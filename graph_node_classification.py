from jax._src.typing import Array
from keras.api import KerasTensor
from typing_extensions import reveal_type
from typing import Tuple, List
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import keras
import keras.api.ops as ops
import networkx as nx
import numpy as np
import os
import pandas as pd

os.environ["KERAS_BACKEND"] = "jax"

def build_cora_graph(citations: pd.DataFrame) -> nx.Graph:
    """
    Build a CoraGraph from a Pandas DataFrame of citations.

    Args:
    citations (pd.DataFrame): A Pandas DataFrame with 'source' and 'target' columns
                              representing paper citations.

    Returns:
    nx.Graph: A NetworkX graph representing the Cora citation network.
    """
    # Create an empty undirected graph
    G = nx.Graph()

    # Convert the Pandas DataFrame to a list of tuples
    edges = citations[['source', 'target']].values.tolist()

    # Add edges to the graph
    G.add_edges_from(edges)

    # Print some basic information about the graph
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    return G

def train_test_split(data: pd.DataFrame, train_fraction: float = 0.5, seed: int = 42) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
    # Set the random seed for reproducibility
    train_data, test_data = [], []

    for _, group_data in data.groupby("subject"):
        # Select around 50% of the dataset for training.
        random_selection = np.random.rand(len(group_data.index)) <= 0.7
        train_data.append(group_data[random_selection])
        test_data.append(group_data[~random_selection])

    train_data = pd.concat(train_data).sample(frac=1)
    test_data = pd.concat(test_data).sample(frac=1)

    print("Train data shape:", train_data.shape)
    print("Test data shape:", test_data.shape)

    return train_data, test_data

def create_ffn(
    hidden_units: List[int], dropout_rate: float, name: str | None = None
)-> keras.Sequential:
    return keras.Sequential(
        [
            layer for units in hidden_units
            for layer in (
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(dropout_rate),
                keras.layers.Dense(units, activation="gelu")
            )
        ],
        name=name
    )

def create_gru(
    hidden_units: List[int],
    dropout_rate: float
) -> keras.Model:
    inputs = keras.Input(shape=(2, hidden_units[0]))
    x = inputs
    for units in hidden_units:
        x = keras.layers.GRU(
            units=units,
            activation="tanh",
            recurrent_activation="sigmoid",
            return_sequences=True,
            dropout=dropout_rate,
            return_state=False,
            recurrent_dropout=dropout_rate,
        )(x)
    return keras.Model(inputs=inputs, outputs=x)

def create_update_fn(
    hidden_units: List[int],
    dropout_rate: float,
    combination: str = "concat"
) -> keras.Sequential | keras.Model:
    if combination is "gru":
        return create_gru(hidden_units=hidden_units, dropout_rate=dropout_rate)
    else:
        return create_ffn(hidden_units=hidden_units, dropout_rate=dropout_rate)



@dataclass
class ModelConfig:
    def __init__(self, hidden_units: List[int] = [32, 32], learning_rate: float = 0.01, dropout_rate: float = 0.5, num_epochs: int = 100, batch_size: int = 64):
        super().__init__()
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size


class GraphConvLayer(keras.Layer):
    def __init__(
        self,
        hidden_units: List[int],
        dropout_rate: float = 0.2,
        reduce: str = "mean",
        combination: str = "concat",
        normalize: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        assert reduce in ["mean", "max", "sum"],
        "reduce: str must be one of the following values: mean, max, sum"
        assert combination in ["concat", "gru", "add"],
        "combination: str must be one of the following values: concat, gru, add"
        self.reduce = reduce
        self.combination = combination
        self.normalize = normalize
        self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        self.update_fn = create_update_fn(hidden_units, dropout_rate, self.combination)

    def prepare(
        self,
        node_representations,
        weights=Array | None
    ) -> Array:
        messages = self.ffn_prepare(node_representations)
        if weights is not None:
            messages = messages * keras.ops.expand_dims(weights, -1)
        return messages

    def reduce(
        self,
        node_indices: Array,
        neighbour_messages: Array,
        node_representations: Array
    ) -> Array:
        n_nodes = node_representations.shape[0]
        reduce_msg = []
        if self.reduce is "sum":
            reduce_msg = keras.ops.segment_sum(
                neighbour_messages,
                node_indices,
                num_segments=n_nodes
            )

        elif self.reduce is "mean":
            sum_msg = keras.ops.segment_sum(
                data=neighbour_messages,
                segment_ids=node_indices,
                num_segments=n_nodes
            )
            count = keras.ops.segment_sum(
                data=keras.ops.ones_like(neighbour_messages[:, 0]),
                segment_ids=node_indices,
                num_segments=n_nodes
            )
            reduce_msg = sum_msg / keras.ops.maximum(count, 1)[:, None]
        elif self.reduce is "max":
            reduce_msg = keras.ops.segment_max(
                neighbour_messages,
                node_indices,
                num_segments=n_nodes
            )
        return jnp.array(reduce_msg)

    def update(
        self,
        node_representations: Array,
        reduced_messages: Array
    ) -> Array:
        h = []
        if self.combination is "gru":
            h = keras.ops.stack(
                [node_representations, reduced_messages],
                axis=1
            )
        elif self.combination is "concat":
            h = keras.ops.concatenate(
                [node_representations, reduced_messages],
                axis=1
            )
        elif self.combination is "add":
            h = node_representations + reduced_messages

        node_embds = self.update_fn(h)
        node_embds = keras.ops.unstack(node_embds, axis=1)[-1] if self.combination is "gru" else node_embds
        node_embds = keras.ops.nn.normalize(node_embds, axis=-1) if self.normalize else node_embds
        return jnp.array(node_embds)

    def call(
        self,
        inputs: Array
    ) -> Array:
        node_representations, edges, edge_W = inputs
        node_idx, neighbour_idx = edges[0], edges[1]
        neighbour_representations = keras.ops.take(node_representations, neighbour_idx)
        neighbour_msgs = self.prepare(neighbour_representations, edge_W)
        reduced_msgs = self.reduce(node_idx, neighbour_msgs, node_representations)
        return self.update(node_representations, reduced_msgs)


class GNNNodeClassifier(keras.Model):
    def __init__(
        self,
        graph_info: Tuple,
        num_classes: int,

    )






class GNNLayer(keras.layers.Layer):
    def __init__(self, units: int, trainable: bool = True , **kwargs):
        super().__init__(**kwargs)
        self.units: int = units
        self.trainable: bool = trainable

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="glorot_uniform",
            trainable=self.trainable,
            name="weight"
        )

        self.b = self.add_weight(
            shape=(self.units,),
            initializer="glorot_uniform",
            trainable=self.trainable,
            name="bias"
        )
        super().build(input_shape)

    def call(self, inputs):
        X, A = inputs
        n_features = ops.matmul(A, X)
        h = ops.matmul(n_features, self.w) + self.b
        return ops.relu(h)

    def compute_output_shape(self, input_shape):
            return (input_shape[0], self.units)


class GNNModel(keras.Model):
    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.gnn_layer1 = GNNLayer(hidden_dim)
        self.gnn_layer2 = GNNLayer(hidden_dim)
        self.output_layer = keras.layers.Dense(output_dim, activation="softmax")

    def call(self, inputs, training: bool = True):
        X, A = inputs
        h = self.gnn_layer1([X, A])
        h = self.gnn_layer2([h, A])
        return self.output_layer(h)

def main():
    zip_file = keras.utils.get_file(
        fname="cora.tgz",
        origin="https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
        extract=True,
    )
    data_dir = os.path.join(os.path.dirname(zip_file), "cora")
    citations = pd.read_csv(
        os.path.join(data_dir, "cora.cites"),
        sep="\t",
        header=None,
        names=["target", "source"]
    )
    print(f"citations: {citations.shape}")
    column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
    papers = pd.read_csv(
        os.path.join(data_dir, "cora.content"),
        sep="\t",
        header=None,
        names=column_names
    )
    print(f"papers: {papers.shape}")

    class_values = sorted(papers["subject"].unique())
    class_idx = {name: id for id, name in enumerate(class_values)}

    # Get unique sorted paper IDs and create paper index
    paper_ids = sorted(papers["paper_id"].unique())
    paper_idx = {name: idx for idx, name in enumerate(paper_ids)}

    # Update paper_id in papers DataFrame
    papers["paper_id"] = papers["paper_id"].apply(lambda x: paper_idx[x])
    papers["subject"] = papers["subject"].apply(lambda x: class_idx[x])

    # Update source and target in citations DataFrame
    citations["source"] = citations["source"].apply(lambda x: paper_idx[x])
    citations["target"] = citations["target"].map(lambda x: paper_idx[x])

    colors = papers["subject"].unique().tolist()
    G = build_cora_graph(citations)
    subjects = papers[papers["subject"].isin(list(G.nodes()))]["subject"].tolist()
    train_data, test_data = train_test_split(papers)

    feature_names = list(set(papers.columns) - {"paper_id", "subject"})

    edges: Array = jnp.array(citations[["source", "target"]]).T
    edge_W: Array = jnp.array(keras.ops.ones(shape=edges.shape[1]))
    node_features = keras.ops.cast(
        jnp.array(papers.sort_values("paper_id")[feature_names]),
        dtype="float32"
    )
    graph_info = (node_features, edges, edge_W)

    print(f"Edges: {edges.shape}")
    print(f"Nodes: {keras.ops.shape(node_features)}")



if __name__ == "__main__":
    main()
