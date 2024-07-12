import os

os.environ["KERAS_BACKEND"] = "jax"
from jax._src.typing import Array
from typing import Tuple, List
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import keras
import keras.api.ops as ops
import networkx as nx
import numpy as np
import pandas as pd


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
    edges = citations[["source", "target"]].values.tolist()

    # Add edges to the graph
    G.add_edges_from(edges)

    # Print some basic information about the graph
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    return G


def build_graph(node_features, new_instances, edges, papers, num_classes):
    # First we add the N new_instances as nodes to the graph
    # by concatenating the new_instance to node_features.
    num_nodes = node_features.shape[0]
    new_node_features = ops.concatenate([node_features, new_instances])

    # Second we add the M edges (citations) from each new node to a set
    # of existing nodes in a particular subject
    new_node_indices = jnp.array([i + num_nodes for i in range(num_classes)])

    def process_subject(subject_idx, subject_papers):
        key = jax.random.PRNGKey(0)  # You might want to use a different seed

        # Select random x papers specific subject.
        key, subkey = jax.random.split(key)
        selected_paper_indices1 = jax.random.choice(
            subkey, jnp.array(subject_papers), shape=(5,), replace=False
        )

        # Select random y papers from any subject (where y < x).
        key, subkey = jax.random.split(key)
        selected_paper_indices2 = jax.random.choice(
            subkey, jnp.array(papers.paper_id), shape=(2,), replace=False
        )

        # Merge the selected paper indices.
        selected_paper_indices = ops.concatenate(
            [selected_paper_indices1, selected_paper_indices2]
        )

        # Create edges between a citing paper idx and the selected cited papers.
        citing_paper_indx = new_node_indices[subject_idx]
        new_citations = ops.stack(
            [
                ops.full_like(selected_paper_indices, citing_paper_indx),
                selected_paper_indices,
            ]
        )

        return new_citations

    # Use vmap to process all subjects
    subjects = jnp.array(papers.groupby("subject").groups.keys())
    subject_papers = jnp.array(
        [jnp.array(group.paper_id) for _, group in papers.groupby("subject")]
    )

    new_citations = jax.vmap(process_subject)(jnp.arange(len(subjects)), subject_papers)
    new_citations = ops.reshape(new_citations, (-1, 2)).T

    new_edges = ops.concatenate([edges, new_citations], axis=1)

    return new_node_features, new_node_indices, new_edges


def train_test_split(
    data: pd.DataFrame, train_fraction: float = 0.5, seed: int = 42
) -> tuple[pd.Series | pd.DataFrame, pd.Series | pd.DataFrame]:
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


def generate_random_instances(num_instances, x_train):
    # Calculate token probability using Keras operations
    token_probability = ops.mean(x_train, axis=0)

    # Convert token_probability to JAX array
    token_probability = jnp.array(token_probability)

    # Generate random numbers and create instances using JAX
    def generate_instance(key):
        probabilities = jax.random.uniform(key, shape=token_probability.shape)
        return (probabilities <= token_probability).astype(jnp.int32)

    # Create a PRNG key
    key = jax.random.PRNGKey(0)

    # Generate instances using JAX's vmap for vectorization
    keys = jax.random.split(key, num_instances)
    instances = jax.vmap(generate_instance)(keys)

    return instances


def create_ffn(
    hidden_units: List[int], dropout_rate: float, name: str | None = None
) -> keras.Sequential:
    return keras.Sequential(
        [
            layer
            for units in hidden_units
            for layer in (
                keras.layers.BatchNormalization(),
                keras.layers.Dropout(dropout_rate),
                keras.layers.Dense(units, activation="gelu"),
            )
        ],
        name=name,
    )


def create_gru(hidden_units: List[int], dropout_rate: float) -> keras.Model:
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
    hidden_units: List[int], dropout_rate: float, combination: str = "concat"
) -> keras.Sequential | keras.Model:
    if combination == "gru":
        return create_gru(hidden_units=hidden_units, dropout_rate=dropout_rate)
    else:
        return create_ffn(hidden_units=hidden_units, dropout_rate=dropout_rate)


@dataclass
class ModelConfig:
    def __init__(
        self,
        hidden_units: List[int] = [32, 32],
        learning_rate: float = 0.01,
        dropout_rate: float = 0.5,
        num_epochs: int = 100,
        batch_size: int = 64,
    ):
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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert reduce in [
            "mean",
            "max",
            "sum",
        ], "reduce: str must be one of the following values: mean, max, sum"
        assert combination in [
            "concat",
            "gru",
            "add",
        ], "combination: str must be one of the following values: concat, gru, add"
        self.reduce = reduce
        self.combination = combination
        self.normalize = normalize
        self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        self.update_fn = create_update_fn(hidden_units, dropout_rate, self.combination)

    def prepare(self, node_representations, weights=Array | None) -> Array:
        messages = self.ffn_prepare(node_representations)
        if weights is not None:
            messages = messages * ops.expand_dims(weights, -1)
        return messages

    def _reduce(
        self,
        node_indices: Array,
        neighbour_messages: Array,
        node_representations: Array,
    ) -> Array:
        n_nodes = node_representations.shape[0]
        reduce_msg = []
        if self.reduce == "sum":
            reduce_msg = ops.segment_sum(
                neighbour_messages, node_indices, num_segments=n_nodes
            )

        elif self.reduce == "mean":
            sum_msg = ops.segment_sum(
                data=neighbour_messages, segment_ids=node_indices, num_segments=n_nodes
            )
            count = ops.segment_sum(
                data=ops.ones_like(neighbour_messages[:, 0]),
                segment_ids=node_indices,
                num_segments=n_nodes,
            )
            reduce_msg = sum_msg / ops.maximum(count, 1)[:, None]
        elif self.reduce == "max":
            reduce_msg = ops.segment_max(
                neighbour_messages, node_indices, num_segments=n_nodes
            )
        return jnp.array(reduce_msg)

    def update(self, node_representations: Array, reduced_messages: Array) -> Array:
        h = []
        if self.combination == "gru":
            h = ops.stack([node_representations, reduced_messages], axis=1)
        elif self.combination == "concat":
            h = ops.concatenate([node_representations, reduced_messages], axis=1)
        elif self.combination == "add":
            h = node_representations + reduced_messages

        node_embds = self.update_fn(h)
        node_embds = (
            ops.unstack(node_embds, axis=1)[-1]
            if self.combination == "gru"
            else node_embds
        )
        node_embds = (
            ops.nn.normalize(node_embds, axis=-1) if self.normalize else node_embds
        )
        return jnp.array(node_embds)

    def call(self, inputs: Array) -> Array:
        node_representations, edges, edge_W = inputs
        node_idx, neighbour_idx = edges[0], edges[1]
        print(node_representations)
        print(neighbour_idx[-1])
        neighbour_representations = ops.take(node_representations, neighbour_idx)
        neighbour_msgs = self.prepare(neighbour_representations, edge_W)
        reduced_msgs = self._reduce(node_idx, neighbour_msgs, node_representations)
        return self.update(node_representations, reduced_msgs)


class GNNNodeClassifier(keras.Model):
    def __init__(
        self,
        graph_info: Tuple[Array, Array, Array | None],
        num_classes: int,
        hidden_units: List[int],
        reduce: str = "sum",
        combination: str = "concat",
        dropout_rate: float = 0.2,
        normalize: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        node_features, edges, edge_W = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_W = edge_W
        if self.edge_W is None:
            self.edge_W = ops.ones(shape=edges.shape[1])
        self.edge_W = self.edge_W / ops.sum(self.edge_W)
        self.preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")
        self.conv1 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            reduce,
            combination,
            normalize,
            name="graph_conv1",
        )
        self.conv2 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            reduce,
            combination,
            normalize,
            name="graph_conv2",
        )
        self.postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")
        self.compute_logits = keras.layers.Dense(units=num_classes, name="logits")

    def call(self, input_node_idx):
        x = self.preprocess(self.node_features)
        x1 = self.conv1((x, self.edges, self.edge_W))
        x = x1 + x
        x2 = self.conv2((x, self.edges, self.edge_W))
        x = x2 + x
        x = self.postprocess(x)
        node_embds = ops.take(x, input_node_idx)
        return self.compute_logits(node_embds)


def main():
    print(f"Backend: {keras.config.backend()}")
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
        names=["target", "source"],
    )
    print(f"citations: {citations.shape}")
    column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
    papers = pd.read_csv(
        os.path.join(data_dir, "cora.content"),
        sep="\t",
        header=None,
        names=column_names,
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
    n_features = len(feature_names)
    n_classes: int = len(class_idx)

    edges: Array[int] = jnp.array(citations[["source", "target"]]).T
    edge_W: Array = jnp.array(ops.ones(shape=edges.shape[1]))
    node_features = ops.cast(
        jnp.array(papers.sort_values("paper_id")[feature_names]), dtype="float32"
    )
    graph_info = (node_features, edges, edge_W)

    print(f"Edges: {edges.shape}")
    print(f"Nodes: {ops.shape(node_features)}")

    config = ModelConfig()

    model = GNNNodeClassifier(
        graph_info=graph_info,
        num_classes=n_classes,
        hidden_units=config.hidden_units,
        dropout_rate=config.dropout_rate,
    )

    gnn_model: keras.Model = model(
        keras.ops.convert_to_tensor([1, 10, 100], dtype="int32")
    )
    print(f"GNN output shape: {gnn_model.shape}")

    gnn_model.summary()
    X_train = jnp.array(train_data["paper_id"])
    y_train = train_data["subject"]

    X_test = jnp.array(test_data["paper_id"])
    y_test = test_data["subject"]

    gnn_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.01),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="Acc")],
    )

    early_callback = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
    )

    _ = gnn_model.fit(
        x=X_train,
        y=y_train,
        epochs=10,
        batch_size=256,
        validation_split=0.15,
        callbacks=[early_callback],
    )

    _, test_accuracy = gnn_model.evalute(X_test, y_test, VERBOSE=0)

    print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")
    # new_instances = generate_random_instances(n_classes, X_train)

    # (new_node_features, new_node_idx ,new_edges) = build_graph(node_features, new_instances, edges, papers, n_classes)

    # print("Original node_features shape:", gnn_model.node_features.shape)
    # print("Original edges shape:", gnn_model.edges.shape)
    # gnn_model.node_features = new_node_features
    # gnn_model.edges = new_edges
    # gnn_model.edge_weights = ops.ones(shape=ops.shape(new_edges)[1])
    # print("New node_features shape:", ops.shape(gnn_model.node_features))
    # print("New edges shape:", ops.shape(gnn_model.edges))

    # logits = gnn_model.predict(ops.convert_to_tensor(new_node_idx))


if __name__ == "__main__":
    main()
