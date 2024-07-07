import keras
import pandas as pd
import numpy as np
from rdkit import Chem
import tensorflow as tf
import os

os.environ["KERAS_BACKEND"] = "jax"

csv_path = keras.utils.get_file(
    "BBBP.csv", "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"
)

df = pd.read_csv(csv_path, usecols=[1, 2, 3])
df.iloc[96:104]


def dynamic_partition(data, partitions, num_partitions):
    res = []
    for i in range(num_partitions):
        res.append(data[partitions == i])
    return res


class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()


atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": {"B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"},
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3"},
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
    }
)


def molecule_from_smiles(smiles):
    # MolFromSmiles(m, sanitize=True) should be equivalent to
    # MolFromSmiles(m, sanitize=False) -> SanitizeMol(m) -> AssignStereochemistry(m, ...)
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)

    # If sanitization is unsuccessful, catch the error, and try again without
    # the sanitization step that caused the error
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def graph_from_molecule(molecule):
    # Initialize graph
    atom_features = []
    bond_features = []
    pair_indices = []

    for atom in molecule.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))

        # Add self-loops
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    return np.array(atom_features), np.array(bond_features), np.array(pair_indices)


def graphs_from_smiles(smiles_list):
    # Initialize graphs
    atom_features_list = []
    bond_features_list = []
    pair_indices_list = []

    for smiles in smiles_list:
        molecule = molecule_from_smiles(smiles)
        atom_features, bond_features, pair_indices = graph_from_molecule(molecule)

        atom_features_list.append(atom_features)
        bond_features_list.append(bond_features)
        pair_indices_list.append(pair_indices)

    # Convert lists to ragged tensors for tf.data.Dataset later on
    return (
        tf.ragged.constant(atom_features_list, dtype="float32"),
        tf.ragged.constant(bond_features_list, dtype="float32"),
        tf.ragged.constant(pair_indices_list, dtype="int64"),
    )


# Shuffle array of indices ranging from 0 to 2049
permuted_indices = np.random.permutation(np.arange(df.shape[0]))

# Train set: 80 % of data
train_index = permuted_indices[: int(df.shape[0] * 0.8)]
x_train = graphs_from_smiles(df.iloc[train_index].smiles)
y_train = df.iloc[train_index].p_np

# Valid set: 19 % of data
valid_index = permuted_indices[int(df.shape[0] * 0.8) : int(df.shape[0] * 0.99)]
x_valid = graphs_from_smiles(df.iloc[valid_index].smiles)
y_valid = df.iloc[valid_index].p_np

# Test set: 1 % of data
test_index = permuted_indices[int(df.shape[0] * 0.99) :]
x_test = graphs_from_smiles(df.iloc[test_index].smiles)
y_test = df.iloc[test_index].p_np


def prepare_batch(x_batch, y_batch):
    """Merges (sub)graphs of batch into a single global (disconnected) graph"""

    atom_features, bond_features, pair_indices = x_batch

    # Obtain number of atoms and bonds for each graph (molecule)
    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()

    # Obtain partition indices (molecule_indicator), which will be used to
    # gather (sub)graphs from global graph in model later on
    molecule_indices = keras.ops.arange(len(num_atoms))
    molecule_indicator = keras.ops.repeat(molecule_indices, num_atoms)

    # Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
    # 'pair_indices' (and merging ragged tensors) actualizes the global graph
    gather_indices = keras.ops.repeat(molecule_indices[:-1], num_bonds[1:])
    increment = keras.ops.cumsum(num_atoms[:-1])
    increment = keras.ops.pad(
        keras.ops.take(increment, gather_indices), [(num_bonds[0], 0)]
    )
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices = pair_indices + increment[:, tf.newaxis]
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

    return (atom_features, bond_features, pair_indices, molecule_indicator), y_batch


def MPNNDataset(X, y, batch_size=32, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
    print(dataset)
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(prepare_batch, -1).prefetch(-1)


class EdgeNetwork(keras.layers.Layer):
    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.bond_dim, self.atom_dim * self.atom_dim),
            initializer="glorot_uniform",
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.atom_dim * self.atom_dim), initializer="zeros", name="bias"
        )
        self.built = True

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs
        bond_features = keras.ops.matmul(bond_features, self.kernel) + self.bias
        bond_features = keras.ops.reshape(
            bond_features, (-1, self.atom_dim, self.atom_dim)
        )
        atom_features_neighbors = keras.ops.take(atom_features, pair_indices[:, 1])
        atom_features_neighbors = keras.ops.expand_dims(
            atom_features_neighbors, axis=-1
        )
        transformed_features = keras.ops.matmul(bond_features, atom_features_neighbors)
        transformed_features = keras.ops.squeeze(transformed_features, axis=-1)
        segment_ids = keras.ops.one_hot(
            pair_indices[:, 0], num_classes=keras.ops.shape(atom_features)[0]
        )
        aggregated_features = keras.ops.segment_sum(
            transformed_features,
            segment_ids,
            num_segments=keras.ops.shape(atom_features)[0],
        )
        return aggregated_features


class MessagePassing(keras.layers.Layer):
    def __init__(self, units, steps=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.message_step = EdgeNetwork()
        self.pad_length = keras.ops.max(0, self.units - self.atom_dim)
        self.update_step = keras.layers.GRUCell(self.atom_dim + self.pad_length)
        self.built = True

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs
        atom_features_updated = keras.ops.pad(
            atom_features, [(0, 0), (0, self.pad_length)]
        )
        # number of message passing steps
        for _ in range(self.steps):
            # aggregated features from neighbors
            atom_features_aggregated = self.message_step(
                [atom_features_updated, bond_features, pair_indices]
            )
            # update node state via a step of GRU
            atom_features_updated, _ = self.update_step(
                atom_features_aggregated, atom_features_updated
            )
        return atom_features_updated


class PartitionPadding(keras.layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs):
        atom_features, molecule_indicator = inputs

        # get the subgraphs
        atom_features_partitioned = dynamic_partition(
            atom_features, molecule_indicator, self.batch_size
        )
        num_atoms = [keras.ops.shape(f)[0] for f in atom_features_partitioned]
        max_num_atoms = keras.ops.max(num_atoms)
        atom_features_stacked = keras.ops.stack(
            [
                keras.ops.pad(f, [(0, max_num_atoms - n), (0, 0)])
                for f, n in zip(atom_features_partitioned, num_atoms)
            ],
            axis=0,
        )

        # remove empty subgraphs
        gather_indices = keras.ops.where(
            keras.ops.sum(atom_features_stacked, (1, 2)) != 0
        )
        gather_indices = keras.ops.squeeze(gather_indices, axis=-1)
        return keras.ops.take(atom_features_stacked, gather_indices, axis=0)


class TransformerEncoderReadout(keras.layers.Layer):
    def __init__(
        self, num_heads=8, embed_dim=64, dense_dim=512, batch_size=32, **kwargs
    ):
        super().__init__(**kwargs)
        self.partition_padding = PartitionPadding(batch_size)
        self.attention = keras.layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [
                keras.layers.Dense(dense_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layerNorm1 = keras.layers.LayerNormalization()
        self.layerNorm2 = keras.layers.LayerNormalization()
        self.average_pooling = keras.layers.GlobalAveragePooling1D()

    def call(self, inputs):
        x = self.partition_padding(inputs)
        padding_mask = keras.ops.any(keras.ops.not_equal(x, 0.0), axis=-1)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        attention_output = self.attention(x, x, attention_mask=padding_mask)
        proj_input = self.layerNorm1(x + attention_output)
        proj_output = self.layerNorm2(proj_input + self.dense_proj(proj_input))
        return self.average_pooling(proj_output)


class MPNNModel(keras.Model):
    def __init__(
        self,
        atom_dim,
        bond_dim,
        batch_size=32,
        message_units=64,
        message_steps=4,
        num_attention_heads=8,
        dense_units=512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.message_units = message_units
        self.message_steps = message_steps
        self.num_attention_heads = num_attention_heads
        self.dense_units = dense_units
        self.atom_features = keras.layers.Input(
            (atom_dim,), dtype="float32", name="atom_features"
        )
        self.bond_features = keras.layers.Input(
            (bond_dim,), dtype="float32", name="bond_features"
        )
        self.pair_indices = keras.layers.Input((2,), dtype="int32", name="pair_indices")
        self.molecule_indicator = keras.layers.Input(
            (), dtype="int32", name="molecule_indicator"
        )
        self.message_passing = MessagePassing(self.message_units, self.message_steps)
        self.transformer_encoder_readout = TransformerEncoderReadout(
            num_attention_heads, message_units, dense_units, batch_size
        )
        self.dense1 = keras.layers.Dense(self.dense_units, activation="relu")
        self.dense2 = keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        atom_features, bond_features, pair_indices, molecule_indicator = inputs
        x = self.message_passing(
            [self.atom_features, self.bond_features, self.pair_indices]
        )
        x = self.transformer_encoder_readout([x, self.molecule_indicator])
        x = self.dense1(x)
        x = self.dense2(x)
        return x


print("=== Initializing the model ===")
mpnn = MPNNModel(atom_dim=x_train[0][0][0].shape[0], bond_dim=x_train[1][0][0].shape[0])
print("=== Compiling the model ===")
mpnn.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer="adam",
    metrics=[keras.metrics.AUC(name="AUC")],
)

print("=== Loading the dataset ===")
train_dataset = MPNNDataset(x_train, y_train)
valid_dataset = MPNNDataset(x_valid, y_valid)
test_dataset = MPNNDataset(x_test, y_test)

print(train_dataset)


print("=== Start the training ===")
history = mpnn.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=40,
    verbose=2,
    class_weight={0: 2.0, 1: 0.5},
)
