from keras import layers, Model, random
import keras.api.ops as ops
import os

os.environ["KERAS_BACKEND"] = "jax"

# Example usage
batch_size: int = 1
num_nodes: int = 10
num_features: int = 5
hidden_units: int = 8  # Changed to match the input shape
num_heads: int = 2
num_classes: int = 1  # For regression tasks
dropout_rate: float = 0.1
max_atoms: int = 10

class GraphAttentionLayer(layers.Layer):
    def __init__(
        self,
        units,
        num_heads: int = 1,
        dropout_rate: float = 0.5,
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.activation = activation

    def build(self, input_shape) -> None:
        self.W = self.add_weight(
            shape=(input_shape[0][-1], self.units * self.num_heads),
            initializer="glorot_uniform",
            name="W",
        )
        self.a = self.add_weight(
            shape=(2 * self.units, 1), initializer="glorot_uniform", name="a"
        )
        self.dropout = layers.Dropout(self.dropout_rate)
        self.activation_f = layers.Activation(self.activation)

    def call(self, inputs):
        X, A = inputs
        batch_size, num_nodes, num_features = X.shape

        print(f"Input shapes: X={X.shape}, A={A.shape}")
        print(f"W shape: {self.W.shape}")

        # Linear transformation
        h = ops.matmul(X, self.W)
        print(f"h shape after linear transformation: {h.shape}")

        h = ops.reshape(h, (batch_size, num_nodes, self.num_heads, self.units))
        print(f"h shape after reshape: {h.shape}")

        # Self-attention
        h_broadcast_1 = ops.repeat(ops.expand_dims(h, axis=2), num_nodes, axis=2)
        h_broadcast_2 = ops.repeat(ops.expand_dims(h, axis=1), num_nodes, axis=1)
        attention_input = ops.concatenate([h_broadcast_1, h_broadcast_2], axis=-1)
        print(f"attention_input shape: {attention_input.shape}")

        e = ops.squeeze(ops.matmul(attention_input, self.a), axis=-1)
        print(f"e shape after attention: {e.shape}")
        # mask attention coeff using A matrix
        mask = ops.expand_dims(A, axis=-1)
        masked_e = ops.where(ops.greater(mask, 0), e, -1e9)
        alpha = ops.softmax(masked_e, axis=2)
        alpha = self.dropout(alpha)
        # apply attention
        h = ops.transpose(h, (0, 3, 1, 2))
        output = ops.matmul(alpha, h)
        output = ops.transpose(output, (0, 2, 3, 1))
        output = ops.reshape(
            output, (batch_size, num_nodes, self.units * self.num_heads)
        )
        return self.activation_f(output)


class GAT(Model):
    def __init__(
        self,
        hidden_units: int,
        num_heads: int,
        num_classes: int,
        dropout_rate: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

    def build(self) -> None:
        self.gat1 = GraphAttentionLayer(self.hidden_units, self.num_heads, self.dropout_rate)
        self.gat2 = GraphAttentionLayer(self.num_classes, 1, self.dropout_rate, activation="linear")
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, inputs):
        X, A = inputs
        h = self.gat1((X, A))
        h = self.dropout(h)
        output = self.gat2((h, A))
        return ops.softmax(ops.squeeze(output, axis=-1), axis=-1)


# Create dummy data
X = random.normal((batch_size, num_nodes, num_features))
A = ops.cast(random.uniform(
    (batch_size, num_nodes, num_nodes)) > 0.5, "float32")

# Create and compile the model
model: GAT = GAT(hidden_units, num_heads, num_classes, dropout_rate)
model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])

# Make a prediction
prediction = model((X, A))
print(f"Prediction shape: {prediction.shape}")

# Summary of the model
model.build([(batch_size, num_nodes, num_features),
            (batch_size, num_nodes, num_nodes)])
model.summary()
