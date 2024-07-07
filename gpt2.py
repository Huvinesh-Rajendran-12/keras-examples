import keras
from dataclasses import dataclass
import logging


class CausalSelfAttention(keras.layers.Layer):
    def __init__(self, config) -> None:
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.n_embed = config.n_embed
        self.n_head = config.n_head
        self.c_attn = keras.layers.Dense(3 * config.n_embed)
        self.c_proj = keras.layers.Dense(config.n_embed)
        self.c_proj.GPT_SCALE_INIT = 1
        self.bias = self.add_weight(
            name="bias",
            trainable=False,
            initializer="ones",
            shape=(1, 1, config.block_size, config.block_size),
        )

    def call(self, x: keras.KerasTensor):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = keras.ops.split(qkv, 3, axis=-1)
        k = keras.ops.reshape(k, (B, T, self.n_head, C // self.n_head))
        k = keras.ops.transpose(k, [1, 2])
        q = keras.ops.reshape(q, [B, T, self.n_head, C // self.n_head])
        q = keras.ops.transpose(q, [1, 2])
        v = keras.ops.reshape(v, (B, T, self.n_head, C // self.n_head))
        v = keras.ops.transpose(v, [1, 2])
        att = keras.ops.multiply(
            keras.ops.matmul(q, keras.ops.transpose(k, [-2, -1])),
            keras.ops.divide(1, keras.ops.sqrt(keras.ops.shape(k)[-1])),
        )
        att = keras.ops.where(self.bias[:, :, T, T] == 0, float("-inf"))
        att = keras.ops.softmax(att, axis=-1)
        y = keras.ops.matmul(att, v)
        y = keras.ops.reshape(keras.ops.transpose(y, [1, 2]), (B, T, C))
        y = self.c_proj(y)
        return y


class MLP(keras.layers.Layer):
    def __init__(self, config) -> None:
        super().__init__()
        self.c_fc = keras.layers.Dense(4 * config.n_embed, activation="gelu")
        self.c_proj = keras.layers.Dense(config.n_embed)

    def call(self, x):
        x = self.c_fc(x)
        x = self.c_proj(x)
        return x


class Block(keras.layers.Layer):
    def __init__(self, config) -> None:
        super().__init__()
        self.ln_1 = keras.layers.LayerNormalization(axis=-1)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = keras.layers.LayerNormalization(axis=-1)
        self.mlp = MLP(config)

    def call(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50527
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


class GPT(keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = keras.Sequential(
            [
                keras.layers.Embedding(config.vocab_size, config.n_embed, name="wte"),
                keras.layers.Embedding(config.block_size, config.n_embed, name="wpe"),
                keras.Sequential(
                    [Block(config) for _ in range(config.n_layer)], name="h"
                ),
                keras.layers.LayerNormalization(axis=-1, name="ln_f"),
            ],
            name="transformer",
        )
        self.lm_head = keras.layers.Dense(
            config.vocab_size, use_bias=False, name="lm_head"
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.transformer.layers:
            if isinstance(layer, keras.layers.Dense):
                std: float = 0.02
                if hasattr(layer, "GPT_SCALE_INIT"):
                    std *= (2 * self.config.n_layer) ** -0.5

                keras.initializers.TruncatedNormal(stddev=std)(layer.kernel)
                if layer.bias is not None:
                    keras.initializers.zeros()(layer.bias)
            elif isinstance(layer, keras.layers.Embedding):
                keras.initializers.TruncatedNormal(stddev=0.02)(layer.embeddings)

    def call(self, idx, targets=None):
        _, T = keras.ops.shape(idx)
        assert T <= self.config.block_size, f"Cannot forward seq of length {
            T}, block size is {self.config.block_size}"
        pos = keras.ops.arange(0, T, dtype="int32")
        pos_emb = self.transformer.get_layer("wpe")(pos)
        tok_emb = self.transformer.get_layer("wte")(idx)
        x = keras.ops.add(pos_emb, tok_emb)
        for block in self.transformer.get_layer("h"):
            x = block(x)
        x = self.transformer.get_layer("ln_f")(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = keras.losses.categorical_crossentropy(
                keras.ops.reshape(logits, (-1, keras.ops.shape(logits)[-1])),
                keras.ops.reshape(targets, (-1)),
            )
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Load a pretrained GPT-2 model weights from huggingface"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import TFAutoModelForCausalLM

        logging.info("Loading weights from pretrained gpt: %s", model_type)
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embed=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embed=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embed=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embed=1600),
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        model_hf = TFAutoModelForCausalLM.from_pretrained(model_type)
        hf_weights = model_hf.get_weights()
        hf_weight_names = [
            weight.name for layer in model_hf.layers for weight in layer.weights
        ]

        name_mapping = {
            "ln_1": "layer_norm1",
            "ln_2": "layer_norm2",
            "ln_f": "layer_norm_final",
            "wte": "token_embedding",
            "wpe": "position_embedding",
            "c_attn": "attention/c_attn",
            "c_proj": "attention/c_proj",
            "c_fc": "mlp/c_fc",
        }
        # Transfer weights
        for i, (name, weight) in enumerate(zip(hf_weight_names, hf_weights)):
            # Skip bias weights in attention
            if "attn.masked_bias" in name or "attn.bias" in name:
                continue

            # Rename layers
            for old, new in name_mapping.items():
                name = name.replace(old, new)

            # Handle transposed weights
            if any(
                substr in name
                for substr in [
                    "attention/c_attn",
                    "attention/c_proj",
                    "mlp/c_fc",
                    "mlp/c_proj",
                ]
            ):
                weight = keras.ops.transpose(weight)

            # Set the weight in the Keras model
            model.get_layer(name.split("/")[0]).set_weights([weight])

        return model
