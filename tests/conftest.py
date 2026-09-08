from types import SimpleNamespace

import torch

from kvbridge.pipeline import ModelBundle


class FakeTokenizer:
    eos_token_id = 2

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        values = [1] if add_special_tokens else []
        values.extend(3 + (ord(character) % 17) for character in text)
        if not values:
            values = [3]
        ids = torch.tensor([values], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, token_ids, **kwargs):
        return "#### 2"


class FakeLayer:
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class FakeCache:
    def __init__(self, length=0, layers=2):
        self.layers = []
        for _ in range(layers):
            values = torch.zeros((1, 2, length, 2))
            self.layers.append(FakeLayer(values.clone(), values.clone()))

    @property
    def length(self):
        return self.layers[0].keys.shape[-2]

    def append(self, count):
        for layer in self.layers:
            addition = torch.zeros((1, 2, count, 2))
            layer.keys = torch.cat((layer.keys, addition), dim=-2)
            layer.values = torch.cat((layer.values, addition), dim=-2)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 4)
        self.config = SimpleNamespace(
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            hidden_size=4,
            head_dim=2,
        )

    def get_input_embeddings(self):
        return self.embedding

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        use_cache=True,
    ):
        cache = past_key_values if past_key_values is not None else FakeCache()
        cache.append(input_ids.shape[-1])
        logits = torch.zeros((1, input_ids.shape[-1], 32))
        logits[..., 4] = 1.0
        return SimpleNamespace(logits=logits, past_key_values=cache)


def fake_models():
    return {
        name: ModelBundle(FakeModel(), FakeTokenizer(), f"revision-{name}")
        for name in ("model_1", "model_2", "model_3")
    }

