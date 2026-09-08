from types import SimpleNamespace

import torch
from transformers import Qwen2Config

from kvbridge.adapter import LinearKVAdapter, cache_layers, relative_layer_map


def config(layers=2, heads=2, width=3):
    return SimpleNamespace(
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        hidden_size=heads * width,
        head_dim=width,
    )


def legacy_cache(layers, heads, tokens, width, transform=None):
    result = []
    for _ in range(layers):
        source = torch.randn(1, heads, tokens, width)
        target = transform(source) if transform else source
        result.append((target, target + 0.25))
    return tuple(result)


def test_relative_depth_mapping_includes_endpoints():
    assert relative_layer_map(3, 5) == (0, 0, 1, 2, 2)


def test_adapter_output_shape_and_finiteness():
    source_config = config(layers=2, heads=3, width=2)
    target_config = Qwen2Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    adapter = LinearKVAdapter("a", "b")
    adapter.layer_mapping = relative_layer_map(2, 3)
    for layer in range(3):
        for kind in ("key", "value"):
            adapter.weights[f"layer.{layer}.{kind}.weight"] = torch.randn(2, 4)
            adapter.weights[f"layer.{layer}.{kind}.bias"] = torch.randn(4)
    adapter.prepare_target(torch.ones((1, 7), dtype=torch.long))
    result = adapter.adapt(legacy_cache(2, 3, 5, 2), source_config, target_config)
    layers = cache_layers(result)
    assert len(layers) == 3
    assert all(layer.key.shape == (1, 2, 7, 4) for layer in layers)
    assert all(torch.isfinite(layer.value).all() for layer in layers)


def test_ridge_fit_recovers_known_affine_projection():
    torch.manual_seed(4)
    source_config = config(layers=1, heads=1, width=2)
    target_config = config(layers=1, heads=1, width=2)
    weight = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    bias = torch.tensor([0.25, -0.75])
    pairs = []
    for _ in range(8):
        source = legacy_cache(1, 1, 10, 2)
        source_layer = cache_layers(source)[0]
        target = ((source_layer.key @ weight + bias, source_layer.value @ weight + bias),)
        pairs.append((source, target))
    adapter = LinearKVAdapter("a", "b", ridge_lambda=1e-8)
    adapter.fit(pairs, source_config, target_config, max_positions=10)
    assert torch.allclose(adapter.weights["layer.0.key.weight"], weight, atol=1e-4)
    assert torch.allclose(adapter.weights["layer.0.key.bias"], bias, atol=1e-4)
