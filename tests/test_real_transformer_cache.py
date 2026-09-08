import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kvbridge import ModelBundle, Pipeline
from kvbridge.adapter import cache_layers

from conftest import FakeTokenizer


def test_real_qwen_attention_matches_cold_and_reused_logits():
    torch.manual_seed(42)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    model = Qwen2ForCausalLM(config).eval()
    pipeline = Pipeline({"qwen": ModelBundle(model, FakeTokenizer())})
    result = pipeline.check_same_model_reuse("qwen", "small prefix", " suffix")
    assert result["max_absolute_logit_difference"] < 1e-6
    assert result["top1_agreement"] == 1.0


def test_real_cache_contains_every_retained_pipeline_token():
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=256,
    )
    bundle = ModelBundle(Qwen2ForCausalLM(config).eval(), FakeTokenizer())
    pipeline = Pipeline({"qwen": bundle}, {"max_new_tokens": 2, "context_limit": 256})
    _, _, state = pipeline.run_steps("1+1?", ["qwen"])
    assert state is not None
    assert cache_layers(state.past_key_values)[0].key.shape[-2] == state.token_ids.shape[-1]
