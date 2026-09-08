from kvbridge import Pipeline

from conftest import FakeCache, fake_models


def test_same_model_reuse_counts_tokens_and_preserves_output():
    models = fake_models()
    cold = Pipeline(models, {"cache_policy": "disabled", "max_new_tokens": 2})
    reuse = Pipeline(models, {"cache_policy": "reuse", "max_new_tokens": 2})
    assignment = ["model_1"] * 3
    cold_result, _ = cold.run("One plus one?", assignment)
    reuse_result, logs = reuse.run("One plus one?", assignment)
    assert cold_result == reuse_result == "#### 2"
    assert logs["steps"][1]["exact_reused_tokens"] > 0
    assert logs["steps"][2]["exact_reused_tokens"] > 0
    assert all(step["transferred_tokens"] == 0 for step in logs["steps"])


def test_every_retained_generated_token_is_cached():
    pipeline = Pipeline(fake_models(), {"max_new_tokens": 2})
    _, _, state = pipeline.run_steps("One plus one?", ["model_1"])
    assert state is not None
    assert state.past_key_values.length == state.token_ids.shape[-1]


def test_same_model_teacher_forced_logits_match():
    pipeline = Pipeline(fake_models(), {"max_new_tokens": 2})
    result = pipeline.check_same_model_reuse("model_1", "prefix", " suffix")
    assert result["max_absolute_logit_difference"] == 0.0
    assert result["top1_agreement"] == 1.0


class RejectedAdapter:
    approved = False
    rejection_reason = "degradation_threshold_exceeded"

    def prepare_target(self, ids):
        self.ids = ids

    def adapt(self, source_cache, source_config, target_config):
        return FakeCache(length=self.ids.shape[-1])


def test_rejected_adapter_falls_back_to_cold_prefill():
    pipeline = Pipeline(
        fake_models(),
        {
            "cache_policy": "reuse",
            "max_new_tokens": 1,
            "adapters": {"model_1->model_2": RejectedAdapter()},
        },
    )
    _, logs, _ = pipeline.run_steps("One plus one?", ["model_1", "model_2"])
    step = logs["steps"][1]
    assert step["adapter_attempted"] is True
    assert step["adapter_accepted"] is False
    assert step["transferred_tokens"] == 0
    assert step["processed_tokens"] == step["prompt_tokens"]
    assert step["fallback_reason"] == "degradation_threshold_exceeded"


def test_missing_adapter_has_explicit_fallback():
    pipeline = Pipeline(fake_models(), {"cache_policy": "reuse", "max_new_tokens": 1})
    _, logs, _ = pipeline.run_steps("One plus one?", ["model_1", "model_2"])
    assert logs["steps"][1]["fallback_reason"] == "adapter_missing"


def test_context_limit_is_reported_without_truncation():
    pipeline = Pipeline(fake_models(), {"context_limit": 2})
    result, logs = pipeline.run("too long", ["model_1"] * 3)
    assert result == ""
    assert logs["failed"] is True
    assert logs["steps"][0]["truncated"] is False
    assert logs["steps"][0]["failure_reason"].startswith("context_limit_")
