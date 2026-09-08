from kvbridge import Pipeline, evaluate
from kvbridge.evaluation import numeric_answer, report_markdown

from conftest import fake_models


def test_numeric_answer_requires_marker_and_preserves_integer_zeros():
    assert numeric_answer("answer 100") is None
    assert numeric_answer("#### 100") == "100"
    assert numeric_answer("#### 1,200.500") == "1200.5"
    assert numeric_answer("#### -0.0") == "0"


def test_evaluation_reports_paired_metrics(tmp_path):
    config = {
        "cache_policy": "reuse",
        "max_new_tokens": 1,
        "warmup": False,
        "records_path": str(tmp_path / "records.jsonl"),
        "report_path": str(tmp_path / "report.json"),
    }
    pipeline = Pipeline(fake_models(), config)
    dataset = [
        {"question": "one plus one", "answer": "work\n#### 2"},
        {"question": "also one plus one", "answer": "#### 2"},
    ]
    assignments = [
        ["model_1"] * 3,
        ["model_2"] * 3,
        ["model_3"] * 3,
        ["model_1", "model_2", "model_3"],
    ]
    report = evaluate(pipeline, dataset, assignments, config)
    setting = report["settings"]["model_1 -> model_1 -> model_1"]
    assert setting["disabled"]["accuracy"] == 1.0
    assert setting["reuse"]["exact_reuse_rate"] > 0
    assert setting["paired_answer_disagreements"] == 0
    assert len((tmp_path / "records.jsonl").read_text().splitlines()) == 16
    assert "Total speedup" in report_markdown(report)
