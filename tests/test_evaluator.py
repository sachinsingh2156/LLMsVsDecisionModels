import pandas as pd
import pytest

import evaluator as ev


def test_parse_bool():
    assert ev.parse_bool("TRUE") is True and ev.parse_bool(0) is False
    assert ev.parse_bool(float("nan")) is None and ev.parse_bool("maybe") is None


def test_binary_metrics():
    m = ev.binary_metrics(pd.Series([True, True, False, False]), pd.Series([True, False, True, False]))
    assert m["accuracy"] == 0.5 and m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5


def test_multiclass_metrics():
    m = ev.multiclass_metrics(pd.Series(["a", "a", "b", "c"]), pd.Series(["a", "b", "b", "a"]), ["a", "b", "c"])
    pc = m["per_class"].set_index("class")
    assert m["accuracy"] == 0.5
    assert pc.loc["a", "precision"] == 0.5 and pc.loc["a", "recall"] == 0.5
    assert pc.loc["b", "precision"] == 0.5 and pc.loc["b", "recall"] == 1.0
    assert m["macro_f1"] == pytest.approx((0.5 + 2 / 3 + 0) / 3)
    assert m["confusion"].loc["c", "a"] == 1


def test_evaluate_and_export(monkeypatch):
    class R:
        latency_ms, input_tokens = 100.0, 10
        answers = {
            "correct": {"type": "noul", "noul": 0.2},
            "misconception": {"type": "choice", "choice": "x", "confidence": 0.7,
                              "probabilities": {"x": 0.9, "y": 0.1}},
        }

    calls = []
    def fake_run(state, questions, **kw):
        calls.append(state)
        if state["student_answer"] == "boom":
            raise ev.ClefError("Could not connect to Ollama.\nmore")
        return R()

    monkeypatch.setattr(ev, "run_clef", fake_run)
    df = pd.DataFrame({"id": [1, 2], "question": ["q", "q"], "student_answer": ["2/5", "boom"],
                       "ground_truth_correct": ["false", "true"], "ground_truth_misconception": ["x", None]})
    progress = []
    res = ev.evaluate_dataset(df, {"correct": {}, "misconception": {}}, model="clef", base_url="u",
                              on_progress=lambda i, n: progress.append((i, n)))
    assert progress == [(1, 2), (2, 2)] and calls[0] == {"question": "q", "student_answer": "2/5"}
    assert res.loc[0, "pred_correct"] == False and res.loc[1, "error"] == "Could not connect to Ollama."
    assert ev.latency_stats(res)["failed_requests"] == 1
    assert ev.correctness_metrics(res)["accuracy"] == 1.0
    out = ev.export_table(res)
    assert list(out.columns[:7]) == ev.EXPORT_COLUMNS and len(out) == 4


def test_evaluate_models_and_export_includes_provider():
    class R:
        latency_ms, input_tokens, output_tokens, cached_tokens = 50.0, 8, 2, None
        answers = {"correct": {"type": "noul", "noul": 0.9}}

    def run_ok(state, questions, **kw):
        return R()

    def run_fail(state, questions, **kw):
        raise ev.ClefError("Gemini down")

    df = pd.DataFrame({"id": [1], "question": ["q"], "student_answer": ["2/5"],
                       "ground_truth_correct": ["true"]})
    progress = []
    res = ev.evaluate_models(
        df, {"correct": {}},
        [("clef", "clef", run_ok), ("gemini", "gemini-3.8-flash", run_fail)],
        on_progress=lambda i, n: progress.append((i, n)),
    )
    assert progress == [(1, 2), (2, 2)]
    assert set(res["provider"]) == {"clef", "gemini"}
    assert res.loc[res["provider"] == "clef", "pred_correct"].iloc[0] == True
    assert res.loc[res["provider"] == "gemini", "error"].iloc[0] == "Gemini down"
    cmp = ev.model_comparison(res)
    assert list(cmp["provider"]) == ["clef", "gemini"] or set(cmp["provider"]) == {"clef", "gemini"}
    assert ev.token_stats(res[res["provider"] == "clef"])["avg_input_tokens"] == 8
    out = ev.export_table(res)
    assert "provider" in out.columns and "model" in out.columns and "input_tokens" in out.columns
