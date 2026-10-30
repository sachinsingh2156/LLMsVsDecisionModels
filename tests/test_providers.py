import pytest

from clef_client import ClefError, ClefResult
from providers import ProviderConfig, run_selected


def test_run_selected_continues_after_failure(monkeypatch):
    import providers as pr

    def fake_one(state, questions, cfg):
        if cfg.provider == "clef":
            raise ClefError("Clef down")
        return ClefResult(
            response={"model": cfg.model, "answers": {"a": {"type": "noul", "noul": 0.6}},
                      "usage": {"input_tokens": 3, "output_tokens": 1}},
            latency_ms=12.0,
            request={"model": cfg.model},
        )

    monkeypatch.setattr(pr, "run_one", fake_one)
    runs = run_selected("s", {"a": {"type": "noul"}}, [
        ProviderConfig("clef", "clef"),
        ProviderConfig("gemini", "gemini-3.8-flash", enabled=False),
        ProviderConfig("typesafe", "jev-latest"),
    ])
    assert len(runs) == 2
    assert runs[0].provider == "clef" and runs[0].error and "Clef down" in runs[0].error
    assert runs[1].ok and runs[1].answers["a"]["noul"] == 0.6
    assert runs[1].total_tokens == 4
    assert runs[1].tokens_per_sec == pytest.approx(1 / 0.012)
