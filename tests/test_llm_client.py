import pytest
import requests

import llm_client as lc


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.ok = status < 400
        self.text = str(body)

    def json(self):
        return self._body

Q_NOUL = {"a": {"type": "noul", "instructions": "x"}}
Q_CHOICE = {"m": {"type": "choice", "instructions": "which",
                  "criteria": {"x": "ex", "y": "why"}}}


def test_parse_json_fences_and_embedded():
    assert lc.parse_json_content('```json\n{"noul": 1}\n```') == {"noul": 1}
    assert lc.parse_json_content('prefix {"ok": true} trailing') == {"ok": True}
    with pytest.raises(lc.ClefError, match="not JSON"):
        lc.parse_json_content("nope")


def test_coerce_noul_percent_and_label():
    answers = lc.answers_from_llm_json({"answers": {"a": {"noul": 80}}}, Q_NOUL)
    assert answers["a"]["noul"] == pytest.approx(0.8)
    answers = lc.answers_from_llm_json({"noul": 0.1}, Q_NOUL)
    assert answers["a"]["noul"] == pytest.approx(0.1)
    answers = lc.answers_from_llm_json({"answers": {"a": {"label": "TRUE"}}}, Q_NOUL)
    assert answers["a"]["noul"] == 1.0


def test_coerce_choice_fills_missing_probs():
    answers = lc.answers_from_llm_json({"answers": {"m": {"choice": "x"}}}, Q_CHOICE)
    assert answers["m"]["choice"] == "x"
    assert answers["m"]["probabilities"]["x"] == pytest.approx(1.0)
    assert answers["m"]["probabilities"]["y"] == pytest.approx(0.0)
    assert answers["m"]["confidence"] == pytest.approx(1.0)


def test_coerce_rejects_missing_noul():
    with pytest.raises(lc.ClefError, match="numeric 'noul'"):
        lc.answers_from_llm_json({"answers": {"a": {"choice": "x"}}}, Q_NOUL)


def test_run_gemini(monkeypatch):
    def fake(url, params=None, json=None, timeout=None):
        assert "gemini-3.8-flash:generateContent" in url
        assert params["key"] == "gk"
        assert json["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "high"
        body = {
            "candidates": [{"content": {"parts": [{"text": '{"answers":{"a":{"noul":0.75}}}'}]}}],
            "usageMetadata": {
                "promptTokenCount": 10, "candidatesTokenCount": 4,
                "thoughtsTokenCount": 20, "cachedContentTokenCount": 2,
            },
        }
        return Resp(200, body)

    monkeypatch.setattr(lc.requests, "post", fake)
    r = lc.run_gemini("s", Q_NOUL, api_key="gk")
    assert r.answers["a"]["noul"] == 0.75
    assert r.input_tokens == 10 and r.output_tokens == 24 and r.cached_tokens == 2


def test_run_gemini_thinking_level_override(monkeypatch):
    def fake(url, params=None, json=None, timeout=None):
        assert json["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "low"
        body = {
            "candidates": [{"content": {"parts": [{"text": '{"answers":{"a":{"noul":0.2}}}'}]}}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
        }
        return Resp(200, body)

    monkeypatch.setattr(lc.requests, "post", fake)
    r = lc.run_gemini("s", Q_NOUL, api_key="gk", thinking_level="low")
    assert r.answers["a"]["noul"] == 0.2


def test_coerce_strips_reasoning():
    answers = lc.answers_from_llm_json(
        {"answers": {"a": {"reasoning": "long analysis " * 20, "noul": 0.2}}}, Q_NOUL
    )
    assert answers["a"] == {"type": "noul", "noul": 0.2}


def test_gemini_output_tokens_adds_thinking():
    assert lc.gemini_output_tokens({"candidatesTokenCount": 3, "thoughtsTokenCount": 7}) == 10
    assert lc.gemini_output_tokens({}) is None


def test_run_gemini_requires_key():
    with pytest.raises(lc.ClefError, match="Gemini API key"):
        lc.run_gemini("s", Q_NOUL, api_key="")


def test_run_gemini_rejects_bad_key(monkeypatch):
    monkeypatch.setattr(
        lc.requests, "post",
        lambda *a, **k: Resp(403, {"error": {"message": "API key not valid"}}),
    )
    with pytest.raises(lc.ClefError, match="rejected the API key"):
        lc.run_gemini("s", Q_NOUL, api_key="bad")


def test_run_qwen(monkeypatch):
    def fake(url, json=None, timeout=None):
        assert url.endswith("/api/chat")
        assert json["format"] == "json" and json["model"] == "qwen2.5"
        return Resp(200, {
            "model": "qwen2.5",
            "message": {"content": '{"answers":{"m":{"choice":"y","probabilities":{"x":0.2,"y":0.8}}}}'},
            "prompt_eval_count": 11,
            "eval_count": 5,
        })

    monkeypatch.setattr(lc.requests, "post", fake)
    r = lc.run_qwen("s", Q_CHOICE, model="qwen2.5")
    assert r.answers["m"]["choice"] == "y"
    assert r.answers["m"]["probabilities"]["y"] == pytest.approx(0.8)
    assert r.input_tokens == 11 and r.output_tokens == 5


def test_run_qwen_timeout(monkeypatch):
    def fake(*a, **k):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(lc.requests, "post", fake)
    with pytest.raises(lc.ClefError, match="did not respond"):
        lc.run_qwen("s", Q_NOUL, model="qwen2.5")


def test_build_decision_prompt_includes_state_and_questions():
    prompt = lc.build_decision_prompt("Hello", Q_NOUL)
    assert "Hello" in prompt and "noul" in prompt
