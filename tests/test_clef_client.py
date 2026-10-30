import pytest
import requests

import clef_client as cc

Q = {"a": {"type": "noul", "instructions": "x"}}


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body = status, body
        self.ok = status < 400
        self.text = str(body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def patch_post(monkeypatch, result):
    def fake(url, json, timeout):
        assert url.endswith("/v1/systemone") and json["model"] == "clef"
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(cc.requests, "post", fake)


def test_success(monkeypatch):
    patch_post(monkeypatch, Resp(200, {"model": "clef", "answers": {"a": {"type": "noul", "noul": 0.9}},
                                       "usage": {"input_tokens": 5, "output_tokens": 0},
                                       "prompt_eval_cached_count": 3}))
    r = cc.run_clef("s", Q)
    assert r.answers["a"]["noul"] == 0.9 and r.input_tokens == 5 and r.cached_tokens == 3


@pytest.mark.parametrize("result, match", [
    (requests.exceptions.ConnectionError(), "ollama serve"),
    (requests.exceptions.ReadTimeout(), "did not respond"),
    (Resp(404, {"error": 'model "clef" not found, try pulling it first'}), "ollama pull"),
    (Resp(400, {"error": "x does not support decision"}), "not a decision model"),
    (Resp(500, {"error": "boom"}), "HTTP 500"),
    (Resp(200, ValueError("bad json")), "not JSON"),
    (Resp(200, {"model": "clef"}), "missing 'answers'"),
    (Resp(200, {"answers": {"a": {"type": "noul"}}}), "numeric 'noul'"),
])
def test_errors(monkeypatch, result, match):
    patch_post(monkeypatch, result)
    with pytest.raises(cc.ClefError, match=match):
        cc.run_clef("s", Q)


def test_typesafe_requires_key():
    with pytest.raises(cc.ClefError, match="TypeSafe API key"):
        cc.run_typesafe("s", Q, api_key="")


def test_typesafe_success_and_auth_header(monkeypatch):
    seen = {}

    def fake(url, json=None, timeout=None, headers=None):
        seen["url"] = url
        seen["headers"] = headers
        seen["json"] = json
        return Resp(200, {"model": "jev-1.13.0", "answers": {"a": {"type": "noul", "noul": 0.42}},
                          "usage": {"input_tokens": 9, "output_tokens": 2}})

    monkeypatch.setattr(cc.requests, "post", fake)
    r = cc.run_typesafe("hello", Q, api_key="sk-test", model="jev-latest")
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["headers"]["Authorization"] == "Bearer sk-test"
    assert seen["json"]["model"] == "jev-latest"
    assert r.answers["a"]["noul"] == 0.42 and r.input_tokens == 9 and r.output_tokens == 2


def test_typesafe_401(monkeypatch):
    monkeypatch.setattr(
        cc.requests, "post",
        lambda *a, **k: Resp(401, {"error": {"message": "invalid api key"}}),
    )
    with pytest.raises(cc.ClefError, match="rejected the API key"):
        cc.run_typesafe("s", Q, api_key="bad")


def test_check_ollama_lists_generate_and_qwen(monkeypatch):
    def fake_get(url, timeout=None):
        if url.endswith("/api/version"):
            return Resp(200, {"version": "0.40.0"})
        return Resp(200, {"models": [
            {"name": "clef:latest", "capabilities": ["decision"]},
            {"name": "qwen2.5:7b", "capabilities": ["completion"]},
            {"name": "llama3.2", "capabilities": ["completion"]},
        ]})

    monkeypatch.setattr(cc.requests, "get", fake_get)
    st = cc.check_ollama("http://localhost:11434")
    assert st.online and st.version == "0.40.0"
    assert "clef:latest" in st.decision_models
    assert st.qwen_models() == ["qwen2.5:7b"]
    assert st.has_generate_model("qwen2.5:7b")
