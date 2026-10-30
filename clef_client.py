"""Clients for System One decision models: local Clef (Ollama) and TypeSafe Jev.

Clef and Jev evaluate predefined decisions and return probabilities rather than
generating a conventional text response. Both are called through
``POST /v1/systemone`` with a ``state`` and a set of ``questions``.
``/api/generate`` and ``/api/chat`` are never used here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import requests

DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "clef"
DEFAULT_TIMEOUT_S = 300.0  # a cold load of a 27.5B model can take a while

TYPESAFE_BASE_URL = "https://api.typesafe.ai"
DEFAULT_JEV_MODEL = "jev-latest"
JEV_MODELS = ("jev-latest", "jev-preview", "jev-1.14.0", "jev-1.13.0")


class ClefError(Exception):
    """A user-presentable error from talking to a decision / LLM provider."""

    def __init__(self, message: str, *, status_code: int | None = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


ProviderError = ClefError


@dataclass
class ClefResult:
    """A successful decision response plus request metadata."""

    response: dict[str, Any]
    latency_ms: float
    request: dict[str, Any] = field(default_factory=dict)

    @property
    def answers(self) -> dict[str, Any]:
        return self.response["answers"]

    @property
    def model(self) -> str:
        return self.response.get("model", self.request.get("model", ""))

    @property
    def input_tokens(self) -> int | None:
        return (self.response.get("usage") or {}).get("input_tokens")

    @property
    def output_tokens(self) -> int | None:
        return (self.response.get("usage") or {}).get("output_tokens")

    @property
    def cached_tokens(self) -> int | None:
        usage = self.response.get("usage") or {}
        cached = usage.get("cached_tokens")
        if cached is not None:
            return cached
        return self.response.get("prompt_eval_cached_count")


def _offline_message(base_url: str) -> str:
    return (
        f"Could not connect to Ollama at {base_url}.\n\n"
        "Make sure Ollama is running:\n    ollama serve"
    )


def _error_from_http(
    resp: requests.Response,
    model: str,
    *,
    host_label: str = "Ollama",
) -> ClefError:
    """Translate an HTTP error into a helpful message."""
    try:
        body = resp.json()
        if isinstance(body, dict):
            err = body.get("error", body)
            if isinstance(err, dict):
                detail = err.get("message") or err.get("error") or str(err)
            else:
                detail = err if isinstance(err, str) else resp.text
        else:
            detail = resp.text
    except ValueError:
        body, detail = resp.text, resp.text

    detail_s = str(detail)
    if resp.status_code in (401, 403):
        msg = (
            f"{host_label} rejected the API key (HTTP {resp.status_code}). "
            f"Check the key in the sidebar or your environment.\n\n{detail_s}"
        )
    elif resp.status_code == 429:
        msg = f"{host_label} rate-limited the request (HTTP 429). Wait and retry.\n\n{detail_s}"
    elif resp.status_code == 404 and "not found" in detail_s.lower() and host_label == "Ollama":
        msg = (
            f"Model '{model}' was not found in Ollama.\n\n"
            f"Pull it first:\n    ollama pull {model}\n\nOllama said: {detail_s}"
        )
    elif "does not support decision" in detail_s:
        msg = (
            f"Model '{model}' is not a decision model, so it cannot serve "
            f"/v1/systemone. Choose a decision model such as 'clef'.\n\nOllama said: {detail_s}"
        )
    elif resp.status_code == 404 and host_label == "Ollama":
        msg = (
            "Ollama returned 404 for /v1/systemone. Your Ollama build may be too old "
            f"for decision models (Clef requires 0.40.0+).\n\nOllama said: {detail_s}"
        )
    else:
        msg = f"{host_label} returned HTTP {resp.status_code}: {detail_s}"
    return ClefError(msg, status_code=resp.status_code, body=body)


def validate_systemone_response(data: Any, questions: dict[str, Any], *, label: str = "Clef") -> None:
    """Raise ClefError if the response does not look like a systemone answer."""
    if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
        raise ClefError(f"Malformed {label} response: missing 'answers' object.", body=data)
    for qid, q in questions.items():
        ans = data["answers"].get(qid)
        if not isinstance(ans, dict):
            raise ClefError(f"Malformed {label} response: no answer for '{qid}'.", body=data)
        qtype = q.get("type")
        if qtype == "noul" and not isinstance(ans.get("noul"), (int, float)):
            raise ClefError(f"Malformed {label} response: '{qid}' has no numeric 'noul'.", body=data)
        if qtype == "choice" and not (
            isinstance(ans.get("choice"), str) and isinstance(ans.get("probabilities"), dict)
        ):
            raise ClefError(
                f"Malformed {label} response: '{qid}' lacks 'choice'/'probabilities'.", body=data
            )


_validate_response = validate_systemone_response  # backward-compatible alias


def run_systemone(
    state: str | dict[str, Any],
    questions: dict[str, Any],
    *,
    model: str,
    base_url: str,
    headers: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT_S,
    host_label: str = "Ollama",
) -> ClefResult:
    """POST ``/v1/systemone`` and return a validated decision result."""
    url = f"{base_url.rstrip('/')}/v1/systemone"
    payload = {"model": model, "state": state, "questions": questions}
    kwargs: dict[str, Any] = {"json": payload, "timeout": timeout}
    if headers:
        kwargs["headers"] = headers

    start = time.perf_counter()
    try:
        resp = requests.post(url, **kwargs)
    except requests.exceptions.ConnectionError as e:
        if host_label == "Ollama":
            raise ClefError(_offline_message(base_url)) from e
        raise ClefError(f"Could not connect to {host_label} at {base_url}.") from e
    except requests.exceptions.Timeout as e:
        who = "Clef" if host_label == "Ollama" else host_label
        raise ClefError(
            f"{who} did not respond within {timeout:.0f} s. The model may still be "
            "loading into memory; try again, or increase the timeout."
        ) from e
    except requests.exceptions.RequestException as e:
        raise ClefError(f"Request to {host_label} failed: {e}") from e
    latency_ms = (time.perf_counter() - start) * 1000

    if not resp.ok:
        raise _error_from_http(resp, model, host_label=host_label)
    try:
        data = resp.json()
    except ValueError as e:
        raise ClefError(f"Malformed {host_label} response: body is not JSON.", body=resp.text) from e
    if isinstance(data, dict) and "error" in data:
        err = data["error"]
        detail = err.get("message", err) if isinstance(err, dict) else err
        raise ClefError(f"{host_label} returned an error: {detail}", body=data)
    validate_systemone_response(data, questions, label=host_label)
    return ClefResult(response=data, latency_ms=latency_ms, request=payload)


def run_clef(
    state: str | dict[str, Any],
    questions: dict[str, Any],
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> ClefResult:
    """Ask Clef one or more decision questions about ``state``."""
    return run_systemone(
        state, questions, model=model, base_url=base_url, timeout=timeout, host_label="Ollama",
    )


def run_typesafe(
    state: str | dict[str, Any],
    questions: dict[str, Any],
    *,
    api_key: str,
    model: str = DEFAULT_JEV_MODEL,
    base_url: str = TYPESAFE_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> ClefResult:
    """Ask TypeSafe Jev the same System One questions as Clef."""
    if not (api_key or "").strip():
        raise ClefError(
            "TypeSafe API key is required. Set TYPESAFE_API_KEY or paste it in the sidebar."
        )
    return run_systemone(
        state,
        questions,
        model=model,
        base_url=base_url,
        headers={"Authorization": f"Bearer {api_key.strip()}", "Content-Type": "application/json"},
        timeout=timeout,
        host_label="TypeSafe",
    )


@dataclass
class OllamaStatus:
    online: bool
    version: str | None = None
    decision_models: list[str] = field(default_factory=list)
    generate_models: list[str] = field(default_factory=list)
    error: str | None = None

    def has_model(self, model: str) -> bool:
        names = set(self.decision_models)
        return model in names or f"{model}:latest" in names

    def has_generate_model(self, model: str) -> bool:
        names = set(self.generate_models)
        return model in names or f"{model}:latest" in names

    def qwen_models(self) -> list[str]:
        return [m for m in self.generate_models if "qwen" in m.lower()]


def check_ollama(base_url: str = DEFAULT_BASE_URL, timeout: float = 3.0) -> OllamaStatus:
    """Cheap health check: /api/version + /api/tags. Never calls Clef itself."""
    base = base_url.rstrip("/")
    try:
        version = requests.get(f"{base}/api/version", timeout=timeout).json().get("version")
    except (requests.exceptions.RequestException, ValueError):
        return OllamaStatus(online=False, error=_offline_message(base_url))

    decision_models: list[str] = []
    generate_models: list[str] = []
    try:
        for m in requests.get(f"{base}/api/tags", timeout=timeout).json().get("models", []):
            name = m.get("name")
            if not name:
                continue
            generate_models.append(name)
            caps = m.get("capabilities") or []
            if "decision" in caps:
                decision_models.append(name)
    except (requests.exceptions.RequestException, ValueError, KeyError):
        pass
    return OllamaStatus(
        online=True,
        version=version,
        decision_models=sorted(set(decision_models)),
        generate_models=sorted(set(generate_models)),
    )


@dataclass
class TypeSafeStatus:
    online: bool
    models: list[str] = field(default_factory=list)
    error: str | None = None


def check_typesafe(
    api_key: str,
    base_url: str = TYPESAFE_BASE_URL,
    timeout: float = 3.0,
) -> TypeSafeStatus:
    """Cheap health check: GET /v1/models. Never evaluates a decision."""
    if not (api_key or "").strip():
        return TypeSafeStatus(online=False, error="TypeSafe API key is missing.")
    url = f"{base_url.rstrip('/')}/v1/models"
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {api_key.strip()}"},
            timeout=timeout,
        )
    except requests.exceptions.RequestException:
        return TypeSafeStatus(online=False, error=f"Could not reach TypeSafe at {base_url}.")
    if resp.status_code in (401, 403):
        return TypeSafeStatus(online=False, error="TypeSafe API key was rejected.")
    if not resp.ok:
        return TypeSafeStatus(online=False, error=f"TypeSafe HTTP {resp.status_code}.")
    names: list[str] = []
    try:
        body = resp.json()
        items = body.get("models", body) if isinstance(body, dict) else body
        if isinstance(items, list):
            for m in items:
                if isinstance(m, dict):
                    n = m.get("name") or m.get("id")
                    if n:
                        names.append(str(n))
                elif isinstance(m, str):
                    names.append(m)
    except ValueError:
        pass
    return TypeSafeStatus(online=True, models=names)
