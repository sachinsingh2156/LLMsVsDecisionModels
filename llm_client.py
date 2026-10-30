"""Gemini and local Ollama Qwen adapters that return Clef-shaped decision JSON.

These models generate text, so they are prompted to emit the same ``noul`` /
``choice`` answer objects that System One models return natively. Probabilities
are model-reported, not calibrated like Clef or Jev.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import requests

from clef_client import (
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT_S,
    ClefError,
    ClefResult,
    validate_systemone_response,
)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_MODELS = ("gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.1-pro-preview")

LLM_SYSTEM = (
    "You are a careful analyst. Think step by step about every question. "
    "Return JSON only, no markdown, in this shape:\n"
    '{"answers": {<question_id>: <answer>}}\n\n'
    "Every answer MUST include a 'reasoning' string: a detailed multi-step analysis "
    "(at least 10 sentences) that quotes or paraphrases the relevant parts of the state, "
    "considers counterarguments, and only then reaches a decision.\n\n"
    'For type "noul": {"type": "noul", "reasoning": "...", "noul": <float 0 to 1>}\n'
    "noul is the probability that the TRUE / positive criteria hold.\n\n"
    'For type "choice": {"type": "choice", "reasoning": "...", "choice": <id>, '
    '"confidence": <float 0-1>, "probabilities": {<id>: <float>, ...}}\n'
    "probabilities must include every choice id and should sum to about 1. "
    "choice is the id with the highest probability. "
    "confidence is your confidence in the selected choice; it is not the same as "
    "that choice's probability."
)


def build_decision_prompt(state: str | dict[str, Any], questions: dict[str, Any]) -> str:
    """User message that asks a generative model for System One-shaped JSON."""
    n = len(questions)
    return (
        f"Evaluate the state against all {n} question(s). "
        "Work slowly. Do not skip a question. Each answer needs a long 'reasoning' field "
        "before the decision fields. Return JSON only.\n\n"
        f"state:\n{json.dumps(state, ensure_ascii=False, indent=2)}\n\n"
        f"questions:\n{json.dumps(questions, ensure_ascii=False, indent=2)}"
    )


def gemini_output_tokens(usage: dict[str, Any]) -> int | None:
    """Candidate tokens plus thinking tokens, so comparison tables show Gemini's real spend."""
    cand = usage.get("candidatesTokenCount")
    thoughts = usage.get("thoughtsTokenCount")
    if cand is None and thoughts is None:
        return None
    return int(cand or 0) + int(thoughts or 0)


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_content(text: str) -> Any:
    """Parse a JSON object, allowing optional markdown fences."""
    raw = _strip_fences(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise ClefError("Malformed LLM response: body is not JSON.", body=text)


def _as_prob(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        x = float(value)
        if 1 < x <= 100:
            x = x / 100.0
        return min(max(x, 0.0), 1.0)
    return None


def _probabilities_from(ans: dict[str, Any], criteria: dict[str, Any] | None) -> dict[str, float]:
    raw = ans.get("probabilities")
    out: dict[str, float] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            p = _as_prob(v)
            if p is not None:
                out[str(k)] = p
    elif isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            cid = item.get("id") or item.get("choice")
            p = _as_prob(item.get("p", item.get("probability", item.get("prob"))))
            if cid is not None and p is not None:
                out[str(cid)] = p
    ids = [str(k) for k in (criteria or {})] or list(out)
    choice = ans.get("choice") or ans.get("selected")
    if choice is not None and str(choice) not in out:
        out[str(choice)] = 1.0
    for cid in ids:
        out.setdefault(cid, 0.0)
    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    elif choice is not None:
        out[str(choice)] = 1.0
    return out


def coerce_llm_answers(data: Any, questions: dict[str, Any]) -> dict[str, Any]:
    """Turn a generative model's JSON into a System One ``answers`` map."""
    if not isinstance(data, dict):
        raise ClefError("Malformed LLM response: expected a JSON object.", body=data)

    answers: Any = data.get("answers", data)
    if isinstance(answers, list):
        mapped: dict[str, Any] = {}
        for item in answers:
            if isinstance(item, dict) and item.get("id"):
                mapped[str(item["id"])] = item
        answers = mapped
    if not isinstance(answers, dict):
        raise ClefError("Malformed LLM response: missing 'answers' object.", body=data)

    out: dict[str, Any] = {}
    for qid, q in questions.items():
        ans = answers.get(qid)
        if ans is None and len(questions) == 1 and (
            "noul" in data or "choice" in data or "selected" in data
        ):
            ans = data
        if not isinstance(ans, dict):
            raise ClefError(f"Malformed LLM response: no answer for '{qid}'.", body=data)

        qtype = q.get("type")
        coerced: dict[str, Any] = {"type": qtype}
        if qtype == "noul":
            noul = _as_prob(ans.get("noul", ans.get("p_true", ans.get("probability"))))
            if noul is None and isinstance(ans.get("label"), str):
                label = ans["label"].strip().upper()
                if label in {"TRUE", "YES"}:
                    noul = 1.0
                elif label in {"FALSE", "NO"}:
                    noul = 0.0
            if noul is None:
                raise ClefError(f"Malformed LLM response: '{qid}' has no numeric 'noul'.", body=data)
            coerced["noul"] = noul
        elif qtype == "choice":
            choice = ans.get("choice", ans.get("selected"))
            if choice is None:
                raise ClefError(f"Malformed LLM response: '{qid}' has no 'choice'.", body=data)
            probs = _probabilities_from(ans, q.get("criteria") if isinstance(q, dict) else None)
            coerced["choice"] = str(choice)
            coerced["probabilities"] = probs
            conf = _as_prob(ans.get("confidence"))
            coerced["confidence"] = conf if conf is not None else probs.get(str(choice), 0.0)
        else:
            coerced.update({k: v for k, v in ans.items() if k != "id"})
        out[qid] = coerced
    return out


def answers_from_llm_json(data: Any, questions: dict[str, Any]) -> dict[str, Any]:
    answers = coerce_llm_answers(data, questions)
    validate_systemone_response({"answers": answers}, questions, label="LLM")
    return answers


def _gemini_text(data: dict[str, Any]) -> str:
    cands = data.get("candidates") or []
    if not cands:
        feedback = data.get("promptFeedback") or data.get("error") or data
        raise ClefError(f"Gemini returned no candidates: {feedback}", body=data)
    parts = ((cands[0].get("content") or {}).get("parts")) or []
    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    text = "\n".join(t for t in texts if t)
    if not text:
        raise ClefError("Gemini returned an empty candidate.", body=data)
    return text


def run_gemini(
    state: str | dict[str, Any],
    questions: dict[str, Any],
    *,
    api_key: str,
    model: str = DEFAULT_GEMINI_MODEL,
    base_url: str = GEMINI_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT_S,
    thinking_level: str = "high",
) -> ClefResult:
    """Ask Gemini to answer System One questions as JSON."""
    if not (api_key or "").strip():
        raise ClefError("Gemini API key is required. Set GEMINI_API_KEY or paste it in the sidebar.")
    model_id = model.removeprefix("models/")
    url = f"{base_url.rstrip('/')}/models/{model_id}:generateContent"
    prompt = build_decision_prompt(state, questions)
    generation_config: dict[str, Any] = {
        "temperature": 0,
        "responseMimeType": "application/json",
    }
    level = (thinking_level or "").strip()
    if level and level.lower() not in {"none", "off"}:
        generation_config["thinkingConfig"] = {"thinkingLevel": level}
    payload = {
        "systemInstruction": {"parts": [{"text": LLM_SYSTEM}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": generation_config,
    }
    start = time.perf_counter()
    try:
        resp = requests.post(
            url,
            params={"key": api_key.strip()},
            json=payload,
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError as e:
        raise ClefError(f"Could not connect to Gemini at {base_url}.") from e
    except requests.exceptions.Timeout as e:
        raise ClefError(f"Gemini did not respond within {timeout:.0f} s.") from e
    except requests.exceptions.RequestException as e:
        raise ClefError(f"Request to Gemini failed: {e}") from e
    latency_ms = (time.perf_counter() - start) * 1000

    if not resp.ok:
        try:
            body = resp.json()
            err = body.get("error", body) if isinstance(body, dict) else body
            detail = err.get("message", err) if isinstance(err, dict) else err
        except ValueError:
            body, detail = resp.text, resp.text
        if resp.status_code in (401, 403):
            raise ClefError(
                f"Gemini rejected the API key (HTTP {resp.status_code}). {detail}",
                status_code=resp.status_code,
                body=body,
            )
        raise ClefError(
            f"Gemini returned HTTP {resp.status_code}: {detail}",
            status_code=resp.status_code,
            body=body,
        )
    try:
        data = resp.json()
    except ValueError as e:
        raise ClefError("Malformed Gemini response: body is not JSON.", body=resp.text) from e

    parsed = parse_json_content(_gemini_text(data))
    answers = answers_from_llm_json(parsed, questions)
    usage = data.get("usageMetadata") or {}
    wrapped = {
        "model": model_id,
        "answers": answers,
        "usage": {
            "input_tokens": usage.get("promptTokenCount"),
            "output_tokens": gemini_output_tokens(usage),
            "cached_tokens": usage.get("cachedContentTokenCount"),
        },
    }
    request = {"model": model_id, "state": state, "questions": questions, "provider": "gemini"}
    return ClefResult(response=wrapped, latency_ms=latency_ms, request=request)


def run_qwen(
    state: str | dict[str, Any],
    questions: dict[str, Any],
    *,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = DEFAULT_TIMEOUT_S,
) -> ClefResult:
    """Ask a local Ollama Qwen (or any chat) model for System One-shaped JSON."""
    if not (model or "").strip():
        raise ClefError("A Qwen / Ollama generate model name is required.")
    url = f"{base_url.rstrip('/')}/api/chat"
    prompt = build_decision_prompt(state, questions)
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    start = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as e:
        raise ClefError(
            f"Could not connect to Ollama at {base_url}.\n\nMake sure Ollama is running:\n    ollama serve"
        ) from e
    except requests.exceptions.Timeout as e:
        raise ClefError(
            f"Qwen did not respond within {timeout:.0f} s. The model may still be loading; try again."
        ) from e
    except requests.exceptions.RequestException as e:
        raise ClefError(f"Request to Ollama failed: {e}") from e
    latency_ms = (time.perf_counter() - start) * 1000

    if not resp.ok:
        try:
            body = resp.json()
            detail = body.get("error", resp.text) if isinstance(body, dict) else resp.text
        except ValueError:
            body, detail = resp.text, resp.text
        if resp.status_code == 404 or (isinstance(detail, str) and "not found" in detail.lower()):
            raise ClefError(
                f"Model '{model}' was not found in Ollama.\n\n"
                f"Pull it first:\n    ollama pull {model}\n\nOllama said: {detail}",
                status_code=resp.status_code,
                body=body,
            )
        raise ClefError(
            f"Ollama returned HTTP {resp.status_code}: {detail}",
            status_code=resp.status_code,
            body=body,
        )
    try:
        data = resp.json()
    except ValueError as e:
        raise ClefError("Malformed Ollama response: body is not JSON.", body=resp.text) from e
    if isinstance(data, dict) and data.get("error"):
        raise ClefError(f"Ollama returned an error: {data['error']}", body=data)

    content = ((data.get("message") or {}).get("content")) if isinstance(data, dict) else None
    if not content:
        raise ClefError("Ollama returned an empty chat message.", body=data)
    parsed = parse_json_content(content) if isinstance(content, str) else content
    answers = answers_from_llm_json(parsed, questions)
    wrapped = {
        "model": data.get("model", model),
        "answers": answers,
        "usage": {
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
        },
    }
    request = {"model": model, "state": state, "questions": questions, "provider": "qwen"}
    return ClefResult(response=wrapped, latency_ms=latency_ms, request=request)


@dataclass
class GeminiStatus:
    online: bool
    error: str | None = None


def check_gemini(
    api_key: str,
    base_url: str = GEMINI_BASE_URL,
    timeout: float = 3.0,
) -> GeminiStatus:
    """Cheap health check: GET /models. Never generates content."""
    if not (api_key or "").strip():
        return GeminiStatus(online=False, error="Gemini API key is missing.")
    try:
        resp = requests.get(
            f"{base_url.rstrip('/')}/models",
            params={"key": api_key.strip()},
            timeout=timeout,
        )
    except requests.exceptions.RequestException:
        return GeminiStatus(online=False, error=f"Could not reach Gemini at {base_url}.")
    if resp.status_code in (401, 403):
        return GeminiStatus(online=False, error="Gemini API key was rejected.")
    if not resp.ok:
        return GeminiStatus(online=False, error=f"Gemini HTTP {resp.status_code}.")
    return GeminiStatus(online=True)
