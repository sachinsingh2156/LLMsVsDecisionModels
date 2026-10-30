"""Unified runner that calls Clef, TypeSafe Jev, Gemini, and local Qwen."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from clef_client import ClefError, ClefResult, run_clef, run_typesafe
from llm_client import run_gemini, run_qwen

PROVIDER_LABELS = {
    "clef": "Clef",
    "typesafe": "TypeSafe Jev",
    "gemini": "Gemini",
    "qwen": "Qwen",
}


@dataclass
class ProviderConfig:
    provider: str
    model: str
    enabled: bool = True
    base_url: str = ""
    api_key: str = ""
    timeout: float | None = None
    thinking_level: str | None = None


@dataclass
class ModelRun:
    """One provider's attempt at a decision, success or failure."""

    provider: str
    model: str
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_tokens: int | None = None
    answers: dict[str, Any] = field(default_factory=dict)
    raw_response: dict[str, Any] = field(default_factory=dict)
    request: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    error_body: Any = None

    @property
    def label(self) -> str:
        return PROVIDER_LABELS.get(self.provider, self.provider)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return (self.input_tokens or 0) + (self.output_tokens or 0)

    @property
    def tokens_per_sec(self) -> float | None:
        if not self.output_tokens or not self.latency_ms:
            return None
        return self.output_tokens / (self.latency_ms / 1000.0)


def model_run_from_result(provider: str, result: ClefResult) -> ModelRun:
    return ModelRun(
        provider=provider,
        model=result.model or "",
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cached_tokens=result.cached_tokens,
        answers=result.answers,
        raw_response=result.response,
        request=result.request,
    )


def model_run_from_error(provider: str, model: str, exc: ClefError) -> ModelRun:
    return ModelRun(
        provider=provider,
        model=model,
        error=str(exc),
        error_body=exc.body,
    )


def _optional_base(cfg: ProviderConfig) -> dict[str, str]:
    return {"base_url": cfg.base_url} if cfg.base_url else {}


def run_one(
    state: str | dict[str, Any],
    questions: dict[str, Any],
    cfg: ProviderConfig,
) -> ClefResult:
    """Call a single provider; raises ClefError on failure."""
    extra: dict[str, Any] = {**_optional_base(cfg)}
    if cfg.timeout is not None:
        extra["timeout"] = cfg.timeout
    if cfg.provider == "clef":
        return run_clef(state, questions, model=cfg.model, **extra)
    if cfg.provider == "typesafe":
        return run_typesafe(state, questions, api_key=cfg.api_key, model=cfg.model, **extra)
    if cfg.provider == "gemini":
        if cfg.thinking_level:
            extra["thinking_level"] = cfg.thinking_level
        return run_gemini(state, questions, api_key=cfg.api_key, model=cfg.model, **extra)
    if cfg.provider == "qwen":
        return run_qwen(state, questions, model=cfg.model, **extra)
    raise ClefError(f"Unknown provider '{cfg.provider}'.")


def run_selected(
    state: str | dict[str, Any],
    questions: dict[str, Any],
    configs: Sequence[ProviderConfig],
) -> list[ModelRun]:
    """Run every enabled provider. A failure in one does not skip the rest."""
    runs: list[ModelRun] = []
    for cfg in configs:
        if not cfg.enabled:
            continue
        try:
            result = run_one(state, questions, cfg)
        except ClefError as e:
            runs.append(model_run_from_error(cfg.provider, cfg.model, e))
        else:
            run = model_run_from_result(cfg.provider, result)
            if not run.model:
                run.model = cfg.model
            runs.append(run)
    return runs
