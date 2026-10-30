"""Batch evaluation of decision providers against a labelled CSV.

Requests are sent sequentially (one per row, carrying every enabled question)
to keep the load on local models predictable. Multiple providers are run one
after another; failures are recorded instead of aborting the batch.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from clef_client import ClefError, run_clef
from models import interpret_answer

REQUIRED_COLUMNS = ["id", "question", "student_answer"]
GT_CORRECT = "ground_truth_correct"
GT_MISCONCEPTION = "ground_truth_misconception"
CORRECT_QID = "correct"
MISCONCEPTION_QID = "misconception"

_TRUE = {"true", "1", "yes", "y", "t", "correct"}
_FALSE = {"false", "0", "no", "n", "f", "incorrect"}

RunFn = Callable[..., Any]


def parse_bool(value: Any) -> bool | None:
    """Parse a ground-truth boolean; None when missing or unrecognised."""
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return None


def clean_label(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    s = str(value).strip()
    return s or None


def validate_dataset(df: pd.DataFrame) -> list[str]:
    """Return a list of problems; empty when the dataset is usable."""
    problems = [f"Missing required column '{c}'." for c in REQUIRED_COLUMNS if c not in df.columns]
    if GT_CORRECT not in df.columns and GT_MISCONCEPTION not in df.columns:
        problems.append(f"Need at least one ground-truth column: '{GT_CORRECT}' or '{GT_MISCONCEPTION}'.")
    if df.empty:
        problems.append("The dataset has no rows.")
    return problems


def row_state(row: pd.Series) -> dict[str, str]:
    """The state sent for one row (structured, as System One accepts JSON objects)."""
    return {"question": str(row["question"]), "student_answer": str(row["student_answer"])}


def _record_success(rec: dict[str, Any], result: Any, questions: dict[str, dict], threshold: float) -> None:
    rec["latency_ms"] = round(result.latency_ms, 1)
    rec["input_tokens"] = getattr(result, "input_tokens", None)
    rec["output_tokens"] = getattr(result, "output_tokens", None)
    rec["cached_tokens"] = getattr(result, "cached_tokens", None)
    if CORRECT_QID in questions:
        d = interpret_answer(result.answers[CORRECT_QID], threshold)
        rec["pred_correct"] = d.label == "TRUE"
        rec["p_true"] = d.noul
        rec["conf_correct"] = d.confidence
    if MISCONCEPTION_QID in questions:
        d = interpret_answer(result.answers[MISCONCEPTION_QID])
        rec["pred_misconception"] = d.label
        rec["prob_misconception"] = d.probability
        rec["conf_misconception"] = d.confidence


def evaluate_dataset(
    df: pd.DataFrame,
    questions: dict[str, dict],
    *,
    model: str = "",
    base_url: str = "",
    threshold: float = 0.5,
    on_progress: Callable[[int, int], None] | None = None,
    run_fn: RunFn | None = None,
) -> pd.DataFrame:
    """Run a provider on every row sequentially and return one result row per input row.

    ``questions`` may contain ``correct`` (noul) and/or ``misconception`` (choice).
    Failed requests are recorded with an ``error`` message instead of aborting.
    ``run_fn(state, questions, **kwargs)`` defaults to ``run_clef``.
    """
    runner = run_fn or run_clef
    records: list[dict[str, Any]] = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        rec: dict[str, Any] = {
            "id": row["id"],
            "question": row["question"],
            "student_answer": row["student_answer"],
            "gt_correct": parse_bool(row.get(GT_CORRECT)),
            "gt_misconception": clean_label(row.get(GT_MISCONCEPTION)),
            "error": None,
        }
        try:
            result = runner(row_state(row), questions, model=model, base_url=base_url)
        except ClefError as e:
            rec["error"] = str(e).splitlines()[0]
            rec["latency_ms"] = None
        else:
            _record_success(rec, result, questions, threshold)
        records.append(rec)
        if on_progress:
            on_progress(i, total)
    return pd.DataFrame(records)


def evaluate_models(
    df: pd.DataFrame,
    questions: dict[str, dict],
    runners: Sequence[tuple[str, str, RunFn]],
    *,
    threshold: float = 0.5,
    on_progress: Callable[[int, int], None] | None = None,
) -> pd.DataFrame:
    """Run every ``(provider, model, run_fn)`` on the dataset and stack the rows."""
    n_rows = len(df)
    total = max(len(runners) * n_rows, 1)
    frames: list[pd.DataFrame] = []
    for r_i, (provider, model, run_fn) in enumerate(runners):
        def progress_row(i: int, _n: int, _offset: int = r_i * n_rows) -> None:
            if on_progress:
                on_progress(_offset + i, total)

        part = evaluate_dataset(
            df, questions, model=model, threshold=threshold, run_fn=run_fn, on_progress=progress_row,
        )
        part["provider"] = provider
        part["model"] = model
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- metrics

def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def binary_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Accuracy / precision / recall / F1 with TRUE as the positive class."""
    t, p = y_true.astype(bool), y_pred.astype(bool)
    tp = int((t & p).sum())
    tn = int((~t & ~p).sum())
    fp = int((~t & p).sum())
    fn = int((t & ~p).sum())
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    return {
        "n": len(t),
        "accuracy": _safe_div(tp + tn, len(t)),
        "precision": precision,
        "recall": recall,
        "f1": _safe_div(2 * precision * recall, precision + recall),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def multiclass_metrics(y_true: pd.Series, y_pred: pd.Series, labels: list[str] | None = None) -> dict[str, Any]:
    """Accuracy, macro F1, per-class precision/recall/F1 and a confusion matrix.

    Rows of the confusion matrix are ground truth, columns are predictions.
    Labels default to the union of observed labels (criteria order first if given).
    """
    y_true, y_pred = y_true.astype(str), y_pred.astype(str)
    all_labels = list(labels or [])
    for lab in list(y_true) + list(y_pred):
        if lab not in all_labels:
            all_labels.append(lab)

    confusion = pd.crosstab(
        pd.Categorical(y_true, categories=all_labels),
        pd.Categorical(y_pred, categories=all_labels),
        dropna=False,
    )
    confusion.index.name, confusion.columns.name = "ground truth", "prediction"

    rows = []
    for lab in all_labels:
        tp = int(confusion.loc[lab, lab])
        fp = int(confusion[lab].sum()) - tp
        fn = int(confusion.loc[lab].sum()) - tp
        precision, recall = _safe_div(tp, tp + fp), _safe_div(tp, tp + fn)
        rows.append({
            "class": lab,
            "precision": precision,
            "recall": recall,
            "f1": _safe_div(2 * precision * recall, precision + recall),
            "support": tp + fn,
        })
    per_class = pd.DataFrame(rows)
    # Macro F1 over classes that occur in the ground truth.
    supported = per_class[per_class["support"] > 0]
    return {
        "n": len(y_true),
        "accuracy": float((y_true.values == y_pred.values).mean()) if len(y_true) else 0.0,
        "macro_f1": float(supported["f1"].mean()) if len(supported) else 0.0,
        "per_class": per_class,
        "confusion": confusion,
    }


def latency_stats(results: pd.DataFrame) -> dict[str, float]:
    ok = results["latency_ms"].dropna() if "latency_ms" in results else pd.Series(dtype=float)
    return {
        "total_requests": len(results),
        "failed_requests": int(results["error"].notna().sum()) if "error" in results else 0,
        "avg_ms": float(ok.mean()) if len(ok) else float("nan"),
        "median_ms": float(ok.median()) if len(ok) else float("nan"),
        "p95_ms": float(ok.quantile(0.95)) if len(ok) else float("nan"),
    }


def _avg_num(results: pd.DataFrame, col: str) -> float:
    if col not in results:
        return float("nan")
    s = pd.to_numeric(results[col], errors="coerce").dropna()
    return float(s.mean()) if len(s) else float("nan")


def token_stats(results: pd.DataFrame) -> dict[str, float]:
    avg_in, avg_out = _avg_num(results, "input_tokens"), _avg_num(results, "output_tokens")
    if pd.isna(avg_in) and pd.isna(avg_out):
        avg_total = float("nan")
    else:
        avg_total = (0.0 if pd.isna(avg_in) else avg_in) + (0.0 if pd.isna(avg_out) else avg_out)
    return {
        "avg_input_tokens": avg_in,
        "avg_output_tokens": avg_out,
        "avg_total_tokens": avg_total,
    }


def correctness_metrics(results: pd.DataFrame) -> dict[str, float] | None:
    if "pred_correct" not in results:
        return None
    sub = results.dropna(subset=["gt_correct", "pred_correct"])
    return binary_metrics(sub["gt_correct"], sub["pred_correct"]) if len(sub) else None


def misconception_metrics(results: pd.DataFrame, labels: list[str] | None = None) -> dict[str, Any] | None:
    if "pred_misconception" not in results:
        return None
    sub = results.dropna(subset=["gt_misconception", "pred_misconception"])
    return multiclass_metrics(sub["gt_misconception"], sub["pred_misconception"], labels) if len(sub) else None


def model_comparison(results: pd.DataFrame, labels: list[str] | None = None) -> pd.DataFrame:
    """One summary row per (provider, model) with accuracy, latency, and tokens."""
    if results.empty:
        return pd.DataFrame()
    if "provider" not in results.columns:
        grouped = [("clef", results.get("model", pd.Series(["clef"])).iloc[0] if len(results) else "", results)]
    else:
        grouped = [
            (provider, model, g)
            for (provider, model), g in results.groupby(["provider", "model"], dropna=False)
        ]
    rows = []
    for provider, model, g in grouped:
        lat = latency_stats(g)
        tok = token_stats(g)
        cm = correctness_metrics(g)
        mm = misconception_metrics(g, labels)
        rows.append({
            "provider": provider,
            "model": model,
            "requests": lat["total_requests"],
            "failed": lat["failed_requests"],
            "accuracy": None if not cm else cm["accuracy"],
            "f1": None if not cm else cm["f1"],
            "macro_f1": None if not mm else mm["macro_f1"],
            "avg_latency_ms": lat["avg_ms"],
            "median_latency_ms": lat["median_ms"],
            "p95_latency_ms": lat["p95_ms"],
            "avg_input_tokens": tok["avg_input_tokens"],
            "avg_output_tokens": tok["avg_output_tokens"],
            "avg_total_tokens": tok["avg_total_tokens"],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- export

EXPORT_COLUMNS = ["id", "question", "student_answer", "ground_truth", "prediction", "confidence", "latency_ms"]


def export_table(results: pd.DataFrame) -> pd.DataFrame:
    """Long-format export: one row per (dataset row, task).

    Columns are the requested export columns plus ``task`` and ``error``.
    """
    extras = [c for c in ("provider", "model", "input_tokens", "output_tokens") if c in results.columns]
    base_cols = ["id", "question", "student_answer", "latency_ms", "error"] + extras
    base = results[base_cols]
    parts = []
    if "pred_correct" in results:
        parts.append(base.assign(
            task=CORRECT_QID,
            ground_truth=results["gt_correct"],
            prediction=results["pred_correct"],
            confidence=results["conf_correct"],
        ))
    if "pred_misconception" in results:
        parts.append(base.assign(
            task=MISCONCEPTION_QID,
            ground_truth=results["gt_misconception"],
            prediction=results["pred_misconception"],
            confidence=results["conf_misconception"],
        ))
    if not parts:  # every request failed
        parts.append(base.assign(task=None, ground_truth=None, prediction=None, confidence=None))
    out = pd.concat(parts, ignore_index=True)
    ordered = EXPORT_COLUMNS + ["task", "error"] + extras
    return out[ordered]
