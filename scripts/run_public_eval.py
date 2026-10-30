#!/usr/bin/env python3
"""Run decision models and LLMs on the public MAP / MathDial slices and record every result.

Run keys: typesafe (Jev), gemini, clef_9b (Clef-flash), clef_27b (Clef), qwen.
Example: python scripts/run_public_eval.py --providers typesafe,gemini,clef_27b

Resumable: each (slice, provider, row id) is appended to a JSONL checkpoint before
the next request. Re-running skips completed rows. After each provider/slice
finishes, metrics, confusion matrices, token totals, and REPORT.md are rewritten.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from clef_client import ClefError, DEFAULT_JEV_MODEL  # noqa: E402
from evaluator import (  # noqa: E402
    binary_metrics,
    clean_label,
    correctness_metrics,
    latency_stats,
    misconception_metrics,
    multiclass_metrics,
    parse_bool,
    token_stats,
)
from llm_client import DEFAULT_GEMINI_MODEL  # noqa: E402
from models import interpret_answer  # noqa: E402
from providers import ProviderConfig, run_one  # noqa: E402

SLICES_DIR = ROOT / "eval_data/slices"
RESULTS_DIR = ROOT / "eval_results/public_slices"
USD_INR = 87.0
JEV_USD_PER_M_INPUT = 0.042
GEMINI_USD_PER_M_INPUT = 0.75
GEMINI_USD_PER_M_OUTPUT = 3.75
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
RETRYABLE_TEXT = ("rate-limited", "timed out", "timeout", "did not respond", "connection")

# Clef native choice questions reject more than 26 candidates. 24 most-frequent MAP
# labels + none + other cover ~96.8% of labelled train rows.
CLEF_9B_MODEL = "clef-flash:latest"
CLEF_27B_MODEL = "clef:latest"
QWEN_MODEL = "qwen2.5:7b"
CLEF_CHOICE_IDS = [
    "Incomplete", "Additive", "Duplication", "Subtraction", "Positive", "Wrong_term",
    "Irrelevant", "Wrong_fraction", "Inversion", "Mult", "Denominator_only_change",
    "Whole_numbers_larger", "Adding_across", "WNB", "Tacking", "Unknowable",
    "Wrong_Fraction", "SwapDividend", "Scale", "Not_variable", "Firstterm",
    "Adding_terms", "Multiplying_by_4", "FlipChange", "none", "other",
]
CLEF_CHOICE_SET = set(CLEF_CHOICE_IDS)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def slice_spec(name: str, run_key: str = "") -> dict[str, Any]:
    manifest = load_json(SLICES_DIR / "manifest.json")
    spec = manifest["slices"][name]
    questions_file = spec["questions"]
    if run_key.startswith("clef") and questions_file == "map_questions.json":
        questions_file = "map_questions_clef.json"
    return {
        "name": name,
        "df": pd.read_csv(SLICES_DIR / spec["csv"]),
        "questions": load_json(SLICES_DIR / questions_file),
        "questions_file": questions_file,
        "manifest": spec,
        "sampling": manifest.get("sampling", ""),
    }


def remap_clef_misconception(label: str | None) -> str | None:
    """Collapse rare MAP codes to ``other`` so Clef stays within the 26-choice cap."""
    if not label:
        return label
    if label in CLEF_CHOICE_SET:
        return label
    return "other"


def provider_config(run_key: str) -> ProviderConfig:
    if run_key == "typesafe":
        key = os.environ.get("TYPESAFE_API_KEY", "")
        if not key.strip():
            raise SystemExit("TYPESAFE_API_KEY is missing from .env")
        return ProviderConfig("typesafe", DEFAULT_JEV_MODEL, api_key=key, timeout=120)
    if run_key == "gemini":
        key = os.environ.get("GEMINI_API_KEY", "")
        if not key.strip():
            raise SystemExit("GEMINI_API_KEY is missing from .env")
        # Low thinking keeps this batch near the agreed slice budget.
        # gemini-3.8-flash does not support thinkingLevel=minimal (HTTP 400).
        # Playground still uses thinkingLevel=high.
        return ProviderConfig(
            "gemini", DEFAULT_GEMINI_MODEL, api_key=key, timeout=300, thinking_level="low",
        )
    if run_key == "clef_9b":
        return ProviderConfig("clef", CLEF_9B_MODEL, timeout=180)
    if run_key == "clef_27b":
        return ProviderConfig("clef", CLEF_27B_MODEL, timeout=300)
    if run_key == "qwen":
        return ProviderConfig("qwen", QWEN_MODEL, timeout=300)
    raise SystemExit(f"Unsupported run key {run_key}")


def checkpoint_path(slice_name: str, run_key: str) -> Path:
    return RESULTS_DIR / "checkpoints" / f"{slice_name}__{run_key}.jsonl"


def load_done_ids(path: Path) -> set[str]:
    """IDs with a successful checkpoint. Failures are retried on resume."""
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("id") is not None and rec.get("ok"):
                done.add(str(rec["id"]))
    return done


def append_jsonl(path: Path, rec: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=_json_default) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def is_retryable(exc: ClefError) -> bool:
    if exc.status_code in RETRYABLE_STATUS:
        return True
    msg = str(exc).lower()
    return any(tok in msg for tok in RETRYABLE_TEXT)


def call_with_retry(cfg: ProviderConfig, state: dict, questions: dict, attempts: int = 5) -> tuple[dict[str, Any], int]:
    last_error: ClefError | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = run_one(state, questions, cfg)
            rec = {
                "ok": True,
                "error": None,
                "error_status": None,
                "latency_ms": round(result.latency_ms, 1),
                "model": result.model or cfg.model,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cached_tokens": result.cached_tokens,
                "answers": result.answers,
                "usage": (result.response or {}).get("usage") or {},
            }
            return rec, attempt
        except ClefError as e:
            last_error = e
            if attempt >= attempts or not is_retryable(e):
                break
            sleep_s = min(60.0, 2.0 ** attempt)
            print(f"    retry {attempt}/{attempts} after {sleep_s:.0f}s: {e}", flush=True)
            time.sleep(sleep_s)
    assert last_error is not None
    rec = {
        "ok": False,
        "error": str(last_error).splitlines()[0],
        "error_status": last_error.status_code,
        "latency_ms": None,
        "model": cfg.model,
        "input_tokens": None,
        "output_tokens": None,
        "cached_tokens": None,
        "answers": {},
        "usage": {},
    }
    return rec, attempts


def interpret_record(row: pd.Series, rec: dict[str, Any], questions: dict, threshold: float = 0.5) -> dict[str, Any]:
    out = {
        "id": row["id"],
        "question": row["question"],
        "student_answer": row["student_answer"],
        "gt_correct": parse_bool(row.get("ground_truth_correct")),
        "gt_misconception": clean_label(row.get("ground_truth_misconception")),
        "error": rec.get("error"),
        "latency_ms": rec.get("latency_ms"),
        "input_tokens": rec.get("input_tokens"),
        "output_tokens": rec.get("output_tokens"),
        "cached_tokens": rec.get("cached_tokens"),
        "model": rec.get("model"),
        "attempts": rec.get("attempts"),
        "timestamp": rec.get("timestamp"),
    }
    for extra in (
        "map_category",
        "mc_answer",
        "explanation_correct",
        "misconception_labelled",
        "qid",
        "student_profile",
        "teacher_described_confusion",
        "self_correctness",
        "gold_solution",
    ):
        if extra in row.index and not (isinstance(row[extra], float) and pd.isna(row[extra])):
            out[extra] = row[extra]
    answers = rec.get("answers") or {}
    if rec.get("ok") and "correct" in questions and "correct" in answers:
        d = interpret_answer(answers["correct"], threshold)
        out["pred_correct"] = d.label == "TRUE"
        out["p_true"] = d.noul
        out["conf_correct"] = d.confidence
        out["correct_reasoning"] = (answers["correct"] or {}).get("reasoning")
    if rec.get("ok") and "misconception" in questions and "misconception" in answers:
        d = interpret_answer(answers["misconception"])
        out["pred_misconception"] = d.label
        out["prob_misconception"] = d.probability
        out["conf_misconception"] = d.confidence
        out["misconception_probabilities"] = d.probabilities
        out["misconception_reasoning"] = (answers["misconception"] or {}).get("reasoning")
    if rec.get("run_key", "").startswith("clef") and out.get("gt_misconception"):
        out["gt_misconception_full"] = out["gt_misconception"]
        out["gt_misconception"] = remap_clef_misconception(out["gt_misconception"])
    return out


def df_from_checkpoint(slice_name: str, run_key: str, cfg: ProviderConfig, questions: dict, source: pd.DataFrame) -> pd.DataFrame:
    path = checkpoint_path(slice_name, run_key)
    if not path.exists():
        return pd.DataFrame()
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(r["id"]): r for r in recs}
    rows = []
    for _, row in source.iterrows():
        rec = by_id.get(str(row["id"]))
        if rec is None:
            continue
        rec = {**rec, "run_key": rec.get("run_key") or run_key}
        interpreted = interpret_record(row, rec, questions)
        interpreted["provider"] = cfg.provider
        interpreted["run_key"] = run_key
        interpreted["model"] = rec.get("model") or cfg.model
        rows.append(interpreted)
    return pd.DataFrame(rows)


def metrics_bundle(results: pd.DataFrame, questions: dict) -> dict[str, Any]:
    lat = latency_stats(results)
    tok = token_stats(results)
    cm = correctness_metrics(results)
    labels = list((questions.get("misconception") or {}).get("criteria") or {})
    mm = misconception_metrics(results, labels=labels or None)
    labelled = None
    if "misconception_labelled" in results.columns and "pred_misconception" in results.columns:
        sub = results[results["misconception_labelled"].fillna(False).astype(bool)].dropna(
            subset=["gt_misconception", "pred_misconception"]
        )
        if len(sub):
            labelled = multiclass_metrics(sub["gt_misconception"], sub["pred_misconception"], labels=labels or None)
    none_vs_other = None
    if "gt_misconception" in results.columns and "pred_misconception" in results.columns:
        sub = results.dropna(subset=["gt_misconception", "pred_misconception"]).copy()
        if len(sub):
            none_vs_other = binary_metrics(
                sub["gt_misconception"].astype(str).ne("none"),
                sub["pred_misconception"].astype(str).ne("none"),
            )
            none_vs_other = {
                **none_vs_other,
                "positive_class": "has_misconception (not none)",
            }
    in_sum = pd.to_numeric(results.get("input_tokens"), errors="coerce").fillna(0).sum()
    out_sum = pd.to_numeric(results.get("output_tokens"), errors="coerce").fillna(0).sum()
    cached_sum = pd.to_numeric(results.get("cached_tokens"), errors="coerce").fillna(0).sum()
    return {
        "n_rows": int(len(results)),
        "latency": lat,
        "tokens": {
            **tok,
            "sum_input_tokens": int(in_sum),
            "sum_output_tokens": int(out_sum),
            "sum_cached_tokens": int(cached_sum),
            "sum_total_tokens": int(in_sum + out_sum),
        },
        "correctness": cm,
        "misconception_all_classes": _metrics_public(mm),
        "misconception_labelled_only": _metrics_public(labelled),
        "has_misconception_vs_none": none_vs_other,
        "pred_correct_rate": (
            float(results["pred_correct"].mean()) if "pred_correct" in results and results["pred_correct"].notna().any() else None
        ),
        "mean_p_true": (
            float(pd.to_numeric(results["p_true"], errors="coerce").mean())
            if "p_true" in results else None
        ),
    }


def _metrics_public(mm: dict[str, Any] | None) -> dict[str, Any] | None:
    if not mm:
        return None
    per_class = mm["per_class"]
    confusion = mm["confusion"]
    return {
        "n": mm["n"],
        "accuracy": mm["accuracy"],
        "macro_f1": mm["macro_f1"],
        "per_class": per_class.to_dict(orient="records"),
        "confusion": confusion.fillna(0).astype(int).to_dict(),
    }


def estimate_cost_usd(provider: str, tokens: dict[str, Any]) -> dict[str, float]:
    inn = float(tokens.get("sum_input_tokens") or 0)
    out = float(tokens.get("sum_output_tokens") or 0)
    if provider == "typesafe":
        usd = inn / 1_000_000 * JEV_USD_PER_M_INPUT
        return {"usd": usd, "inr": usd * USD_INR, "note": "Jev bills input tokens only at $0.042/MTok"}
    if provider == "gemini":
        usd = inn / 1_000_000 * GEMINI_USD_PER_M_INPUT + out / 1_000_000 * GEMINI_USD_PER_M_OUTPUT
        return {
            "usd": usd,
            "inr": usd * USD_INR,
            "note": "Gemini 3.8 Flash intro pricing $0.75/M in + $3.75/M out; output includes thinking tokens",
        }
    if provider in {"clef", "qwen"}:
        return {
            "usd": 0.0,
            "inr": 0.0,
            "note": "Local Ollama model: no API bill. Electricity / GPU time not priced.",
        }
    return {"usd": 0.0, "inr": 0.0, "note": "unknown provider"}


def write_slice_artifacts(slice_name: str, run_key: str, cfg: ProviderConfig, spec: dict[str, Any]) -> dict[str, Any]:
    results = df_from_checkpoint(slice_name, run_key, cfg, spec["questions"], spec["df"])
    out_dir = RESULTS_DIR / slice_name / run_key
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "per_row.csv", index=False)
    # Keep a JSON copy of every interpreted row as well.
    results.to_json(out_dir / "per_row.json", orient="records", indent=2, force_ascii=False)
    bundle = metrics_bundle(results, spec["questions"])
    cost = estimate_cost_usd(cfg.provider, bundle["tokens"])
    summary = {
        "slice": slice_name,
        "run_key": run_key,
        "provider": cfg.provider,
        "model": cfg.model,
        "finished_at": utc_now(),
        "fx_inr_per_usd": USD_INR,
        "cost": cost,
        "metrics": bundle,
        "sampling": spec.get("sampling"),
        "slice_manifest": spec.get("manifest"),
    }
    dump_json(out_dir / "summary.json", summary)
    if bundle.get("correctness"):
        cm = bundle["correctness"]
        pd.DataFrame(
            [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
            index=["gt_false", "gt_true"],
            columns=["pred_false", "pred_true"],
        ).to_csv(out_dir / "confusion_correct.csv")
    mm = bundle.get("misconception_all_classes")
    if mm and mm.get("confusion"):
        pd.DataFrame(mm["confusion"]).to_csv(out_dir / "confusion_misconception.csv")
    labelled = bundle.get("misconception_labelled_only")
    if labelled and labelled.get("confusion"):
        pd.DataFrame(labelled["confusion"]).to_csv(out_dir / "confusion_misconception_labelled.csv")
    return summary


def fmt_pct(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{100 * float(x):.1f}%"


def fmt_num(x: Any, nd: int = 1) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{float(x):.{nd}f}"


def write_report(all_summaries: list[dict[str, Any]]) -> None:
    lines = [
        "# Public-slice evaluation report",
        "",
        f"Generated: {utc_now()}",
        "",
        "This file is rewritten after every completed (slice, provider) run. Use it plus the",
        "`per_row.csv` / `per_row.json` / checkpoint JSONL files as the source for a written report.",
        "",
        "## What was run",
        "",
        "- Datasets: MAP train (stratified 50 and 500) and MathDial **test** (599).",
        "- Models: TypeSafe Jev (`jev-latest`); Gemini (`gemini-3.8-flash`, thinkingLevel=low); local Clef 9.5B (`clef-flash:latest`); local Clef 27.5B (`clef:latest`); local Qwen (`qwen2.5:7b`).",
        "- MAP tasks: MC-answer correctness (`noul`) + MAP misconception taxonomy (`choice`). Jev/Gemini/Qwen use 35 labels + `none`. Clef is capped at 26 choice candidates by Ollama, so it uses the 24 most common MAP labels + `none` + `other`.",
        "- MathDial task: correctness of the student's initial incorrect solution (`noul` only). Gold label is FALSE for every row by construction.",
        "- Sampling: MAP is stratified by `Category` with seed 42. Sequential head(50)/head(500) would be 100% `True_*` and is not used.",
        f"- FX used for ₹ estimates: {USD_INR:.0f} INR / USD. Jev $0.042/MTok input-only. Gemini 3.8 Flash intro $0.75/M in + $3.75/M out.",
        "",
        "## Headline comparison",
        "",
        "| Slice | Provider | Model | N | Failed | Correct acc | Correct F1 | Misc. acc (all) | Misc. macro-F1 (labelled) | Avg latency ms | Input tok | Output tok | Est. ₹ |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in all_summaries:
        m = s["metrics"]
        c = m.get("correctness") or {}
        mall = m.get("misconception_all_classes") or {}
        lab = m.get("misconception_labelled_only") or {}
        lines.append(
            "| {slice} | {provider} | {model} | {n} | {failed} | {acc} | {f1} | {macc} | {mf1} | {lat} | {tin} | {tout} | {inr} |".format(
                slice=s["slice"],
                provider=s.get("run_key") or s["provider"],
                model=s["model"],
                n=m["n_rows"],
                failed=m["latency"]["failed_requests"],
                acc=fmt_pct(c.get("accuracy")),
                f1=fmt_pct(c.get("f1")),
                macc=fmt_pct(mall.get("accuracy")),
                mf1=fmt_pct(lab.get("macro_f1")),
                lat=fmt_num(m["latency"].get("avg_ms"), 0),
                tin=m["tokens"]["sum_input_tokens"],
                tout=m["tokens"]["sum_output_tokens"],
                inr=fmt_num(s["cost"]["inr"], 2),
            )
        )
    lines += ["", "## Per-run detail", ""]
    for s in all_summaries:
        m = s["metrics"]
        c = m.get("correctness")
        lines += [
            f"### {s['slice']} / {s.get('run_key') or s['provider']} (`{s['model']}`)",
            "",
            f"- Finished: {s['finished_at']}",
            f"- Rows: {m['n_rows']}; failed requests: {m['latency']['failed_requests']}",
            f"- Latency ms: avg {fmt_num(m['latency']['avg_ms'], 1)}, median {fmt_num(m['latency']['median_ms'], 1)}, p95 {fmt_num(m['latency']['p95_ms'], 1)}",
            f"- Tokens: input {m['tokens']['sum_input_tokens']}, output {m['tokens']['sum_output_tokens']}, cached {m['tokens']['sum_cached_tokens']}, avg total {fmt_num(m['tokens']['avg_total_tokens'], 1)}",
            f"- Estimated cost: ${s['cost']['usd']:.4f} (₹{s['cost']['inr']:.2f}). {s['cost']['note']}",
            "",
        ]
        if c:
            lines += [
                "Correctness (TRUE = selected MC answer is right; MathDial TRUE would mean the written solution is right):",
                "",
                f"- n={c['n']} accuracy={c['accuracy']:.4f} precision={c['precision']:.4f} recall={c['recall']:.4f} F1={c['f1']:.4f}",
                f"- TP={c['tp']} TN={c['tn']} FP={c['fp']} FN={c['fn']}",
                f"- Model predicted TRUE on {fmt_pct(m.get('pred_correct_rate'))} of scored rows; mean noul P(true)={fmt_num(m.get('mean_p_true'), 3)}",
                "",
            ]
        mall = m.get("misconception_all_classes")
        if mall:
            lines += [
                f"Misconception, all classes including `none`: n={mall['n']} accuracy={mall['accuracy']:.4f} macro-F1={mall['macro_f1']:.4f}",
                "",
            ]
        lab = m.get("misconception_labelled_only")
        if lab:
            lines += [
                f"Misconception, labelled rows only (MAP `*_Misconception`): n={lab['n']} accuracy={lab['accuracy']:.4f} macro-F1={lab['macro_f1']:.4f}",
                "",
            ]
        hvm = m.get("has_misconception_vs_none")
        if hvm:
            lines += [
                "Has-misconception vs `none` (binary, positive = not none):",
                f"- n={hvm['n']} accuracy={hvm['accuracy']:.4f} precision={hvm['precision']:.4f} recall={hvm['recall']:.4f} F1={hvm['f1']:.4f} TP={hvm['tp']} TN={hvm['tn']} FP={hvm['fp']} FN={hvm['fn']}",
                "",
            ]
        lines += [
            "Artifacts:",
            f"- `{s['slice']}/{s.get('run_key') or s['provider']}/per_row.csv`",
            f"- `{s['slice']}/{s.get('run_key') or s['provider']}/per_row.json`",
            f"- `{s['slice']}/{s.get('run_key') or s['provider']}/summary.json`",
            f"- `checkpoints/{s['slice']}__{s.get('run_key') or s['provider']}.jsonl` (raw answers, usage, errors)",
            "",
        ]
    lines += [
        "## File map",
        "",
        "```",
        "eval_results/public_slices/",
        "  REPORT.md                  this file",
        "  comparison.csv             one row per completed run",
        "  checkpoints/*.jsonl        every API response, including failures and retries",
        "  <slice>/<provider>/",
        "    per_row.csv              one row per item with GT, prediction, tokens, latency",
        "    per_row.json             same, with reasoning and probability vectors",
        "    summary.json             metrics + cost",
        "    confusion_*.csv          confusion matrices",
        "```",
        "",
        "## Notes for the written report",
        "",
        "- Gemini output tokens include thinking tokens (`thoughtsTokenCount` + `candidatesTokenCount`).",
        "- Jev output is a small JSON decision object; Gemini is prompted to emit long `reasoning` fields.",
        "- MathDial has no correct student solutions in this split, so correctness F1 for the TRUE class is expected to be 0 or near 0; accuracy then equals the true-negative rate.",
        "- MAP `Wrong_fraction` and `Wrong_Fraction` are distinct labels in the source data and were kept distinct.",
        "",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    rows = []
    for s in all_summaries:
        m = s["metrics"]
        c = m.get("correctness") or {}
        mall = m.get("misconception_all_classes") or {}
        lab = m.get("misconception_labelled_only") or {}
        rows.append({
            "slice": s["slice"],
            "run_key": s.get("run_key") or s["provider"],
            "provider": s["provider"],
            "model": s["model"],
            "n": m["n_rows"],
            "failed": m["latency"]["failed_requests"],
            "correct_accuracy": c.get("accuracy"),
            "correct_precision": c.get("precision"),
            "correct_recall": c.get("recall"),
            "correct_f1": c.get("f1"),
            "correct_tp": c.get("tp"),
            "correct_tn": c.get("tn"),
            "correct_fp": c.get("fp"),
            "correct_fn": c.get("fn"),
            "misc_accuracy_all": mall.get("accuracy"),
            "misc_macro_f1_all": mall.get("macro_f1"),
            "misc_accuracy_labelled": lab.get("accuracy"),
            "misc_macro_f1_labelled": lab.get("macro_f1"),
            "avg_latency_ms": m["latency"]["avg_ms"],
            "median_latency_ms": m["latency"]["median_ms"],
            "p95_latency_ms": m["latency"]["p95_ms"],
            "sum_input_tokens": m["tokens"]["sum_input_tokens"],
            "sum_output_tokens": m["tokens"]["sum_output_tokens"],
            "sum_total_tokens": m["tokens"]["sum_total_tokens"],
            "avg_total_tokens": m["tokens"]["avg_total_tokens"],
            "cost_usd": s["cost"]["usd"],
            "cost_inr": s["cost"]["inr"],
            "finished_at": s["finished_at"],
        })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "comparison.csv", index=False)


def collect_summaries() -> list[dict[str, Any]]:
    summaries = []
    for path in sorted(RESULTS_DIR.glob("*/*/summary.json")):
        summaries.append(load_json(path))
    slice_order = {"map_50": 0, "map_500": 1, "mathdial_test": 2}
    key_order = {"typesafe": 0, "gemini": 1, "clef_9b": 2, "clef_27b": 3, "qwen": 4}
    summaries.sort(key=lambda s: (
        slice_order.get(s["slice"], 9),
        key_order.get(s.get("run_key") or s["provider"], 9),
    ))
    return summaries


def run_slice_provider(slice_name: str, run_key: str) -> None:
    spec = slice_spec(slice_name, run_key)
    cfg = provider_config(run_key)
    df = spec["df"]
    questions = spec["questions"]
    ckpt = checkpoint_path(slice_name, run_key)
    done = load_done_ids(ckpt)
    total = len(df)
    remaining = [i for i, row in df.iterrows() if str(row["id"]) not in done]
    print(f"\n=== {slice_name} / {run_key} ({cfg.model}): {len(done)} done, {len(remaining)} remaining of {total} ===", flush=True)
    started = time.perf_counter()
    for n, idx in enumerate(remaining, start=1):
        row = df.loc[idx]
        state = {"question": str(row["question"]), "student_answer": str(row["student_answer"])}
        rec, attempts = call_with_retry(cfg, state, questions)
        rec.update({
            "id": row["id"],
            "slice": slice_name,
            "run_key": run_key,
            "provider": cfg.provider,
            "attempts": attempts,
            "timestamp": utc_now(),
        })
        append_jsonl(ckpt, rec)
        status = "ok" if rec["ok"] else f"FAIL {rec['error']}"
        toks = f"in={rec.get('input_tokens')} out={rec.get('output_tokens')} {rec.get('latency_ms')}ms"
        print(f"  [{len(done) + n}/{total}] {row['id']} {status} {toks}", flush=True)
        progress = {
            "slice": slice_name,
            "run_key": run_key,
            "provider": cfg.provider,
            "model": cfg.model,
            "done": len(done) + n,
            "total": total,
            "last_id": row["id"],
            "last_status": status,
            "updated": utc_now(),
            "elapsed_s": round(time.perf_counter() - started, 1),
        }
        dump_json(RESULTS_DIR / "progress.json", progress)
    summary = write_slice_artifacts(slice_name, run_key, cfg, spec)
    write_report(collect_summaries())
    print(
        f"Finished {slice_name}/{run_key}: acc={fmt_pct((summary['metrics'].get('correctness') or {}).get('accuracy'))} "
        f"₹{summary['cost']['inr']:.2f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slices", default="map_50,map_500,mathdial_test")
    parser.add_argument("--providers", default="typesafe,gemini")
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dump_json(RESULTS_DIR / "run_meta.json", {
        "started": utc_now(),
        "slices": args.slices.split(","),
        "providers": args.providers.split(","),
        "models": {
            "typesafe": DEFAULT_JEV_MODEL,
            "gemini": DEFAULT_GEMINI_MODEL,
            "clef_9b": CLEF_9B_MODEL,
            "clef_27b": CLEF_27B_MODEL,
            "qwen": QWEN_MODEL,
        },
        "gemini_thinking_level": "low",
        "pricing": {
            "usd_inr": USD_INR,
            "jev_usd_per_m_input": JEV_USD_PER_M_INPUT,
            "gemini_usd_per_m_input": GEMINI_USD_PER_M_INPUT,
            "gemini_usd_per_m_output": GEMINI_USD_PER_M_OUTPUT,
        },
    })
    try:
        for provider in [p.strip() for p in args.providers.split(",") if p.strip()]:
            for slice_name in [s.strip() for s in args.slices.split(",") if s.strip()]:
                run_slice_provider(slice_name, provider)
    except Exception:
        dump_json(RESULTS_DIR / "crash.json", {"at": utc_now(), "traceback": traceback.format_exc()})
        raise
    dump_json(RESULTS_DIR / "run_meta_finished.json", {"finished": utc_now()})
    print("All requested slices complete.", flush=True)


if __name__ == "__main__":
    main()
