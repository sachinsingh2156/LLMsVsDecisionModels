#!/usr/bin/env python3
"""Build extra tables and a report-ready markdown pack from completed eval artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluator import binary_metrics, multiclass_metrics  # noqa: E402
from scripts.run_public_eval import (  # noqa: E402
    RESULTS_DIR,
    collect_summaries,
    estimate_cost_usd,
    fmt_num,
    fmt_pct,
    write_report,
)

ANALYSIS = RESULTS_DIR / "analysis"


def load_rows(slice_name: str, run_key: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / slice_name / run_key / "per_row.csv")


def run_label(summary: dict) -> str:
    return summary.get("run_key") or summary["provider"]


def all_run_keys(summaries: list[dict]) -> list[str]:
    seen: list[str] = []
    for s in summaries:
        k = run_label(s)
        if k not in seen:
            seen.append(k)
    return seen


def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    if "map_category" not in df.columns or "pred_correct" not in df.columns:
        return pd.DataFrame()
    sub = df.dropna(subset=["gt_correct", "pred_correct", "map_category"]).copy()
    rows = []
    for cat, g in sub.groupby("map_category"):
        cm = binary_metrics(g["gt_correct"], g["pred_correct"])
        misc_n = misc_acc = None
        if "gt_misconception" in g and "pred_misconception" in g:
            m = g.dropna(subset=["gt_misconception", "pred_misconception"])
            if len(m):
                misc_n = len(m)
                misc_acc = float((m["gt_misconception"].astype(str) == m["pred_misconception"].astype(str)).mean())
        rows.append({
            "map_category": cat,
            "n": cm["n"],
            "correct_accuracy": cm["accuracy"],
            "tp": cm["tp"], "tn": cm["tn"], "fp": cm["fp"], "fn": cm["fn"],
            "pred_true_rate": float(g["pred_correct"].mean()),
            "mean_p_true": float(pd.to_numeric(g.get("p_true"), errors="coerce").mean()) if "p_true" in g else None,
            "misconception_n": misc_n,
            "misconception_accuracy": misc_acc,
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def error_cases(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if {"gt_correct", "pred_correct"}.issubset(df.columns):
        sub = df.dropna(subset=["gt_correct", "pred_correct"])
        bad = sub[sub["gt_correct"].astype(bool) != sub["pred_correct"].astype(bool)]
        for _, r in bad.iterrows():
            rows.append({
                "task": "correct",
                "id": r["id"],
                "gt": r["gt_correct"],
                "pred": r["pred_correct"],
                "p_true": r.get("p_true"),
                "map_category": r.get("map_category"),
                "gt_misconception": r.get("gt_misconception"),
                "pred_misconception": r.get("pred_misconception"),
                "question": str(r.get("question", ""))[:240],
                "student_answer": str(r.get("student_answer", ""))[:240],
            })
    if {"gt_misconception", "pred_misconception", "misconception_labelled"}.issubset(df.columns):
        sub = df[df["misconception_labelled"].fillna(False).astype(bool)].dropna(
            subset=["gt_misconception", "pred_misconception"]
        )
        bad = sub[sub["gt_misconception"].astype(str) != sub["pred_misconception"].astype(str)]
        for _, r in bad.iterrows():
            rows.append({
                "task": "misconception_labelled",
                "id": r["id"],
                "gt": r["gt_misconception"],
                "pred": r["pred_misconception"],
                "p_true": r.get("p_true"),
                "map_category": r.get("map_category"),
                "gt_misconception": r.get("gt_misconception"),
                "pred_misconception": r.get("pred_misconception"),
                "question": str(r.get("question", ""))[:240],
                "student_answer": str(r.get("student_answer", ""))[:240],
            })
    return pd.DataFrame(rows)


def unique_mathdial(df: pd.DataFrame) -> dict:
    u = df.drop_duplicates("id", keep="last")
    scored = u.dropna(subset=["gt_correct", "pred_correct"])
    cm = binary_metrics(scored["gt_correct"], scored["pred_correct"]) if len(scored) else {}
    return {
        "dialogue_rows": int(len(df)),
        "unique_qids": int(u["id"].nunique()),
        "failed_unique": int(u["error"].notna().sum()) if "error" in u else 0,
        "correctness_unique": cm,
        "pred_true_rate_unique": float(scored["pred_correct"].mean()) if len(scored) else None,
        "mean_p_true_unique": float(pd.to_numeric(scored.get("p_true"), errors="coerce").mean()) if len(scored) else None,
    }


def labelled_misc_table(df: pd.DataFrame) -> pd.DataFrame:
    if not {"gt_misconception", "pred_misconception", "misconception_labelled"}.issubset(df.columns):
        return pd.DataFrame()
    sub = df[df["misconception_labelled"].fillna(False).astype(bool)].dropna(
        subset=["gt_misconception", "pred_misconception"]
    )
    if sub.empty:
        return pd.DataFrame()
    mm = multiclass_metrics(sub["gt_misconception"], sub["pred_misconception"])
    return mm["per_class"]


def totals(summaries: list[dict]) -> dict:
    by_p: dict[str, dict] = {}
    for s in summaries:
        p = s["provider"]
        slot = by_p.setdefault(p, {"usd": 0.0, "inr": 0.0, "input": 0, "output": 0, "n": 0, "failed": 0})
        slot["usd"] += s["cost"]["usd"]
        slot["inr"] += s["cost"]["inr"]
        slot["input"] += s["metrics"]["tokens"]["sum_input_tokens"]
        slot["output"] += s["metrics"]["tokens"]["sum_output_tokens"]
        slot["n"] += s["metrics"]["n_rows"]
        slot["failed"] += s["metrics"]["latency"]["failed_requests"]
    grand = {
        "usd": sum(v["usd"] for v in by_p.values()),
        "inr": sum(v["inr"] for v in by_p.values()),
        "n": sum(v["n"] for v in by_p.values()),
        "failed": sum(v["failed"] for v in by_p.values()),
        "by_provider": by_p,
    }
    return grand


def md_table(df: pd.DataFrame, cols: list[str] | None = None, fmt: dict[str, str] | None = None) -> str:
    if df is None or df.empty:
        return "_none_\n"
    use = df if cols is None else df[cols]
    fmt = fmt or {}
    header = "| " + " | ".join(use.columns) + " |"
    sep = "|" + "|".join(["---"] * len(use.columns)) + "|"
    lines = [header, sep]
    for _, r in use.iterrows():
        cells = []
        for c in use.columns:
            v = r[c]
            if c in fmt:
                try:
                    cells.append(fmt[c].format(v))
                except Exception:
                    cells.append(str(v))
            elif isinstance(v, float):
                cells.append(f"{v:.4f}" if abs(v) < 10 else f"{v:.1f}")
            else:
                cells.append("" if pd.isna(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    summaries = collect_summaries()
    write_report(summaries)

    extra_md = []
    unique_stats = {}
    keys = all_run_keys(summaries)
    for s in summaries:
        slice_name, run_key = s["slice"], run_label(s)
        path = RESULTS_DIR / slice_name / run_key / "per_row.csv"
        if not path.exists():
            continue
        df = load_rows(slice_name, run_key)
        cat = category_breakdown(df)
        if not cat.empty:
            cat.to_csv(ANALYSIS / f"{slice_name}__{run_key}__by_category.csv", index=False)
        err = error_cases(df)
        if not err.empty:
            err.to_csv(ANALYSIS / f"{slice_name}__{run_key}__errors.csv", index=False)
        lab = labelled_misc_table(df)
        if not lab.empty:
            lab.to_csv(ANALYSIS / f"{slice_name}__{run_key}__misconception_per_class.csv", index=False)
        if slice_name == "mathdial_test":
            unique_stats[run_key] = unique_mathdial(df)

    (ANALYSIS / "mathdial_unique.json").write_text(json.dumps(unique_stats, indent=2), encoding="utf-8")
    tot = totals(summaries)
    (ANALYSIS / "cost_totals.json").write_text(json.dumps(tot, indent=2), encoding="utf-8")

    jev = tot["by_provider"].get("typesafe", {})
    gem = tot["by_provider"].get("gemini", {})
    extra_md += [
        "## Cost vs the planned cheaper slices",
        "",
        f"- Planned (from the slice table): MAP-50 Gemini ~₹15–20, MAP-500 Gemini ~₹150–200, MathDial-599 Gemini ~₹230; Jev under ₹1 / ~₹1 / ~₹2.",
        f"- Actual at 87 INR/USD: Jev total ₹{jev.get('inr', 0):.2f}; Gemini total ₹{gem.get('inr', 0):.2f}; local Clef/Qwen ₹0 API. Grand API ₹{tot['inr']:.2f} (${tot['usd']:.4f}).",
        f"- Requests: {tot['n']} scored rows across all models; remaining failures after retry: {tot['failed']}.",
        "- Gemini batch used `thinkingLevel=low`. Clef MAP uses a 26-class taxonomy (Ollama cap). Qwen uses the same 36-class taxonomy as Jev/Gemini.",
        "",
        "## MathDial unique problems vs 599 dialogues",
        "",
        "The official MathDial test file has 599 tutoring dialogues but only **394 unique `qid`s**. Duplicate qids share the same question and the same `student_incorrect_solution`; they differ in student profile, scenario, and the teacher's free-text confusion note. The model state is `{question, student_answer}`, so unique-qid metrics are the right unit for correctness. Dialogue-weighted metrics (N=599) are also stored because that is the official split size.",
        "",
        "| Provider | Dialogues N | Unique qid N | Unique acc (TNR) | Unique predicted-TRUE rate | Dialogue acc | Dialogue predicted-TRUE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run_key in keys:
        u = unique_stats.get(run_key)
        if not u:
            continue
        c = u["correctness_unique"]
        s = next(x for x in summaries if x["slice"] == "mathdial_test" and run_label(x) == run_key)
        dc = s["metrics"]["correctness"]
        extra_md.append(
            f"| {run_key} | {u['dialogue_rows']} | {u['unique_qids']} | {fmt_pct(c.get('accuracy'))} | {fmt_pct(u.get('pred_true_rate_unique'))} | {fmt_pct(dc.get('accuracy'))} | {fmt_pct(s['metrics'].get('pred_correct_rate'))} |"
        )
    extra_md += ["", "## MAP-500 correctness by Category", ""]
    for run_key in keys:
        path = ANALYSIS / f"map_500__{run_key}__by_category.csv"
        if not path.exists():
            continue
        extra_md += [f"### {run_key}", "", md_table(pd.read_csv(path), fmt={
            "correct_accuracy": "{:.1%}",
            "pred_true_rate": "{:.1%}",
            "mean_p_true": "{:.3f}",
            "misconception_accuracy": "{:.1%}",
        })]

    extra_md += [
        "## What each number means",
        "",
        "- **Correct (MAP)**: `noul` on whether the student's *selected MC answer* is the right option (`Category` starts with `True_`). Not whether the explanation is good.",
        "- **Misconception all-classes**: 36-way MAP taxonomy including `none` for unlabelled rows. Accuracy is dominated by `none` because most MAP rows have no misconception label.",
        "- **Misconception labelled-only**: scored only on `True_Misconception` and `False_Misconception` rows. This is the hard 35-way task.",
        "- **Has-misconception vs none**: binary collapse of the taxonomy.",
        "- **MathDial correct**: gold is FALSE for every row (the field is `student_incorrect_solution`). Accuracy = true-negative rate. TRUE-class F1 is 0 by construction. Jev predicted TRUE on 37% of dialogues (over-confident that a wrong solution is right); Gemini predicted TRUE on 3.7%.",
        "- **Latency / tokens**: one HTTP request per row, sequential. Gemini output tokens = candidate tokens + thinking tokens.",
        "",
        "## Error files for qualitative write-up",
        "",
        "Every correctness mistake and every labelled-misconception mistake is in `analysis/<slice>__<provider>__errors.csv` with the question snippet and student answer snippet.",
        "",
    ]

    report = (RESULTS_DIR / "REPORT.md").read_text(encoding="utf-8")
    marker = "## File map"
    if marker in report:
        report = report.replace(marker, "\n".join(extra_md) + "\n" + marker)
    else:
        report += "\n" + "\n".join(extra_md)
    (RESULTS_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    print("Wrote", RESULTS_DIR / "REPORT.md")
    print(json.dumps(tot, indent=2))


if __name__ == "__main__":
    main()
