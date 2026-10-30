#!/usr/bin/env python3
"""Build the cheaper MAP / MathDial evaluation slices used for the report.

Sampling is stratified by MAP Category. Taking the first 50/500 train rows would
yield only True_* labels, which is useless for a correctness report.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import build_choice_question, build_noul_question  # noqa: E402

RAW_MAP = ROOT / "eval_data/raw/map/train.csv"
RAW_MATHDIAL = ROOT / "eval_data/raw/mathdial/test.jsonl"
OUT = ROOT / "eval_data/slices"
SEED = 42

MAP_MISCONCEPTION_HELP = {
    "Adding_across": "Added numerators with numerators and denominators with denominators, or added across place values incorrectly.",
    "Adding_terms": "Added terms that should have been multiplied, subtracted, or otherwise combined differently.",
    "Additive": "Treated a multiplicative or proportional relationship as if it were additive.",
    "Base_rate": "Ignored or inverted a base rate / proportion when comparing quantities.",
    "Certainty": "Treated an uncertain or estimated quantity as if it were exact.",
    "Definition": "Used an incorrect definition of a mathematical term or operation.",
    "Denominator_only_change": "Changed only the denominator of a fraction (or only the numerator) instead of an equivalent transformation.",
    "Division": "Divided when the situation required a different operation, or inverted the dividend and divisor.",
    "Duplication": "Counted the same quantity twice or duplicated a step.",
    "Firstterm": "Used only the first term or first piece of given information and ignored the rest.",
    "FlipChange": "Flipped a fraction or ratio incorrectly when changing operations (e.g. invert-and-multiply misuse).",
    "Ignores_zeroes": "Dropped, ignored, or misplaced zeroes in a number or decimal.",
    "Incomplete": "Stopped before finishing a required step; the method is otherwise on track but unfinished.",
    "Incorrect_equivalent_fraction_addition": "Tried to add fractions by making equivalents but produced a non-equivalent fraction or added incorrectly after converting.",
    "Interior": "Used an interior / intermediate number from the problem as if it were the answer.",
    "Inverse_operation": "Applied the inverse of the required operation.",
    "Inversion": "Inverted a fraction, ratio, or difference when it should not have been inverted.",
    "Irrelevant": "The explanation is off-topic or does not address the mathematics of the question.",
    "Longer_is_bigger": "Assumed the longer decimal / more digits means a larger number.",
    "Mult": "Multiplied when the situation required a different operation.",
    "Multiplying_by_4": "Multiplied by 4 (or another salient number in the item) without a valid reason.",
    "Not_variable": "Treated a variable as a specific number, or treated a number as if it were a variable.",
    "Positive": "Dropped a negative sign or treated a signed quantity as positive.",
    "Scale": "Used the wrong scale factor or failed to scale a diagram / unit consistently.",
    "Shorter_is_bigger": "Assumed a shorter decimal / fewer digits means a larger number.",
    "Subtraction": "Subtracted when the situation required a different operation, or subtracted in the wrong order.",
    "SwapDividend": "Swapped dividend and divisor.",
    "Tacking": "Tacked an extra unmatched digit, term, or unit onto an otherwise reasonable answer.",
    "Unknowable": "Claimed the answer cannot be determined when it can, or invented missing information.",
    "WNB": "Whole-number bias: interpreted a fraction using whole-number thinking.",
    "Whole_numbers_larger": "Assumed a whole number is larger than a fraction (or vice versa) without comparing values.",
    "Wrong_Fraction": "Wrote or selected the wrong fraction (MAP label Wrong_Fraction).",
    "Wrong_Operation": "Chose the wrong arithmetic operation for the situation.",
    "Wrong_fraction": "Wrote or selected the wrong fraction (MAP label Wrong_fraction).",
    "Wrong_term": "Used the wrong term, addend, factor, or variable from the question.",
    "none": "No labelled MAP misconception. The explanation is correct, or it is neither correct nor a known misconception.",
}


def slug_misconception(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    if not slug:
        return None
    if slug[0].isdigit():
        slug = f"m_{slug}"
    return slug


def stratified_sample(df: pd.DataFrame, n: int, *, col: str, seed: int) -> pd.DataFrame:
    """Proportional sample by ``col``, keeping at least one row per class when possible."""
    sizes = df[col].value_counts()
    raw = (sizes / len(df) * n).round().astype(int)
    for label in sizes.index:
        raw[label] = max(1, int(raw[label]))
        raw[label] = min(int(raw[label]), int(sizes[label]))
    while int(raw.sum()) > n:
        growable = raw[raw > 1]
        raw[growable.idxmax()] -= 1
    while int(raw.sum()) < n:
        room = sizes - raw
        room = room[room > 0]
        if room.empty:
            break
        raw[room.idxmax()] += 1
    parts = [
        df[df[col] == label].sample(n=int(k), random_state=seed)
        for label, k in raw.items()
        if k > 0
    ]
    out = pd.concat(parts, ignore_index=False)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def map_row(row: pd.Series) -> dict:
    mc = "" if pd.isna(row["MC_Answer"]) else str(row["MC_Answer"]).strip()
    expl = "" if pd.isna(row["StudentExplanation"]) else str(row["StudentExplanation"]).strip()
    category = str(row["Category"])
    misconception = slug_misconception(row["Misconception"])
    return {
        "id": f"map_{row['row_id']}",
        "source_row_id": int(row["row_id"]),
        "question_id": int(row["QuestionId"]),
        "question": str(row["QuestionText"]),
        "student_answer": f"Selected answer: {mc}\n\nExplanation: {expl}",
        "mc_answer": mc,
        "student_explanation": expl,
        "map_category": category,
        "ground_truth_correct": category.startswith("True_"),
        "explanation_correct": category.endswith("_Correct"),
        "ground_truth_misconception": misconception or "none",
        "misconception_labelled": bool(misconception),
    }


def prepare_map() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(RAW_MAP)
    full_labels = sorted({slug_misconception(v) for v in raw["Misconception"].dropna().unique()} | {"none"})
    map_50 = pd.DataFrame([map_row(r) for _, r in stratified_sample(raw, 50, col="Category", seed=SEED).iterrows()])
    map_500 = pd.DataFrame([map_row(r) for _, r in stratified_sample(raw, 500, col="Category", seed=SEED).iterrows()])
    questions = {
        "correct": build_noul_question(
            "Decide whether the student's selected multiple-choice answer is the correct "
            "answer to the question. Judge the selected answer, not the quality of the explanation.",
            true_criteria="The selected answer matches the mathematically correct option.",
            false_criteria="The selected answer is not the correct option.",
        ),
        "misconception": build_choice_question(
            "Classify the misconception in the student's explanation using the MAP taxonomy. "
            "Choose none if the explanation is correct or does not match a listed misconception.",
            [(cid, MAP_MISCONCEPTION_HELP[cid]) for cid in full_labels],
        ),
    }
    return map_50, map_500, questions


def prepare_mathdial() -> tuple[pd.DataFrame, dict]:
    rows = []
    for line in RAW_MATHDIAL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        rows.append({
            "id": f"mathdial_{rec.get('qid')}",
            "qid": rec.get("qid"),
            "scenario": rec.get("scenario"),
            "question": rec.get("question") or "",
            "student_answer": rec.get("student_incorrect_solution") or "",
            "gold_solution": rec.get("ground_truth") or "",
            "student_profile": rec.get("student_profile") or "",
            "teacher_described_confusion": rec.get("teacher_described_confusion") or "",
            "self_correctness": rec.get("self-correctness"),
            "ground_truth_correct": False,
            "ground_truth_misconception": "",
        })
    df = pd.DataFrame(rows)
    questions = {
        "correct": build_noul_question(
            "Decide whether the student's written solution is mathematically correct and complete "
            "for the question. The student has not yet been tutored.",
            true_criteria="The written solution reaches the right answer with valid reasoning.",
            false_criteria="The written solution is incorrect, incomplete, or uses invalid reasoning.",
        ),
    }
    return df, questions


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    map_50, map_500, map_questions = prepare_map()
    mathdial, mathdial_questions = prepare_mathdial()

    map_50.to_csv(OUT / "map_50.csv", index=False)
    map_500.to_csv(OUT / "map_500.csv", index=False)
    mathdial.to_csv(OUT / "mathdial_test.csv", index=False)
    (OUT / "map_questions.json").write_text(json.dumps(map_questions, indent=2), encoding="utf-8")
    (OUT / "mathdial_questions.json").write_text(json.dumps(mathdial_questions, indent=2), encoding="utf-8")

    def mix(df: pd.DataFrame) -> dict:
        out: dict[str, object] = {}
        if "map_category" in df.columns:
            out["map_category"] = df["map_category"].value_counts().to_dict()
        if "ground_truth_correct" in df.columns:
            out["ground_truth_correct"] = {
                "true": int(df["ground_truth_correct"].astype(bool).sum()),
                "false": int((~df["ground_truth_correct"].astype(bool)).sum()),
            }
        if "misconception_labelled" in df.columns:
            out["misconception_labelled"] = int(df["misconception_labelled"].astype(bool).sum())
        return out

    manifest = {
        "seed": SEED,
        "sampling": (
            "MAP slices are stratified by Category with a fixed seed, not the first N train rows. "
            "The first 50/500 MAP train rows are all True_* (correct MC answers), which would make "
            "correctness accuracy uninterpretable. MathDial uses the official test split (599 rows)."
        ),
        "slices": {
            "map_50": {"rows": len(map_50), "mix": mix(map_50), "csv": "map_50.csv", "questions": "map_questions.json"},
            "map_500": {"rows": len(map_500), "mix": mix(map_500), "csv": "map_500.csv", "questions": "map_questions.json"},
            "mathdial_test": {
                "rows": len(mathdial),
                "mix": mix(mathdial),
                "csv": "mathdial_test.csv",
                "questions": "mathdial_questions.json",
            },
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
