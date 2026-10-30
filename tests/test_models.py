import pytest

from models import (
    MISCONCEPTION_CHOICES,
    PRESETS,
    build_choice_question,
    build_noul_question,
    build_questions,
    interpret_answer,
)


def test_noul_question_matches_spec():
    q = build_questions("correct", build_noul_question(
        "Is the answer correct?", "The answer is mathematically correct.", "The answer is mathematically incorrect."))
    assert q == {"correct": {
        "type": "noul",
        "instructions": "Is the answer correct?",
        "criteria": {"true": "The answer is mathematically correct.",
                     "false": "The answer is mathematically incorrect."},
    }}


def test_noul_criteria_optional_but_paired():
    assert "criteria" not in build_noul_question("Q?")
    with pytest.raises(ValueError):
        build_noul_question("Q?", "only true", "")


def test_choice_question_skips_blank_rows_and_rejects_duplicates():
    q = build_choice_question("Which?", MISCONCEPTION_CHOICES + [("", "")])
    assert q["type"] == "choice" and list(q["criteria"]) == [c for c, _ in MISCONCEPTION_CHOICES]
    with pytest.raises(ValueError, match="Duplicate"):
        build_choice_question("Which?", [("a", "x"), ("a", "y")])
    with pytest.raises(ValueError, match="at least two"):
        build_choice_question("Which?", [("a", "x")])


def test_invalid_decision_name():
    with pytest.raises(ValueError):
        build_questions("bad name", build_noul_question("Q?"))


def test_interpret_noul():
    d = interpret_answer({"type": "noul", "noul": 0.9752018398044143})
    assert d.label == "TRUE" and d.probability == pytest.approx(0.9752, abs=1e-4)
    assert d.probabilities["FALSE"] == pytest.approx(0.0248, abs=1e-4)
    d = interpret_answer({"type": "noul", "noul": 0.0108})
    assert d.label == "FALSE" and d.probability == pytest.approx(0.9892)


def test_interpret_choice_sorted():
    d = interpret_answer({
        "type": "choice", "choice": "add_numerators_denominators", "confidence": 0.6896,
        "probabilities": {"add_numerators_denominators": 0.8997, "common_denominator_error": 0.0328,
                          "whole_number_reasoning": 0.0473, "other": 0.0202},
    })
    assert d.label == "add_numerators_denominators"
    assert d.probability == 0.8997 and d.confidence == 0.6896
    assert list(d.probabilities)[:2] == ["add_numerators_denominators", "whole_number_reasoning"]


def test_hard_presets_and_token_stress_bundle():
    assert "Hard: fallacious proof (2=1)" in PRESETS
    assert "Token stress: 8 proof questions" in PRESETS
    proof = PRESETS["Hard: fallacious proof (2=1)"]
    assert "a = b" in proof["state"] and len(proof["state"]) > 800
    bundle = PRESETS["Token stress: 8 proof questions"]["bundle_questions"]
    assert len(bundle) == 8
    assert bundle["proof_is_valid"]["type"] == "noul"
    assert bundle["hidden_error"]["type"] == "choice"
    assert "division_by_zero" in bundle["hidden_error"]["criteria"]
