"""Decision definitions: build Clef `questions` JSON and interpret answers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

NOUL = "noul"
CHOICE = "choice"
DECISION_TYPES = (NOUL, CHOICE)

_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_id(value: str, what: str = "ID") -> str:
    """Return a stripped identifier or raise ValueError."""
    value = (value or "").strip()
    if not value:
        raise ValueError(f"{what} is required.")
    if not _ID_RE.match(value):
        raise ValueError(
            f"{what} '{value}' must start with a letter or underscore and contain only "
            "letters, digits, and underscores."
        )
    return value


def build_noul_question(instructions: str, true_criteria: str = "", false_criteria: str = "") -> dict:
    """Build a yes/no (`noul`) question. Criteria are optional but sent as a pair."""
    if not instructions.strip():
        raise ValueError("Instructions are required.")
    q: dict[str, Any] = {"type": NOUL, "instructions": instructions.strip()}
    t, f = true_criteria.strip(), false_criteria.strip()
    if t or f:
        if not (t and f):
            raise ValueError("Provide both True and False criteria, or neither.")
        q["criteria"] = {"true": t, "false": f}
    return q


def build_choice_question(instructions: str, choices: list[tuple[str, str]]) -> dict:
    """Build a `choice` question from (id, description) pairs."""
    if not instructions.strip():
        raise ValueError("Instructions are required.")
    criteria: dict[str, str] = {}
    for cid, desc in choices:
        if not (cid or "").strip() and not (desc or "").strip():
            continue  # ignore blank editor rows
        cid = validate_id(cid, "Choice ID")
        if cid in criteria:
            raise ValueError(f"Duplicate choice ID '{cid}'.")
        if not (desc or "").strip():
            raise ValueError(f"Choice '{cid}' needs a description.")
        criteria[cid] = desc.strip()
    if len(criteria) < 2:
        raise ValueError("A choice decision needs at least two choices.")
    return {"type": CHOICE, "instructions": instructions.strip(), "criteria": criteria}


def build_questions(decision_id: str, question: dict) -> dict[str, dict]:
    """Wrap a single question under its decision ID, as /v1/systemone expects."""
    return {validate_id(decision_id, "Decision name"): question}


@dataclass
class Decision:
    """A uniform, display-ready view of one Clef answer."""

    type: str
    label: str                     # "TRUE"/"FALSE" for noul; choice ID for choice
    probability: float             # probability of `label`
    confidence: float              # noul: same as probability; choice: Clef's confidence
    probabilities: dict[str, float] = field(default_factory=dict)  # sorted high→low
    noul: float | None = None      # raw P(true) for noul


def interpret_answer(answer: dict[str, Any], threshold: float = 0.5) -> Decision:
    """Turn a raw Clef answer into a Decision.

    noul: ``noul`` ≈ P(true). The decision is TRUE when noul ≥ threshold.
    choice: ``choice`` is the selected ID, ``probabilities`` the distribution,
    ``confidence`` Clef's own confidence (not the same as the top probability).
    """
    atype = answer.get("type")
    if atype == NOUL:
        p_true = float(answer["noul"])
        is_true = p_true >= threshold
        p = p_true if is_true else 1 - p_true
        return Decision(
            type=NOUL,
            label="TRUE" if is_true else "FALSE",
            probability=p,
            confidence=p,
            probabilities={"TRUE": p_true, "FALSE": 1 - p_true},
            noul=p_true,
        )
    if atype == CHOICE:
        probs = {k: float(v) for k, v in answer["probabilities"].items()}
        probs = dict(sorted(probs.items(), key=lambda kv: kv[1], reverse=True))
        label = answer["choice"]
        return Decision(
            type=CHOICE,
            label=label,
            probability=probs.get(label, 0.0),
            confidence=float(answer.get("confidence", probs.get(label, 0.0))),
            probabilities=probs,
        )
    raise ValueError(f"Unsupported answer type: {atype!r}")


# --------------------------------------------------------------------------- presets

FRACTION_STATE = "Question: What is 1/2 + 1/3?\nStudent answer: {answer}"

MISCONCEPTION_CHOICES: list[tuple[str, str]] = [
    ("add_numerators_denominators", "The student adds numerator and denominator directly."),
    ("common_denominator_error", "The student attempts to find a common denominator but does it incorrectly."),
    ("whole_number_reasoning", "The student treats fractions as ordinary whole numbers."),
    ("other", "Another misconception or no identifiable misconception."),
]

PROOF_ERROR_CHOICES: list[tuple[str, str]] = [
    ("division_by_zero", "The proof divides by zero (or cancels a factor that can be zero)."),
    ("false_premise", "An early equation is asserted without justification and is false."),
    ("invalid_square_root", "A square-root or ± step drops a sign or domain restriction."),
    ("circular_reasoning", "The conclusion is assumed in a disguised form."),
    ("notation_abuse", "Equals signs chain quantities that are not actually equal."),
    ("no_error", "The argument is valid; there is no algebraic error."),
]

RUBRIC_CHOICES: list[tuple[str, str]] = [
    ("mastery", "The work is essentially complete, correct, and well justified."),
    ("partial_credit", "The approach is mostly right but a non-trivial step is wrong or missing."),
    ("concept_error", "The student misunderstands a core concept, not just arithmetic."),
    ("off_task", "The writing does not address the assigned prompt."),
    ("copied_without_understanding", "Correct-looking statements appear, but later steps contradict them."),
    ("insufficient_evidence", "Too little work is shown to judge understanding."),
]

POLICY_CHOICES: list[tuple[str, str]] = [
    ("data_exfiltration", "Confidential data was sent to an unauthorized destination."),
    ("access_violation", "Someone used credentials or a role they were not allowed to use."),
    ("unsafe_change", "A production change skipped required review or rollback planning."),
    ("harassment", "The incident is primarily interpersonal misconduct, not a systems issue."),
    ("false_alarm", "No policy was broken; the report is a misunderstanding."),
    ("multiple_violations", "Two or more distinct policy classes apply with similar severity."),
]

FALLACIOUS_PROOF_STATE = """
Assignment: Prove or disprove that if a = b, then 2 = 1. Student: Priya. Course: Intro to Proofs.

Student write-up (submitted as a complete proof):

Let a and b be real numbers with a = b, and assume both are nonzero. Start from the given equality
    a = b.
Multiply both sides by a:
    a² = a b.
Subtract b² from both sides:
    a² − b² = a b − b².
Factor:
    (a − b)(a + b) = b (a − b).
The student writes: "Since a = b, the factor (a − b) appears on both sides, so we may cancel it, as one cancels a common term in an equation, obtaining
    a + b = b."
Substitute a = b:
    b + b = b, hence 2b = b. Because b is nonzero, divide by b:
    2 = 1.

The student then adds a commentary meant to sound careful:

"I checked a few numbers. If a = b = 3, then a² = 9 and ab = 9, so the second line holds. a² − b² = 0 and ab − b² = 0, so the third line holds. Factoring 0 = 0 is true, and cancelling the common factor of 0 feels like cancelling 0/0, but in algebra class we cancelled (x−2) from (x−2)(x+2) = 5(x−2) and got x+2 = 5, which is valid when x ≠ 2. Here a = b is given, so a − b is 0, which is exactly the excluded case of that cancellation. I still think the proof shows that 2 = 1 under the hypothesis a = b, because every line after cancellation is ordinary arithmetic. Therefore the theorem is true, and the usual claim that 1 ≠ 2 must have a hidden extra assumption.

I also considered starting from a² − 2ab + b² = 0, which is (a−b)² = 0, so a = b, which we already know. That does not produce 2 = 1. The cancellation route seems to be the only way to reach the surprising conclusion, so it must be the essential idea.

Finally, if someone objects that we divided by zero, I would reply that we divided by (a−b) only in the sense of cancelling a common factor, and later we divided by b, which is not zero. So there is no division by zero."

Instructor note (ignore for the decision except as context): The algebraic identity (a−b)(a+b) = b(a−b) is true whenever a = b, because both sides are 0. Cancelling (a−b) is invalid precisely when a = b. The later division by b is valid. The student's numerical check of the pre-cancellation lines is consistent with 0 = 0 and does not justify cancellation.
""".strip()

LAB_REPORT_STATE = """
Lab prompt: A cart on a low-friction track is given an initial velocity v0 and then coasts. Students must decide whether a graph of velocity vs time should be closer to constant, linearly decreasing, or exponentially decaying, justify the model with Newton's laws, estimate the coefficient of kinetic friction μk from the data table, and discuss at least one systematic error.

Raw motion data (time t in seconds, velocity v in m/s), collected at 0.20 s intervals:
t: 0.00, 0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40, 1.60, 1.80, 2.00
v: 0.82, 0.79, 0.74, 0.71, 0.66, 0.62, 0.58, 0.53, 0.49, 0.44, 0.41

Track length 2.00 m, cart mass 0.250 kg, local g = 9.80 m/s². Photogate timing uncertainty ±0.01 s. The track was not re-leveled after being bumped at t ≈ 0.9 s.

Student report:

"The velocity is clearly exponentially decaying because it keeps getting smaller and never hits a straight line. Exponential decay is what batteries and cooling coffee do, so a cart should be the same. Newton's first law says objects in motion stay in motion, so without friction v would be constant; with friction it should fall like e^(−kt). I fitted by eye k ≈ 0.35 s⁻¹.

For friction, F = μk N = μk mg, a = −μk g. If the motion were linear, Δv/Δt ≈ (0.41−0.82)/2.00 = −0.205 m/s², so μk = −a/g ≈ 0.021. I will still report μk = 0.35 because that matches my exponential k, and having two methods disagree means we should trust the more sophisticated one.

The graph looks slightly bent, which proves it is not linear. Air resistance is an exponential effect, and even though the speeds are under 1 m/s I think air resistance dominates wood-on-aluminum friction. The bump at 0.9 s is visible as a tiny wiggle and can be ignored.

Systematic error: the photogate might have been 1 cm too high, which would change nothing important. I did not propagate uncertainty. The conclusion is that Newton's laws plus exponential friction explain the table, and μk is 0.35 with no need for a ± range."

Additional unused notes the student pasted from a website: "For small Reynolds number, drag is linear in v; for high Reynolds number, drag is quadratic. Neither is a decaying exponential in v(t) for constant μk on a horizontal track. Horizontal kinetic friction of constant μk predicts v(t) = v0 − μk g t until stop."
""".strip()

POLICY_INCIDENT_STATE = """
Company policy excerpts (abridged):

Access: Production SSH is forbidden except via the break-glass role `prod-oncall`, which requires a logged ticket and a second approver. Personal AWS keys must never be pasted into chat, tickets, or CI logs. Customer PII (email, phone, address, payment tokens) may leave the VPC only through the approved export job `pii-export`, which writes to a company S3 bucket with object lock.

Change management: Weekday production deploys need a reviewed pull request and a passing staging run. Hotfixes on an incident still need a post-facto ticket within 4 hours. Dropping a table, rotating a master secret, or changing auth redirect URLs is a Sev-1 change and needs an explicit rollback command in the ticket.

Conduct: Mocking a coworker's identity, sharing recordings of 1:1s, or using slurs is a conduct violation handled by HR, not security.

Incident timeline (Slack + laptop, times UTC):

14:02 Mira (backend): staging tests failed on payments, I'll patch prod directly so the demo at 15:00 works.
14:04 Mira pastes a stack trace containing customer emails and last4 of cards into #eng-random (public to all 200 employees).
14:11 Mira SSHes to prod with her laptop user `mira`, not `prod-oncall`. No ticket exists.
14:18 Mira runs `psql` and executes DELETE FROM invoices WHERE created_at > now() - interval '2 hours'; intending to clear test rows. Staging and prod share a read-replica naming scheme; she is on prod. 14,402 rows gone. No backup restore command is written down.
14:26 Mira posts in #eng-random: "prod is fine, I used my key because break-glass is slow, do not tell security we can rotate later."
14:31 Jordan replies with a joke about Mira's password being her dog's name. Mira laughs. No slur, no 1:1 recording.
14:40 A customer tweets that invoices vanished. Mira opens a Sev-2 ticket at 16:12 describing only "replica lag."

Question for the model: which policy class best describes the *primary* violation to act on first?
""".strip()

PROOF_AUDIT_INSTRUCTIONS = (
    "Read the entire student proof, the numerical check, and the cancellation argument. "
    "Decide whether the proof is a valid demonstration that 2 = 1. "
    "Consider hidden division by zero, cancellation of (a−b) when a = b, and whether later "
    "arithmetic (including dividing by b) could independently justify the conclusion. "
    "A valid proof must be sound for all values allowed by the hypotheses."
)

LAB_RUBRIC_INSTRUCTIONS = (
    "Score the lab report against the assigned prompt: choose a model for v(t), justify it "
    "with Newton's laws, estimate μk from the table, and discuss systematic error. "
    "Weigh the data (which is close to linear), the student's exponential claim, the two "
    "incompatible μk values, and the unused note about constant friction. Pick the single "
    "best rubric bucket."
)

POLICY_INSTRUCTIONS = (
    "Identify the primary policy violation to act on first. Weigh production SSH without "
    "break-glass, PII in a public channel, an unreviewed destructive SQL on prod, the late "
    "misleading ticket, and the joke in Slack. Do not treat the joke as the main issue unless "
    "nothing else qualifies. Choose exactly one class."
)


def _bundle_token_stress() -> dict[str, dict]:
    """Several decisions on one long proof — compact for Clef/Jev, heavy for LLMs."""
    return {
        "proof_is_valid": build_noul_question(
            PROOF_AUDIT_INSTRUCTIONS,
            "Every step is valid and the conclusion 2 = 1 follows from a = b.",
            "At least one step is invalid, so the proof does not establish 2 = 1.",
        ),
        "cancellation_is_legal": build_noul_question(
            "Is cancelling the common factor (a − b) valid in this student's setting?",
            "Cancellation is valid for the values allowed by the hypotheses (including a = b).",
            "Cancellation is invalid because a = b makes (a − b) = 0.",
        ),
        "numerical_check_saves_the_proof": build_noul_question(
            "Does substituting a = b = 3 and seeing 0 = 0 before cancellation prove the later steps?",
            "The numerical check independently justifies cancellation and the conclusion 2 = 1.",
            "The check only confirms 0 = 0 and does not justify cancelling (a − b).",
        ),
        "divided_by_nonzero_b_only": build_noul_question(
            "The student says there is no division by zero because the last division is by b ≠ 0. Is that a complete defense?",
            "Yes: the only division is by b, which is nonzero, so there is no division-by-zero issue.",
            "No: the earlier cancellation of (a − b) is the division-by-zero (or undefined cancellation) issue.",
        ),
        "hidden_error": build_choice_question(
            "Which description best names the fatal error in the argument?",
            PROOF_ERROR_CHOICES,
        ),
        "should_receive_full_credit": build_noul_question(
            "Should this write-up receive full credit as a correct proof?",
            "Yes, full credit: the theorem is established.",
            "No: the argument is not a correct proof.",
        ),
        "student_noticed_the_danger": build_noul_question(
            "Did the student clearly recognize that cancelling (a − b) is the excluded case, and then correctly reject the proof for that reason?",
            "Yes: they identified the excluded case and therefore withdrew the claim 2 = 1.",
            "No: they mentioned the excluded case but still endorsed the proof.",
        ),
        "instructor_note_agrees_with_student": build_noul_question(
            "Does the instructor note agree that the proof is valid?",
            "The instructor note endorses the student's conclusion that 2 = 1.",
            "The instructor note identifies invalid cancellation when a = b.",
        ),
    }


PRESETS: dict[str, dict[str, Any]] = {
    "Greeting": {
        "decision_name": "says_hello",
        "state": "Hello World",
        "type": NOUL,
        "instructions": "Does the state text contain a greeting?",
        "true": "The state text contains a greeting.",
        "false": "The state text does not contain a greeting.",
    },
    "Fraction correctness": {
        "decision_name": "correct",
        "state": FRACTION_STATE.format(answer="2/5"),
        "type": NOUL,
        "instructions": "Is the student answer mathematically correct?",
        "true": "The student's answer is mathematically correct.",
        "false": "The student's answer is mathematically incorrect.",
    },
    "Fraction misconception (2/5)": {
        "decision_name": "misconception",
        "state": FRACTION_STATE.format(answer="2/5"),
        "type": CHOICE,
        "instructions": "Which misconception best explains the student answer?",
        "choices": MISCONCEPTION_CHOICES,
    },
    "Another student (5/6)": {
        "decision_name": "misconception",
        "state": FRACTION_STATE.format(answer="5/6"),
        "type": CHOICE,
        "instructions": "Which misconception best explains the student answer?",
        "choices": MISCONCEPTION_CHOICES,
    },
    "Hard: fallacious proof (2=1)": {
        "decision_name": "proof_is_valid",
        "state": FALLACIOUS_PROOF_STATE,
        "type": NOUL,
        "instructions": PROOF_AUDIT_INSTRUCTIONS,
        "true": "Every step is valid and the conclusion 2 = 1 follows from a = b.",
        "false": "At least one step is invalid, so the proof does not establish 2 = 1.",
    },
    "Hard: where the proof breaks": {
        "decision_name": "hidden_error",
        "state": FALLACIOUS_PROOF_STATE,
        "type": CHOICE,
        "instructions": "Which description best names the fatal error in the argument? Use the full write-up, not just the slogan 'division by zero'.",
        "choices": PROOF_ERROR_CHOICES,
    },
    "Hard: messy lab report": {
        "decision_name": "lab_rubric",
        "state": LAB_REPORT_STATE,
        "type": CHOICE,
        "instructions": LAB_RUBRIC_INSTRUCTIONS,
        "choices": RUBRIC_CHOICES,
    },
    "Hard: policy incident": {
        "decision_name": "primary_violation",
        "state": POLICY_INCIDENT_STATE,
        "type": CHOICE,
        "instructions": POLICY_INSTRUCTIONS,
        "choices": POLICY_CHOICES,
    },
    "Token stress: 8 proof questions": {
        "decision_name": "proof_is_valid",
        "state": FALLACIOUS_PROOF_STATE,
        "type": NOUL,
        "instructions": PROOF_AUDIT_INSTRUCTIONS,
        "true": "Every step is valid and the conclusion 2 = 1 follows from a = b.",
        "false": "At least one step is invalid, so the proof does not establish 2 = 1.",
        "bundle_questions": _bundle_token_stress(),
    },
}
