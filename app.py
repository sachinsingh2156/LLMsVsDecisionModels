"""Clef Decision Lab — compare Clef, TypeSafe Jev, Gemini, and local Qwen."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

import evaluator as ev
from clef_client import (
    DEFAULT_BASE_URL,
    DEFAULT_JEV_MODEL,
    DEFAULT_MODEL,
    JEV_MODELS,
    check_ollama,
    check_typesafe,
)
from llm_client import DEFAULT_GEMINI_MODEL, GEMINI_MODELS, check_gemini
from models import (
    DECISION_TYPES,
    MISCONCEPTION_CHOICES,
    NOUL,
    PRESETS,
    build_choice_question,
    build_noul_question,
    build_questions,
    interpret_answer,
)
from providers import ProviderConfig, run_one, run_selected

# Palette (validated reference instance): blue = selected / TRUE, red = FALSE, gray = rest.
BLUE = "#2a78d6"
RED = "#e34948"

HELP_CLEF = (
    "Clef and TypeSafe Jev are decision models: they evaluate predefined decisions "
    "and return probabilities rather than generating a conventional text response. "
    "Gemini and local Qwen are prompted to return the same JSON; their probabilities "
    "are model-reported, not calibrated."
)
HELP_NOUL = "noul ≈ probability of the positive/true decision. TRUE when noul ≥ 0.5."
HELP_CHOICE = (
    "choice = selected category · probabilities = distribution over categories · "
    "confidence = the model's own confidence in the decision (reported separately; it is "
    "not the same as the top probability)."
)

st.set_page_config(page_title="Clef Decision Lab", page_icon="◆", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 2.2rem;}
.lab-title {font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: -0.01em;}
.lab-sub {color: #8a8984; margin: 0 0 .25rem 0;}
.status {display:inline-flex; gap:.5rem; align-items:center; padding:.35rem .75rem;
         border:1px solid rgba(128,128,128,.25); border-radius:999px; font-size:.85rem;}
.dot {width:.6rem; height:.6rem; border-radius:50%; display:inline-block;}
.card {border:1px solid rgba(128,128,128,.25); border-radius:10px; padding:.7rem .9rem;
       background: rgba(128,128,128,.04); margin-bottom:.5rem; overflow:visible;}
.card .k {font-size:.7rem; text-transform:uppercase; letter-spacing:.06em; color:#8a8984;}
.card .v, .card .big {font-size:1.05rem; font-weight:700; word-break:break-word;
                      overflow-wrap:anywhere; line-height:1.3;}
.split {display:flex; height:14px; border-radius:7px; overflow:hidden; gap:2px; margin:.4rem 0;}
.legend-row {display:flex; justify-content:space-between; font-size:.85rem; color:#8a8984;}
.badge-row {display:flex; flex-wrap:wrap; gap:.4rem; justify-content:flex-end; margin-top:.8rem;}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- state

def _init_state() -> None:
    defaults = {
        "f_name": "student_answer_evaluation",
        "f_state": "Question: What is 1/2 + 1/3?\nStudent answer: 2/5",
        "f_type": NOUL,
        "f_instr": "Is the student's answer mathematically correct?",
        "f_true": "The student's answer is mathematically correct.",
        "f_false": "The student's answer is mathematically incorrect.",
        "choices": [("", ""), ("", "")],
        "editor_ver": 0,
        "last": None,        # {"qid", "questions", "runs": [ModelRun]} or None
        "bundle_questions": None,
        "history": [],
        "typesafe_key": os.environ.get("TYPESAFE_API_KEY", ""),
        "gemini_key": os.environ.get("GEMINI_API_KEY", ""),
        "use_clef": True,
        "use_jev": bool(os.environ.get("TYPESAFE_API_KEY", "")),
        "use_gemini": bool(os.environ.get("GEMINI_API_KEY", "")),
        "use_qwen": False,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def load_preset(name: str) -> None:
    p = PRESETS[name]
    st.session_state.f_name = p["decision_name"]
    st.session_state.f_state = p["state"]
    st.session_state.f_type = p["type"]
    st.session_state.f_instr = p["instructions"]
    st.session_state.f_true = p.get("true", "")
    st.session_state.f_false = p.get("false", "")
    st.session_state.choices = list(p.get("choices", [("", ""), ("", "")]))
    st.session_state.bundle_questions = p.get("bundle_questions")
    st.session_state.editor_ver += 1
    st.session_state.last = None


def clear_form() -> None:
    for k in ("f_name", "f_state", "f_instr", "f_true", "f_false"):
        st.session_state[k] = ""
    st.session_state.choices = [("", ""), ("", "")]
    st.session_state.bundle_questions = None
    st.session_state.editor_ver += 1
    st.session_state.last = None


def _opt_index(options: list[str], preferred: str) -> int:
    for i, o in enumerate(options):
        if o in (preferred, f"{preferred}:latest"):
            return i
    return 0


_init_state()


@st.cache_data(ttl=15, show_spinner=False)
def cached_status(base_url: str):
    return check_ollama(base_url)


@st.cache_data(ttl=15, show_spinner=False)
def cached_typesafe(api_key: str):
    return check_typesafe(api_key)


@st.cache_data(ttl=15, show_spinner=False)
def cached_gemini(api_key: str):
    return check_gemini(api_key)


# --------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.subheader("Ollama")
    base_url = st.text_input("Ollama URL", DEFAULT_BASE_URL)
    status = cached_status(base_url)
    if st.button("Recheck connections", width="stretch"):
        cached_status.clear()
        cached_typesafe.clear()
        cached_gemini.clear()
        st.rerun()

    st.divider()
    st.subheader("Models to compare")

    if "use_qwen_init" not in st.session_state:
        st.session_state.use_qwen = bool(status.qwen_models())
        st.session_state.use_qwen_init = True

    use_clef = st.checkbox("Clef (local decision)", key="use_clef",
                           help="Ollama POST /v1/systemone. Requires a decision model such as clef.")
    decision_options = status.decision_models or [DEFAULT_MODEL]
    clef_model = st.selectbox(
        "Clef model", decision_options, index=_opt_index(decision_options, DEFAULT_MODEL),
        disabled=not use_clef,
        help="Only models advertising the 'decision' capability in /api/tags are listed.",
    )

    use_jev = st.checkbox(
        "TypeSafe Jev",
        key="use_jev",
        help="Hosted System One model. Same noul/choice JSON as Clef.",
    )
    jev_model = st.selectbox("Jev model", list(JEV_MODELS), index=0, disabled=not use_jev)
    st.text_input("TypeSafe API key", type="password", key="typesafe_key", disabled=not use_jev)

    use_gemini = st.checkbox(
        "Gemini",
        key="use_gemini",
        help="Google Gemini, prompted for the same decision JSON. Probabilities are not calibrated.",
    )
    gemini_model = st.selectbox(
        "Gemini model", list(GEMINI_MODELS),
        index=_opt_index(list(GEMINI_MODELS), DEFAULT_GEMINI_MODEL),
        disabled=not use_gemini,
    )
    st.text_input("Gemini API key", type="password", key="gemini_key", disabled=not use_gemini)

    qwen_options = status.qwen_models() or [m for m in status.generate_models if m]
    use_qwen = st.checkbox(
        "Qwen (local Ollama)",
        key="use_qwen",
        help="Local chat model via Ollama /api/chat, prompted for the same decision JSON.",
    )
    if qwen_options:
        qwen_model = st.selectbox("Qwen model", qwen_options, disabled=not use_qwen)
    else:
        qwen_model = st.text_input("Qwen model", value="qwen2.5", disabled=not use_qwen,
                                   help="No qwen* model found in Ollama. Pull one, or type the tag.")

    st.divider()
    st.subheader("Example Experiments")
    for preset_name in PRESETS:
        st.button(preset_name, on_click=load_preset, args=(preset_name,), width="stretch")

    st.divider()
    with st.expander("About the models", expanded=False):
        st.markdown(f"{HELP_CLEF}\n\n**noul** — {HELP_NOUL}\n\n**choice** — {HELP_CHOICE}")
        st.caption(
            "Clef/Jev: `POST /v1/systemone`. `ollama run clef` does not work. "
            "Qwen uses `POST /api/chat` and `ollama run qwen*` is fine."
        )


configs: list[ProviderConfig] = []
if use_clef:
    configs.append(ProviderConfig("clef", clef_model, base_url=base_url))
if use_jev:
    configs.append(ProviderConfig("typesafe", jev_model, api_key=st.session_state.typesafe_key))
if use_gemini:
    configs.append(ProviderConfig("gemini", gemini_model, api_key=st.session_state.gemini_key))
if use_qwen:
    configs.append(ProviderConfig("qwen", qwen_model, base_url=base_url))


def _badge(color: str, text: str) -> str:
    return (
        f'<span class="status"><span class="dot" style="background:{color}"></span>{text}</span>'
    )


badges: list[str] = []
if use_clef:
    if status.online and status.has_model(clef_model):
        badges.append(_badge("#1baf7a", f"<b>Clef</b> · {clef_model}"))
    elif status.online:
        badges.append(_badge("#eda100", f"<b>Clef</b> · “{clef_model}” not found"))
    else:
        badges.append(_badge("#e34948", "<b>Clef</b> · Ollama offline"))
if use_jev:
    if st.session_state.typesafe_key:
        js = cached_typesafe(st.session_state.typesafe_key)
        if js.online:
            badges.append(_badge("#1baf7a", f"<b>Jev</b> · {jev_model}"))
        else:
            badges.append(_badge("#e34948", f"<b>Jev</b> · {js.error or 'offline'}"))
    else:
        badges.append(_badge("#eda100", "<b>Jev</b> · API key missing"))
if use_gemini:
    if st.session_state.gemini_key:
        gs = cached_gemini(st.session_state.gemini_key)
        if gs.online:
            badges.append(_badge("#1baf7a", f"<b>Gemini</b> · {gemini_model}"))
        else:
            badges.append(_badge("#e34948", f"<b>Gemini</b> · {gs.error or 'offline'}"))
    else:
        badges.append(_badge("#eda100", "<b>Gemini</b> · API key missing"))
if use_qwen:
    if status.online and status.has_generate_model(qwen_model):
        badges.append(_badge("#1baf7a", f"<b>Qwen</b> · {qwen_model}"))
    elif status.online:
        badges.append(_badge("#eda100", f"<b>Qwen</b> · “{qwen_model}” not listed"))
    else:
        badges.append(_badge("#e34948", "<b>Qwen</b> · Ollama offline"))
if not badges:
    badges.append(_badge("#eda100", "<b>No models selected</b>"))


# --------------------------------------------------------------------------- header

head_l, head_r = st.columns([3, 2])
with head_l:
    st.markdown('<p class="lab-title">Clef Decision Lab</p>', unsafe_allow_html=True)
    st.markdown('<p class="lab-sub">Compare Clef, TypeSafe Jev, Gemini, and local Qwen</p>',
                unsafe_allow_html=True)
with head_r:
    st.markdown(f'<div class="badge-row">{"".join(badges)}</div>', unsafe_allow_html=True)
if use_clef and not status.online:
    st.error(status.error)
st.caption(HELP_CLEF)


# --------------------------------------------------------------------------- result rendering

def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def render_noul(d) -> None:
    p_true = d.noul
    st.dataframe(
        pd.DataFrame([{"Decision": d.label, "P(true)": p_true, "P(false)": 1 - p_true}]),
        width="stretch",
        hide_index=True,
        column_config={
            "Decision": st.column_config.TextColumn("Decision", width="small"),
            "P(true)": st.column_config.ProgressColumn("P(true)", min_value=0, max_value=1, format="percent"),
            "P(false)": st.column_config.ProgressColumn("P(false)", min_value=0, max_value=1, format="percent"),
        },
    )
    st.markdown(
        f'<div class="split"><div style="width:{p_true*100:.2f}%;background:{BLUE}"></div>'
        f'<div style="flex:1;background:{RED}"></div></div>'
        f'<div class="legend-row"><span>TRUE {pct(p_true)}</span><span>FALSE {pct(1-p_true)}</span></div>',
        unsafe_allow_html=True,
    )


def render_choice(d) -> None:
    st.dataframe(
        pd.DataFrame([{"Selected": d.label, "Probability": d.probability, "Confidence": d.confidence}]),
        width="stretch",
        hide_index=True,
        column_config={
            "Selected": st.column_config.TextColumn("Selected decision", width="large"),
            "Probability": st.column_config.ProgressColumn("Probability", min_value=0, max_value=1, format="percent"),
            "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="percent"),
        },
    )
    st.caption("Choice probabilities", help=HELP_CHOICE)
    st.dataframe(
        pd.DataFrame({"choice": list(d.probabilities), "probability": list(d.probabilities.values())}),
        width="stretch",
        hide_index=True,
        column_config={
            "choice": st.column_config.TextColumn("choice", width="large"),
            "probability": st.column_config.ProgressColumn("probability", min_value=0, max_value=1, format="percent"),
        },
    )


def render_meta(run) -> None:
    tok_s = run.tokens_per_sec
    st.dataframe(
        pd.DataFrame([{
            "Model": run.model,
            "Latency (ms)": None if run.latency_ms is None else round(run.latency_ms),
            "Input tokens": run.input_tokens,
            "Output tokens": run.output_tokens,
            "Total tokens": run.total_tokens,
            "Tok/s": None if tok_s is None else round(tok_s, 1),
        }]),
        width="stretch",
        hide_index=True,
        column_config={
            "Model": st.column_config.TextColumn("Model", width="large"),
        },
    )


def comparison_table(runs, qids) -> pd.DataFrame:
    if isinstance(qids, str):
        qids = [qids]
    rows = []
    multi = len(qids) > 1
    for qid in qids:
        for run in runs:
            row = {
                "Provider": run.label,
                "Model": run.model,
                "Decision": "ERROR" if not run.ok else None,
                "Probability": None,
                "Confidence": None,
                "Latency (ms)": None if run.latency_ms is None else round(run.latency_ms),
                "Input tokens": run.input_tokens,
                "Output tokens": run.output_tokens,
                "Total tokens": run.total_tokens,
                "Tok/s": None if run.tokens_per_sec is None else round(run.tokens_per_sec, 2),
                "Error": run.error.splitlines()[0] if run.error else None,
            }
            if multi:
                row = {"Question": qid, **row}
            if run.ok and qid in run.answers:
                d = interpret_answer(run.answers[qid])
                row["Decision"] = d.label
                row["Probability"] = d.probability
                row["Confidence"] = d.confidence
            rows.append(row)
    return pd.DataFrame(rows)


def history_entry(run, *, time_s: str, qid: str, qtype: str, state: str) -> dict:
    entry = {
        "Time": time_s,
        "Provider": run.label,
        "Model": run.model,
        "Decision": qid,
        "Type": qtype,
        "State": state,
        "Result": "ERROR",
        "Confidence": None,
        "Probability": None,
        "Latency (ms)": None if run.latency_ms is None else round(run.latency_ms),
        "Input tokens": run.input_tokens,
        "Output tokens": run.output_tokens,
        "Total tokens": run.total_tokens,
    }
    if run.ok and qid in run.answers:
        d = interpret_answer(run.answers[qid])
        entry.update(Result=d.label, Confidence=d.confidence, Probability=d.probability)
    return entry


def render_run_card(run, qid: str) -> None:
    if not run.ok:
        st.error(run.error)
        with st.expander("Error details"):
            st.json({"request": run.request, "error_body": run.error_body})
        return
    d = interpret_answer(run.answers[qid])
    st.caption(f"Decision `{qid}` · type `{d.type}`",
               help=HELP_NOUL if d.type == NOUL else HELP_CHOICE)
    if d.type == NOUL:
        render_noul(d)
    else:
        render_choice(d)
    render_meta(run)
    with st.expander("Raw response"):
        st.json(run.raw_response)
        st.markdown("**Request payload**")
        st.json(run.request)


# --------------------------------------------------------------------------- tabs

tab_play, tab_eval = st.tabs(["Playground", "Evaluation"])

with tab_play:
    st.subheader("Input")
    st.text_input("Decision name", key="f_name",
                  help="Becomes the question key in the request, e.g. student_answer_evaluation.")
    st.text_area("State / Context", key="f_state", height=160,
                 help="The text each model evaluates. Sent as the `state` field.")
    st.selectbox("Decision type", DECISION_TYPES, key="f_type",
                 help=f"noul: {HELP_NOUL}\n\nchoice: {HELP_CHOICE}")
    st.text_area("Instructions", key="f_instr", height=80,
                 help="The decision question answered about the state.")

    current_choices = st.session_state.choices
    if st.session_state.f_type == NOUL:
        st.text_area("True criteria", key="f_true", height=68,
                     help="Optional. Describes when the answer is TRUE.")
        st.text_area("False criteria", key="f_false", height=68,
                     help="Optional. Describes when the answer is FALSE.")
    else:
        st.markdown("**Choices**", help="Add rows with the + below the table. ID must be an identifier.")
        edited = st.data_editor(
            pd.DataFrame(st.session_state.choices, columns=["id", "description"]),
            key=f"choices_{st.session_state.editor_ver}",
            num_rows="dynamic",
            width="stretch",
            hide_index=True,
            column_config={
                "id": st.column_config.TextColumn("ID", width="medium", required=True),
                "description": st.column_config.TextColumn("Description", width="large"),
            },
        )
        current_choices = [
            (str(r["id"] or ""), str(r["description"] or ""))
            for _, r in edited.fillna("").iterrows()
        ]

    bundle = st.session_state.get("bundle_questions")
    if bundle:
        st.info(
            f"This experiment sends **{len(bundle)} decisions in one request**. "
            "Clef and Jev still return compact probabilities; Gemini and Qwen must reason "
            "through every question, which is what drives the token gap. "
            "The question set stays fixed until you pick another experiment or Clear."
        )

    b1, b2 = st.columns(2)
    run_clicked = b1.button("Run comparison", type="primary", width="stretch",
                            disabled=not configs)
    b2.button("Clear", on_click=clear_form, width="stretch")
    if not configs:
        st.caption("Select at least one model in the sidebar.")

    if run_clicked:
        try:
            if st.session_state.f_type == NOUL:
                q = build_noul_question(st.session_state.f_instr, st.session_state.f_true,
                                        st.session_state.f_false)
            else:
                q = build_choice_question(st.session_state.f_instr, current_choices)
                st.session_state.choices = current_choices
            bundle_q = st.session_state.get("bundle_questions")
            questions = dict(bundle_q) if bundle_q else build_questions(st.session_state.f_name, q)
            if not st.session_state.f_state.strip():
                raise ValueError("State / Context is required.")
        except ValueError as e:
            st.warning(str(e))
        else:
            qid = next(iter(questions))
            names = ", ".join(c.model for c in configs)
            with st.spinner(f"Asking {names}…"):
                runs = run_selected(st.session_state.f_state, questions, configs)
            st.session_state.last = {"qid": qid, "questions": questions, "runs": runs}
            now = datetime.now().strftime("%H:%M:%S")
            for run in runs:
                for hqid, hq in questions.items():
                    st.session_state.history.append(history_entry(
                        run, time_s=now, qid=hqid, qtype=hq.get("type", q["type"]),
                        state=st.session_state.f_state,
                    ))

    st.divider()
    st.subheader("Result")
    last = st.session_state.last
    if last is None:
        st.info("Define a decision, select models in the sidebar, and click **Run comparison**.")
    else:
        qids = list(last["questions"])
        qid = last["qid"]
        runs = last["runs"]
        cmp_cfg = {
            "Question": st.column_config.TextColumn("Question", width="medium"),
            "Provider": st.column_config.TextColumn("Provider", width="medium"),
            "Model": st.column_config.TextColumn("Model", width="medium"),
            "Decision": st.column_config.TextColumn("Decision", width="medium"),
            "Probability": st.column_config.ProgressColumn("Probability", min_value=0, max_value=1, format="percent"),
            "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="percent"),
            "Error": st.column_config.TextColumn("Error", width="large"),
        }
        st.markdown("**Comparison**")
        st.dataframe(comparison_table(runs, qids), width="stretch", hide_index=True, column_config=cmp_cfg)
        if any(r.provider in ("gemini", "qwen") and r.ok for r in runs):
            st.caption(
                "Gemini and Qwen probabilities come from prompted JSON, not a native decision head. "
                "Their token counts include written reasoning (and Gemini thinking tokens)."
            )
        st.markdown("**Per model**")
        for run in runs:
            with st.expander(f"{run.label} · {run.model}", expanded=False):
                render_run_card(run, qid)

    st.divider()
    st.subheader("Test history")
    hist = st.session_state.history
    if not hist:
        st.caption("No runs yet in this session.")
    else:
        hdf = pd.DataFrame(hist)
        ok = hdf[hdf["Result"] != "ERROR"]
        m = st.columns(5)
        m[0].metric("Runs", len(hdf))
        m[1].metric("Failed", int((hdf["Result"] == "ERROR").sum()))
        m[2].metric("Avg latency", f"{ok['Latency (ms)'].mean():,.0f} ms" if len(ok) else "—")
        m[3].metric("Median latency", f"{ok['Latency (ms)'].median():,.0f} ms" if len(ok) else "—")
        m[4].metric("Avg confidence", pct(ok["Confidence"].mean()) if len(ok) else "—")
        hist_cols = [c for c in (
            "Time", "Provider", "Model", "Decision", "Result", "Confidence",
            "Latency (ms)", "Input tokens", "Output tokens", "Total tokens", "Type", "State",
        ) if c in hdf.columns]
        st.dataframe(
            hdf.iloc[::-1][hist_cols],
            width="stretch",
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="percent"),
                "State": st.column_config.TextColumn("State / question", width="large"),
            },
        )
        st.button("Clear History", on_click=lambda: st.session_state.update(history=[]))

SAMPLE_CSV = Path(__file__).parent / "sample_data" / "fractions_sample.csv"
ORANGE = "#eb6834"


def fmt_ms(x: float) -> str:
    return "—" if pd.isna(x) else f"{x:,.0f} ms"


def fmt_num(x, digits: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:,.{digits}f}"


def confusion_heatmap(cm: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=cm.values, x=list(cm.columns), y=list(cm.index),
        colorscale=[[0, "#f0efec"], [0.0001, "#cde2fb"], [0.5, "#5598e7"], [1, "#104281"]],
        text=cm.values, texttemplate="%{text}", showscale=False, xgap=2, ygap=2,
        hovertemplate="truth <b>%{y}</b><br>predicted <b>%{x}</b><br>%{z} rows<extra></extra>",
    ))
    fig.update_layout(
        height=max(260, 60 * len(cm) + 120), margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="prediction", side="bottom", automargin=True),
        yaxis=dict(title="ground truth", autorange="reversed", automargin=True),
    )
    return fig


def per_class_chart(per_class: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, color in (("precision", BLUE), ("recall", ORANGE)):
        fig.add_bar(
            y=per_class["class"], x=per_class[col], name=col, orientation="h",
            marker=dict(color=color, line=dict(width=0)),
            hovertemplate=f"<b>%{{y}}</b><br>{col} %{{x:.1%}}<extra></extra>",
        )
    fig.update_layout(
        barmode="group", bargap=0.3, bargroupgap=0.08,
        height=max(220, 70 * len(per_class) + 60), margin=dict(l=10, r=10, t=30, b=30),
        xaxis=dict(range=[0, 1], tickformat=".0%", gridcolor="rgba(128,128,128,0.15)"),
        yaxis=dict(autorange="reversed", automargin=True),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return fig


def render_eval_group(results: pd.DataFrame, labels: list[str] | None, heading: str | None) -> None:
    if heading:
        st.markdown(heading)
    lat = ev.latency_stats(results)
    tok = ev.token_stats(results)
    st.markdown("**Model / system**")
    m = st.columns(6)
    m[0].metric("Total requests", lat["total_requests"])
    m[1].metric("Failed requests", lat["failed_requests"])
    m[2].metric("Avg latency", fmt_ms(lat["avg_ms"]))
    m[3].metric("P95 latency", fmt_ms(lat["p95_ms"]))
    m[4].metric("Avg input tok", fmt_num(tok["avg_input_tokens"], 0))
    m[5].metric("Avg output tok", fmt_num(tok["avg_output_tokens"], 0))
    if lat["failed_requests"]:
        st.warning(f"{lat['failed_requests']} request(s) failed, for example: "
                   f"{results['error'].dropna().iloc[0]}")

    cm_ = ev.correctness_metrics(results)
    if cm_:
        st.markdown(f"**Correctness** (TRUE = positive class, n = {cm_['n']})")
        m = st.columns(4)
        m[0].metric("Accuracy", pct(cm_["accuracy"]))
        m[1].metric("Precision", pct(cm_["precision"]))
        m[2].metric("Recall", pct(cm_["recall"]))
        m[3].metric("F1", pct(cm_["f1"]))
        st.caption(f"TP {cm_['tp']} · FP {cm_['fp']} · FN {cm_['fn']} · TN {cm_['tn']}")

    mm = ev.misconception_metrics(results, labels)
    if mm:
        st.markdown(f"**Misconception** (rows with a ground-truth label, n = {mm['n']})")
        m = st.columns(2)
        m[0].metric("Accuracy", pct(mm["accuracy"]))
        m[1].metric("Macro F1", pct(mm["macro_f1"]),
                    help="Mean F1 over classes that appear in the ground truth.")
        g_l, g_r = st.columns(2, gap="large")
        with g_l:
            st.markdown("Per-class precision & recall")
            st.plotly_chart(per_class_chart(mm["per_class"]), width="stretch")
        with g_r:
            st.markdown("Confusion matrix")
            st.plotly_chart(confusion_heatmap(mm["confusion"]), width="stretch")
        st.dataframe(
            mm["per_class"], hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="percent")
                           for c in ("precision", "recall", "f1")},
        )


with tab_eval:
    st.markdown(
        "Upload a labelled CSV and score every selected model against ground truth. Each row is sent "
        "**sequentially** as one request containing the enabled decisions. The state is "
        '`{"question": ..., "student_answer": ...}`.'
    )
    st.caption(
        f"Expected columns: `id`, `question`, `student_answer`, `{ev.GT_CORRECT}`, `{ev.GT_MISCONCEPTION}`. "
        "Leave the misconception empty for rows with no misconception; those rows are skipped for misconception metrics. "
        "Models are selected in the sidebar."
    )

    up_l, up_r = st.columns([3, 1])
    uploaded = up_l.file_uploader("Dataset CSV", type=["csv"])
    use_sample = up_r.toggle("Use sample dataset", value=uploaded is None,
                             help=f"10 toy fraction items in {SAMPLE_CSV.relative_to(SAMPLE_CSV.parent.parent)}")

    dataset: pd.DataFrame | None = None
    try:
        if uploaded is not None and not use_sample:
            dataset = pd.read_csv(uploaded)
        elif use_sample:
            dataset = pd.read_csv(SAMPLE_CSV)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

    if dataset is not None:
        problems = ev.validate_dataset(dataset)
        with st.expander(f"Dataset preview — {len(dataset)} rows", expanded=False):
            st.dataframe(dataset, width="stretch", hide_index=True)
        for p in problems:
            st.error(p)

        st.markdown("#### Decisions to evaluate")
        c_l, c_r = st.columns(2, gap="large")
        with c_l:
            do_correct = st.checkbox("Correctness (noul)", value=ev.GT_CORRECT in dataset.columns,
                                     disabled=ev.GT_CORRECT not in dataset.columns, help=HELP_NOUL)
            e_instr = st.text_area("Instructions", "Is the student answer mathematically correct?",
                                   key="e_instr", height=68, disabled=not do_correct)
            e_true = st.text_input("True criteria", "The student's answer is mathematically correct.",
                                   key="e_true", disabled=not do_correct)
            e_false = st.text_input("False criteria", "The student's answer is mathematically incorrect.",
                                    key="e_false", disabled=not do_correct)
            threshold = st.slider("TRUE threshold on noul", 0.05, 0.95, 0.5, 0.05, disabled=not do_correct,
                                  help="Predict TRUE when noul ≥ threshold.")
        with c_r:
            do_misc = st.checkbox("Misconception (choice)", value=ev.GT_MISCONCEPTION in dataset.columns,
                                  disabled=ev.GT_MISCONCEPTION not in dataset.columns, help=HELP_CHOICE)
            m_instr = st.text_area("Instructions", "Which misconception best explains the student answer?",
                                   key="m_instr", height=68, disabled=not do_misc)
            m_choices = st.data_editor(
                pd.DataFrame(MISCONCEPTION_CHOICES, columns=["id", "description"]),
                key="m_choices", num_rows="dynamic", width="stretch", hide_index=True, disabled=not do_misc,
            )

        n = len(dataset)
        go_clicked = st.button(
            f"Run evaluation on {n} rows × {len(configs)} model(s)",
            type="primary",
            disabled=bool(problems) or not (do_correct or do_misc) or not configs,
        )
        if go_clicked:
            try:
                questions: dict[str, dict] = {}
                if do_correct:
                    questions[ev.CORRECT_QID] = build_noul_question(e_instr, e_true, e_false)
                if do_misc:
                    pairs = [(str(r["id"] or ""), str(r["description"] or ""))
                             for _, r in m_choices.fillna("").iterrows()]
                    questions[ev.MISCONCEPTION_QID] = build_choice_question(m_instr, pairs)
            except ValueError as e:
                st.warning(str(e))
            else:
                runners = [
                    (cfg.provider, cfg.model, lambda s, q, c=cfg, **kw: run_one(s, q, c))
                    for cfg in configs
                ]
                bar = st.progress(0.0, text=f"Evaluating 0 / {n * len(runners)}")
                results = ev.evaluate_models(
                    dataset, questions, runners, threshold=threshold,
                    on_progress=lambda i, total: bar.progress(i / total, text=f"Evaluating {i} / {total}"),
                )
                bar.empty()
                labels = list(questions[ev.MISCONCEPTION_QID]["criteria"]) if do_misc else None
                st.session_state.eval = {
                    "results": results, "labels": labels,
                    "models": [f"{c.provider}:{c.model}" for c in configs],
                    "questions": questions, "threshold": threshold,
                    "at": datetime.now().isoformat(timespec="seconds"),
                }

    evr = st.session_state.get("eval")
    if evr:
        results: pd.DataFrame = evr["results"]
        st.divider()
        st.markdown(f"#### Results · {evr['at']}")

        summary = ev.model_comparison(results, evr["labels"])
        if not summary.empty:
            st.markdown("**By model**")
            show = summary.copy()
            st.dataframe(
                show,
                width="stretch",
                hide_index=True,
                column_config={
                    "accuracy": st.column_config.ProgressColumn("accuracy", min_value=0, max_value=1, format="percent"),
                    "f1": st.column_config.ProgressColumn("f1", min_value=0, max_value=1, format="percent"),
                    "macro_f1": st.column_config.ProgressColumn("macro_f1", min_value=0, max_value=1, format="percent"),
                    "avg_latency_ms": st.column_config.NumberColumn("avg latency (ms)", format="%.0f"),
                    "median_latency_ms": st.column_config.NumberColumn("median latency (ms)", format="%.0f"),
                    "p95_latency_ms": st.column_config.NumberColumn("p95 latency (ms)", format="%.0f"),
                    "avg_input_tokens": st.column_config.NumberColumn("avg in tok", format="%.0f"),
                    "avg_output_tokens": st.column_config.NumberColumn("avg out tok", format="%.0f"),
                    "avg_total_tokens": st.column_config.NumberColumn("avg total tok", format="%.0f"),
                },
            )

        if "provider" in results.columns:
            for (provider, model), g in results.groupby(["provider", "model"], dropna=False):
                with st.expander(f"{provider} · {model}", expanded=len(summary) == 1):
                    render_eval_group(g, evr["labels"], None)
        else:
            render_eval_group(results, evr["labels"], None)

        with st.expander("Per-row results", expanded=False):
            st.dataframe(results, width="stretch", hide_index=True)

        export = ev.export_table(results)
        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button("Download CSV", export.to_csv(index=False), "clef_evaluation.csv", "text/csv",
                           width="stretch")
        lat = ev.latency_stats(results)
        cm_all = ev.correctness_metrics(results)
        mm_all = ev.misconception_metrics(results, evr["labels"])
        payload = {
            "models": evr.get("models"), "evaluated_at": evr["at"], "threshold": evr["threshold"],
            "questions": evr["questions"], "system": lat,
            "by_model": json.loads(summary.to_json(orient="records")) if not summary.empty else [],
            "correctness": cm_all,
            "misconception": None if not mm_all else {
                "n": mm_all["n"], "accuracy": mm_all["accuracy"], "macro_f1": mm_all["macro_f1"],
                "per_class": mm_all["per_class"].to_dict(orient="records"),
                "confusion": mm_all["confusion"].to_dict(),
            },
            "rows": json.loads(export.to_json(orient="records")),
        }
        d2.download_button("Download JSON", json.dumps(payload, indent=2, default=str),
                           "clef_evaluation.json", "application/json", width="stretch")
