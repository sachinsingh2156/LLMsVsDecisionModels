# Clef Decision Lab

**Decision models vs. generative LLMs, cloud vs. local — for grading student mathematics.**

This repository holds a Streamlit dashboard and a reproducible benchmark that compare five
systems on two public math-education datasets:

| | Decision model (typed answers + probabilities) | Generative LLM (writes JSON) |
|---|---|---|
| **Cloud API** | TypeSafe **Jev** (`jev-latest`) | Google **Gemini 3.8 Flash** |
| **Local (Ollama)** | Cloudflare **Clef** 27.5B, **Clef-flash** 9.5B | **Qwen 2.5 7B** |

Each system answers the same two questions per item: *is the student's answer correct?*
(`noul`, a probability) and *which misconception explains it?* (`choice` over the MAP taxonomy).

## Key results

| Model | MAP-500 correctness | MathDial true-negative rate | Labelled misconception acc. | Mean latency (MAP) | API cost, 1,149 items |
|---|---:|---:|---:|---:|---:|
| Gemini 3.8 Flash | **99.8%** | **95.7%** | **53.7%** | 9.0 s | ₹350.76 |
| TypeSafe Jev | 98.0% | 64.5% | 46.3% | **0.37 s** | ₹4.18 |
| Clef 27.5B | 89.2% | 85.3% | 32.1% (26-way) | 8.9 s | ₹0 (local) |
| Clef-flash 9.5B | 84.4% | 34.8% | 24.6% (26-way) | 2.6 s | ₹0 (local) |
| Qwen 2.5 7B | 68.5% | 30.1%* | 12.7% | 12.2 s | ₹0 (local) |

\* Qwen returned no parseable JSON for 355 of 599 MathDial dialogues; its rate covers only the 166 unique problems it answered.

**Takeaways**

- **Short multiple-choice answers:** a decision model (Jev) is almost as accurate as Gemini, about 24× faster and about 84× cheaper.
- **Long worked solutions:** Gemini is clearly best. Clef 27.5B is the only local model that rejects most incorrect write-ups.
- **Local models:** the decision model beats the generative model of similar size (Clef-flash 9.5B > Qwen 2.5 7B), and size matters (Clef 27.5B ≫ Clef-flash 9.5B on MathDial).
- **Naming the misconception** is unsolved for every system (best: 53.7%).

Full write-up: [`eval_results/public_slices/ANALYSIS_REPORT.md`](eval_results/public_slices/ANALYSIS_REPORT.md).
Auto-generated metrics: [`REPORT.md`](eval_results/public_slices/REPORT.md) and [`comparison.csv`](eval_results/public_slices/comparison.csv).

## Repository layout

```
clef-decision-lab/
├── app.py                     # Streamlit dashboard (Playground + Evaluation tabs)
├── clef_client.py             # Clef (Ollama) + TypeSafe Jev  → POST /v1/systemone
├── llm_client.py              # Gemini + Ollama Qwen JSON adapters
├── providers.py               # run_selected(): run one decision on several providers
├── models.py                  # question builders, answer interpretation, presets
├── evaluator.py               # batch evaluation, metrics, export
├── scripts/
│   ├── prepare_eval_slices.py # build MAP-50 / MAP-500 / MathDial slices from raw data
│   ├── run_public_eval.py     # resumable benchmark runner (writes eval_results/)
│   └── finalize_eval_report.py# extra tables: per-category, errors, cost totals
├── eval_data/slices/          # the exact items and questions used in the benchmark
├── eval_results/public_slices/
│   ├── ANALYSIS_REPORT.md     # narrative analysis of the results
│   ├── REPORT.md, comparison.csv
│   ├── {slice}/{run_key}/     # per_row.csv/.json, summary.json, confusion matrices
│   ├── checkpoints/           # raw model responses (JSONL)
│   └── analysis/              # error lists, per-category and per-class breakdowns
├── sample_data/               # small hand-written CSVs for the Evaluation tab
├── tests/                     # pytest; HTTP is faked, no API keys needed
├── .env.example
└── requirements.txt
```

`run_key` values: `typesafe`, `gemini`, `clef_9b`, `clef_27b`, `qwen`.

## Setup

Requirements: Python 3.10+, [Ollama](https://ollama.com) 0.40.0+ for the local models,
and optional API keys for TypeSafe and Gemini.

```bash
git clone <this-repo-url> clef-decision-lab
cd clef-decision-lab
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add TYPESAFE_API_KEY / GEMINI_API_KEY; never commit .env

ollama pull clef              # Clef 27.5B
ollama pull clef-flash        # Clef-flash 9.5B
ollama pull qwen2.5:7b        # optional generative baseline
```

## Run the dashboard

```bash
streamlit run app.py          # http://localhost:8501
```

- **Playground:** define any `noul` or `choice` decision, run it on the selected models, and compare decisions, probabilities, latency and tokens side by side.
- **Evaluation:** upload a CSV (`id,question,student_answer,ground_truth_correct,ground_truth_misconception`) or use `sample_data/fractions_sample.csv` to get accuracy, F1, confusion matrices and CSV/JSON exports.

## Reproduce the benchmark

```bash
# 1. Download the raw datasets (not committed)
hf download kaggle-map/data train.csv --repo-type dataset --local-dir eval_data/raw/map
hf download eth-nlped/mathdial test.jsonl --repo-type dataset --local-dir eval_data/raw/mathdial

# 2. Build the slices (stratified by MAP Category, seed 42)
python scripts/prepare_eval_slices.py

# 3. Run any subset of models (resumable; re-running skips finished rows)
python scripts/run_public_eval.py --providers typesafe,gemini,clef_9b,clef_27b,qwen

# 4. Build the extra analysis tables
python scripts/finalize_eval_report.py
```

The committed `eval_data/slices/` and `eval_results/` let you inspect every prediction
without re-running anything.

## Notes on the providers

| Provider | Endpoint | Default model |
|---|---|---|
| Clef / Clef-flash | `POST http://localhost:11434/v1/systemone` | `clef`, `clef-flash` |
| TypeSafe Jev | `POST https://api.typesafe.ai/v1/systemone` | `jev-latest` |
| Gemini | `POST …/models/{id}:generateContent` (JSON schema) | `gemini-3.8-flash` |
| Qwen | `POST http://localhost:11434/api/chat` with `format: json` | `qwen2.5:7b` |

- Clef has no chat capability, so `ollama run clef` does not work; it is called only through `/v1/systemone`.
- `noul` ≈ P(TRUE); the default decision threshold is 0.5. `choice` returns a label, a probability distribution and a separate `confidence`.
- Ollama limits Clef `choice` questions to 26 candidates, so Clef uses the 24 most frequent MAP labels plus `none` and `other` (96.8% of labelled rows). Quote Clef misconception scores as 26-way.
- Gemini and Qwen probabilities are self-reported, not calibrated like Clef/Jev.

## Tests

```bash
python -m pytest -q
```

## Data and attribution

- **MAP — Charting Student Math Misunderstandings** (Kaggle competition; Hugging Face mirror [`kaggle-map/data`](https://huggingface.co/datasets/kaggle-map/data), MIT licence). See also S. Crossley *et al.*, "Automated approaches for scoring math misunderstandings in student self-explanations," AIME-Con 2026.
- **MathDial** — J. Macina *et al.*, "MathDial: A dialogue tutoring dataset with rich pedagogical properties grounded in math reasoning problems," *Findings of EMNLP 2023* ([`eth-nlped/mathdial`](https://huggingface.co/datasets/eth-nlped/mathdial), CC BY 4.0).

`eval_data/slices/` and `eval_results/` contain items derived from these datasets and are
redistributed under their original licences.

## Troubleshooting

| Message | Fix |
|---|---|
| *Could not connect to Ollama* | Start Ollama (`ollama serve`) and check the URL in the sidebar. |
| *Model 'clef' was not found* | `ollama pull clef` |
| *is not a decision model* | Use `clef` / `clef-flash` for Clef, not a chat model. |
| *TypeSafe / Gemini API key* | Paste the key in the sidebar or set it in `.env`. |
| *did not respond within 300 s* | The first Clef call loads the weights; retry once it is warm. |
