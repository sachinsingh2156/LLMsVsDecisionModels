# Decision-model evaluation report

**Five systems on MAP and MathDial:** TypeSafe Jev, Gemini 3.8 Flash, Clef 9.5B, Clef 27.5B, Qwen 2.5 7B

Date of run: 26–27 October 2026 (UTC)  
Lab: Clef Decision Lab (`clef-decision-lab`)  
Primary artifacts: `eval_results/public_slices/`

---

## 1. Executive summary

The same three cheap public slices were scored for every model in the Decision Lab:

- MAP stratified 50 (pipeline)
- MAP stratified 500
- MathDial official test (599 dialogues / 394 unique problems)

**Ranking on MAP-500 correctness (is the selected MC option right?)**

| Rank | Model | Accuracy | F1 | Latency |
|---:|---|---:|---:|---:|
| 1 | Gemini 3.8 Flash | **99.8%** | 0.998 | 9.0 s |
| 2 | TypeSafe Jev | **98.0%** | 0.982 | **0.37 s** |
| 3 | Clef 27.5B (`clef:latest`) | 89.2% | 0.894 | 8.9 s |
| 4 | Clef 9.5B (`clef-flash`) | 84.4% | 0.852 | 2.6 s |
| 5 | Qwen 2.5 7B | 68.5% | 0.763 | 12.2 s |

**Ranking on MAP-500 labelled misconceptions (n=134)**

Gemini 53.7% → Jev 46.3% → Clef 27B 32.1% → Clef 9B 24.6% → Qwen 12.7%.  
Clef was given a *smaller* 26-class taxonomy (Ollama cap), so its lower accuracy is not an artifact of a harder label set.

**Ranking on MathDial unique true-negative rate (gold is always FALSE)**

Gemini 95.7% → Clef 27B 85.3% → Jev 64.5% → Clef 9B 34.8% → Qwen 30.1% (on the 166 unique rows it could parse).  
Clef 9B and Qwen are worse than a constant-FALSE baseline would be, because they call too many wrong solutions “correct.”

**Efficiency.** Jev is two orders of magnitude cheaper than Gemini (₹4.18 vs ₹350.76 for 1,149 rows) and ~24× faster on MAP. Local Clef/Qwen have ₹0 API cost. Clef 9B is the fast local decision model (~2.6 s); Clef 27B matches Gemini’s latency on MAP (~9 s) with lower accuracy; Qwen is slowest and least reliable (355/599 MathDial rows failed JSON).

**One-line product reading.** Use **Jev** for MAP-style option checking. Use **Gemini** (or **Clef 27B** if you must stay local) to read long wrong write-ups. Do not use **Clef 9B** or **Qwen 7B** as a MathDial-style solution checker. No model names MAP misconceptions well enough to ship as a 35-way classifier.

---

## 2. What was run

### 2.1 Models

| Run key | System | Model id | Where | Notes |
|---|---|---|---|---|
| `typesafe` | TypeSafe Jev | `jev-latest` | API | Native System One. Input-only billing. |
| `gemini` | Gemini | `gemini-3.8-flash` | API | JSON schema + thinking **low**. |
| `clef_9b` | Clef Flash | `clef-flash:latest` | local Ollama | 9.5B, mxfp8, decision capability. |
| `clef_27b` | Clef | `clef:latest` | local Ollama | 27.5B, nvfp4, decision capability. |
| `qwen` | Qwen | `qwen2.5:7b` | local Ollama | Generative; same JSON prompt as Gemini. |

Clef and Jev return `noul` / `choice` natively. Gemini and Qwen are prompted to emit the same JSON. Clef `output_tokens` is always 0 (Ollama decision convention). Gemini output tokens include thinking.

### 2.2 Datasets and tasks

Unchanged from the Jev/Gemini study:

- **MAP** (Hugging Face `kaggle-map/data` train). Stratified by `Category`, seed 42. Correctness = MC option is `True_*`. Misconception = MAP taxonomy, or `none`.
- **MathDial** test. Gold correctness is FALSE for every row (`student_incorrect_solution`). 599 dialogues, 394 unique `qid`s.

MAP-50 is nested in MAP-500 under this seed (pipeline check, not an independent draw).

### 2.3 Clef 26-class cap

Ollama rejects Clef `choice` questions with more than 26 candidates (`HTTP 400: criteria must contain 2–26 candidates`). Jev/Gemini/Qwen used 35 MAP labels + `none` (36-way). Clef used the **24 most common MAP labels + `none` + `other`**.

Those 24 labels cover **96.8%** of labelled MAP train rows. Rare gold codes (`Division`, `Definition`, `Interior`, …) were remapped to `other` only for Clef scoring. On MAP-500 labelled rows, gold `other` is 3.0% (4/134). Clef never predicted `other`. The collapse does not explain Clef’s lower taxonomy accuracy.

MathDial is `noul` only, so Clef, Jev, Gemini, and Qwen used the same question.

### 2.4 Completeness

| Model | Rows requested | Failed after retry |
|---|---:|---:|
| Jev | 1,149 | 0 |
| Gemini | 1,149 | 0 |
| Clef 9B | 1,149 | 0 |
| Clef 27B | 1,149 | 0 |
| Qwen | 1,149 | **357** (2 MAP-500 + 355 MathDial) |

Qwen MathDial failures are `Malformed LLM response: no answer for 'correct'` and Ollama `token repeat limit reached`. Retries did not recover them. MathDial Qwen metrics below are on the **scored subset** (244/599 dialogues, 166/394 unique qids).

---

## 3. Headline tables

### 3.1 MAP-50 (pipeline)

| Model | Acc | P | R | F1 | Labelled misc. (n=14) | Avg ms | ₹ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gemini | 100% | 1.00 | 1.00 | 1.00 | 57.1% | 9,402 | 21.45 |
| Jev | 98.0% | 1.00 | 0.96 | 0.98 | 42.9% | 394 | 0.27 |
| Clef 9B | 90.0% | 1.00 | 0.81 | 0.90 | 42.9% | 2,640 | 0 |
| Clef 27B | 90.0% | 1.00 | 0.81 | 0.90 | 42.9% | 9,141 | 0 |
| Qwen | 68.0% | 0.64 | 0.93 | 0.76 | 7.1% | 12,219 | 0 |

Clef 9B and 27B posted **identical** MAP-50 correctness confusion (TP 22, TN 23, FP 0, FN 5). Both are conservative: they never call a wrong option correct, and they miss five true options. Qwen is the opposite: high recall, floods FALSE rows with TRUE (14 FP).

### 3.2 MAP-500 (main MAP result)

| Model | Acc | TP / TN / FP / FN | F1 | Misc all | Misc labelled | Macro-F1 labelled | Avg ms |
|---|---:|---|---:|---:|---:|---:|---:|
| Gemini | **99.8%** | 279 / 220 / 1 / 0 | 0.998 | 73.0% | **53.7%** | 0.408 | 9,044 |
| Jev | **98.0%** | 277 / 213 / 8 / 2 | 0.982 | 64.4% | 46.3% | 0.290 | **371** |
| Clef 27B | 89.2% | 227 / 219 / **2** / 52 | 0.894 | 45.6% | 32.1% | 0.256 | 8,930 |
| Clef 9B | 84.4% | 225 / 197 / 24 / 54 | 0.852 | 45.4% | 24.6% | 0.195 | 2,561 |
| Qwen | 68.5% | 253 / 88 / 133 / 24 | 0.763 | 43.8% | 12.7% | 0.093 | 12,188 |

Clef 9B vs 27B correctness agreement is 84.4%. Scaling Clef from 9.5B to 27.5B buys **4.8 points** of MAP accuracy, almost all of it from killing false positives (24 → 2) while false negatives stay ~52–54. The 27B model is a stricter, better-calibrated version of the 9B behaviour, not a different decision style.

Qwen on MAP-500: mean `noul` when the option is **wrong** is still 0.57 — it does not separate the classes.

### 3.3 MathDial test

Gold TRUE count = 0, so F1 of the TRUE class is 0 for every model. Accuracy = true-negative rate.

| Model | Dialogues TNR | Unique TNR (n=394, Qwen n=166) | Predicted TRUE (unique) | Avg ms | Failures |
|---|---:|---:|---:|---:|---:|
| Gemini | **96.3%** | **95.7%** | 4.3% | 5,572 | 0 |
| Clef 27B | 86.3% | 85.3% | 14.7% | 2,050 | 0 |
| Jev | 63.3% | 64.5% | 35.5% | **364** | 0 |
| Clef 9B | 35.9% | 34.8% | 65.2% | 561 | 0 |
| Qwen | 27.9% of 244 scored | 30.1% of 166 scored | 69.9% of scored | 6,820 | **355** |

Clef 27B is the only *local* model that is actually useful on MathDial. Clef 9B is anti-useful: majority vote TRUE on gold-FALSE data. Qwen failed to return parseable JSON on most of the split; when it answered, it was also majority TRUE.

---

## 4. MAP correctness in detail

### 4.1 Error style

| Model | Typical miss | `noul` picture |
|---|---|---|
| Gemini | One FP (`noul`=0.95) on a garbled “greatest number” explanation. No FN. | Peaked: almost all mass at 0 or 1. |
| Jev | 8 FP + 2 FN, all near the 0.5 fence (`noul` 0.35–0.66). | Calibrated. Mean `noul` 0.95 on true options, 0.14 on false. |
| Clef 27B | Almost no FP (2), many FN (52). | Mean 0.72 on true, 0.15 on false. Conservative threshold behaviour. |
| Clef 9B | Same FN pile (54) plus 24 FP. | Mean 0.69 on true, 0.25 on false. Softer than 27B. |
| Qwen | 133 FP. | Mean 0.89 on true, **0.57 on false**. Not a detector. |

Clef is biased toward FALSE on MAP. That is the opposite of Qwen (biased toward TRUE) and different from Jev (near-calibrated). If a teacher would rather skip a correct student than flag a wrong one, Clef 27B’s 2 FP / 52 FN may be acceptable. If the job is “don’t miss wrong work,” Gemini and Jev dominate.

### 4.2 MAP-500 by Category (Clef / Qwen)

**Clef 27B** keeps precision 1.0 on every `False_*` class except a sliver of `False_Neither` (the 2 FP). Recall on `True_Correct` is the hole (~81% of true options found, matching the 52 FN).

**Clef 9B** leaks FPs into `False_Misconception` / `False_Neither` and still misses the same true-option mass.

**Qwen** is noisy on every `False_*` bin (pred-TRUE rate 50%+ on wrong options).

Full per-category CSVs: `analysis/map_500__{clef_9b,clef_27b,qwen}__by_category.csv`.

### 4.3 Calibration note for Clef

Clef `noul` is a real probability, unlike Gemini’s peaked self-report. 27B’s false-option mean of 0.15 vs 9B’s 0.25 is the scaling gain: the larger model pushes more wrong options below 0.5 without lifting true options enough to cut the FN pile. A lower threshold (e.g. 0.4) would trade FP for FN; it was not swept in this run.

---

## 5. MAP misconceptions in detail

### 5.1 The hard task is still unsolved

Even Gemini, the best system, names the labelled MAP code only **53.7%** of the time. Local models are not close.

| Model | Label set | Labelled acc | Dump class |
|---|---|---:|---|
| Gemini | 36-way | 53.7% | `Incomplete` |
| Jev | 36-way | 46.3% | `Incomplete` / `none` |
| Clef 27B | 26-way | 32.1% | `Wrong_Fraction`, `Denominator_only_change` |
| Clef 9B | 26-way | 24.6% | **`Wrong_Fraction` (54 of 134 labelled preds)** |
| Qwen | 36-way | 12.7% | near-random |

Clef 9B’s `Wrong_Fraction` magnet is a real failure mode: it is not reading the taxonomy, it is collapsing into one frequent fraction-looking bucket.

### 5.2 Fairness of the 26-way Clef set

If the 26-way set were easier, Clef should beat Jev on labelled accuracy. It does not (32% vs 46%). The cap is a limitation of the Clef API, not an advantage. Do not quote Clef 26-way accuracy as if it were 36-way.

---

## 6. MathDial in detail

### 6.1 Architecture split, now with local models

Short MAP options vs long derivations:

| | Short MC option (MAP-500 acc) | Long wrong write-up (unique TNR) |
|---|---:|---:|
| Gemini | 99.8% | 95.7% |
| Jev | 98.0% | 64.5% |
| Clef 27B | 89.2% | **85.3%** |
| Clef 9B | 84.4% | 34.8% |
| Qwen | 68.5% | 30.1% scored |

**Clef 27B is the interesting local result.** It is worse than Jev on MAP option-checking, but **much better than Jev on MathDial** (85% vs 65% unique TNR), and faster than Gemini (2.1 s vs 5.6 s) with no API bill. Scaling Clef from 9.5B to 27.5B is what creates that MathDial skill: 9B predicted TRUE on 65% of unique wrong solutions; 27B only 15%.

Jev remains the MAP specialist. Clef 27B is the local “this write-up is wrong” specialist. Gemini wins both, at API cost.

### 6.2 Qwen MathDial is not a scored bake-off

355/599 dialogue rows never produced a `correct` answer. Unique-level: 228/394 failed. Ollama aborted many generations with `token repeat limit reached`. Among the rows that parsed, predicted-TRUE was 70%. Treat Qwen MathDial as **an engineering failure of the JSON adapter on 7B**, not as a 28% TNR competitor.

---

## 7. Latency, tokens, cost

### 7.1 MAP-500 latency

| Model | Avg | Median | p95 | vs Jev |
|---|---:|---:|---:|---:|
| Jev | 371 ms | 371 | 399 | 1× |
| Clef 9B | 2,561 ms | 2,569 | 2,710 | 7× |
| Clef 27B | 8,930 ms | 8,938 | 9,452 | 24× |
| Gemini | 9,044 ms | 8,626 | 12,615 | 24× |
| Qwen | 12,188 ms | 11,407 | 16,940 | 33× |

Clef is stable (p95 ≈ mean). Gemini and Qwen have a long tail. Clef 27B and Gemini are tied on MAP wall-clock; Clef 27B is faster on MathDial because there is no 36-way choice table.

### 7.2 Tokens (MAP-500)

| Model | Input sum | Output sum | Note |
|---|---:|---:|---|
| Jev | 748,017 | 181,911 | Choice table inflates input; small JSON out |
| Gemini | 673,834 | 507,766 | Thinking + reasoning |
| Clef 9B / 27B | 582,487 | **0** | Decision model; Ollama reports no decode tokens |
| Qwen | 638,527 | 235,417 | Same long-reasoning prompt as Gemini, no thinking |

### 7.3 Cost

| Model | API ₹ for 1,149 rows |
|---|---:|
| Jev | 4.18 |
| Gemini | 350.76 |
| Clef 9B / 27B / Qwen | **0** (local GPU / electricity not priced) |

Grand API spend remains **₹354.94**. Local runs add wall-clock (~5 h Clef 9B+27B + ~5 h Qwen on this machine) but no token invoice.

---

## 8. Limitations

All previous caveats still apply (stratified MAP sample, MC-option vs explanation quality, MathDial has no positive class, Gemini thinking was `low`, images are text placeholders). New ones:

1. **Clef misconception scores are 26-way, not 36-way.** Quote them with that label.
2. **Qwen MathDial is incomplete.** 59% of dialogue rows failed. Do not rank Qwen on TNR against models that scored 599/599.
3. **Clef and Qwen share the local Ollama GPU.** They were run sequentially. Streamlit on port 8502 was also up; contention was not measured.
4. **No threshold sweep** on Clef `noul`. Reported accuracy uses the lab default 0.5.
5. **Qwen 2.5 7B is Q4_K_M.** A larger Qwen would likely JSON-fail less and maybe detect errors better; it was not run.
6. Clef 9B and 27B MAP-50 metrics are identical; that is a small-n coincidence on the pipeline slice, not proof the models are the same (MAP-500 and MathDial separate them).

---

## 9. Conclusions

1. **Jev vs Gemini is unchanged:** Jev is the cheap, calibrated MAP option-checker (98.0%, 371 ms, ₹2.73/500). Gemini is slightly more accurate on MAP (99.8%) and far better on long wrong solutions (96% TNR) at ~84× API cost.
2. **Clef 27B is the best local model in this study**, but it is not a Jev clone. It is conservative on MAP (89%, almost no FP, many FN) and surprisingly strong on MathDial (85% unique TNR), beating Jev there. Latency matches Gemini on MAP (~9 s).
3. **Clef 9B is a fast local MAP filter** (84% MAP-500, 2.6 s) and **not a write-up checker** (35% MathDial TNR). Scaling to 27B is required for MathDial.
4. **Qwen 2.5 7B, with the lab’s long-reasoning JSON prompt, is not competitive:** 68.5% MAP-500, 13% labelled misconceptions, and a majority of MathDial rows that never parse. A shorter classification prompt might change this; that was out of scope for a fair schema-matched comparison.
5. **Misconception IDs remain unsolved for every stack**, local or API.

**Practical routing for the Decision Lab**

- MAP / short selected-answer: **Jev**, fallback Gemini.
- Long student write-up, cloud OK: **Gemini**.
- Long student write-up, must be local: **Clef 27B**, not Clef 9B, not Qwen 7B.
- Named MAP misconception: none of these models as a 35-way `choice`; treat as research, not a product feature.

---

## 10. Numbers that are safe to quote

- MAP-500 correctness: Gemini **99.8%**, Jev **98.0%**, Clef 27B **89.2%**, Clef 9B **84.4%**, Qwen **68.5%** (Qwen 2 failed requests).
- MAP-500 labelled misconception: Gemini **53.7%**, Jev **46.3%**, Clef 27B **32.1% (26-way)**, Clef 9B **24.6% (26-way)**, Qwen **12.7%**.
- MathDial unique TNR: Gemini **95.7%**, Clef 27B **85.3%**, Jev **64.5%**, Clef 9B **34.8%**, Qwen **30.1% on 166/394 parsed qids**.
- MAP-500 latency: Jev **0.37 s**, Clef 9B **2.6 s**, Clef 27B **8.9 s**, Gemini **9.0 s**, Qwen **12.2 s**.
- API cost for the original Jev+Gemini 1,149-row pair: **₹354.94**. Local Clef/Qwen: **₹0**.

---

## 11. Artifact index

Under `clef-decision-lab/eval_results/public_slices/`:

| Path | Contents |
|---|---|
| `ANALYSIS_REPORT.md` | This narrative |
| `REPORT.md` | Auto-generated metric dump |
| `comparison.csv` | One row per (slice, run_key) |
| `{slice}/{run_key}/per_row.csv` | GT, prediction, `noul`, tokens, latency |
| `{slice}/{run_key}/summary.json` | Metrics + cost |
| `checkpoints/{slice}__{run_key}.jsonl` | Raw API / Ollama answers |
| `analysis/{slice}__{run_key}__errors.csv` | Every correctness / labelled-misc mistake |
| `analysis/{slice}__{run_key}__by_category.csv` | MAP Category breakdown |
| `eval_data/slices/map_questions_clef.json` | 26-class Clef taxonomy |
| `eval_data/slices/clef_misconception_ids.json` | The 24+none+other id list |

`run_key` values: `typesafe`, `gemini`, `clef_9b`, `clef_27b`, `qwen`.
