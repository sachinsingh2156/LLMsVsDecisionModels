# Public-slice evaluation report

Generated: 2026-10-27T13:57:58Z

This file is rewritten after every completed (slice, provider) run. Use it plus the
`per_row.csv` / `per_row.json` / checkpoint JSONL files as the source for a written report.

## What was run

- Datasets: MAP train (stratified 50 and 500) and MathDial **test** (599).
- Models: TypeSafe Jev (`jev-latest`); Gemini (`gemini-3.8-flash`, thinkingLevel=low); local Clef 9.5B (`clef-flash:latest`); local Clef 27.5B (`clef:latest`); local Qwen (`qwen2.5:7b`).
- MAP tasks: MC-answer correctness (`noul`) + MAP misconception taxonomy (`choice`). Jev/Gemini/Qwen use 35 labels + `none`. Clef is capped at 26 choice candidates by Ollama, so it uses the 24 most common MAP labels + `none` + `other`.
- MathDial task: correctness of the student's initial incorrect solution (`noul` only). Gold label is FALSE for every row by construction.
- Sampling: MAP is stratified by `Category` with seed 42. Sequential head(50)/head(500) would be 100% `True_*` and is not used.
- FX used for ₹ estimates: 87 INR / USD. Jev $0.042/MTok input-only. Gemini 3.8 Flash intro $0.75/M in + $3.75/M out.

## Headline comparison

| Slice | Provider | Model | N | Failed | Correct acc | Correct F1 | Misc. acc (all) | Misc. macro-F1 (labelled) | Avg latency ms | Input tok | Output tok | Est. ₹ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| map_50 | typesafe | jev-latest | 50 | 0 | 98.0% | 98.1% | 58.0% | 36.4% | 394 | 74587 | 18193 | 0.27 |
| map_50 | gemini | gemini-3.8-flash | 50 | 0 | 100.0% | 100.0% | 74.0% | 45.5% | 9402 | 67177 | 52309 | 21.45 |
| map_50 | clef_9b | clef-flash:latest | 50 | 0 | 90.0% | 89.8% | 40.0% | 34.5% | 2640 | 58028 | 0 | 0.00 |
| map_50 | clef_27b | clef:latest | 50 | 0 | 90.0% | 89.8% | 42.0% | 36.4% | 9141 | 58028 | 0 | 0.00 |
| map_50 | qwen | qwen2.5:7b | 50 | 0 | 68.0% | 75.8% | 40.0% | 6.1% | 12219 | 63887 | 23571 | 0.00 |
| map_500 | typesafe | jev-latest | 500 | 0 | 98.0% | 98.2% | 64.4% | 29.0% | 371 | 748017 | 181911 | 2.73 |
| map_500 | gemini | gemini-3.8-flash | 500 | 0 | 99.8% | 99.8% | 73.0% | 40.8% | 9044 | 673834 | 507766 | 209.63 |
| map_500 | clef_9b | clef-flash:latest | 500 | 0 | 84.4% | 85.2% | 45.4% | 19.5% | 2561 | 582487 | 0 | 0.00 |
| map_500 | clef_27b | clef:latest | 500 | 0 | 89.2% | 89.4% | 45.6% | 25.6% | 8930 | 582487 | 0 | 0.00 |
| map_500 | qwen | qwen2.5:7b | 500 | 2 | 68.5% | 76.3% | 43.8% | 9.3% | 12188 | 638527 | 235417 | 0.00 |
| mathdial_test | typesafe | jev-latest | 599 | 0 | 63.3% | 0.0% | n/a | n/a | 364 | 321537 | 11980 | 1.17 |
| mathdial_test | gemini | gemini-3.8-flash | 599 | 0 | 96.3% | 0.0% | n/a | n/a | 5572 | 347046 | 297432 | 119.68 |
| mathdial_test | clef_9b | clef-flash:latest | 599 | 0 | 35.9% | 0.0% | n/a | n/a | 561 | 219359 | 0 | 0.00 |
| mathdial_test | clef_27b | clef:latest | 599 | 0 | 86.3% | 0.0% | n/a | n/a | 2050 | 219359 | 0 | 0.00 |
| mathdial_test | qwen | qwen2.5:7b | 599 | 355 | 27.9% | 0.0% | n/a | n/a | 6820 | 139411 | 74719 | 0.00 |

## Per-run detail

### map_50 / typesafe (`jev-latest`)

- Finished: 2026-10-26T19:51:59Z
- Rows: 50; failed requests: 0
- Latency ms: avg 394.3, median 380.0, p95 477.5
- Tokens: input 74587, output 18193, cached 0, avg total 1855.6
- Estimated cost: $0.0031 (₹0.27). Jev bills input tokens only at $0.042/MTok

Correctness (TRUE = selected MC answer is right; MathDial TRUE would mean the written solution is right):

- n=50 accuracy=0.9800 precision=1.0000 recall=0.9630 F1=0.9811
- TP=26 TN=23 FP=0 FN=1
- Model predicted TRUE on 52.0% of scored rows; mean noul P(true)=0.544

Misconception, all classes including `none`: n=50 accuracy=0.5800 macro-F1=0.3972

Misconception, labelled rows only (MAP `*_Misconception`): n=14 accuracy=0.4286 macro-F1=0.3636

Has-misconception vs `none` (binary, positive = not none):
- n=50 accuracy=0.7200 precision=0.5000 recall=0.9286 F1=0.6500 TP=13 TN=23 FP=13 FN=1

Artifacts:
- `map_50/typesafe/per_row.csv`
- `map_50/typesafe/per_row.json`
- `map_50/typesafe/summary.json`
- `checkpoints/map_50__typesafe.jsonl` (raw answers, usage, errors)

### map_50 / gemini (`gemini-3.8-flash`)

- Finished: 2026-10-26T20:08:48Z
- Rows: 50; failed requests: 0
- Latency ms: avg 9401.8, median 8933.3, p95 13097.5
- Tokens: input 67177, output 52309, cached 0, avg total 2389.7
- Estimated cost: $0.2465 (₹21.45). Gemini 3.8 Flash intro pricing $0.75/M in + $3.75/M out; output includes thinking tokens

Correctness (TRUE = selected MC answer is right; MathDial TRUE would mean the written solution is right):

- n=50 accuracy=1.0000 precision=1.0000 recall=1.0000 F1=1.0000
- TP=27 TN=23 FP=0 FN=0
- Model predicted TRUE on 54.0% of scored rows; mean noul P(true)=0.540

Misconception, all classes including `none`: n=50 accuracy=0.7400 macro-F1=0.4910

Misconception, labelled rows only (MAP `*_Misconception`): n=14 accuracy=0.5714 macro-F1=0.4545

Has-misconception vs `none` (binary, positive = not none):
- n=50 accuracy=0.8600 precision=0.6667 recall=1.0000 F1=0.8000 TP=14 TN=29 FP=7 FN=0

Artifacts:
- `map_50/gemini/per_row.csv`
- `map_50/gemini/per_row.json`
- `map_50/gemini/summary.json`
- `checkpoints/map_50__gemini.jsonl` (raw answers, usage, errors)

### map_50 / clef_9b (`clef-flash:latest`)

- Finished: 2026-10-27T08:03:53Z
- Rows: 50; failed requests: 0
- Latency ms: avg 2639.7, median 2633.4, p95 2822.9
- Tokens: input 58028, output 0, cached 2744, avg total 1160.6
- Estimated cost: $0.0000 (₹0.00). Local Ollama model: no API bill. Electricity / GPU time not priced.

Correctness (TRUE = selected MC answer is right; MathDial TRUE would mean the written solution is right):

- n=50 accuracy=0.9000 precision=1.0000 recall=0.8148 F1=0.8980
- TP=22 TN=23 FP=0 FN=5
- Model predicted TRUE on 44.0% of scored rows; mean noul P(true)=0.444

Misconception, all classes including `none`: n=50 accuracy=0.4000 macro-F1=0.3165

Misconception, labelled rows only (MAP `*_Misconception`): n=14 accuracy=0.4286 macro-F1=0.3455

Has-misconception vs `none` (binary, positive = not none):
- n=50 accuracy=0.5600 precision=0.3889 recall=1.0000 F1=0.5600 TP=14 TN=14 FP=22 FN=0

Artifacts:
- `map_50/clef_9b/per_row.csv`
- `map_50/clef_9b/per_row.json`
- `map_50/clef_9b/summary.json`
- `checkpoints/map_50__clef_9b.jsonl` (raw answers, usage, errors)

### map_50 / clef_27b (`clef:latest`)

- Finished: 2026-10-27T08:38:47Z
- Rows: 50; failed requests: 0
- Latency ms: avg 9141.3, median 9005.0, p95 9375.9
- Tokens: input 58028, output 0, cached 2568, avg total 1160.6
- Estimated cost: $0.0000 (₹0.00). Local Ollama model: no API bill. Electricity / GPU time not priced.

Correctness (TRUE = selected MC answer is right; MathDial TRUE would mean the written solution is right):

- n=50 accuracy=0.9000 precision=1.0000 recall=0.8148 F1=0.8980
- TP=22 TN=23 FP=0 FN=5
- Model predicted TRUE on 44.0% of scored rows; mean noul P(true)=0.422

Misconception, all classes including `none`: n=50 accuracy=0.4200 macro-F1=0.3536

Misconception, labelled rows only (MAP `*_Misconception`): n=14 accuracy=0.4286 macro-F1=0.3636

Has-misconception vs `none` (binary, positive = not none):
- n=50 accuracy=0.5600 precision=0.3824 recall=0.9286 F1=0.5417 TP=13 TN=15 FP=21 FN=1

Artifacts:
- `map_50/clef_27b/per_row.csv`
- `map_50/clef_27b/per_row.json`
- `map_50/clef_27b/summary.json`
- `checkpoints/map_50__clef_27b.jsonl` (raw answers, usage, errors)

### map_50 / qwen (`qwen2.5:7b`)

- Finished: 2026-10-27T10:24:19Z
- Rows: 50; failed requests: 0
- Latency ms: avg 12219.2, median 11061.3, p95 17299.3
- Tokens: input 63887, output 23571, cached 0, avg total 1749.2
- Estimated cost: $0.0000 (₹0.00). Local Ollama model: no API bill. Electricity / GPU time not priced.

Correctness (TRUE = selected MC answer is right; MathDial TRUE would mean the written solution is right):

- n=50 accuracy=0.6800 precision=0.6410 recall=0.9259 F1=0.7576
- TP=25 TN=9 FP=14 FN=2
- Model predicted TRUE on 78.0% of scored rows; mean noul P(true)=0.752

Misconception, all classes including `none`: n=50 accuracy=0.4000 macro-F1=0.0972

Misconception, labelled rows only (MAP `*_Misconception`): n=14 accuracy=0.0714 macro-F1=0.0606

Has-misconception vs `none` (binary, positive = not none):
- n=50 accuracy=0.6200 precision=0.4138 recall=0.8571 F1=0.5581 TP=12 TN=19 FP=17 FN=2

Artifacts:
- `map_50/qwen/per_row.csv`
- `map_50/qwen/per_row.json`
- `map_50/qwen/summary.json`
- `checkpoints/map_50__qwen.jsonl` (raw answers, usage, errors)

### map_500 / typesafe (`jev-latest`)

- Finished: 2026-10-26T19:55:06Z
- Rows: 500; failed requests: 0
- Latency ms: avg 370.8, median 371.3, p95 399.2
- Tokens: input 748017, output 181911, cached 0, avg total 1859.9
- Estimated cost: $0.0314 (₹2.73). Jev bills input tokens only at $0.042/MTok

Correctness (TRUE = selected MC answer is right; MathDial TRUE would mean the written solution is right):

- n=500 accuracy=0.9800 precision=0.9719 recall=0.9928 F1=0.9823
- TP=277 TN=213 FP=8 FN=2
- Model predicted TRUE on 57.0% of scored rows; mean noul P(true)=0.593

Misconception, all classes including `none`: n=500 accuracy=0.6440 macro-F1=0.2471

Misconception, labelled rows only (MAP `*_Misconception`): n=134 accuracy=0.4627 macro-F1=0.2897

Has-misconception vs `none` (binary, positive = not none):
- n=500 accuracy=0.7540 precision=0.5247 recall=0.8731 F1=0.6555 TP=117 TN=260 FP=106 FN=17

Artifacts:
- `map_500/typesafe/per_row.csv`
- `map_500/typesafe/per_row.json`
- `map_500/typesafe/summary.json`
- `checkpoints/map_500__typesafe.jsonl` (raw answers, usage, errors)

### map_500 / gemini (`gemini-3.8-flash`)

- Finished: 2026-10-26T21:24:12Z
- Rows: 500; failed requests: 0
- Latency ms: avg 9043.8, median 8625.6, p95 12614.8
- Tokens: input 673834, output 507766, cached 0, avg total 2363.2
- Estimated cost: $2.4095 (₹209.63). Gemini 3.8 Flash intro pricing $0.75/M in + $3.75/M out; output includes thinking tokens

Correctness (TRUE = selected MC answer is right; MathDial TRUE would mean the written solution is right):

- n=500 accuracy=0.9980 precision=0.9964 recall=1.0000 F1=0.9982
- TP=279 TN=220 FP=1 FN=0
- Model predicted TRUE on 56.0% of scored rows; mean noul P(true)=0.559

Misconception, all classes including `none`: n=500 accuracy=0.7300 macro-F1=0.3536

Misconception, labelled rows only (MAP `*_Misconception`): n=134 accuracy=0.5373 macro-F1=0.4082

Has-misconception vs `none` (binary, positive = not none):
- n=500 accuracy=0.8180 precision=0.6138 recall=0.8657 F1=0.7183 TP=116 TN=293 FP=73 FN=18

Artifacts:
- `map_500/gemini/per_row.csv`
- `map_500/gemini/per_row.json`
- `map_500/gemini/summary.json`
- `checkpoints/map_500__gemini.jsonl` (raw answers, usage, errors)

### map_500 / clef_9b (`clef-flash:latest`)

- Finished: 2026-10-27T08:25:14Z
- Rows: 500; failed requests: 0
- Latency ms: avg 2561.0, median 2568.9, p95 2710.2
- Tokens: input 582487, output 0, cached 45424, avg total 1165.0
- Estimated cost: $0.0000 (₹0.00). Local Ollama model: no API bill. Electricity / GPU time not priced.

Correctness (TRUE = selected MC answer is right; MathDial TRUE would mean the written solution is right):

- n=500 accuracy=0.8440 precision=0.9036 recall=0.8065 F1=0.8523
- TP=225 TN=197 FP=24 FN=54
- Model predicted TRUE on 49.8% of scored rows; mean noul P(true)=0.493

Misconception, all classes including `none`: n=500 accuracy=0.4540 macro-F1=0.1669

Misconception, labelled rows only (MAP `*_Misconception`): n=134 accuracy=0.2463 macro-F1=0.1946

Has-misconception vs `none` (binary, positive = not none):
- n=500 accuracy=0.6440 precision=0.4267 recall=0.9552 F1=0.5899 TP=128 TN=194 FP=172 FN=6

Artifacts:
- `map_500/clef_9b/per_row.csv`
- `map_500/clef_9b/per_row.json`
- `map_500/clef_9b/summary.json`
- `checkpoints/map_500__clef_9b.jsonl` (raw answers, usage, errors)

### map_500 / clef_27b (`clef:latest`)

- Finished: 2026-10-27T09:53:13Z
- Rows: 500; failed requests: 0
- Latency ms: avg 8929.6, median 8938.1, p95 9452.4
- Tokens: input 582487, output 0, cached 32346, avg total 1165.0
- Estimated cost: $0.0000 (₹0.00). Local Ollama model: no API bill. Electricity / GPU time not priced.

Correctness (TRUE = selected MC answer is right; MathDial TRUE would mean the written solution is right):

- n=500 accuracy=0.8920 precision=0.9913 recall=0.8136 F1=0.8937
- TP=227 TN=219 FP=2 FN=52
- Model predicted TRUE on 45.8% of scored rows; mean noul P(true)=0.469

Misconception, all classes including `none`: n=500 accuracy=0.4560 macro-F1=0.2122

Misconception, labelled rows only (MAP `*_Misconception`): n=134 accuracy=0.3209 macro-F1=0.2564

Has-misconception vs `none` (binary, positive = not none):
- n=500 accuracy=0.6220 precision=0.4104 recall=0.9403 F1=0.5714 TP=126 TN=185 FP=181 FN=8

Artifacts:
- `map_500/clef_27b/per_row.csv`
- `map_500/clef_27b/per_row.json`
- `map_500/clef_27b/summary.json`
- `checkpoints/map_500__clef_27b.jsonl` (raw answers, usage, errors)

### map_500 / qwen (`qwen2.5:7b`)

- Finished: 2026-10-27T13:16:37Z
- Rows: 500; failed requests: 2
- Latency ms: avg 12188.1, median 11407.1, p95 16939.5
- Tokens: input 638527, output 235417, cached 0, avg total 1754.9
- Estimated cost: $0.0000 (₹0.00). Local Ollama model: no API bill. Electricity / GPU time not priced.

Correctness (TRUE = selected MC answer is right; MathDial TRUE would mean the written solution is right):

- n=498 accuracy=0.6847 precision=0.6554 recall=0.9134 F1=0.7632
- TP=253 TN=88 FP=133 FN=24
- Model predicted TRUE on 77.5% of scored rows; mean noul P(true)=0.748

Misconception, all classes including `none`: n=498 accuracy=0.4378 macro-F1=0.0913

Misconception, labelled rows only (MAP `*_Misconception`): n=134 accuracy=0.1269 macro-F1=0.0927

Has-misconception vs `none` (binary, positive = not none):
- n=498 accuracy=0.5964 precision=0.3707 recall=0.7164 F1=0.4885 TP=96 TN=201 FP=163 FN=38

Artifacts:
- `map_500/qwen/per_row.csv`
- `map_500/qwen/per_row.json`
- `map_500/qwen/summary.json`
- `checkpoints/map_500__qwen.jsonl` (raw answers, usage, errors)

### mathdial_test / typesafe (`jev-latest`)

- Finished: 2026-10-26T19:58:47Z
- Rows: 599; failed requests: 0
- Latency ms: avg 363.5, median 361.6, p95 402.4
- Tokens: input 321537, output 11980, cached 0, avg total 556.8
- Estimated cost: $0.0135 (₹1.17). Jev bills input tokens only at $0.042/MTok

Correctness (TRUE = selected MC answer is right; MathDial TRUE would mean the written solution is right):

- n=599 accuracy=0.6327 precision=0.0000 recall=0.0000 F1=0.0000
- TP=0 TN=379 FP=220 FN=0
- Model predicted TRUE on 36.7% of scored rows; mean noul P(true)=0.411

Artifacts:
- `mathdial_test/typesafe/per_row.csv`
- `mathdial_test/typesafe/per_row.json`
- `mathdial_test/typesafe/summary.json`
- `checkpoints/mathdial_test__typesafe.jsonl` (raw answers, usage, errors)

### mathdial_test / gemini (`gemini-3.8-flash`)

- Finished: 2026-10-26T22:21:42Z
- Rows: 599; failed requests: 0
- Latency ms: avg 5571.8, median 5258.4, p95 8083.6
- Tokens: input 347046, output 297432, cached 0, avg total 1075.9
- Estimated cost: $1.3757 (₹119.68). Gemini 3.8 Flash intro pricing $0.75/M in + $3.75/M out; output includes thinking tokens

Correctness (TRUE = selected MC answer is right; MathDial TRUE would mean the written solution is right):

- n=599 accuracy=0.9633 precision=0.0000 recall=0.0000 F1=0.0000
- TP=0 TN=577 FP=22 FN=0
- Model predicted TRUE on 3.7% of scored rows; mean noul P(true)=0.037

Artifacts:
- `mathdial_test/gemini/per_row.csv`
- `mathdial_test/gemini/per_row.json`
- `mathdial_test/gemini/summary.json`
- `checkpoints/mathdial_test__gemini.jsonl` (raw answers, usage, errors)

### mathdial_test / clef_9b (`clef-flash:latest`)

- Finished: 2026-10-27T08:31:10Z
- Rows: 599; failed requests: 0
- Latency ms: avg 560.9, median 761.0, p95 912.4
- Tokens: input 219359, output 0, cached 85933, avg total 366.2
- Estimated cost: $0.0000 (₹0.00). Local Ollama model: no API bill. Electricity / GPU time not priced.

Correctness (TRUE = selected MC answer is right; MathDial TRUE would mean the written solution is right):

- n=599 accuracy=0.3589 precision=0.0000 recall=0.0000 F1=0.0000
- TP=0 TN=215 FP=384 FN=0
- Model predicted TRUE on 64.1% of scored rows; mean noul P(true)=0.597

Artifacts:
- `mathdial_test/clef_9b/per_row.csv`
- `mathdial_test/clef_9b/per_row.json`
- `mathdial_test/clef_9b/summary.json`
- `checkpoints/mathdial_test__clef_9b.jsonl` (raw answers, usage, errors)

### mathdial_test / clef_27b (`clef:latest`)

- Finished: 2026-10-27T10:14:08Z
- Rows: 599; failed requests: 0
- Latency ms: avg 2050.3, median 2553.5, p95 3052.5
- Tokens: input 219359, output 0, cached 73035, avg total 366.2
- Estimated cost: $0.0000 (₹0.00). Local Ollama model: no API bill. Electricity / GPU time not priced.

Correctness (TRUE = selected MC answer is right; MathDial TRUE would mean the written solution is right):

- n=599 accuracy=0.8631 precision=0.0000 recall=0.0000 F1=0.0000
- TP=0 TN=517 FP=82 FN=0
- Model predicted TRUE on 13.7% of scored rows; mean noul P(true)=0.195

Artifacts:
- `mathdial_test/clef_27b/per_row.csv`
- `mathdial_test/clef_27b/per_row.json`
- `mathdial_test/clef_27b/summary.json`
- `checkpoints/mathdial_test__clef_27b.jsonl` (raw answers, usage, errors)

### mathdial_test / qwen (`qwen2.5:7b`)

- Finished: 2026-10-27T13:57:47Z
- Rows: 599; failed requests: 355
- Latency ms: avg 6820.3, median 6487.2, p95 9543.5
- Tokens: input 139411, output 74719, cached 0, avg total 877.6
- Estimated cost: $0.0000 (₹0.00). Local Ollama model: no API bill. Electricity / GPU time not priced.

Correctness (TRUE = selected MC answer is right; MathDial TRUE would mean the written solution is right):

- n=244 accuracy=0.2787 precision=0.0000 recall=0.0000 F1=0.0000
- TP=0 TN=68 FP=176 FN=0
- Model predicted TRUE on 72.1% of scored rows; mean noul P(true)=0.670

Artifacts:
- `mathdial_test/qwen/per_row.csv`
- `mathdial_test/qwen/per_row.json`
- `mathdial_test/qwen/summary.json`
- `checkpoints/mathdial_test__qwen.jsonl` (raw answers, usage, errors)

## Cost vs the planned cheaper slices

- Planned (from the slice table): MAP-50 Gemini ~₹15–20, MAP-500 Gemini ~₹150–200, MathDial-599 Gemini ~₹230; Jev under ₹1 / ~₹1 / ~₹2.
- Actual at 87 INR/USD: Jev total ₹4.18; Gemini total ₹350.76; local Clef/Qwen ₹0 API. Grand API ₹354.94 ($4.0797).
- Requests: 5745 scored rows across all models; remaining failures after retry: 357.
- Gemini batch used `thinkingLevel=low`. Clef MAP uses a 26-class taxonomy (Ollama cap). Qwen uses the same 36-class taxonomy as Jev/Gemini.

## MathDial unique problems vs 599 dialogues

The official MathDial test file has 599 tutoring dialogues but only **394 unique `qid`s**. Duplicate qids share the same question and the same `student_incorrect_solution`; they differ in student profile, scenario, and the teacher's free-text confusion note. The model state is `{question, student_answer}`, so unique-qid metrics are the right unit for correctness. Dialogue-weighted metrics (N=599) are also stored because that is the official split size.

| Provider | Dialogues N | Unique qid N | Unique acc (TNR) | Unique predicted-TRUE rate | Dialogue acc | Dialogue predicted-TRUE |
|---|---:|---:|---:|---:|---:|---:|
| typesafe | 599 | 394 | 64.5% | 35.5% | 63.3% | 36.7% |
| gemini | 599 | 394 | 95.7% | 4.3% | 96.3% | 3.7% |
| clef_9b | 599 | 394 | 34.8% | 65.2% | 35.9% | 64.1% |
| clef_27b | 599 | 394 | 85.3% | 14.7% | 86.3% | 13.7% |
| qwen | 599 | 394 | 30.1% | 69.9% | 27.9% | 72.1% |

## MAP-500 correctness by Category

### typesafe

| map_category | n | correct_accuracy | tp | tn | fp | fn | pred_true_rate | mean_p_true | misconception_n | misconception_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| True_Correct | 202 | 99.5% | 201 | 0 | 0 | 1 | 99.5% | 0.959 | 202 | 96.5% |
| False_Misconception | 129 | 96.1% | 0 | 124 | 5 | 0 | 3.9% | 0.173 | 129 | 47.3% |
| False_Neither | 89 | 96.6% | 0 | 86 | 3 | 0 | 3.4% | 0.098 | 89 | 9.0% |
| True_Neither | 72 | 98.6% | 71 | 0 | 0 | 1 | 98.6% | 0.920 | 72 | 76.4% |
| True_Misconception | 5 | 100.0% | 5 | 0 | 0 | 0 | 100.0% | 0.978 | 5 | 20.0% |
| False_Correct | 3 | 100.0% | 0 | 3 | 0 | 0 | 0.0% | 0.163 | 3 | 66.7% |

### gemini

| map_category | n | correct_accuracy | tp | tn | fp | fn | pred_true_rate | mean_p_true | misconception_n | misconception_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| True_Correct | 202 | 100.0% | 202 | 0 | 0 | 0 | 100.0% | 0.998 | 202 | 100.0% |
| False_Misconception | 129 | 100.0% | 0 | 129 | 0 | 0 | 0.0% | 0.003 | 129 | 54.3% |
| False_Neither | 89 | 98.9% | 0 | 88 | 1 | 0 | 1.1% | 0.013 | 89 | 28.1% |
| True_Neither | 72 | 100.0% | 72 | 0 | 0 | 0 | 100.0% | 0.992 | 72 | 88.9% |
| True_Misconception | 5 | 100.0% | 5 | 0 | 0 | 0 | 100.0% | 1.000 | 5 | 40.0% |
| False_Correct | 3 | 100.0% | 0 | 3 | 0 | 0 | 0.0% | 0.000 | 3 | 66.7% |

### clef_9b

| map_category | n | correct_accuracy | tp | tn | fp | fn | pred_true_rate | mean_p_true | misconception_n | misconception_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| True_Correct | 202 | 83.7% | 169 | 0 | 0 | 33 | 83.7% | 0.721 | 202 | 73.8% |
| False_Misconception | 129 | 89.9% | 0 | 116 | 13 | 0 | 10.1% | 0.278 | 129 | 24.8% |
| False_Neither | 89 | 87.6% | 0 | 78 | 11 | 0 | 12.4% | 0.206 | 89 | 9.0% |
| True_Neither | 72 | 72.2% | 52 | 0 | 0 | 20 | 72.2% | 0.607 | 72 | 51.4% |
| True_Misconception | 5 | 80.0% | 4 | 0 | 0 | 1 | 80.0% | 0.554 | 5 | 20.0% |
| False_Correct | 3 | 100.0% | 0 | 3 | 0 | 0 | 0.0% | 0.087 | 3 | 0.0% |

### clef_27b

| map_category | n | correct_accuracy | tp | tn | fp | fn | pred_true_rate | mean_p_true | misconception_n | misconception_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| True_Correct | 202 | 89.6% | 181 | 0 | 0 | 21 | 89.6% | 0.784 | 202 | 74.8% |
| False_Misconception | 129 | 98.4% | 0 | 127 | 2 | 0 | 1.6% | 0.178 | 129 | 32.6% |
| False_Neither | 89 | 100.0% | 0 | 89 | 0 | 0 | 0.0% | 0.102 | 89 | 5.6% |
| True_Neither | 72 | 61.1% | 44 | 0 | 0 | 28 | 61.1% | 0.575 | 72 | 38.9% |
| True_Misconception | 5 | 40.0% | 2 | 0 | 0 | 3 | 40.0% | 0.412 | 5 | 20.0% |
| False_Correct | 3 | 100.0% | 0 | 3 | 0 | 0 | 0.0% | 0.173 | 3 | 33.3% |

### qwen

| map_category | n | correct_accuracy | tp | tn | fp | fn | pred_true_rate | mean_p_true | misconception_n | misconception_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| True_Correct | 201 | 93.0% | 187 | 0 | 0 | 14 | 93.0% | 0.909 | 201 | 69.7% |
| False_Misconception | 129 | 36.4% | 0 | 47 | 82 | 0 | 63.6% | 0.602 | 129 | 12.4% |
| False_Neither | 89 | 42.7% | 0 | 38 | 51 | 0 | 57.3% | 0.539 | 89 | 20.2% |
| True_Neither | 71 | 88.7% | 63 | 0 | 0 | 8 | 88.7% | 0.859 | 71 | 60.6% |
| True_Misconception | 5 | 60.0% | 3 | 0 | 0 | 2 | 60.0% | 0.600 | 5 | 20.0% |
| False_Correct | 3 | 100.0% | 0 | 3 | 0 | 0 | 0.0% | 0.000 | 3 | 0.0% |

## What each number means

- **Correct (MAP)**: `noul` on whether the student's *selected MC answer* is the right option (`Category` starts with `True_`). Not whether the explanation is good.
- **Misconception all-classes**: 36-way MAP taxonomy including `none` for unlabelled rows. Accuracy is dominated by `none` because most MAP rows have no misconception label.
- **Misconception labelled-only**: scored only on `True_Misconception` and `False_Misconception` rows. This is the hard 35-way task.
- **Has-misconception vs none**: binary collapse of the taxonomy.
- **MathDial correct**: gold is FALSE for every row (the field is `student_incorrect_solution`). Accuracy = true-negative rate. TRUE-class F1 is 0 by construction. Jev predicted TRUE on 37% of dialogues (over-confident that a wrong solution is right); Gemini predicted TRUE on 3.7%.
- **Latency / tokens**: one HTTP request per row, sequential. Gemini output tokens = candidate tokens + thinking tokens.

## Error files for qualitative write-up

Every correctness mistake and every labelled-misconception mistake is in `analysis/<slice>__<provider>__errors.csv` with the question snippet and student answer snippet.

## File map

```
eval_results/public_slices/
  REPORT.md                  this file
  comparison.csv             one row per completed run
  checkpoints/*.jsonl        every API response, including failures and retries
  <slice>/<provider>/
    per_row.csv              one row per item with GT, prediction, tokens, latency
    per_row.json             same, with reasoning and probability vectors
    summary.json             metrics + cost
    confusion_*.csv          confusion matrices
```

## Notes for the written report

- Gemini output tokens include thinking tokens (`thoughtsTokenCount` + `candidatesTokenCount`).
- Jev output is a small JSON decision object; Gemini is prompted to emit long `reasoning` fields.
- MathDial has no correct student solutions in this split, so correctness F1 for the TRUE class is expected to be 0 or near 0; accuracy then equals the true-negative rate.
- MAP `Wrong_fraction` and `Wrong_Fraction` are distinct labels in the source data and were kept distinct.
