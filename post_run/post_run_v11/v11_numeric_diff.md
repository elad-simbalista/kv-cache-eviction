# v1.1 numeric diff

Every number below is recomputed from the per-item records by `post_run_calc.py`. The `v1 printed` column is transcribed from the manuscript. Edit the paper from this table; do not hand-copy from anywhere else.


## 1. The two value-origin metrics

`most-grounded` = v1's mutually exclusive trial label (precedence exact > near > novel). `any-novel` = the trial emitted at least one value matching nothing in the input. **Only `any-novel` supports the phrase "appears nowhere in the input".**

| model        | task        | press         |   kept |   n_answered |   n_exact_copy |   n_near_copy |   n_novel |   novel_rate |   n_any_novel |   any_novel_rate |   any_novel_ci_low |   any_novel_ci_high |
|:-------------|:------------|:--------------|-------:|-------------:|---------------:|--------------:|----------:|-------------:|--------------:|-----------------:|-------------------:|--------------------:|
| llama31-8b   | niah_none   | knorm         |   0.25 |          348 |            167 |           176 |         5 |       0.0144 |            47 |           0.1351 |             0.1031 |              0.175  |
| llama31-8b   | niah_none   | knorm         |   0.5  |          348 |            275 |            72 |         1 |       0.0029 |            39 |           0.1121 |             0.0831 |              0.1495 |
| llama31-8b   | niah_none   | none          |   1    |          397 |            396 |             0 |         1 |       0.0025 |             3 |           0.0076 |             0.0026 |              0.022  |
| llama31-8b   | niah_none   | snapkv        |   0.25 |          225 |            208 |             8 |         9 |       0.04   |            18 |           0.08   |             0.0512 |              0.1229 |
| llama31-8b   | niah_none   | snapkv        |   0.5  |          312 |            294 |             9 |         9 |       0.0288 |            35 |           0.1122 |             0.0818 |              0.152  |
| llama31-8b   | niah_none   | streaming_llm |   0.25 |          228 |            227 |             0 |         1 |       0.0044 |             1 |           0.0044 |             0.0008 |              0.0244 |
| llama31-8b   | niah_none   | streaming_llm |   0.5  |          365 |            355 |             0 |        10 |       0.0274 |            12 |           0.0329 |             0.0189 |              0.0566 |
| llama31-8b   | niah_single | knorm         |   0.25 |          432 |            150 |           272 |        10 |       0.0231 |            13 |           0.0301 |             0.0177 |              0.0508 |
| llama31-8b   | niah_single | knorm         |   0.5  |          636 |            424 |           209 |         3 |       0.0047 |             4 |           0.0063 |             0.0024 |              0.0161 |
| llama31-8b   | niah_single | none          |   1    |          652 |            652 |             0 |         0 |       0      |             0 |           0      |             0      |              0.0059 |
| llama31-8b   | niah_single | snapkv        |   0.25 |          302 |            268 |            21 |        13 |       0.043  |            13 |           0.043  |             0.0253 |              0.0722 |
| llama31-8b   | niah_single | snapkv        |   0.5  |          494 |            459 |            26 |         9 |       0.0182 |             9 |           0.0182 |             0.0096 |              0.0343 |
| llama31-8b   | niah_single | streaming_llm |   0.25 |          176 |            174 |             0 |         2 |       0.0114 |             2 |           0.0114 |             0.0031 |              0.0405 |
| llama31-8b   | niah_single | streaming_llm |   0.5  |          308 |            306 |             0 |         2 |       0.0065 |             2 |           0.0065 |             0.0018 |              0.0234 |
| qwen25-7b-1m | niah_none   | knorm         |   0.25 |           13 |              5 |             6 |         2 |       0.1538 |             9 |           0.6923 |             0.4237 |              0.8732 |
| qwen25-7b-1m | niah_none   | knorm         |   0.5  |           61 |             52 |             7 |         2 |       0.0328 |             6 |           0.0984 |             0.0459 |              0.1984 |
| qwen25-7b-1m | niah_none   | none          |   1    |          155 |            155 |             0 |         0 |       0      |             0 |           0      |             0      |              0.0242 |
| qwen25-7b-1m | niah_none   | snapkv        |   0.25 |           44 |             37 |             3 |         4 |       0.0909 |            15 |           0.3409 |             0.2188 |              0.4886 |
| qwen25-7b-1m | niah_none   | snapkv        |   0.5  |           71 |             60 |             6 |         5 |       0.0704 |            10 |           0.1408 |             0.0783 |              0.2402 |
| qwen25-7b-1m | niah_none   | streaming_llm |   0.25 |           67 |             67 |             0 |         0 |       0      |             3 |           0.0448 |             0.0153 |              0.1236 |
| qwen25-7b-1m | niah_none   | streaming_llm |   0.5  |           88 |             88 |             0 |         0 |       0      |             2 |           0.0227 |             0.0063 |              0.0791 |
| qwen25-7b-1m | niah_single | knorm         |   0.25 |           96 |             13 |            58 |        25 |       0.2604 |            26 |           0.2708 |             0.192  |              0.3673 |
| qwen25-7b-1m | niah_single | knorm         |   0.5  |          322 |            112 |           179 |        31 |       0.0963 |            31 |           0.0963 |             0.0687 |              0.1334 |
| qwen25-7b-1m | niah_single | none          |   1    |          641 |            640 |             1 |         0 |       0      |             0 |           0      |             0      |              0.006  |
| qwen25-7b-1m | niah_single | snapkv        |   0.25 |          265 |            110 |            82 |        73 |       0.2755 |            73 |           0.2755 |             0.2252 |              0.3322 |
| qwen25-7b-1m | niah_single | snapkv        |   0.5  |          464 |            258 |           133 |        73 |       0.1573 |            75 |           0.1616 |             0.1309 |              0.1979 |
| qwen25-7b-1m | niah_single | streaming_llm |   0.25 |          177 |            177 |             0 |         0 |       0      |             0 |           0      |             0      |              0.0212 |
| qwen25-7b-1m | niah_single | streaming_llm |   0.5  |          294 |            294 |             0 |         0 |       0      |             0 |           0      |             0      |              0.0129 |

**Abstract sentence.** Llama KNorm-25%, answered absent trials: most-grounded-novel 5/348 = 1.4%; **any-novel 47/348 = 13.5% [10.3%, 17.5%]**. v1 printed the first number with the second number's wording.


**"<= 4% in every Llama condition".** Under most-grounded the max is 4.3%; under any-novel it is 13.5%. Scope the sentence to the taxonomy or replace the number.


## 2. Strict scoring, by scorer and task

- **strict_none** — 12 of 98 cells shift >= 0.05 (7 >= 0.10). By model: llama31-8b: 11, qwen25-7b-1m: 1.

- **lenient_single** — 11 of 196 cells shift >= 0.05 (2 >= 0.10). By model: llama31-8b: 8, qwen25-7b-1m: 3.


Section 5.2 must quote the `strict_none` row only. The `lenient_single` row answers a different question and was summed into v1's count of 23.


## 3. Present-trial outcome composition (Figure 2)

| model        | press         |   kept |   n_trials |   correct |   answered_contains_target_not_credited |   wrong_near_copy_of_target |   wrong_not_near_copy |   abstention |
|:-------------|:--------------|-------:|-----------:|----------:|----------------------------------------:|----------------------------:|----------------------:|-------------:|
| llama31-8b   | knorm         |   0.25 |        700 |       145 |                                       5 |                         272 |                    10 |          268 |
| llama31-8b   | none          |   1    |        700 |       639 |                                      13 |                           0 |                     0 |           48 |
| llama31-8b   | snapkv        |   0.25 |        700 |       245 |                                      23 |                          21 |                    13 |          398 |
| llama31-8b   | streaming_llm |   0.25 |        700 |       174 |                                       0 |                           0 |                     2 |          524 |
| qwen25-7b-1m | knorm         |   0.25 |        700 |        12 |                                       1 |                          58 |                    25 |          604 |
| qwen25-7b-1m | none          |   1    |        700 |       635 |                                       5 |                           1 |                     0 |           59 |
| qwen25-7b-1m | snapkv        |   0.25 |        700 |       109 |                                       1 |                          82 |                    73 |          435 |
| qwen25-7b-1m | streaming_llm |   0.25 |        700 |       173 |                                       4 |                           0 |                     0 |          523 |

`answered_contains_target_not_credited` is the band v1's caption does not name: the output contains the true value but the official scorer rejects it. Decide where it belongs and say so in the caption.


## 4. v1 printed claims to re-check

| paper location | claim | v1 printed |
|---|---|---|
| Abstract / sec 4.2 | Llama KNorm-25% absent: answers containing a value appearing nowhere in the input | 1.4% [0.6, 3.3]  (n=5/348) |
| Abstract / sec 4.2 | Llama KNorm-25% absent: near-copies of a distractor | 176/348 (51%) |
| Sec 4.2 | Llama KNorm-25% present: answers lacking the true value that are near-copies of it | 272 of 282 |
| Sec 4.2 | ...of which truncations | 262 |
| Sec 4.2 | entirely novel values, every Llama condition | <= 4% of answered trials |
| Sec 4.2 | Qwen KNorm-25% present novel | 26% [18, 36] |
| Sec 4.2 | Qwen SnapKV-25% present novel | 28% [23, 33] |
| Sec 5.2 | cells shifting >= 0.05 under strict scoring | 23 of 294 (19 involve Llama) |
| Table 2 | Llama Streaming-50% absent exact/near/novel | 355 / 0 / 10 |
| Table 2 | Qwen SnapKV-25% absent exact/near/novel | 37 / 3 / 4 |
| Fig 2 | Llama KNorm-25% present bands (correct / near / not-near / abstain) | 0.21 / 0.40 / <0.05 / 0.38 |
| Fig 2 | Llama SnapKV-25% present bands | 0.35 / 0.06 / - / 0.57 |

Compare each against sections 1-3 above before editing.
