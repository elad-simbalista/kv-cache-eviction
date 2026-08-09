# post_run_calc v2 — value origin and protocol contrast

Both stages run on the EXISTING per-item records. No new generations.


## 7. Value origin: exact copy / near copy / entirely novel

**Answered ABSENT trials** (the population behind the draft's distractor-copy / fabrication numbers):

| model        | press         |   kept |   n_answered |   n_exact_copy |   n_near_copy |   n_novel |   near_rate |   novel_rate |   novel_ci_low |   novel_ci_high |
|:-------------|:--------------|-------:|-------------:|---------------:|--------------:|----------:|------------:|-------------:|---------------:|----------------:|
| llama31-8b   | knorm         |   0.25 |          348 |            167 |           176 |         5 |      0.5057 |       0.0144 |         0.0062 |          0.0332 |
| llama31-8b   | knorm         |   0.5  |          348 |            275 |            72 |         1 |      0.2069 |       0.0029 |         0.0005 |          0.0161 |
| llama31-8b   | none          |   1    |          397 |            396 |             0 |         1 |      0      |       0.0025 |         0.0004 |          0.0141 |
| llama31-8b   | snapkv        |   0.25 |          225 |            208 |             8 |         9 |      0.0356 |       0.04   |         0.0212 |          0.0743 |
| llama31-8b   | snapkv        |   0.5  |          312 |            294 |             9 |         9 |      0.0288 |       0.0288 |         0.0152 |          0.0539 |
| llama31-8b   | streaming_llm |   0.25 |          228 |            227 |             0 |         1 |      0      |       0.0044 |         0.0008 |          0.0244 |
| llama31-8b   | streaming_llm |   0.5  |          365 |            355 |             2 |         8 |      0.0055 |       0.0219 |         0.0111 |          0.0426 |
| qwen25-7b-1m | knorm         |   0.25 |           13 |              5 |             6 |         2 |      0.4615 |       0.1538 |         0.0433 |          0.4224 |
| qwen25-7b-1m | knorm         |   0.5  |           61 |             52 |             7 |         2 |      0.1148 |       0.0328 |         0.009  |          0.1119 |
| qwen25-7b-1m | none          |   1    |          155 |            155 |             0 |         0 |      0      |       0      |         0      |          0.0242 |
| qwen25-7b-1m | snapkv        |   0.25 |           44 |             37 |             4 |         3 |      0.0909 |       0.0682 |         0.0235 |          0.1823 |
| qwen25-7b-1m | snapkv        |   0.5  |           71 |             60 |             6 |         5 |      0.0845 |       0.0704 |         0.0305 |          0.1545 |
| qwen25-7b-1m | streaming_llm |   0.25 |           67 |             67 |             0 |         0 |      0      |       0      |         0      |          0.0542 |
| qwen25-7b-1m | streaming_llm |   0.5  |           88 |             88 |             0 |         0 |      0      |       0      |         0      |          0.0418 |

**Answered PRESENT trials** (`TARGET_EXACT` = correct retrieval; near-copies here are corrupted values):

| model        | press         |   kept |   n_answered |   n_exact_copy |   n_near_copy |   n_novel |   near_rate |   novel_rate |   novel_ci_low |   novel_ci_high |
|:-------------|:--------------|-------:|-------------:|---------------:|--------------:|----------:|------------:|-------------:|---------------:|----------------:|
| llama31-8b   | knorm         |   0.25 |          432 |            150 |           272 |        10 |      0.6296 |       0.0231 |         0.0126 |          0.0421 |
| llama31-8b   | knorm         |   0.5  |          636 |            424 |           209 |         3 |      0.3286 |       0.0047 |         0.0016 |          0.0138 |
| llama31-8b   | none          |   1    |          652 |            652 |             0 |         0 |      0      |       0      |         0      |          0.0059 |
| llama31-8b   | snapkv        |   0.25 |          302 |            268 |            21 |        13 |      0.0695 |       0.043  |         0.0253 |          0.0722 |
| llama31-8b   | snapkv        |   0.5  |          494 |            459 |            26 |         9 |      0.0526 |       0.0182 |         0.0096 |          0.0343 |
| llama31-8b   | streaming_llm |   0.25 |          176 |            174 |             0 |         2 |      0      |       0.0114 |         0.0031 |          0.0405 |
| llama31-8b   | streaming_llm |   0.5  |          308 |            306 |             0 |         2 |      0      |       0.0065 |         0.0018 |          0.0234 |
| qwen25-7b-1m | knorm         |   0.25 |           96 |             13 |            58 |        25 |      0.6042 |       0.2604 |         0.1831 |          0.3562 |
| qwen25-7b-1m | knorm         |   0.5  |          322 |            112 |           179 |        31 |      0.5559 |       0.0963 |         0.0687 |          0.1334 |
| qwen25-7b-1m | none          |   1    |          641 |            640 |             1 |         0 |      0.0016 |       0      |         0      |          0.006  |
| qwen25-7b-1m | snapkv        |   0.25 |          265 |            110 |            82 |        73 |      0.3094 |       0.2755 |         0.2252 |          0.3322 |
| qwen25-7b-1m | snapkv        |   0.5  |          464 |            258 |           133 |        73 |      0.2866 |       0.1573 |         0.127  |          0.1933 |
| qwen25-7b-1m | streaming_llm |   0.25 |          177 |            177 |             0 |         0 |      0      |       0      |         0      |          0.0212 |
| qwen25-7b-1m | streaming_llm |   0.5  |          294 |            294 |             0 |         0 |      0      |       0      |         0      |          0.0129 |

A value is a NEAR COPY iff it is within Levenshtein distance 2 of some value present in the input, or is a prefix/suffix of one sharing at least 6 digits. `ENTIRELY_NOVEL` is the only category for which the phrase "appears nowhere in the input" is defensible.


Chance check: a RANDOM value of the same width is misfiled as grounded 0.070% of the time (`value_origin_chance.csv`), so the near-copy tier is not an artefact of a loose rule.


Other-input reference used: **per-item distractors + language-level haystack union**. Pass `--data-dir` for per-item full-input matching.


## 8. Query-agnostic vs query-aware (present trials)

| model        | press         |   kept |   acc_qa |   acc_joint |   d_acc |   d_acc_lo |   d_acc_hi |   coverage_qa |   coverage_joint |   prec_qa |   prec_joint |   d_prec | prec_sig   |
|:-------------|:--------------|-------:|---------:|------------:|--------:|-----------:|-----------:|--------------:|-----------------:|----------:|-------------:|---------:|:-----------|
| llama31-8b   | knorm         |   0.25 |   0.2071 |      0.1229 | -0.0843 |    -0.1071 |    -0.0629 |        0.6171 |           0.5371 |    0.3356 |       0.2287 |  -0.1069 | True       |
| llama31-8b   | knorm         |   0.5  |   0.6014 |      0.49   | -0.1114 |    -0.1357 |    -0.0857 |        0.9086 |           0.79   |    0.6619 |       0.6203 |  -0.0417 | True       |
| llama31-8b   | none          |   1    |   0.9129 |      0.9157 |  0.0029 |    -0.0057 |     0.01   |        0.9314 |           0.9343 |    0.9801 |       0.9801 |   0.0001 | False      |
| llama31-8b   | snapkv        |   0.25 |   0.35   |      0.9129 |  0.5629 |     0.5314 |     0.5957 |        0.4314 |           0.93   |    0.8113 |       0.9816 |   0.1703 | True       |
| llama31-8b   | snapkv        |   0.5  |   0.6314 |      0.9143 |  0.2829 |     0.2529 |     0.3129 |        0.7057 |           0.9314 |    0.8947 |       0.9816 |   0.0869 | True       |
| llama31-8b   | streaming_llm |   0.25 |   0.2486 |      0.23   | -0.0186 |    -0.0329 |    -0.0043 |        0.2514 |           0.2557 |    0.9886 |       0.8994 |  -0.0892 | True       |
| llama31-8b   | streaming_llm |   0.5  |   0.4343 |      0.4071 | -0.0271 |    -0.0429 |    -0.0114 |        0.44   |           0.4357 |    0.987  |       0.9344 |  -0.0526 | True       |
| qwen25-7b-1m | knorm         |   0.25 |   0.0171 |      0.0129 | -0.0043 |    -0.0157 |     0.0071 |        0.1371 |           0.2243 |    0.125  |       0.0573 |  -0.0677 | True       |
| qwen25-7b-1m | knorm         |   0.5  |   0.1586 |      0.1643 |  0.0057 |    -0.0157 |     0.0257 |        0.46   |           0.5543 |    0.3447 |       0.2964 |  -0.0483 | True       |
| qwen25-7b-1m | none          |   1    |   0.9071 |      0.9114 |  0.0043 |    -0.0086 |     0.0157 |        0.9157 |           0.9286 |    0.9906 |       0.9815 |  -0.0091 | False      |
| qwen25-7b-1m | snapkv        |   0.25 |   0.1557 |      0.9029 |  0.7471 |     0.7214 |     0.7729 |        0.3786 |           0.9214 |    0.4113 |       0.9798 |   0.5685 | True       |
| qwen25-7b-1m | snapkv        |   0.5  |   0.3657 |      0.9214 |  0.5557 |     0.5229 |     0.5872 |        0.6629 |           0.9314 |    0.5517 |       0.9893 |   0.4375 | True       |
| qwen25-7b-1m | streaming_llm |   0.25 |   0.2471 |      0.2543 |  0.0071 |    -0.0043 |     0.0186 |        0.2529 |           0.4114 |    0.9774 |       0.6181 |  -0.3593 | True       |
| qwen25-7b-1m | streaming_llm |   0.5  |   0.4171 |      0.4343 |  0.0171 |     0.0043 |     0.03   |        0.42   |           0.5229 |    0.9932 |       0.8306 |  -0.1626 | True       |

`qa` compresses the context only (question prefilled afterwards, kvpress-leaderboard semantics); `joint` compresses the whole templated prompt, which puts SnapKV's observation window on the question -- a query-AWARE variant, though not the canonical one (KNorm can evict question tokens there, and the budget denominator is the prompt rather than the context, a difference of ~0.4%). Differences carry a paired item bootstrap; the two arms share sample_ids within each language.
