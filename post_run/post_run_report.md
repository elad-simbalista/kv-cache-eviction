# post_run_calc — recomputed results

All numbers below are computed from **raw per-item records**, not from reconstructed cell means. Intervals are bootstrap over items (paired within language where the cells share items).


## 1. Dispersion and amplification (replaces §6.3 + Appendix A)

| model        | arm   | task        | press         |   kept |   phi_base |   phi_comp |   amplification |   ci_low |   ci_high | ci_excludes_1   |
|:-------------|:------|:------------|:--------------|-------:|-----------:|-----------:|----------------:|---------:|----------:|:----------------|
| llama31-8b   | qa    | niah_single | knorm         |   0.25 |       8.91 |      46.39 |           5.205 |    2.961 |     8.793 | True            |
| llama31-8b   | qa    | niah_single | knorm         |   0.5  |       8.91 |      20.13 |           2.258 |    1.234 |     3.94  | True            |
| llama31-8b   | qa    | niah_single | snapkv        |   0.25 |       8.91 |      47.03 |           5.277 |    3.147 |     8.459 | True            |
| llama31-8b   | qa    | niah_single | snapkv        |   0.5  |       8.91 |      38.65 |           4.336 |    2.515 |     6.951 | True            |
| llama31-8b   | qa    | niah_single | streaming_llm |   0.25 |       8.91 |       0.65 |           0.073 |    0.04  |     0.435 | True            |
| llama31-8b   | qa    | niah_single | streaming_llm |   0.5  |       8.91 |       0.55 |           0.062 |    0.036 |     0.403 | True            |
| qwen25-7b-1m | qa    | niah_single | knorm         |   0.25 |      42.37 |       1.33 |           0.031 |    0.02  |     0.115 | True            |
| qwen25-7b-1m | qa    | niah_single | knorm         |   0.5  |      42.37 |      36.46 |           0.861 |    0.604 |     1.318 | False           |
| qwen25-7b-1m | qa    | niah_single | snapkv        |   0.25 |      42.37 |      49.67 |           1.172 |    0.815 |     1.747 | False           |
| qwen25-7b-1m | qa    | niah_single | snapkv        |   0.5  |      42.37 |      35.18 |           0.83  |    0.634 |     1.188 | False           |
| qwen25-7b-1m | qa    | niah_single | streaming_llm |   0.25 |      42.37 |       3.31 |           0.078 |    0.052 |     0.17  | True            |
| qwen25-7b-1m | qa    | niah_single | streaming_llm |   0.5  |      42.37 |       7.78 |           0.184 |    0.127 |     0.325 | True            |

`amplification` is phi_comp/phi_base with a paired bootstrap CI. Any value whose CI excludes 1 is a real change in cross-language dispersion; anything else is not.


## 2. Ordering / reallocation (replaces §6.4)

| arm   | task        | press         |   kept |   k_langs |     rho |   exact_p |   rho_boot_lo |   rho_boot_hi |   rho_floor_gated |   p_floor_gated |   k_floor_gated |
|:------|:------------|:--------------|-------:|----------:|--------:|----------:|--------------:|--------------:|------------------:|----------------:|----------------:|
| joint | niah_single | knorm         |   0.25 |         7 |  0.7769 |    0.0571 |         0.112 |         0.926 |            0.7715 |          0.1    |               6 |
| joint | niah_single | knorm         |   0.5  |         7 |  0.0546 |    0.9238 |        -0.609 |         0.649 |            0.1739 |          0.7556 |               6 |
| joint | niah_single | snapkv        |   0.25 |         7 | -0.3762 |    0.4103 |        -0.655 |        -0.108 |           -0.5822 |          0.2333 |               6 |
| joint | niah_single | snapkv        |   0.5  |         7 | -0.3545 |    0.4302 |        -0.636 |         0.138 |           -0.5441 |          0.2611 |               6 |
| joint | niah_single | streaming_llm |   0.25 |         7 | -0.835  |    0.0333 |        -0.716 |         0.703 |           -0.7775 |          0.1333 |               6 |
| joint | niah_single | streaming_llm |   0.5  |         7 | -0.2572 |    0.569  |        -0.667 |         0.667 |           -0.4227 |          0.35   |               6 |
| qa    | niah_none   | knorm         |   0.25 |         7 |  0.6307 |    0.1437 |         0.282 |         0.823 |          nan      |        nan      |               3 |
| qa    | niah_none   | knorm         |   0.5  |         7 |  0.4324 |    0.327  |         0.126 |         0.523 |          nan      |        nan      |               3 |
| qa    | niah_none   | snapkv        |   0.25 |         7 |  0.6071 |    0.1667 |         0.252 |         0.873 |          nan      |        nan      |               3 |
| qa    | niah_none   | snapkv        |   0.5  |         7 |  0.3214 |    0.4976 |         0.143 |         0.577 |          nan      |        nan      |               3 |
| qa    | niah_none   | streaming_llm |   0.25 |         7 | -0.4183 |    0.3524 |        -0.631 |        -0.222 |          nan      |        nan      |               3 |
| qa    | niah_none   | streaming_llm |   0.5  |         7 |  0.3273 |    0.475  |        -0.126 |         0.714 |          nan      |        nan      |               3 |
| qa    | niah_single | knorm         |   0.25 |         7 |  0.8532 |    0.0206 |         0.209 |         0.954 |            0.8971 |          0.0222 |               6 |
| qa    | niah_single | knorm         |   0.5  |         7 |  0.1786 |    0.7131 |        -0.091 |         0.429 |            0.2571 |          0.6583 |               6 |
| qa    | niah_single | snapkv        |   0.25 |         7 |  0.7857 |    0.048  |         0.216 |         0.893 |            0.8857 |          0.0333 |               6 |
| qa    | niah_single | snapkv        |   0.5  |         7 |  0.25   |    0.5948 |        -0.036 |         0.45  |            0.4857 |          0.3556 |               6 |
| qa    | niah_single | streaming_llm |   0.25 |         7 | -0.3495 |    0.4524 |        -0.419 |         0.846 |           -0.3131 |          0.5833 |               6 |
| qa    | niah_single | streaming_llm |   0.5  |         7 | -0.0901 |    0.8587 |        -0.355 |         0.883 |           -0.1429 |          0.8028 |               6 |

`rho_boot_lo/hi` is the item bootstrap on rho itself — the honest replacement for the draft's rounding worry. `rho_floor_gated` drops languages whose baseline is below 0.6 in either model (this removes Swahili on Qwen, whose baseline is 0.48 and which the draft's own floor rule excludes but §6.4 uses).


See the `leave_one_out` column of `ordering.csv`: if the correlation depends on Korean, that must be stated, because §3 of the redundancy analysis below gives a specific alternative explanation for Korean.


## 3. Haystack redundancy — the confound test

| lang   |   n_chars |   n_words |   n_sentences |   n_unique_sentences |   dup_sentence_frac |   max_sentence_repeats |   implied_repeats |   distinct_char_48gram |   distinct_word_10gram |   gzip_ratio |   n_samples_parsed |   n_samples_unparsed |
|:-------|----------:|----------:|--------------:|---------------------:|--------------------:|-----------------------:|------------------:|-----------------------:|-----------------------:|-------------:|-------------------:|---------------------:|
| en     |  139002   |   25938   |          1611 |                 1546 |             0.04035 |                     44 |            1.042  |                1       |                1       |      0.38402 |                 24 |                    0 |
| pl     |   83661.2 |   13883   |          1186 |                 1178 |             0.00675 |                      3 |            1.0068 |                1       |                1       |      0.40096 |                 24 |                    0 |
| zh     |   36134.7 |     261   |          1464 |                 1460 |             0.00273 |                      2 |            1.0027 |                1       |                1       |      0.44455 |                 24 |                    0 |
| ja     |   45923   |     270   |          1264 |                 1264 |             0       |                      1 |            1      |                1       |                1       |      0.32889 |                 24 |                    0 |
| ko     |   51032.8 |   13029   |          1589 |                  341 |             0.7854  |                    171 |            4.6598 |                0.24681 |                0.24647 |      0.10024 |                 24 |                    0 |
| vi     |  109652   |   25012.5 |          1200 |                 1027 |             0.14417 |                     93 |            1.1685 |                1       |                0.99996 |      0.33077 |                 24 |                    0 |
| sw     |   82658.3 |   16016   |           882 |                  879 |             0.0034  |                      2 |            1.0034 |                0.99981 |                0.99881 |      0.36895 |                 24 |                    0 |

Redundancy is measured **tokenizer-free** (sentences, characters, gzip) on purpose: a token-based measure would entangle redundancy with fertility, which is a competing explanation tested separately.


**Redundancy vs content-press retention:**

| model        | press   |   kept | redundancy_metric    |   k_langs |     rho |   exact_p | leave_one_out                                                                                      |     q_bh |
|:-------------|:--------|-------:|:---------------------|----------:|--------:|----------:|:---------------------------------------------------------------------------------------------------|---------:|
| llama31-8b   | knorm   |   0.25 | implied_repeats      |         7 |  0.3214 |    0.4976 | {"en": 0.257, "pl": 0.371, "zh": 0.314, "ja": 0.771, "ko": -0.086, "vi": 0.257, "sw": 0.314}       | 0.758248 |
| llama31-8b   | knorm   |   0.25 | dup_sentence_frac    |         7 |  0.3214 |    0.4976 | {"en": 0.257, "pl": 0.371, "zh": 0.314, "ja": 0.771, "ko": -0.086, "vi": 0.257, "sw": 0.314}       | 0.758248 |
| llama31-8b   | knorm   |   0.25 | distinct_char_48gram |         7 | -0.4009 |    0.4286 | {"en": -0.507, "pl": -0.338, "zh": -0.338, "ja": -0.507, "ko": 0.131, "vi": -0.507, "sw": -0.655}  | 0.758248 |
| llama31-8b   | knorm   |   0.25 | gzip_ratio           |         7 | -0.8571 |    0.0238 | {"en": -0.943, "pl": -0.829, "zh": -0.829, "ja": -0.771, "ko": -0.771, "vi": -0.886, "sw": -0.886} | 0.30464  |
| llama31-8b   | knorm   |   0.5  | implied_repeats      |         7 |  0.6786 |    0.1095 | {"en": 0.6, "pl": 0.886, "zh": 0.714, "ja": 0.714, "ko": 0.486, "vi": 0.486, "sw": 0.771}          | 0.389333 |
| llama31-8b   | knorm   |   0.5  | dup_sentence_frac    |         7 |  0.6786 |    0.1095 | {"en": 0.6, "pl": 0.886, "zh": 0.714, "ja": 0.714, "ko": 0.486, "vi": 0.486, "sw": 0.771}          | 0.389333 |
| llama31-8b   | knorm   |   0.5  | distinct_char_48gram |         7 | -0.6682 |    0.1429 | {"en": -0.676, "pl": -0.676, "zh": -0.676, "ja": -0.676, "ko": -0.393, "vi": -0.845, "sw": -0.655} | 0.45728  |
| llama31-8b   | knorm   |   0.5  | gzip_ratio           |         7 | -0.75   |    0.0663 | {"en": -0.771, "pl": -0.657, "zh": -0.657, "ja": -0.943, "ko": -0.6, "vi": -0.771, "sw": -0.771}   | 0.3536   |
| llama31-8b   | snapkv  |   0.25 | implied_repeats      |         7 |  0.1071 |    0.8397 | {"en": -0.029, "pl": 0.2, "zh": 0.029, "ja": 0.029, "ko": -0.2, "vi": 0.371, "sw": 0.371}          | 0.89568  |
| llama31-8b   | snapkv  |   0.25 | dup_sentence_frac    |         7 |  0.1071 |    0.8397 | {"en": -0.029, "pl": 0.2, "zh": 0.029, "ja": 0.029, "ko": -0.2, "vi": 0.371, "sw": 0.371}          | 0.89568  |
| llama31-8b   | snapkv  |   0.25 | distinct_char_48gram |         7 | -0.7572 |    0.0952 | {"en": -0.778, "pl": -0.778, "zh": -0.778, "ja": -0.778, "ko": -0.655, "vi": -0.778, "sw": -0.655} | 0.389333 |
| llama31-8b   | snapkv  |   0.25 | gzip_ratio           |         7 | -0.25   |    0.5948 | {"en": -0.2, "pl": 0.029, "zh": -0.371, "ja": -0.429, "ko": 0.029, "vi": -0.429, "sw": -0.371}     | 0.865164 |
| llama31-8b   | snapkv  |   0.5  | implied_repeats      |         7 |  0.1429 |    0.7825 | {"en": 0.029, "pl": 0.314, "zh": 0.143, "ja": 0.143, "ko": -0.371, "vi": 0.486, "sw": 0.2}         | 0.89568  |
| llama31-8b   | snapkv  |   0.5  | dup_sentence_frac    |         7 |  0.1429 |    0.7825 | {"en": 0.029, "pl": 0.314, "zh": 0.143, "ja": 0.143, "ko": -0.371, "vi": 0.486, "sw": 0.2}         | 0.89568  |
| llama31-8b   | snapkv  |   0.5  | distinct_char_48gram |         7 | -0.8018 |    0.0476 | {"en": -0.845, "pl": -0.845, "zh": -0.845, "ja": -0.845, "ko": -0.655, "vi": -0.845, "sw": -0.655} | 0.30464  |
| llama31-8b   | snapkv  |   0.5  | gzip_ratio           |         7 | -0.4286 |    0.3536 | {"en": -0.486, "pl": -0.371, "zh": -0.371, "ja": -0.371, "ko": -0.086, "vi": -0.771, "sw": -0.486} | 0.754347 |
| qwen25-7b-1m | knorm   |   0.25 | implied_repeats      |         7 |  0.4505 |    0.3111 | {"en": 0.406, "pl": 0.406, "zh": 0.429, "ja": 0.986, "ko": 0.116, "vi": 0.406, "sw": 0.429}        | 0.711086 |
| qwen25-7b-1m | knorm   |   0.25 | dup_sentence_frac    |         7 |  0.4505 |    0.3111 | {"en": 0.406, "pl": 0.406, "zh": 0.429, "ja": 0.986, "ko": 0.116, "vi": 0.406, "sw": 0.429}        | 0.711086 |
| qwen25-7b-1m | knorm   |   0.25 | distinct_char_48gram |         7 | -0.2023 |    0.6429 | {"en": -0.257, "pl": -0.257, "zh": -0.169, "ja": -0.257, "ko": 0.531, "vi": -0.257, "sw": -0.655}  | 0.89447  |
| qwen25-7b-1m | knorm   |   0.25 | gzip_ratio           |         7 | -0.8469 |    0.0254 | {"en": -0.899, "pl": -0.899, "zh": -0.829, "ja": -0.754, "ko": -0.754, "vi": -0.754, "sw": -1.0}   | 0.30464  |
| qwen25-7b-1m | knorm   |   0.5  | implied_repeats      |         7 |  0.5    |    0.2667 | {"en": 0.714, "pl": 0.486, "zh": 0.6, "ja": 0.6, "ko": 0.2, "vi": 0.314, "sw": 0.486}              | 0.711086 |
| qwen25-7b-1m | knorm   |   0.5  | dup_sentence_frac    |         7 |  0.5    |    0.2667 | {"en": 0.714, "pl": 0.486, "zh": 0.6, "ja": 0.6, "ko": 0.2, "vi": 0.314, "sw": 0.486}              | 0.711086 |
| qwen25-7b-1m | knorm   |   0.5  | distinct_char_48gram |         7 | -0.1336 |    0.8095 | {"en": -0.169, "pl": -0.169, "zh": -0.169, "ja": -0.169, "ko": 0.655, "vi": -0.169, "sw": -0.655}  | 0.89568  |
| qwen25-7b-1m | knorm   |   0.5  | gzip_ratio           |         7 | -0.3929 |    0.3956 | {"en": -0.486, "pl": -0.714, "zh": -0.314, "ja": -0.314, "ko": -0.029, "vi": -0.314, "sw": -0.486} | 0.758248 |
| qwen25-7b-1m | snapkv  |   0.25 | implied_repeats      |         7 |  0.0714 |    0.9063 | {"en": 0.086, "pl": 0.257, "zh": 0.143, "ja": -0.029, "ko": -0.486, "vi": 0.371, "sw": 0.143}      | 0.9063   |
| qwen25-7b-1m | snapkv  |   0.25 | dup_sentence_frac    |         7 |  0.0714 |    0.9063 | {"en": 0.086, "pl": 0.257, "zh": 0.143, "ja": -0.029, "ko": -0.486, "vi": 0.371, "sw": 0.143}      | 0.9063   |
| qwen25-7b-1m | snapkv  |   0.25 | distinct_char_48gram |         7 | -0.8018 |    0.0476 | {"en": -0.845, "pl": -0.845, "zh": -0.845, "ja": -0.845, "ko": -0.655, "vi": -0.845, "sw": -0.655} | 0.30464  |
| qwen25-7b-1m | snapkv  |   0.25 | gzip_ratio           |         7 | -0.1786 |    0.7131 | {"en": -0.257, "pl": -0.086, "zh": -0.371, "ja": -0.257, "ko": 0.314, "vi": -0.429, "sw": -0.143}  | 0.89568  |
| qwen25-7b-1m | snapkv  |   0.5  | implied_repeats      |         7 |  0.3214 |    0.4976 | {"en": 0.371, "pl": 0.371, "zh": 0.2, "ja": 0.2, "ko": -0.086, "vi": 0.771, "sw": 0.371}           | 0.758248 |
| qwen25-7b-1m | snapkv  |   0.5  | dup_sentence_frac    |         7 |  0.3214 |    0.4976 | {"en": 0.371, "pl": 0.371, "zh": 0.2, "ja": 0.2, "ko": -0.086, "vi": 0.771, "sw": 0.371}           | 0.758248 |
| qwen25-7b-1m | snapkv  |   0.5  | distinct_char_48gram |         7 | -0.8018 |    0.0476 | {"en": -0.845, "pl": -0.845, "zh": -0.845, "ja": -0.845, "ko": -0.655, "vi": -0.845, "sw": -0.655} | 0.30464  |
| qwen25-7b-1m | snapkv  |   0.5  | gzip_ratio           |         7 | -0.1071 |    0.8397 | {"en": -0.143, "pl": -0.143, "zh": -0.086, "ja": -0.371, "ko": 0.429, "vi": -0.371, "sw": -0.086}  | 0.89568  |

If these correlations are strong, the cross-model ordering of §6.4 is at least partly a property of the ONERULER corpus rather than of the languages: both models read the *same* generated text, so cross-model replication cannot distinguish a language effect from a corpus effect.


## 4. Scorer sensitivity (§6.7)

23 cells differ by >= 0.05 between the official and the robustness scorer. Largest offenders:

| model        | arm   | task        | lang   | press         |   kept |   n |   acc_official |   acc_recomputed |   acc_robust |   delta_robust |
|:-------------|:------|:------------|:-------|:--------------|-------:|----:|---------------:|-----------------:|-------------:|---------------:|
| llama31-8b   | qa    | niah_none   | ko     | none          |   1    | 100 |           0.83 |             0.83 |         0.12 |           0.71 |
| llama31-8b   | qa    | niah_none   | ko     | knorm         |   0.5  | 100 |           0.71 |             0.71 |         0.08 |           0.63 |
| llama31-8b   | qa    | niah_none   | ko     | snapkv        |   0.5  | 100 |           0.65 |             0.65 |         0.11 |           0.54 |
| llama31-8b   | qa    | niah_none   | ko     | snapkv        |   0.25 | 100 |           0.32 |             0.32 |         0.09 |           0.23 |
| llama31-8b   | qa    | niah_none   | ko     | knorm         |   0.25 | 100 |           0.34 |             0.34 |         0.12 |           0.22 |
| llama31-8b   | qa    | niah_none   | sw     | none          |   1    | 100 |           0.21 |             0.21 |         0.06 |           0.15 |
| llama31-8b   | qa    | niah_none   | ko     | streaming_llm |   0.5  | 100 |           0.26 |             0.26 |         0.14 |           0.12 |
| qwen25-7b-1m | joint | niah_single | ja     | none          |   1    | 100 |           0.9  |             0.9  |         1    |          -0.1  |
| llama31-8b   | qa    | niah_single | ko     | snapkv        |   0.25 | 100 |           0.62 |             0.62 |         0.72 |          -0.1  |
| qwen25-7b-1m | qa    | niah_none   | sw     | none          |   1    | 100 |           0.58 |             0.58 |         0.49 |           0.09 |
| llama31-8b   | qa    | niah_none   | sw     | snapkv        |   0.5  | 100 |           0.13 |             0.13 |         0.05 |           0.08 |
| qwen25-7b-1m | joint | niah_single | ja     | snapkv        |   0.25 | 100 |           0.92 |             0.92 |         0.99 |          -0.07 |
| llama31-8b   | qa    | niah_none   | ko     | streaming_llm |   0.25 | 100 |           0.38 |             0.38 |         0.31 |           0.07 |
| llama31-8b   | qa    | niah_none   | ja     | none          |   1    | 100 |           0.51 |             0.51 |         0.44 |           0.07 |
| llama31-8b   | joint | niah_single | ko     | streaming_llm |   0.5  | 100 |           0.32 |             0.32 |         0.38 |          -0.06 |

## 5. Controls

- **Positional law** (streaming acc ~= baseline x kept): 45/56 cells within 0.05, mean |err| 0.036, max 0.130

- **Arm invariance** (press=none must be identical across arms): 6/14 cells show per-item disagreement

| model        | task        | lang   |   n_discordant |   exact_p |
|:-------------|:------------|:-------|---------------:|----------:|
| llama31-8b   | niah_single | pl     |              2 |    1      |
| llama31-8b   | niah_single | sw     |              2 |    1      |
| llama31-8b   | niah_single | vi     |              4 |    0.625  |
| qwen25-7b-1m | niah_single | ja     |              6 |    0.0312 |
| qwen25-7b-1m | niah_single | pl     |              1 |    1      |
| qwen25-7b-1m | niah_single | sw     |             14 |    0.0574 |

- **Fertility**: 0/72 tests survive BH. Sign flips across arms/presses indicate no stable relationship.


## 6. Design / power (Appendix D2)

|          |   0.5 |   0.6 |   0.7 |   0.8 |
|:---------|------:|------:|------:|------:|
| (7, 100) | 0.189 | 0.283 | 0.371 | 0.527 |
| (10, 70) | 0.257 | 0.363 | 0.519 | 0.719 |
| (12, 58) | 0.3   | 0.45  | 0.628 | 0.806 |
| (15, 46) | 0.373 | 0.519 | 0.709 | 0.883 |
| (20, 35) | 0.489 | 0.685 | 0.846 | 0.951 |
| (26, 26) | 0.609 | 0.797 | 0.929 | 0.986 |

Equal GPU cost across rows. Power scales with the number of languages, not items per language.
