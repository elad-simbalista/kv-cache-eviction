"""
Multilingual KV-Cache Eviction Audit — single-file Modal app (v4).
Supersedes modal_kv_audit_v3.py (whose commented-out v1/v2 graveyard is
dropped; v3 itself is preserved in git / the uploads folder).

CHANGELOG v4 (the four review additions + the three analysis fixes):

  1. QUERY-AGNOSTIC ARM (`--arm qa`, the NEW PRIMARY ARM).
     v3 compressed the whole chat-templated prompt, so SnapKV's observation
     window sat on the question — SnapKV was query-AWARE and its near-lossless
     result was partially structural. v4 adds arm="qa": the press sees ONLY the
     context; the question is prefilled afterwards against the compressed
     cache, exactly the kvpress-leaderboard protocol. arm="joint" reproduces
     v3's protocol so the joint-vs-qa contrast (fig5) isolates the
     query-awareness effect per (lang, press, budget).
     Mechanics (deliberate, please don't "simplify"):
       * The ONERULER input is split into (context, question) by a marker
         DERIVED FROM THE DATA (template-tail anchoring + pairwise
         re-synchronization + a template-consistency filter that rejects
         haystack coincidences, see _derive_question_marker) and verified by
         hard invariants (question = marker+key+tail everywhere; needle in
         context, never in question; queried key shared between needle
         window and question; systematic key-adjacency leak check for
         double-key templates). `--stage
         split-audit` prints every language's marker + example question for
         a human eyeball BEFORE sweeping.
       * Budgets in the qa arm are fractions of the CONTEXT (question tokens
         are always retained — the realistic serving semantics).
       * Positions: question/generated tokens continue at their ORIGINAL
         indices (cache_position = k, k+1, ...), the same convention the
         joint arm's decode uses. The multi-token question chunk therefore
         needs an explicit 4D causal mask (transformers would otherwise
         build a wrong mask from compressed cache length); a token-by-token
         fallback path exists and smoke PROVES chunked==stepwise under real
         compression, and qa(press=None)==joint EXACTLY.
  2. niah_none WIRED END TO END (`--task niah_none`). Directly tests
     compression-induced false absence/presence. gen-data already knew the
     flags; v4 adds distractor-number capture at load, task-aware failure
     modes (false_presence_distractor vs _hallucinated), toy none samples,
     and smoke/selftest coverage.
  3. SECOND MODEL (`--model`). MODELS registry: llama31-8b (pinned) and
     qwen25-7b-1m (Qwen2.5-7B-Instruct-1M: multilingual-centric, 1M window
     so 32K+64 never overflows, qwen2 arch = kvpress-supported). Same
     generated text for both models (matched TEXT, model-native tokenization;
     budgets are fractional so this is fair); per-model fertility files;
     smoke logs the resolved Qwen commit — PIN IT in MODELS after first
     smoke. Baselines below the 0.6 floor are FLAGGED (base_floor) rather
     than silently compared.
  4. DIAGNOSE STAGE (`--stage diagnose`). Answers "does KnormPress evict the
     QUESTION/instruction rather than the needle?" (the vi collapse smell).
     Captures per-(layer, kv-head) key L2 norms via k_proj hooks (RoPE is
     norm-preserving, so pre-RoPE norms == cached-key norms), replays
     KnormPress's keep-lowest-norm selection offline, and reports keep-rates
     for needle-value / needle-window / question / haystack spans + the
     fraction of heads that fully evict the needle value. Both norm
     orientations are recorded as a safety column.
  5. STATS FIXES from the external review:
       * ga_lo_sig computed and used to star fig1b (v3 starred the log-odds
         heatmap with the RAW flags — bug).
       * bootstrap two-sided p-values + Benjamini–Hochberg q for BOTH gap
         metrics (36-test family per facet; v3 had no multiplicity control).
       * exact-permutation p for the fertility Spearman rho (n<=8 -> exact)
         plus a leave-one-out table (the v3 "driven by ko" claim was
         backwards; the LOO table settles such claims mechanically).
       * base_acc + base_floor columns in the gap table.
  6. TOKENIZATION: chat template applied then tokenized with
     add_special_tokens=False everywhere (v3 double-added BOS; harmless but
     v4 makes joint and qa arms share EXACTLY the same token ids).
  7. Cell identity now includes (model, arm); hashes change; use
     `--stage wipe-results --confirm` once before the v4 campaign (keeps
     generated data, the ONERULER repo and the HF model cache).
  8. v4.1 EXTERNAL-REVIEW FIXES (2026-07-24):
       * fig4 stacks SIGNED per-mode deltas with separate +/- baselines
         (naive cumulative bottoms overlap on mixed signs — real bug).
       * fig1/fig1b render BOTH budgets with dual-notation stars (* raw
         CI, ** BH q<.05); fig5/arm_contrast.csv gain q + sig_bh too.
       * smoke [11]: transformers must carry cache_position INCREMENTALLY
         so joint decode continues at ORIGINAL positions under compression
         (verified in the pinned 5.2.0 source; asserted against pin drift).
         smoke [12]: joint positional twin of [6] + on-text ratio check;
         run_batch also verifies the press on the FIRST joint sample of
         every compressed cell (scorer presses keep a fixed count — the
         assert guards library changes, not content).
       * diagnose is ARM-AWARE (`--arm qa`: context-only selection with the
         question protected, mirroring _generate_qa; kvpress-exact floor
         int(kept*len); ambiguous-gold samples skipped).
       * analyze audits cross-language needle-DEPTH balance (KS vs en,
         warnings, fig6 ECDF, streaming-by-depth table) — depth imbalance
         would masquerade as a language effect, and for streaming_llm it
         would BE the effect.
       * failure-modes adds scorer_sensitivity.csv (strict-none /
         lenient-single robustness scorers; official scorer stays the
         headline) + an audit of officially-'correct' niah_none preds.
       * _classify_row canonicalizes native-script digits via int().
       * fertility: dead script-dataset fallbacks removed (unloadable under
         datasets>=3) and fertility is ALSO measured on the actual eval
         texts (fig3 prefers those columns; source recorded in summary).
       * marker derivation uses the FIRST 40 samples (split-audit and
         run_batch provably derive the same marker) + logged marker hash;
         max_key_span aligned with the 64-char invariant.
       * raw-results readers tolerate a truncated FINAL line; gen-data
         paths are SEED-AWARE; per-batch cell summaries (no cross-container
         append race on one shared file).

RUN ORDER (v4 campaign; full runbook with costs in RUNBOOK.md):
  modal run modal_kv_audit_v4.py --stage wipe-results --confirm
  modal run modal_kv_audit_v4.py --stage smoke --model llama31-8b
  modal run modal_kv_audit_v4.py --stage smoke --model qwen25-7b-1m   # then PIN rev
  modal run modal_kv_audit_v4.py --stage selftest
  modal run modal_kv_audit_v4.py --stage prepare-data                  # idempotent
  modal run --detach modal_kv_audit_v4.py --stage gen-data --task niah_single --n 100 --langs en,pl,zh,ja,ko,vi,sw
  modal run --detach modal_kv_audit_v4.py --stage gen-data --task niah_none   --n 100 --langs en,pl,zh,ja,ko,vi,sw
  modal run modal_kv_audit_v4.py --stage split-audit --task niah_single
  modal run modal_kv_audit_v4.py --stage split-audit --task niah_none   # EYEBALL both
  modal run --detach modal_kv_audit_v4.py --stage sweep --model llama31-8b  --arm qa    --task niah_single --n 100
  modal run --detach modal_kv_audit_v4.py --stage sweep --model llama31-8b  --arm joint --task niah_single --n 100
  modal run --detach modal_kv_audit_v4.py --stage sweep --model llama31-8b  --arm qa    --task niah_none   --n 100
  modal run --detach modal_kv_audit_v4.py --stage sweep --model qwen25-7b-1m --arm qa    --task niah_single --n 100
  modal run --detach modal_kv_audit_v4.py --stage sweep --model qwen25-7b-1m --arm joint --task niah_single --n 100
  modal run --detach modal_kv_audit_v4.py --stage sweep --model qwen25-7b-1m --arm qa    --task niah_none   --n 100
  modal run modal_kv_audit_v4.py --stage status --model <m> --arm <a> --task <t>
  modal run modal_kv_audit_v4.py --stage fertility --model llama31-8b  --langs en,pl,ru,zh,ja,ko,vi,hi,sw,ta
  modal run modal_kv_audit_v4.py --stage fertility --model qwen25-7b-1m --langs en,pl,ru,zh,ja,ko,vi,hi,sw,ta
  modal run modal_kv_audit_v4.py --stage aggregate
  modal run modal_kv_audit_v4.py --stage analyze
  modal run modal_kv_audit_v4.py --stage failure-modes
  modal run modal_kv_audit_v4.py --stage diagnose --model llama31-8b --arm qa --langs vi,en,ko --press knorm --kept 0.5 --n 24
  modal volume get kv-audit-vol results ./results ; modal volume get kv-audit-vol figures ./figures

FACTS PULLED FROM THE ONERULER REPO (github.com/mungg/OneRuler, July 2026)
  * Languages: cs da de en es fa fi fr hi hu it ja ko nl no pl pt ru sr st
    sv sw ta uk vi zh.
  * niah_single (config/synthetic.yaml): type_haystack=book,
    type_needle_k=words, type_needle_v=numbers, num_needle_k/v/q=1;
    tokens_to_generate=30 (synthetic/constants.py). niah_none:
    num_needle_k=4, relevant_needle=0 (queried key absent; gold = none-word).
  * Output rows {"index","input","outputs","length"} — raw text, NO chat
    template; we apply the model's chat template ourselves, identically
    across all settings.
  * Needle depth: sampled per-sample from 40 values in [0,100]; NOT recorded
    (we recover it post-hoc, see load_samples).
"""

import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path

import modal

# ----------------------------------------------------------------------------
# Frozen experiment config (change only with a DECISIONS.md entry)
# ----------------------------------------------------------------------------
MODELS = {
    # tag -> spec. `revision` None means "resolve latest and LOG it";
    # pin the logged commit here right after the first green smoke.
    "llama31-8b": dict(
        hf_id="meta-llama/Llama-3.1-8B-Instruct",
        revision="0e9e39f249a16976918f6564b8830bc894c89659",
        max_len=131072),
    "qwen25-7b-1m": dict(
        hf_id="Qwen/Qwen2.5-7B-Instruct-1M",
        # pinned 2026-07-25 from the first green smoke (rev logged there)
        revision="e28526f7bb80e2a9c8af03b831a9af3812f18fba",
        max_len=1010000),
}
DEFAULT_MODEL = "llama31-8b"
ARMS = ["qa", "joint"]        # qa = query-agnostic (PRIMARY), joint = v3 protocol
PRIMARY_ARM = "qa"
TASKS = ["niah_single", "niah_none"]

GPU_TYPE = "L40S"                                # 48 GB; alt "A100-40GB"
ATTN_IMPL = "sdpa"     # "flash_attention_2" only if you add flash-attn below
CTX_LEN = 32768                                  # frozen fallback: 16384
TASK = "niah_single"                             # default task
PRESSES = ["snapkv", "streaming_llm", "knorm"]   # streaming = positional CONTROL
BUDGETS_KEPT = [0.5, 0.25]                       # fraction of KV cache KEPT
MAX_NEW_TOKENS = 64          # v3 fix kept: >30 for verbose-style languages (sw)
DEFAULT_SEED = 42                                # ONERULER's default
PRESS_TOL = 0.05                                 # measured-vs-requested kept
N_BOOT = 2000                                    # paired bootstrap iterations
TOY_DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
QA_CHUNKED = True    # question fed as one masked chunk (fast); auto-falls back
FLOOR_ACC = 0.6      # baselines below this are flagged base_floor in analysis

ONERULER_GIT = "https://github.com/mungg/OneRuler.git"
ONERULER_ALL_LANGS = ["cs", "da", "de", "en", "es", "fa", "fi", "fr", "hi",
                      "hu", "it", "ja", "ko", "nl", "no", "pl", "pt", "ru",
                      "sr", "st", "sv", "sw", "ta", "uk", "vi", "zh"]

# Frozen 7 (floor filter 2026-07-21, DECISIONS.md §4). Same set for BOTH
# models for comparability; per-model floor problems are FLAGGED, not dropped.
LANGS = ["en", "pl", "zh", "ja", "ko", "vi", "sw"]
LANG_TO_FLORES = {  # FLORES-200 config names, all 26
    "cs": "ces_Latn", "da": "dan_Latn", "de": "deu_Latn", "en": "eng_Latn",
    "es": "spa_Latn", "fa": "pes_Arab", "fi": "fin_Latn", "fr": "fra_Latn",
    "hi": "hin_Deva", "hu": "hun_Latn", "it": "ita_Latn", "ja": "jpn_Jpan",
    "ko": "kor_Hang", "nl": "nld_Latn", "no": "nob_Latn", "pl": "pol_Latn",
    "pt": "por_Latn", "ru": "rus_Cyrl", "sr": "srp_Cyrl", "st": "sot_Latn",
    "sv": "swe_Latn", "sw": "swh_Latn", "ta": "tam_Taml", "uk": "ukr_Cyrl",
    "vi": "vie_Latn", "zh": "zho_Hans",
}

# kvpress semantics: True => compression_ratio means "fraction REMOVED".
# smoke verifies empirically and hard-fails with instructions if wrong.
KVPRESS_RATIO_MEANS_REMOVED = True

VOL = "/vol"
RAW_DIR = f"{VOL}/results/raw"
# v4.1: per-batch summary files (cells/<model>__<arm>__<task>__<lang>.jsonl)
# instead of one shared cells.jsonl — 7 containers appending to a single
# volume file race on commit and can drop lines. Informational only; all
# analysis recomputes from raw/.
CELLS_DIR = f"{VOL}/results/cells"
DATA_DIR = f"{VOL}/data"
REPO_DIR = f"{VOL}/oneruler_repo"
LOG_DIR = f"{VOL}/logs"

# ----------------------------------------------------------------------------
# Modal plumbing
# ----------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        # Pinned after first green v3 smoke (2026-07-21, L40S) — see
        # /vol/logs/env_smoke_*.txt for the full freeze.
        "torch==2.13.0", "transformers==5.2.0", "kvpress==0.5.4",
        "accelerate", "datasets==5.0.0",
        "pandas", "pyarrow", "numpy", "matplotlib", "sentencepiece",
        "protobuf", "hf_transfer",
        # ONERULER generation deps (their requirement.txt + missing stanza)
        "stanza", "nltk", "wonderwords", "tenacity", "tiktoken", "tqdm",
        "pyyaml",
    )
    .env({"HF_HOME": f"{VOL}/hf_cache",
          "HF_HUB_ENABLE_HF_TRANSFER": "1",
          "STANZA_RESOURCES_DIR": f"{VOL}/stanza"})
)
app = modal.App("kv-eviction-audit")
vol = modal.Volume.from_name("kv-audit-vol", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface-secret")]
GPU_KW = dict(image=image, gpu=GPU_TYPE, volumes={VOL: vol}, secrets=secrets)
CPU_KW = dict(image=image, volumes={VOL: vol}, secrets=secrets)

# ----------------------------------------------------------------------------
# Logging + progress helpers (unchanged from v3)
# ----------------------------------------------------------------------------
_LOGFILE = None
_ENV_DUMPED = False


def log_open(stage: str):
    """Mirror all log() output to /vol/logs/<stage>_<ts>.log and dump the
    container's environment (pip freeze, torch/CUDA) once for reproducibility."""
    global _LOGFILE, _ENV_DUMPED
    os.makedirs(LOG_DIR, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    _LOGFILE = open(f"{LOG_DIR}/{stage}_{ts}.log", "a")
    log(f"===== stage={stage} ts={ts} ctx={CTX_LEN} attn={ATTN_IMPL} "
        f"seed_default={DEFAULT_SEED} =====")
    if not _ENV_DUMPED:
        _ENV_DUMPED = True
        try:
            freeze = subprocess.check_output(
                [sys.executable, "-m", "pip", "freeze"], text=True)
            extras = []
            try:
                import torch
                dev = (torch.cuda.get_device_name(0)
                       if torch.cuda.is_available() else "cpu")
                extras = [f"# torch.cuda={torch.version.cuda} device={dev}"]
            except Exception:
                pass
            env_path = f"{LOG_DIR}/env_{stage}_{ts}.txt"
            Path(env_path).write_text("\n".join(extras) + "\n" + freeze)
            keylibs = [l for l in freeze.splitlines()
                       if l.split("==")[0].lower() in
                       ("torch", "transformers", "kvpress", "datasets")]
            log("env: " + " | ".join(keylibs) + f" -> {env_path}")
        except Exception as e:
            log(f"env dump failed (non-fatal): {e}")


def log(msg: str):
    line = f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOGFILE:
        _LOGFILE.write(line + "\n")
        _LOGFILE.flush()


def bar(done: int, total: int, t0: float, extra: str = "") -> str:
    frac = done / max(total, 1)
    filled = int(frac * 24)
    eta = (time.time() - t0) / max(done, 1) * (total - done)
    return (f"[{'#' * filled}{'.' * (24 - filled)}] {done}/{total} "
            f"({frac * 100:4.1f}%) | ETA {eta / 60:5.1f}m {extra}")


def _load_fertility(model_tag: str = DEFAULT_MODEL):
    """Fertility DataFrame indexed by lang for THIS model's tokenizer, or
    None. Prefers results/fertility_<model>.csv; falls back to the legacy
    results/fertility.csv (llama tokenizer) with a warning."""
    import pandas as pd
    for fp in (Path(f"{VOL}/results/fertility_{model_tag}.csv"),
               Path(f"{VOL}/results/fertility.csv")):
        if not fp.exists() or fp.stat().st_size == 0:
            continue
        try:
            fert = pd.read_csv(fp)
        except pd.errors.EmptyDataError:
            continue
        if fert.empty or "lang" not in fert.columns:
            continue
        if fp.name == "fertility.csv" and model_tag != DEFAULT_MODEL:
            log(f"WARNING: using legacy fertility.csv for model={model_tag} "
                f"— run --stage fertility --model {model_tag}")
        return fert.set_index("lang")
    log(f"no fertility file for model={model_tag} — fig3/Spearman skipped "
        f"(run --stage fertility --model {model_tag})")
    return None


# ----------------------------------------------------------------------------
# Official ONERULER scoring — ported from OneRuler/eval/evaluate.py (MIT).
# niah_single -> compare_numbers ; niah_none -> compare_none.
# ----------------------------------------------------------------------------
NONE_DICT = {
    "en": ["none"], "ko": ["없음"], "pl": ["brak"], "zh": ["无"],
    "vi": ["Không có"], "ja": ["なし", "数字はありません"], "ta": ["ஏதுமில்லை"],
    "hu": ["nincs"], "fr": ["aucun"], "no": ["ingen"],
    "uk": ["немає", "Нема"], "ru": ["нет"], "de": ["Keine vorhanden"],
    "es": ["ninguno"], "sv": ["inga"], "fi": ["ei mikään"],
    "cs": ["žádné", "žádná"], "sr": ["nema"], "pt": ["nenhum"],
    "it": ["nessuno"], "fa": ["هیچ کدام"], "sw": ["hakuna"], "nl": ["geen"],
    "st": ["ha ho letho"], "hi": ["कोई नहीं"], "da": ["ingen"],
}


def _clean_text(text: str) -> str:
    return text.strip().lower().replace("\u200c", "").replace(" ", "")


def compare_numbers(lang: str, correct_answer: list, model_answer: str) -> bool:
    inst_lang = lang.split("-")[1] if "-" in lang else lang
    if not model_answer:
        return False
    processed = unicodedata.normalize("NFKC", model_answer)
    for word in NONE_DICT[inst_lang]:
        if word in processed or _clean_text(word) in processed:
            return False
    nums = re.findall(r"\d+", processed)
    nums = [n for n in nums if len(n) > 1]
    nums = list(dict.fromkeys(nums))
    if not nums:
        return False
    try:
        extracted = [int(n) for n in nums]
        correct = [int(x) for x in correct_answer]
    except Exception:
        return False
    return len(extracted) == len(correct) and set(extracted) == set(correct)


def compare_none(lang: str, correct_answer: list, model_answer: str) -> bool:
    inst_lang = lang.split("-")[1] if "-" in lang else lang
    processed = _clean_text(unicodedata.normalize("NFKC", model_answer))
    none_words = [_clean_text(w) for w in NONE_DICT[inst_lang]]
    processed = re.sub(r"\b\d\b", "", processed)
    return any(w in processed for w in none_words)


def score(task: str, lang: str, golds: list, pred: str) -> bool:
    if "niah_none" in task:
        return compare_none(lang, golds, pred)
    return compare_numbers(lang, golds, pred)


def _has_multi_digit(pred: str) -> bool:
    proc = unicodedata.normalize("NFKC", str(pred))
    return any(len(n) > 1 for n in re.findall(r"\d+", proc))


def compare_none_strict(lang: str, correct_answer: list,
                        model_answer: str) -> bool:
    """v4.1 robustness scorer for niah_none. The OFFICIAL compare_none is
    substring-lax in a language-dependent way (zh '无' and ja 'なし' occur
    inside ordinary words), so a compression-degraded output that ALSO emits
    a number can still score 'correct' — deflating measured false-presence
    exactly where damage is expected. Strict = official AND no multi-digit
    number anywhere in the prediction. Used for the scorer-sensitivity
    table only; headline accuracy stays official for comparability."""
    return (compare_none(lang, correct_answer, model_answer)
            and not _has_multi_digit(model_answer))


def compare_numbers_lenient(lang: str, correct_answer: list,
                            model_answer: str) -> bool:
    """v4.1 robustness scorer for niah_single. Official compare_numbers
    voids an otherwise-exact numeric answer if ANY in-language none-word
    appears as a substring (penalizes verbose CJK answers asymmetrically).
    Lenient = exact number-set match, ignoring the none-word veto."""
    if not model_answer:
        return False
    processed = unicodedata.normalize("NFKC", model_answer)
    nums = [n for n in re.findall(r"\d+", processed) if len(n) > 1]
    nums = list(dict.fromkeys(nums))
    if not nums:
        return False
    try:
        extracted = [int(n) for n in nums]
        correct = [int(x) for x in correct_answer]
    except Exception:
        return False
    return len(extracted) == len(correct) and set(extracted) == set(correct)


def score_robust(task: str, lang: str, golds: list, pred: str) -> bool:
    """Secondary scorer for scorer-sensitivity analysis (NEVER the headline
    metric): strict on niah_none, lenient on niah_single."""
    if "niah_none" in task:
        return compare_none_strict(lang, golds, pred)
    return compare_numbers_lenient(lang, golds, pred)


def _scorer_unit_tests():
    assert compare_numbers("en", ["4815162"], "<Answer> 4815162 </Answer>")
    assert compare_numbers("en", ["4815162"], "it is 4815162.")
    assert not compare_numbers("en", ["4815162"], "4815162 and 999")   # extra
    assert not compare_numbers("en", ["4815162"], "none")              # none
    assert not compare_numbers("en", ["4815162"], "no idea")
    assert compare_numbers("hi", ["1234567"], "१२३४५६७")               # Devanagari
    assert compare_none("pl", ["brak"], "Brak takich liczb.")
    assert not compare_none("pl", ["brak"], "1234567")
    # v4: niah_none coverage
    assert compare_none("en", [], "<Answer> none </Answer>")
    assert not compare_none("en", [], "<Answer> 4815162 </Answer>")
    assert compare_none("vi", [], "<Câu trả lời> Không có </Câu trả lời>")
    assert compare_none("ko", [], "없음")
    # v4.1: robustness scorers (official stays the headline metric)
    assert compare_none("zh", [], "无法确定，可能是1234567")     # official: lax
    assert not compare_none_strict("zh", [], "无法确定，可能是1234567")
    assert compare_none_strict("zh", [], "<答案> 无 </答案>")
    assert not compare_numbers("zh", ["1234567"], "无关的数字是1234567")
    assert compare_numbers_lenient("zh", ["1234567"], "无关的数字是1234567")
    assert not compare_numbers_lenient("en", ["1234567"], "none")
    assert not compare_none_strict("ja", [], "その数字は存在しなしい 9999999")
    print("scorer unit tests (official ONERULER logic, v4 extended): PASS")


# ----------------------------------------------------------------------------
# Model / press helpers
# ----------------------------------------------------------------------------
_MODEL, _TOK, _MODEL_TAG = None, None, None


def _get_model(model_tag: str = DEFAULT_MODEL):
    """Load (and cache) the model for `model_tag`. Reloads if the tag changes
    (never happens inside one container run in practice). Logs the RESOLVED
    hub revision so unpinned models (revision=None) can be pinned afterwards."""
    global _MODEL, _TOK, _MODEL_TAG
    if _MODEL is not None and _MODEL_TAG == model_tag:
        return _MODEL, _TOK
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = MODELS[model_tag]
    t0 = time.time()
    kw = dict(revision=spec["revision"]) if spec["revision"] else {}
    _TOK = AutoTokenizer.from_pretrained(spec["hf_id"], **kw)
    assert _TOK.is_fast, f"{model_tag}: fast tokenizer required (offsets)"
    _MODEL = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation=ATTN_IMPL, **kw,
    ).eval()
    _MODEL_TAG = model_tag
    rev = getattr(_MODEL.config, "_commit_hash", "unknown")
    log(f"model loaded: {model_tag} ({spec['hf_id']}) rev={rev} "
        f"({time.time() - t0:.0f}s)"
        + ("" if spec["revision"] else
           f"  <-- UNPINNED: paste rev={rev} into MODELS['{model_tag}']"))
    return _MODEL, _TOK


def _seed_everything(seed: int):
    import random as _r
    import numpy as _np
    import torch as _t
    _r.seed(seed)
    _np.random.seed(seed)
    _t.manual_seed(seed)


def _make_press(name: str, kept: float):
    from kvpress import KnormPress, SnapKVPress, StreamingLLMPress
    ratio = (1.0 - kept) if KVPRESS_RATIO_MEANS_REMOVED else kept
    cls = {"snapkv": SnapKVPress, "streaming_llm": StreamingLLMPress,
           "knorm": KnormPress}[name]
    return cls(compression_ratio=ratio)


def _logits_kw(model) -> dict:
    """{kwarg: 1} to keep only the last position's logits on a long prefill
    (an un-truncated 32K x vocab logits tensor is ~8 GB). Name differs across
    transformers versions; detect from the forward signature."""
    import inspect
    try:
        params = inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return {}
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return {name: 1}
    return {}


def _eos_ids(model, tok) -> set:
    ids = set()
    for src in (getattr(model.generation_config, "eos_token_id", None),
                tok.eos_token_id):
        if src is None:
            continue
        if isinstance(src, (list, tuple)):
            ids.update(int(i) for i in src)
        else:
            ids.add(int(src))
    return ids


def _cell_key(cfg: dict) -> dict:
    """Semantic identity of a cell. Deliberately EXCLUDES `n` so that runs at
    different n share one results file and resume/top-up in place. v4 ADDS
    (model, arm) — all v4 hashes differ from v3's (wipe results first)."""
    return {k: cfg[k] for k in
            ("model", "arm", "lang", "ctx_len", "press", "kept", "task",
             "seed", "toy")}


def _cfg_hash(cfg: dict) -> str:
    return hashlib.sha1(
        json.dumps(_cell_key(cfg), sort_keys=True).encode()).hexdigest()[:16]


def _read_jsonl(path):
    """Parse a results JSONL file, tolerating a truncated FINAL line (a
    container kill or volume commit can land mid-append; v3 would brick
    resume for that cell). A malformed NON-final line is real corruption
    and still raises. NOT used for generated data files — those must stay
    loud."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    rows = []
    for i, l in enumerate(lines):
        if not l.strip():
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                log(f"WARNING {path}: dropping truncated final line "
                    f"({len(l)} chars) — that sample will be re-run")
                continue
            raise
    return rows


# ----------------------------------------------------------------------------
# Generation — joint arm (v3 protocol) and query-agnostic arm (v4 primary)
# ----------------------------------------------------------------------------
def _templated(tok, user_text: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True)


def _generate_joint(model, tok, prompt_text: str, press=None,
                    expect_kept: float = None) -> str:
    """v3 protocol: the WHOLE templated prompt (context + question) is
    prefilled under the press. Attention-window presses (SnapKV) therefore
    see the question -> query-AWARE. Kept as arm='joint' for the contrast.
    v4 change vs v3: add_special_tokens=False (template already carries BOS;
    v3 double-added it — uniform then, uniform now, but now also identical
    to the qa arm's token ids).
    v4.1: expect_kept (first joint sample of each compressed cell) makes
    generate return its cache; post-prefill compressed length is recovered
    as final_cache - (n_generated - 1) and asserted within PRESS_TOL on the
    REAL multilingual prompt. Scorer presses keep a fixed count, so this can
    only fail if library semantics change — which is what asserts are for."""
    import torch
    from contextlib import nullcontext
    prompt = _templated(tok, prompt_text)
    ids = tok(prompt, return_tensors="pt",
              add_special_tokens=False).to("cuda")
    L = ids["input_ids"].shape[1]
    gen_kw = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                  pad_token_id=tok.eos_token_id)
    if expect_kept is not None:
        gen_kw["return_dict_in_generate"] = True
    ctx = press(model) if press is not None else nullcontext()
    with torch.no_grad(), ctx:
        out = model.generate(**ids, **gen_kw)
    if expect_kept is not None:
        seqs = out.sequences
        n_new = seqs.shape[1] - L
        c = int(out.past_key_values.get_seq_length()) - max(n_new - 1, 0)
        ratio = c / L
        log(f"joint press check: prompt kept {ratio:.3f} "
            f"(want {expect_kept})")
        assert abs(ratio - expect_kept) < PRESS_TOL, \
            "joint-arm kept fraction off on real text — abort"
        out = seqs
    return tok.decode(out[0, L:], skip_special_tokens=True)


_CHUNK_OK = True   # latched False if the 4D-mask chunk path errors once


def _qa_question_forward(model, tail_ids, past, k_idx, chunked: bool):
    """Feed the question tokens against the (possibly compressed) context
    cache, at their ORIGINAL positions k_idx..k_idx+q-1 — the same position
    convention transformers' generate uses for decode in the joint arm.

    Why the explicit 4D mask: with a compressed cache (c < k_idx) and
    cache_position starting at k_idx, transformers' auto-built causal mask
    compares cache_position against CACHE indices and would let question
    tokens attend FORWARD within the chunk (non-causal leak). We build the
    correct mask ourselves: attend all cached context keys + causal within
    the question chunk. The stepwise path is ground truth (one token at a
    time needs no mask); smoke asserts chunked == stepwise under a real
    press, and run_batch auto-falls back if the chunk path raises."""
    import torch
    global _CHUNK_OK
    dev = tail_ids.device
    q = tail_ids.shape[1]
    c = int(past.get_seq_length())
    if chunked and _CHUNK_OK:
        try:
            keep = torch.ones(q, c + q, dtype=torch.bool, device=dev)
            keep[:, c:] = torch.tril(
                torch.ones(q, q, dtype=torch.bool, device=dev))
            neg = torch.finfo(model.dtype).min
            mask4d = torch.zeros(q, c + q, dtype=model.dtype, device=dev
                                 ).masked_fill(~keep, neg)[None, None]
            with torch.no_grad():
                out = model(input_ids=tail_ids, past_key_values=past,
                            use_cache=True, attention_mask=mask4d,
                            cache_position=torch.arange(
                                k_idx, k_idx + q, device=dev),
                            **_logits_kw(model))
            return out.logits[:, -1], out.past_key_values
        except Exception as e:
            _CHUNK_OK = False
            log(f"WARNING: chunked question forward failed ({e!r}) — "
                f"falling back to stepwise for the rest of this container")
            # a mid-forward failure may have appended question keys to some
            # layers already — restore the cache to context-only first
            if hasattr(past, "crop") and int(past.get_seq_length()) != c:
                past.crop(c)
    logits = None
    for i in range(q):
        with torch.no_grad():
            out = model(input_ids=tail_ids[:, i:i + 1], past_key_values=past,
                        use_cache=True,
                        cache_position=torch.arange(
                            k_idx + i, k_idx + i + 1, device=dev))
        past = out.past_key_values
        logits = out.logits
    return logits[:, -1], past


def _generate_qa(model, tok, context_text: str, question_text: str,
                 press=None, chunked: bool = None, keep_ids: bool = False):
    """Query-agnostic generation:
      1. templated(context+question) tokenized ONCE (offsets kept);
      2. split at the first token starting at/after the question's char
         offset (a rare template char straddling the seam goes to context —
         harmless);
      3. context prefilled under the press (press NEVER sees the question);
      4. question chunk + greedy decode OUTSIDE the press context, at
         original positions.
    Returns (pred, meta) with meta = {context_tokens, question_tokens,
    compressed_len}. With press=None this must (and in smoke, does) equal
    _generate_joint on context+question exactly."""
    import torch
    from contextlib import nullcontext
    if chunked is None:
        chunked = QA_CHUNKED
    prompt = _templated(tok, context_text + question_text)
    qpos = prompt.rindex(question_text)
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = torch.tensor([enc["input_ids"]], device="cuda")
    starts = torch.tensor([o[0] for o in enc["offset_mapping"]])
    hit = (starts >= qpos).nonzero()
    assert len(hit) > 0, "question offset not found in token offsets"
    k_idx = int(hit[0])
    assert 0 < k_idx < ids.shape[1] - 1, \
        f"degenerate split: k={k_idx} of {ids.shape[1]}"
    ctx_mgr = press(model) if press is not None else nullcontext()
    with torch.no_grad(), ctx_mgr:
        out = model(input_ids=ids[:, :k_idx], use_cache=True,
                    **_logits_kw(model))
    past = out.past_key_values
    c = int(past.get_seq_length())
    last_logits, past = _qa_question_forward(
        model, ids[:, k_idx:], past, k_idx, chunked)
    eos = _eos_ids(model, tok)
    toks = []
    nxt = int(last_logits.argmax(-1))
    pos = ids.shape[1]
    for _ in range(MAX_NEW_TOKENS):
        toks.append(nxt)
        if nxt in eos:
            break
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[nxt]], device="cuda"),
                        past_key_values=past, use_cache=True,
                        cache_position=torch.tensor([pos], device="cuda"))
        past = out.past_key_values
        pos += 1
        nxt = int(out.logits[:, -1].argmax(-1))
    pred = tok.decode(toks, skip_special_tokens=True)
    meta = dict(context_tokens=k_idx,
                question_tokens=int(ids.shape[1] - k_idx),
                compressed_len=c)
    if keep_ids:
        meta["gen_ids"] = toks
    return pred, meta


def _measured_kept_fraction(model, tok, press, n_tokens=4000) -> float:
    import torch
    ids = tok("word " * n_tokens, return_tensors="pt", truncation=True,
              max_length=n_tokens).to("cuda")
    with torch.no_grad(), press(model):
        out = model(**ids, use_cache=True)
    return out.past_key_values.get_seq_length() / ids["input_ids"].shape[1]


# ----------------------------------------------------------------------------
# Question/context split — derived from the data, verified by invariants
# ----------------------------------------------------------------------------
def _common_suffix(a: str, b: str) -> str:
    i, m = 0, min(len(a), len(b))
    while i < m and a[len(a) - 1 - i] == b[len(b) - 1 - i]:
        i += 1
    return a[len(a) - i:] if i else ""


def _common_suffix_all(strs) -> str:
    out = strs[0]
    for s in strs[1:]:
        out = _common_suffix(out, s)
        if not out:
            break
    return out


def _tmpl_ok(cand, pres, span):
    """True iff every pre-tail string ends with  ...cand + (1..span chars).
    This is the structural signature of the TRUE question marker: it is
    followed only by the per-sample key, then the (already stripped) tail.
    Haystack-coincidence candidates fail because their text does not sit
    within `span` chars of the key anchor in EVERY sample."""
    for p in pres:
        lo = max(0, len(p) - span - len(cand))
        end = len(p)
        while True:
            pos = p.rfind(cand, lo, end)
            if pos < 0:
                return False
            rem = len(p) - (pos + len(cand))
            if 1 <= rem <= span:
                break            # found a template-consistent occurrence
            end = pos + len(cand) - 1
    return True


def _derive_question_marker(inputs, max_key_span=64, min_marker=12,
                            max_pairs=12, max_marker_len=200):
    """Derive the per-(language, task) string at which the ONERULER question
    section begins, WITHOUT knowing the language's template.

    Structure of every input: [instr][haystack_i][Qprefix][key_i][tail],
    where only haystack_i and key_i vary across samples (the key is a single
    word — max_key_span bounds its length). Steps:
      1. tail = common suffix of ALL inputs (template after the key).
      2. For pairs with DIFFERENT keys (same-key pairs — detected by a long
         untrimmed common suffix — are skipped), trim 1..max_key_span chars
         off each pre-tail string and collect EVERY re-synchronized common
         suffix >= min_marker as a candidate (deduped by ending). The true
         Qprefix is always among them; so are haystack coincidences (short
         repetitive filler units, e.g. CJK, produce many). v4.1: pairs are
         drawn from indices spread EVENLY across all available samples, not
         just the first 8 — a haystack pool small/ordered enough that a
         handful of early samples happen to share a duplicated passage can
         otherwise contaminate every candidate the search ever sees.
      3. Select the LONGEST candidate suffix that is TEMPLATE-CONSISTENT
         (_tmpl_ok): it must be followed by 1..max_key_span chars (the key)
         at the end of EVERY sample's pre-tail string. Consistency is
         monotone in suffix length, so binary-search each candidate,
         CAPPED at max_marker_len. v4.1: every real template header we have
         ever observed is 60-75 chars; a candidate anywhere near the cap is
         the signature of a duplicated-haystack coincidence (confirmed via
         repro), not a real template — capping forces the search toward the
         true short boundary instead of silently accepting a long
         coincidental run of shared book prose as "the marker".
    Splitting each input at the marker's LAST occurrence yields
    (context, question = marker + key + tail). Hard invariants live in
    get_question_split()."""
    uniq = list(dict.fromkeys(inputs))
    assert len(uniq) >= 3, "need >=3 distinct samples to derive the split"
    tail = _common_suffix_all(uniq)
    assert len(tail) >= 8, f"no common template tail (got {tail!r})"
    pres = [s[: len(s) - len(tail)] for s in uniq]

    # ---- PRIMARY (v4.1c): anchor the split on the haystack-closing tag.
    # The common-suffix search below is fooled whenever the template header
    # is SHORTER than max_key_span: trimming 1..span chars removes the whole
    # header (tag + interrogative + key) at once, after which the samples'
    # shared haystack tails match, stranding the marker in book prose with
    # the header absorbed into the "key" slot. Observed on real zh/ja/ko
    # niah_none, where </文本> through the key is only ~30 chars.
    # Every ONERULER template wraps the haystack in a tag, so: find the LAST
    # closing tag inside the PRE-TAIL string (the tail carries the
    # template's own </Question>/</Answer>, so searching pre-tail isolates
    # the haystack boundary), then take the common PREFIX of the resulting
    # question sections. That prefix is exactly the template header — it
    # stops where the per-sample key first differs. Language-agnostic; no
    # template hardcoding.
    tag_re = re.compile(r"</[^<>\n]{1,32}>")
    qsecs = []
    for s, p in zip(uniq, pres):
        ms = list(tag_re.finditer(p))
        if not ms:
            qsecs = None
            break
        qsecs.append(s[ms[-1].start():])
    if qsecs:
        pref = qsecs[0]
        for q in qsecs[1:]:
            i, lim = 0, min(len(pref), len(q))
            while i < lim and pref[i] == q[i]:
                i += 1
            pref = pref[:i]
            if len(pref) < 8:
                break
        # leave >=1 char of key before the tail, and respect the length cap
        room = min(len(q) - len(tail) - 1 for q in qsecs)
        pref = pref[: max(0, min(room, max_marker_len - 1))]
        if len(pref) >= 8 and all(s.count(pref) == 1 for s in uniq):
            return pref, tail

    # ---- FALLBACK: legacy common-suffix search (templates with no tag) ----
    n_idx = min(len(pres), 40)
    if n_idx <= 8:
        idxs = list(range(n_idx))
    else:
        idxs = sorted(set(round(k * (n_idx - 1) / 7) for k in range(8)))
    pairs = list(itertools.combinations(idxs, 2))
    cands, used = {}, 0          # ending-fingerprint -> longest candidate
    for (i, j) in pairs:
        a, b = pres[i], pres[j]
        if len(_common_suffix(a, b)) >= 4:
            continue  # same key — trimming would run into the question
        found = False
        for da in range(1, max_key_span + 1):
            if da >= len(a):
                break
            aa = a[:-da]
            for db in range(1, max_key_span + 1):
                if db >= len(b):
                    break
                cs = _common_suffix(aa, b[:-db])
                if len(cs) >= min_marker:
                    found = True
                    fp = cs[-min_marker:]
                    if len(cs) > len(cands.get(fp, "")):
                        cands[fp] = cs
        used += int(found)
        if used >= max_pairs or len(cands) > 400:
            break
    assert cands, ("could not align any differing-key sample pair — "
                   "inspect the raw inputs with --stage split-audit")
    best = ""
    for cs in sorted(set(cands.values()), key=len, reverse=True):
        if len(cs) <= len(best):
            break                # sorted desc: nothing shorter can win
        lo_len, hi_len, good = 1, min(len(cs), max_marker_len), 0
        while lo_len <= hi_len:  # longest template-consistent suffix of cs,
                                 # capped — see max_marker_len note above
            mid = (lo_len + hi_len) // 2
            if _tmpl_ok(cs[-mid:], pres, max_key_span):
                good, lo_len = mid, mid + 1
            else:
                hi_len = mid - 1
        if good > len(best):
            best = cs[-good:]
    assert len(best) >= min_marker, (
        f"no template-consistent marker >= {min_marker} chars (best: "
        f"{best!r}). Possible causes: the key exceeds max_key_span="
        f"{max_key_span} chars, or the template is unusual — inspect with "
        f"--stage split-audit before spending GPU money.")
    # v4.1b: ANCHOR ON THE STRUCTURAL BOUNDARY. The common-suffix search
    # legitimately absorbs shared HAYSTACK text whenever samples share book
    # material near the question: niah_single inserts ONE needle at a
    # varying depth into an otherwise identical haystack, so the common
    # suffix runs back to the deepest sample's needle — thousands of chars
    # of book prose (observed: 1313-5603 char "markers", every language).
    # That text would land in the question region, which the qa arm NEVER
    # compresses, exempting a language-varying slice of each document's
    # tail from eviction — a confound directly under the primary
    # cross-language metric. Every ONERULER template closes the haystack
    # with an XML-ish tag (</text>, </文本>, </글>, </Văn bản>, ...), so trim
    # back to the LAST closing tag: language-agnostic, no template
    # hardcoding. Trimming preserves _tmpl_ok (a suffix of a consistent
    # suffix matches at the same end positions) and keeps rindex
    # unambiguous (the closing tag occurs once, after all haystack).
    tags = list(re.finditer(r"</[^<>\n]{1,32}>", best))
    if tags and tags[-1].start() > 0:
        trimmed = best[tags[-1].start():]
        if len(trimmed) >= 8:
            best = trimmed
    assert len(best) >= 8, (
        f"marker collapsed to {best!r} after structural trimming — "
        f"inspect with --stage split-audit")
    assert len(best) < max_marker_len, (
        f"derived marker is {len(best)} chars (cap {max_marker_len}) and "
        f"carries no haystack-closing tag to anchor on. Real template "
        f"headers are 20-80 chars, so this is shared HAYSTACK prose, not "
        f"template: the question region would absorb real document text "
        f"that the qa arm never compresses. Inspect the printed marker "
        f"with --stage split-audit; do NOT sweep.")
    return best, tail


def _split_sample(input_str: str, marker: str):
    qs = input_str.rindex(marker)
    return input_str[:qs], input_str[qs:]


def _lcs(a: str, b: str) -> str:
    """Longest common substring (rolling DP) — used to verify the queried
    key is shared between the needle window and the question section."""
    if not a or not b:
        return ""
    prev = [0] * (len(b) + 1)
    best_len, best_end = 0, 0
    for i, ca in enumerate(a, 1):
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best_len:
                    best_len, best_end = cur[j], i
        prev = cur
    return a[best_end - best_len:best_end]


def get_question_split(samples, task: str, lang: str) -> str:
    """Derive + VERIFY the question marker for one (lang, task) sample set.
    v4.1: derivation uses only the FIRST 40 samples (deterministic order
    from load_samples), so split-audit and run_batch provably derive the
    SAME marker for the same data regardless of n; a marker hash is logged
    in both for eyeball comparison. Invariants are verified over ALL
    samples passed.
    Hard invariants (any failure aborts before GPU money is spent):
      * marker present in every input; context >= 200 chars; question length
        sane; question = marker + key + tail with 0 < len(key) <= 64 for
        EVERY sample (catches markers that re-occur inside the tail, and
        template shapes the derivation cannot represent);
      * niah_single: every gold value in the CONTEXT and NEVER in the
        question (the split may not leak the needle, nor orphan it);
      * niah_single: the queried key must be shared between the needle
        window and the question (LCS >= 3 chars, checked on 5 samples);
      * MID-QUESTION guard (both tasks): if the template mentions the key
        TWICE, derivation lands between the mentions and the first mention —
        including the key — leaks into the compressed span (SnapKV would be
        query-aware again). Detector: the extracted key sitting IMMEDIATELY
        before the split point. A needle can legitimately sit there only at
        depth ~1, so those samples are excluded and the pattern must be
        systematic (>=3 of 5).
    Rarity of the shared key in the haystack is logged as a warning only
    (book text may repeat common words)."""
    inputs = [s["input"] for s in samples[:40]]
    marker, tail = _derive_question_marker(inputs)
    mh = hashlib.sha1(marker.encode()).hexdigest()[:8]
    qlens, leak_hits, audited = [], 0, 0
    for s in samples:
        ctx, q = _split_sample(s["input"], marker)
        qlens.append(len(q))
        assert len(ctx) >= 200, f"{lang}/{task}: context too short after split"
        assert len(marker) <= len(q) <= 8000, \
            f"{lang}/{task}: implausible question length {len(q)}"
        key_s = q[len(marker): len(q) - len(tail)]
        assert 0 < len(key_s) <= 64, \
            (f"{lang}/{task}: question is not marker+key+tail "
             f"(key={key_s[:80]!r}) — marker unusable, run --stage "
             f"split-audit")
        if task == "niah_single":
            for g in s["outputs"]:
                assert g in ctx, f"{lang}: gold {g!r} not in context part"
                assert g not in q, f"{lang}: gold {g!r} leaked into question"
        if audited < 5 and s["depth"] < 0.9:
            audited += 1
            if key_s in ctx[-(len(key_s) + 16):]:
                leak_hits += 1
    assert leak_hits < 3, \
        (f"{lang}/{task}: the queried key sits immediately before the split "
         f"point in {leak_hits}/{audited} low-depth samples — the marker is "
         f"MID-QUESTION (template repeats the key) and the first mention "
         f"leaks into the compressed span. Do NOT sweep; run --stage "
         f"split-audit.")
    if task == "niah_single":
        for s in samples[:5]:
            ctx, q = _split_sample(s["input"], marker)
            g = s["outputs"][0]
            pos = ctx.find(g)
            win = ctx[max(0, pos - 90): pos + len(g) + 30]
            shared = _lcs(win, q)
            assert len(shared) >= 3, \
                (f"{lang}: no shared key between needle window and question "
                 f"— split looks wrong, run --stage split-audit")
            if ctx.count(shared) > 8:
                log(f"{lang}: key-check substring {shared!r} is common in the"
                    f" haystack ({ctx.count(shared)}x) — eyeball split-audit")
    log(f"split[{lang}/{task}]: marker#{mh} {marker[-60:]!r} "
        f"question chars min/med/max = {min(qlens)}/"
        f"{sorted(qlens)[len(qlens)//2]}/{max(qlens)}")
    return marker


# ----------------------------------------------------------------------------
# Data: ONERULER loader (+ distractor capture) and toy generator (+ none)
# ----------------------------------------------------------------------------
def data_path(lang: str, ctx_len: int, task: str,
              seed: int = DEFAULT_SEED) -> str:
    """v4.1: the path is SEED-AWARE (non-default seeds land in
    <task>__seed<seed>/), so gen-data's skip check can never silently reuse
    data from a different seed while _cfg_hash claims the new one. seed=42
    keeps the legacy path for back-compatibility with existing volumes."""
    name = task if seed == DEFAULT_SEED else f"{task}__seed{seed}"
    return f"{DATA_DIR}/oneruler/{lang}/{ctx_len}/{name}/validation.jsonl"


def load_samples(lang, ctx_len, n, task=TASK, toy=False, seed=DEFAULT_SEED):
    """-> [{sample_id, input, outputs, depth, length, distractors?}],
    deterministic order.

    niah_single: asserts every gold occurs in the input (guards silent
    generation/truncation bugs) and recovers needle depth as first-occurrence
    char offset / input length.
    niah_none (v4): the queried key is absent, gold = the in-language
    none-word; 4 DISTRACTOR needles are present. We capture their numbers
    (>=6-digit runs; ONERULER values are 7-digit, the floor avoids years in
    book text) so failure_modes can split false-presence into
    'retrieved a real distractor' vs 'hallucinated a number'."""
    if toy:
        return _make_toy_samples(lang, ctx_len, n, seed,
                                 none=("niah_none" in task))
    p = Path(data_path(lang, ctx_len, task, seed))
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — run: modal run <file> --stage gen-data "
            f"--task {task} --langs {lang} --ctx-len {ctx_len} "
            f"--seed {seed} --n <N>")
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if len(rows) < n:
        log(f"WARNING {lang}: only {len(rows)} samples on disk, need {n} — "
            f"regenerate with a larger --n")
    out, ambiguous = [], 0
    for r in rows[:n]:
        inp = r["input"]
        golds = [str(o) for o in r["outputs"]]
        depth, distractors = -1.0, None
        if task == "niah_single":
            pos = inp.find(golds[0])
            if pos < 0:
                raise AssertionError(
                    f"INVARIANT VIOLATION: gold {golds[0]!r} not found in "
                    f"input (lang={lang}, index={r['index']}) — generation "
                    f"or truncation bug; regenerate this language.")
            depth = round(pos / max(len(inp), 1), 4)
            if inp.count(golds[0]) > 1:
                ambiguous += 1
        elif task == "niah_none":
            distractors = sorted(set(re.findall(r"\d{6,}", inp)))
            if not distractors:
                log(f"WARNING {lang}/niah_none idx={r['index']}: no "
                    f"distractor numbers found — check generation flags")
        rec = {"sample_id": r["index"], "input": inp, "outputs": golds,
               "depth": depth, "length": r.get("length", -1)}
        if distractors is not None:
            rec["distractors"] = distractors
        out.append(rec)
    if task == "niah_single":
        log(f"{lang}: depth recovered for {len(out)} samples "
            f"({ambiguous} with multiple gold occurrences — first used)")
    return out


def _make_toy_samples(lang, ctx_len, n, seed, none=False):
    """Synthetic English NIAH mimicking ONERULER's needle phrasing, so the
    OFFICIAL scorer applies. Explicit depths. none=True (v4): the question
    asks about a key that is NOT the inserted needle's key -> gold 'none';
    the inserted needle's value doubles as the recorded distractor."""
    import random
    _, tok = _get_model(_MODEL_TAG or DEFAULT_MODEL)
    filler = ("The sky above the harbor was the color of television, tuned to "
              "a dead channel. Merchants argued about grain prices while "
              "gulls circled the masts and children counted the boats. ")
    rng = random.Random(seed)
    keys = ["lantern", "compass", "anchor", "harbor"]
    unit = len(tok(filler)["input_ids"])
    reps = -(-n // len(TOY_DEPTHS))
    depth_seq = (TOY_DEPTHS * reps)[:n]   # exactly n; all depths covered
    samples = []
    for sid, depth in enumerate(depth_seq):
        code = rng.randrange(10**6, 10**7)
        key = rng.choice(keys)
        others = [k for k in keys if k != key]
        q_key = others[sid % len(others)] if none else key
        needle = f' The special magic number for "{key}" is: {code}. '
        n_units = max(4, (ctx_len - 250) // unit)
        pos = int(n_units * depth)
        hay = filler * pos + needle + filler * (n_units - pos)
        prompt = ("Please read and memorize the text below. I will ask "
                  "you about it later.\n\n<text>\n" + hay + "\n</text>\n\n"
                  f'<Question> What special magic numbers associated with '
                  f'"{q_key}" are mentioned in the provided text? Please '
                  f'list all that apply. If no such numbers exist, please '
                  f'answer "none".</Question>\n\nPlease provide your '
                  f"answer in the following format:\n<Answer> List all "
                  f"numbers here </Answer>")
        rec = {"sample_id": sid, "input": prompt,
               "outputs": (["none"] if none else [str(code)]),
               "depth": depth, "length": -1}
        if none:
            rec["distractors"] = [str(code)]
        samples.append(rec)
    return samples


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------
@app.function(**CPU_KW, timeout=15 * 60)
def prepare_data():
    """Clone the ONERULER repo to the volume (idempotent)."""
    log_open("prepare_data")
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(f"{REPO_DIR}/OneRuler/synthetic/niah.py"):
        log(f"repo already present at {REPO_DIR}")
    else:
        log(f"cloning {ONERULER_GIT} -> {REPO_DIR}")
        subprocess.run(["git", "clone", "--depth", "1", ONERULER_GIT,
                        REPO_DIR], check=True)
        log("clone OK")
    vol.commit()
    log("next: --stage gen-data --task <task> --langs <codes> --n <N>")


@app.function(**CPU_KW, timeout=8 * 3600)
def gen_data(langs: list[str], ctx_len: int, n: int, seed: int, task: str):
    """Generate ONERULER data by invoking synthetic/niah.py directly.
    Data is tokenized/length-matched with the LLAMA tokenizer for BOTH
    models (matched TEXT, model-native tokenization at run time — budgets
    are fractional, and smoke's tokcheck asserts fit in each model)."""
    log_open("gen_data")
    assert os.path.exists(f"{REPO_DIR}/OneRuler/synthetic/niah.py"), \
        "repo missing — run --stage prepare-data first"
    os.makedirs(os.environ["STANZA_RESOURCES_DIR"], exist_ok=True)
    bad = [l for l in langs if l not in ONERULER_ALL_LANGS]
    assert not bad, f"not ONERULER languages: {bad}"
    task_args = {  # from config/synthetic.yaml
        "niah_single": ["--num_needle_k", "1", "--num_needle_v", "1",
                        "--num_needle_q", "1"],
        "niah_none": ["--num_needle_k", "4", "--num_needle_v", "1",
                      "--num_needle_q", "1", "--relevant_needle", "0"],
    }[task]
    save_name = task if seed == DEFAULT_SEED else f"{task}__seed{seed}"
    t0 = time.time()
    for i, lang in enumerate(langs):
        out_file = Path(data_path(lang, ctx_len, task, seed))
        if out_file.exists() and len(out_file.read_text(
                encoding="utf-8").splitlines()) >= n:
            log(f"{bar(i + 1, len(langs), t0)} {lang}: exists for "
                f"seed={seed}, skip")
            continue
        save_dir = str(out_file.parent.parent)  # niah.py appends /{name}/
        os.makedirs(save_dir, exist_ok=True)
        cmd = ["python", f"{REPO_DIR}/OneRuler/synthetic/niah.py",
               "--save_dir", save_dir, "--save_name", save_name,
               "--subset", "validation", "--lang", lang,
               "--tokenizer_path", MODELS[DEFAULT_MODEL]["hf_id"],
               "--tokenizer_type", "hf",
               "--max_seq_length", str(ctx_len),
               "--tokens_to_generate", "30",       # constants.py: niah
               "--num_samples", str(n),
               "--random_seed", str(seed),
               "--type_haystack", "book",
               "--type_needle_k", "words", "--type_needle_v", "numbers",
               ] + task_args
        log(f"--- {lang}: generating task={task} n={n} ctx={ctx_len} "
            f"seed={seed}")
        log("    cmd: " + " ".join(cmd))
        r = subprocess.run(cmd, capture_output=True, text=True)
        for line in (r.stdout + "\n" + r.stderr).splitlines():
            if line.strip():
                log(f"    [niah.py] {line[:160]}")
        assert r.returncode == 0, f"generation failed for {lang}"
        assert out_file.exists(), f"{out_file} not produced"
        rows = [json.loads(l) for l in
                out_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        lens = [row["length"] for row in rows]
        log(f"{bar(i + 1, len(langs), t0)} {lang}: {len(rows)} samples, "
            f"token length min/mean/max = {min(lens)}/"
            f"{sum(lens) / len(lens):.0f}/{max(lens)}")
        vol.commit()
    log(f"gen-data DONE ({task}) for all languages")


@app.function(**CPU_KW, timeout=30 * 60)
def split_audit(langs: list[str], ctx_len: int, task: str, n: int,
                seed: int):
    """Derive + verify the context/question split for every language and
    PRINT the marker (+hash) and an example question section for a human
    eyeball. Pure text, no GPU. This is the GATE before any qa-arm sweep.
    Derivation uses the first 40 samples, so the marker (and its logged
    hash) is provably identical to run_batch's for the same data."""
    log_open("split_audit")
    ok, bad = [], []
    for lang in langs:
        try:
            samples = load_samples(lang, ctx_len, min(n, 40), task=task,
                                   seed=seed)
            marker = get_question_split(samples, task, lang)
            ctx, q = _split_sample(samples[0]["input"], marker)
            log(f"--- {lang}/{task} PASS")
            log(f"    marker(repr, last 80): {marker[-80:]!r}")
            preview = min(len(q), 2000)
            log(f"    example question (first {preview} chars): "
                f"{q[:preview]!r}")
            log(f"    example context tail (last 120): {ctx[-120:]!r}")
            ok.append(lang)
        except Exception as e:
            log(f"--- {lang}/{task} FAIL: {e}")
            bad.append(lang)
    log(f"split-audit: PASS={ok} FAIL={bad}")
    assert not bad, f"split derivation failed for: {bad}"
    log("EYEBALL the printed question sections (they must contain the "
        "interrogative + queried key + answer-format tail) before sweeping.")


@app.function(**GPU_KW, timeout=45 * 60)
def smoke(model_tag: str):
    """v4 smoke: every new mechanism is exercised on toy data BEFORE any
    real GPU spend (12 checks). Hard asserts = plumbing; behavioral checks
    that depend on the model retrieving from a ~3K toy context are also
    asserted (an 8B instruct model is reliable there) but each carries a
    diagnostic message. On an exact-equality failure ([4]/[8]) the top-2
    logit margin at the first divergent step is reported, separating a
    plumbing bug (large margin) from bf16 near-tie nondeterminism (~0
    margin — kernel tiling can differ with q_len, so bitwise equality of a
    split prefill vs a fused one is a property of the stack, not a law)."""
    log_open(f"smoke_{model_tag}")
    _seed_everything(DEFAULT_SEED)
    model, tok = _get_model(model_tag)
    _scorer_unit_tests()

    def _diverge_report(tag, sample_input, qa_pred, qa_ids, joint_pred):
        import torch
        prompt = _templated(tok, sample_input)
        ids = tok(prompt, return_tensors="pt",
                  add_special_tokens=False).to("cuda")
        with torch.no_grad():
            g = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS,
                               do_sample=False,
                               pad_token_id=tok.eos_token_id,
                               return_dict_in_generate=True,
                               output_scores=True)
        j_ids = g.sequences[0, ids["input_ids"].shape[1]:].tolist()
        i = next((k for k, (a, b) in enumerate(zip(qa_ids, j_ids))
                  if a != b), min(len(qa_ids), len(j_ids)))
        margin = float("nan")
        if i < len(g.scores):
            top2 = torch.topk(g.scores[i][0].float(), 2).values
            margin = float(top2[0] - top2[1])
        return (f"[{tag}] qa != joint. First divergence at step {i}; "
                f"joint top-2 logit margin there = {margin:.4f}. "
                f"Margin >> 0.05 -> split/mask/position plumbing bug, do "
                f"not sweep. Margin ~ 0 -> bf16 near-tie argmax "
                f"nondeterminism between fused and split prefill (rerun "
                f"smoke; if it persists only at tiny margins, treat as "
                f"numerics, not plumbing).\n"
                f" qa   ={qa_pred!r}\n joint={joint_pred!r}")

    # [1] press semantics on a flat probe (v3 check, kept verbatim)
    for name in PRESSES:
        press = _make_press(name, kept=0.5)
        kept = _measured_kept_fraction(model, tok, press)
        log(f"[1] {name}: requested kept=0.50 measured kept={kept:.3f}")
        assert abs(kept - 0.5) < PRESS_TOL, (
            f"{name}: measured kept={kept:.3f} outside +-{PRESS_TOL} of 0.5. "
            f"Either kvpress compression_ratio semantics differ (flip "
            f"KVPRESS_RATIO_MEANS_REMOVED) or this press has structural "
            f"offsets — investigate before sweeping.")

    # toy sets: niah_single at explicit depths + niah_none
    toys = _make_toy_samples("en", 3000, 6, DEFAULT_SEED)
    toys_none = _make_toy_samples("en", 3000, 3, DEFAULT_SEED, none=True)

    # [2] joint baseline generates + scores (v3 check)
    s_mid = next(t for t in toys if t["depth"] == 0.5)
    full = _generate_joint(model, tok, s_mid["input"])
    log(f"[2] joint full-cache answer: {full[:90]!r} | score="
        f"{score('niah_single', 'en', s_mid['outputs'], full)}")
    assert score("niah_single", "en", s_mid["outputs"], full), \
        "joint full-cache toy retrieval failed — model/template problem"

    # [3] split derivation on the toy sets + invariants
    marker = get_question_split(toys, "niah_single", "en")
    assert "<Question" in marker or "</text>" in marker, \
        f"toy marker unexpected: {marker!r}"
    marker_none = get_question_split(toys_none, "niah_none", "en")
    log(f"[3] toy markers OK: {marker[-40:]!r} / {marker_none[-40:]!r}")

    # [4] qa(press=None) must EXACTLY equal joint on the same sample
    ctx_t, q_t = _split_sample(s_mid["input"], marker)
    qa_pred, meta = _generate_qa(model, tok, ctx_t, q_t, press=None,
                                 keep_ids=True)
    if qa_pred != full:
        raise AssertionError(_diverge_report(
            "4", s_mid["input"], qa_pred, meta["gen_ids"], full))
    log(f"[4] qa(None) == joint EXACTLY "
        f"(k={meta['context_tokens']} q={meta['question_tokens']})")

    # [5] chunked vs stepwise question forward UNDER COMPRESSION must match
    press = _make_press("streaming_llm", kept=0.5)
    p_chunk, m_chunk = _generate_qa(model, tok, ctx_t, q_t, press=press,
                                    chunked=True)
    p_step, m_step = _generate_qa(model, tok, ctx_t, q_t, press=press,
                                  chunked=False)
    assert p_chunk == p_step, (
        f"[5] chunked != stepwise under compression:\n chunk={p_chunk!r}\n"
        f" step ={p_step!r}\n-> 4D mask / cache_position bug.")
    assert m_chunk["compressed_len"] == m_step["compressed_len"]
    log(f"[5] chunked == stepwise under streaming@0.5 "
        f"(c={m_chunk['compressed_len']}/{m_chunk['context_tokens']})")

    # [6] qa positional behavior: streaming@0.5 keeps depth-0.9, evicts 0.1
    s_hi = next(t for t in toys if t["depth"] == 0.9)
    s_lo = next(t for t in toys if t["depth"] == 0.1)
    for s, want in ((s_hi, True), (s_lo, False)):
        c_txt, q_txt = _split_sample(s["input"], marker)
        pred, meta = _generate_qa(model, tok, c_txt, q_txt,
                                  press=_make_press("streaming_llm", 0.5))
        got = score("niah_single", "en", s["outputs"], pred)
        log(f"[6] qa streaming@0.5 depth={s['depth']}: pred={pred[:60]!r} "
            f"correct={got} (expect {want})")
        assert got == want, (
            f"[6] positional behavior wrong at depth {s['depth']} — kept "
            f"window/split interaction broken (pred={pred!r}).")
        assert abs(meta["compressed_len"] / meta["context_tokens"] - 0.5) \
            < PRESS_TOL, "[6] qa-arm measured kept fraction off"

    # [7] qa + snapkv@0.5: compresses the CONTEXT only, still answers
    pred, meta = _generate_qa(model, tok, ctx_t, q_t,
                              press=_make_press("snapkv", 0.5))
    assert abs(meta["compressed_len"] / meta["context_tokens"] - 0.5) \
        < PRESS_TOL, "[7] snapkv qa kept fraction off"
    log(f"[7] qa snapkv@0.5 answer: {pred[:70]!r} | score="
        f"{score('niah_single', 'en', s_mid['outputs'], pred)}")

    # [8] niah_none toy: baseline should answer the none-word (both arms)
    sn = toys_none[0]
    pj = _generate_joint(model, tok, sn["input"])
    cn_txt, qn_txt = _split_sample(sn["input"], marker_none)
    pq, mq = _generate_qa(model, tok, cn_txt, qn_txt, press=None,
                          keep_ids=True)
    log(f"[8] niah_none baseline joint={pj[:60]!r} qa={pq[:60]!r}")
    assert score("niah_none", "en", sn["outputs"], pj), (
        "[8] joint baseline failed toy niah_none (should say 'none') — "
        f"pred={pj!r}; check template/model behavior before the none sweep.")
    if pq != pj:
        raise AssertionError(_diverge_report(
            "8", sn["input"], pq, mq["gen_ids"], pj))

    # [9] knorm diagnosis machinery on a toy: qa replay (primary arm) —
    #     spans found, exact keep fraction, question fully protected
    d = _knorm_diagnose_sample(model, tok, s_mid,
                               marker, kept=0.5, task="niah_single",
                               arm="qa")
    log(f"[9] diagnose toy (qa replay): keep ctx={d['keep_overall']:.3f} "
        f"value={d['keep_value']:.3f} question={d['keep_question']:.3f} "
        f"haystack={d['keep_haystack']:.3f}")
    assert abs(d["keep_overall"] - 0.5) < 0.01, \
        "[9] offline knorm selection does not hit the requested fraction"
    assert d["keep_question"] == 1.0, \
        "[9] qa replay must protect the question span entirely"
    assert d["n_value_tokens"] > 0 and d["n_question_tokens"] > 0, \
        "[9] diagnosis spans empty"

    # [10] tokcheck: real data (if generated) fits this model's window
    spec = MODELS[model_tag]
    for lang in LANGS[:3]:
        p = Path(data_path(lang, CTX_LEN, "niah_single"))
        if not p.exists():
            log(f"[10] tokcheck {lang}: no data yet (gen-data later) — skip")
            continue
        row = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
        n_tok = len(tok(_templated(tok, row["input"]),
                        add_special_tokens=False)["input_ids"])
        log(f"[10] tokcheck {lang}: {n_tok} tokens under {model_tag} "
            f"(limit {spec['max_len']})")
        assert n_tok + MAX_NEW_TOKENS + 8 <= spec["max_len"], \
            f"{lang} overflows {model_tag}'s window — pick a longer-ctx model"

    # [11] joint decode positions under compression: transformers must carry
    #      cache_position INCREMENTALLY (verified in the 5.2.0 source:
    #      generation/utils.py sets cache_position[-1:] + num_new_tokens),
    #      never re-derive it from the compressed cache. If it re-derived,
    #      joint decode would continue at position c (~kept*L) while the qa
    #      arm continues at L, and fig5's arm contrast would silently
    #      include a RoPE position shift. Asserted here against pin drift.
    import torch
    positions = []

    def _pos_hook(_m, _args, kwargs):
        cp = kwargs.get("cache_position")
        if cp is not None:
            positions.append((int(cp[0]), int(cp[-1]), int(cp.shape[0])))

    hd = model.register_forward_pre_hook(_pos_hook, with_kwargs=True)
    try:
        prompt11 = _templated(tok, s_mid["input"])
        jids = tok(prompt11, return_tensors="pt",
                   add_special_tokens=False).to("cuda")
        L11 = jids["input_ids"].shape[1]
        with torch.no_grad(), _make_press("streaming_llm", 0.5)(model):
            model.generate(**jids, max_new_tokens=3, do_sample=False,
                           pad_token_id=tok.eos_token_id)
    finally:
        hd.remove()
    assert positions and positions[0] == (0, L11 - 1, L11), \
        f"[11] prefill cache_position unexpected: {positions[:1]} (L={L11})"
    assert len(positions) >= 2 and positions[1] == (L11, L11, 1), (
        f"[11] first decode step at cache_position {positions[1]}, expected "
        f"({L11}, {L11}, 1): transformers is re-deriving positions from the "
        f"COMPRESSED cache — joint decode would be position-shifted and the "
        f"arm contrast invalid. Do NOT sweep on this stack.")
    log(f"[11] joint decode continues at original position {L11} under "
        f"compression (cache_position carried incrementally)")

    # [12] joint positional behavior (twin of [6]) + on-text ratio check —
    #      also exercises run_batch's per-cell joint press verification path
    for s, want in ((s_hi, True), (s_lo, False)):
        pj2 = _generate_joint(model, tok, s["input"],
                              press=_make_press("streaming_llm", 0.5),
                              expect_kept=0.5)
        got = score("niah_single", "en", s["outputs"], pj2)
        log(f"[12] joint streaming@0.5 depth={s['depth']}: "
            f"pred={pj2[:60]!r} correct={got} (expect {want})")
        assert got == want, (
            f"[12] JOINT positional behavior wrong at depth {s['depth']} "
            f"(pred={pj2!r}) — joint-side position/keep-window handling "
            f"broken.")
    vol.commit()
    log(f"SMOKE PASS ({model_tag}) — if this model was unpinned, paste the "
        f"logged revision into MODELS and record in DECISIONS.md")


@app.function(**GPU_KW, timeout=6 * 3600, retries=1)
def run_batch(cells: list[dict]):
    """Run cells (one language's settings for one model+arm+task).
    Resumable, seeded, progress-barred; per-sample rows -> raw/{hash}.jsonl
    (n-independent hash: higher-n runs top up the same file);
    summary -> cells.jsonl."""
    c0 = cells[0]
    log_open(f"batch_{c0['lang']}_{c0['model']}_{c0['arm']}_{c0['task']}")
    model_tag = c0["model"]
    assert all(c["model"] == model_tag for c in cells), "mixed models in batch"
    model, tok = _get_model(model_tag)
    os.makedirs(RAW_DIR, exist_ok=True)
    markers = {}   # (lang, task, toy) -> derived question marker (qa arm)
    for ci, cfg in enumerate(cells):
        _seed_everything(cfg["seed"])
        h = _cfg_hash(cfg)
        raw = Path(f"{RAW_DIR}/{h}.jsonl")
        done = set()
        if raw.exists():
            done = {r["sample_id"] for r in _read_jsonl(raw)}
        samples = load_samples(cfg["lang"], cfg["ctx_len"], cfg["n"],
                               task=cfg["task"], toy=cfg["toy"],
                               seed=cfg["seed"])
        todo = [s for s in samples if s["sample_id"] not in done]
        press = (None if cfg["press"] == "none"
                 else _make_press(cfg["press"], cfg["kept"]))
        if press is not None:
            kept = _measured_kept_fraction(model, tok, press)
            log(f"press check: want kept={cfg['kept']} measured={kept:.3f}")
            assert abs(kept - cfg["kept"]) < PRESS_TOL, \
                "press semantics drifted beyond tolerance"
        marker = None
        if cfg["arm"] == "qa":
            mkey = (cfg["lang"], cfg["task"], cfg["toy"])
            if mkey not in markers:
                markers[mkey] = get_question_split(samples, cfg["task"],
                                                   cfg["lang"])
            marker = markers[mkey]
        log(f"=== cell {ci + 1}/{len(cells)} [{h}] {_cell_key(cfg)} "
            f"n={cfg['n']} | resume: {len(done)} done, {len(todo)} to run")
        t0, ok_run, qa_checked = time.time(), 0, False
        joint_checked = False
        with raw.open("a", encoding="utf-8") as f:
            for k, s in enumerate(todo, 1):
                ts = time.time()
                meta = {}
                if cfg["arm"] == "qa":
                    ctx_txt, q_txt = _split_sample(s["input"], marker)
                    pred, meta = _generate_qa(model, tok, ctx_txt, q_txt,
                                              press=press)
                    if press is not None and not qa_checked:
                        ratio = (meta["compressed_len"]
                                 / meta["context_tokens"])
                        log(f"qa press check: context kept {ratio:.3f} "
                            f"(want {cfg['kept']})")
                        assert abs(ratio - cfg["kept"]) < PRESS_TOL, \
                            "qa-arm context kept fraction off — abort"
                        qa_checked = True
                else:
                    # v4.1: verify the press on the first REAL prompt of the
                    # cell for the joint arm too (qa gets the same above)
                    pred = _generate_joint(
                        model, tok, s["input"], press=press,
                        expect_kept=(cfg["kept"] if press is not None
                                     and not joint_checked else None))
                    joint_checked = True
                ok = score(cfg["task"], cfg["lang"], s["outputs"], pred)
                ok_run += int(ok)
                row = {
                    **_cell_key(cfg),
                    "config_hash": h, "sample_id": s["sample_id"],
                    "depth": s["depth"], "input_length": s["length"],
                    "correct": bool(ok), "pred": pred[:300],
                    "gold": s["outputs"],
                    "wall_s": round(time.time() - ts, 2),
                    "model_id": MODELS[model_tag]["hf_id"],
                    "ts": int(time.time()), **meta,
                }
                if "distractors" in s:
                    row["distractors"] = s["distractors"]
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if k % 10 == 0 or k == len(todo):
                    f.flush()
                    vol.commit()
                    log(bar(k, len(todo), t0,
                            f"| sess-acc {ok_run / k:.2f} "
                            f"| {(time.time() - t0) / k:.1f}s/sample"))
        rows = _read_jsonl(raw)
        acc = sum(r["correct"] for r in rows) / len(rows)
        summary = {**_cell_key(cfg), "config_hash": h, "n_done": len(rows),
                   "acc": round(acc, 4),
                   "mean_wall_s": round(sum(r["wall_s"] for r in rows)
                                        / len(rows), 2),
                   "finished_ts": int(time.time())}
        os.makedirs(CELLS_DIR, exist_ok=True)
        cpath = (f"{CELLS_DIR}/{c0['model']}__{c0['arm']}__{c0['task']}"
                 f"__{c0['lang']}.jsonl")
        with open(cpath, "a", encoding="utf-8") as cf:
            cf.write(json.dumps(summary, ensure_ascii=False) + "\n")
        vol.commit()
        log(f"CELL DONE [{h}] acc={acc:.3f} n={len(rows)}")
    return (f"{c0['lang']}/{model_tag}/{c0['arm']}/{c0['task']}: "
            f"{len(cells)} cells done")


# FLORES+ (openlanguagedata/flores_plus) renamed some FLORES-200 configs. For
# our language set only zh is affected: zho_Hans -> cmn_Hans.
FLORES_PLUS_CONFIG_ALIASES = {"zho_Hans": "cmn_Hans", "zho_Hant": "cmn_Hant"}


@app.function(**CPU_KW, timeout=20 * 60)
def fertility(model_tag: str, langs: list[str]):
    """Tokenizer fertility (tokens/byte, tokens/char) PER MODEL, from two
    sources:
      * FLORES+ devtest (openlanguagedata/flores_plus) — broad language
        coverage incl. langs without generated data. The legacy
        script-based fallbacks (facebook/flores, Muennighoff/flores200)
        were removed in v4.1: script datasets are unloadable under
        datasets>=3, so they were dead code giving false comfort.
      * v4.1: the ACTUAL evaluation texts (generated niah_single inputs at
        CTX_LEN, default seed) -> tokens_per_byte_eval — same domain as the
        experiment, so analyze/fig3 PREFERS these columns when complete.
    Writes results/fertility_<model>.csv (top-up merge, v3 semantics)."""
    log_open(f"fertility_{model_tag}")
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import pandas as pd
    spec = MODELS[model_tag]
    kw = dict(revision=spec["revision"]) if spec["revision"] else {}
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], **kw)
    rows, skipped, t0 = [], [], time.time()

    def _rates(texts):
        enc = tok([str(t) for t in texts],
                  add_special_tokens=False)["input_ids"]
        n_tok = sum(len(e) for e in enc)
        n_b = sum(len(str(t).encode("utf-8")) for t in texts)
        n_c = sum(len(str(t)) for t in texts)
        return round(n_tok / n_b, 5), round(n_tok / n_c, 5)

    for i, lang in enumerate(langs):
        flores = FLORES_PLUS_CONFIG_ALIASES.get(LANG_TO_FLORES[lang],
                                                LANG_TO_FLORES[lang])
        row = {"lang": lang, "flores": flores}
        try:
            ds = load_dataset("openlanguagedata/flores_plus", flores,
                              split="devtest")
            texts = list(ds["text"])
            tpb, tpc = _rates(texts)
            row.update(tokens_per_byte=tpb, tokens_per_char=tpc,
                       n_sentences=len(texts))
            log(f"{lang}: FLORES+ {len(texts)} sentences ({flores})")
        except Exception as e:
            log(f"{lang}: openlanguagedata/flores_plus:{flores} failed "
                f"({str(e)[:140]}) — no fallback exists (script datasets "
                f"are unloadable under datasets>=3); fix flores_plus "
                f"access (accept terms + HF_TOKEN)")
        dp = Path(data_path(lang, CTX_LEN, "niah_single"))
        if dp.exists():
            inputs = [json.loads(l)["input"] for l in
                      dp.read_text(encoding="utf-8").splitlines()
                      if l.strip()][:24]
            tpb, tpc = _rates(inputs)
            row.update(tokens_per_byte_eval=tpb, tokens_per_char_eval=tpc,
                       n_eval_docs=len(inputs))
            log(f"{lang}: eval-text fertility from {len(inputs)} generated "
                f"inputs (tokens/byte={tpb})")
        if len(row) <= 2:
            skipped.append(lang)
            log(f"{lang}: SKIPPED — neither FLORES+ nor generated eval data")
            continue
        rows.append(row)
        log(f"{bar(i + 1, len(langs), t0)} {row}")
    if not rows:
        log("FATAL: every language SKIPPED — NOT writing fertility csv. "
            "Fix flores_plus access (accept terms + HF_TOKEN), rerun.")
        vol.commit()
        return
    os.makedirs(f"{VOL}/results", exist_ok=True)
    out = pd.DataFrame(rows)
    fp = Path(f"{VOL}/results/fertility_{model_tag}.csv")
    if fp.exists() and fp.stat().st_size > 0:
        try:
            out = (pd.concat([pd.read_csv(fp), out])
                   .drop_duplicates(subset="lang", keep="last"))
        except pd.errors.EmptyDataError:
            pass
    out.to_csv(fp, index=False)
    if skipped:
        log(f"WARNING: wrote {len(rows)} langs; still missing {skipped}")
    vol.commit()
    log(f"wrote {fp} ({len(out)} languages total)")


# ----------------------------------------------------------------------------
# KnormPress eviction diagnosis (change 4 of the review)
# ----------------------------------------------------------------------------
def _capture_key_norms(model, ids):
    """Forward `ids` once, capturing per-layer, per-KV-head key L2 norms via
    k_proj forward hooks. RoPE rotates 2D pairs (orthogonal), so pre-RoPE
    k_proj norms equal the cached post-RoPE key norms KnormPress scores.
    Returns list[layer] of float32 cpu tensors (seq, n_kv_heads)."""
    import torch
    cfg = model.config
    n_kv = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", None) or \
        cfg.hidden_size // cfg.num_attention_heads
    norms, handles = [], []

    def _mk():
        def hook(_mod, _inp, out):
            b, s, d = out.shape
            k = out.view(b, s, n_kv, head_dim)
            norms.append(k.norm(dim=-1)[0].float().cpu())
        return hook

    for layer in model.model.layers:
        handles.append(layer.self_attn.k_proj.register_forward_hook(_mk()))
    try:
        with torch.no_grad():
            model(input_ids=ids, use_cache=False, **_logits_kw(model))
    finally:
        for hd in handles:
            hd.remove()
    return norms


def _span_mask(starts, ends, lo, hi):
    """Boolean token mask for tokens overlapping char range [lo, hi)."""
    import torch
    return (torch.tensor(starts) < hi) & (torch.tensor(ends) > lo)


def _knorm_diagnose_sample(model, tok, sample, marker, kept: float,
                           task: str, arm: str = "qa"):
    """Replay KnormPress's per-(layer, head) keep-lowest-norm selection
    offline and report keep-rates by span:
      keep_value    — the needle's digit tokens (niah_single only)
      keep_window   — needle sentence neighborhood (+-90/+30 chars)
      keep_question — tokens from the qa split point on (index-based, same
                      first-token rule as _generate_qa)
      keep_haystack — the rest
    plus frac_value_evicted = fraction of (layer, head) pairs whose ENTIRE
    needle-value span is evicted.
    v4.1 ARM SEMANTICS (the primary sweeps are arm='qa', so the replay must
    match): arm='qa' scores and budgets ONLY the context tokens [0, k_idx)
    — the press never sees the question, which is retained wholesale
    (keep_question == 1.0 by construction; keep_overall = keep rate over
    the context, the budgeted quantity). arm='joint' replays the v3
    protocol: selection over the whole prompt, question competes for
    budget. n_keep uses kvpress's exact floor (scorer_press.py:
    int(k_len * (1 - compression_ratio))), not round.
    keep_*_high columns replay the OPPOSITE orientation (keep highest
    norms) as a sign-flip safety check."""
    import torch
    prompt = _templated(tok, sample["input"])
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids = torch.tensor([enc["input_ids"]], device="cuda")
    starts = [o[0] for o in enc["offset_mapping"]]
    ends = [o[1] for o in enc["offset_mapping"]]
    S = ids.shape[1]
    _, q_txt = _split_sample(sample["input"], marker)
    qpos = prompt.rindex(q_txt)
    k_idx = next((i for i, st in enumerate(starts) if st >= qpos), S)
    q_mask = torch.zeros(S, dtype=torch.bool)
    q_mask[k_idx:] = True
    if task == "niah_single":
        g = sample["outputs"][0]
        vpos = prompt.index(g)          # callers skip ambiguous samples
        v_mask = _span_mask(starts, ends, vpos, vpos + len(g)) & ~q_mask
        w_mask = _span_mask(starts, ends, max(0, vpos - 90),
                            vpos + len(g) + 30) & ~q_mask
    else:
        v_mask = torch.zeros(S, dtype=torch.bool)
        w_mask = v_mask.clone()
    h_mask = ~(q_mask | v_mask | w_mask)
    sel_len = k_idx if arm == "qa" else S
    o_mask = torch.zeros(S, dtype=torch.bool)
    o_mask[:sel_len] = True             # the budgeted (selection) domain
    norms = _capture_key_norms(model, ids)          # [L] x (S, n_kv)
    n_keep = max(1, int(kept * sel_len))            # kvpress floors
    tot = {k: 0.0 for k in ("overall", "value", "window", "question",
                            "haystack")}
    tot_hi = {k: 0.0 for k in tot}
    n_lh, full_evict = 0, 0
    for lyr in norms:
        for hidx in range(lyr.shape[1]):
            col = lyr[:sel_len, hidx]
            keep_lo = torch.zeros(S, dtype=torch.bool)
            keep_lo[torch.topk(-col, n_keep).indices] = True   # keep LOWEST
            keep_hi = torch.zeros(S, dtype=torch.bool)
            keep_hi[torch.topk(col, n_keep).indices] = True    # opposite
            if arm == "qa":             # question retained by construction
                keep_lo[k_idx:] = True
                keep_hi[k_idx:] = True
            for name, m in (("overall", o_mask), ("value", v_mask),
                            ("window", w_mask), ("question", q_mask),
                            ("haystack", h_mask)):
                if int(m.sum()):
                    tot[name] += keep_lo[m].float().mean().item()
                    tot_hi[name] += keep_hi[m].float().mean().item()
            if int(v_mask.sum()) and not bool(keep_lo[v_mask].any()):
                full_evict += 1
            n_lh += 1
    out = {f"keep_{k}": v / n_lh for k, v in tot.items()}
    out.update({f"keep_{k}_high": v / n_lh for k, v in tot_hi.items()})
    out.update(dict(
        arm=arm,
        frac_value_evicted=(full_evict / n_lh if int(v_mask.sum()) else
                            float("nan")),
        n_tokens=S, n_value_tokens=int(v_mask.sum()),
        n_question_tokens=int(q_mask.sum()), depth=sample["depth"]))
    return out


@app.function(**GPU_KW, timeout=3 * 3600)
def diagnose(model_tag: str, arm: str, langs: list[str], press_name: str,
             kept: float, task: str, ctx_len: int, seed: int, n_diag: int):
    """Question-vs-needle eviction diagnosis (currently knorm only — SnapKV
    scoring needs attention capture; StreamingLLM's mask is analytic).
    Motivated by vi/knorm's budget-INSENSITIVE collapse (0.09@50 vs 0.06@25):
    graded needle loss cannot explain a step function, but evicting the
    question/instruction tokens can. v4.1: ARM-AWARE — run with the arm you
    are explaining (qa is the primary protocol; under qa the question is
    protected by construction, so the informative spans are value/window/
    haystack; use arm=joint to explain v3's joint-arm results). Samples
    where the gold occurs more than once in the input are skipped (needle
    location would be ambiguous). ~1 forward per sample, no generation."""
    assert press_name == "knorm", "diagnose currently supports knorm only"
    assert arm in ("qa", "joint"), f"unknown arm {arm!r}"
    log_open(f"diagnose_{model_tag}_{arm}")
    import pandas as pd
    model, tok = _get_model(model_tag)
    recs = []
    for lang in langs:
        samples = load_samples(lang, ctx_len, max(n_diag, 12), task=task,
                               seed=seed)
        if task == "niah_single":
            n0 = len(samples)
            samples = [s for s in samples
                       if s["input"].count(s["outputs"][0]) == 1]
            if len(samples) < n0:
                log(f"{lang}: skipping {n0 - len(samples)} ambiguous "
                    f"samples (gold occurs >1x — needle location "
                    f"uncertain)")
        marker = get_question_split(samples, task, lang)
        for s in samples[:n_diag]:
            d = _knorm_diagnose_sample(model, tok, s, marker, kept, task,
                                       arm=arm)
            recs.append({"model": model_tag, "lang": lang, "task": task,
                         "kept": kept, "sample_id": s["sample_id"], **d})
        sub = pd.DataFrame([r for r in recs if r["lang"] == lang])
        log(f"{lang}: keep-rate value={sub.keep_value.mean():.3f} "
            f"window={sub.keep_window.mean():.3f} "
            f"question={sub.keep_question.mean():.3f} "
            f"haystack={sub.keep_haystack.mean():.3f} "
            f"| frac(value fully evicted per head)="
            f"{sub.frac_value_evicted.mean():.3f}")
    df = pd.DataFrame(recs)
    os.makedirs(f"{VOL}/results", exist_ok=True)
    fp = (f"{VOL}/results/diagnosis_knorm_{model_tag}_{arm}"
          f"_{task}_kept{kept}.csv")
    df.to_csv(fp, index=False)
    # fig7: keep-rate by span per language
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    means = df.groupby("lang")[["keep_value", "keep_window",
                                "keep_question", "keep_haystack"]].mean()
    x = np.arange(len(means))
    fig, ax = plt.subplots(figsize=(1.6 * len(means) + 3, 3.4))
    for off, col in zip((-0.3, -0.1, 0.1, 0.3), means.columns):
        ax.bar(x + off, means[col], width=0.18, label=col.replace("keep_", ""))
    ax.axhline(kept, color="k", lw=0.8, ls="--",
               label=f"uniform ({kept})")
    ax.set_xticks(x, means.index)
    ax.set_ylabel("mean keep rate (layers x kv-heads)")
    ax.set_title(f"KnormPress selection replay — {model_tag}, arm={arm}, "
                 f"kept={kept}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(f"{VOL}/figures", exist_ok=True)
    fig.savefig(f"{VOL}/figures/fig7_knorm_diagnosis_{model_tag}_{arm}"
                f"_{task}_kept{kept}.png", dpi=200)
    vol.commit()
    if arm == "qa":
        log(f"wrote {fp} + fig7. Reading (qa: question protected by "
            f"construction, keep_question=1.0): the informative signal is "
            f"frac_value_evicted and keep_value/keep_window vs the uniform "
            f"line — ~1 full-eviction rate with a flat haystack = needle "
            f"loss explains the collapse; if NOT, the qa collapse is not "
            f"about knorm's selection and needs another mechanism.")
    else:
        log(f"wrote {fp} + fig7. Reading: question keep-rate << {kept} in "
            f"a language that collapses to false-absence = instruction "
            f"damage; frac_value_evicted ~ 1 with intact question = "
            f"needle loss.")


# ----------------------------------------------------------------------------
# Aggregation + statistics helpers
# ----------------------------------------------------------------------------
AGG_KEYS = ["model", "arm", "lang", "ctx_len", "press", "kept", "task",
            "seed", "toy"]


@app.function(**CPU_KW, timeout=20 * 60)
def aggregate():
    log_open("aggregate")
    import numpy as np
    import pandas as pd
    files = sorted(Path(RAW_DIR).glob("*.jsonl"))
    if not files:
        log("no raw results yet")
        return
    # v4.1: tolerant reader (drops a truncated FINAL line with a warning,
    # raises on mid-file corruption). From-dicts construction keeps the hex
    # config_hash a string (the v3 ArrowTypeError fix, preserved).
    frames = []
    for f in files:
        rs = _read_jsonl(f)
        if rs:
            frames.append(pd.DataFrame(rs))
    if not frames:
        log("no parsable raw results yet")
        return
    df = pd.concat(frames, ignore_index=True)
    for col, default in (("arm", "joint"), ("model", DEFAULT_MODEL)):
        if col not in df.columns:
            df[col] = default
        df[col] = df[col].fillna(default)
    for col in ("config_hash", "lang", "press", "task", "model", "arm"):
        df[col] = df[col].astype(str)
    df = df.drop_duplicates(subset=AGG_KEYS + ["sample_id"])
    n_toy = int(df["toy"].sum())
    if n_toy:
        log(f"excluding {n_toy} toy selftest rows from aggregation")
    real = df[~df["toy"]].copy()
    os.makedirs(f"{VOL}/results", exist_ok=True)
    df.to_parquet(f"{VOL}/results/results.parquet")  # full, incl. toy-tagged
    if real.empty:
        log("no real (non-toy) results yet")
        vol.commit()
        return
    rng = np.random.default_rng(DEFAULT_SEED)

    def _ci(v):
        v = np.asarray(v, dtype=float)
        boots = rng.choice(v, size=(1000, len(v)), replace=True).mean(axis=1)
        return (round(float(np.percentile(boots, 2.5)), 3),
                round(float(np.percentile(boots, 97.5)), 3))

    recs = []
    for key, g in real.groupby(AGG_KEYS):
        lo, hi = _ci(g["correct"].values)
        recs.append({**dict(zip(AGG_KEYS, key)),
                     "acc": round(g["correct"].mean(), 3),
                     "ci_low": lo, "ci_high": hi, "n": len(g),
                     "mean_wall_s": round(g["wall_s"].mean(), 2)})
    cell = pd.DataFrame(recs).sort_values(
        ["model", "arm", "task", "press", "kept", "lang"])
    cell.to_csv(f"{VOL}/results/cell_table.csv", index=False)
    log("per-cell table (95% bootstrap CIs):\n" + cell.to_string(index=False))
    for (m, a, t), g in cell.groupby(["model", "arm", "task"]):
        piv = g.pivot_table(index=["press", "kept"], columns="lang",
                            values="acc")
        log(f"accuracy pivot [{m}/{a}/{t}]:\n" + piv.to_string())
    log(f"rows={len(real)} cells={len(cell)} -> results.parquet, "
        f"cell_table.csv")
    vol.commit()


def _bh_q(pvals):
    """Benjamini–Hochberg adjusted q-values."""
    import numpy as np
    p = np.asarray(pvals, float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0.0, 1.0)
    return out


def _boot_p(boot):
    """Two-sided bootstrap p (probability the effect's sign is unstable),
    floored at 2/(B+1)."""
    import numpy as np
    b = np.asarray(boot, float)
    lo, hi = float((b <= 0).mean()), float((b >= 0).mean())
    return float(min(1.0, max(2 * min(lo, hi), 2.0 / (len(b) + 1))))


def _ks_stat(a, b):
    """Two-sample Kolmogorov-Smirnov statistic (max |ECDF difference|),
    pure numpy — used by the v4.1 depth-balance audit (no scipy in the
    image). Statistic only; the audit thresholds it directly rather than
    converting to a p-value."""
    import numpy as np
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    allv = np.concatenate([a, b])
    ca = np.searchsorted(a, allv, side="right") / len(a)
    cb = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def _rankdata(v):
    import numpy as np
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), float)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _spearman(x, y):
    import numpy as np
    rx, ry = _rankdata(x), _rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def _perm_p_spearman(x, y, max_perm=20000, seed=DEFAULT_SEED):
    """Two-sided permutation p for Spearman rho — EXACT enumeration when
    n! <= max_perm (n=7 -> 5040, exact). No scipy dependency."""
    import numpy as np
    n = len(x)
    rho = _spearman(x, y)
    rx = _rankdata(x)
    rxc = rx - rx.mean()
    denx = float((rxc ** 2).sum())
    ry = _rankdata(y)

    def _rho_of(p_ry):
        ryc = p_ry - p_ry.mean()
        den = math.sqrt(denx * float((ryc ** 2).sum()))
        return float((rxc * ryc).sum() / den) if den > 0 else 0.0

    if math.factorial(n) <= max_perm:
        vals = [_rho_of(np.asarray(p)) for p in itertools.permutations(ry)]
        exact = True
    else:
        rng = np.random.default_rng(seed)
        vals = [_rho_of(rng.permutation(ry)) for _ in range(max_perm)]
        exact = False
    cnt = sum(1 for v in vals if abs(v) >= abs(rho) - 1e-12)
    return rho, cnt / len(vals), exact


@app.function(**CPU_KW, timeout=60 * 60)
def analyze():
    """Per-facet (model, arm, task): paired bootstrap gap amplification vs EN
    in raw accuracy AND smoothed log-odds, with CIs, two-sided bootstrap p,
    and BH q (36-test family per facet). v4.1: a cross-language needle-DEPTH
    balance audit (KS vs en + fig6 ECDF + streaming-by-depth table) runs
    first; fig1 raw-GA and fig1b log-odds-GA heatmaps render at BOTH budgets
    with dual-notation stars (* per-test CI, ** BH q<.05); fig2 degradation
    curves; fig3 fertility scatter (PREFERS fertility measured on the actual
    eval texts, falls back to FLORES+; source recorded in summary) with
    exact-permutation p and leave-one-out rho table. Then the v4 money
    analysis: fig5 arm contrast (joint - qa, paired over shared sample_ids)
    per model with per-model BH q, isolating the query-awareness effect per
    (lang, press, budget)."""
    log_open("analyze")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    os.makedirs(f"{VOL}/figures", exist_ok=True)
    df = pd.read_parquet(f"{VOL}/results/results.parquet")
    df = df[(~df["toy"]) & (df.ctx_len == CTX_LEN)]
    if df.empty:
        log("no real rows for analysis")
        return
    comp_cols = [(p, k) for p in PRESSES for k in BUDGETS_KEPT]
    base_col = ("none", 1.0)
    summary = {"ctx_len": CTX_LEN, "n_boot": N_BOOT,
               "boot_seed": DEFAULT_SEED, "floor_acc": FLOOR_ACC,
               "facets": {}}
    all_ga = []

    def _lo(p, n):  # smoothed log-odds (Haldane-Anscombe)
        return np.log((p * n + 0.5) / (n - p * n + 0.5))

    # ---- v4.1 depth-balance audit: needle depth is a property of the
    # GENERATED DATA (identical across models/arms/settings for a lang), so
    # cross-language imbalance would masquerade as a language effect — and
    # for streaming_llm (purely positional) it would BE the entire effect.
    dsing = df[(df.task == "niah_single") & (df.press == "none")]
    if len(dsing):
        dd = dsing.drop_duplicates(subset=["lang", "sample_id"])[
            ["lang", "sample_id", "depth"]]
        en_d = dd[dd.lang == "en"].depth.to_numpy(dtype=float)
        if not len(en_d):
            log("depth audit skipped: no en baseline rows yet")
        else:
            drecs = []
            for lang, g in dd.groupby("lang"):
                v = g.depth.to_numpy(dtype=float)
                ks = _ks_stat(v, en_d)
                md = float(v.mean() - en_d.mean())
                drecs.append({"lang": lang, "n": len(v),
                              "depth_mean": round(float(v.mean()), 3),
                              "depth_p10": round(float(np.percentile(v, 10)),
                                                 3),
                              "depth_p50": round(float(np.percentile(v, 50)),
                                                 3),
                              "depth_p90": round(float(np.percentile(v, 90)),
                                                 3),
                              "ks_vs_en": round(ks, 3),
                              "mean_diff_vs_en": round(md, 3)})
                if lang != "en" and (ks > 0.25 or abs(md) > 0.08):
                    log(f"WARNING depth imbalance {lang} vs en: "
                        f"KS={ks:.3f} mean diff={md:+.3f} — treat "
                        f"cross-language claims (especially streaming_llm) "
                        f"with care; condition on depth before believing "
                        f"them")
            dbal = pd.DataFrame(drecs).sort_values("lang")
            dbal.to_csv(f"{VOL}/results/depth_balance.csv", index=False)
            log("needle-depth balance across languages "
                "(niah_single, per-sample, deduped):\n"
                + dbal.to_string(index=False))
            summary["depth_balance"] = dbal.set_index("lang").to_dict(
                "index")
            fig, ax = plt.subplots(figsize=(5, 3.4))
            for lang, g in dd.groupby("lang"):
                v = np.sort(g.depth.to_numpy(dtype=float))
                ax.step(v, np.arange(1, len(v) + 1) / len(v),
                        where="post", label=lang)
            ax.set_xlabel("needle depth (fraction of context)")
            ax.set_ylabel("ECDF")
            ax.set_title("needle-depth distributions by language "
                         "(niah_single)", fontsize=9)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(f"{VOL}/figures/fig6_depth_ecdf.png", dpi=200)
            plt.close(fig)
        st = df[(df.task == "niah_single")
                & (df.press == "streaming_llm")].copy()
        if len(st):
            st["depth_bin"] = pd.cut(
                st.depth.astype(float), [0, 0.25, 0.5, 0.75, 1.0001],
                labels=["0-.25", ".25-.5", ".5-.75", ".75-1"])
            sb = (st.groupby(["model", "arm", "kept", "lang", "depth_bin"],
                             observed=True)["correct"].mean().round(3)
                  .unstack("depth_bin"))
            sb.to_csv(f"{VOL}/results/streaming_by_depth.csv")
            log("streaming_llm accuracy by needle depth (sanity: should be "
                "~monotone in depth and language-flat GIVEN depth):\n"
                + sb.to_string())

    facets = sorted(set(zip(df.model, df.arm, df.task)))
    for (model, arm, task) in facets:
        ftag = f"{model}__{arm}__{task}"
        sub = df[(df.model == model) & (df.arm == arm) & (df.task == task)]
        langs_present = sorted(sub.lang.unique())
        mats, ns = {}, {}
        for lang in langs_present:
            piv = (sub[sub.lang == lang]
                   .pivot_table(index="sample_id", columns=["press", "kept"],
                                values="correct", aggfunc="first"))
            have = [c for c in [base_col] + comp_cols if c in piv.columns]
            piv = piv[have].dropna()
            mats[lang], ns[lang] = piv, len(piv)
            log(f"[{ftag}] {lang}: {len(piv)} paired samples x "
                f"{len(have)} settings")
        if "en" not in mats or base_col not in mats["en"].columns:
            log(f"[{ftag}] English baseline missing — GA skipped")
            continue
        rng = np.random.default_rng(DEFAULT_SEED)
        boot_acc = {}
        for lang, piv in mats.items():
            m = piv.to_numpy(dtype=float)
            idx = rng.integers(0, len(m), size=(N_BOOT, len(m)))
            boot_acc[lang] = {c: m[idx, j].mean(axis=1)
                              for j, c in enumerate(piv.columns)}
        recs = []
        for lang in langs_present:
            piv = mats[lang]
            if base_col not in piv.columns:
                continue
            base_acc = float(piv[base_col].mean())
            for c in comp_cols:
                if c not in piv.columns or c not in mats["en"].columns:
                    continue
                point_d = base_acc - piv[c].mean()
                en_d = (mats["en"][base_col].mean()
                        - mats["en"][c].mean())
                d_boot = boot_acc[lang][base_col] - boot_acc[lang][c]
                den_boot = boot_acc["en"][base_col] - boot_acc["en"][c]
                ga_boot = d_boot - den_boot
                ci = np.percentile(ga_boot, [2.5, 97.5])
                lo_point = ((_lo(piv[base_col].mean(), ns[lang])
                             - _lo(piv[c].mean(), ns[lang]))
                            - (_lo(mats["en"][base_col].mean(), ns["en"])
                               - _lo(mats["en"][c].mean(), ns["en"])))
                lo_boot = (_lo(boot_acc[lang][base_col], ns[lang])
                           - _lo(boot_acc[lang][c], ns[lang])
                           - _lo(boot_acc["en"][base_col], ns["en"])
                           + _lo(boot_acc["en"][c], ns["en"]))
                lo_ci = np.percentile(lo_boot, [2.5, 97.5])
                is_en = (lang == "en")
                recs.append({
                    "model": model, "arm": arm, "task": task,
                    "lang": lang, "press": c[0], "kept": c[1],
                    "base_acc": round(base_acc, 4),
                    "base_floor": bool(base_acc < FLOOR_ACC),
                    "delta": round(point_d, 4),
                    "delta_ci_low": round(float(np.percentile(d_boot, 2.5)),
                                          4),
                    "delta_ci_high": round(float(np.percentile(d_boot,
                                                               97.5)), 4),
                    "gap_amp": round(point_d - en_d, 4),
                    "ga_ci_low": round(float(ci[0]), 4),
                    "ga_ci_high": round(float(ci[1]), 4),
                    "ga_sig": bool(ci[0] > 0 or ci[1] < 0),
                    "ga_p": 1.0 if is_en else _boot_p(ga_boot),
                    "gap_amp_logodds": round(float(lo_point), 4),
                    "ga_lo_ci_low": round(float(lo_ci[0]), 4),
                    "ga_lo_ci_high": round(float(lo_ci[1]), 4),
                    "ga_lo_sig": bool(lo_ci[0] > 0 or lo_ci[1] < 0),
                    "ga_lo_p": 1.0 if is_en else _boot_p(lo_boot),
                })
        ga = pd.DataFrame(recs)
        if ga.empty:
            log(f"[{ftag}] no compressed cells yet")
            continue
        ne = ga.lang != "en"
        ga.loc[ne, "ga_q"] = _bh_q(ga.loc[ne, "ga_p"].to_numpy())
        ga.loc[ne, "ga_lo_q"] = _bh_q(ga.loc[ne, "ga_lo_p"].to_numpy())
        ga[["ga_q", "ga_lo_q"]] = ga[["ga_q", "ga_lo_q"]].fillna(1.0)
        ga["ga_sig_bh"] = ga.ga_q < 0.05
        ga["ga_lo_sig_bh"] = ga.ga_lo_q < 0.05
        all_ga.append(ga)
        log(f"[{ftag}] gap table (paired bootstrap, B={N_BOOT}):\n"
            + ga.to_string(index=False))
        surv = ga[ne & (ga.ga_sig_bh | ga.ga_lo_sig_bh)]
        log(f"[{ftag}] BH(q<.05) survivors: "
            + (surv[["lang", "press", "kept", "gap_amp", "ga_q",
                     "gap_amp_logodds", "ga_lo_q"]].to_string(index=False)
               if len(surv) else "none"))
        floors = ga[ga.base_floor].lang.unique().tolist()
        if floors:
            log(f"[{ftag}] WARNING base_acc<{FLOOR_ACC} (interpret GA with "
                f"care, cells flagged base_floor): {floors}")

        fert = _load_fertility(model)
        if fert is not None:
            ocol = ("tokens_per_byte"
                    if "tokens_per_byte" in fert.columns
                    and fert["tokens_per_byte"].notna().any()
                    else "tokens_per_byte_eval")
            order = list(fert.dropna(subset=[ocol]).sort_values(ocol).index)
        else:
            order = list(langs_present)
        order = [l for l in order if l in langs_present]
        order += [l for l in langs_present if l not in order]

        def _heat(vcol, scol_raw, scol_bh, kept_v, fname, label):
            gk = ga[ga.kept == kept_v]
            if gk.empty:
                return
            piv = gk.pivot(index="press", columns="lang", values=vcol)
            piv = piv.reindex(index=PRESSES, columns=order)
            sig_r = gk.pivot(index="press", columns="lang",
                             values=scol_raw).reindex(
                                 index=PRESSES, columns=order)
            sig_b = gk.pivot(index="press", columns="lang",
                             values=scol_bh).reindex(
                                 index=PRESSES, columns=order)
            finite = np.abs(piv.values[np.isfinite(
                piv.values.astype(float))])
            vmax = max(0.1, (float(finite.max()) if finite.size else 0.1)
                       * 1.05)
            fig, ax = plt.subplots(
                figsize=(1.2 * len(order) + 2.5, 3))
            im = ax.imshow(piv.values.astype(float), cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(len(piv.columns)), piv.columns)
            ax.set_yticks(range(len(piv.index)), piv.index)
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    v = piv.values[i, j]
                    if v is not None and np.isfinite(float(v)):
                        # == True (not `is True`) so NaN never stars
                        star = ("**" if sig_b.values[i, j] == True   # noqa
                                else "*" if sig_r.values[i, j] == True  # noqa
                                else "")
                        ax.text(j, i, f"{float(v):+.2f}{star}", ha="center",
                                va="center", fontsize=8)
            ax.set_title(f"{ftag}  kept={kept_v}", fontsize=8)
            fig.colorbar(im, label=label)
            fig.tight_layout()
            fig.savefig(f"{VOL}/figures/{fname}", dpi=200)
            plt.close(fig)

        for kv in BUDGETS_KEPT:
            _heat("gap_amp", "ga_sig", "ga_sig_bh", kv,
                  f"fig1_gap_heatmap_kept{kv}__{ftag}.png",
                  "gap amplification vs EN (* raw CI, ** BH q<.05)")
            _heat("gap_amp_logodds", "ga_lo_sig", "ga_lo_sig_bh", kv,
                  f"fig1b_gap_heatmap_logodds_kept{kv}__{ftag}.png",
                  "GA vs EN, smoothed log-odds (* LO CI, ** BH q<.05)")

        fig, axes = plt.subplots(1, len(PRESSES),
                                 figsize=(4 * len(PRESSES), 3), sharey=True)
        for ax, press in zip(np.atleast_1d(axes), PRESSES):
            for lang in order:
                xs, ys, lo_, hi_ = [], [], [], []
                for c in [base_col] + [(press, k) for k in
                                       sorted(BUDGETS_KEPT, reverse=True)]:
                    if c in mats[lang].columns:
                        xs.append(c[1] if c != base_col else 1.0)
                        p = mats[lang][c].mean()
                        b = boot_acc[lang][c]
                        ys.append(p)
                        lo_.append(p - np.percentile(b, 2.5))
                        hi_.append(np.percentile(b, 97.5) - p)
                ax.errorbar(xs, ys, yerr=[lo_, hi_], marker="o", capsize=2,
                            label=lang)
            ax.set_title(press + ("  (positional control)"
                                  if press == "streaming_llm" else ""))
            ax.set_xlabel("fraction of KV kept")
            ax.invert_xaxis()
        np.atleast_1d(axes)[0].set_ylabel("accuracy")
        np.atleast_1d(axes)[-1].legend(fontsize=7)
        fig.suptitle(ftag, fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{VOL}/figures/fig2_degradation__{ftag}.png", dpi=200)
        plt.close(fig)

        fstats = {"paired_n_per_lang": ns}
        if fert is not None:
            has_eval = "tokens_per_byte_eval" in fert.columns
            fig, axes = plt.subplots(1, len(PRESSES),
                                     figsize=(4.2 * len(PRESSES), 3.4),
                                     sharey=True)
            rhos = {}
            for ax, press in zip(np.atleast_1d(axes), PRESSES):
                pts = (ga[(ga.kept == 0.25) & (ga.press == press)]
                       .set_index("lang")[["delta"]].join(fert))
                # v4.1: prefer fertility measured on the ACTUAL eval texts
                # (same domain as the experiment); fall back to FLORES+
                # only when eval-text values are incomplete for this facet.
                xcol = ("tokens_per_byte_eval"
                        if has_eval
                        and pts["tokens_per_byte_eval"].notna().all()
                        else "tokens_per_byte")
                if xcol not in pts.columns:
                    continue
                pts = pts.dropna(subset=[xcol])
                if len(pts) < 4:
                    continue
                src = ("eval_texts" if xcol.endswith("_eval")
                       else "flores_plus")
                x = pts[xcol].to_numpy()
                y = pts["delta"].to_numpy()
                rho, p, exact = _perm_p_spearman(x, y)
                loo = {drop: round(_spearman(
                    np.delete(x, i), np.delete(y, i)), 3)
                    for i, drop in enumerate(pts.index)}
                rhos[press] = {"rho": round(rho, 4), "p_perm": round(p, 4),
                               "exact_perm": exact, "n": len(pts),
                               "fertility_source": src,
                               "leave_one_out": loo}
                ax.scatter(x, y)
                for lang, r in pts.iterrows():
                    ax.annotate(lang, (r[xcol], r["delta"]),
                                fontsize=8)
                ax.set_title(f"{press}: rho={rho:.2f} p={p:.3f} (n={len(pts)})"
                             + ("\n(positional control)"
                                if press == "streaming_llm" else ""))
                ax.set_xlabel(f"tokens per UTF-8 byte ({src})")
            np.atleast_1d(axes)[0].set_ylabel("accuracy drop at kept=0.25")
            fig.suptitle(ftag, fontsize=9)
            fig.tight_layout()
            fig.savefig(f"{VOL}/figures/fig3_fertility__{ftag}.png", dpi=200)
            plt.close(fig)
            fstats["spearman_delta25_vs_fertility"] = rhos
            log(f"[{ftag}] Spearman(delta@0.25 vs tokens/byte): "
                + json.dumps(rhos))
        summary["facets"][ftag] = fstats

    if all_ga:
        ga_all = pd.concat(all_ga, ignore_index=True)
        ga_all.to_csv(f"{VOL}/results/gap_table.csv", index=False)

    # --- arm contrast: joint - qa (query-awareness effect), niah_single ----
    contrast = []
    single = df[df.task == "niah_single"]
    rng = np.random.default_rng(DEFAULT_SEED)
    for model in sorted(single.model.unique()):
        msub = single[single.model == model]
        if not {"qa", "joint"} <= set(msub.arm.unique()):
            continue
        for lang in sorted(msub.lang.unique()):
            piv = (msub[msub.lang == lang]
                   .pivot_table(index="sample_id",
                                columns=["arm", "press", "kept"],
                                values="correct", aggfunc="first"))
            settings = [base_col] + comp_cols
            have = [(a, p, k) for a in ("joint", "qa")
                    for (p, k) in settings
                    if (a, p, k) in piv.columns]
            piv = piv[have].dropna()
            if len(piv) == 0:
                continue
            idx = rng.integers(0, len(piv), size=(N_BOOT, len(piv)))
            for (p, k) in settings:
                jc, qc = ("joint", p, k), ("qa", p, k)
                if jc not in piv.columns or qc not in piv.columns:
                    continue
                jj = piv[jc].to_numpy(float)
                qq = piv[qc].to_numpy(float)
                boots = jj[idx].mean(axis=1) - qq[idx].mean(axis=1)
                ci = np.percentile(boots, [2.5, 97.5])
                contrast.append(dict(
                    model=model, lang=lang, press=p, kept=k,
                    acc_joint=round(float(jj.mean()), 4),
                    acc_qa=round(float(qq.mean()), 4),
                    d_joint_minus_qa=round(float(jj.mean() - qq.mean()), 4),
                    ci_low=round(float(ci[0]), 4),
                    ci_high=round(float(ci[1]), 4),
                    sig=bool(ci[0] > 0 or ci[1] < 0),
                    p=_boot_p(boots), n=len(piv)))
            jb, qb = ("joint",) + base_col, ("qa",) + base_col
            if jb in piv.columns and qb in piv.columns:
                mism = int((piv[jb] != piv[qb]).sum())
                if mism:
                    log(f"WARNING [{model}/{lang}] joint vs qa BASELINE "
                        f"disagree on {mism} samples — arms should be "
                        f"identical at press=none; investigate before "
                        f"trusting the contrast")
    if contrast:
        cdf = pd.DataFrame(contrast)
        # v4.1: BH within each model's family of compressed-setting tests
        # (baseline rows keep q=1); fig5 stars use the same dual notation
        # as fig1 (* per-test CI, ** BH survivor).
        cdf["q"] = 1.0
        for model in cdf.model.unique():
            m = (cdf.model == model) & (cdf.press != "none")
            if m.any():
                cdf.loc[m, "q"] = _bh_q(cdf.loc[m, "p"].to_numpy())
        cdf["sig_bh"] = cdf["q"] < 0.05
        cdf.to_csv(f"{VOL}/results/arm_contrast.csv", index=False)
        log("arm contrast (joint - qa; positive = query-awareness helps):\n"
            + cdf.to_string(index=False))
        for model in cdf.model.unique():
            sub = cdf[(cdf.model == model) & (cdf.press != "none")]
            if sub.empty:
                continue
            langs_o = sorted(sub.lang.unique())
            fig, axes = plt.subplots(1, len(BUDGETS_KEPT),
                                     figsize=(1.1 * len(langs_o)
                                              * len(BUDGETS_KEPT) + 3, 3))
            for ax, k in zip(np.atleast_1d(axes),
                             sorted(BUDGETS_KEPT, reverse=True)):
                piv = sub[sub.kept == k].pivot(index="press", columns="lang",
                                               values="d_joint_minus_qa")
                piv = piv.reindex(index=PRESSES, columns=langs_o)
                sg = sub[sub.kept == k].pivot(index="press", columns="lang",
                                              values="sig").reindex(
                    index=PRESSES, columns=langs_o)
                sgb = sub[sub.kept == k].pivot(index="press", columns="lang",
                                               values="sig_bh").reindex(
                    index=PRESSES, columns=langs_o)
                vmax = max(0.1, np.nanmax(np.abs(piv.values.astype(float)))
                           * 1.05)
                im = ax.imshow(piv.values.astype(float), cmap="RdBu_r",
                               vmin=-vmax, vmax=vmax)
                ax.set_xticks(range(len(piv.columns)), piv.columns)
                ax.set_yticks(range(len(piv.index)), piv.index)
                for i in range(piv.shape[0]):
                    for j in range(piv.shape[1]):
                        v = piv.values[i, j]
                        if v is not None and np.isfinite(float(v)):
                            star = ("**" if sgb.values[i, j] == True  # noqa
                                    else "*" if sg.values[i, j] == True  # noqa
                                    else "")
                            ax.text(j, i, f"{float(v):+.2f}{star}",
                                    ha="center", va="center", fontsize=8)
                ax.set_title(f"kept={k}")
                fig.colorbar(im, ax=ax, fraction=0.046)
            fig.suptitle(f"joint - qa accuracy (query-awareness effect) — "
                         f"{model}  (* CI, ** BH q<.05)", fontsize=9)
            fig.tight_layout()
            fig.savefig(f"{VOL}/figures/fig5_arm_contrast__{model}.png",
                        dpi=200)
            plt.close(fig)
        summary["arm_contrast"] = "results/arm_contrast.csv"

    summary["gap_table"] = "results/gap_table.csv"
    with open(f"{VOL}/results/summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)
    vol.commit()
    log("figures -> /vol/figures ; gap_table.csv + arm_contrast.csv + "
        "summary_stats.json -> /vol/results")


# ----------------------------------------------------------------------------
# Failure modes — task-aware in v4
# ----------------------------------------------------------------------------
def _stack_bottoms(seq_vals):
    """Bottoms for a SIGNED stacked bar chart: positive segments stack up
    from 0, negative segments stack down from 0. v4.1 fix — matplotlib's
    naive cumulative `bottom += vals` draws overlapping segments whenever
    the per-mode deltas have mixed signs (the EXPECTED case here: e.g.
    false_absence up while wrong_number down in the same cell). Returns one
    bottom array per input array; segments then tile
    [sum(negatives), sum(positives)] exactly, no overlap."""
    import numpy as np
    n = len(seq_vals[0])
    pos, neg, out = np.zeros(n), np.zeros(n), []
    for vals in seq_vals:
        vals = np.asarray(vals, float)
        out.append(np.where(vals >= 0, pos, neg))
        pos = pos + np.clip(vals, 0, None)
        neg = neg + np.clip(vals, None, 0)
    return out


MODES_BY_TASK = {
    "niah_single": ["correct", "false_absence", "wrong_number", "other"],
    "niah_none": ["correct", "false_presence_distractor",
                  "false_presence_hallucinated", "other"],
}


def _classify_row(row):
    """niah_single: false_absence (in-language none-word) / wrong_number /
    other. niah_none: false_presence_distractor (emitted a number that IS one
    of the 4 real-but-irrelevant needles — mis-retrieval) vs
    false_presence_hallucinated (emitted a number not in the haystack) /
    other. 'correct' always defers to the OFFICIAL scorer's verdict.
    v4.1: numbers are canonicalized via int() on BOTH sides before the
    distractor comparison — Python's \\d matches native-script decimal
    digits (e.g. Devanagari), which previously compared as unequal strings
    against ASCII distractors and were misfiled as hallucinated."""
    if bool(row["correct"]):
        return "correct"

    def _canon(n):
        try:
            return str(int(n))
        except (ValueError, TypeError):
            return str(n)

    pred = str(row["pred"])
    proc = unicodedata.normalize("NFKC", pred)
    nums = [_canon(n) for n in re.findall(r"\d+", proc) if len(n) > 1]
    if str(row["task"]) == "niah_none":
        d = row.get("distractors")
        try:
            distr = set(_canon(x) for x in list(d)) if d is not None \
                else set()
        except TypeError:
            distr = set()
        if nums and distr and any(n in distr for n in nums):
            return "false_presence_distractor"
        if nums:
            return "false_presence_hallucinated"
        return "other"
    if compare_none(row["lang"], [], pred):
        return "false_absence"
    if nums:
        return "wrong_number"
    return "other"


@app.function(**CPU_KW, timeout=30 * 60)
def failure_modes():
    """Decompose every error into task-appropriate channels and attribute
    each cell's accuracy drop to them, per facet (model, arm, task). Zero
    extra GPU cost — operates on stored `pred` strings."""
    log_open("failure_modes")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    os.makedirs(f"{VOL}/figures", exist_ok=True)
    df = pd.read_parquet(f"{VOL}/results/results.parquet")
    df = df[(~df["toy"]) & (df.ctx_len == CTX_LEN)].copy()
    if df.empty:
        log("no real rows for failure-mode analysis")
        return
    if "distractors" not in df.columns:
        df["distractors"] = None
    df["mode"] = df.apply(_classify_row, axis=1)
    # ---- v4.1 scorer-sensitivity: re-score every row with the robustness
    # scorers (strict on niah_none, lenient on niah_single) and table the
    # per-cell difference. Zero GPU cost; the OFFICIAL scorer remains the
    # headline metric everywhere else — this table bounds how much of any
    # headline effect could be a scorer-strictness artifact.
    df["correct_robust"] = df.apply(
        lambda r: score_robust(str(r["task"]), str(r["lang"]),
                               [str(g) for g in (r["gold"]
                                                 if r["gold"] is not None
                                                 else [])],
                               str(r["pred"])), axis=1)
    sens = (df.groupby(["model", "arm", "task", "lang", "press", "kept"])
            .agg(acc=("correct", "mean"),
                 acc_robust=("correct_robust", "mean"),
                 n=("correct", "size")).reset_index())
    sens["delta_robust"] = (sens["acc"] - sens["acc_robust"]).round(3)
    sens[["acc", "acc_robust"]] = sens[["acc", "acc_robust"]].round(3)
    os.makedirs(f"{VOL}/results", exist_ok=True)
    sens.to_csv(f"{VOL}/results/scorer_sensitivity.csv", index=False)
    hot = sens[sens.delta_robust.abs() >= 0.05]
    log("scorer sensitivity (official minus strict-none/lenient-single):\n"
        + (hot.to_string(index=False) if len(hot)
           else "all cells within 0.05 — headline effects are not "
                "scorer-strictness artifacts"))
    nz = df[(df.task == "niah_none") & df.correct.astype(bool)]
    if len(nz):
        pl = (nz.assign(pred_chars=nz.pred.astype(str).str.len(),
                        with_number=nz.pred.map(_has_multi_digit))
              .groupby("lang")
              .agg(n=("pred", "size"),
                   pred_chars_mean=("pred_chars", "mean"),
                   pred_chars_max=("pred_chars", "max"),
                   frac_with_number=("with_number", "mean")))
        log("audit of officially-'correct' niah_none predictions (long "
            "preds / numbers present = substring-laxity risk):\n"
            + pl.round(3).to_string())
    all_rates, all_deltas = [], []
    for (model, arm, task), sub in df.groupby(["model", "arm", "task"]):
        ftag = f"{model}__{arm}__{task}"
        modes = MODES_BY_TASK.get(task, MODES_BY_TASK["niah_single"])
        rates = (sub.groupby(["lang", "press", "kept"])["mode"]
                 .value_counts(normalize=True).unstack(fill_value=0.0))
        for m in modes:
            if m not in rates.columns:
                rates[m] = 0.0
        rates = rates[modes].reset_index()
        rates.insert(0, "task", task)
        rates.insert(0, "arm", arm)
        rates.insert(0, "model", model)
        all_rates.append(rates)
        log(f"[{ftag}] failure-mode rates (rows sum to 1):\n"
            + rates.round(3).to_string(index=False))
        base = (rates[rates.press == "none"].set_index("lang")[modes]
                .rename(columns={m: f"base_{m}" for m in modes}))
        comp = rates[rates.press != "none"].join(base, on="lang")
        if comp.empty:
            log(f"[{ftag}] no compressed cells yet — baseline-only")
            continue
        for m in modes:
            comp[f"d_{m}"] = comp[m] - comp[f"base_{m}"]
        drop = -comp["d_correct"]
        comp["drop"] = drop
        err_modes = [m for m in modes if m != "correct"]
        for m in err_modes:
            comp[f"share_{m}"] = np.where(
                drop > 1e-9, comp[f"d_{m}"] / drop.replace(0, np.nan),
                np.nan)
        keep_cols = (["model", "arm", "task", "lang", "press", "kept",
                      "drop"] + [f"d_{m}" for m in err_modes]
                     + [f"share_{m}" for m in err_modes])
        out = comp[keep_cols].sort_values(["press", "kept", "lang"])
        all_deltas.append(out)
        log(f"[{ftag}] damage attribution (drop = base_acc - cell_acc):\n"
            + out.round(3).to_string(index=False))

        fert = _load_fertility(model)
        if fert is not None:
            ocol = ("tokens_per_byte"
                    if "tokens_per_byte" in fert.columns
                    and fert["tokens_per_byte"].notna().any()
                    else "tokens_per_byte_eval")
            order = list(fert.dropna(subset=[ocol]).sort_values(ocol).index)
        else:
            order = sorted(comp.lang.unique())
        order = [l for l in order if l in set(comp.lang)]
        order += [l for l in sorted(comp.lang.unique()) if l not in order]
        tight = min(BUDGETS_KEPT)
        fig, axes = plt.subplots(1, len(PRESSES),
                                 figsize=(4.2 * len(PRESSES), 3.8),
                                 sharey=True)
        for ax, press in zip(np.atleast_1d(axes), PRESSES):
            s2 = comp[(comp.press == press) & (comp.kept == tight)]
            s2 = s2.set_index("lang").reindex(order)
            x = np.arange(len(s2))
            vals_list = [s2[f"d_{m}"].fillna(0).to_numpy()
                         for m in err_modes]
            for m, vals, b in zip(err_modes, vals_list,
                                  _stack_bottoms(vals_list)):
                ax.bar(x, vals, bottom=b, label=f"Δ {m}")
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(x, s2.index)
            ax.set_title(press + (" (positional control)"
                                  if press == "streaming_llm" else ""))
        np.atleast_1d(axes)[0].set_ylabel(
            f"rate change vs full cache (kept={tight})")
        np.atleast_1d(axes)[-1].legend(fontsize=7)
        fig.suptitle(ftag, fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{VOL}/figures/fig4_failure_modes__{ftag}.png", dpi=200)
        plt.close(fig)
    if all_rates:
        pd.concat(all_rates, ignore_index=True).to_csv(
            f"{VOL}/results/failure_modes.csv", index=False)
    if all_deltas:
        pd.concat(all_deltas, ignore_index=True).to_csv(
            f"{VOL}/results/failure_mode_deltas.csv", index=False)
    vol.commit()
    log("wrote results/failure_modes.csv, results/failure_mode_deltas.csv, "
        "results/scorer_sensitivity.csv, "
        "figures/fig4_failure_modes__<facet>.png")


# ----------------------------------------------------------------------------
# status / inspect / wipe
# ----------------------------------------------------------------------------
@app.function(**CPU_KW, timeout=10 * 60)
def status(cells_json: str):
    log_open("status")
    cells = json.loads(cells_json)
    t0 = time.time()
    total_done = total_need = 0
    for cfg in cells:
        h = _cfg_hash(cfg)
        p = Path(f"{RAW_DIR}/{h}.jsonl")
        n = len(_read_jsonl(p)) if p.exists() else 0
        total_done += min(n, cfg["n"])
        total_need += cfg["n"]
        tag = (f"{cfg['model']}/{cfg['arm']}/{cfg['task']} "
               f"{cfg['lang']:>3} {cfg['press']:<14} kept={cfg['kept']:<5}")
        log(f"{tag} {bar(min(n, cfg['n']), cfg['n'], t0)}")
    log(f"OVERALL: {total_done}/{total_need} samples "
        f"({100 * total_done / max(total_need, 1):.1f}%)")


@app.function(**CPU_KW, timeout=10 * 60)
def inspect_preds(model_tag: str, arm: str, lang: str, press: str,
                  kept: float, task: str, ctx_len: int, seed: int,
                  n_show: int, only_wrong: bool = False):
    """Print sample gold/pred pairs for one cell, straight from the volume."""
    log_open("inspect")
    cfg = dict(model=model_tag, arm=arm, lang=lang, ctx_len=ctx_len,
               press=press, kept=kept, n=0, seed=seed, task=task, toy=False)
    h = _cfg_hash(cfg)
    p = Path(f"{RAW_DIR}/{h}.jsonl")
    if not p.exists():
        log(f"no results file for {model_tag}/{arm}/{task} lang={lang} "
            f"press={press} kept={kept} (expected hash {h})")
        avail = sorted(Path(RAW_DIR).glob("*.jsonl"))
        log(f"{len(avail)} raw files on volume: "
            + ", ".join(f.stem for f in avail))
        return
    rows = _read_jsonl(p)
    n_ok = sum(r["correct"] for r in rows)
    lens = [len(r["pred"]) for r in rows]
    log(f"cell {h} | {model_tag}/{arm}/{task} {lang} press={press} "
        f"kept={kept} | acc={n_ok / len(rows):.3f} n={len(rows)} | "
        f"pred_chars min/mean/max = {min(lens)}/{sum(lens) / len(lens):.0f}/"
        f"{max(lens)}")
    shown = [r for r in rows if not r["correct"]] if only_wrong else rows
    if only_wrong:
        log(f"showing up to {n_show} of {len(shown)} INCORRECT samples")
    for r in shown[:n_show]:
        log("-" * 70)
        log(f"  gold={r['gold']} | correct={r['correct']} | "
            f"depth={r['depth']} | pred_chars={len(r['pred'])} | "
            f"input_tokens={r.get('input_length')} | "
            f"ctx_tok={r.get('context_tokens')} "
            f"comp_len={r.get('compressed_len')}")
        log(f"  pred={r['pred']!r}")
    log("-" * 70)


@app.function(**CPU_KW, timeout=10 * 60)
def wipe_results(confirm: bool):
    """Delete /vol/results (raw rows, cells.jsonl, CSVs, parquet, fertility)
    so the v4 campaign starts clean — v4 cell hashes include (model, arm), so
    stale v3 files would otherwise linger as dead weight. PRESERVES generated
    data, the ONERULER repo, the HF model cache, and logs."""
    log_open("wipe_results")
    assert confirm, "refusing to wipe without --confirm"
    import shutil
    tgt = f"{VOL}/results"
    if os.path.exists(tgt):
        n = sum(1 for _ in Path(tgt).rglob("*"))
        shutil.rmtree(tgt)
        log(f"removed {tgt} ({n} entries)")
    else:
        log(f"{tgt} did not exist")
    os.makedirs(RAW_DIR, exist_ok=True)
    vol.commit()
    log("results wiped. Preserved: /vol/data, /vol/oneruler_repo, "
        "/vol/hf_cache, /vol/logs. Re-run fertility for each model.")


# ----------------------------------------------------------------------------
# Config assembly + entrypoint
# ----------------------------------------------------------------------------
def _all_cells(model, arm, langs, ctx_len, n, seed, task, toy=False,
               baselines_only=False):
    cells = []
    for lang in langs:
        cells.append(dict(model=model, arm=arm, lang=lang, ctx_len=ctx_len,
                          press="none", kept=1.0, n=n, seed=seed, task=task,
                          toy=toy))
        if not baselines_only:
            for press in PRESSES:
                for kept in BUDGETS_KEPT:
                    cells.append(dict(model=model, arm=arm, lang=lang,
                                      ctx_len=ctx_len, press=press,
                                      kept=kept, n=n, seed=seed, task=task,
                                      toy=toy))
    return cells


def _by_lang(cells):
    g = {}
    for c in cells:
        g.setdefault(c["lang"], []).append(c)
    return list(g.values())


@app.local_entrypoint()
def main(stage: str = "status", n: int = 100, ctx_len: int = CTX_LEN,
         langs: str = "", seed: int = DEFAULT_SEED, task: str = TASK,
         press: str = "none", kept: float = 1.0, only_wrong: bool = False,
         model: str = DEFAULT_MODEL, arm: str = PRIMARY_ARM,
         confirm: bool = False):
    lang_list = [l for l in langs.split(",") if l] or LANGS
    assert model in MODELS, f"--model must be one of {list(MODELS)}"
    assert arm in ARMS, f"--arm must be one of {ARMS}"
    assert task in TASKS, f"--task must be one of {TASKS}"
    print(f"stage={stage} model={model} arm={arm} task={task} n={n} "
          f"ctx_len={ctx_len} seed={seed} langs={lang_list}")
    if stage == "smoke":
        smoke.remote(model)
    elif stage == "selftest":
        _scorer_unit_tests()
        cells = []
        for a in ARMS:
            for p_, k_ in (("none", 1.0), ("snapkv", 0.5), ("knorm", 0.5)):
                cells.append(dict(model=model, arm=a, lang="en",
                                  ctx_len=4096, press=p_, kept=k_,
                                  n=min(n, 6), seed=seed,
                                  task="niah_single", toy=True))
        cells.append(dict(model=model, arm="qa", lang="en", ctx_len=4096,
                          press="none", kept=1.0, n=min(n, 6), seed=seed,
                          task="niah_none", toy=True))
        for r in run_batch.map(_by_lang(cells)):
            print(r)
        aggregate.remote()
        print("selftest: end-to-end OK (toy data, both arms + niah_none; "
              "toy rows are tagged and excluded from real aggregation)")
    elif stage == "prepare-data":
        prepare_data.remote()
    elif stage == "gen-data":
        gen_data.remote(lang_list, ctx_len, n, seed, task)
    elif stage == "split-audit":
        split_audit.remote(lang_list, ctx_len, task, n, seed)
    elif stage == "baselines":
        cells = _all_cells(model, arm, lang_list, ctx_len, n, seed, task,
                           baselines_only=True)
        for r in run_batch.map(_by_lang(cells)):
            print(r)
        aggregate.remote()
    elif stage == "sweep":
        cells = _all_cells(model, arm, lang_list, ctx_len, n, seed, task)
        for r in run_batch.map(_by_lang(cells)):
            print(r)
        aggregate.remote()
    elif stage == "fertility":
        fertility.remote(model, lang_list)
    elif stage == "aggregate":
        aggregate.remote()
    elif stage == "analyze":
        analyze.remote()
    elif stage == "failure-modes":
        failure_modes.remote()
    elif stage == "diagnose":
        diagnose.remote(model, arm, lang_list, press if press != "none"
                        else "knorm", kept if kept < 1.0 else 0.5, task,
                        ctx_len, seed, min(n, 40))
    elif stage == "status":
        status.remote(json.dumps(
            _all_cells(model, arm, lang_list, ctx_len, n, seed, task)))
    elif stage == "inspect":
        inspect_preds.remote(model, arm, lang_list[0], press, kept, task,
                             ctx_len, seed,
                             (15 if only_wrong else 5) if n > 20 else n,
                             only_wrong=only_wrong)
    elif stage == "wipe-results":
        wipe_results.remote(confirm)
    else:
        raise SystemExit(f"unknown stage: {stage}")
