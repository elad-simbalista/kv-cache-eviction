#!/usr/bin/env python3
"""
post_run_calc.py — post-hoc recomputation and confound analysis for the v4
multilingual KV-cache eviction audit.

WHY THIS EXISTS
===============
The v4 paper draft computes its two headline statistics (the overdispersion
factor phi and the cross-model ordering rho) from *cell-level accuracies read
out of log tables*, some of which were reconstructed rather than quoted.  One
reconstruction is wrong: the Llama niah_single baseline was inferred as
mean 0.956 when the true value is 0.913, which makes phi_base 17.9 instead of
8.9 and turns a 5.3x amplification into a reported 2.6-2.9x.  Every number
needed to fix this is already on disk in results.parquet.

This script recomputes everything from RAW PER-ITEM RECORDS, adds paired
bootstrap intervals the draft lacks, and runs the confound tests the draft is
missing -- above all the haystack-redundancy test, which asks whether the
"reproducible language ordering" is a property of the language or of the
ONERULER corpus.

WHAT IT PRODUCES
================
  cells.csv                per-cell accuracy + item-bootstrap CI (replaces
                           every reconstructed number in Appendix A)
  dispersion.csv           phi per cell with bootstrap CI
  amplification.csv        phi_comp / phi_base with a PAIRED CI
  ordering.csv             Spearman rho, exact permutation p, leave-one-out,
                           floor-gated variants, item-bootstrap CI on rho
  redundancy.csv           per-language haystack redundancy (tokenizer-free)
  redundancy_link.csv      redundancy vs content-press retention  <-- the
                           confound test for the Korean result
  scorers.csv              official vs strict vs lenient per cell
  errors.csv               failure modes, incl. distractor-vs-hallucination
  depth.csv                depth balance + depth-conditioned accuracy
  positional_law.csv       streaming acc vs baseline * kept
  arm_invariance.csv       press=none qa-vs-joint per-item agreement
  detection.csv            Youden J / balanced accuracy on paired tasks
  fertility.csv            fertility correlations with LOO
  power.csv                power of the ordering test vs (k, n)
  post_run_report.md       a written summary with the numbers filled in

  --- v2 additions (the two computations the v1 paper edits depend on) ---
  value_origin_trials.csv  per-cell value-origin taxonomy at TRIAL level:
                           exact copy / near copy / entirely novel, with
                           Wilson CIs and raw numerators.  REPLACES the
                           draft's "a majority are fabricated values that
                           appear nowhere in the input", which is false:
                           almost every non-exact-copy value is a corrupted
                           near-copy of a value that IS in the input.
  value_origin_cands.csv   the same taxonomy at CANDIDATE level (a single
                           output can emit several numbers)
  value_origin_chance.csv  Monte-Carlo false-positive rate of the near-copy
                           rule, so the near-copy tier is defensible
  protocol_contrast.csv    query-agnostic (arm=qa) vs query-aware
                           (arm=joint) present-trial operating points:
                           present-task accuracy, coverage (= answer rate
                           H), precision-given-answer, per condition, with
                           a PAIRED item bootstrap on the difference.  The
                           joint arm was swept for niah_single at every
                           press and budget on both models, so this needs
                           no new GPU time.
  protocol_by_lang.csv     the same contrast per language

USAGE
=====
  # everything that needs only predictions (free, ~2 min):
  python post_run_calc.py --v4-dir ./results --out ./post_run

  # add the input-text analyses (redundancy, distractor membership).
  # EITHER download the generated data once:
  #   modal volume get kv-audit-vol data ./data
  # and then:
  python post_run_calc.py --v4-dir ./results --data-dir ./data --out ./post_run

  # OR compute those on the volume without downloading 180MB:
  modal run post_run_calc.py --stage text-stats
  modal volume get kv-audit-vol postrun ./post_run_remote
  python post_run_calc.py --v4-dir ./results --remote-text ./post_run_remote --out ./post_run

  # verify the machinery on fabricated data with known answers:
  python post_run_calc.py --selftest

  # ONLY the two v2 stages (skips everything else; seconds, no --data-dir
  # needed, though --data-dir makes the value-origin check exact):
  python post_run_calc.py --v4-dir ./results --out ./post_run --only v2

COST
====
Local stages: $0.  The optional Modal text-stats stage is CPU-only and runs
in a few minutes: well under $0.10.
"""

import argparse
import gzip
import itertools
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
LANGS = ["en", "pl", "zh", "ja", "ko", "vi", "sw"]
CONTENT_PRESSES = ["snapkv", "knorm"]
POSITIONAL_PRESS = "streaming_llm"
FLOOR = 0.60          # baseline below this -> ratio statistics unstable
N_BOOT = 2000
SEED = 42

# Fallback only.  The v4 module's own scorers are preferred and are used
# whenever --v4-script resolves; the two are cross-checked on real rows.
NONE_WORDS_FALLBACK = {
    "en": ["none"], "pl": ["brak"], "zh": ["无"], "ja": ["なし"],
    "ko": ["없음"], "vi": ["không có"], "sw": ["hakuna"],
}


# ---------------------------------------------------------------------------
# scorers
# ---------------------------------------------------------------------------
def _nums(pred):
    proc = unicodedata.normalize("NFKC", str(pred))
    return [n for n in re.findall(r"\d+", proc) if len(n) > 1]


def _has_none_word(lang, pred, none_words):
    low = unicodedata.normalize("NFKC", str(pred)).lower()
    return any(w.lower() in low for w in none_words.get(lang, []))


# ---------------------------------------------------------------------------
# v2: detection coding and value-origin machinery
# ---------------------------------------------------------------------------
VALUE_MIN_DIGITS = 6      # ONERULER values are 7-digit; 6 excludes years
NEAR_MAX_LEV = 2          # documented near-copy threshold


def candidate_values(pred):
    """The paper's response coding, made explicit and reusable: NFKC, then
    every digit run of length >= VALUE_MIN_DIGITS.  A trial is `answered`
    iff this returns anything."""
    proc = unicodedata.normalize("NFKC", str(pred))
    return re.findall(r"\d{%d,}" % VALUE_MIN_DIGITS, proc)


def answered(pred):
    return bool(candidate_values(pred))


def candidate_values_sep(pred):
    """Sensitivity variant: also accepts separator-formatted values such as
    '1,234,567' or '1 234 567'.  Reported as a robustness column, never as
    the headline coding."""
    proc = unicodedata.normalize("NFKC", str(pred))
    got = list(candidate_values(proc))
    for m in re.finditer(r"\d{1,3}(?:[,.\u00a0\s]\d{3}){1,4}", proc):
        digits = re.sub(r"\D", "", m.group())
        if len(digits) >= VALUE_MIN_DIGITS:
            got.append(digits)
    return list(dict.fromkeys(got))


def canon_num(x):
    """Canonicalize a numeric string (strips leading zeros, normalizes
    native-script digits already handled by NFKC upstream)."""
    try:
        return str(int(x))
    except (TypeError, ValueError):
        return str(x)


def lev(a, b, max_d=NEAR_MAX_LEV):
    """Levenshtein distance with an early exit once it exceeds max_d."""
    if abs(len(a) - len(b)) > max_d:
        return max_d + 1
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        if min(cur) > max_d:
            return max_d + 1
        prev = cur
    return prev[n]


def is_near_copy(v, pool, max_d=NEAR_MAX_LEV, min_share=VALUE_MIN_DIGITS):
    """NEAR-COPY RULE (documented; report this verbatim in the paper).

    v is a near copy of some value in `pool` iff EITHER
      (a) Levenshtein(v, p) <= max_d for some p in pool -- this covers a
          one-digit substitution, a dropped digit, a repeated digit and a
          transposition; OR
      (b) one of v, p is a prefix or suffix of the other and the shorter
          string is at least min_share digits long -- a truncated copy.

    Rule (a) alone is enough for almost every case in this data; (b) is
    kept because a truncated 7-digit value read out of the context is
    plainly not a fabrication."""
    for p in pool:
        if lev(v, p, max_d) <= max_d:
            return True
        s, t = (v, p) if len(v) <= len(p) else (p, v)
        if len(s) >= min_share and (t.startswith(s) or t.endswith(s)):
            return True
    return False


def near_copy_kind(v, pool, max_d=NEAR_MAX_LEV, min_share=VALUE_MIN_DIGITS):
    """As is_near_copy, but reports WHICH clause fired.

    v1.1: the paper quotes "262 of those are truncations -- the model emits
    the first six digits of the 7-digit needle and stops".  That number is
    not recoverable from a boolean near-copy flag, so the taxonomy has to
    carry the match type.  Returns one of
    "prefix" | "suffix" | "levenshtein" | None, preferring the truncation
    reading when both clauses fire (a 6-digit prefix of a 7-digit value is
    also at Levenshtein distance 1)."""
    best = None
    for p in pool:
        s_, t_ = (v, p) if len(v) <= len(p) else (p, v)
        if len(s_) >= min_share and t_.startswith(s_):
            return "prefix"
        if len(s_) >= min_share and t_.endswith(s_):
            best = best or "suffix"
        if lev(v, p, max_d) <= max_d:
            best = best or "levenshtein"
    return best


def wilson(k, n, z=1.96):
    """Wilson score interval -- correct at the small denominators this
    analysis has (some cells have n=13 answered trials)."""
    if n <= 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


class Scorers:
    """Official ONERULER scorers plus the two robustness variants.

    Prefers the v4 module's implementations (identical code path as the run
    that produced the data); falls back to a local reimplementation and
    reports which was used.  When both are available they are cross-checked
    on the real predictions and any disagreement is reported loudly.
    """

    def __init__(self, v4_script=None):
        self.mod = None
        self.source = "builtin-fallback"
        if v4_script:
            p = Path(v4_script)
            if p.exists():
                try:
                    sys.path.insert(0, str(p.resolve().parent))
                    self.mod = __import__(p.stem)
                    self.source = f"v4 module ({p.name})"
                except Exception as e:                       # pragma: no cover
                    print(f"  ! could not import {p}: {type(e).__name__}: {e}")
                    print("    falling back to built-in scorers")
        self.none_words = NONE_WORDS_FALLBACK
        if self.mod is not None:
            for attr in ("NONE_WORDS", "NONE_WORD", "NONE_ANSWERS"):
                if hasattr(self.mod, attr):
                    self.none_words = getattr(self.mod, attr)
                    break

    # -- official -----------------------------------------------------------
    def official(self, task, lang, gold, pred):
        if self.mod is not None and hasattr(self.mod, "score"):
            return bool(self.mod.score(task, lang, list(gold), str(pred)))
        return self._official_local(task, lang, gold, pred)

    def _official_local(self, task, lang, gold, pred):
        if "niah_none" in str(task):
            return _has_none_word(lang, pred, self.none_words)
        if _has_none_word(lang, pred, self.none_words):
            return False
        got = list(dict.fromkeys(_nums(pred)))
        try:
            return (len(got) == len(gold)
                    and {int(x) for x in got} == {int(g) for g in gold})
        except (TypeError, ValueError):
            return False

    # -- robustness ---------------------------------------------------------
    def strict_none(self, lang, gold, pred):
        """niah_none: official AND no multi-digit number in the output."""
        if self.mod is not None and hasattr(self.mod, "compare_none_strict"):
            return bool(self.mod.compare_none_strict(lang, list(gold),
                                                     str(pred)))
        return (self._official_local("niah_none", lang, gold, pred)
                and not _nums(pred))

    def lenient_single(self, lang, gold, pred):
        """niah_single: number-set match WITHOUT the none-word veto."""
        if self.mod is not None and hasattr(self.mod,
                                            "compare_numbers_lenient"):
            return bool(self.mod.compare_numbers_lenient(lang, list(gold),
                                                         str(pred)))
        got = list(dict.fromkeys(_nums(pred)))
        try:
            return (len(got) == len(gold)
                    and {int(x) for x in got} == {int(g) for g in gold})
        except (TypeError, ValueError):
            return False

    def robust(self, task, lang, gold, pred):
        if "niah_none" in str(task):
            return self.strict_none(lang, gold, pred)
        return self.lenient_single(lang, gold, pred)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def phi_from_counts(x, n):
    """Overdispersion of k proportions vs the binomial null at the pooled
    rate.  x = successes per group, n = items per group (scalar or array)."""
    x = np.asarray(x, float)
    n = np.asarray(n, float) * np.ones_like(x)
    k = len(x)
    pbar = x.sum() / n.sum()
    if pbar <= 0 or pbar >= 1:
        return np.nan, np.nan, pbar
    X2 = (((x - n * pbar) ** 2) / (n * pbar * (1 - pbar))).sum()
    return X2, X2 / (k - 1), pbar


def phi_bootstrap(per_lang_correct, B=N_BOOT, seed=SEED):
    """Item bootstrap on phi: resample items WITHIN each language."""
    rng = np.random.default_rng(seed)
    langs = sorted(per_lang_correct)
    arrs = [np.asarray(per_lang_correct[l], float) for l in langs]
    obs = phi_from_counts([a.sum() for a in arrs], [len(a) for a in arrs])[1]
    out = np.empty(B)
    for b in range(B):
        xs, ns = [], []
        for a in arrs:
            s = rng.integers(0, len(a), len(a))
            xs.append(a[s].sum())
            ns.append(len(a))
        out[b] = phi_from_counts(xs, ns)[1]
    ok = out[np.isfinite(out)]
    lo, hi = (np.percentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan))
    return obs, lo, hi, out


def amplification_paired(base_by_lang, comp_by_lang, B=N_BOOT, seed=SEED):
    """phi_comp / phi_base with a PAIRED bootstrap: the same resampled
    sample_ids are applied to the baseline and compressed cells within each
    language, because those cells share items."""
    rng = np.random.default_rng(seed)
    langs = sorted(set(base_by_lang) & set(comp_by_lang))
    bl = {l: np.asarray(base_by_lang[l], float) for l in langs}
    cp = {l: np.asarray(comp_by_lang[l], float) for l in langs}
    pb = phi_from_counts([bl[l].sum() for l in langs],
                         [len(bl[l]) for l in langs])[1]
    pc = phi_from_counts([cp[l].sum() for l in langs],
                         [len(cp[l]) for l in langs])[1]
    obs = pc / pb if (pb and np.isfinite(pb) and pb > 0) else np.nan
    out = np.empty(B)
    for b in range(B):
        xb, xc, ns = [], [], []
        for l in langs:
            m = min(len(bl[l]), len(cp[l]))
            s = rng.integers(0, m, m)
            xb.append(bl[l][:m][s].sum())
            xc.append(cp[l][:m][s].sum())
            ns.append(m)
        fb = phi_from_counts(xb, ns)[1]
        fc = phi_from_counts(xc, ns)[1]
        out[b] = fc / fb if (fb and np.isfinite(fb) and fb > 0) else np.nan
    ok = out[np.isfinite(out)]
    lo, hi = (np.percentile(ok, [2.5, 97.5]) if len(ok) else (np.nan, np.nan))
    return obs, pb, pc, lo, hi


def _ranks(v):
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float)
    r[order] = np.arange(1, len(v) + 1)
    # average ties
    for val in np.unique(v):
        m = v == val
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = _ranks(a), _ranks(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def spearman_exact_p(a, b, max_k=8):
    """Exact two-sided permutation p over all k! relabelings."""
    a = list(a)
    b = list(b)
    k = len(a)
    r = spearman(a, b)                 # SIGNED rho is what we report
    r0 = abs(r)                        # magnitude drives the two-sided p
    if not np.isfinite(r0):
        return np.nan, np.nan
    if k > max_k:
        return r, np.nan
    hit = tot = 0
    for perm in itertools.permutations(range(k)):
        tot += 1
        if abs(spearman(a, [b[i] for i in perm])) >= r0 - 1e-12:
            hit += 1
    return r, hit / tot


def bh_q(p):
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(o[::-1]):
        j = n - rank
        prev = min(prev, p[i] * n / j)
        q[i] = prev
    return q


def boot_ci_mean(x, B=N_BOOT, seed=SEED):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = np.array([x[rng.integers(0, len(x), len(x))].mean()
                   for _ in range(B)])
    return float(x.mean()), float(np.percentile(bs, 2.5)), \
        float(np.percentile(bs, 97.5))


# ---------------------------------------------------------------------------
# text statistics (tokenizer-free by design)
# ---------------------------------------------------------------------------
def extract_haystack(text):
    """Return the haystack (book text) from a ONERULER prompt.

    ONERULER wraps the haystack in a native-language tag pair, e.g.
    <text>...</text>, <文本>...</文本>, <글>...</글>.  Matching the tag pair by
    NAME is language-agnostic and needs no per-language table."""
    m = re.search(r"<([^/<>\n]{1,32})>\s*\n(.*?)\n\s*</\1>", text, re.S)
    if m:
        return m.group(2)
    m = re.search(r"<([^/<>\n]{1,32})>(.*?)</\1>", text, re.S)
    if m:
        return m.group(2)
    # fallback: between the first blank line and the last closing tag
    tags = list(re.finditer(r"</[^<>\n]{1,32}>", text))
    if tags:
        head = text.find("\n\n")
        if 0 <= head < tags[-1].start():
            return text[head + 2:tags[-1].start()]
    return None


def _sentences(text):
    parts = re.split(r'(?<=[.!?"。！？；;])\s*', text)
    return [p.strip() for p in parts if p.strip()]


def distinct_ngram_ratio(seq, n):
    if len(seq) < n:
        return 1.0
    grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
    return len(set(grams)) / len(grams)


def haystack_stats(hay):
    """Tokenizer-free redundancy statistics.

    Deliberately NOT token-based: token counts differ across languages by
    tokenizer fertility, and using them here would entangle the redundancy
    measure with the very variable §6.5 tests as a competing explanation.
    Characters, words, sentences and gzip are all fertility-neutral."""
    chars = hay
    words = hay.split()
    sents = _sentences(hay)
    uniq_sents = len(set(sents))
    raw = hay.encode("utf-8")
    gz = len(gzip.compress(raw, compresslevel=6)) / max(len(raw), 1)
    dup_frac = 1.0 - uniq_sents / max(len(sents), 1)
    counts = {}
    for s in sents:
        counts[s] = counts.get(s, 0) + 1
    return {
        "n_chars": len(chars),
        "n_words": len(words),
        "n_sentences": len(sents),
        "n_unique_sentences": uniq_sents,
        "dup_sentence_frac": round(dup_frac, 5),
        "max_sentence_repeats": max(counts.values()) if counts else 0,
        "implied_repeats": round(len(sents) / max(uniq_sents, 1), 4),
        "distinct_char_48gram": round(distinct_ngram_ratio(chars, 48), 5),
        "distinct_word_10gram": round(distinct_ngram_ratio(words, 10), 5),
        "gzip_ratio": round(gz, 5),
    }


NUM_RE = re.compile(r"\d+")


def haystack_numerals(hay, min_len=2):
    proc = unicodedata.normalize("NFKC", hay)
    return {n for n in NUM_RE.findall(proc) if len(n) >= min_len}


def text_stats_for_file(path, max_samples=24):
    """Redundancy + numeral stats + the numeral SET for one language/task."""
    rows, numerals = [], set()
    parsed = failed = 0
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_samples or not line.strip():
                if i >= max_samples:
                    break
                continue
            rec = json.loads(line)
            hay = extract_haystack(rec["input"])
            if hay is None:
                failed += 1
                continue
            parsed += 1
            rows.append(haystack_stats(hay))
            numerals |= haystack_numerals(hay)
    if not rows:
        return None, set(), parsed, failed
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    agg["n_samples_parsed"] = parsed
    agg["n_samples_unparsed"] = failed
    return agg, numerals, parsed, failed


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
ITEM_KEYS = ["model", "arm", "task", "lang", "press", "kept", "sample_id"]


def load_raw(v4_dir, dedup=True, expect_n=None):
    v4 = Path(v4_dir)
    pq = v4 / "results.parquet"
    if pq.exists():
        df = pd.read_parquet(pq)
        src = str(pq)
    else:
        raw = v4 / "raw"
        files = sorted(raw.glob("*.jsonl")) if raw.exists() else []
        if not files:
            raise SystemExit(
                f"no results.parquet and no raw/*.jsonl under {v4}.\n"
                f"Run:  modal volume get kv-audit-vol results ./results")
        frames = []
        for f in files:
            rs = [json.loads(l) for l in f.read_text(encoding="utf-8")
                  .splitlines() if l.strip()]
            if rs:
                frames.append(pd.DataFrame(rs))
        df = pd.concat(frames, ignore_index=True)
        src = f"{len(files)} raw jsonl files"
    n_raw = len(df)
    n_toy = 0
    if "toy" in df.columns:
        n_toy = int(df["toy"].astype(bool).sum())
        df = df[~df["toy"].astype(bool)]
        if n_toy:
            # v1.1: report this explicitly.  The parquet ships 29,442 rows of
            # which 42 are toy smoke-test rows -- NOT re-runs.  They reuse
            # real sample_ids, so this filter MUST precede the dedup below;
            # any dedup that runs first (including a timestamp-based one) can
            # substitute smoke-test outputs for real data.
            print(f"  toy filter: dropped {n_toy} smoke-test rows "
                  f"({n_raw} -> {len(df)})")
    if dedup:
        # GUARD, not a fix: on the current results.parquet this drops
        # NOTHING (all 294 real cells are exactly n=100).  It exists because
        # cells were topped up in place during the campaign (30 -> 100
        # baselines), so a re-run or a merged volume could reintroduce
        # duplicate (cell, sample_id) rows, which would silently reweight a
        # language.  Note the toy smoke-test rows reuse the same sample_ids
        # as real rows -- so this MUST run after the toy filter above, or it
        # would delete real data.  --no-dedup disables it.
        before = len(df)
        df = df.drop_duplicates(subset=ITEM_KEYS, keep="first")
        if before != len(df):
            print(f"  dedup: dropped {before - len(df)} duplicate "
                  f"(cell, sample_id) rows [--no-dedup to disable]")
    df["correct"] = df["correct"].astype(int)
    for c in ("kept", "depth"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"  loaded {len(df):,} rows from {src}  "
          f"({df.groupby(['model', 'arm', 'task']).ngroups} facets, "
          f"{df.config_hash.nunique()} cells)")
    if expect_n is not None and len(df) != expect_n:
        raise SystemExit(
            f"FATAL: expected {expect_n:,} real rows after the toy filter and "
            f"dedup, got {len(df):,}.  Do not proceed -- the corpus is not the "
            f"one the paper reports.  (--expect-n 0 disables this check.)")
    return df


CELL_KEYS = ["model", "arm", "task", "lang", "press", "kept"]


def cell_vectors(df, model, arm, task, press, kept):
    """{lang: np.array of per-item correct} for one cell, ordered by
    sample_id so that paired resampling lines up across cells."""
    s = df[(df.model == model) & (df.arm == arm) & (df.task == task)
           & (df.press == press) & (df.kept == kept)]
    return {l: g.sort_values("sample_id").correct.to_numpy()
            for l, g in s.groupby("lang")}


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_cells(df, out):
    rows = []
    for keys, g in df.groupby(CELL_KEYS):
        m, lo, hi = boot_ci_mean(g.correct.to_numpy())
        rows.append(dict(zip(CELL_KEYS, keys), n=len(g), acc=round(m, 4),
                         ci_low=round(lo, 4), ci_high=round(hi, 4)))
    c = pd.DataFrame(rows).sort_values(CELL_KEYS)
    c.to_csv(out / "cells.csv", index=False)
    print(f"  cells.csv: {len(c)} cells recomputed from raw items")
    return c


def stage_dispersion(df, cells, out):
    disp, amp = [], []
    for (m, a, t, p, k), g in df.groupby(["model", "arm", "task",
                                          "press", "kept"]):
        vec = {l: gg.sort_values("sample_id").correct.to_numpy()
               for l, gg in g.groupby("lang")}
        if len(vec) < 3:
            continue
        obs, lo, hi, _ = phi_bootstrap(vec)
        X2, _, pbar = phi_from_counts([v.sum() for v in vec.values()],
                                      [len(v) for v in vec.values()])
        accs = np.array([v.mean() for v in vec.values()])
        disp.append(dict(model=m, arm=a, task=t, press=p, kept=k,
                         k_langs=len(vec), mean_acc=round(float(pbar), 4),
                         raw_spread=round(float(accs.max() - accs.min()), 4),
                         X2=round(float(X2), 1), phi=round(float(obs), 2),
                         phi_ci_low=round(float(lo), 2),
                         phi_ci_high=round(float(hi), 2)))
    d = pd.DataFrame(disp).sort_values(["model", "arm", "task", "press",
                                        "kept"])
    d.to_csv(out / "dispersion.csv", index=False)

    for (m, a, t), g in df.groupby(["model", "arm", "task"]):
        base = cell_vectors(df, m, a, t, "none", 1.0)
        if not base:
            continue
        for (p, k), _ in g[g.press != "none"].groupby(["press", "kept"]):
            comp = cell_vectors(df, m, a, t, p, k)
            if len(comp) < 3:
                continue
            obs, pb, pc, lo, hi = amplification_paired(base, comp)
            amp.append(dict(model=m, arm=a, task=t, press=p, kept=k,
                            phi_base=round(float(pb), 2),
                            phi_comp=round(float(pc), 2),
                            amplification=round(float(obs), 3),
                            ci_low=round(float(lo), 3),
                            ci_high=round(float(hi), 3),
                            ci_excludes_1=bool(lo > 1 or hi < 1)))
    ampdf = pd.DataFrame(amp)
    ampdf.to_csv(out / "amplification.csv", index=False)
    print(f"  dispersion.csv: {len(d)} cells   "
          f"amplification.csv: {len(ampdf)} contrasts (paired CI)")
    return d, ampdf


def stage_ordering(df, out, B=1000):
    """Every ordering contrast the paper needs, with exact permutation p,
    leave-one-out, floor gating, and an ITEM bootstrap on rho itself.

    The item bootstrap is the honest replacement for the draft's
    'robust to rounding?' worry: the two models saw the SAME generated
    items, so sample_ids are resampled jointly across models."""
    rows = []
    rng = np.random.default_rng(SEED)

    def base_acc(model, arm, task):
        v = cell_vectors(df, model, arm, task, "none", 1.0)
        return {l: a.mean() for l, a in v.items()}

    combos = sorted({(m, a, t, p, k) for m, a, t, p, k in
                     df[df.press != "none"][["model", "arm", "task", "press",
                                             "kept"]].itertuples(index=False)})

    # ---- cross-model transfer at matched (arm, task, press, kept) ----------
    models = sorted(df.model.unique())
    for arm in sorted(df.arm.unique()):
        for task in sorted(df.task.unique()):
            for press in sorted(df[df.press != "none"].press.unique()):
                for kept in sorted(df[df.press != "none"].kept.unique()):
                    va = cell_vectors(df, models[0], arm, task, press, kept)
                    vb = (cell_vectors(df, models[-1], arm, task, press, kept)
                          if len(models) > 1 else {})
                    langs = sorted(set(va) & set(vb))
                    if len(langs) < 4:
                        continue
                    a = [va[l].mean() for l in langs]
                    b = [vb[l].mean() for l in langs]
                    r, p = spearman_exact_p(a, b)
                    # floor gating on either model's baseline
                    ba = base_acc(models[0], arm, task)
                    bb = base_acc(models[-1], arm, task)
                    keep = [l for l in langs
                            if ba.get(l, 1) >= FLOOR and bb.get(l, 1) >= FLOOR]
                    rf = pf = np.nan
                    if len(keep) >= 4:
                        rf, pf = spearman_exact_p([va[l].mean() for l in keep],
                                                  [vb[l].mean() for l in keep])
                    # leave-one-out
                    loo = {}
                    for l in langs:
                        sub = [x for x in langs if x != l]
                        loo[l] = round(spearman([va[x].mean() for x in sub],
                                                [vb[x].mean() for x in sub]), 3)
                    # item bootstrap on rho (joint resample of sample_ids)
                    bs = []
                    for _ in range(B):
                        aa, bb2 = [], []
                        for l in langs:
                            n = min(len(va[l]), len(vb[l]))
                            s = rng.integers(0, n, n)
                            aa.append(va[l][:n][s].mean())
                            bb2.append(vb[l][:n][s].mean())
                        bs.append(spearman(aa, bb2))
                    bs = np.array([x for x in bs if np.isfinite(x)])
                    rows.append(dict(
                        contrast="cross_model", arm=arm, task=task,
                        press=press, kept=kept, k_langs=len(langs),
                        langs=",".join(langs), rho=round(r, 4),
                        exact_p=round(p, 4) if np.isfinite(p) else np.nan,
                        rho_boot_lo=round(float(np.percentile(bs, 2.5)), 3),
                        rho_boot_hi=round(float(np.percentile(bs, 97.5)), 3),
                        frac_boot_below_crit=round(
                            float((bs < 0.75).mean()), 3),
                        rho_floor_gated=round(rf, 4) if np.isfinite(rf)
                        else np.nan,
                        p_floor_gated=round(pf, 4) if np.isfinite(pf)
                        else np.nan,
                        k_floor_gated=len(keep),
                        leave_one_out=json.dumps(loo)))

    # ---- baseline -> compressed (does compression scale or reorder?) -------
    for (m, arm, task, press, kept) in combos:
        vb = cell_vectors(df, m, arm, task, "none", 1.0)
        vc = cell_vectors(df, m, arm, task, press, kept)
        langs = sorted(set(vb) & set(vc))
        if len(langs) < 4:
            continue
        a = [vb[l].mean() for l in langs]
        b = [vc[l].mean() for l in langs]
        r, p = spearman_exact_p(a, b)
        rows.append(dict(contrast="baseline_to_compressed", model=m, arm=arm,
                         task=task, press=press, kept=kept, k_langs=len(langs),
                         langs=",".join(langs), rho=round(r, 4),
                         exact_p=round(p, 4) if np.isfinite(p) else np.nan))

    # ---- cross-press within model (is there a 'fragile language' axis?) ----
    for m in models:
        for arm in sorted(df.arm.unique()):
            for task in sorted(df.task.unique()):
                for kept in sorted(df[df.press != "none"].kept.unique()):
                    v1 = cell_vectors(df, m, arm, task, "snapkv", kept)
                    v2 = cell_vectors(df, m, arm, task, "knorm", kept)
                    langs = sorted(set(v1) & set(v2))
                    if len(langs) < 4:
                        continue
                    a = [v1[l].mean() for l in langs]
                    b = [v2[l].mean() for l in langs]
                    r, p = spearman_exact_p(a, b)
                    rows.append(dict(contrast="cross_press", model=m, arm=arm,
                                     task=task, kept=kept, k_langs=len(langs),
                                     langs=",".join(langs), rho=round(r, 4),
                                     exact_p=round(p, 4) if np.isfinite(p)
                                     else np.nan,
                                     note="floor: both cells near 0 -> ranks "
                                          "are noise" if max(a + b) < 0.1
                                     else ""))
    o = pd.DataFrame(rows)
    if not o.empty and "exact_p" in o.columns:
        cm = o.contrast == "cross_model"
        if cm.any():
            o.loc[cm, "q_bh"] = bh_q(o.loc[cm, "exact_p"].fillna(1).to_numpy())
    o.to_csv(out / "ordering.csv", index=False)
    print(f"  ordering.csv: {len(o)} contrasts "
          f"(exact permutation p, LOO, floor-gated, item-bootstrap CI)")
    return o


def stage_scorers(df, sc, out):
    d = df.copy()
    d["gold_l"] = d["gold"].map(lambda g: list(g) if g is not None else [])
    d["official"] = [sc.official(t, l, g, p) for t, l, g, p in
                     zip(d.task, d.lang, d.gold_l, d.pred)]
    d["robust"] = [sc.robust(t, l, g, p) for t, l, g, p in
                   zip(d.task, d.lang, d.gold_l, d.pred)]
    mismatch = int((d["official"].astype(int) != d["correct"]).sum())
    print(f"  scorer cross-check: {mismatch}/{len(d)} rows where the "
          f"recomputed official scorer disagrees with the stored label"
          + ("  <-- investigate" if mismatch else "  (exact match)"))
    rows = []
    for keys, g in d.groupby(CELL_KEYS):
        is_none = "niah_none" in str(dict(zip(CELL_KEYS, keys))["task"])
        rows.append(dict(zip(CELL_KEYS, keys), n=len(g),
                         acc_official=round(g.correct.mean(), 4),
                         acc_recomputed=round(g["official"].mean(), 4),
                         acc_robust=round(g["robust"].mean(), 4),
                         delta_robust=round(g.correct.mean()
                                            - g["robust"].mean(), 4),
                         # v1.1: name WHICH robustness scorer produced the
                         # delta.  Scorers.robust() dispatches to two
                         # different rules by task -- strict_none on
                         # niah_none (official AND no >=6-digit value: the
                         # none-word substring defect of paper section 5.2)
                         # and lenient_single on niah_single (number match
                         # WITHOUT the none-word veto: a different question
                         # entirely).  v1 summed the two into one count.
                         robust_scorer=("strict_none" if is_none
                                        else "lenient_single"),
                         delta_strict_none=(round(g.correct.mean()
                                                  - g["robust"].mean(), 4)
                                            if is_none else float("nan")),
                         delta_lenient_single=(float("nan") if is_none
                                               else round(g.correct.mean()
                                                          - g["robust"].mean(),
                                                          4))))
    s = pd.DataFrame(rows).sort_values("delta_robust",
                                       key=lambda c: -c.abs())
    s.to_csv(out / "scorers.csv", index=False)

    # ---- v1.1: the section 5.2 statistic, with the right denominator ------
    # The none-word substring defect can only fire on absent-needle cells.
    # v1 reported "23 of 294 cells" -- 294 counts 98 qa-absent + 98 qa-present
    # + 98 aware-present, and 11 of the 23 were present cells moved by the
    # OTHER scorer.  Report the two populations separately, always.
    strict = s[s.robust_scorer == "strict_none"]
    lenient = s[s.robust_scorer == "lenient_single"]
    for tag, sub, col in (("strict_none  (niah_none cells)", strict,
                           "delta_strict_none"),
                          ("lenient_single (niah_single cells)", lenient,
                           "delta_lenient_single")):
        if not len(sub):
            continue
        b5 = sub[sub[col].abs() >= 0.05]
        b10 = sub[sub[col].abs() >= 0.10]
        by_model = b5.groupby("model").size().to_dict()
        print(f"  {tag}: {len(b5)}/{len(sub)} cells shift >= 0.05 "
              f"({len(b10)} >= 0.10); by model {by_model}; "
              f"largest {sub[col].abs().max():.3f}")
    strict.to_csv(out / "scorers_strict_none.csv", index=False)
    lenient.to_csv(out / "scorers_lenient_single.csv", index=False)
    return s, d


def stage_errors(df, sc, out, numerals_by_lang=None):
    """Failure modes, and — the D4 measurement — whether a wrong number was
    RETRIEVED from the haystack (distractor enrichment) or INVENTED."""
    rows = []
    for r in df.itertuples(index=False):
        pred = str(r.pred)
        nums = _nums(pred)
        gold = list(r.gold) if r.gold is not None else []
        if r.correct:
            mode = "correct"
        elif "niah_none" in str(r.task):
            distr = set(map(str, r.distractors)) if getattr(
                r, "distractors", None) is not None else set()
            canon = set()
            for x in distr:
                try:
                    canon.add(str(int(x)))
                except ValueError:
                    canon.add(x)
            got = {str(int(n)) for n in nums} if nums else set()
            if got & canon:
                mode = "false_presence_distractor"
            elif nums:
                mode = "false_presence_hallucinated"
            else:
                mode = "other"
        else:
            if _has_none_word(r.lang, pred, sc.none_words):
                mode = "false_absence"
            elif nums:
                mode = "wrong_number"
            else:
                mode = "other"
        in_hay = np.nan
        if (numerals_by_lang and mode == "wrong_number"
                and r.lang in numerals_by_lang):
            hay = numerals_by_lang[r.lang]
            in_hay = any(n in hay for n in nums)
        rows.append(dict(model=r.model, arm=r.arm, task=r.task, lang=r.lang,
                         press=r.press, kept=r.kept, mode=mode,
                         wrong_number_in_haystack=in_hay))
    e = pd.DataFrame(rows)
    tab = (e.groupby(CELL_KEYS + ["mode"]).size().rename("n").reset_index())
    tot = tab.groupby(CELL_KEYS)["n"].transform("sum")
    tab["rate"] = (tab["n"] / tot).round(4)
    tab.to_csv(out / "errors.csv", index=False)
    if numerals_by_lang:
        w = e[(e["mode"] == "wrong_number")
              & e.wrong_number_in_haystack.notna()]
        if len(w):
            d4 = (w.groupby(["model", "lang", "press", "kept"])
                  .wrong_number_in_haystack.agg(["mean", "size"])
                  .rename(columns={"mean": "frac_from_haystack",
                                   "size": "n_wrong_number"})
                  .round(4).reset_index())
            d4.to_csv(out / "distractor_vs_hallucination.csv", index=False)
            print(f"  distractor_vs_hallucination.csv: {len(d4)} cells "
                  f"(overall {w.wrong_number_in_haystack.mean():.1%} of "
                  f"wrong numbers ARE haystack numerals)")
    print(f"  errors.csv: {tab['mode'].nunique()} modes x {len(tab)} rows")
    return tab


def load_item_numerals(data_dir, ctx_len=32768, min_digits=VALUE_MIN_DIGITS):
    """{(lang, task, sample_id): set of every >= min_digits digit run in the
    ENTIRE serialized input} -- prompt, instructions, question and haystack.

    This is what makes the "appears nowhere in the input" claim checkable at
    the item level.  Requires the generated data (--data-dir).  Without it
    the value-origin stage falls back to per-item distractors plus the
    language-level numeral union from haystack_numerals.json, which is an
    approximation and is labelled as such in the output."""
    if not data_dir:
        return {}
    idx = {}
    for lang in LANGS:
        for task in ("niah_single", "niah_none"):
            hits = list(Path(data_dir).glob(
                f"**/oneruler/{lang}/{ctx_len}/{task}/validation.jsonl"))
            if not hits:
                continue
            with open(hits[0], encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue          # tolerate a truncated final line
                    proc = unicodedata.normalize("NFKC", rec["input"])
                    idx[(lang, task, rec["index"])] = {
                        canon_num(x) for x in
                        re.findall(r"\d{%d,}" % min_digits, proc)}
    if idx:
        print(f"    per-item full-input numerals loaded for {len(idx)} items")
    return idx


ORIGIN_ORDER = ["TARGET_EXACT", "DISTRACTOR_EXACT", "OTHER_INPUT_EXACT",
                "TARGET_NEAR_COPY", "DISTRACTOR_NEAR_COPY",
                "OTHER_INPUT_NEAR_COPY", "ENTIRELY_NOVEL"]


def _classify_candidate(v, gold, distr, other):
    """Most-grounded-wins classification of one candidate value."""
    if v in gold:
        return "TARGET_EXACT"
    if v in distr:
        return "DISTRACTOR_EXACT"
    if v in other:
        return "OTHER_INPUT_EXACT"
    if gold and is_near_copy(v, gold):
        return "TARGET_NEAR_COPY"
    if distr and is_near_copy(v, distr):
        return "DISTRACTOR_NEAR_COPY"
    if other and is_near_copy(v, other):
        return "OTHER_INPUT_NEAR_COPY"
    return "ENTIRELY_NOVEL"


def stage_value_origin(df, out, numerals=None, item_numerals=None,
                       chance_draws=20000, seed=SEED):
    """THE v1-blocking computation.

    The draft says that under KNorm at 25% "a majority [of wrong answers]
    are fabricated values that appear nowhere in the input".  The existing
    pipeline only ever asked "is this value an EXACT match to something in
    the input?" -- so every corrupted copy (one dropped digit, one
    substituted digit) was filed as a fabrication.  This stage adds the
    near-copy tier and reports exact / near / novel with Wilson CIs and raw
    numerators, at trial level (what the paper quotes) and candidate level.

    A Monte-Carlo chance rate is also computed: the fraction of RANDOM
    7-digit values that the near-copy rule would misfile as grounded.  If
    that rate is ~0.1% and the observed near-copy rate is ~50%, the tier is
    measuring something real."""
    rng = np.random.default_rng(seed)
    numerals = numerals or {}
    item_numerals = item_numerals or {}
    exact_mode = "per-item full input" if item_numerals else (
        "per-item distractors + language-level haystack union"
        if numerals else "per-item distractors ONLY")
    print(f"    other-input reference: {exact_mode}")

    trial_rows, cand_rows = [], []
    for r in df.itertuples(index=False):
        vals = [canon_num(v) for v in candidate_values(r.pred)]
        if not vals:
            continue                       # abstention: not an answered trial
        gold = {canon_num(g) for g in (list(r.gold)
                                       if r.gold is not None else [])}
        d = getattr(r, "distractors", None)
        try:
            distr = {canon_num(x) for x in list(d)} if d is not None else set()
        except TypeError:
            distr = set()
        key = (r.lang, str(r.task), r.sample_id)
        if key in item_numerals:
            other = item_numerals[key] - gold - distr
        else:
            other = {canon_num(x) for x in numerals.get(r.lang, set())
                     if len(str(x)) >= VALUE_MIN_DIGITS} - gold - distr
        labs, kinds = [], []
        for v in dict.fromkeys(vals):
            lab = _classify_candidate(v, gold, distr, other)
            labs.append(lab)
            # v1.1: which near-copy clause fired, so truncations are countable
            if lab.endswith("NEAR_COPY"):
                pool = (gold if lab.startswith("TARGET") else
                        distr if lab.startswith("DISTRACTOR") else other)
                kind = near_copy_kind(v, pool) or "levenshtein"
            elif lab.endswith("EXACT"):
                kind = "exact"
            else:
                kind = "none"
            kinds.append((lab, kind))
            cand_rows.append(dict(model=r.model, arm=r.arm, task=r.task,
                                  lang=r.lang, press=r.press, kept=r.kept,
                                  sample_id=r.sample_id, value=v, origin=lab,
                                  match_type=kind))
        # trial label = the most grounded of its candidates
        trial = min(labs, key=ORIGIN_ORDER.index)
        # ---- v1.1: the SECOND metric -------------------------------------
        # `origin` above is MOST-GROUNDED-WINS: a trial that emits the true
        # value AND an invented one is filed as TARGET_EXACT.  That is the
        # right label for a composition table, but it is NOT what the
        # sentence "contains a value appearing nowhere in the input" means.
        # The claim needs an existential over the trial's candidates, so we
        # record it separately.  Never report one metric with the other's
        # wording.
        labset = set(labs)
        trial_rows.append(dict(
            model=r.model, arm=r.arm, task=r.task, lang=r.lang,
            press=r.press, kept=r.kept, sample_id=r.sample_id,
            origin=trial, n_candidates=len(labs),
            contains_any_exact=bool(labset & set(ORIGIN_ORDER[:3])),
            contains_any_near=bool(labset & set(ORIGIN_ORDER[3:6])),
            contains_any_novel=("ENTIRELY_NOVEL" in labset),
            # target-specific flags: these drive the present-trial outcome
            # composition (Figure 2), where the categories are defined
            # relative to the TRUE value, not to any input value.
            contains_target_exact=("TARGET_EXACT" in labset),
            contains_target_near=("TARGET_NEAR_COPY" in labset),
            target_near_is_truncation=any(
                l == "TARGET_NEAR_COPY" and k in ("prefix", "suffix")
                for l, k in kinds),
            target_near_is_prefix=any(
                l == "TARGET_NEAR_COPY" and k == "prefix" for l, k in kinds),
            official_correct=int(getattr(r, "correct", 0))))
    if not trial_rows:
        print("  value_origin: no answered trials found")
        return None, None, None
    tr = pd.DataFrame(trial_rows)
    cn = pd.DataFrame(cand_rows)
    cn.to_csv(out / "value_origin_cands.csv", index=False)
    # v1.1: persist the per-trial rows (one line per answered trial, with the
    # contains_any_* indicators).  stage_figure2 consumes this, and it is the
    # artifact a reviewer needs to recheck either metric.
    tr.to_csv(out / "value_origin_trials_items.csv", index=False)

    def _summarize(g):
        n = len(g)
        exact = int(g.origin.isin(ORIGIN_ORDER[:3]).sum())
        near = int(g.origin.isin(ORIGIN_ORDER[3:6]).sum())
        novel = int((g.origin == "ENTIRELY_NOVEL").sum())
        d_ex = int((g.origin == "DISTRACTOR_EXACT").sum())
        p, lo, hi = wilson(novel, n)
        pn, nlo, nhi = wilson(near, n)
        # v1.1: the existential metric, reported alongside, never instead
        any_nov = int(g.contains_any_novel.sum())
        any_near = int(g.contains_any_near.sum())
        any_ex = int(g.contains_any_exact.sum())
        ap_, alo, ahi = wilson(any_nov, n)
        return pd.Series(dict(
            n_answered=n,
            n_exact_copy=exact, n_near_copy=near, n_novel=novel,
            n_distractor_exact=d_ex,
            exact_rate=round(exact / n, 4),
            near_rate=round(pn, 4), near_ci_low=round(nlo, 4),
            near_ci_high=round(nhi, 4),
            novel_rate=round(p, 4), novel_ci_low=round(lo, 4),
            novel_ci_high=round(hi, 4),
            n_target_near_truncation=int(g.target_near_is_truncation.sum())
            if "target_near_is_truncation" in g else 0,
            n_target_near_prefix=int(g.target_near_is_prefix.sum())
            if "target_near_is_prefix" in g else 0,
            n_any_exact=any_ex, n_any_near=any_near, n_any_novel=any_nov,
            any_novel_rate=round(ap_, 4),
            any_novel_ci_low=round(alo, 4),
            any_novel_ci_high=round(ahi, 4)))

    keys = ["model", "arm", "task", "press", "kept"]
    summ = tr.groupby(keys, observed=True).apply(
        _summarize, include_groups=False).reset_index()
    summ["other_input_reference"] = exact_mode
    summ.to_csv(out / "value_origin_trials.csv", index=False)

    bylang = tr.groupby(keys + ["lang"], observed=True).apply(
        _summarize, include_groups=False).reset_index()
    bylang.to_csv(out / "value_origin_by_lang.csv", index=False)

    # ---- chance rate of the near-copy rule ---------------------------------
    pools = []
    for r in df.itertuples(index=False):
        d = getattr(r, "distractors", None)
        try:
            p = {canon_num(x) for x in list(d)} if d is not None else set()
        except TypeError:
            p = set()
        if p:
            pools.append(p)
    ch_rows = []
    if pools:
        widths = sorted({len(x) for p in pools for x in p}) or [7]
        for w in widths:
            hit = 0
            for _ in range(chance_draws):
                pool = pools[rng.integers(0, len(pools))]
                v = str(rng.integers(10 ** (w - 1), 10 ** w))
                if v in pool or is_near_copy(v, pool):
                    hit += 1
            p, lo, hi = wilson(hit, chance_draws)
            ch_rows.append(dict(value_width=w, draws=chance_draws, hits=hit,
                                chance_grounded_rate=round(p, 6),
                                ci_low=round(lo, 6), ci_high=round(hi, 6)))
        pd.DataFrame(ch_rows).to_csv(out / "value_origin_chance.csv",
                                     index=False)
    print(f"  value_origin_trials.csv: {len(summ)} cells "
          f"({len(tr):,} answered trials, {len(cn):,} candidate values)")
    if ch_rows:
        print(f"    near-copy rule chance false-positive rate: "
              f"{ch_rows[-1]['chance_grounded_rate']:.4%}")
    return summ, bylang, (pd.DataFrame(ch_rows) if ch_rows else None)


def stage_figure2(df, tr, out):
    """Present-trial outcome composition, with the categories spelled out.

    Figure 2 of v1 stacks four bands: correct / wrong-but-near-copy-of-the-
    true-value / wrong-and-not-a-near-copy / abstention.  Those do not
    reconstruct from the records, because a fifth population exists and the
    caption never says where it goes: trials that DO contain the true value
    but are scored incorrect by the official scorer (formatting, or the
    none-word veto).  There are 5 such trials in Llama KNorm-25% and 23 in
    Llama SnapKV-25%.  We emit all five bands; the caption must state which
    band absorbs `answered_contains_target_not_credited`."""
    keys = ["model", "arm", "task", "press", "kept"]
    tr_idx = tr.set_index(["model", "arm", "task", "lang", "press", "kept",
                           "sample_id"])
    rows = []
    for k, g in df[df.task == "niah_single"].groupby(keys, observed=True):
        n = len(g)
        idx = pd.MultiIndex.from_arrays(
            [g.model, g.arm, g.task, g.lang, g.press, g.kept, g.sample_id])
        sub = tr_idx.reindex(idx)
        answered = sub.origin.notna()
        correct = g.correct.to_numpy().astype(bool)
        tgt_ex = sub.contains_target_exact.fillna(False).to_numpy().astype(bool)
        tgt_nr = sub.contains_target_near.fillna(False).to_numpy().astype(bool)
        ans = answered.to_numpy()
        band_correct = int(correct.sum())
        band_uncredited = int((ans & ~correct & tgt_ex).sum())
        band_near = int((ans & ~correct & ~tgt_ex & tgt_nr).sum())
        band_other = int((ans & ~correct & ~tgt_ex & ~tgt_nr).sum())
        band_abstain = int((~ans).sum())
        rows.append(dict(zip(keys, k), n_trials=n,
                         correct=band_correct,
                         answered_contains_target_not_credited=band_uncredited,
                         wrong_near_copy_of_target=band_near,
                         wrong_not_near_copy=band_other,
                         abstention=band_abstain,
                         frac_correct=round(band_correct / n, 4),
                         frac_uncredited=round(band_uncredited / n, 4),
                         frac_near=round(band_near / n, 4),
                         frac_other=round(band_other / n, 4),
                         frac_abstain=round(band_abstain / n, 4),
                         bands_sum=band_correct + band_uncredited + band_near
                         + band_other + band_abstain))
    f2 = pd.DataFrame(rows)
    bad = f2[f2.bands_sum != f2.n_trials]
    if len(bad):
        print(f"  WARNING: {len(bad)} cells whose bands do not sum to n")
    f2.to_csv(out / "figure2_composition.csv", index=False)
    print(f"  figure2_composition.csv: {len(f2)} present-trial cells "
          f"(5 bands, incl. the uncredited-but-contains-target band)")
    return f2


# v1 printed claims, transcribed from the manuscript, for the diff report.
# (paper_location, description, printed_value)
V1_PRINTED = [
    ("Abstract / sec 4.2", "Llama KNorm-25% absent: answers containing a "
     "value appearing nowhere in the input", "1.4% [0.6, 3.3]  (n=5/348)"),
    ("Abstract / sec 4.2", "Llama KNorm-25% absent: near-copies of a "
     "distractor", "176/348 (51%)"),
    ("Sec 4.2", "Llama KNorm-25% present: answers lacking the true value "
     "that are near-copies of it", "272 of 282"),
    ("Sec 4.2", "...of which truncations", "262"),
    ("Sec 4.2", "entirely novel values, every Llama condition",
     "<= 4% of answered trials"),
    ("Sec 4.2", "Qwen KNorm-25% present novel", "26% [18, 36]"),
    ("Sec 4.2", "Qwen SnapKV-25% present novel", "28% [23, 33]"),
    ("Sec 5.2", "cells shifting >= 0.05 under strict scoring",
     "23 of 294 (19 involve Llama)"),
    ("Table 2", "Llama Streaming-50% absent exact/near/novel", "355 / 0 / 10"),
    ("Table 2", "Qwen SnapKV-25% absent exact/near/novel", "37 / 3 / 4"),
    ("Fig 2", "Llama KNorm-25% present bands "
     "(correct / near / not-near / abstain)", "0.21 / 0.40 / <0.05 / 0.38"),
    ("Fig 2", "Llama SnapKV-25% present bands",
     "0.35 / 0.06 / - / 0.57"),
]


def stage_v11_claims(out, vo, scor, f2):
    """The v1.1 numeric diff: every paper-facing number this release changes,
    printed old -> new, so the manuscript edit is a checklist."""
    L = ["# v1.1 numeric diff\n",
         "Every number below is recomputed from the per-item records by "
         "`post_run_calc.py`. The `v1 printed` column is transcribed from "
         "the manuscript. Edit the paper from this table; do not hand-copy "
         "from anywhere else.\n"]
    A = L.append

    def cell(task, model, press, kept):
        if vo is None:
            return None
        m = vo[(vo.arm == "qa") & (vo.task == task) & (vo.model == model)
               & (vo.press == press) & (vo.kept == kept)]
        return m.iloc[0] if len(m) else None

    A("\n## 1. The two value-origin metrics\n")
    A("`most-grounded` = v1's mutually exclusive trial label (precedence "
      "exact > near > novel). `any-novel` = the trial emitted at least one "
      "value matching nothing in the input. **Only `any-novel` supports the "
      "phrase \"appears nowhere in the input\".**\n")
    if vo is not None and len(vo):
        cols = ["model", "task", "press", "kept", "n_answered",
                "n_exact_copy", "n_near_copy", "n_novel", "novel_rate",
                "n_any_novel", "any_novel_rate", "any_novel_ci_low",
                "any_novel_ci_high"]
        q = vo[vo.arm == "qa"]
        A(_md(q[[c for c in cols if c in q.columns]]))
        c = cell("niah_none", "llama31-8b", "knorm", 0.25)
        if c is not None:
            A(f"\n**Abstract sentence.** Llama KNorm-25%, answered absent "
              f"trials: most-grounded-novel {int(c.n_novel)}/"
              f"{int(c.n_answered)} = {c.novel_rate:.1%}; "
              f"**any-novel {int(c.n_any_novel)}/{int(c.n_answered)} = "
              f"{c.any_novel_rate:.1%} "
              f"[{c.any_novel_ci_low:.1%}, {c.any_novel_ci_high:.1%}]**. "
              f"v1 printed the first number with the second number's "
              f"wording.\n")
        lla = q[(q.model == "llama31-8b")]
        if len(lla):
            A(f"\n**\"<= 4% in every Llama condition\".** Under most-grounded "
              f"the max is {lla.novel_rate.max():.1%}; under any-novel it is "
              f"{lla.any_novel_rate.max():.1%}. Scope the sentence to the "
              f"taxonomy or replace the number.\n")

    A("\n## 2. Strict scoring, by scorer and task\n")
    if scor is not None and "robust_scorer" in scor.columns:
        for tag, col in (("strict_none", "delta_strict_none"),
                         ("lenient_single", "delta_lenient_single")):
            sub = scor[scor.robust_scorer == tag]
            if not len(sub):
                continue
            b5 = sub[sub[col].abs() >= 0.05]
            b10 = sub[sub[col].abs() >= 0.10]
            bym = ", ".join(f"{k}: {v}" for k, v in
                            sorted(b5.groupby("model").size().items()))
            A(f"- **{tag}** — {len(b5)} of {len(sub)} cells shift >= 0.05 "
              f"({len(b10)} >= 0.10). By model: {bym or 'none'}.\n")
        A("\nSection 5.2 must quote the `strict_none` row only. The "
          "`lenient_single` row answers a different question and was summed "
          "into v1's count of 23.\n")

    A("\n## 3. Present-trial outcome composition (Figure 2)\n")
    if f2 is not None and len(f2):
        sel = f2[(f2.arm == "qa") & (f2.kept.isin([0.25, 1.0]))]
        A(_md(sel[["model", "press", "kept", "n_trials", "correct",
                   "answered_contains_target_not_credited",
                   "wrong_near_copy_of_target", "wrong_not_near_copy",
                   "abstention"]]))
        A("\n`answered_contains_target_not_credited` is the band v1's caption "
          "does not name: the output contains the true value but the official "
          "scorer rejects it. Decide where it belongs and say so in the "
          "caption.\n")

    A("\n## 4. v1 printed claims to re-check\n")
    A("| paper location | claim | v1 printed |")
    A("|---|---|---|")
    for loc, desc, val in V1_PRINTED:
        A(f"| {loc} | {desc} | {val} |")
    A("\nCompare each against sections 1-3 above before editing.\n")

    (out / "v11_numeric_diff.md").write_text("\n".join(L), encoding="utf-8")
    print(f"  v11_numeric_diff.md written")


def stage_protocol(df, out, B=N_BOOT, seed=SEED):
    """Query-agnostic (arm=qa) vs query-aware (arm=joint) present-trial
    operating points.  No new GPU time: the joint arm was swept for
    niah_single at every press and budget on both models.

    Reports, per (model, press, budget, arm): present-task accuracy
    (macro-averaged over languages, the leaderboard statistic), coverage
    (= the answer rate H under the scorer-independent coding) and
    precision-given-answer (pooled), plus the joint-minus-qa difference
    with a PAIRED item bootstrap -- the two arms share sample_ids within
    each language, so the same resampled indices are applied to both."""
    d = df[df.task == "niah_single"].copy()
    if d.empty or d.arm.nunique() < 2:
        print("  protocol: SKIPPED -- need both arms on niah_single")
        return None, None
    d["ans"] = d.pred.map(answered).astype(int)
    rng = np.random.default_rng(seed)

    def _vecs(model, arm, press, kept):
        s = d[(d.model == model) & (d.arm == arm) & (d.press == press)
              & (d.kept == kept)]
        return {l: g.sort_values("sample_id")[["correct", "ans"]].to_numpy()
                for l, g in s.groupby("lang")}

    def _point(v):
        """(macro accuracy, macro coverage, pooled precision, n_answered)."""
        if not v:
            return (np.nan,) * 3 + (0,)
        acc = float(np.mean([a[:, 0].mean() for a in v.values()]))
        cov = float(np.mean([a[:, 1].mean() for a in v.values()]))
        allv = np.concatenate(list(v.values()))
        na = int(allv[:, 1].sum())
        prec = float(allv[allv[:, 1] == 1][:, 0].mean()) if na else np.nan
        return acc, cov, prec, na

    rows, lang_rows = [], []
    for (m, p, k), _ in d.groupby(["model", "press", "kept"]):
        vq, vj = _vecs(m, "qa", p, k), _vecs(m, "joint", p, k)
        if not vq or not vj:
            continue
        aq, cq, pq, nq = _point(vq)
        aj, cj, pj, nj = _point(vj)
        # paired item bootstrap on the joint-minus-qa differences
        langs = sorted(set(vq) & set(vj))
        da, dc, dp = np.empty(B), np.empty(B), np.empty(B)
        for b in range(B):
            sq, sj = {}, {}
            for l in langs:
                n = min(len(vq[l]), len(vj[l]))
                s = rng.integers(0, n, n)
                sq[l] = vq[l][:n][s]
                sj[l] = vj[l][:n][s]
            a1, c1, p1, _ = _point(sq)
            a2, c2, p2, _ = _point(sj)
            da[b], dc[b], dp[b] = a2 - a1, c2 - c1, p2 - p1

        def _ci(x):
            x = x[np.isfinite(x)]
            return (round(float(np.percentile(x, 2.5)), 4),
                    round(float(np.percentile(x, 97.5)), 4)) \
                if len(x) else (np.nan, np.nan)

        alo, ahi = _ci(da)
        clo, chi = _ci(dc)
        plo, phi_ = _ci(dp)
        rows.append(dict(
            model=m, press=p, kept=k,
            acc_qa=round(aq, 4), acc_joint=round(aj, 4),
            d_acc=round(aj - aq, 4), d_acc_lo=alo, d_acc_hi=ahi,
            acc_sig=bool(alo > 0 or ahi < 0),
            coverage_qa=round(cq, 4), coverage_joint=round(cj, 4),
            d_coverage=round(cj - cq, 4), d_cov_lo=clo, d_cov_hi=chi,
            cov_sig=bool(clo > 0 or chi < 0),
            prec_qa=round(pq, 4), prec_joint=round(pj, 4),
            d_prec=round(pj - pq, 4), d_prec_lo=plo, d_prec_hi=phi_,
            prec_sig=bool(plo > 0 or phi_ < 0),
            n_answered_qa=nq, n_answered_joint=nj))
        for l in langs:
            gq, gj = vq[l], vj[l]
            lang_rows.append(dict(
                model=m, press=p, kept=k, lang=l,
                acc_qa=round(float(gq[:, 0].mean()), 4),
                acc_joint=round(float(gj[:, 0].mean()), 4),
                coverage_qa=round(float(gq[:, 1].mean()), 4),
                coverage_joint=round(float(gj[:, 1].mean()), 4),
                n=int(min(len(gq), len(gj)))))
    pr = pd.DataFrame(rows).sort_values(["model", "press", "kept"])
    pr.to_csv(out / "protocol_contrast.csv", index=False)
    pl = pd.DataFrame(lang_rows)
    pl.to_csv(out / "protocol_by_lang.csv", index=False)
    print(f"  protocol_contrast.csv: {len(pr)} conditions "
          f"(paired bootstrap B={B}), protocol_by_lang.csv: {len(pl)} rows")
    return pr, pl


def stage_depth(df, out):
    d = df[df.depth.notna() & (df.depth >= 0)].copy()
    if d.empty:
        print("  depth: no usable depth column")
        return None, None
    bal = []
    ds = d[(d.task == "niah_single") & (d.press == "none")]
    if len(ds):
        dd = ds.drop_duplicates(subset=["lang", "sample_id"])
        en = dd[dd.lang == "en"].depth.to_numpy()
        for l, g in dd.groupby("lang"):
            v = g.depth.to_numpy()
            if len(en):
                allv = np.concatenate([v, en])
                ks = float(np.max(np.abs(
                    np.searchsorted(np.sort(v), allv, "right") / len(v)
                    - np.searchsorted(np.sort(en), allv, "right") / len(en))))
            else:
                ks = np.nan
            bal.append(dict(lang=l, n=len(v), mean=round(float(v.mean()), 4),
                            p10=round(float(np.percentile(v, 10)), 3),
                            p50=round(float(np.percentile(v, 50)), 3),
                            p90=round(float(np.percentile(v, 90)), 3),
                            ks_vs_en=round(ks, 4),
                            mean_diff_vs_en=round(
                                float(v.mean() - en.mean()), 4)
                            if len(en) else np.nan))
    b = pd.DataFrame(bal)
    b.to_csv(out / "depth_balance.csv", index=False)

    d["bin"] = pd.cut(d.depth, [0, .25, .5, .75, 1.0001],
                      labels=["0-.25", ".25-.5", ".5-.75", ".75-1"])
    dc = (d.groupby(["model", "arm", "task", "press", "kept", "lang", "bin"],
                    observed=True).correct.agg(["mean", "size"])
          .rename(columns={"mean": "acc", "size": "n"}).round(4).reset_index())
    dc.to_csv(out / "depth.csv", index=False)

    law = []
    for (m, a, t, k), g in d[d.press == POSITIONAL_PRESS].groupby(
            ["model", "arm", "task", "kept"]):
        base = df[(df.model == m) & (df.arm == a) & (df.task == t)
                  & (df.press == "none")]
        for l, gg in g.groupby("lang"):
            bl = base[base.lang == l].correct.mean()
            if np.isnan(bl):
                continue
            law.append(dict(model=m, arm=a, task=t, kept=k, lang=l,
                            predicted=round(bl * k, 4),
                            observed=round(gg.correct.mean(), 4),
                            abs_err=round(abs(bl * k - gg.correct.mean()), 4)))
    lw = pd.DataFrame(law)
    lw.to_csv(out / "positional_law.csv", index=False)
    if len(lw):
        print(f"  depth.csv + depth_balance.csv + positional_law.csv "
              f"({(lw.abs_err <= .05).sum()}/{len(lw)} cells within 0.05, "
              f"mean |err| {lw.abs_err.mean():.3f})")
    return b, lw


def stage_arms(df, out):
    rows = []
    b = df[df.press == "none"]
    for (m, t, l), g in b.groupby(["model", "task", "lang"]):
        arms = {a: gg.set_index("sample_id").correct
                for a, gg in g.groupby("arm")}
        if len(arms) < 2:
            continue
        (a1, v1), (a2, v2) = sorted(arms.items())
        idx = v1.index.intersection(v2.index)
        if not len(idx):
            continue
        x, y = v1.loc[idx].to_numpy(), v2.loc[idx].to_numpy()
        disc = int((x != y).sum())
        b01 = int(((x == 0) & (y == 1)).sum())
        b10 = int(((x == 1) & (y == 0)).sum())
        # exact two-sided binomial on discordant pairs
        p = np.nan
        if disc:
            from math import comb
            tail = sum(comb(disc, i) for i in range(min(b01, b10) + 1))
            p = min(1.0, 2 * tail / (2 ** disc))
        rows.append(dict(model=m, task=t, lang=l, arm_a=a1, arm_b=a2,
                         n_paired=len(idx), acc_a=round(float(x.mean()), 4),
                         acc_b=round(float(y.mean()), 4),
                         n_discordant=disc, b_a0b1=b01, b_a1b0=b10,
                         exact_p=round(p, 4) if np.isfinite(p) else np.nan,
                         violates=bool(disc > 0)))
    a = pd.DataFrame(rows)
    a.to_csv(out / "arm_invariance.csv", index=False)
    if len(a):
        v = a[a.violates]
        print(f"  arm_invariance.csv: {len(v)}/{len(a)} baseline cells with "
              f"ANY per-item disagreement (must be 0 by construction)")
    return a


def stage_detection(df, out):
    rows = []
    for (m, a, p, k), g in df[df.arm == "qa"].groupby(
            ["model", "arm", "press", "kept"]):
        for l, gg in g.groupby("lang"):
            hit = gg[gg.task == "niah_single"].correct.mean()
            rej = gg[gg.task == "niah_none"].correct.mean()
            if np.isnan(hit) or np.isnan(rej):
                continue
            rows.append(dict(model=m, press=p, kept=k, lang=l,
                             hit_rate=round(hit, 4),
                             correct_rejection=round(rej, 4),
                             naive_avg=round((hit + rej) / 2, 4),
                             youden_j=round(hit + rej - 1, 4),
                             balanced_acc=round((hit + rej) / 2, 4)))
    d = pd.DataFrame(rows)
    d.to_csv(out / "detection.csv", index=False)
    print(f"  detection.csv: {len(d)} (model, press, budget, language) rows")
    return d


def stage_fertility(df, v4_dir, out):
    rows = []
    for f in Path(v4_dir).glob("fertility_*.csv"):
        model = f.stem.replace("fertility_", "")
        fert = pd.read_csv(f).set_index("lang")
        col = ("tokens_per_byte_eval"
               if "tokens_per_byte_eval" in fert.columns
               and fert["tokens_per_byte_eval"].notna().any()
               else "tokens_per_byte")
        for (a, t, p, k), g in df[(df.model == model)
                                  & (df.press != "none")].groupby(
                ["arm", "task", "press", "kept"]):
            acc = g.groupby("lang").correct.mean()
            base = (df[(df.model == model) & (df.arm == a) & (df.task == t)
                       & (df.press == "none")].groupby("lang").correct.mean())
            langs = [l for l in acc.index if l in fert.index
                     and pd.notna(fert.loc[l, col])]
            if len(langs) < 4:
                continue
            x = [float(fert.loc[l, col]) for l in langs]
            for outcome, y in (("accuracy", [float(acc[l]) for l in langs]),
                               ("retention", [float(acc[l] / base[l])
                                              if base.get(l, 0) > 0 else np.nan
                                              for l in langs])):
                if any(not np.isfinite(v) for v in y):
                    continue
                r, pp = spearman_exact_p(x, y)
                loo = {l: round(spearman([x[i] for i in range(len(langs))
                                          if i != j],
                                         [y[i] for i in range(len(langs))
                                          if i != j]), 3)
                       for j, l in enumerate(langs)}
                rows.append(dict(model=model, arm=a, task=t, press=p, kept=k,
                                 outcome=outcome, fertility_col=col,
                                 k_langs=len(langs), rho=round(r, 4),
                                 exact_p=round(pp, 4), leave_one_out=
                                 json.dumps(loo)))
    fdf = pd.DataFrame(rows)
    if len(fdf):
        fdf["q_bh"] = bh_q(fdf.exact_p.to_numpy())
        fdf["sig_bh"] = fdf.q_bh < 0.05
    fdf.to_csv(out / "fertility.csv", index=False)
    print(f"  fertility.csv: {len(fdf)} tests, "
          f"{int(fdf.sig_bh.sum()) if len(fdf) else 0} survive BH")
    return fdf


def read_possibly_concatenated(path):
    """Read a redundancy table that may have other files glued onto it.

    `modal volume get <volume> <dir> <dest>` can collapse a directory into a
    single file by CONCATENATING its contents, so redundancy.csv arrives with
    haystack_numerals.json appended.  Keep the leading run of lines that have
    the header's field count, then try to parse the remainder as the numerals
    JSON.  Returns (DataFrame|None, {lang: set(numerals)})."""
    import io
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = [l for l in txt.splitlines()]
    if not lines:
        return None, {}
    ncol = lines[0].count(",") + 1
    keep, i = [lines[0]], 1
    while (i < len(lines) and lines[i].strip()
           and lines[i].count(",") + 1 == ncol):
        keep.append(lines[i])
        i += 1
    red = None
    try:
        red = pd.read_csv(io.StringIO("\n".join(keep)))
    except Exception:
        return None, {}
    numerals = {}
    rest = "\n".join(lines[i:]).strip()
    if rest:
        j = rest.find("{")
        if j >= 0:
            try:
                numerals = {k: set(v) for k, v in
                            json.loads(rest[j:]).items()}
            except Exception:
                pass
    if len(keep) < len(lines):
        print(f"    note: {path} contained {len(lines)} lines; used the "
              f"first {len(keep)} as the table"
              + (f" and recovered numerals for {len(numerals)} languages "
                 f"from the remainder" if numerals else ""))
    return red, numerals


def stage_redundancy(df, data_dir, remote_text, out):
    """Per-language haystack redundancy + the link to content-press
    retention.  THE confound test for the Korean result."""
    red = None
    numerals = {}
    if remote_text:
        p = Path(remote_text)
        if not p.exists():
            print(f"    ! --remote-text {p} does not exist "
                  f"(cwd={Path.cwd()})")
        else:
            if p.is_file():
                # Modal's `volume get <dir>` can collapse a directory into a
                # single extensionless file, so do NOT require a .csv suffix.
                cand = [p]
            else:
                cand = sorted(p.rglob("redundancy.csv"))
            if cand:
                try:
                    red, _num = read_possibly_concatenated(cand[0])
                    if _num and not numerals:
                        numerals = _num
                    if red is None or "lang" not in red.columns:
                        raise ValueError(
                            "no 'lang' column; got "
                            + str(list(red.columns)[:6] if red is not None
                                  else None))
                    print(f"    read {cand[0]} ({len(red)} languages)")
                except Exception as e:
                    head = ""
                    try:
                        head = cand[0].read_text(
                            encoding="utf-8", errors="replace")[:160]
                    except Exception:
                        pass
                    print(f"    ! {cand[0]} is not a readable redundancy "
                          f"table: {type(e).__name__}: {e}")
                    print(f"      first bytes: {head!r}")
                    red = None
            else:
                found = sorted(str(x.relative_to(p))
                               for x in p.rglob("*") if x.is_file())[:25]
                print(f"    ! no redundancy.csv under {p}")
                print(f"      files present: {found if found else '(none)'}")
                print(f"      pass the CSV directly if it is elsewhere, e.g."
                      f"  --remote-text path/to/redundancy.csv")
            nf = sorted(p.rglob("haystack_numerals.json")) if p.is_dir() \
                else list((p.parent).glob("haystack_numerals.json"))
            if nf:
                numerals = {k: set(v) for k, v in
                            json.loads(nf[0].read_text(
                                encoding="utf-8")).items()}
                print(f"    read {nf[0]} ({len(numerals)} languages)")
    if red is None and data_dir:
        rows = []
        for lang in LANGS:
            hits = list(Path(data_dir).glob(
                f"**/oneruler/{lang}/*/niah_single/validation.jsonl"))
            if not hits:
                continue
            agg, nums, parsed, failed = text_stats_for_file(hits[0])
            if agg is None:
                print(f"    ! {lang}: could not parse any haystack")
                continue
            numerals[lang] = nums
            rows.append(dict(lang=lang, **{k: round(v, 5)
                                           for k, v in agg.items()}))
            print(f"    {lang}: implied_repeats={agg['implied_repeats']:.2f} "
                  f"dup_sentences={agg['dup_sentence_frac']:.3f} "
                  f"gzip={agg['gzip_ratio']:.4f} ({parsed} parsed)")
        red = pd.DataFrame(rows) if rows else None
    if red is None or red.empty:
        if not (remote_text or data_dir):
            print("  redundancy: SKIPPED — neither --data-dir nor "
                  "--remote-text was given. Get it with either:\n"
                  "      modal volume get kv-audit-vol data ./data  (~180MB)\n"
                  "      modal run post_run_calc.py --stage text-stats")
        else:
            print("  redundancy: SKIPPED — a source WAS given but no usable "
                  "redundancy.csv was read (see the diagnostics above).")
        return None, None, numerals
    red.to_csv(out / "redundancy.csv", index=False)

    # link: redundancy vs content-press retention, per (model, press, kept)
    rows = []
    for (m, p, k), g in df[(df.arm == "qa") & (df.task == "niah_single")
                           & df.press.isin(CONTENT_PRESSES)].groupby(
            ["model", "press", "kept"]):
        acc = g.groupby("lang").correct.mean()
        base = (df[(df.model == m) & (df.arm == "qa")
                   & (df.task == "niah_single")
                   & (df.press == "none")].groupby("lang").correct.mean())
        langs = [l for l in red.lang if l in acc.index and base.get(l, 0) > 0]
        if len(langs) < 4:
            continue
        ret = [float(acc[l] / base[l]) for l in langs]
        for col in ("implied_repeats", "dup_sentence_frac",
                    "distinct_char_48gram", "gzip_ratio"):
            if col not in red.columns:
                continue
            x = [float(red.set_index("lang").loc[l, col]) for l in langs]
            r, pp = spearman_exact_p(x, ret)
            loo = {l: round(spearman([x[i] for i in range(len(langs))
                                      if i != j],
                                     [ret[i] for i in range(len(langs))
                                      if i != j]), 3)
                   for j, l in enumerate(langs)}
            rows.append(dict(model=m, press=p, kept=k, redundancy_metric=col,
                             k_langs=len(langs), rho=round(r, 4),
                             exact_p=round(pp, 4),
                             leave_one_out=json.dumps(loo)))
    link = pd.DataFrame(rows)
    if len(link):
        link["q_bh"] = bh_q(link.exact_p.to_numpy())
    link.to_csv(out / "redundancy_link.csv", index=False)
    print(f"  redundancy.csv ({len(red)} languages) + "
          f"redundancy_link.csv ({len(link)} correlations)")
    return red, link, numerals


def stage_power(out, ks=(7, 10, 12, 15, 20, 26), rhos=(0.5, 0.6, 0.7, 0.8),
                budget=700, B=4000, seed=SEED):
    """Power of the cross-model ordering test at equal GPU cost.
    budget = k * n items per condition (700 = 7 languages x 100)."""
    rng = np.random.default_rng(seed)
    crit = {}
    for k in ks:
        if k <= 8:
            base = list(range(k))
            null = [abs(spearman(base, list(p)))
                    for p in itertools.permutations(base)]
            crit[k] = float(np.percentile(null, 95))
        else:
            null = [abs(spearman(rng.permutation(k), rng.permutation(k)))
                    for _ in range(20000)]
            crit[k] = float(np.percentile(null, 95))
    rows = []
    for k in ks:
        n = max(10, budget // k)
        for rt in rhos:
            hits = 0
            for _ in range(B):
                z1 = rng.normal(size=k)
                z2 = rt * z1 + np.sqrt(max(1e-9, 1 - rt ** 2)) * rng.normal(size=k)
                p1 = np.clip(0.5 + 0.28 * z1, 0.02, 0.98)
                p2 = np.clip(0.5 + 0.28 * z2, 0.02, 0.98)
                a = rng.binomial(n, p1) / n
                b = rng.binomial(n, p2) / n
                if abs(spearman(a, b)) >= crit[k]:
                    hits += 1
            rows.append(dict(k_langs=k, items_per_cell=n,
                             critical_abs_rho=round(crit[k], 3),
                             rho_true=rt, power=round(hits / B, 3)))
    p = pd.DataFrame(rows)
    p.to_csv(out / "power.csv", index=False)
    print(f"  power.csv: {len(p)} (k, rho) design points at equal cost")
    return p


# ---------------------------------------------------------------------------
# report

def _md(df, index=False):
    """Markdown table without requiring the optional `tabulate` package.
    Falls back to a hand-rolled pipe table so the report always renders."""
    try:
        return df.to_markdown(index=index)
    except Exception:
        d = df.reset_index() if index else df
        cols = [str(c) for c in d.columns]
        rows = [[("" if v is None else str(v)) for v in r]
                for r in d.astype(object).where(d.notna(), "").values]
        w = [max(len(cols[i]), *(len(r[i]) for r in rows)) if rows
             else len(cols[i]) for i in range(len(cols))]
        out = ["| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cols)) + " |",
               "|" + "|".join("-" * (w[i] + 2) for i in range(len(cols))) + "|"]
        for r in rows:
            out.append("| " + " | ".join(r[i].ljust(w[i])
                                         for i in range(len(cols))) + " |")
        return "\n".join(out)


# ---------------------------------------------------------------------------
def _v2_lines(vo, vch, prot):
    """Shared body of the v2 report, so the standalone and the appended
    version can never drift apart."""
    L = []
    A = L.append
    A("\n## 7. Value origin: exact copy / near copy / entirely novel\n")
    if vo is not None and len(vo):
        cols = ["model", "press", "kept", "n_answered", "n_exact_copy",
                "n_near_copy", "n_novel", "near_rate", "novel_rate",
                "novel_ci_low", "novel_ci_high"]
        n = vo[(vo.arm == "qa") & (vo.task == "niah_none")]
        if len(n):
            A("**Answered ABSENT trials** (the population behind the "
              "draft's distractor-copy / fabrication numbers):\n")
            A(_md(n[[c for c in cols if c in n.columns]]))
        s = vo[(vo.arm == "qa") & (vo.task == "niah_single")]
        if len(s):
            A("\n**Answered PRESENT trials** (`TARGET_EXACT` = correct "
              "retrieval; near-copies here are corrupted values):\n")
            A(_md(s[[c for c in cols if c in s.columns]]))
        A("\nA value is a NEAR COPY iff it is within Levenshtein distance 2 "
          "of some value present in the input, or is a prefix/suffix of one "
          "sharing at least 6 digits. `ENTIRELY_NOVEL` is the only category "
          "for which the phrase \"appears nowhere in the input\" is "
          "defensible.\n")
        if vch is not None and len(vch):
            r = float(vch.chance_grounded_rate.iloc[-1])
            A(f"\nChance check: a RANDOM value of the same width is "
              f"misfiled as grounded {r:.3%} of the time "
              f"(`value_origin_chance.csv`), so the near-copy tier is not "
              f"an artefact of a loose rule.\n")
        if vo is not None and "other_input_reference" in vo.columns:
            A(f"\nOther-input reference used: "
              f"**{vo.other_input_reference.iloc[0]}**. Pass `--data-dir` "
              f"for per-item full-input matching.\n")
    else:
        A("_Not run._\n")
    A("\n## 8. Query-agnostic vs query-aware (present trials)\n")
    if prot is not None and len(prot):
        A(_md(prot[["model", "press", "kept", "acc_qa", "acc_joint", "d_acc",
                    "d_acc_lo", "d_acc_hi", "coverage_qa", "coverage_joint",
                    "prec_qa", "prec_joint", "d_prec", "prec_sig"]]))
        A("\n`qa` compresses the context only (question prefilled "
          "afterwards, kvpress-leaderboard semantics); `joint` compresses "
          "the whole templated prompt, which puts SnapKV's observation "
          "window on the question -- a query-AWARE variant, though not the "
          "canonical one (KNorm can evict question tokens there, and the "
          "budget denominator is the prompt rather than the context, a "
          "difference of ~0.4%). Differences carry a paired item bootstrap; "
          "the two arms share sample_ids within each language.\n")
    else:
        A("_Not run: needs both arms on niah_single._\n")
    return L


def write_v2_report(out, vo, vch, prot):
    L = ["# post_run_calc v2 — value origin and protocol contrast\n",
         "Both stages run on the EXISTING per-item records. No new "
         "generations.\n"]
    L += _v2_lines(vo, vch, prot)
    (out / "post_run_v2_report.md").write_text("\n".join(L), encoding="utf-8")


def write_report(out, cells, disp, amp, order, scor, red, link, law, arms,
                 fert, power, det, vo=None, vch=None, prot=None):
    L = []
    A = L.append
    A("# post_run_calc — recomputed results\n")
    A("All numbers below are computed from **raw per-item records**, not "
      "from reconstructed cell means. Intervals are bootstrap over items "
      "(paired within language where the cells share items).\n")

    A("\n## 1. Dispersion and amplification (replaces §6.3 + Appendix A)\n")
    if amp is not None and len(amp):
        a = amp[(amp.arm == "qa") & (amp.task == "niah_single")].copy()
        A(_md(a))
        A("\n`amplification` is phi_comp/phi_base with a paired bootstrap CI. "
          "Any value whose CI excludes 1 is a real change in cross-language "
          "dispersion; anything else is not.\n")
    A("\n## 2. Ordering / reallocation (replaces §6.4)\n")
    if order is not None and len(order):
        cm = order[order.contrast == "cross_model"]
        if len(cm):
            cols = [c for c in ["arm", "task", "press", "kept", "k_langs",
                                "rho", "exact_p", "rho_boot_lo", "rho_boot_hi",
                                "rho_floor_gated", "p_floor_gated",
                                "k_floor_gated"] if c in cm.columns]
            A(_md(cm[cols]))
            A("\n`rho_boot_lo/hi` is the item bootstrap on rho itself — the "
              "honest replacement for the draft's rounding worry. "
              "`rho_floor_gated` drops languages whose baseline is below "
              f"{FLOOR} in either model (this removes Swahili on Qwen, whose "
              "baseline is 0.48 and which the draft's own floor rule "
              "excludes but §6.4 uses).\n")
            A("\nSee the `leave_one_out` column of `ordering.csv`: if the "
              "correlation depends on Korean, that must be stated, because "
              "§3 of the redundancy analysis below gives a specific "
              "alternative explanation for Korean.\n")
    A("\n## 3. Haystack redundancy — the confound test\n")
    if red is not None and len(red):
        A(_md(red))
        A("\nRedundancy is measured **tokenizer-free** (sentences, characters, "
          "gzip) on purpose: a token-based measure would entangle redundancy "
          "with fertility, which is a competing explanation tested "
          "separately.\n")
        if link is not None and len(link):
            A("\n**Redundancy vs content-press retention:**\n")
            A(_md(link))
            A("\nIf these correlations are strong, the cross-model ordering "
              "of §6.4 is at least partly a property of the ONERULER corpus "
              "rather than of the languages: both models read the *same* "
              "generated text, so cross-model replication cannot distinguish "
              "a language effect from a corpus effect.\n")
    else:
        A("_Not run: supply `--data-dir` or `--remote-text`._\n")
    A("\n## 4. Scorer sensitivity (§6.7)\n")
    if scor is not None and len(scor):
        big = scor[scor.delta_robust.abs() >= 0.05]
        A(f"{len(big)} cells differ by >= 0.05 between the official and the "
          f"robustness scorer. Largest offenders:\n")
        A(_md(big.head(15)))
    A("\n## 5. Controls\n")
    if law is not None and len(law):
        A(f"- **Positional law** (streaming acc ~= baseline x kept): "
          f"{(law.abs_err <= .05).sum()}/{len(law)} cells within 0.05, "
          f"mean |err| {law.abs_err.mean():.3f}, max {law.abs_err.max():.3f}\n")
    if arms is not None and len(arms):
        v = arms[arms.violates]
        A(f"- **Arm invariance** (press=none must be identical across arms): "
          f"{len(v)}/{len(arms)} cells show per-item disagreement\n")
        if len(v):
            A(_md(v[["model", "task", "lang", "n_discordant", "exact_p"]]))
    if fert is not None and len(fert):
        A(f"\n- **Fertility**: {int(fert.sig_bh.sum())}/{len(fert)} tests "
          f"survive BH. Sign flips across arms/presses indicate no stable "
          f"relationship.\n")
    A("\n## 6. Design / power (Appendix D2)\n")
    if power is not None and len(power):
        A(_md(power.pivot(index=["k_langs", "items_per_cell"],
                          columns="rho_true", values="power"), index=True))
        A("\nEqual GPU cost across rows. Power scales with the number of "
          "languages, not items per language.\n")
    if vo is not None or prot is not None:
        L += _v2_lines(vo, vch, prot)
    (out / "post_run_report.md").write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------
def selftest():
    print("=== selftest: fabricated data with known answers ===")
    ok = True
    # phi against a hand-computed case
    X2, phi, pbar = phi_from_counts([100, 98, 77, 84, 92, 90, 98], 100)
    print(f"  phi(v4 llama baseline) = {phi:.2f} "
          f"(expected 8.92, mean {pbar:.4f} expected 0.9129)")
    ok &= abs(phi - 8.92) < 0.05 and abs(pbar - 0.9129) < 1e-3
    X2c, phic, _ = phi_from_counts([45, 21, 62, 0, 90, 3, 24], 100)
    print(f"  phi(v4 llama snapkv@.25) = {phic:.2f} (expected 47.03), "
          f"amplification {phic/phi:.2f} (expected 5.28)")
    ok &= abs(phic - 47.03) < 0.1 and abs(phic / phi - 5.28) < 0.05
    # spearman + exact p against the known v4 cross-model value
    lla = [0.45, 0.00, 0.24, 0.21, 0.62, 0.03, 0.90]
    qwe = [0.07, 0.04, 0.11, 0.06, 0.73, 0.00, 0.08]
    r, p = spearman_exact_p(lla, qwe)
    print(f"  cross-model rho = {r:.3f} exact p = {p:.4f} "
          f"(expected 0.786 / 0.0480)")
    ok &= abs(r - 0.786) < 1e-3 and abs(p - 0.048) < 1e-3
    # redundancy metrics
    uniq = " ".join(f"w{i} x{i} y{i} z{i}." for i in range(400))
    rep = " ".join((f"a{i} b{i} c{i} d{i}." for i in range(100))) 
    rep4 = " ".join([rep] * 4)
    su, sr = haystack_stats(uniq), haystack_stats(rep4)
    print(f"  redundancy: unique implied_repeats={su['implied_repeats']:.2f} "
          f"vs 4x-repeated {sr['implied_repeats']:.2f} "
          f"(gzip {su['gzip_ratio']:.3f} vs {sr['gzip_ratio']:.3f})")
    ok &= su["implied_repeats"] < 1.05 and sr["implied_repeats"] > 3.5
    ok &= sr["gzip_ratio"] < su["gzip_ratio"]
    # haystack extraction on each template shape
    shapes = [("text", "Hello world."), ("文本", "你好世界。"),
              ("テキスト", "こんにちは。"), ("글", "안녕하세요."),
              ("Văn bản", "Xin chào."), ("Maandishi", "Habari."),
              ("tekst", "Dzien dobry.")]
    for tag, body in shapes:
        prompt = f"Instr\n\n<{tag}>\n{body}\n</{tag}>\n\n<Q> key </Q>"
        got = extract_haystack(prompt)
        if got is None or body not in got:
            print(f"  ! haystack extraction FAILED for <{tag}>")
            ok = False
    print(f"  haystack extraction: {len(shapes)} template shapes")
    # bh monotone
    q = bh_q([0.001, 0.02, 0.04, 0.5])
    ok &= all(q[i] <= q[i + 1] + 1e-12 for i in range(3))

    # ---- v2: detection coding, Levenshtein, near-copy, Wilson -------------
    ok &= candidate_values("<answer> 2867825 </answer>") == ["2867825"]
    ok &= candidate_values("in 1998 he wrote 12345") == []      # year floor
    ok &= answered("value is 4744854") and not answered("없음")
    ok &= candidate_values("\uff12\uff18\uff16\uff17\uff18\uff12\uff15") == \
        ["2867825"]                                             # NFKC digits
    ok &= candidate_values_sep("1,234,567") == ["1234567"]
    ok &= candidate_values("1,234,567") == []                   # headline
    print(f"  response coding: digit-run floor, NFKC, separator variant")
    ok &= lev("2867825", "2867825") == 0
    ok &= lev("286782", "2867825") == 1                          # dropped
    ok &= lev("2867815", "2867825") == 1                         # substituted
    ok &= lev("2868725", "2867825") == 2                         # transposed
    ok &= lev("9999999", "2867825") > NEAR_MAX_LEV
    pool = {"2867825", "4744854"}
    ok &= is_near_copy("286782", pool) and is_near_copy("2867815", pool)
    ok &= not is_near_copy("1111111", pool)
    print(f"  near-copy rule: drop/substitute/transpose in, random out")
    ok &= _classify_candidate("7", {"7"}, set(), set()) == "TARGET_EXACT"
    ok &= _classify_candidate("2867815", set(), {"2867825"}, set()) == \
        "DISTRACTOR_NEAR_COPY"
    ok &= _classify_candidate("1111111", set(), {"2867825"}, set()) == \
        "ENTIRELY_NOVEL"
    p, lo, hi = wilson(8, 13)
    ok &= lo < p < hi and 0.35 < lo < 0.40 and 0.82 < hi < 0.88
    print(f"  wilson(8/13) = {p:.2f} [{lo:.2f}, {hi:.2f}] "
          f"(small-n interval stays wide)")
    print(f"\n  SELFTEST {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# optional Modal stage (only needed if the ONERULER data is not downloaded)
# ---------------------------------------------------------------------------
try:
    import modal
    _image = (modal.Image.debian_slim(python_version="3.11")
              .pip_install("numpy", "pandas"))
    _app = modal.App("post-run-calc")
    _vol = modal.Volume.from_name("kv-audit-vol", create_if_missing=False)

    @_app.function(image=_image, volumes={"/vol": _vol}, timeout=30 * 60)
    def _remote_text_stats(ctx_len: int = 32768):
        """Compute haystack redundancy + numeral sets on the volume, so the
        180MB of generated data never has to be downloaded."""
        import json as _json
        rows, numerals = [], {}
        for lang in LANGS:
            p = Path(f"/vol/data/oneruler/{lang}/{ctx_len}/niah_single/"
                     f"validation.jsonl")
            if not p.exists():
                print(f"{lang}: missing {p}")
                continue
            agg, nums, parsed, failed = text_stats_for_file(str(p))
            if agg is None:
                print(f"{lang}: no haystack parsed")
                continue
            numerals[lang] = sorted(nums)
            rows.append(dict(lang=lang, **{k: round(v, 5)
                                           for k, v in agg.items()}))
            print(f"{lang}: implied_repeats={agg['implied_repeats']:.2f} "
                  f"dup_sent={agg['dup_sentence_frac']:.3f} "
                  f"gzip={agg['gzip_ratio']:.4f} parsed={parsed} "
                  f"unparsed={failed}")
        os.makedirs("/vol/postrun", exist_ok=True)
        pd.DataFrame(rows).to_csv("/vol/postrun/redundancy.csv", index=False)
        Path("/vol/postrun/haystack_numerals.json").write_text(
            _json.dumps(numerals), encoding="utf-8")
        _vol.commit()
        print("wrote /vol/postrun/redundancy.csv + haystack_numerals.json")
        print("now run:  modal volume get kv-audit-vol postrun ./post_run_remote")

    @_app.local_entrypoint()
    def modal_main(stage: str = "text-stats", ctx_len: int = 32768):
        if stage != "text-stats":
            raise SystemExit("the only Modal stage is --stage text-stats; "
                             "everything else runs locally")
        _remote_text_stats.remote(ctx_len)
except Exception:                                            # pragma: no cover
    pass


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v4-dir", default="./results",
                    help="downloaded v4 results/ (results.parquet or raw/)")
    ap.add_argument("--data-dir", default=None,
                    help="downloaded v4 data/ (for redundancy + D4)")
    ap.add_argument("--remote-text", default=None,
                    help="output of `modal run post_run_calc.py "
                         "--stage text-stats`")
    ap.add_argument("--v4-script", default="./modal_kv_audit_v4.py",
                    help="v4 script, imported for its exact scorers")
    ap.add_argument("--out", default="./post_run")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--only", default="all", choices=["all", "v2", "v11"],
                    help="'v2' runs ONLY the two new stages "
                         "(value origin + protocol contrast)")
    ap.add_argument("--no-dedup", action="store_true",
                    help="keep duplicate (cell, sample_id) rows")
    ap.add_argument("--expect-n", type=int, default=29400,
                    help="abort unless this many real rows survive the toy "
                         "filter and dedup (0 disables)")
    ap.add_argument("--ctx-len", type=int, default=32768,
                    help="context length of the generated data (--data-dir)")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"post_run_calc -> {out.resolve()}\n")
    print("[load]")
    df = load_raw(a.v4_dir, dedup=not a.no_dedup,
                  expect_n=(a.expect_n or None))
    sc = Scorers(a.v4_script)
    print(f"  scorers: {sc.source}")

    if a.only == "v11":
        # ---- v1.1: the CPU-only release ----------------------------------
        # A2 value-origin (adds contains_any_novel), A3 scorer split,
        # Figure 2 composition, and the numeric diff.  No GPU, no new data.
        numerals = {}
        if a.remote_text:
            p = Path(a.remote_text)
            nf = (sorted(p.rglob("haystack_numerals.json")) if p.is_dir()
                  else list(p.parent.glob("haystack_numerals.json")))
            if nf:
                numerals = {k: set(v) for k, v in json.loads(
                    nf[0].read_text(encoding="utf-8")).items()}
                print(f"  read {nf[0]} ({len(numerals)} languages)")
        print("\n[v11-a] value origin: most-grounded AND contains-any-novel")
        item_nums = load_item_numerals(a.data_dir, a.ctx_len)
        if not item_nums:
            print("    NOTE: no --data-dir; using the language-level "
                  "approximation.  The Table 2 caption must say so.")
        vo, vol_, vch = stage_value_origin(df, out, numerals, item_nums)
        tr = pd.read_csv(out / "value_origin_trials_items.csv")
        print("\n[v11-b] scorer sensitivity, split by robustness scorer")
        scor, _ = stage_scorers(df, sc, out)
        print("\n[v11-c] present-trial outcome composition (Figure 2)")
        f2 = stage_figure2(df, tr, out)
        print("\n[v11-d] numeric diff for the manuscript edit")
        stage_v11_claims(out, vo, scor, f2)
        print(f"\nDONE (v1.1 stages). Read {out / 'v11_numeric_diff.md'}.")
        return

    if a.only == "v2":
        numerals = {}
        if a.remote_text:
            p = Path(a.remote_text)
            nf = (sorted(p.rglob("haystack_numerals.json")) if p.is_dir()
                  else list(p.parent.glob("haystack_numerals.json")))
            if nf:
                numerals = {k: set(v) for k, v in json.loads(
                    nf[0].read_text(encoding="utf-8")).items()}
                print(f"  read {nf[0]} ({len(numerals)} languages)")
        print("\n[v2-a] value-origin taxonomy (exact / near copy / novel)")
        item_nums = load_item_numerals(a.data_dir, a.ctx_len)
        vo, vol_, vch = stage_value_origin(df, out, numerals, item_nums)
        print("\n[v2-b] query-agnostic vs query-aware present-trial contrast")
        prot, protl = stage_protocol(df, out, B=a.boot)
        write_v2_report(out, vo, vch, prot)
        print(f"\nDONE (v2 stages only). "
              f"Read {out / 'post_run_v2_report.md'}.")
        return

    print("\n[1] per-cell accuracies with item-bootstrap CIs")
    cells = stage_cells(df, out)
    print("\n[2] dispersion (phi) and amplification with paired CIs")
    disp, amp = stage_dispersion(df, cells, out)
    print("\n[3] ordering contrasts")
    order = stage_ordering(df, out)
    print("\n[4] haystack redundancy and its link to retention")
    red, link, numerals = stage_redundancy(df, a.data_dir, a.remote_text, out)
    print("\n[5] scorer sensitivity")
    scor, _ = stage_scorers(df, sc, out)
    print("\n[6] failure modes (+ distractor vs hallucination if data present)")
    stage_errors(df, sc, out, numerals)
    print("\n[7] depth balance, depth-conditioned accuracy, positional law")
    bal, law = stage_depth(df, out)
    print("\n[8] arm invariance")
    arms = stage_arms(df, out)
    print("\n[9] detection metric (paired tasks)")
    det = stage_detection(df, out)
    print("\n[10] fertility null")
    fert = stage_fertility(df, a.v4_dir, out)
    print("\n[11] power simulation")
    power = stage_power(out)
    print("\n[12] value-origin taxonomy (exact / near copy / novel)  [v2]")
    item_nums = load_item_numerals(a.data_dir, a.ctx_len)
    vo, vol_, vch = stage_value_origin(df, out, numerals, item_nums)
    print("\n[13] query-agnostic vs query-aware contrast  [v2]")
    prot, protl = stage_protocol(df, out, B=a.boot)

    write_report(out, cells, disp, amp, order, scor, red, link, law, arms,
                 fert, power, det, vo, vch, prot)
    write_v2_report(out, vo, vch, prot)
    print(f"\nDONE. Read {out / 'post_run_report.md'} first, then "
          f"{out / 'post_run_v2_report.md'}.")


if __name__ == "__main__":
    main()
