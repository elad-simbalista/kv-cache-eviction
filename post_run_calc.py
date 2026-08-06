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
def load_raw(v4_dir):
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
    if "toy" in df.columns:
        df = df[~df["toy"].astype(bool)]
    df["correct"] = df["correct"].astype(int)
    for c in ("kept", "depth"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    print(f"  loaded {len(df):,} rows from {src}  "
          f"({df.groupby(['model', 'arm', 'task']).ngroups} facets, "
          f"{df.config_hash.nunique()} cells)")
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
        rows.append(dict(zip(CELL_KEYS, keys), n=len(g),
                         acc_official=round(g.correct.mean(), 4),
                         acc_recomputed=round(g["official"].mean(), 4),
                         acc_robust=round(g["robust"].mean(), 4),
                         delta_robust=round(g.correct.mean()
                                            - g["robust"].mean(), 4)))
    s = pd.DataFrame(rows).sort_values("delta_robust",
                                       key=lambda c: -c.abs())
    s.to_csv(out / "scorers.csv", index=False)
    big = s[s.delta_robust.abs() >= 0.05]
    print(f"  scorers.csv: {len(big)} cells with |delta| >= 0.05 "
          f"(largest {s.delta_robust.abs().max():.3f})")
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
def write_report(out, cells, disp, amp, order, scor, red, link, law, arms,
                 fert, power, det):
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
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"post_run_calc -> {out.resolve()}\n")
    print("[load]")
    df = load_raw(a.v4_dir)
    sc = Scorers(a.v4_script)
    print(f"  scorers: {sc.source}")

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

    write_report(out, cells, disp, amp, order, scor, red, link, law, arms,
                 fert, power, det)
    print(f"\nDONE. Read {out / 'post_run_report.md'} first.")


if __name__ == "__main__":
    main()
