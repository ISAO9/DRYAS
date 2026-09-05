#!/usr/bin/env python3
# =============================================================================
# DRYAS  |  Script 007  —  REPLICATING PUBLISHED FAR-FIELD METHODS AGAINST RANDOM-YEAR NULLS
# =============================================================================
# WHY
# ---------------------------------------------------------------------------
# Several recent studies report far-field earthquake signals in ITRDB-type
# tree-ring networks: Gao et al. (2024, Nature Geoscience; global ITRDB vs USGS
# catalog, MMI >= 4, SEA with the Rao et al. 2019 double bootstrap, Lloret-type
# resilience indices), Pandey et al. (2023, JGR Biogeosciences; 46 Himalayan
# sites, "64-77% of trees responsive", declines 1-5 years after the event,
# distance dependence to ~200 km) and Owczarek et al. (2017; signals to 230 km).
# 004-006 found no signal in Japan with a detection statistic calibrated by
# injection. A referee from that community will ask: "did you apply OUR
# criteria?"  This script applies the published family of criteria to the same
# Japanese data and, crucially, applies each criterion to RANDOM years as well,
# so that every number has its null distribution next to it.
#
# WHAT IS COMPUTED (all on the growing-season effective year; all with nulls)
#   A. RESPONSIVE-TREE FRACTIONS at event years, under criteria commonly used in
#      dendro-ecology / dendrogeomorphology:
#        pct10/20/30/40 : ring width in year t (or t+1) < (1 - x) * mean(t-3..t-1)
#        schw40         : Schweingruber-type abrupt growth change, >= 40% reduction
#                         of the 4-yr mean after vs 4-yr mean before
#        schw25         : same with 25%
#        cropper        : Cropper (1979) pointer index < -0.75 (5-yr window)
#      Each fraction is compared with the same fraction at K random years per
#      site (null median and 95th percentile).
#   B. DOUBLE-BOOTSTRAP SEA (Rao et al. 2019): pooled over all (site, event)
#      pairs with prior p_response >= p_target; significance from (i) random
#      event-year sets and (ii) bootstrap of the actual composite. Also by
#      distance bin (0-100, 100-200, 200-300 km) and by trench.
#   C. LAG-TOLERANT DETECTION: for each pair, the minimum growth change over lags
#      0..5 (most negative), compared with the same statistic at random years
#      (this is generous to the "1-5 years after" claim).
#   D. LLORET (2011) RESILIENCE INDICES at events vs random years:
#        resistance = Dr / PreDr, recovery = PostDr / Dr, resilience = PostDr / PreDr
#      with Dr = growth in the event year(+1), PreDr/PostDr = 5-yr means.
#   E. RAW WIDTHS (no detrending, mean-normalized per tree) for A and C, because
#      some published analyses use raw or BAI series.
#
# The synthetic testbed (001 --mode synthetic) contains embedded suppressions:
# on it every criterion MUST separate events from nulls. On the real data the
# script reports whether any does.
#
# INPUTS  data/treering_network.parquet, data/site_metadata.parquet, data/chronologies.parquet,
#         data/002_selection.json, data/graph_nodes.parquet, data/site_event_pga.parquet,
#         data/eq_segment_assignment.parquet
# OUTPUTS data/007_responsive_fractions.parquet, data/007_sea_pooled.parquet,
#         data/007_lag_tolerant.parquet, data/007_resilience.parquet, data/007_summary.json
#         PDF/007a_responsive_fraction_vs_null.pdf, 007b_pooled_sea_double_bootstrap.pdf,
#             007c_lag_tolerant_and_resilience.pdf
#
# USAGE   uv run python src/007_replication_nulls.py   [--n-null 500] [--quick]
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=RuntimeWarning)

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
DATA_DIR = ROOT / "data"
PDF_DIR = ROOT / "PDF"
PDF_DIR.mkdir(parents=True, exist_ok=True)


def _import(name, fname):
    spec = importlib.util.spec_from_file_location(name, SRC_DIR / fname)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


@dataclass
class Config:
    p_target: float = 0.5          # prior p_response for "should respond" pairs (SEA, fractions)
    p_min_pairs: float = 0.2       # for lag-tolerant / resilience pair sets
    n_random_per_site: int = 200   # random years per site for fraction nulls
    n_null_sea: int = 1000         # random event-year sets for SEA
    n_boot_sea: int = 1000         # bootstrap of the real composite
    sea_window: tuple = (-5, 10)
    lags: tuple = (0, 1, 2, 3, 4, 5)
    dist_bins: tuple = (0, 100, 200, 300, 450)
    criteria: tuple = ("pct10", "pct20", "pct30", "pct40", "schw25", "schw40", "cropper")
    n_shift_null: int = 500        # year-shift null draws (common offset for all pairs)
    shift_min_abs: int = 8         # |offset| below this would overlap the true windows
    seed: int = 3
    quick: bool = False


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Tree-level criteria (operate on one tree series x, index i of the event year)
# ---------------------------------------------------------------------------
def _pre_mean(x, i, n=3):
    seg = x[max(0, i - n): i]
    return np.nanmean(seg) if np.isfinite(seg).sum() >= 2 else np.nan


def crit_pct(x, i, frac):
    """Reduction >= frac in year i OR i+1 relative to the mean of the 3 preceding years."""
    pre = _pre_mean(x, i)
    if not np.isfinite(pre) or pre <= 0:
        return np.nan
    post = x[i: i + 2]
    post = post[np.isfinite(post)]
    if len(post) == 0:
        return np.nan
    return float(np.min(post) <= (1 - frac) * pre)


def crit_schweingruber(x, i, frac, n=4):
    """Abrupt growth change: mean of n years from i vs mean of n years before i."""
    pre = x[max(0, i - n): i]; post = x[i: i + n]
    if np.isfinite(pre).sum() < 3 or np.isfinite(post).sum() < 3:
        return np.nan
    a, b = np.nanmean(pre), np.nanmean(post)
    if a <= 0:
        return np.nan
    return float((b - a) / a <= -frac)


def crit_cropper(x, i, thr=-0.75, w=5):
    """Cropper (1979) pointer index: (x_i - mean(window)) / sd(window), window = i-w..i-1 (+i+1..i+w)."""
    win = np.concatenate([x[max(0, i - w): i], x[i + 1: i + 1 + w]])
    win = win[np.isfinite(win)]
    if len(win) < 5 or not np.isfinite(x[i]):
        return np.nan
    sd = np.std(win)
    if sd == 0:
        return np.nan
    return float((x[i] - np.mean(win)) / sd <= thr)


CRITERIA = {
    "pct10": lambda x, i: crit_pct(x, i, 0.10), "pct20": lambda x, i: crit_pct(x, i, 0.20),
    "pct30": lambda x, i: crit_pct(x, i, 0.30), "pct40": lambda x, i: crit_pct(x, i, 0.40),
    "schw25": lambda x, i: crit_schweingruber(x, i, 0.25), "schw40": lambda x, i: crit_schweingruber(x, i, 0.40),
    "cropper": lambda x, i: crit_cropper(x, i),
}


def responsive_fraction(idx: np.ndarray, i: int, crit) -> tuple[float, int]:
    vals = [crit(idx[:, j], i) for j in range(idx.shape[1]) if np.isfinite(idx[i, j])]
    vals = [v for v in vals if np.isfinite(v)]
    return (float(np.mean(vals)), len(vals)) if vals else (np.nan, 0)


# ---------------------------------------------------------------------------
# Lloret et al. (2011) resilience indices on the site chronology
# ---------------------------------------------------------------------------
def lloret(chron: np.ndarray, i: int, w: int = 5):
    pre = chron[max(0, i - w): i]; post = chron[i + 2: i + 2 + w]; dr = chron[i: i + 2]
    if np.isfinite(pre).sum() < 3 or np.isfinite(post).sum() < 3 or np.isfinite(dr).sum() < 1:
        return np.nan, np.nan, np.nan
    pre_m, post_m, dr_m = np.nanmean(pre), np.nanmean(post), np.nanmean(dr)
    if pre_m <= 0 or dr_m <= 0:
        return np.nan, np.nan, np.nan
    return dr_m / pre_m, post_m / dr_m, post_m / pre_m


# ---------------------------------------------------------------------------
# Pooled double-bootstrap SEA (Rao et al. 2019 style)
# ---------------------------------------------------------------------------
def composite(series_by_site: dict, pairs: list, lo: int, hi: int):
    """pairs: list of (site_id, year). Returns list of normalized windows."""
    out = []
    for s, y in pairs:
        yrs, ch = series_by_site[s]
        i = int(np.searchsorted(yrs, y))
        if i + lo < 0 or i + hi >= len(yrs) or yrs[i] != y:
            continue
        base = np.nanmean(ch[i - 5: i])
        seg = ch[i + lo: i + hi + 1]
        if np.isfinite(base) and base > 0 and np.isfinite(seg).sum() >= (hi - lo - 1):
            out.append(seg / base)
    return out


def double_bootstrap_sea(series_by_site: dict, pairs: list, cfg: Config, rng) -> dict:
    lo, hi = cfg.sea_window
    wins = composite(series_by_site, pairs, lo, hi)
    if len(wins) < 3:
        return {}
    W = np.vstack(wins); comp = np.nanmean(W, axis=0); n = len(wins)
    # (i) random-event null: same number of (site, random year) draws, sites resampled with replacement
    site_list = [p[0] for p in pairs]
    nulls = []
    for _ in range(cfg.n_null_sea):
        draws = []
        for s in rng.choice(site_list, size=n, replace=True):
            yrs, ch = series_by_site[s]
            ok = yrs[(np.arange(len(yrs)) >= 5) & (np.arange(len(yrs)) < len(yrs) - hi - 1) & np.isfinite(ch)]
            if len(ok):
                draws.append((s, int(rng.choice(ok))))
        w = composite(series_by_site, draws, lo, hi)
        if len(w) >= 3:
            nulls.append(np.nanmean(np.vstack(w), axis=0))
    nulls = np.vstack(nulls)
    # (ii) bootstrap of the real composite
    boots = np.vstack([np.nanmean(W[rng.integers(0, n, n)], axis=0) for _ in range(cfg.n_boot_sea)])
    lags = np.arange(lo, hi + 1)
    i0 = -lo
    stat = float(np.nanmean(comp[i0: i0 + 2])); null_stat = np.nanmean(nulls[:, i0: i0 + 2], axis=1)
    # lag-tolerant version of the pooled test: min over lags 0..5
    stat_min = float(np.nanmin(comp[i0: i0 + 6])); null_min = np.nanmin(nulls[:, i0: i0 + 6], axis=1)
    return {"lags": lags, "composite": comp, "n": n,
            "null_q05": np.nanpercentile(nulls, 5, axis=0), "null_q95": np.nanpercentile(nulls, 95, axis=0),
            "boot_q05": np.nanpercentile(boots, 5, axis=0), "boot_q95": np.nanpercentile(boots, 95, axis=0),
            "stat_y01": stat, "p_y01": float(np.mean(null_stat <= stat)),
            "stat_min_lag0_5": stat_min, "p_min_lag0_5": float(np.mean(null_min <= stat_min))}


# ---------------------------------------------------------------------------
# Year-shift null: apply ONE common offset to all event years so that the
# spatial co-occurrence of events (six sites sharing 1952) and the regional
# climate correlation between sites are preserved. Independent-random-year
# nulls (including the Rao et al. 2019 double bootstrap) destroy that structure
# and therefore underestimate the variance of pooled statistics.
# ---------------------------------------------------------------------------
def shifted_pairs(pairs: list, delta: int, mat: dict) -> list:
    out = []
    for s, y in pairs:
        M = mat[s]; yy = y + delta
        i = int(np.searchsorted(M["yrs"], yy))
        if 5 <= i < len(M["yrs"]) - 8 and M["yrs"][i] == yy and M["usable"][i]:
            out.append((s, int(yy)))
    return out


def shift_offsets(cfg: Config, rng, max_abs: int = 250) -> np.ndarray:
    """Candidate common offsets; more candidates than needed because offsets that push
    most events outside the usable spans are discarded by shift_null_test."""
    d = np.arange(-max_abs, max_abs + 1)
    d = d[np.abs(d) >= cfg.shift_min_abs]
    return rng.choice(d, size=cfg.n_shift_null * 6, replace=True)


_FRAC_CACHE: dict = {}


def cached_fraction(M: dict, site: str, series: str, cname: str, i: int):
    key = (site, series, cname, i)
    if key not in _FRAC_CACHE:
        _FRAC_CACHE[key] = responsive_fraction(M[series], i, CRITERIA[cname])
    return _FRAC_CACHE[key]


def stat_fraction_mean(pairs: list, mat: dict, series: str, cname: str) -> float:
    vals = []
    for s, y in pairs:
        M = mat[s]
        f, n = cached_fraction(M, s, series, cname, int(np.searchsorted(M["yrs"], y)))
        if np.isfinite(f) and n >= 5:
            vals.append(f)
    return float(np.mean(vals)) if len(vals) >= 3 else np.nan


def stat_sea_y01(pairs: list, series_by_site: dict, lo: int, hi: int) -> float:
    w = composite(series_by_site, pairs, lo, hi)
    if len(w) < 3:
        return np.nan
    c = np.nanmean(np.vstack(w), axis=0); i0 = -lo
    return float(np.nanmean(c[i0: i0 + 2]))


def stat_resistance_median(pairs: list, mat: dict) -> float:
    vals = [lloret(mat[s]["chron"], int(np.searchsorted(mat[s]["yrs"], y)))[0] for s, y in pairs]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.median(vals)) if len(vals) >= 3 else np.nan


def stat_min_gc_median(pairs: list, mat: dict, m4, cfg4) -> float:
    vals = []
    for s, y in pairs:
        M = mat[s]; gc = M.setdefault("_gc0", m4.growth_change(M["chron"], cfg4.pre_window, (0, 0)))
        i = int(np.searchsorted(M["yrs"], y)); seg = gc[i: i + 6]; seg = seg[np.isfinite(seg)]
        if len(seg) >= 3:
            vals.append(float(np.min(seg)))
    return float(np.median(vals)) if len(vals) >= 3 else np.nan


def stat_weighted_z(pairs: list, mat: dict, weights: dict, m4, cfg4) -> float:
    """PRIMARY (pre-specified) statistic: prior-weighted stack of per-pair growth-change z-scores,
    z_w = sum(w_i z_i) / sqrt(sum w_i^2), w_i = prior p_response of the (site, event) pair."""
    num, den = 0.0, 0.0
    for s, y in pairs:
        M = mat[s]
        gc = M.setdefault("_gc01", m4.growth_change(M["chron"], cfg4.pre_window, cfg4.post_window))
        if "_gc_mu" not in M:
            g = gc[M["usable"] & np.isfinite(gc)]; M["_gc_mu"], M["_gc_sd"] = float(np.mean(g)), float(np.std(g))
        i = int(np.searchsorted(M["yrs"], y))
        if i < len(gc) and np.isfinite(gc[i]) and M["_gc_sd"] > 0:
            z = -(gc[i] - M["_gc_mu"]) / M["_gc_sd"]
            w = weights.get((s, y), weights.get(s, 1.0))
            num += w * z; den += w * w
    return float(num / np.sqrt(den)) if den > 0 else np.nan


def shift_null_test(pairs: list, mat: dict, stat_fn, cfg: Config, rng, lower_is_signal: bool = True) -> dict:
    obs = stat_fn(pairs)
    nulls = []
    for d in shift_offsets(cfg, rng):
        sp = shifted_pairs(pairs, int(d), mat)
        if len(sp) >= max(3, int(0.6 * len(pairs))):
            v = stat_fn(sp)
            if np.isfinite(v):
                nulls.append(v)
        if len(nulls) >= cfg.n_shift_null:
            break
    nulls = np.array(nulls)
    if not np.isfinite(obs) or len(nulls) < 30:
        return {"obs": obs, "n_null": int(len(nulls)), "p": np.nan, "null_median": np.nan, "null_q05": np.nan, "null_q95": np.nan}
    p = float(np.mean(nulls <= obs)) if lower_is_signal else float(np.mean(nulls >= obs))
    return {"obs": float(obs), "n_null": int(len(nulls)), "p": p, "null_median": float(np.median(nulls)),
            "null_q05": float(np.percentile(nulls, 5)), "null_q95": float(np.percentile(nulls, 95))}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_fractions(frac_tab: pd.DataFrame, cfg: Config, out: Path):
    crits = list(cfg.criteria)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor="white")
    for ax, proxy in zip(axes, ("index", "raw")):
        d = frac_tab[frac_tab["series"] == proxy]; ax.set_facecolor("white")
        x = np.arange(len(crits))
        ev = [d.loc[d["criterion"] == c, "event_mean"].mean() for c in crits]
        nm = [d.loc[d["criterion"] == c, "nullmean_median"].mean() for c in crits]
        n95 = [d.loc[d["criterion"] == c, "nullmean_q95"].mean() for c in crits]
        n05 = [d.loc[d["criterion"] == c, "nullmean_q05"].mean() for c in crits]
        ax.bar(x - 0.18, nm, width=0.36, color="#bdbdbd", label="random years: mean of n_ev draws (median, 5-95%)")
        ax.errorbar(x - 0.18, nm, yerr=[np.array(nm) - np.array(n05), np.array(n95) - np.array(nm)], fmt="none", ecolor="#666", capsize=3)
        ax.bar(x + 0.18, ev, width=0.36, color="#d62728", label="event years (prior p>=%.1f)" % cfg.p_target)
        ax.set_xticks(x); ax.set_xticklabels(crits, fontsize=8)
        ax.set_ylabel("Fraction of trees classified as responsive")
        ax.set_title("Detrended indices" if proxy == "index" else "Raw ring widths (mean-normalized)")
        ax.set_ylim(0, 1); ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    fig.suptitle("DRYAS - 'Responsive tree' fractions at earthquake years vs the same criteria at random years", y=1.02)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_sea(results: dict, cfg: Config, out: Path):
    keys = [k for k, v in results.items() if v]
    if not keys:
        return
    ncol = 3; nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.0 * nrow), facecolor="white", squeeze=False)
    for ax, k in zip(axes.ravel(), keys):
        r = results[k]; ax.set_facecolor("white")
        ax.fill_between(r["lags"], r["null_q05"], r["null_q95"], color="#cccccc", alpha=0.6, linewidth=0)
        ax.fill_between(r["lags"], r["boot_q05"], r["boot_q95"], color="#f4a582", alpha=0.45, linewidth=0)
        ax.plot(r["lags"], r["composite"], color="#b2182b", linewidth=1.6)
        ax.axhline(1, color="#777", linewidth=0.6); ax.axvline(0, color="#777", linewidth=0.6, linestyle=":")
        ax.set_title(f"{k}  (n={r['n']}, p0-1={r['p_y01']:.2f}, p_min0-5={r['p_min_lag0_5']:.2f})", fontsize=8.5)
        ax.tick_params(labelsize=7); ax.grid(True, linestyle=":", linewidth=0.4)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Years relative to event", fontsize=8)
    for row in axes:
        row[0].set_ylabel("Index / pre-event mean", fontsize=8)
    handles = [Line2D([0], [0], color="#b2182b", label="Pooled composite"),
               Line2D([0], [0], color="#f4a582", linewidth=8, label="Bootstrap 5-95% of the composite"),
               Line2D([0], [0], color="#cccccc", linewidth=8, label=f"Random-event null 5-95% ({cfg.n_null_sea} sets)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 0.98), frameon=False, fontsize=8)
    fig.suptitle("DRYAS - Pooled double-bootstrap SEA (Rao et al. 2019 design) on Japanese ITRDB sites", y=1.01)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_lag_resilience(lag: pd.DataFrame, res: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), facecolor="white")
    ax = axes[0]; ax.set_facecolor("white")
    for series, col in (("index", "#d62728"), ("raw", "#1f77b4")):
        d = lag[lag["series"] == series]
        ax.hist(d[d["kind"] == "null"]["min_gc"].dropna(), bins=40, density=True, histtype="step", color="#777", linewidth=1 if series == "index" else 0.6)
        ax.hist(d[d["kind"] == "event"]["min_gc"].dropna(), bins=40, density=True, histtype="step", color=col, linewidth=1.6, label=f"events ({series})")
    ax.set_xlabel("Most negative growth change over lags 0-5"); ax.set_ylabel("Density"); ax.set_title("Lag-tolerant statistic")
    ax.legend(frameon=False, fontsize=8); ax.grid(True, linestyle=":", linewidth=0.5)
    for ax, k in zip(axes[1:], ("resistance", "recovery", "resilience")):
        ax.set_facecolor("white")
        ev = res[res["kind"] == "event"][k].dropna(); nl = res[res["kind"] == "null"][k].dropna()
        ax.boxplot([nl, ev], widths=0.5, showfliers=False)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["random years", "events"])
        ax.set_title(f"Lloret {k} (n_ev={len(ev)})"); ax.grid(True, axis="y", linestyle=":", linewidth=0.5)
        ax.axhline(1, color="#999", linewidth=0.6)
    fig.suptitle("DRYAS - Lag-tolerant detection and resilience indices: events vs random years", y=1.03)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def save_table(df, name, records):
    p = DATA_DIR / name
    df.to_parquet(p, index=False)
    records[name] = {"rows": int(len(df)), "sha256": _sha(p)}
    print(f"  wrote {p}  ({len(df):,} rows)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="DRYAS 007 — replication of published criteria with nulls")
    ap.add_argument("--n-null", type=int, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    if args.quick:
        cfg.quick = True; cfg.n_random_per_site = 40; cfg.n_null_sea = 200; cfg.n_boot_sea = 200; cfg.n_shift_null = 60
    if args.n_null:
        cfg.n_null_sea = args.n_null; cfg.n_boot_sea = args.n_null
    rng = np.random.default_rng(cfg.seed)
    m2 = _import("dryas002", "002_chronology_standardization.py")
    m4 = _import("dryas004", "004_response_features.py")
    cfg2, cfg4 = m2.Config(), m4.Config()

    rings = pd.read_parquet(DATA_DIR / "treering_network.parquet")
    sites = pd.read_parquet(DATA_DIR / "site_metadata.parquet")
    sel = json.load(open(DATA_DIR / "002_selection.json"))
    nodes = pd.read_parquet(DATA_DIR / "graph_nodes.parquet")
    se = pd.read_parquet(DATA_DIR / "site_event_pga.parquet")
    prov = json.load(open(DATA_DIR / "provenance.json"))
    method = sel["ring_width"]["selected"]
    ycol = "eff_year" if "eff_year" in se.columns else "year"
    names = dict(zip(nodes["site_id"], nodes["site_name"]))
    print(f"DRYAS 007 — method={method}, criteria={cfg.criteria}, random years/site={cfg.n_random_per_site}")

    rings, sites_m, _ = m2.merge_clusters(rings, sites, cfg2, "ring_width_mm")
    site_ids = [s for s in nodes["site_id"] if s in set(rings["site_id"])]

    # per-site material: detrended indices + raw normalized + chronologies
    mat = {}
    series_idx, series_raw = {}, {}
    for s in site_ids:
        yrs, W, tree_ids = m2.to_wide(rings[rings["site_id"] == s], "ring_width_mm")
        res = m2.build_site(W, method, cfg2, rcs_group=None if method not in ("rcs", "sfrcs") else W)
        raw = W / np.nanmean(W, axis=0, keepdims=True)
        n = nodes[nodes["site_id"] == s].iloc[0]
        usable = (yrs >= n["usable_depth_first"]) & (yrs <= n["usable_depth_last"])
        mat[s] = {"yrs": yrs, "index": res["index"], "raw": raw, "chron": res["chron"],
                  "chron_raw": m2.chronology_from_indices(raw), "usable": usable}
        series_idx[s] = (yrs, res["chron"]); series_raw[s] = (yrs, m2.chronology_from_indices(raw))

    # event pairs
    tgt = se[(se["p_response"] >= cfg.p_target) & se["in_usable_span"] & se["site_id"].isin(site_ids)]
    tgt_pairs = sorted(set(zip(tgt["site_id"], tgt[ycol].astype(int))))
    wide = se[(se["p_response"] >= cfg.p_min_pairs) & se["in_usable_span"] & se["site_id"].isin(site_ids)]
    wide_pairs = sorted(set(zip(wide["site_id"], wide[ycol].astype(int))))
    print(f"  pairs: {len(tgt_pairs)} with p>={cfg.p_target}, {len(wide_pairs)} with p>={cfg.p_min_pairs}")

    # ---- A. responsive fractions with nulls ----------------------------
    frac_rows = []
    for s in site_ids:
        M = mat[s]; yrs = M["yrs"]
        ev_years = [y for (ss, y) in tgt_pairs if ss == s]
        cand = yrs[M["usable"] & (np.arange(len(yrs)) >= 5) & (np.arange(len(yrs)) < len(yrs) - 6)]
        if len(cand) == 0:
            continue
        rand_years = rng.choice(cand, size=min(cfg.n_random_per_site, len(cand)), replace=False)
        for series in ("index", "raw"):
            X = M[series]
            for c in cfg.criteria:
                crit = CRITERIA[c]
                ev = [responsive_fraction(X, int(np.searchsorted(yrs, y)), crit)[0] for y in ev_years if y in set(yrs)]
                ev = [v for v in ev if np.isfinite(v)]
                nl = [responsive_fraction(X, int(np.searchsorted(yrs, y)), crit)[0] for y in rand_years]
                nl = np.array([v for v in nl if np.isfinite(v)])
                if len(nl) < 10 or not ev:
                    continue
                # null of the MEAN over n_ev random years (fair comparison with event_mean)
                k = len(ev)
                nl_mean = np.array([np.mean(rng.choice(nl, size=k, replace=True)) for _ in range(500)])
                frac_rows.append({"site_id": s, "series": series, "criterion": c, "n_events": k,
                                  "event_mean": float(np.mean(ev)),
                                  "null_median": float(np.median(nl)), "null_q05": float(np.percentile(nl, 5)),
                                  "null_q95": float(np.percentile(nl, 95)),
                                  "nullmean_median": float(np.median(nl_mean)),
                                  "nullmean_q05": float(np.percentile(nl_mean, 5)),
                                  "nullmean_q95": float(np.percentile(nl_mean, 95)),
                                  "p_site_one_sided": float(np.mean(nl_mean >= np.mean(ev)))})
    frac_tab = pd.DataFrame(frac_rows)

    # ---- B. pooled double-bootstrap SEA ---------------------------------
    sea_results = {}
    sea_results["all sites, p>=%.1f" % cfg.p_target] = double_bootstrap_sea(series_idx, tgt_pairs, cfg, rng)
    sea_results["raw widths, p>=%.1f" % cfg.p_target] = double_bootstrap_sea(series_raw, tgt_pairs, cfg, rng)
    # distance bins (closest distance of the pair)
    tgt_d = tgt.copy()
    for a, b in zip(cfg.dist_bins[:-1], cfg.dist_bins[1:]):
        sub = tgt_d[(tgt_d["dist_km"] >= a) & (tgt_d["dist_km"] < b)]
        prs = sorted(set(zip(sub["site_id"], sub[ycol].astype(int))))
        if len(prs) >= 3:
            sea_results[f"{a:.0f}-{b:.0f} km"] = double_bootstrap_sea(series_idx, prs, cfg, rng)
    # by trench (from segment prefix)
    tgt_d["trench"] = tgt_d["segment"].str.split("-").str[0]
    for tr, sub in tgt_d.groupby("trench"):
        prs = sorted(set(zip(sub["site_id"], sub[ycol].astype(int))))
        if len(prs) >= 4:
            sea_results[f"trench {tr}"] = double_bootstrap_sea(series_idx, prs, cfg, rng)
    sea_rows = []
    for k, r in sea_results.items():
        if r:
            sea_rows.append({"subset": k, "n": r["n"], "stat_y01": r["stat_y01"], "p_y01": r["p_y01"],
                             "stat_min_lag0_5": r["stat_min_lag0_5"], "p_min_lag0_5": r["p_min_lag0_5"]})
    sea_tab = pd.DataFrame(sea_rows)

    # ---- C. lag-tolerant per-pair statistic ------------------------------
    lag_rows = []
    for series in ("index", "raw"):
        key = "chron" if series == "index" else "chron_raw"
        for s in site_ids:
            M = mat[s]; yrs = M["yrs"]; ch = M[key]
            gc = m4.growth_change(ch, cfg4.pre_window, (0, 0))
            def min_gc(i):
                seg = gc[i: i + 6]; seg = seg[np.isfinite(seg)]
                return float(np.min(seg)) if len(seg) >= 3 else np.nan
            for (ss, y) in wide_pairs:
                if ss == s and y in set(yrs):
                    lag_rows.append({"series": series, "kind": "event", "site_id": s, "year": y,
                                     "min_gc": min_gc(int(np.searchsorted(yrs, y)))})
            cand = yrs[M["usable"] & (np.arange(len(yrs)) >= 5) & (np.arange(len(yrs)) < len(yrs) - 7)]
            for y in rng.choice(cand, size=min(cfg.n_random_per_site, len(cand)), replace=False):
                lag_rows.append({"series": series, "kind": "null", "site_id": s, "year": int(y),
                                 "min_gc": min_gc(int(np.searchsorted(yrs, y)))})
    lag = pd.DataFrame(lag_rows)

    # ---- D. Lloret resilience indices -----------------------------------
    res_rows = []
    for s in site_ids:
        M = mat[s]; yrs = M["yrs"]; ch = M["chron"]
        for (ss, y) in wide_pairs:
            if ss == s and y in set(yrs):
                r, c, z = lloret(ch, int(np.searchsorted(yrs, y)))
                res_rows.append({"kind": "event", "site_id": s, "year": y, "resistance": r, "recovery": c, "resilience": z})
        cand = yrs[M["usable"] & (np.arange(len(yrs)) >= 6) & (np.arange(len(yrs)) < len(yrs) - 8)]
        for y in rng.choice(cand, size=min(cfg.n_random_per_site, len(cand)), replace=False):
            r, c, z = lloret(ch, int(np.searchsorted(yrs, y)))
            res_rows.append({"kind": "null", "site_id": s, "year": int(y), "resistance": r, "recovery": c, "resilience": z})
    res = pd.DataFrame(res_rows)

    # ---- F. year-shift nulls (spatially coherent) for the pooled statistics ----
    lo, hi = cfg.sea_window
    shift_rows = []
    # ---- PRIMARY pre-specified test: prior-weighted z stack over all pairs p>=0.2 ----
    w_pair = {(r["site_id"], int(r[ycol])): float(r["p_response"]) for _, r in wide.iterrows()}
    w_site = wide.groupby("site_id")["p_response"].mean().to_dict()
    weights = {**w_site, **w_pair}
    primary = shift_null_test(wide_pairs, mat, lambda pr: stat_weighted_z(pr, mat, weights, m4, cfg4), cfg, rng,
                              lower_is_signal=False)   # larger z_w = more suppression
    primary["test"] = "PRIMARY: prior-weighted z stack (p>=0.2)"; primary["n_pairs"] = len(wide_pairs)
    slopes_p = DATA_DIR / "005_sensitivity_slopes.parquet"
    if slopes_p.exists():
        sl = pd.read_parquet(slopes_p).set_index("site_id")["slope"].to_dict()
        num = sum(weights.get(k, 1.0) * sl.get(k[0], np.nan) for k in wide_pairs if k[0] in sl)
        den = np.sqrt(sum(weights.get(k, 1.0) ** 2 for k in wide_pairs if k[0] in sl))
        slope_w = num / den if den > 0 else np.nan
        primary["A_UB_pooled"] = max(0.0, (primary["obs"] + 1.645) / slope_w) if np.isfinite(slope_w) else np.nan
    print(f"  [PRIMARY] z_w={primary['obs']:.2f}  shift-null median={primary['null_median']:.2f} "
          f"[{primary['null_q05']:.2f},{primary['null_q95']:.2f}]  p={primary['p']:.3f}  "
          f"pooled A_UB={primary.get('A_UB_pooled', float('nan')):.3f}")

    tests = {
        "fraction pct20 (index)": (tgt_pairs, lambda pr: stat_fraction_mean(pr, mat, "index", "pct20"), False),
        "fraction pct40 (index)": (tgt_pairs, lambda pr: stat_fraction_mean(pr, mat, "index", "pct40"), False),
        "fraction schw25 (index)": (tgt_pairs, lambda pr: stat_fraction_mean(pr, mat, "index", "schw25"), False),
        "SEA years 0-1 (index, p>=0.5)": (tgt_pairs, lambda pr: stat_sea_y01(pr, series_idx, lo, hi), True),
        "SEA years 0-1 (raw, p>=0.5)": (tgt_pairs, lambda pr: stat_sea_y01(pr, series_raw, lo, hi), True),
        "Lloret resistance median (p>=0.2)": (wide_pairs, lambda pr: stat_resistance_median(pr, mat), True),
        "lag-tolerant min gc median (p>=0.2)": (wide_pairs, lambda pr: stat_min_gc_median(pr, mat, m4, cfg4), True),
    }
    for name, (prs, fn, lower) in tests.items():
        r = shift_null_test(prs, mat, fn, cfg, rng, lower_is_signal=lower)
        r["test"] = name; r["n_pairs"] = len(prs); shift_rows.append(r)
        print(f"  [shift-null] {name}: obs={r['obs']:.3f} null={r['null_median']:.3f} "
              f"[{r['null_q05']:.3f},{r['null_q95']:.3f}] p={r['p']:.3f} (n_null={r['n_null']})")
    shift_tab = pd.DataFrame([primary] + shift_rows)

    # ---- summary statistics --------------------------------------------
    def mw_p(a, b):
        """One-sided Mann-Whitney: is 'a' (events) stochastically smaller than 'b' (null)?"""
        from scipy.stats import mannwhitneyu
        a = np.asarray(a)[np.isfinite(a)]; b = np.asarray(b)[np.isfinite(b)]
        if len(a) < 3 or len(b) < 3:
            return np.nan
        return float(mannwhitneyu(a, b, alternative="less").pvalue)
    n_tests_total = 14 + len(sea_tab) * 2 + 2 + 3 + len(shift_tab)
    summary = {"method": method, "n_pairs_p_target": len(tgt_pairs), "n_pairs_p_min": len(wide_pairs),
               "responsive_fraction": {}, "sea": sea_tab.to_dict(orient="records"),
               "lag_tolerant": {}, "resilience": {},
               "shift_null": shift_tab.to_dict(orient="records"),
               "n_tests_total": int(n_tests_total),
               "bonferroni_alpha_0.05": float(0.05 / n_tests_total)}
    for series in ("index", "raw"):
        for c in cfg.criteria:
            d = frac_tab[(frac_tab["series"] == series) & (frac_tab["criterion"] == c)]
            summary["responsive_fraction"][f"{series}:{c}"] = {
                "event_mean": float(d["event_mean"].mean()), "null_median": float(d["null_median"].mean()),
                "null_q95": float(d["null_q95"].mean()),
                "nullmean_median": float(d["nullmean_median"].mean()), "nullmean_q95": float(d["nullmean_q95"].mean()),
                "sites_event_above_nullmean_q95": int((d["event_mean"] > d["nullmean_q95"]).sum()), "n_sites": int(len(d)),
                "median_site_p": float(d["p_site_one_sided"].median())}
        d = lag[lag["series"] == series]
        summary["lag_tolerant"][series] = {
            "event_median": float(d[d["kind"] == "event"]["min_gc"].median()),
            "null_median": float(d[d["kind"] == "null"]["min_gc"].median()),
            "p_events_lower": mw_p(d[d["kind"] == "event"]["min_gc"], d[d["kind"] == "null"]["min_gc"])}
    for k in ("resistance", "recovery", "resilience"):
        summary["resilience"][k] = {"event_median": float(res[res["kind"] == "event"][k].median()),
                                    "null_median": float(res[res["kind"] == "null"][k].median()),
                                    "p_events_lower": mw_p(res[res["kind"] == "event"][k], res[res["kind"] == "null"][k])}

    records = {}
    save_table(frac_tab, "007_responsive_fractions.parquet", records)
    save_table(sea_tab, "007_sea_pooled.parquet", records)
    save_table(lag, "007_lag_tolerant.parquet", records)
    save_table(res, "007_resilience.parquet", records)
    save_table(shift_tab, "007_shift_null.parquet", records)
    json.dump(summary, open(DATA_DIR / "007_summary.json", "w"), indent=2, default=float)
    prov["007"] = {"script": "007_replication_nulls.py", "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "config": asdict(cfg), "artifacts": records, "summary": summary}
    json.dump(prov, open(DATA_DIR / "provenance.json", "w"), indent=2, default=str)

    print("rendering figures ...")
    if len(frac_tab):
        fig_fractions(frac_tab, cfg, PDF_DIR / "007a_responsive_fraction_vs_null.pdf")
    fig_sea(sea_results, cfg, PDF_DIR / "007b_pooled_sea_double_bootstrap.pdf")
    if len(lag) and len(res):
        fig_lag_resilience(lag, res, PDF_DIR / "007c_lag_tolerant_and_resilience.pdf")

    print("\n=== SUMMARY ===")
    print("  responsive-tree fraction (mean over sites): event | null(mean of n_ev) median | 95th | sites above 95th | median site p")
    for k, v in summary["responsive_fraction"].items():
        print(f"    {k:>14}: {v['event_mean']:.2f} | {v['nullmean_median']:.2f} | {v['nullmean_q95']:.2f} | "
              f"{v['sites_event_above_nullmean_q95']}/{v['n_sites']} | {v['median_site_p']:.2f}")
    print("  pooled double-bootstrap SEA:")
    for r in summary["sea"]:
        print(f"    {r['subset']:>22}: n={r['n']:>3}  comp(0-1)={r['stat_y01']:.3f} p={r['p_y01']:.3f}  "
              f"min(lag0-5)={r['stat_min_lag0_5']:.3f} p={r['p_min_lag0_5']:.3f}")
    print("  lag-tolerant min growth change (events vs random years), one-sided Mann-Whitney:")
    for k, v in summary["lag_tolerant"].items():
        print(f"    {k:>6}: event median={v['event_median']:.3f}  null median={v['null_median']:.3f}  p={v['p_events_lower']:.3f}")
    print("  Lloret indices (events vs random years), one-sided Mann-Whitney (events lower):")
    for k, v in summary["resilience"].items():
        print(f"    {k:>10}: event median={v['event_median']:.3f}  null median={v['null_median']:.3f}  p={v['p_events_lower']:.3f}")
    print("  YEAR-SHIFT nulls (spatial co-occurrence preserved). First row = PRIMARY pre-specified test; the rest are exploratory:")
    for r in summary["shift_null"]:
        print(f"    {r['test']:>36}: obs={r['obs']:.3f} null={r['null_median']:.3f} [{r['null_q05']:.3f}, {r['null_q95']:.3f}]  p={r['p']:.3f}")
    print(f"  total tests reported: {summary['n_tests_total']}  (Bonferroni alpha = {summary['bonferroni_alpha_0.05']:.4f})")
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
