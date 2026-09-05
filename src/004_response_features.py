#!/usr/bin/env python3
# =============================================================================
# DRYAS  |  Script 004  —  EARTHQUAKE RESPONSE FEATURES, SEA & EMPIRICAL CALIBRATION
# =============================================================================
# WHAT THIS SCRIPT DOES
# ---------------------------------------------------------------------------
# 003 produced a PHYSICAL prior for which events each tree site should feel
# (PGA from Si & Midorikawa 1999 -> p_response). This script confronts that
# prior with the trees themselves and turns the outcome into (i) calibrated
# response parameters and (ii) the per-site-year feature table that the source
# separation model (005) will learn from.
#
#   1. PER-TREE INDICES are rebuilt with the method selected in 002 (functions
#      imported from 002, so the standardization is identical), including the
#      merged Yakushima site. From them we compute, for every site-year:
#        gc         growth change of the site chronology:
#                   mean(index[t..t+1]) / mean(index[t-5..t-1]) - 1
#        frac_supp  fraction of living trees whose own growth change is below
#                   their personal 10th percentile (tree-level "responders")
#        depth      living trees
#      and a cluster-level coherence feature (mean gc and mean frac_supp of the
#      other sites of the same regional cluster, which is what distinguishes a
#      fault-scale forcing from a stand-scale one).
#
#   2. SUPERPOSED EPOCH ANALYSIS (SEA). For each site, the chronology is
#      stacked from -5 to +10 years around the events that the prior says it
#      should feel (p_response >= p_target, inside the detection-usable span),
#      and compared with 2,000 random-year composites of the same size. This
#      gives a per-site significance of the composite response and the
#      composite recovery shape (the shape assumed in 001/002 is tested here).
#
#   3. PER-EVENT RESPONSE for every (site, event) pair with M >= mmin inside the
#      usable span: r = -gc at the event year (positive = suppression), its
#      z-score against the site's own gc distribution, and the tree-level
#      responding fraction f. A pair is "detected" if z >= z_detect.
#
#   4. EMPIRICAL CALIBRATION of the prior: logistic regression of detected vs
#      log10(PGA_prior) fitted by maximum likelihood -> (x0, k) replacing the
#      003 placeholders; AUC of the prior; the median f among detected pairs
#      replaces the injection assumption f = 0.6 of 002 if it differs.
#      Everything is written to data/004_response_calibration.json.
#
#   5. MULTI-PROXY ALIGNMENT: maximum-density (MXD) chronologies are aligned
#      on the same site-year grid; their correlation with ring width and their
#      own SEA are reported (MXD tracks summer temperature and is the climate
#      covariate 005 will use to explain away non-seismic suppression).
#
# INPUTS  data/treering_network.parquet, data/treering_density.parquet (opt),
#         data/site_metadata.parquet, data/chronologies.parquet,
#         data/002_selection.json, data/graph_nodes.parquet,
#         data/site_event_pga.parquet, data/eq_segment_assignment.parquet,
#         data/provenance.json
# OUTPUTS data/features.parquet (site_id, year, rw_index, mxd_index, gc, frac_supp,
#           depth, cluster, cluster_gc, cluster_frac, usable_depth, max_pga_prior)
#         data/004_sea.parquet, data/004_event_response.parquet,
#         data/004_response_calibration.json
#         PDF/004a_sea_composites.pdf, 004b_prior_calibration.pdf,
#             004c_responding_fraction.pdf, 004d_rw_vs_mxd.pdf
#
# USAGE  uv run python src/004_response_features.py
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
from scipy.optimize import minimize

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


def _import_002():
    spec = importlib.util.spec_from_file_location("dryas002", SRC_DIR / "002_chronology_standardization.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["dryas002"] = m
    spec.loader.exec_module(m)
    return m


@dataclass
class Config:
    pre_window: tuple = (-5, -1)      # baseline years relative to event
    post_window: tuple = (0, 1)       # response years
    sea_window: tuple = (-5, 10)
    n_null: int = 2000
    p_target: float = 0.5             # prior probability to count as "should respond" in SEA
    mmin_pairs: float = 6.5           # (site, event) pairs used for calibration
    z_detect: float = 1.645           # one-sided 95%
    tree_pct: float = 10.0            # personal percentile for tree-level responders
    seed: int = 7


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Growth-change features
# ---------------------------------------------------------------------------
def growth_change(x: np.ndarray, pre=(-5, -1), post=(0, 1)) -> np.ndarray:
    """gc[t] = mean(x[t+post]) / mean(x[t+pre]) - 1  (NaN-aware, needs >=3 pre and all post)."""
    n = len(x)
    out = np.full(n, np.nan)
    for t in range(n):
        a, b = t + pre[0], t + pre[1] + 1
        c, d = t + post[0], t + post[1] + 1
        if a < 0 or d > n:
            continue
        pre_v, post_v = x[a:b], x[c:d]
        if np.isfinite(pre_v).sum() >= 3 and np.isfinite(post_v).all() and np.nanmean(pre_v) > 0:
            out[t] = np.nanmean(post_v) / np.nanmean(pre_v) - 1.0
    return out


def site_features(idx: np.ndarray, chron: np.ndarray, years: np.ndarray, cfg: Config) -> pd.DataFrame:
    gc = growth_change(chron, cfg.pre_window, cfg.post_window)
    depth = np.isfinite(idx).sum(axis=1)
    # tree-level responders: personal gc below personal percentile
    n_years, n_trees = idx.shape
    resp = np.zeros(n_years); alive = np.zeros(n_years)
    for j in range(n_trees):
        g = growth_change(idx[:, j], cfg.pre_window, cfg.post_window)
        ok = np.isfinite(g)
        if ok.sum() < 30:
            continue
        thr = np.nanpercentile(g[ok], cfg.tree_pct)
        resp[ok & (g < thr)] += 1
        alive[ok] += 1
    frac = np.where(alive > 0, resp / np.maximum(alive, 1), np.nan)
    return pd.DataFrame({"year": years, "rw_index": chron, "gc": gc, "frac_supp": frac, "depth": depth,
                         "n_gc_trees": alive})


# ---------------------------------------------------------------------------
# SEA
# ---------------------------------------------------------------------------
def sea(chron: np.ndarray, years: np.ndarray, event_years: list, cfg: Config, rng) -> dict:
    lo, hi = cfg.sea_window
    L = hi - lo + 1
    def composite(evs):
        stack = []
        for y in evs:
            i = int(np.searchsorted(years, y))
            if i + lo < 0 or i + hi >= len(years) or years[i] != y:
                continue
            seg = chron[i + lo: i + hi + 1]
            base = np.nanmean(chron[i + cfg.pre_window[0]: i + cfg.pre_window[1] + 1])
            if np.isfinite(base) and base > 0 and np.isfinite(seg).sum() >= L - 2:
                stack.append(seg / base)
        return (np.nanmean(np.vstack(stack), axis=0), len(stack)) if stack else (np.full(L, np.nan), 0)
    comp, n_ev = composite(event_years)
    if n_ev == 0:
        return {}
    valid = years[(np.isfinite(chron))]
    valid = valid[(valid >= years.min() - lo + 5) & (valid <= years.max() - hi - 1)]
    nulls = []
    for _ in range(cfg.n_null):
        rnd = rng.choice(valid, size=n_ev, replace=False)
        c, _ = composite(rnd)
        nulls.append(c)
    nulls = np.vstack(nulls)
    q05, q95 = np.nanpercentile(nulls, 5, axis=0), np.nanpercentile(nulls, 95, axis=0)
    # one-sided significance of the year-0/+1 mean
    i0 = -lo
    stat = np.nanmean(comp[i0: i0 + 2]); null_stat = np.nanmean(nulls[:, i0: i0 + 2], axis=1)
    p_val = float(np.mean(null_stat <= stat))
    return {"lags": np.arange(lo, hi + 1), "composite": comp, "q05": q05, "q95": q95,
            "n_events": n_ev, "p_value": p_val, "stat": float(stat), "null_mean": float(np.nanmean(null_stat))}


# ---------------------------------------------------------------------------
# Logistic calibration
# ---------------------------------------------------------------------------
def fit_logistic(x: np.ndarray, y: np.ndarray):
    """p = 1/(1+exp(-k (x - x0))) by max likelihood (bounded: x0 in [0.5, 4.5], k in [0.3, 20]).
    Returns (x0, k, converged_inside_bounds)."""
    def nll(p):
        x0, k = p
        z = np.clip(k * (x - x0), -50, 50)
        ll = y * (-np.logaddexp(0, -z)) + (1 - y) * (-np.logaddexp(0, z))
        return -np.sum(ll) + 0.01 * k ** 2
    best = None
    for x0 in (1.5, 2.0, 2.5, 3.0):
        for k in (1.0, 3.0, 6.0):
            r = minimize(nll, [x0, k], method="L-BFGS-B", bounds=[(0.5, 4.5), (0.3, 20.0)])
            if best is None or r.fun < best.fun:
                best = r
    x0, k = float(best.x[0]), float(best.x[1])
    inside = 0.5 + 1e-3 < x0 < 4.5 - 1e-3 and 0.3 + 1e-3 < k < 20.0 - 1e-3
    return x0, k, inside


def auc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_sea(results: dict, names: dict, cfg: Config, out: Path):
    keys = [k for k, v in results.items() if v]
    if not keys:
        return
    n = len(keys); ncol = 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.7 * nrow), facecolor="white", squeeze=False)
    for ax, k in zip(axes.ravel(), keys):
        r = results[k]; ax.set_facecolor("white")
        ax.fill_between(r["lags"], r["q05"], r["q95"], color="#cccccc", alpha=0.6, linewidth=0)
        ax.plot(r["lags"], r["composite"], color="#d62728", linewidth=1.6)
        ax.axhline(1.0, color="#777", linewidth=0.6); ax.axvline(0, color="#777", linewidth=0.6, linestyle=":")
        ax.set_title(f"{names.get(k, k)}  (n={r['n_events']}, p={r['p_value']:.3f})", fontsize=8.5)
        ax.tick_params(labelsize=7); ax.grid(True, linestyle=":", linewidth=0.4)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("Years relative to event", fontsize=8)
    for row in axes:
        row[0].set_ylabel("Index / pre-event mean", fontsize=8)
    handles = [Line2D([0], [0], color="#d62728", label=f"Composite of events with prior p>={cfg.p_target}"),
               Line2D([0], [0], color="#cccccc", linewidth=8, label=f"5-95% of {cfg.n_null} random composites")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 0.98), frameon=False, fontsize=8)
    fig.suptitle("DRYAS - Superposed Epoch Analysis per Site (ring width, selected method)", y=1.01)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_calibration(pairs: pd.DataFrame, calib: dict, cfg: Config, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white")
    ax = axes[0]; ax.set_facecolor("white")
    x = np.log10(pairs["pga_gal"].clip(lower=0.1)); y = pairs["z"]
    det = pairs["detected"].to_numpy()
    ax.scatter(x[~det], y[~det], s=12, color="#9e9e9e", alpha=0.6)
    ax.scatter(x[det], y[det], s=18, color="#d62728", alpha=0.85)
    ax.axhline(cfg.z_detect, color="#777", linestyle="--", linewidth=0.8)
    ax.set_xlabel("log10 prior PGA (gal)"); ax.set_ylabel("Response z-score (site gc distribution)")
    ax.set_title("Per (site, event) response vs prior ground motion")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax = axes[1]; ax.set_facecolor("white")
    bins = np.arange(0, 3.6, 0.25)
    ctr, frac, cnt = [], [], []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (x >= a) & (x < b)
        if m.sum() >= 3:
            ctr.append((a + b) / 2); frac.append(det[m].mean()); cnt.append(m.sum())
    ax.scatter(ctr, frac, s=np.array(cnt) * 4 + 10, color="#1f77b4", label="Observed detection rate (size ~ n pairs)")
    xx = np.linspace(0, 3.5, 200)
    ax.plot(xx, 1 / (1 + np.exp(-calib["prior_k"] * (xx - calib["prior_x0"]))), color="#9e9e9e", linestyle="--",
            label=f"003 prior (x0={calib['prior_x0']:.2f}, k={calib['prior_k']:.1f})")
    ax.plot(xx, 1 / (1 + np.exp(-calib["k"] * (xx - calib["x0"]))), color="#d62728",
            label=f"fitted (x0={calib['x0']:.2f}, k={calib['k']:.1f}), AUC={calib['auc']:.2f}")
    ax.set_xlabel("log10 prior PGA (gal)"); ax.set_ylabel("P(detected)"); ax.set_ylim(-0.02, 1.02)
    ax.set_title("Empirical calibration of the tree-response prior")
    ax.grid(True, linestyle=":", linewidth=0.5)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", label=f"detected (z>={cfg.z_detect})"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="#9e9e9e", label="not detected")]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.18), frameon=False, fontsize=8, ncol=2)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    fig.suptitle("DRYAS - Confronting the Physical Prior with the Trees", y=1.03)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_fraction(pairs: pd.DataFrame, calib: dict, out: Path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor="white"); ax.set_facecolor("white")
    det = pairs[pairs["detected"]]["frac_supp"].dropna(); nd = pairs[~pairs["detected"]]["frac_supp"].dropna()
    bins = np.linspace(0, 1, 21)
    ax.hist(nd, bins=bins, color="#9e9e9e", alpha=0.6, density=True, label=f"not detected (n={len(nd)})")
    ax.hist(det, bins=bins, color="#d62728", alpha=0.7, density=True, label=f"detected (n={len(det)})")
    if np.isfinite(calib.get("f_median_detected", np.nan)):
        ax.axvline(calib["f_median_detected"], color="#d62728", linestyle="--",
                   label=f"median f (detected) = {calib['f_median_detected']:.2f}")
    ax.axvline(0.6, color="#333", linestyle=":", label="002 injection assumption f = 0.6")
    ax.set_xlabel("Fraction of living trees below their personal 10th-percentile growth change")
    ax.set_ylabel("Density"); ax.set_title("Tree-level responding fraction at event years")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_rw_mxd(feat: pd.DataFrame, names: dict, out: Path):
    f = feat.dropna(subset=["mxd_index"])
    if f.empty:
        return
    sites = f["site_id"].unique()[:6]
    fig, axes = plt.subplots(len(sites), 1, figsize=(11, 1.9 * len(sites) + 1), sharex=True, facecolor="white", squeeze=False)
    for ax, s in zip(axes[:, 0], sites):
        d = f[f["site_id"] == s].sort_values("year"); ax.set_facecolor("white")
        ax.plot(d["year"], d["rw_index"], color="#1b5e20", linewidth=0.7, label="ring width")
        ax.plot(d["year"], d["mxd_index"], color="#8e24aa", linewidth=0.7, label="max density")
        r = d[["rw_index", "mxd_index"]].corr().iloc[0, 1]
        ax.text(0.005, 0.9, f"{names.get(s, s)}  r(RW,MXD)={r:.2f}", transform=ax.transAxes, fontsize=8, va="top")
        ax.grid(True, linestyle=":", linewidth=0.4); ax.tick_params(labelsize=7)
    axes[-1, 0].set_xlabel("Year (CE)")
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    fig.suptitle("DRYAS - Ring Width vs Maximum Latewood Density (aligned site-year grid)", y=1.0)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ---------------------------------------------------------------------------
def save_table(df, name, records):
    p = DATA_DIR / name
    df.to_parquet(p, index=False)
    records[name] = {"rows": int(len(df)), "sha256": _sha256_of_file(p), "path": str(p)}
    print(f"  wrote {p}  ({len(df):,} rows)")


def main():
    ap = argparse.ArgumentParser(description="DRYAS 004 — response features & calibration")
    ap.add_argument("--n-null", type=int, default=None)
    args = ap.parse_args()
    cfg = Config()
    if args.n_null:
        cfg.n_null = args.n_null
    rng = np.random.default_rng(cfg.seed)
    m2 = _import_002()

    rings = pd.read_parquet(DATA_DIR / "treering_network.parquet")
    sites = pd.read_parquet(DATA_DIR / "site_metadata.parquet")
    sel = json.load(open(DATA_DIR / "002_selection.json"))
    nodes = pd.read_parquet(DATA_DIR / "graph_nodes.parquet")
    se = pd.read_parquet(DATA_DIR / "site_event_pga.parquet")
    ev = pd.read_parquet(DATA_DIR / "eq_segment_assignment.parquet")
    chron_all = pd.read_parquet(DATA_DIR / "chronologies.parquet")
    prov = json.load(open(DATA_DIR / "provenance.json"))
    prior_cfg = prov.get("003", {}).get("config", {})
    method = sel["ring_width"]["selected"]
    cfg2 = m2.Config()
    dens_p = DATA_DIR / "treering_density.parquet"
    density = pd.read_parquet(dens_p) if dens_p.exists() else None
    names = dict(zip(nodes["site_id"], nodes["site_name"]))
    print(f"DRYAS 004 — method={method}, {len(nodes)} sites, {len(ev)} assigned events")

    # rebuild per-tree indices (identical standardization to 002), incl. merged sites
    rings, sites_m, merged = m2.merge_clusters(rings, sites, cfg2, "ring_width_mm")
    groups = m2.species_group(sites_m)
    feats, sea_rows, sea_results = [], [], {}
    site_ids = [s for s in nodes["site_id"] if s in set(rings["site_id"])]
    per_site = {}
    for s in site_ids:
        yrs, W, tree_ids = m2.to_wide(rings[rings["site_id"] == s], "ring_width_mm")
        res = m2.build_site(W, method, cfg2, rcs_group=None if method not in ("rcs", "sfrcs") else W)
        f = site_features(res["index"], res["chron"], yrs, cfg)
        f.insert(0, "site_id", s)
        per_site[s] = (yrs, res["chron"], f)
        feats.append(f)
        print(f"  [{s}] {W.shape[1]} cores, gc computed for {np.isfinite(f['gc']).sum()} years")
    feat = pd.concat(feats, ignore_index=True)

    # MXD alignment
    feat["mxd_index"] = np.nan
    if "max_density" in set(chron_all["proxy"]):
        mx = chron_all[(chron_all["proxy"] == "max_density") & (chron_all["method"] == method)][["site_id", "year", "index"]]
        mx = mx.rename(columns={"index": "mxd_index"})
        feat = feat.drop(columns="mxd_index").merge(mx, on=["site_id", "year"], how="left")

    # cluster & usable span from nodes
    feat = feat.merge(nodes[["site_id", "cluster", "usable_depth_first", "usable_depth_last"]], on="site_id", how="left")
    feat["usable_depth"] = (feat["year"] >= feat["usable_depth_first"]) & (feat["year"] <= feat["usable_depth_last"])
    # cluster coherence features (leave-one-out mean of other sites in the same cluster)
    cl = feat.groupby(["cluster", "year"]).agg(sum_gc=("gc", "sum"), n_gc=("gc", "count"),
                                                sum_fr=("frac_supp", "sum"), n_fr=("frac_supp", "count")).reset_index()
    feat = feat.merge(cl, on=["cluster", "year"], how="left")
    feat["cluster_gc"] = np.where(feat["n_gc"] > 1, (feat["sum_gc"] - feat["gc"].fillna(0)) / (feat["n_gc"] - 1), np.nan)
    feat["cluster_frac"] = np.where(feat["n_fr"] > 1, (feat["sum_fr"] - feat["frac_supp"].fillna(0)) / (feat["n_fr"] - 1), np.nan)
    feat = feat.drop(columns=["sum_gc", "n_gc", "sum_fr", "n_fr"])
    # max prior PGA of any assigned event that year (for supervised use only)
    ycol = "eff_year" if "eff_year" in se.columns else "year"
    mp = se.groupby(["site_id", ycol])["pga_gal"].max().rename("max_pga_prior").reset_index().rename(columns={ycol: "year"})
    feat = feat.merge(mp, on=["site_id", "year"], how="left")

    # per (site, event) response pairs
    # (site, event) pairs keyed on the growing-season effective year (003: eff_year)
    pairs = se[(se["mag"] >= cfg.mmin_pairs) & se["in_usable_span"]].merge(
        feat[["site_id", "year", "gc", "frac_supp", "depth"]].rename(columns={"year": ycol}),
        on=["site_id", ycol], how="inner")
    zs = []
    for s, d in feat.groupby("site_id"):
        g = d.loc[d["usable_depth"], "gc"].dropna()
        mu, sd = g.mean(), g.std()
        zs.append(pd.DataFrame({"site_id": [s], "gc_mu": [mu], "gc_sd": [sd]}))
    pairs = pairs.merge(pd.concat(zs), on="site_id", how="left")
    pairs["z"] = -(pairs["gc"] - pairs["gc_mu"]) / pairs["gc_sd"]
    pairs["detected"] = pairs["z"] >= cfg.z_detect
    pairs = pairs.dropna(subset=["z"]).reset_index(drop=True)

    # SEA per site on prior-selected events
    for s in site_ids:
        yrs, chron, f = per_site[s]
        tgt = se[(se["site_id"] == s) & (se["p_response"] >= cfg.p_target) & se["in_usable_span"]][ycol].unique().tolist()
        r = sea(chron, yrs, sorted(tgt), cfg, rng)
        sea_results[s] = r
        if r:
            for lag, c, a, b in zip(r["lags"], r["composite"], r["q05"], r["q95"]):
                sea_rows.append({"site_id": s, "lag": int(lag), "composite": float(c), "q05": float(a), "q95": float(b),
                                 "n_events": r["n_events"], "p_value": r["p_value"]})
    sea_df = pd.DataFrame(sea_rows)

    # calibration
    x = np.log10(pairs["pga_gal"].clip(lower=0.1).to_numpy()); y = pairs["detected"].to_numpy().astype(float)
    if len(pairs) >= 10 and y.sum() >= 3 and (1 - y).sum() >= 3:
        x0, k, inside = fit_logistic(x, y); a = auc(x, y.astype(int))
    else:
        x0, k, a, inside = float(prior_cfg.get("resp_x0_log10gal", 2.0)), float(prior_cfg.get("resp_k", 5.5)), np.nan, False
    fd = pairs[pairs["detected"]]["frac_supp"].dropna()
    calib = {"method": method, "n_pairs": int(len(pairs)), "n_detected": int(y.sum()),
             "prior_x0": float(prior_cfg.get("resp_x0_log10gal", 2.0)), "prior_k": float(prior_cfg.get("resp_k", 5.5)),
             "x0": x0, "k": k, "auc": a, "fit_converged_inside_bounds": bool(inside),
             "verdict": ("prior predicts detections" if (np.isfinite(a) and a >= 0.6 and inside) else
                         "NO usable relation between prior PGA and detection (AUC ~ chance or fit at bounds)"),
             "f_median_detected": float(fd.median()) if len(fd) else np.nan,
             "f_iqr_detected": [float(fd.quantile(0.25)), float(fd.quantile(0.75))] if len(fd) else None,
             "base_rate_detected": float(y.mean()) if len(y) else np.nan,
             "sea_p_values": {s: r["p_value"] for s, r in sea_results.items() if r},
             "sea_composite_year0_1": {s: r["stat"] for s, r in sea_results.items() if r}}

    records = {}
    save_table(feat, "features.parquet", records)
    save_table(sea_df, "004_sea.parquet", records)
    save_table(pairs, "004_event_response.parquet", records)
    with open(DATA_DIR / "004_response_calibration.json", "w") as fh:
        json.dump(calib, fh, indent=2, default=float)
    print(f"  wrote {DATA_DIR / '004_response_calibration.json'}")
    prov["004"] = {"script": "004_response_features.py", "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "config": asdict(cfg), "artifacts": records, "calibration": calib}
    json.dump(prov, open(DATA_DIR / "provenance.json", "w"), indent=2, default=str)

    print("rendering figures ...")
    fig_sea(sea_results, names, cfg, PDF_DIR / "004a_sea_composites.pdf")
    if len(pairs):
        fig_calibration(pairs, calib, cfg, PDF_DIR / "004b_prior_calibration.pdf")
        fig_fraction(pairs, calib, PDF_DIR / "004c_responding_fraction.pdf")
    fig_rw_mxd(feat, names, PDF_DIR / "004d_rw_vs_mxd.pdf")

    print("\n=== SUMMARY ===")
    print(f"  (site,event) pairs M>={cfg.mmin_pairs} in usable span: {len(pairs)}; detected (z>={cfg.z_detect}): {int(y.sum())} "
          f"(base rate {y.mean():.2f})")
    print(f"  prior logistic  : x0={calib['prior_x0']:.2f} k={calib['prior_k']:.1f}")
    print(f"  fitted logistic : x0={x0:.2f} (=> p=0.5 at {10**min(x0, 6):.0f} gal), k={k:.1f}, AUC={a:.2f}, "
          f"inside bounds={inside}")
    print(f"  verdict         : {calib['verdict']}")
    print(f"  responding fraction f (detected pairs): median {calib['f_median_detected']:.2f}, IQR {calib['f_iqr_detected']}")
    print("  SEA per site (composite index at years 0-1 vs null mean; one-sided p):")
    for s, r in sea_results.items():
        if r:
            print(f"    {names.get(s, s):>28}: n={r['n_events']:>2}  comp={r['stat']:.3f}  null={r['null_mean']:.3f}  p={r['p_value']:.3f}")
    strong = pairs[pairs["detected"]].sort_values("z", ascending=False).head(15)
    print("  strongest detected pairs:")
    for _, r in strong.iterrows():
        print(f"    {r['year']} {r['segment']:>8} M{r['mag']:.1f}  {names.get(r['site_id'], r['site_id']):>26}  "
              f"z={r['z']:.2f} f={r['frac_supp']:.2f} PGA={r['pga_gal']:.0f} gal")
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
