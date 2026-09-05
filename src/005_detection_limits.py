#!/usr/bin/env python3
# =============================================================================
# DRYAS  |  Script 005  —  DETECTION LIMITS, AMPLITUDE UPPER BOUNDS & ARRAY DESIGN
# =============================================================================
# WHY THIS SCRIPT EXISTS (pre-registered decision, see HANDOFF 7d)
# ---------------------------------------------------------------------------
# 004 found no relation between the physical ground-motion prior and observed
# growth suppression (AUC 0.46, detection rate = false-positive rate, SEA
# p >= 0.15 at all 12 sites, including the five sites that "should" have seen
# the 1952 Tokachi-oki M8.1). The pre-registered rule turns the project from a
# reconstruction into a quantitative detection-limit study:
#
#   "How large a growth suppression could a great subduction earthquake have
#    caused at ITRDB-type forest sites 100-250 km from the rupture, given that
#    none was detected?"  and  "what array WOULD detect one?"
#
# WHAT THIS SCRIPT DOES
#   A. DETECTION POWER per site. The 002 injection machinery (same selected
#      standardization, same tree-level responding fraction f as measured in
#      004) injects suppressions of amplitude A at K random usable years and
#      recomputes the 004 detection statistic z. This gives P(detect | A) per
#      site and the per-site sensitivity slope  dz/dA  (z-units per unit A).
#
#   B. AMPLITUDE UPPER BOUNDS. For every (site, event) pair with prior
#      p_response >= p_min, the 95% one-sided upper bound on the true
#      suppression amplitude consistent with the observed z is
#            A_UB = (z_obs + 1.645) / slope_site
#      (under amplitude A the statistic shifts by slope*A relative to its
#      null distribution, which is standard normal by construction in 004).
#      Events seen by several sites get an ARRAY bound from the stacked
#      statistic  z_arr = sum(z_i)/sqrt(S),  slope_arr = sum(slope_i)/sqrt(S).
#      Result: an empirical sensitivity curve  A_UB versus prior PGA, the first
#      array-scale quantification of far-field tree-ring insensitivity.
#
#   C. MAXIMUM DENSITY as a second proxy: the 004 SEA and (site, event) z-test
#      are repeated on the MXD chronologies.
#
#   D. ARRAY DESIGN. From the fitted scaling of slope with tree depth, the
#      number of sites S and trees per site n needed to detect an amplitude A
#      at 3 sigma:   z = slope(n) * A * sqrt(S) >= 3.  Written as a lookup
#      table and figure — this is the quantitative requirement for a purpose-
#      built dendroseismic array (near-fault sampling; cf. ARIADNE).
#
# INPUTS  001-004 outputs (see loaders in main)
# OUTPUTS data/005_power.parquet, data/005_upper_bounds.parquet,
#         data/005_mxd_sea.parquet, data/005_mxd_pairs.parquet,
#         data/005_array_design.json
#         PDF/005a_detection_power.pdf, 005b_amplitude_upper_bounds.pdf,
#             005c_mxd_sea.pdf, 005d_array_design.pdf
#
# USAGE   uv run python src/005_detection_limits.py        (~minutes)
#         uv run python src/005_detection_limits.py --quick
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
from scipy.stats import norm

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


def _import(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, SRC_DIR / fname)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


@dataclass
class Config:
    amplitudes: tuple = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
    n_inject: int = 30
    responding_fraction: float | None = None   # None -> median f measured in 004 (fallback 0.45)
    z_detect: float = 1.645
    p_min_pairs: float = 0.2                   # prior p_response threshold for upper-bound pairs
    key_events_mmin: float = 7.5
    design_target_z: float = 3.0
    design_amplitudes: tuple = (0.05, 0.10, 0.15, 0.20, 0.30)
    design_trees: tuple = (10, 20, 30, 50, 100)
    seed: int = 11
    quick: bool = False


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Detection statistic identical to 004 (gc-based z on the biweight chronology)
# ---------------------------------------------------------------------------
def z_at(chron: np.ndarray, years: np.ndarray, t0: int, mu: float, sd: float, m4) -> float:
    gc = m4.growth_change(chron, m4.Config().pre_window, m4.Config().post_window)
    i0 = int(np.searchsorted(years, t0))
    if i0 >= len(gc) or not np.isfinite(gc[i0]) or not (sd > 0):
        return np.nan
    return float(-(gc[i0] - mu) / sd)


def power_for_site(W, yrs, tree_ids, chron, idx, mu, sd, usable_mask, eq_years, m2, m4, cfg2, cfg, method, rng):
    """Inject amplitude A at K random usable years; return rows (A, t0, z_clean, z_inj)."""
    depth = np.isfinite(idx).sum(axis=1)
    cand = yrs[usable_mask & (depth >= cfg2.depth_for_detection)]
    cand = cand[(cand >= yrs.min() + 10) & (cand <= yrs.max() - 5)]
    bad = set()
    for y in eq_years:
        bad.update(range(int(y) - 10, int(y) + 11))
    cand = np.array([c for c in cand if c not in bad])
    if len(cand) == 0:
        return []
    rng.shuffle(cand)
    t0s = sorted(cand[: min(cfg.n_inject, len(cand))])
    rows = []
    for t0 in t0s:
        z_clean = z_at(chron, yrs, int(t0), mu, sd, m4)
        for A in cfg.amplitudes:
            W_inj, hit = m2.inject(W, yrs, int(t0), A, cfg2, rng)
            if method in ("rcs", "sfrcs"):
                res = m2.build_site(W_inj, method, cfg2, rcs_group=W_inj)
                ch = res["chron"]
            else:
                idx_i = idx.copy()
                for j in hit:
                    idx_i[:, j] = W_inj[:, j] / m2.detrend_series(W_inj[:, j], method, cfg2)
                ch = m2.chronology_from_indices(idx_i)
            rows.append({"t0": int(t0), "A": A, "z_clean": z_clean, "z_inj": z_at(ch, yrs, int(t0), mu, sd, m4),
                         "n_hit": int(len(hit)), "depth": int(depth[np.searchsorted(yrs, t0)])})
    return rows


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_power(power: pd.DataFrame, slopes: pd.DataFrame, names: dict, cfg: Config, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor="white")
    ax = axes[0]; ax.set_facecolor("white")
    cmap = plt.get_cmap("tab20")
    for i, (s, d) in enumerate(power.groupby("site_id")):
        g = d.groupby("A")["detected"].mean()
        ax.plot(g.index, g.values, "-o", color=cmap(i % 20), markersize=4, label=names.get(s, s))
    ax.axhline(0.95, color="#999", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Injected suppression amplitude A (year-0 growth loss, fraction)")
    ax.set_ylabel(f"P(detect) at z>={cfg.z_detect}"); ax.set_ylim(0, 1.02)
    ax.set_title("Single-site detection power (injection-recovery)")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.18), ncol=4, frameon=False, fontsize=7)
    ax = axes[1]; ax.set_facecolor("white")
    ax.scatter(slopes["depth_mean"], slopes["slope"], color="#1f77b4", s=30)
    for _, r in slopes.iterrows():
        ax.annotate(names.get(r["site_id"], r["site_id"]).split(" ")[0], (r["depth_mean"], r["slope"]),
                    xytext=(4, 3), textcoords="offset points", fontsize=7)
    if len(slopes) >= 3:
        c = np.polyfit(np.log(slopes["depth_mean"]), np.log(slopes["slope"]), 1)
        xx = np.linspace(slopes["depth_mean"].min(), slopes["depth_mean"].max(), 50)
        ax.plot(xx, np.exp(c[1]) * xx ** c[0], color="#d62728", linestyle="--", label=f"fit: slope ~ n^{c[0]:.2f}")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    ax.set_xlabel("Mean living-tree depth at injection years"); ax.set_ylabel("Sensitivity slope dz/dA (z per unit A)")
    ax.set_title("Sensitivity vs sample depth"); ax.grid(True, linestyle=":", linewidth=0.5)
    fig.suptitle("DRYAS - What the ITRDB Array Can Detect", y=1.02)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_bounds(ub: pd.DataFrame, arr: pd.DataFrame, names: dict, cfg: Config, out: Path):
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor="white"); ax.set_facecolor("white")
    x = np.log10(ub["pga_gal"].clip(lower=0.1)); y = ub["A_UB"]
    ax.scatter(x, y, s=14, color="#9e9e9e", alpha=0.6)
    if len(arr):
        xa = np.log10(arr["pga_mean_gal"].clip(lower=0.1))
        ax.scatter(xa, arr["A_UB_array"], s=60, color="#d62728", edgecolor="black", zorder=4)
        for _, r in arr.iterrows():
            ax.annotate(f"{r['year']} {r['segment']} M{r['mag']:.1f} (S={r['n_sites']})",
                        (np.log10(max(r['pga_mean_gal'], 0.1)), r["A_UB_array"]), xytext=(5, 4),
                        textcoords="offset points", fontsize=6.5, color="#7b1a1a")
    ax.set_xlabel("log10 prior PGA at the site (gal)")
    ax.set_ylabel("95% upper bound on year-0 growth suppression A")
    ax.set_title("Empirical far-field sensitivity bound: how much suppression could have occurred unseen")
    ax.set_ylim(0, min(1.0, max(0.05, np.nanpercentile(y, 99) * 1.1)))
    ax.grid(True, linestyle=":", linewidth=0.5)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#9e9e9e", label=f"(site, event) pairs, prior p>={cfg.p_min_pairs}"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markeredgecolor="black", markersize=9,
                      label=f"array stacks, M>={cfg.key_events_mmin}, >=2 sites")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_design(design: dict, cfg: Config, out: Path):
    fig, ax = plt.subplots(figsize=(8.5, 5), facecolor="white"); ax.set_facecolor("white")
    cmap = plt.get_cmap("viridis")
    for i, n in enumerate(cfg.design_trees):
        S = [design["required_sites"][str(n)][str(A)] for A in cfg.design_amplitudes]
        ax.plot(cfg.design_amplitudes, S, "-o", color=cmap(i / max(1, len(cfg.design_trees) - 1)), label=f"{n} trees / site")
    ax.set_yscale("log"); ax.set_xlabel("Target suppression amplitude A to detect at 3 sigma")
    ax.set_ylabel("Required number of coherent sites S")
    ax.set_title(f"Array design from the measured sensitivity scaling (slope ~ n^{design['depth_exponent']:.2f})")
    ax.grid(True, linestyle=":", linewidth=0.5, which="both")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def save_table(df, name, records):
    p = DATA_DIR / name
    df.to_parquet(p, index=False)
    records[name] = {"rows": int(len(df)), "sha256": _sha256_of_file(p), "path": str(p)}
    print(f"  wrote {p}  ({len(df):,} rows)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="DRYAS 005 — detection limits & array design")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-inject", type=int, default=None)
    args = ap.parse_args()
    cfg = Config()
    if args.quick:
        cfg.quick = True; cfg.n_inject = 8; cfg.amplitudes = (0.1, 0.2, 0.4)
    if args.n_inject:
        cfg.n_inject = args.n_inject
    rng = np.random.default_rng(cfg.seed)
    m2 = _import("dryas002", "002_chronology_standardization.py")
    m4 = _import("dryas004", "004_response_features.py")
    cfg2 = m2.Config(); cfg4 = m4.Config()

    rings = pd.read_parquet(DATA_DIR / "treering_network.parquet")
    sites = pd.read_parquet(DATA_DIR / "site_metadata.parquet")
    sel = json.load(open(DATA_DIR / "002_selection.json"))
    nodes = pd.read_parquet(DATA_DIR / "graph_nodes.parquet")
    se = pd.read_parquet(DATA_DIR / "site_event_pga.parquet")
    ev = pd.read_parquet(DATA_DIR / "eq_segment_assignment.parquet")
    pairs4 = pd.read_parquet(DATA_DIR / "004_event_response.parquet")
    calib = json.load(open(DATA_DIR / "004_response_calibration.json"))
    chron_all = pd.read_parquet(DATA_DIR / "chronologies.parquet")
    prov = json.load(open(DATA_DIR / "provenance.json"))
    method = sel["ring_width"]["selected"]
    f_meas = calib.get("f_median_detected")
    cfg2.responding_fraction = cfg.responding_fraction or (float(f_meas) if f_meas and np.isfinite(f_meas) else 0.45)
    names = dict(zip(nodes["site_id"], nodes["site_name"]))
    print(f"DRYAS 005 — method={method}, f={cfg2.responding_fraction:.2f} (from 004), "
          f"amplitudes={cfg.amplitudes}, K={cfg.n_inject}")

    # ---- A. detection power per site ------------------------------------
    rings, sites_m, merged = m2.merge_clusters(rings, sites, cfg2, "ring_width_mm")
    site_ids = [s for s in nodes["site_id"] if s in set(rings["site_id"])]
    power_rows, slope_rows = [], []
    for s in site_ids:
        yrs, W, tree_ids = m2.to_wide(rings[rings["site_id"] == s], "ring_width_mm")
        res = m2.build_site(W, method, cfg2, rcs_group=None if method not in ("rcs", "sfrcs") else W)
        chron, idx = res["chron"], res["index"]
        n = nodes[nodes["site_id"] == s].iloc[0]
        usable = (yrs >= n["usable_depth_first"]) & (yrs <= n["usable_depth_last"])
        gc = m4.growth_change(chron, cfg4.pre_window, cfg4.post_window)
        g = gc[usable & np.isfinite(gc)]
        mu, sd = float(np.mean(g)), float(np.std(g))
        ycol = "eff_year" if "eff_year" in se.columns else "year"
        eq_years = se[(se["site_id"] == s) & (se["p_response"] >= 0.2)][ycol].unique()
        rows = power_for_site(W, yrs, tree_ids, chron, idx, mu, sd, usable, eq_years, m2, m4, cfg2, cfg, method, rng)
        for r in rows:
            r["site_id"] = s
        power_rows += rows
        d = pd.DataFrame(rows)
        if len(d):
            d["dz"] = d["z_inj"] - d["z_clean"]
            slope = float(np.nansum(d["dz"] * d["A"]) / np.nansum(d["A"] ** 2))   # through-origin LS
            slope_rows.append({"site_id": s, "slope": slope, "depth_mean": float(d["depth"].mean()),
                               "gc_sd": sd, "n_trials": int(len(d))})
            print(f"  [{names.get(s, s)}] slope dz/dA = {slope:.2f}  (depth ~{d['depth'].mean():.0f})")
    power = pd.DataFrame(power_rows)
    power["detected"] = power["z_inj"] >= cfg.z_detect
    slopes = pd.DataFrame(slope_rows)

    # ---- B. amplitude upper bounds -------------------------------------
    ub = pairs4.merge(slopes[["site_id", "slope"]], on="site_id", how="inner")
    ub = ub[ub["p_response"] >= cfg.p_min_pairs].copy()
    ub["A_UB"] = (ub["z"] + 1.645) / ub["slope"]
    ub.loc[ub["A_UB"] < 0, "A_UB"] = 0.0
    # array stacks for key events
    arr_rows = []
    for eid, d in ub[ub["mag"] >= cfg.key_events_mmin].groupby("event_id"):
        if len(d) < 2:
            continue
        S = len(d)
        z_arr = float(d["z"].sum() / np.sqrt(S)); slope_arr = float(d["slope"].sum() / np.sqrt(S))
        arr_rows.append({"event_id": eid, "year": int(d["year"].iloc[0]), "segment": d["segment"].iloc[0],
                         "mag": float(d["mag"].iloc[0]), "n_sites": S, "sites": ",".join(sorted(d["site_id"])),
                         "z_array": z_arr, "p_one_sided": float(1 - norm.cdf(z_arr)),
                         "A_UB_array": max(0.0, (z_arr + 1.645) / slope_arr),
                         "pga_mean_gal": float(d["pga_gal"].mean()), "pga_max_gal": float(d["pga_gal"].max())})
    arr = pd.DataFrame(arr_rows).sort_values("year") if arr_rows else pd.DataFrame()

    # ---- C. MXD second proxy -------------------------------------------
    mxd_sea_rows, mxd_pairs, mxd_res = [], [], {}
    if "max_density" in set(chron_all["proxy"]):
        mx = chron_all[(chron_all["proxy"] == "max_density") & (chron_all["method"] == method)]
        for s, d in mx.groupby("site_id"):
            d = d.sort_values("year"); yrs_m = d["year"].to_numpy(); ch = d["index"].to_numpy()
            n = nodes[nodes["site_id"] == s]
            if n.empty:
                continue
            n = n.iloc[0]
            ycol = "eff_year" if "eff_year" in se.columns else "year"
            tgt = se[(se["site_id"] == s) & (se["p_response"] >= cfg4.p_target) & se["in_usable_span"]][ycol].unique().tolist()
            r = m4.sea(ch, yrs_m, sorted(tgt), cfg4, rng) if tgt else {}
            mxd_res[s] = r
            if r:
                mxd_sea_rows.append({"site_id": s, "n_events": r["n_events"], "p_value": r["p_value"], "stat": r["stat"]})
            gc = m4.growth_change(ch, cfg4.pre_window, cfg4.post_window)
            usable = (yrs_m >= n["usable_depth_first"]) & (yrs_m <= n["usable_depth_last"])
            g = gc[usable & np.isfinite(gc)]
            if len(g) < 30:
                continue
            mu, sd = float(np.mean(g)), float(np.std(g))
            for _, e in se[(se["site_id"] == s) & (se["mag"] >= cfg4.mmin_pairs) & se["in_usable_span"]].iterrows():
                ey = int(e[ycol])
                i0 = int(np.searchsorted(yrs_m, ey))
                if i0 < len(gc) and yrs_m[i0] == ey and np.isfinite(gc[i0]):
                    mxd_pairs.append({"site_id": s, "event_id": e["event_id"], "year": int(e["year"]), "mag": e["mag"],
                                      "pga_gal": e["pga_gal"], "z_mxd": -(gc[i0] - mu) / sd})
    mxd_sea = pd.DataFrame(mxd_sea_rows); mxd_pairs = pd.DataFrame(mxd_pairs)
    mxd_auc = np.nan
    if len(mxd_pairs):
        yv = (mxd_pairs["z_mxd"] >= cfg.z_detect).astype(int).to_numpy()
        mxd_auc = m4.auc(np.log10(mxd_pairs["pga_gal"].clip(lower=0.1).to_numpy()), yv)

    # ---- D. array design -----------------------------------------------
    design = {}
    if len(slopes) >= 3:
        c = np.polyfit(np.log(slopes["depth_mean"]), np.log(slopes["slope"]), 1)
        expo, coef = float(c[0]), float(np.exp(c[1]))
        req = {}
        for n in cfg.design_trees:
            slope_n = coef * n ** expo
            req[str(n)] = {str(A): float(np.ceil((cfg.design_target_z / (slope_n * A)) ** 2)) for A in cfg.design_amplitudes}
        design = {"depth_exponent": expo, "slope_coef": coef, "target_z": cfg.design_target_z,
                  "required_sites": req, "f_used": cfg2.responding_fraction,
                  "note": "z = slope(n) * A * sqrt(S); slope(n) fitted on the ITRDB sites (far-field noise level)."}

    # ---- persist ---------------------------------------------------------
    records = {}
    save_table(power, "005_power.parquet", records)
    save_table(slopes, "005_sensitivity_slopes.parquet", records)
    save_table(ub, "005_upper_bounds.parquet", records)
    if len(arr):
        save_table(arr, "005_upper_bounds_array.parquet", records)
    if len(mxd_sea):
        save_table(mxd_sea, "005_mxd_sea.parquet", records)
    if len(mxd_pairs):
        save_table(mxd_pairs, "005_mxd_pairs.parquet", records)
    design["mxd_auc"] = float(mxd_auc) if np.isfinite(mxd_auc) else None
    json.dump(design, open(DATA_DIR / "005_array_design.json", "w"), indent=2)
    prov["005"] = {"script": "005_detection_limits.py", "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "config": asdict(cfg), "artifacts": records, "design": design}
    json.dump(prov, open(DATA_DIR / "provenance.json", "w"), indent=2, default=str)

    print("rendering figures ...")
    if len(power):
        fig_power(power, slopes, names, cfg, PDF_DIR / "005a_detection_power.pdf")
    if len(ub):
        fig_bounds(ub, arr, names, cfg, PDF_DIR / "005b_amplitude_upper_bounds.pdf")
    if mxd_res and any(mxd_res.values()):
        m4.fig_sea({k: v for k, v in mxd_res.items() if v}, names, cfg4, PDF_DIR / "005c_mxd_sea.pdf")
    if design.get("required_sites"):
        fig_design(design, cfg, PDF_DIR / "005d_array_design.pdf")

    print("\n=== SUMMARY ===")
    print("  single-site sensitivity (z per unit amplitude) and amplitude for 95% detection:")
    for _, r in slopes.sort_values("slope", ascending=False).iterrows():
        d = power[power["site_id"] == r["site_id"]].groupby("A")["detected"].mean()
        a95 = next((a for a, p in d.items() if p >= 0.95), None)
        print(f"    {names.get(r['site_id'], r['site_id']):>28}: slope={r['slope']:.2f}  A95={a95 if a95 else '>%.2f' % max(cfg.amplitudes)}")
    if len(arr):
        print("  array upper bounds (95%) for key events:")
        for _, r in arr.iterrows():
            print(f"    {r['year']} {r['segment']:>7} M{r['mag']:.1f}  S={r['n_sites']}  z_arr={r['z_array']:+.2f} "
                  f"p={r['p_one_sided']:.2f}  A_UB={r['A_UB_array']:.3f}  PGA~{r['pga_mean_gal']:.0f} gal")
    if len(ub):
        hi = ub[ub["pga_gal"] >= 100]
        print(f"  pairs with prior PGA>=100 gal: n={len(hi)}, median A_UB={hi['A_UB'].median():.3f}, "
              f"10th pct={hi['A_UB'].quantile(0.1):.3f}")
    if len(mxd_sea):
        print("  MXD SEA p-values: " + ", ".join(f"{names.get(r['site_id'], r['site_id']).split(' ')[0]}={r['p_value']:.2f}"
                                               for _, r in mxd_sea.iterrows()))
        print(f"  MXD pair AUC vs prior PGA: {mxd_auc:.2f}")
    if design.get("required_sites"):
        print(f"  design: slope ~ n^{design['depth_exponent']:.2f}; sites needed at 3 sigma (30 trees/site): " +
              ", ".join(f"A={A}: S={int(design['required_sites']['30'][str(A)])}" for A in cfg.design_amplitudes))
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
