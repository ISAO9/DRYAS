#!/usr/bin/env python3
# =============================================================================
# DRYAS  |  Script 006  —  ABLATIONS, HEADLINE FIGURES & PRE-REGISTRATION RECORD
# =============================================================================
# WHAT THIS SCRIPT DOES
# ---------------------------------------------------------------------------
# 005 produced the paper's central result: 95% upper bounds on the growth
# suppression that great subduction earthquakes could have caused at the ITRDB
# Japan sites. A reviewer will ask how much those bounds depend on the choices
# made along the way. This script answers with five ablations, all recomputed
# from the same inputs and reported for the same key events:
#
#   1. TIMING      calendar year vs growing-season effective year
#                  (003: eff_year = year + [month >= 8]). This is physics, not a
#                  free choice: an autumn earthquake cannot alter a ring that is
#                  already formed. Baseline = effective year.
#   2. RESPONDING  the bound on A scales as 1/f (slope is proportional to f),
#      FRACTION f  so the f-independent quantity is the SITE-MEAN growth loss
#                  A*f. Bounds are reported in both units; A is shown for
#                  f in {0.30, 0.45 (measured), 0.60, 1.00}.
#   3. STACKING    equal-weight stack z_arr = sum(z)/sqrt(S) vs PGA-prior-
#                  weighted stack  z_w = sum(w z)/sqrt(sum w^2), w = p_response.
#   4. GEOMETRY    fault-segment rectangles expanded / shrunk by +-50 km
#                  (about the centroid); PGA, p_response, pair selection and
#                  bounds recomputed. Tests dependence on the approximate
#                  source geometry.
#   5. METHOD      the whole chain (per-tree indices -> gc -> sensitivity slope
#                  by injection -> pairs -> array bound) repeated with the two
#                  runner-up standardizations of 002 (negexp, spline67).
#
# It then writes the headline figure in f-independent units, an ablation
# summary figure, and a dated pre-registration record that lists every
# decision rule declared before the corresponding result was seen (from the
# HANDOFF) together with the outcome, so the paper can cite it.
#
# INPUTS  data/features.parquet, data/site_event_pga.parquet, data/graph_nodes.parquet,
#         data/eq_segment_assignment.parquet, data/fault_segments.geojson,
#         data/005_sensitivity_slopes.parquet, data/005_upper_bounds_array.parquet,
#         data/004_response_calibration.json, data/002_selection.json,
#         data/treering_network.parquet, data/site_metadata.parquet, data/provenance.json
# OUTPUTS data/006_ablations.parquet          one row per (variant, event)
#         data/006_ablation_summary.json
#         data/006_preregistration_record.md
#         PDF/006a_headline_bounds_Af.pdf     site-mean growth-loss bound vs PGA (array + pairs)
#         PDF/006b_ablation_summary.pdf       bounds per key event under every variant
#         PDF/006c_geometry_sensitivity.pdf   bound vs geometry perturbation
#
# USAGE   uv run python src/006_ablations_and_record.py   (method ablation ~minutes)
#         uv run python src/006_ablations_and_record.py --skip-method
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


def _import(name, fname):
    spec = importlib.util.spec_from_file_location(name, SRC_DIR / fname)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


@dataclass
class Config:
    z_one_sided: float = 1.645
    p_min_pairs: float = 0.2
    key_events_mmin: float = 7.5
    f_grid: tuple = (0.30, 0.45, 0.60, 1.00)
    geometry_shifts_km: tuple = (-50.0, -25.0, 0.0, 25.0, 50.0)
    methods_ablation: tuple = ("negexp", "spline67")
    method_n_inject: int = 12
    method_amplitudes: tuple = (0.1, 0.2, 0.4)
    seed: int = 5


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Core recomputations
# ---------------------------------------------------------------------------
def site_gc_stats(feat: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s, d in feat.groupby("site_id"):
        g = d.loc[d["usable_depth"], "gc"].dropna()
        rows.append({"site_id": s, "gc_mu": g.mean(), "gc_sd": g.std()})
    return pd.DataFrame(rows)


def pairs_from(se: pd.DataFrame, feat: pd.DataFrame, ycol: str, stats: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    p = se[(se["p_response"] >= cfg.p_min_pairs) & se["in_usable_span"]].merge(
        feat[["site_id", "year", "gc"]].rename(columns={"year": ycol}), on=["site_id", ycol], how="inner")
    p = p.merge(stats, on="site_id", how="left")
    p["z"] = -(p["gc"] - p["gc_mu"]) / p["gc_sd"]
    return p.dropna(subset=["z"]).reset_index(drop=True)


def array_bounds(pairs: pd.DataFrame, slopes: pd.DataFrame, cfg: Config, weighted: bool = False,
                 events: set | None = None) -> pd.DataFrame:
    p = pairs.merge(slopes[["site_id", "slope"]], on="site_id", how="inner")
    rows = []
    for eid, d in p.groupby("event_id"):
        if events is not None and eid not in events:
            continue
        if len(d) < 2 or d["mag"].iloc[0] < cfg.key_events_mmin:
            continue
        w = d["p_response"].to_numpy() if weighted else np.ones(len(d))
        z = float(np.sum(w * d["z"]) / np.sqrt(np.sum(w ** 2)))
        sl = float(np.sum(w * d["slope"]) / np.sqrt(np.sum(w ** 2)))
        rows.append({"event_id": eid, "year": int(d["year"].iloc[0]), "segment": d["segment"].iloc[0],
                     "mag": float(d["mag"].iloc[0]), "n_sites": int(len(d)), "z_array": z,
                     "p_one_sided": float(1 - norm.cdf(z)), "A_UB": max(0.0, (z + cfg.z_one_sided) / sl),
                     "pga_mean_gal": float(d["pga_gal"].mean())})
    return pd.DataFrame(rows)


def perturbed_site_event(nodes, segments, ev, m3, cfg3, shift_km: float) -> pd.DataFrame:
    """Recompute site x event PGA with each rectangle grown/shrunk by shift_km about its centroid."""
    segs = []
    for s in segments:
        cx, cy = m3.polygon_centroid(s["corners"])
        new = []
        for (x, y) in s["corners"]:
            dx, dy = x - cx, y - cy
            dist = np.hypot(dx * 111.0 * np.cos(np.radians(cy)), dy * 111.0)
            if dist == 0:
                new.append((x, y)); continue
            scale = max(0.05, (dist + shift_km) / dist)
            new.append((cx + dx * scale, cy + dy * scale))
        segs.append({**s, "corners": new})
    rows = []
    for _, e in ev.iterrows():
        seg = next((s for s in segs if s["id"] == e["segment"]), None)
        for _, n in nodes.iterrows():
            if seg is not None and e["mech"] != "intraplate":
                xs = m3.dist_point_to_polygon_km(n["lon"], n["lat"], seg["corners"])
                xs = max(xs, 10.0) if xs == 0 else xs
                x = float(np.hypot(xs, min(e["depth_km"], 30.0)))
            else:
                x = float(np.hypot(m3.haversine_km(n["lat"], n["lon"], e["lat"], e["lon"]), e["depth_km"]))
            pga = float(m3.pga_si_midorikawa_1999(e["mag"], np.array([x]), e["depth_km"], e["mech"])[0])
            rows.append({"site_id": n["site_id"], "event_id": e["event_id"], "year": int(e["year"]),
                         "eff_year": int(e["eff_year"]), "mag": e["mag"], "segment": e["segment"],
                         "pga_gal": pga, "p_response": float(m3.p_response(np.array([pga]), cfg3)[0]),
                         "in_usable_span": bool(n["usable_depth_first"] <= e["eff_year"] <= n["usable_depth_last"])})
    return pd.DataFrame(rows)


def method_chain(method: str, rings, sites, nodes, se, feat_cols, m2, m4, m5, cfg2, cfg4, cfg5, cfg, rng):
    """Rebuild gc features and injection slopes for another standardization method."""
    rows_feat, rows_slope = [], []
    site_ids = [s for s in nodes["site_id"] if s in set(rings["site_id"])]
    ycol = "eff_year" if "eff_year" in se.columns else "year"
    for s in site_ids:
        yrs, W, tree_ids = m2.to_wide(rings[rings["site_id"] == s], "ring_width_mm")
        res = m2.build_site(W, method, cfg2, rcs_group=None if method not in ("rcs", "sfrcs") else W)
        f = m4.site_features(res["index"], res["chron"], yrs, cfg4); f.insert(0, "site_id", s)
        n = nodes[nodes["site_id"] == s].iloc[0]
        usable = (yrs >= n["usable_depth_first"]) & (yrs <= n["usable_depth_last"])
        f["usable_depth"] = usable
        rows_feat.append(f)
        g = f.loc[usable, "gc"].dropna(); mu, sd = float(g.mean()), float(g.std())
        eq_years = se[(se["site_id"] == s) & (se["p_response"] >= 0.2)][ycol].unique()
        pr = m5.power_for_site(W, yrs, tree_ids, res["chron"], res["index"], mu, sd, usable, eq_years,
                               m2, m4, cfg2, cfg5, method, rng)
        d = pd.DataFrame(pr)
        if len(d):
            d["dz"] = d["z_inj"] - d["z_clean"]
            rows_slope.append({"site_id": s, "slope": float(np.nansum(d["dz"] * d["A"]) / np.nansum(d["A"] ** 2))})
        print(f"    [{method}] {s}: slope={rows_slope[-1]['slope'] if rows_slope and rows_slope[-1]['site_id']==s else float('nan'):.2f}")
    return pd.concat(rows_feat, ignore_index=True), pd.DataFrame(rows_slope)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_headline(pairs, arr, f_meas, cfg: Config, out: Path):
    fig, ax = plt.subplots(figsize=(10, 5.8), facecolor="white"); ax.set_facecolor("white")
    pairs = pairs.copy(); pairs["Af"] = pairs["A_UB"] * f_meas
    x = np.log10(pairs["pga_gal"].clip(lower=1))
    ax.scatter(x, pairs["Af"], s=14, color="#9e9e9e", alpha=0.55)
    if len(arr):
        xa = np.log10(arr["pga_mean_gal"].clip(lower=1)); ya = arr["A_UB"] * f_meas
        ax.scatter(xa, ya, s=70, color="#d62728", edgecolor="black", zorder=4)
        for _, r in arr.iterrows():
            ax.annotate(f"{r['year']} {r['segment']} M{r['mag']:.1f} (S={r['n_sites']})",
                        (np.log10(max(r['pga_mean_gal'], 1)), r["A_UB"] * f_meas), xytext=(6, 3),
                        textcoords="offset points", fontsize=6.8, color="#7b1a1a")
    ax.set_xlabel("log10 prior PGA at the site (gal), Si & Midorikawa (1999)")
    ax.set_ylabel("95% upper bound on site-mean year-0 growth loss  (A x f)")
    ax.set_title("Far-field tree-ring insensitivity to great Japanese earthquakes (ITRDB array, 13 sites)")
    ax.set_ylim(0, min(0.6, max(0.1, np.nanpercentile(pairs["Af"], 99) * 1.15)))
    ax.grid(True, linestyle=":", linewidth=0.5)
    for lvl, lab in ((0.05, "5%"), (0.10, "10%")):
        ax.axhline(lvl, color="#bbbbbb", linewidth=0.6, linestyle="--"); ax.text(ax.get_xlim()[0] + 0.02, lvl + 0.004, lab, fontsize=7, color="#777")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#9e9e9e", label=f"(site, event) pairs, prior p>={cfg.p_min_pairs}, M>=6.5"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markeredgecolor="black", markersize=9,
                      label=f"array stacks, M>={cfg.key_events_mmin}, >=2 sites (effective year)")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_ablation(abl: pd.DataFrame, f_meas: float, out: Path):
    ev_order = abl[abl["variant"] == "baseline"].sort_values("year")["event_id"].tolist()
    variants = [v for v in abl["variant"].unique()]
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(ev_order) + 2.5), facecolor="white"); ax.set_facecolor("white")
    cmap = plt.get_cmap("tab10")
    labels = {}
    for i, eid in enumerate(ev_order):
        d = abl[abl["event_id"] == eid]
        r0 = d[d["variant"] == "baseline"]
        labels[i] = f"{int(r0['year'].iloc[0])} {r0['segment'].iloc[0]} M{r0['mag'].iloc[0]:.1f} (S={int(r0['n_sites'].iloc[0])})" if len(r0) else eid
        for j, v in enumerate(variants):
            rv = d[d["variant"] == v]
            if len(rv):
                ax.scatter(rv["A_UB"] * f_meas, i + (j - len(variants) / 2) * 0.09, color=cmap(j % 10), s=32 if v == "baseline" else 18,
                           edgecolor="black" if v == "baseline" else "none", zorder=3)
    ax.set_yticks(range(len(ev_order))); ax.set_yticklabels([labels[i] for i in range(len(ev_order))], fontsize=8)
    ax.set_xlabel(f"95% upper bound on site-mean growth loss (A x f, f={f_meas:.2f})")
    ax.set_title("Ablations: how the array upper bounds move under alternative choices")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5)
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(j % 10), markeredgecolor="black" if v == "baseline" else "none",
                      label=v, markersize=8) for j, v in enumerate(variants)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_geometry(geo: pd.DataFrame, f_meas: float, out: Path):
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor="white"); ax.set_facecolor("white")
    cmap = plt.get_cmap("tab10")
    for i, (eid, d) in enumerate(geo.groupby("event_id")):
        d = d.sort_values("shift_km")
        ax.plot(d["shift_km"], d["A_UB"] * f_meas, "-o", color=cmap(i % 10), markersize=4,
                label=f"{int(d['year'].iloc[0])} {d['segment'].iloc[0]} M{d['mag'].iloc[0]:.1f}")
    ax.set_xlabel("Segment rectangle expanded (+) / shrunk (-) by (km)"); ax.set_ylabel("Array bound on A x f")
    ax.set_title("Sensitivity of the array bounds to the approximate source geometry")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ---------------------------------------------------------------------------
def write_record(prov: dict, abl: pd.DataFrame, calib: dict, summary: dict, path: Path):
    def ts(k):
        return prov.get(k, {}).get("generated_utc", "n/a")
    lines = [
        "# DRYAS — Pre-registration record (auto-generated by 006)", "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
        "## Declared rules and outcomes", "",
        "| # | Rule (declared before the result) | Declared in | Outcome | Evidence |", "|---|---|---|---|---|",
        "| R1 | Standardization methods with median injected-signal preservation (A=0.35) < 0.70 are rejected; among survivors pick highest median SNR | 002 design (HANDOFF §7), before real run | all 5 methods passed; spline100 selected (SNR 1.08) | 002_selection.json (%s) |" % ts("002"),
        "| R2 | Usable years for event detection = living-tree depth >= 10 (EPS >= 0.85 gate kept only as climate-quality flag) | 002 v2, after seeing EPS cut Yakushima | YAKU-M usable 644-2005 | chronologies.parquet |",
        "| R3 | Tree-response prior: logistic in log10 PGA (x0=2.0, k=5.5) declared as placeholder to be re-fitted | 003 design | fitted x0 at bound (4.5), AUC %.2f -> no relation | 004_response_calibration.json (%s) |" % (calib.get("auc", float("nan")), ts("004")),
        "| R4 | If AUC < 0.6 and Hokkaido SEA (incl. 1952) p > 0.05 after bug fixes, the project becomes a detection-limit study | HANDOFF §7d, before 004 rerun | condition met (AUC %.2f, all SEA p >= 0.15) | 004 SUMMARY, 004a/004b |" % calib.get("auc", float("nan")),
        "| R5 | Upper bound A_UB = (z_obs + 1.645)/slope with slope from injection at the measured f; array stack equal-weight | 005 design | reported in 005/006 | 005_upper_bounds_array.parquet (%s) |" % ts("005"),
        "| R6 | Growing-season effective year (month >= 8 -> next ring) adopted as baseline before seeing its effect | 006 design (HANDOFF §7f) | see ablation 'timing' below | 006_ablations.parquet |",
        "", "## Ablation summary (array bounds on site-mean growth loss A x f)", "",
    ]
    if len(abl):
        piv = abl.pivot_table(index="event_id", columns="variant", values="A_UB")
        piv = piv.loc[abl[abl["variant"] == "baseline"].sort_values("year")["event_id"]]
        f = summary["f_measured"]
        lines.append("| event | " + " | ".join(piv.columns) + " |")
        lines.append("|---|" + "---|" * len(piv.columns))
        meta = abl[abl["variant"] == "baseline"].set_index("event_id")
        for eid, r in piv.iterrows():
            lab = f"{int(meta.loc[eid, 'year'])} {meta.loc[eid, 'segment']} M{meta.loc[eid, 'mag']:.1f}"
            lines.append(f"| {lab} | " + " | ".join("" if not np.isfinite(v) else f"{v * f:.3f}" for v in r.values) + " |")
    lines += ["", "## Key numbers", "", "```", json.dumps(summary, indent=2, default=float), "```"]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="DRYAS 006 — ablations & record")
    ap.add_argument("--skip-method", action="store_true")
    args = ap.parse_args()
    cfg = Config(); rng = np.random.default_rng(cfg.seed)
    m2 = _import("dryas002", "002_chronology_standardization.py")
    m3 = _import("dryas003", "003_spatial_graph.py")
    m4 = _import("dryas004", "004_response_features.py")
    m5 = _import("dryas005", "005_detection_limits.py")
    cfg2, cfg3, cfg4, cfg5 = m2.Config(), m3.Config(), m4.Config(), m5.Config()
    cfg5.n_inject = cfg.method_n_inject; cfg5.amplitudes = cfg.method_amplitudes

    feat = pd.read_parquet(DATA_DIR / "features.parquet")
    se = pd.read_parquet(DATA_DIR / "site_event_pga.parquet")
    nodes = pd.read_parquet(DATA_DIR / "graph_nodes.parquet")
    ev = pd.read_parquet(DATA_DIR / "eq_segment_assignment.parquet")
    slopes = pd.read_parquet(DATA_DIR / "005_sensitivity_slopes.parquet")
    calib = json.load(open(DATA_DIR / "004_response_calibration.json"))
    sel = json.load(open(DATA_DIR / "002_selection.json"))
    prov = json.load(open(DATA_DIR / "provenance.json"))
    segments = [{**ft["properties"], "corners": [tuple(c) for c in ft["geometry"]["coordinates"][0][:-1]]}
                for ft in json.load(open(DATA_DIR / "fault_segments.geojson"))["features"]]
    f_meas = float(calib.get("f_median_detected") or 0.45)
    cfg2.responding_fraction = f_meas
    stats = site_gc_stats(feat)
    if "eff_year" not in se.columns:
        raise SystemExit("site_event_pga.parquet has no eff_year: rerun 001 -> 003 -> 004 -> 005 first")
    print(f"DRYAS 006 — method={sel['ring_width']['selected']}, f={f_meas:.2f}, {len(slopes)} sites with slopes")

    # baseline (effective year, equal weights, measured f)
    p_eff = pairs_from(se, feat, "eff_year", stats, cfg)
    base = array_bounds(p_eff, slopes, cfg); base["variant"] = "baseline"
    key = set(base["event_id"])
    abl = [base]
    # 1. timing
    p_cal = pairs_from(se, feat, "year", stats, cfg)
    t = array_bounds(p_cal, slopes, cfg, events=None); t["variant"] = "timing: calendar year"; abl.append(t)
    # 3. weighted stack
    w = array_bounds(p_eff, slopes, cfg, weighted=True, events=key); w["variant"] = "stack: PGA-weighted"; abl.append(w)
    # 4. geometry
    geo_rows = []
    for sh in cfg.geometry_shifts_km:
        se_p = perturbed_site_event(nodes, segments, ev, m3, cfg3, sh)
        pp = pairs_from(se_p, feat, "eff_year", stats, cfg)
        g = array_bounds(pp, slopes, cfg, events=key); g["shift_km"] = sh; geo_rows.append(g)
        if sh != 0.0:
            gg = g.copy(); gg["variant"] = f"geometry: {sh:+.0f} km"; abl.append(gg)
        print(f"  geometry shift {sh:+.0f} km: {len(g)} key events recomputed")
    geo = pd.concat(geo_rows, ignore_index=True)
    # 5. method
    if not args.skip_method:
        rings = pd.read_parquet(DATA_DIR / "treering_network.parquet")
        sites = pd.read_parquet(DATA_DIR / "site_metadata.parquet")
        rings, sites_m, _ = m2.merge_clusters(rings, sites, cfg2, "ring_width_mm")
        for meth in cfg.methods_ablation:
            print(f"  method ablation: {meth}")
            f_m, s_m = method_chain(meth, rings, sites_m, nodes, se, None, m2, m4, m5, cfg2, cfg4, cfg5, cfg, rng)
            st_m = site_gc_stats(f_m)
            pm = pairs_from(se, f_m, "eff_year", st_m, cfg)
            am = array_bounds(pm, s_m, cfg, events=key); am["variant"] = f"method: {meth}"; abl.append(am)
    abl = pd.concat(abl, ignore_index=True)
    # 2. f grid (pure rescaling, reported in the summary)
    f_table = {str(f): {r["event_id"]: float(r["A_UB"] * f_meas / f) for _, r in base.iterrows()} for f in cfg.f_grid}

    # persist
    records = {}
    abl.to_parquet(DATA_DIR / "006_ablations.parquet", index=False)
    records["006_ablations.parquet"] = {"rows": int(len(abl)), "sha256": _sha(DATA_DIR / "006_ablations.parquet")}
    print(f"  wrote {DATA_DIR / '006_ablations.parquet'}  ({len(abl):,} rows)")
    piv = abl.pivot_table(index="event_id", columns="variant", values="A_UB")
    spread = (piv.max(axis=1) - piv.min(axis=1))
    summary = {"f_measured": f_meas, "baseline_bounds_A": base.set_index("event_id")["A_UB"].round(4).to_dict(),
               "baseline_bounds_Af": (base.set_index("event_id")["A_UB"] * f_meas).round(4).to_dict(),
               "max_spread_across_variants_A": spread.round(4).to_dict(),
               "A_for_f_grid": f_table,
               "timing_changed_events": [e for e in key if e in set(t["event_id"]) and
                                         abs(float(t.set_index("event_id").loc[e, "A_UB"]) - float(base.set_index("event_id").loc[e, "A_UB"])) > 0.02]}
    json.dump(summary, open(DATA_DIR / "006_ablation_summary.json", "w"), indent=2, default=float)
    write_record(prov, abl, calib, summary, DATA_DIR / "006_preregistration_record.md")
    print(f"  wrote {DATA_DIR / '006_preregistration_record.md'}")
    prov["006"] = {"script": "006_ablations_and_record.py", "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "config": asdict(cfg), "artifacts": records, "summary": summary}
    json.dump(prov, open(DATA_DIR / "provenance.json", "w"), indent=2, default=str)

    print("rendering figures ...")
    pairs_ub = p_eff.merge(slopes[["site_id", "slope"]], on="site_id", how="inner")
    pairs_ub["A_UB"] = ((pairs_ub["z"] + cfg.z_one_sided) / pairs_ub["slope"]).clip(lower=0)
    fig_headline(pairs_ub, base, f_meas, cfg, PDF_DIR / "006a_headline_bounds_Af.pdf")
    fig_ablation(abl, f_meas, PDF_DIR / "006b_ablation_summary.pdf")
    fig_geometry(geo, f_meas, PDF_DIR / "006c_geometry_sensitivity.pdf")

    print("\n=== SUMMARY ===")
    print(f"  f (measured) = {f_meas:.2f}; bounds below are site-mean growth loss A x f (95%, one-sided)")
    print(piv.mul(f_meas).round(3).loc[base.sort_values('year')['event_id']].to_string())
    print(f"  events whose bound moved > 0.02 (A) with calendar-year timing: {summary['timing_changed_events']}")
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
