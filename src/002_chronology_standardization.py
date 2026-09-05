#!/usr/bin/env python3
# =============================================================================
# DRYAS  |  Script 002  —  CHRONOLOGY BUILDING & SIGNAL-PRESERVING STANDARDIZATION
# =============================================================================
# WHAT THIS SCRIPT DOES
# ---------------------------------------------------------------------------
# Tree-ring widths carry a biological age trend and a climate signal on top of
# the earthquake response we want. Standard dendro "detrending" removes the
# first two — but the usual choices (flexible splines, signal-free RCS) can also
# erase a 1-5 year growth suppression, i.e. exactly the earthquake signature.
# (MOIRAI lesson: a separation step that is not validated against the target
# signal will silently destroy it.)
#
# This script therefore does three things for every site (ring width and, where
# available, maximum latewood density):
#
#   1. DETRENDING, five ways, from conservative to aggressive
#        negexp    : modified negative exponential / non-positive linear / mean
#        spline100 : smoothing spline, 50% frequency response at 100 yr (rigid)
#        spline67  : smoothing spline, 50% response at 2/3 of series length
#                    (the classic Cook & Peters 1981 choice; flexible)
#        rcs       : Regional Curve Standardization (one age-curve per species
#                    group; cambial age = ring count from first measured ring,
#                    because ITRDB .rwl files carry no pith-offset information)
#        sfrcs     : signal-free RCS (Melvin & Briffa 2008, iterated)
#      Indices are ratios (width / curve). Site chronologies are Tukey biweight
#      robust means; sample depth, running rbar and EPS (Wigley et al. 1984)
#      are computed so later scripts can gate usable years (EPS >= 0.85, n >= 5).
#
#   2. INJECTION-RECOVERY TEST (the core validation)
#      For each site and method, an earthquake-shaped multiplicative suppression
#      (profile 1.0/0.6/0.35/0.2/0.1 over 5 years, amplitude A) is injected into
#      a random fraction f of the living trees at K random years away from any
#      catalogued M>=7 event, the site is re-standardized, and the chronology
#      difference (clean minus injected) is matched against the profile:
#          preservation = recovered amplitude (arithmetic-mean chronology) / (A*f)
#                         -> 1.0 means detrending removed nothing of the signal
#          leakage      = mean absolute chronology distortion outside the window
#          snr          = recovered amplitude / matched-filter noise of the clean
#                         chronology (how detectable a real event would be)
#      The paired (clean vs injected) design cancels climate noise exactly, so
#      the numbers measure the method's transfer function for the target signal.
#
#   3. METHOD SELECTION, with the rule written down:
#      a method is REJECTED if median preservation (A = 0.35) < 0.70; among the
#      survivors the one with the highest median SNR is selected per proxy.
#      The chronologies of ALL methods are saved (downstream scripts may run
#      sensitivity analyses), and the selection is recorded in JSON.
#
# INPUTS  (from 001)
#   data/treering_network.parquet, data/site_metadata.parquet,
#   data/eq_catalog.parquet, data/treering_density.parquet (optional)
#
# OUTPUTS
#   data/chronologies.parquet          site_id, proxy, method, year, index, n,
#                                      rbar, eps, usable
#   data/002_preservation.parquet      one row per injection trial
#   data/002_climate_metrics.parquet   variance removed / lag-1 AC per site-method
#   data/002_selection.json            selected method per proxy + rationale
#   data/provenance.json               (appended: "002" section)
#   PDF/002a_chronologies_selected.pdf     chronology per site (selected method),
#                                          sample depth, EPS-usable years, M>=7 events
#   PDF/002b_preservation_tradeoff.pdf     preservation vs climate removal per method
#   PDF/002c_eps_sample_depth.pdf          EPS and depth per site for the selected method
#   PDF/002d_injection_example.pdf         worked example of one injection trial
#
# FIGURE CONVENTIONS: white background, English, legends in the margin.
#
# USAGE
#   uv run python src/002_chronology_standardization.py
#   uv run python src/002_chronology_standardization.py --n-inject 10 --quick
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solveh_banded
from scipy.optimize import curve_fit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore", category=RuntimeWarning)

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
DATA_DIR = ROOT / "data"
PDF_DIR = ROOT / "PDF"
PDF_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    methods: tuple = ("negexp", "spline100", "spline67", "rcs", "sfrcs")
    spline_rigid_period: float = 100.0     # years, 50% frequency response
    spline_flex_fraction: float = 2.0 / 3  # of series length (Cook & Peters 1981)
    min_series_len: int = 30               # shorter cores are not detrended
    sf_iterations: int = 5                 # signal-free RCS iterations
    rcs_curve_smooth_period: float = 30.0  # smoothing of the regional age curve
    # chronology quality
    eps_window: int = 50
    eps_step: int = 10
    eps_threshold: float = 0.85
    min_depth: int = 5                     # minimum depth for the EPS-based (climate-quality) gate
    depth_for_detection: int = 10          # depth gate for event detection (usable_depth)
    merge_km: float = 15.0                 # merge same-species collections closer than this
    # injection-recovery test
    profile: tuple = (1.0, 0.6, 0.35, 0.2, 0.1)   # same recovery shape as 001 synthetic
    amplitudes: tuple = (0.20, 0.35, 0.50)
    responding_fraction: float = 0.6
    n_inject: int = 25                      # injection years per site
    exclusion_mag: float = 7.0              # keep injections away from real M>=7 events
    exclusion_km: float = 250.0
    exclusion_years: int = 10
    leak_window: tuple = (-10, 15)          # years relative to t0 used for leakage
    # selection rule
    preservation_threshold: float = 0.70
    selection_amplitude: float = 0.35
    seed: int = 21
    quick: bool = False


PROFILE_LEN = 5


# ---------------------------------------------------------------------------
# Small numerical helpers
# ---------------------------------------------------------------------------
def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def lam_for_period(p: float) -> float:
    """
    Smoothing parameter of a second-difference (Whittaker / Hodrick-Prescott)
    smoother whose amplitude response is exactly 50% at period p:
        H(w) = 1 / (1 + lam * (2 - 2 cos w)^2),  H = 1/2  at w = 2*pi/p.
    This is the discrete analogue of the Cook & Peters (1981) frequency-response
    definition used by dendro software for smoothing splines.
    """
    w = 2 * np.pi / p
    return 1.0 / (2 - 2 * np.cos(w)) ** 2


def whittaker_smooth(y: np.ndarray, lam: float) -> np.ndarray:
    """Second-difference penalized smoother (banded Cholesky solve). y: no NaNs."""
    n = len(y)
    if n < 4:
        return np.full(n, np.nan if n == 0 else np.nanmean(y))
    # (I + lam D'D) x = y, D = second difference matrix; build upper banded form
    ab = np.zeros((3, n))
    main = np.full(n, 1.0 + 6.0 * lam)
    main[0] = main[-1] = 1.0 + lam
    main[1] = main[-2] = 1.0 + 5.0 * lam
    off1 = np.full(n - 1, -4.0 * lam)
    off1[0] = off1[-1] = -2.0 * lam
    off2 = np.full(n - 2, lam)
    ab[2, :] = main
    ab[1, 1:] = off1
    ab[0, 2:] = off2
    return solveh_banded(ab, y, lower=False)


def _fill_gaps(y: np.ndarray) -> np.ndarray:
    """Linear interpolation of interior NaNs (edges left as-is)."""
    y = y.copy()
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return y
    idx = np.arange(len(y))
    y[~ok] = np.interp(idx[~ok], idx[ok], y[ok])
    return y


def fit_negexp(y: np.ndarray) -> np.ndarray:
    """
    Modified negative exponential (Fritts 1969): y = a*exp(-b t) + k with
    a > 0, b > 0, k > 0. Falls back to a non-positive-slope line, then to the mean.
    """
    t = np.arange(len(y), dtype=float)
    ok = np.isfinite(y)
    yy, tt = y[ok], t[ok]
    if ok.sum() >= 8:
        try:
            k0 = max(np.nanmin(yy), 1e-3)
            a0 = max(yy[: max(3, len(yy) // 10)].mean() - k0, 1e-3)
            p, _ = curve_fit(lambda t, a, b, k: a * np.exp(-b * t) + k, tt, yy,
                             p0=(a0, 0.02, k0), bounds=([0, 1e-6, 0], [np.inf, 5, np.inf]),
                             maxfev=4000)
            curve = p[0] * np.exp(-p[1] * t) + p[2]
            if np.all(curve > 0):
                return curve
        except Exception:
            pass
    slope, inter = np.polyfit(tt, yy, 1)
    if slope <= 0:
        curve = inter + slope * t
        if np.all(curve > 0):
            return curve
    return np.full(len(y), np.nanmean(yy))


def biweight_mean(x: np.ndarray, c: float = 9.0) -> float:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    if len(x) < 3:
        return float(np.mean(x))
    m = np.median(x)
    mad = np.median(np.abs(x - m))
    if mad == 0:
        return float(m)
    u = (x - m) / (c * mad)
    w = (1 - u ** 2) ** 2
    w[np.abs(u) >= 1] = 0
    if w.sum() == 0:
        return float(m)
    return float(np.sum(w * x) / np.sum(w))


# ---------------------------------------------------------------------------
# Detrending per series and per site
# ---------------------------------------------------------------------------
def detrend_series(y: np.ndarray, method: str, cfg: Config) -> np.ndarray:
    """Return the growth curve for one core (same length as y, NaN where y is NaN)."""
    ok = np.isfinite(y)
    if ok.sum() < cfg.min_series_len:
        return np.full(len(y), np.nan)
    first, last = np.argmax(ok), len(y) - np.argmax(ok[::-1])
    seg = _fill_gaps(y[first:last])
    if method == "negexp":
        curve = fit_negexp(seg)
    elif method == "spline100":
        curve = whittaker_smooth(seg, lam_for_period(cfg.spline_rigid_period))
    elif method == "spline67":
        curve = whittaker_smooth(seg, lam_for_period(max(cfg.spline_flex_fraction * len(seg), 10)))
    else:
        raise ValueError(method)
    curve = np.where(curve > 1e-6, curve, 1e-6)
    out = np.full(len(y), np.nan)
    out[first:last] = curve
    out[~ok] = np.nan
    return out


def rcs_curve(wide: np.ndarray, smooth_period: float) -> np.ndarray:
    """
    Regional curve: mean value by cambial age (age = ring count from first
    measured ring), smoothed, with a minimum replication of 3 trees per age.
    Returns array indexed by age (0-based).
    """
    n_years, n_trees = wide.shape
    ages_max = n_years
    sums = np.zeros(ages_max)
    cnts = np.zeros(ages_max)
    for j in range(n_trees):
        col = wide[:, j]
        ok = np.isfinite(col)
        if not ok.any():
            continue
        first = np.argmax(ok)
        idx = np.arange(first, n_years)
        v = col[first:]
        m = np.isfinite(v)
        sums[idx[m] - first] += v[m]
        cnts[idx[m] - first] += 1
    curve = np.where(cnts >= 3, sums / np.maximum(cnts, 1), np.nan)
    valid = np.isfinite(curve)
    if valid.sum() < 5:
        return np.full(ages_max, np.nanmean(wide))
    last_valid = np.max(np.where(valid)[0])
    seg = _fill_gaps(curve[: last_valid + 1])
    seg = whittaker_smooth(seg, lam_for_period(smooth_period))
    seg = np.where(seg > 1e-6, seg, 1e-6)
    out = np.full(ages_max, seg[-1])   # extrapolate flat beyond replication
    out[: last_valid + 1] = seg
    return out


def _apply_rcs(wide: np.ndarray, curve: np.ndarray) -> np.ndarray:
    n_years, n_trees = wide.shape
    idx = np.full_like(wide, np.nan, dtype=float)
    for j in range(n_trees):
        col = wide[:, j]
        ok = np.isfinite(col)
        if not ok.any() or ok.sum() < 1:
            continue
        first = np.argmax(ok)
        ages = np.arange(n_years) - first
        m = ok & (ages >= 0)
        idx[m, j] = col[m] / curve[ages[m]]
    return idx


def chronology_from_indices(idx: np.ndarray, robust: bool = True) -> np.ndarray:
    """Site chronology: Tukey biweight mean (product) or arithmetic mean (used for the
    injection bookkeeping, where the expected response is exactly A*f*profile)."""
    if robust:
        return np.array([biweight_mean(idx[i]) for i in range(idx.shape[0])])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(idx, axis=1)


def build_site(wide: np.ndarray, method: str, cfg: Config, rcs_group: np.ndarray | None = None):
    """
    wide: years x trees matrix (NaN = no ring). rcs_group: optional larger
    matrix (same year axis) used to fit the regional curve for rcs/sfrcs.
    Returns dict(index=wide indices, chron=chronology, curve=growth curves).
    """
    n_years, n_trees = wide.shape
    # drop cores shorter than min_series_len from the chronology
    keep = np.isfinite(wide).sum(axis=0) >= cfg.min_series_len
    w = wide.copy()
    w[:, ~keep] = np.nan
    if method in ("negexp", "spline100", "spline67"):
        curves = np.column_stack([detrend_series(w[:, j], method, cfg) for j in range(n_trees)]) \
            if n_trees else np.zeros((n_years, 0))
        idx = w / curves
        return {"index": idx, "chron": chronology_from_indices(idx), "curve": curves}
    src = rcs_group if rcs_group is not None else w
    if method == "rcs":
        curve = rcs_curve(src, cfg.rcs_curve_smooth_period)
        idx = _apply_rcs(w, curve)
        return {"index": idx, "chron": chronology_from_indices(idx), "curve": curve}
    if method == "sfrcs":
        curve = rcs_curve(src, cfg.rcs_curve_smooth_period)
        idx = _apply_rcs(w, curve)
        chron = chronology_from_indices(idx)
        for _ in range(cfg.sf_iterations):
            # signal-free measurements: divide out the common signal, refit curve
            cf = np.where(np.isfinite(chron) & (chron > 0.05), chron, 1.0)
            sf = src / cf[:, None]
            curve = rcs_curve(sf, cfg.rcs_curve_smooth_period)
            idx = _apply_rcs(w, curve)
            chron = chronology_from_indices(idx)
        return {"index": idx, "chron": chron, "curve": curve}
    raise ValueError(method)


def eps_stats(idx: np.ndarray, years: np.ndarray, cfg: Config) -> pd.DataFrame:
    """Running rbar / EPS in windows; returns per-year n, rbar, eps (interpolated)."""
    n_years = idx.shape[0]
    depth = np.isfinite(idx).sum(axis=1)
    centers, rbars = [], []
    for start in range(0, n_years - cfg.eps_window + 1, cfg.eps_step):
        block = idx[start: start + cfg.eps_window]
        ok_cols = np.isfinite(block).sum(axis=0) >= int(0.8 * cfg.eps_window)
        b = block[:, ok_cols]
        if b.shape[1] < 2:
            continue
        c = pd.DataFrame(b).corr(min_periods=int(0.8 * cfg.eps_window)).to_numpy()
        iu = np.triu_indices(b.shape[1], 1)
        vals = c[iu]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        centers.append(start + cfg.eps_window / 2)
        rbars.append(float(np.mean(vals)))
    rbar = np.full(n_years, np.nan)
    if centers:
        rbar = np.interp(np.arange(n_years), centers, rbars, left=rbars[0], right=rbars[-1])
    rb = np.clip(rbar, 1e-3, 1)
    eps = depth * rb / (1 + (depth - 1) * rb)
    eps[depth < 2] = np.nan
    usable = (eps >= cfg.eps_threshold) & (depth >= cfg.min_depth)
    usable_depth = depth >= cfg.depth_for_detection
    return pd.DataFrame({"year": years, "n": depth, "rbar": rbar, "eps": eps,
                         "usable": usable, "usable_depth": usable_depth})


# ---------------------------------------------------------------------------
# Injection-recovery test
# ---------------------------------------------------------------------------
def choose_injection_years(years: np.ndarray, depth: np.ndarray, eq_near: np.ndarray,
                           cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Random years with depth >= min_depth, away from real M>=7 events and each other."""
    cand = years[(depth >= cfg.min_depth)]
    cand = cand[(cand >= years.min() + 30) & (cand <= years.max() - 20)]
    bad = set()
    for y in eq_near:
        bad.update(range(int(y) - cfg.exclusion_years, int(y) + cfg.exclusion_years + 1))
    cand = np.array([c for c in cand if c not in bad])
    rng.shuffle(cand)
    chosen = []
    for c in cand:
        if all(abs(c - x) > cfg.exclusion_years + PROFILE_LEN for x in chosen):
            chosen.append(c)
        if len(chosen) >= cfg.n_inject:
            break
    return np.array(sorted(chosen))


def inject(wide: np.ndarray, years: np.ndarray, t0: int, A: float, cfg: Config,
           rng: np.random.Generator):
    i0 = int(np.searchsorted(years, t0))
    alive = np.isfinite(wide[i0])
    trees = np.where(alive)[0]
    k = max(1, int(round(cfg.responding_fraction * len(trees))))
    hit = rng.choice(trees, size=k, replace=False)
    w = wide.copy()
    for d, g in enumerate(cfg.profile):
        if i0 + d < len(years):
            w[i0 + d, hit] *= (1.0 - A * g)
    return w, hit


def matched_amplitude(chron_clean: np.ndarray, chron_inj: np.ndarray, years: np.ndarray,
                      t0: int, cfg: Config) -> float:
    """Profile-matched amplitude of the paired chronology difference in the injection window."""
    i0 = int(np.searchsorted(years, t0))
    g = np.array(cfg.profile)
    d_win = (1.0 - chron_inj / chron_clean)[i0: i0 + PROFILE_LEN]
    ok = np.isfinite(d_win)
    if ok.sum() < 3:
        return np.nan
    return float(np.sum(d_win[ok] * g[ok]) / np.sum(g[ok] ** 2))


def recover(chron_clean: np.ndarray, chron_inj: np.ndarray, years: np.ndarray, t0: int,
            A: float, cfg: Config, A_ref: float = np.nan) -> dict:
    """
    Match the paired chronology difference against the injected profile.
    A_hat is the profile-matched amplitude on the biweight chronology (used for
    SNR and leakage). Preservation itself is computed by the caller on the
    arithmetic-mean chronology, whose expected response is exactly A*f*profile,
    so that 1.0 means detrending removed nothing from the injected signal.
    """
    i0 = int(np.searchsorted(years, t0))
    g = np.array(cfg.profile)
    win = slice(i0, i0 + PROFILE_LEN)
    delta = 1.0 - chron_inj / chron_clean          # positive = suppression
    d_win = delta[win]
    ok = np.isfinite(d_win)
    if ok.sum() < 3:
        return {}
    A_hat = float(np.sum(d_win[ok] * g[ok]) / np.sum(g[ok] ** 2))
    expected = A_ref if np.isfinite(A_ref) and A_ref > 0 else A * cfg.responding_fraction
    # leakage: distortion outside the window (before and after)
    lo, hi = cfg.leak_window
    before = delta[max(0, i0 + lo): i0]
    after = delta[i0 + PROFILE_LEN: i0 + hi + 1]
    leak = np.concatenate([before, after])
    leak = leak[np.isfinite(leak)]
    leakage = float(np.mean(np.abs(leak))) if len(leak) else np.nan
    # matched-filter noise of the CLEAN chronology (how detectable would A_hat be?)
    c = chron_clean.copy()
    c = c / np.nanmedian(c)
    dev = 1.0 - c
    mf = np.array([np.nansum(dev[i:i + PROFILE_LEN] * g) / np.sum(g ** 2)
                   for i in range(0, len(dev) - PROFILE_LEN)])
    mf = mf[np.isfinite(mf)]
    noise = float(1.4826 * np.median(np.abs(mf - np.median(mf)))) if len(mf) > 20 else np.nan
    return {"A_hat": A_hat, "A_ref": float(A_ref), "expected": expected,
            "preservation": A_hat / expected if expected > 0 else np.nan,
            "leakage": leakage, "noise": noise,
            "snr": A_hat / noise if noise and noise > 0 else np.nan}


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def to_wide(df: pd.DataFrame, value_col: str):
    piv = df.pivot_table(index="year", columns="tree_id", values=value_col, aggfunc="mean")
    years = piv.index.to_numpy().astype(int)
    full = np.arange(years.min(), years.max() + 1)
    piv = piv.reindex(full)
    return full, piv.to_numpy(dtype=float), list(piv.columns)


def species_group(sites: pd.DataFrame) -> dict:
    """Map site_id -> group key for regional (RCS) curves: same species code."""
    if "species_code" in sites.columns and sites["species_code"].notna().any():
        return dict(zip(sites["site_id"], sites["species_code"].fillna(sites["site_id"])))
    if "species" in sites.columns and sites["species"].notna().any():
        return dict(zip(sites["site_id"], sites["species"].fillna(sites["site_id"])))
    return {s: s for s in sites["site_id"]}


def events_near_site(eq: pd.DataFrame, lat: float, lon: float, cfg: Config) -> np.ndarray:
    if not np.isfinite(lat) or not np.isfinite(lon):
        return eq.loc[eq["mag"] >= cfg.exclusion_mag, "year"].to_numpy()
    d = haversine_km(lat, lon, eq["lat"].to_numpy(), eq["lon"].to_numpy())
    m = (eq["mag"].to_numpy() >= cfg.exclusion_mag) & (d <= cfg.exclusion_km)
    return eq.loc[m, "year"].to_numpy()


# ---------------------------------------------------------------------------
# Cluster merging: same-species collections within merge_km become one extra
# "merged" site (originals are kept). Rationale: for event detection the
# relevant quantity is living-tree depth in a given year; Yakushima A/C/020 are
# one forest sampled three times.
# ---------------------------------------------------------------------------
def merge_clusters(rings: pd.DataFrame, sites: pd.DataFrame, cfg: Config, value_col: str):
    sp_col = "species_code" if "species_code" in sites.columns else ("species" if "species" in sites.columns else None)
    ok = sites.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    used = np.zeros(len(ok), dtype=bool)
    new_rings, new_sites, merged_ids = [], [], {}
    for i in range(len(ok)):
        if used[i]:
            continue
        d = haversine_km(ok.loc[i, "lat"], ok.loc[i, "lon"], ok["lat"].to_numpy(), ok["lon"].to_numpy())
        same_sp = (ok[sp_col] == ok.loc[i, sp_col]).to_numpy() if sp_col else np.ones(len(ok), bool)
        members = np.where((d <= cfg.merge_km) & same_sp & (~used))[0]
        used[members] = True
        if len(members) < 2:
            continue
        ids = ok.loc[members, "site_id"].tolist()
        base = str(ok.loc[i, "site_name"]).split(" ")[0] if "site_name" in ok.columns else ids[0]
        mid = f"{base.upper()[:4]}-M"
        sub = rings[rings["site_id"].isin(ids)].copy()
        sub["tree_id"] = sub["site_id"] + ":" + sub["tree_id"].astype(str)
        sub["site_id"] = mid
        new_rings.append(sub)
        row = ok.loc[i].copy()
        row["site_id"] = mid
        row["site_name"] = f"{base} (merged {'+'.join(ids)})"
        row["lat"], row["lon"] = ok.loc[members, "lat"].mean(), ok.loc[members, "lon"].mean()
        new_sites.append(row)
        merged_ids[mid] = ids
        print(f"  [merge] {mid} <- {ids}  ({sub['tree_id'].nunique()} cores)")
    if not new_rings:
        return rings, sites, merged_ids
    return (pd.concat([rings] + new_rings, ignore_index=True),
            pd.concat([sites, pd.DataFrame(new_sites)], ignore_index=True), merged_ids)


# ---------------------------------------------------------------------------
# Main processing per proxy
# ---------------------------------------------------------------------------
def process_proxy(rings: pd.DataFrame, value_col: str, proxy: str, sites: pd.DataFrame,
                  eq: pd.DataFrame, cfg: Config, rng: np.random.Generator, do_injection: bool):
    groups = species_group(sites)
    site_ids = [s for s in sites["site_id"] if s in set(rings["site_id"])]
    wides = {s: to_wide(rings[rings["site_id"] == s], value_col) for s in site_ids}

    # regional matrices per species group (aligned on a common year axis)
    group_mats = {}
    for g in set(groups[s] for s in site_ids):
        members = [s for s in site_ids if groups[s] == g]
        y0 = min(wides[s][0].min() for s in members)
        y1 = max(wides[s][0].max() for s in members)
        full = np.arange(y0, y1 + 1)
        blocks = []
        for s in members:
            yrs, W, _ = wides[s]
            B = np.full((len(full), W.shape[1]), np.nan)
            B[np.searchsorted(full, yrs), :] = W
            blocks.append(B)
        group_mats[g] = (full, np.column_stack(blocks))

    chron_rows, pres_rows, clim_rows, example = [], [], [], None
    for s in site_ids:
        yrs, W, tree_ids = wides[s]
        srow = sites.loc[sites["site_id"] == s].iloc[0]
        lat, lon = float(srow.get("lat", np.nan)), float(srow.get("lon", np.nan))
        eq_near = events_near_site(eq, lat, lon, cfg)
        full_g, G = group_mats[groups[s]]
        sel = slice(np.searchsorted(full_g, yrs.min()), np.searchsorted(full_g, yrs.max()) + 1)
        G_site = G[sel]
        print(f"  [{proxy}] {s}: {W.shape[1]} cores, {yrs.min()}-{yrs.max()}")

        # raw (un-detrended) reference for climate metrics
        raw_norm = W / np.nanmean(W, axis=0, keepdims=True)
        raw_chron = chronology_from_indices(raw_norm)

        for method in cfg.methods:
            res = build_site(W, method, cfg, rcs_group=G_site if method in ("rcs", "sfrcs") else None)
            idx, chron = res["index"], res["chron"]
            chron_mean = chronology_from_indices(idx, robust=False)
            stats = eps_stats(idx, yrs, cfg)
            for i, y in enumerate(yrs):
                chron_rows.append((s, proxy, method, int(y), float(chron[i]), int(stats["n"][i]),
                                   float(stats["rbar"][i]), float(stats["eps"][i]), bool(stats["usable"][i]),
                                   bool(stats["usable_depth"][i])))
            # climate / trend removal metrics
            ok = np.isfinite(chron) & np.isfinite(raw_chron)
            if ok.sum() > 30:
                v_raw = np.nanvar(raw_chron[ok]); v_idx = np.nanvar(chron[ok])
                ac1 = float(pd.Series(chron[ok]).autocorr(1))
                ac1_raw = float(pd.Series(raw_chron[ok]).autocorr(1))
                clim_rows.append({"site_id": s, "proxy": proxy, "method": method,
                                  "variance_removed": float(1 - v_idx / v_raw) if v_raw > 0 else np.nan,
                                  "lag1_ac_index": ac1, "lag1_ac_raw": ac1_raw,
                                  "mean_rbar": float(np.nanmean(stats["rbar"])),
                                  "usable_years": int(stats["usable"].sum()),
                                  "usable_first_year": int(yrs[stats["usable"].to_numpy()].min())
                                  if stats["usable"].any() else -1,
                                  "usable_depth_years": int(stats["usable_depth"].sum()),
                                  "usable_depth_first_year": int(yrs[stats["usable_depth"].to_numpy()].min())
                                  if stats["usable_depth"].any() else -1})

            if not do_injection:
                continue
            depth = stats["n"].to_numpy()
            inj_years = choose_injection_years(yrs, depth, eq_near, cfg, rng)
            for t0 in inj_years:
                for A in cfg.amplitudes:
                    W_inj, hit = inject(W, yrs, int(t0), A, cfg, rng)
                    if method in ("rcs", "sfrcs"):
                        # regional curve refitted with the injected trees as well
                        G_inj = G_site.copy()
                        # locate this site's columns inside the group matrix
                        offset = sum(wides[m][1].shape[1] for m in site_ids
                                     if groups[m] == groups[s] and site_ids.index(m) < site_ids.index(s))
                        G_inj[:, offset: offset + W.shape[1]] = W_inj
                        res_i = build_site(W_inj, method, cfg, rcs_group=G_inj)
                        res_i["chron_mean"] = chronology_from_indices(res_i["index"], robust=False)
                    else:
                        # only the hit cores change -> refit those only (exact for per-series methods)
                        idx_i = idx.copy()
                        for j in hit:
                            curve_j = detrend_series(W_inj[:, j], method, cfg)
                            idx_i[:, j] = W_inj[:, j] / curve_j
                        res_i = {"chron": chronology_from_indices(idx_i),
                                 "chron_mean": chronology_from_indices(idx_i, robust=False)}
                    # preservation is measured on arithmetic-mean chronologies (expected = A*f*profile);
                    # SNR/leakage on the biweight product chronology
                    A_ref = A * cfg.responding_fraction
                    A_mean = matched_amplitude(chron_mean, res_i["chron_mean"], yrs, int(t0), cfg)
                    r = recover(chron, res_i["chron"], yrs, int(t0), A, cfg, A_ref=A_ref)
                    if r:
                        r["A_hat_mean"] = A_mean
                        r["preservation"] = A_mean / A_ref if A_ref > 0 else np.nan
                    if not r:
                        continue
                    r.update({"site_id": s, "proxy": proxy, "method": method, "t0": int(t0),
                              "A": A, "n_hit": int(len(hit)), "depth": int(depth[np.searchsorted(yrs, t0)])})
                    pres_rows.append(r)
                    if example is None and abs(A - cfg.selection_amplitude) < 1e-9 and method == "spline67":
                        example = {"site_id": s, "years": yrs, "clean": chron, "inj": res_i["chron"],
                                   "t0": int(t0), "A": A, "method": method}
    return chron_rows, pres_rows, clim_rows, example


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def select_methods(pres: pd.DataFrame, cfg: Config) -> dict:
    out = {}
    for proxy, sub in pres.groupby("proxy"):
        sub = sub[np.isclose(sub["A"], cfg.selection_amplitude)]
        summ = (sub.groupby("method")
                .agg(preservation=("preservation", "median"), leakage=("leakage", "median"),
                     snr=("snr", "median"), n_trials=("preservation", "size"))
                .reset_index())
        summ["passes"] = summ["preservation"] >= cfg.preservation_threshold
        cand = summ[summ["passes"]]
        if len(cand):
            best = cand.sort_values("snr", ascending=False).iloc[0]["method"]
            rule = (f"reject if median preservation(A={cfg.selection_amplitude}) < "
                    f"{cfg.preservation_threshold}; pick highest median SNR among survivors")
        else:
            best = summ.sort_values("preservation", ascending=False).iloc[0]["method"]
            rule = "NO method passed the preservation threshold; least-bad chosen — flag for review"
        out[proxy] = {"selected": best, "rule": rule,
                      "table": summ.round(4).to_dict(orient="records")}
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_chronologies(chron: pd.DataFrame, sites: pd.DataFrame, eq: pd.DataFrame, sel: dict,
                     cfg: Config, out: Path):
    method = sel["ring_width"]["selected"]
    sub = chron[(chron["proxy"] == "ring_width") & (chron["method"] == method)]
    order = (sub.groupby("site_id")["year"].min().sort_values().index.tolist())
    names = dict(zip(sites["site_id"], sites.get("site_name", sites["site_id"]).fillna(sites["site_id"])))
    n = len(order)
    fig, axes = plt.subplots(n, 1, figsize=(12, 1.55 * n + 1.2), sharex=True, facecolor="white")
    axes = np.atleast_1d(axes)
    for ax, s in zip(axes, order):
        d = sub[sub["site_id"] == s].sort_values("year")
        ax.set_facecolor("white")
        # usable-year shading
        us = d["usable"].to_numpy(); ud = d["usable_depth"].to_numpy(); yrs = d["year"].to_numpy()
        if ud.any():
            ax.fill_between(yrs, 0, 1, where=ud, transform=ax.get_xaxis_transform(),
                            color="#fff3e0", zorder=0)
        if us.any():
            ax.fill_between(yrs, 0, 1, where=us, transform=ax.get_xaxis_transform(),
                            color="#e8f5e9", zorder=0)
        ax.plot(yrs, d["index"], color="#1b5e20", linewidth=0.7, zorder=3)
        ax.axhline(1.0, color="#999", linewidth=0.5, zorder=2)
        ax2 = ax.twinx()
        ax2.fill_between(yrs, 0, d["n"], color="#bbbbbb", alpha=0.35, linewidth=0, zorder=1)
        ax2.set_ylim(0, max(d["n"].max() * 3, 10)); ax2.set_yticks([])
        srow = sites[sites["site_id"] == s].iloc[0]
        lat, lon = float(srow.get("lat", np.nan)), float(srow.get("lon", np.nan))
        for y in events_near_site(eq, lat, lon, cfg):
            ax.axvline(y, color="#B5651D", linewidth=0.7, alpha=0.7, zorder=2)
        ax.set_ylim(max(0, np.nanpercentile(d["index"], 0.5) - 0.1),
                    np.nanpercentile(d["index"], 99.5) + 0.1)
        ax.set_ylabel("index", fontsize=8)
        ax.text(0.005, 0.92, f"{names.get(s, s)} [{s}]", transform=ax.transAxes,
                fontsize=8.5, va="top", ha="left", fontweight="bold")
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("Year (CE)")
    axes[0].set_title(f"DRYAS - Site Chronologies ({method}, ring width) with Sample Depth and Nearby M>={cfg.exclusion_mag:g} Events")
    handles = [Line2D([0], [0], color="#1b5e20", label="Chronology index"),
               Patch(facecolor="#bbbbbb", alpha=0.35, label="Sample depth (right axis, unscaled)"),
               Patch(facecolor="#e8f5e9", label=f"Climate-quality years (EPS>={cfg.eps_threshold}, n>={cfg.min_depth})"),
               Patch(facecolor="#fff3e0", label=f"Detection-usable years (depth>={cfg.depth_for_detection})"),
               Line2D([0], [0], color="#B5651D", label=f"M>={cfg.exclusion_mag:g} within {cfg.exclusion_km:.0f} km")]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_tradeoff(pres: pd.DataFrame, clim: pd.DataFrame, sel: dict, cfg: Config, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), facecolor="white")
    colors = dict(zip(cfg.methods, ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]))
    p = pres[pres["proxy"] == "ring_width"]
    c = clim[clim["proxy"] == "ring_width"]
    # left: preservation vs variance removed
    ax = axes[0]; ax.set_facecolor("white")
    for m in cfg.methods:
        pm = p[(p["method"] == m) & np.isclose(p["A"], cfg.selection_amplitude)]
        cm = c[c["method"] == m]
        if len(pm) == 0 or len(cm) == 0:
            continue
        x, y = cm["variance_removed"].median(), pm["preservation"].median()
        xe = np.abs(np.nanpercentile(cm["variance_removed"], [25, 75]) - x)[:, None]
        ye = np.abs(np.nanpercentile(pm["preservation"], [25, 75]) - y)[:, None]
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", color=colors[m], capsize=3, label=m,
                    markersize=8, markeredgecolor="black" if m == sel["ring_width"]["selected"] else colors[m],
                    markeredgewidth=1.5)
    ax.axhline(cfg.preservation_threshold, color="#999", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Variance removed relative to raw site mean (median over sites)")
    ax.set_ylabel(f"Preservation of injected suppression (A={cfg.selection_amplitude}, median)")
    ax.set_title("Signal preservation vs trend/climate removal")
    ax.grid(True, linestyle=":", linewidth=0.5)
    # right: preservation by amplitude
    ax = axes[1]; ax.set_facecolor("white")
    for m in cfg.methods:
        pm = p[p["method"] == m]
        if len(pm) == 0:
            continue
        g = pm.groupby("A")["preservation"].agg(["median", lambda v: np.nanpercentile(v, 25),
                                                  lambda v: np.nanpercentile(v, 75)])
        g.columns = ["med", "q1", "q3"]
        ax.plot(g.index, g["med"], "-o", color=colors[m], label=m)
        ax.fill_between(g.index, g["q1"], g["q3"], color=colors[m], alpha=0.12, linewidth=0)
    ax.axhline(cfg.preservation_threshold, color="#999", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Injected amplitude A (fraction of growth lost in year 0)")
    ax.set_ylabel("Preservation (median, IQR band)")
    ax.set_title("Preservation across amplitudes")
    ax.grid(True, linestyle=":", linewidth=0.5)
    handles = [Line2D([0], [0], marker="o", color=colors[m], label=m) for m in cfg.methods]
    handles.append(Line2D([0], [0], color="#999", linestyle="--", label=f"threshold {cfg.preservation_threshold}"))
    handles.append(Line2D([0], [0], marker="o", color="w", markeredgecolor="black", markerfacecolor="w",
                          markeredgewidth=1.5, label="selected method (black ring)"))
    axes[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8.5)
    fig.suptitle("DRYAS - Injection-Recovery Test of Detrending Methods", y=1.02)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_eps(chron: pd.DataFrame, sites: pd.DataFrame, sel: dict, cfg: Config, out: Path):
    method = sel["ring_width"]["selected"]
    sub = chron[(chron["proxy"] == "ring_width") & (chron["method"] == method)]
    order = sub.groupby("site_id")["year"].min().sort_values().index.tolist()
    names = dict(zip(sites["site_id"], sites.get("site_name", sites["site_id"]).fillna(sites["site_id"])))
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.42 * len(order) + 2.5), facecolor="white")
    ax = axes[0]; ax.set_facecolor("white")
    for i, s in enumerate(order):
        d = sub[sub["site_id"] == s].sort_values("year")
        ax.plot(d["year"], d["eps"] + 0 * i, color="#1f77b4", linewidth=0.6, alpha=0.7)
    ax.axhline(cfg.eps_threshold, color="#d62728", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Year (CE)"); ax.set_ylabel("EPS (running, all sites overlaid)")
    ax.set_ylim(0, 1.02); ax.grid(True, linestyle=":", linewidth=0.5)
    ax.set_title("Expressed Population Signal")
    ax = axes[1]; ax.set_facecolor("white")
    for i, s in enumerate(order):
        d = sub[sub["site_id"] == s].sort_values("year")
        us = d["usable"].to_numpy(); ud = d["usable_depth"].to_numpy(); yrs = d["year"].to_numpy()
        ax.plot([yrs.min(), yrs.max()], [i, i], color="#cccccc", linewidth=6, solid_capstyle="butt")
        if ud.any():
            ax.scatter(yrs[ud], np.full(ud.sum(), i + 0.18), color="#ff9800", s=10, marker="|", linewidths=1.2)
        if us.any():
            ax.scatter(yrs[us], np.full(us.sum(), i - 0.18), color="#2e7d32", s=10, marker="|", linewidths=1.2)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([names.get(s, s) for s in order], fontsize=8)
    ax.set_xlabel("Year (CE)"); ax.set_title("Chronology span (grey), climate-quality (green) and detection-usable (orange) years")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5)
    handles = [Line2D([0], [0], color="#1f77b4", label="EPS per site"),
               Line2D([0], [0], color="#d62728", linestyle="--", label=f"EPS threshold {cfg.eps_threshold}"),
               Line2D([0], [0], color="#cccccc", linewidth=6, label="Full span"),
               Line2D([0], [0], color="#2e7d32", linewidth=3, label=f"Climate-quality (EPS>={cfg.eps_threshold}, n>={cfg.min_depth})"),
               Line2D([0], [0], color="#ff9800", linewidth=3, label=f"Detection-usable (depth>={cfg.depth_for_detection})")]
    axes[1].legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8.5)
    fig.suptitle(f"DRYAS - Chronology Quality ({method})", y=1.01)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_example(example: dict, cfg: Config, out: Path):
    yrs, clean, inj, t0 = example["years"], example["clean"], example["inj"], example["t0"]
    i0 = int(np.searchsorted(yrs, t0)); lo, hi = i0 - 25, i0 + 25
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, facecolor="white")
    ax = axes[0]; ax.set_facecolor("white")
    ax.plot(yrs[lo:hi], clean[lo:hi], color="#1b5e20", label="Clean chronology (biweight)")
    ax.plot(yrs[lo:hi], inj[lo:hi], color="#d62728", linestyle="--", label="After injection (biweight)")
    ax.axvspan(t0 - 0.5, t0 + PROFILE_LEN - 0.5, color="#ffe0b2", alpha=0.6)
    ax.set_ylabel("Chronology index")
    ax.set_title(f"Injection example: {example['site_id']}, method={example['method']}, "
                 f"A={example['A']}, f={cfg.responding_fraction}, t0={t0}")
    ax = axes[1]; ax.set_facecolor("white")
    delta = 1 - inj / clean
    ax.bar(yrs[lo:hi], delta[lo:hi], color="#555", width=0.8, label="1 - injected/clean")
    exp_prof = np.zeros(hi - lo)
    exp_prof[i0 - lo: i0 - lo + PROFILE_LEN] = example["A"] * cfg.responding_fraction * np.array(cfg.profile)
    ax.plot(yrs[lo:hi], exp_prof, color="#ff7f0e", drawstyle="steps-mid",
            label="Expected for arithmetic mean (A*f*profile);\nbiweight down-weights non-responders -> larger")
    ax.set_ylabel("Suppression"); ax.set_xlabel("Year (CE)")
    for a in axes:
        a.grid(True, linestyle=":", linewidth=0.5)
        a.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ---------------------------------------------------------------------------
# Provenance / IO
# ---------------------------------------------------------------------------
def save_table(df: pd.DataFrame, name: str, records: dict):
    p = DATA_DIR / name
    df.to_parquet(p, index=False)
    records[name] = {"rows": int(len(df)), "sha256": _sha256_of_file(p), "path": str(p)}
    print(f"  wrote {p}  ({len(df):,} rows)")


def append_provenance(records: dict, cfg: Config, extra: dict):
    p = DATA_DIR / "provenance.json"
    prov = json.load(open(p)) if p.exists() else {}
    prov["002"] = {"script": "002_chronology_standardization.py",
                   "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "config": asdict(cfg), "artifacts": records, **extra}
    json.dump(prov, open(p, "w"), indent=2, default=str)
    print(f"  updated {p}")


def main():
    ap = argparse.ArgumentParser(description="DRYAS 002 — chronology & standardization")
    ap.add_argument("--n-inject", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="fewer injections/amplitudes for a smoke test")
    ap.add_argument("--no-density", action="store_true")
    args = ap.parse_args()
    cfg = Config()
    if args.n_inject is not None:
        cfg.n_inject = args.n_inject
    if args.seed is not None:
        cfg.seed = args.seed
    if args.quick:
        cfg.quick = True; cfg.n_inject = min(cfg.n_inject, 6); cfg.amplitudes = (0.35,)
    rng = np.random.default_rng(cfg.seed)

    rings = pd.read_parquet(DATA_DIR / "treering_network.parquet")
    sites = pd.read_parquet(DATA_DIR / "site_metadata.parquet")
    eq = pd.read_parquet(DATA_DIR / "eq_catalog.parquet")
    dens_p = DATA_DIR / "treering_density.parquet"
    density = pd.read_parquet(dens_p) if (dens_p.exists() and not args.no_density) else None
    print(f"DRYAS 002 — {rings['site_id'].nunique()} sites, {len(rings):,} rings, "
          f"density={'yes' if density is not None else 'no'}, methods={cfg.methods}")

    rings, sites, merged_ids = merge_clusters(rings, sites, cfg, "ring_width_mm")
    print("[ring width] building chronologies + injection test ...")
    ch, pr, cl, example = process_proxy(rings, "ring_width_mm", "ring_width", sites, eq, cfg, rng, True)
    chron_rows, pres_rows, clim_rows = list(ch), list(pr), list(cl)
    if density is not None and len(density):
        print("[max density] building chronologies (no injection: MXD earthquake response unknown) ...")
        d = density.copy()
        d["site_id"] = d["site_id"].str.replace(r"x$", "", regex=True)   # japa008x -> japa008
        d, _, _ = merge_clusters(d, sites[~sites["site_id"].isin(merged_ids)], cfg, "density")
        ch, pr, cl, _ = process_proxy(d, "density", "max_density", sites, eq, cfg, rng, False)
        chron_rows += ch; clim_rows += cl

    chron = pd.DataFrame(chron_rows, columns=["site_id", "proxy", "method", "year", "index",
                                              "n", "rbar", "eps", "usable", "usable_depth"])
    pres = pd.DataFrame(pres_rows)
    clim = pd.DataFrame(clim_rows)
    sel = select_methods(pres, cfg)
    sel_dens = None
    if "max_density" in set(chron["proxy"]):
        # no injection evidence for MXD -> inherit the ring-width choice, documented
        sel["max_density"] = {"selected": sel["ring_width"]["selected"],
                              "rule": "inherited from ring_width (no injection test for MXD)", "table": []}

    records = {}
    save_table(chron, "chronologies.parquet", records)
    save_table(pres, "002_preservation.parquet", records)
    save_table(clim, "002_climate_metrics.parquet", records)
    with open(DATA_DIR / "002_selection.json", "w") as f:
        json.dump(sel, f, indent=2)
    with open(DATA_DIR / "002_merged_sites.json", "w") as f:
        json.dump(merged_ids, f, indent=2)
    print(f"  wrote {DATA_DIR / '002_selection.json'}")

    print("rendering figures ...")
    fig_chronologies(chron, sites, eq, sel, cfg, PDF_DIR / "002a_chronologies_selected.pdf")
    fig_tradeoff(pres, clim, sel, cfg, PDF_DIR / "002b_preservation_tradeoff.pdf")
    fig_eps(chron, sites, sel, cfg, PDF_DIR / "002c_eps_sample_depth.pdf")
    if example is not None:
        fig_example(example, cfg, PDF_DIR / "002d_injection_example.pdf")
    for f in ("002a_chronologies_selected", "002b_preservation_tradeoff", "002c_eps_sample_depth", "002d_injection_example"):
        print(f"  wrote {PDF_DIR / (f + '.pdf')}")

    summary = {"selection": {k: v["selected"] for k, v in sel.items()},
               "n_injection_trials": int(len(pres)), "merged_sites": merged_ids}
    append_provenance(records, cfg, summary)

    print("\n=== SUMMARY ===")
    for k, v in sel.items():
        print(f"  {k}: selected = {v['selected']}   ({v['rule']})")
    tab = pd.DataFrame(sel["ring_width"]["table"])
    if len(tab):
        print(tab.to_string(index=False))
    us = clim[(clim["proxy"] == "ring_width") & (clim["method"] == sel["ring_width"]["selected"])]
    print("  usable years per site (selected method):  EPS gate | depth gate (n>=%d)" % cfg.depth_for_detection)
    for _, r in us.sort_values("usable_depth_first_year").head(20).iterrows():
        print(f"    {r['site_id']:>8}: EPS from {r['usable_first_year']:>5} ({r['usable_years']:>4} yrs) | "
              f"depth from {r['usable_depth_first_year']:>5} ({r['usable_depth_years']:>4} yrs)")
    if len(us) > 20:
        print(f"    ... ({len(us) - 20} more sites)")
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
