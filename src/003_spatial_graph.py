#!/usr/bin/env python3
# =============================================================================
# DRYAS  |  Script 003  —  SPATIAL GRAPH, FAULT GEOMETRY & GROUND-MOTION PRIOR
# =============================================================================
# WHAT THIS SCRIPT DOES
# ---------------------------------------------------------------------------
# DRYAS treats the tree-ring sites as a passive sensor array. To separate an
# earthquake signal from climate/fire/insect signals (005) and to test cascade
# structure (008/010) we need the geometry that ties trees to faults:
#
#   1. FAULT SEGMENTS. Approximate rectangular source areas for the seismogenic
#      segments that the array can see: Nankai Trough (Hyuga-nada plus Ando-1975
#      segments A-E), Kuril Trench (Tokachi-oki, Nemuro-oki, Shikotan-oki,
#      Etorofu-oki), Sanriku/Japan Trench (for completeness of the catalog
#      assignment), and the two inland zones that host tree sites (ISTL near
#      Senjo, MTL near Omogo). Corner coordinates are textbook-level
#      approximations (see NOTE) and are written to GeoJSON so they can be
#      inspected and, if necessary, replaced by an official model.
#
#   2. EXPECTED GROUND MOTION per (site, event). Peak ground acceleration from
#      the Si & Midorikawa (1999) attenuation relation, using the closest
#      distance from the site to the event's source (rectangle for segment-
#      assigned events, hypocentre for the rest). This is the physical prior
#      for 005: an earthquake forcing should scale with PGA and be synchronous
#      across sites facing the same segment, whereas climate is smooth over
#      hundreds of km and fire/insects are patchy. PGA is also converted to a
#      first-order "expected tree response" probability with a logistic in
#      log-PGA anchored at the literature thresholds (Mw ~5.3 local, ~7.4
#      regional; Hough 2026 argues thresholds are high) — the anchors are
#      CONFIG values, to be re-fitted in 006 against real events.
#
#   3. CATALOG-TO-SEGMENT ASSIGNMENT. Each catalog event (M >= 6.5) is assigned
#      to the segment whose rectangle contains its epicentre (or the nearest
#      within 60 km); great events (M >= 8.0) are allowed to span adjacent
#      segments of the same trench (multi-segment rupture flag). The result is
#      the discrete event list used by the cascade analysis and the forecast
#      test.
#
#   4. SPATIAL GRAPH. Site-level nodes (with the merged Yakushima site from 002)
#      and two edge families: (a) site-site edges weighted by great-circle
#      distance, (b) site-segment edges with fault distance and per-segment
#      expected PGA for a reference magnitude. Written as parquet for 005.
#
# INPUTS (001, 002)
#   data/site_metadata.parquet, data/eq_catalog.parquet,
#   data/chronologies.parquet (only to inherit the merged-site definition and
#   detection-usable spans), data/002_selection.json
#
# OUTPUTS
#   data/fault_segments.geojson         segment polygons + attributes
#   data/graph_nodes.parquet            site nodes (id, name, lat, lon, cluster,
#                                       usable_depth_first/last, n_series)
#   data/graph_edges_site_site.parquet  distances between sites
#   data/graph_edges_site_segment.parquet  fault distance + reference PGA
#   data/eq_segment_assignment.parquet  events with segment, distance, PGA per site
#   data/site_event_pga.parquet         long table (site_id, event_id, dist_km,
#                                       pga_gal, p_response)
#   PDF/003a_array_and_segments.pdf     PyGMT map: sites, segments, assigned M>=7
#   PDF/003b_expected_response_matrix.pdf  site x event expected response
#   PDF/003c_attenuation_curves.pdf     the prior itself (PGA & p_response vs distance)
#
# NOTE ON GEOMETRY
#   Segment rectangles are simplified (4 corners, lon/lat) and intended for
#   distance calculations at the ~50 km fidelity that annual tree-ring data can
#   resolve. They are NOT a substitute for HERP/JSHIS source models; replacing
#   them is a one-file change (edit SEGMENTS below or supply --segments file).
#
# USAGE
#   uv run python src/003_spatial_graph.py
#   uv run python src/003_spatial_graph.py --segments my_segments.json
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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
    assign_mmin: float = 6.5          # catalog events considered for segment assignment
    assign_max_km: float = 60.0       # epicentre may lie this far outside a rectangle
    multiseg_mmin: float = 8.0        # events at/above may span adjacent segments
    reference_mag: float = 8.0        # for site-segment reference PGA
    depth_default_km: float = 20.0    # if the catalog has no depth (historical)
    intraslab_depth_km: float = 60.0  # deeper events use the intraplate coefficient (d = 0.22)
    # Growing-season timing: rings of year Y are essentially complete by the end
    # of the growing season; an earthquake in month >= season_end_month can only
    # affect the ring of Y+1. eff_year = year + (month >= season_end_month).
    season_end_month: int = 8
    # tree-response prior: logistic in log10(PGA[gal]); p = 1/(1+exp(-k (log PGA - x0)))
    # x0 anchored so that p = 0.5 at ~100 gal (JMA intensity ~5-lower), k chosen so
    # that p ~ 0.1 at 40 gal and ~0.9 at 250 gal. Re-fitted in 006.
    resp_x0_log10gal: float = 2.0
    resp_k: float = 5.5
    map_region: tuple = (127.0, 149.0, 29.0, 46.5)


# ---------------------------------------------------------------------------
# Segment geometry (approximate rectangles; lon/lat corners, counter-clockwise)
# trench: which fault system; mechanism: "interplate" | "crustal"
# ---------------------------------------------------------------------------
SEGMENTS = [
    # --- Nankai Trough (Ando 1975 A-E) + Hyuga-nada ------------------------
    {"id": "HYUGA",  "name": "Hyuga-nada",            "trench": "Nankai", "mech": "interplate",
     "corners": [(131.2, 31.3), (132.6, 31.3), (132.6, 33.0), (131.2, 33.0)]},
    {"id": "NK-A",   "name": "Nankai A (Tosa Bay)",   "trench": "Nankai", "mech": "interplate",
     "corners": [(132.6, 31.8), (134.4, 32.4), (134.0, 33.7), (132.5, 33.1)]},
    {"id": "NK-B",   "name": "Nankai B (Kii Channel)","trench": "Nankai", "mech": "interplate",
     "corners": [(134.4, 32.4), (135.8, 32.6), (135.6, 34.0), (134.0, 33.7)]},
    {"id": "NK-C",   "name": "Tonankai C (Kumano)",   "trench": "Nankai", "mech": "interplate",
     "corners": [(135.8, 32.6), (137.2, 33.0), (137.0, 34.4), (135.6, 34.0)]},
    {"id": "NK-D",   "name": "Tonankai D (Enshu)",    "trench": "Nankai", "mech": "interplate",
     "corners": [(137.2, 33.0), (138.4, 33.4), (138.2, 34.7), (137.0, 34.4)]},
    {"id": "NK-E",   "name": "Tokai E (Suruga Bay)",  "trench": "Nankai", "mech": "interplate",
     "corners": [(138.2, 34.0), (138.9, 34.0), (138.9, 35.1), (138.2, 35.1)]},
    # --- Sagami Trough (Kanto) --------------------------------------------
    {"id": "SAGAMI", "name": "Sagami Trough",         "trench": "Sagami", "mech": "interplate",
     "corners": [(139.0, 34.6), (140.6, 34.6), (140.6, 35.6), (139.0, 35.6)]},
    # --- Japan Trench (Tohoku, for catalog completeness) --------------------
    {"id": "JT-S",   "name": "Japan Trench south (Fukushima/Ibaraki)", "trench": "Japan", "mech": "interplate",
     "corners": [(140.5, 35.6), (142.5, 35.6), (142.5, 37.5), (140.5, 37.5)]},
    {"id": "JT-M",   "name": "Japan Trench middle (Miyagi)",           "trench": "Japan", "mech": "interplate",
     "corners": [(140.8, 37.5), (143.5, 37.5), (143.5, 39.0), (140.8, 39.0)]},
    {"id": "JT-N",   "name": "Japan Trench north (Sanriku)",           "trench": "Japan", "mech": "interplate",
     "corners": [(141.5, 39.0), (145.0, 39.0), (145.0, 41.3), (141.5, 41.3)]},
    # --- Kuril Trench (Hokkaido) -----------------------------------------
    {"id": "KT-TOK", "name": "Tokachi-oki",           "trench": "Kuril", "mech": "interplate",
     "corners": [(142.5, 41.3), (144.8, 41.3), (144.8, 42.8), (142.5, 42.8)]},
    {"id": "KT-NEM", "name": "Nemuro-oki",            "trench": "Kuril", "mech": "interplate",
     "corners": [(144.8, 41.8), (146.5, 41.8), (146.5, 43.5), (144.8, 43.5)]},
    {"id": "KT-SHI", "name": "Shikotan-oki",          "trench": "Kuril", "mech": "interplate",
     "corners": [(146.5, 42.5), (148.0, 42.5), (148.0, 44.2), (146.5, 44.2)]},
    {"id": "KT-ETO", "name": "Etorofu-oki",           "trench": "Kuril", "mech": "interplate",
     "corners": [(148.0, 43.5), (150.0, 43.5), (150.0, 45.5), (148.0, 45.5)]},
    # --- Hidaka / Hokkaido interior (Urakawa-oki, Ishikari) -----------------
    {"id": "HIDAKA", "name": "Hidaka collision zone", "trench": "Hokkaido", "mech": "crustal",
     "corners": [(142.0, 41.8), (143.4, 41.8), (143.4, 43.2), (142.0, 43.2)]},
    # --- Inland zones hosting tree sites ----------------------------------
    {"id": "ISTL",   "name": "Itoigawa-Shizuoka Tectonic Line", "trench": "Inland", "mech": "crustal",
     "corners": [(137.6, 35.2), (138.4, 35.2), (138.4, 37.0), (137.6, 37.0)]},
    {"id": "MTL-SHK","name": "Median Tectonic Line (Shikoku)",  "trench": "Inland", "mech": "crustal",
     "corners": [(132.7, 33.6), (134.6, 33.6), (134.6, 34.3), (132.7, 34.3)]},
    {"id": "NOBI",   "name": "Nobi / Neodani",        "trench": "Inland", "mech": "crustal",
     "corners": [(136.0, 35.2), (137.0, 35.2), (137.0, 36.0), (136.0, 36.0)]},
    {"id": "NIIGATA","name": "Niigata-Shinano (Echigo/Zenkoji)", "trench": "Inland", "mech": "crustal",
     "corners": [(138.0, 36.5), (139.6, 36.5), (139.6, 37.9), (138.0, 37.9)]},
    {"id": "NOTO",   "name": "Noto Peninsula",        "trench": "Inland", "mech": "crustal",
     "corners": [(136.5, 36.9), (137.8, 36.9), (137.8, 37.8), (136.5, 37.8)]},
    {"id": "SANIN",  "name": "San-in coast (Tango/Tottori)", "trench": "Inland", "mech": "crustal",
     "corners": [(133.0, 35.1), (135.5, 35.1), (135.5, 35.9), (133.0, 35.9)]},
    {"id": "ROKKO",  "name": "Rokko-Awaji (Kobe)",    "trench": "Inland", "mech": "crustal",
     "corners": [(134.7, 34.2), (135.5, 34.2), (135.5, 34.9), (134.7, 34.9)]},
    {"id": "BEPPU",  "name": "Beppu-Shimabara (Kumamoto)", "trench": "Inland", "mech": "crustal",
     "corners": [(130.2, 32.4), (131.8, 32.4), (131.8, 33.4), (130.2, 33.4)]},
]

TRENCH_ORDER = {"Nankai": ["HYUGA", "NK-A", "NK-B", "NK-C", "NK-D", "NK-E"],
                "Japan": ["JT-S", "JT-M", "JT-N"],
                "Kuril": ["KT-TOK", "KT-NEM", "KT-SHI", "KT-ETO"]}


# ---------------------------------------------------------------------------
# Geometry helpers
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


def point_in_polygon(lon, lat, corners) -> bool:
    x, y = lon, lat
    inside = False
    n = len(corners)
    for i in range(n):
        x1, y1 = corners[i]; x2, y2 = corners[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xint:
                inside = not inside
    return inside


def dist_point_to_polygon_km(lon, lat, corners) -> float:
    """Closest distance (km) from a point to a lon/lat polygon (0 if inside).
    Edges are densified in lon/lat and haversine distance is taken — adequate
    at the ~5 km level for segment-sized rectangles."""
    if point_in_polygon(lon, lat, corners):
        return 0.0
    best = np.inf
    for i in range(len(corners)):
        (x1, y1), (x2, y2) = corners[i], corners[(i + 1) % len(corners)]
        t = np.linspace(0, 1, 60)
        xs, ys = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
        best = min(best, float(np.min(haversine_km(lat, lon, ys, xs))))
    return best


def polygon_centroid(corners):
    xs = np.array([c[0] for c in corners]); ys = np.array([c[1] for c in corners])
    return float(xs.mean()), float(ys.mean())


# ---------------------------------------------------------------------------
# Ground-motion prior
# ---------------------------------------------------------------------------
def pga_si_midorikawa_1999(mw: float, x_km: np.ndarray, depth_km: float, mech: str) -> np.ndarray:
    """
    Si & Midorikawa (1999) PGA on stiff soil (cm/s^2 = gal), fault-distance form:
      log10 A = 0.50 Mw + 0.0043 D + d + 0.61 - log10(X + 0.0055*10^(0.50 Mw)) - 0.003 X
      d = 0.00 crustal, 0.01 interplate, 0.22 intraplate
    X: closest distance to the fault (km), D: focal depth (km).
    """
    d = {"crustal": 0.00, "interplate": 0.01, "intraplate": 0.22}.get(mech, 0.0)
    x = np.maximum(np.asarray(x_km, dtype=float), 0.0)
    log_a = (0.50 * mw + 0.0043 * depth_km + d + 0.61
             - np.log10(x + 0.0055 * 10 ** (0.50 * mw)) - 0.003 * x)
    return 10 ** log_a


def p_response(pga_gal: np.ndarray, cfg: Config) -> np.ndarray:
    z = cfg.resp_k * (np.log10(np.maximum(pga_gal, 1e-3)) - cfg.resp_x0_log10gal)
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Catalog assignment
# ---------------------------------------------------------------------------
def assign_segments(eq: pd.DataFrame, segments: list, cfg: Config) -> pd.DataFrame:
    rows = []
    for k, e in eq.iterrows():
        lon, lat, mag = float(e["lon"]), float(e["lat"]), float(e["mag"])
        if mag < cfg.assign_mmin:
            continue
        dists = {s["id"]: dist_point_to_polygon_km(lon, lat, s["corners"]) for s in segments}
        inside = [s for s in segments if dists[s["id"]] == 0.0]
        if len(inside) > 1:
            # overlap between an interplate rectangle and an inland zone: great events
            # go to the plate boundary, smaller ones to the crustal zone
            crustal = [s for s in inside if s["mech"] == "crustal"]
            sid = (crustal[0]["id"] if crustal and mag < cfg.multiseg_mmin else
                   next(s["id"] for s in inside if s["mech"] != "crustal") if any(s["mech"] != "crustal" for s in inside)
                   else inside[0]["id"])
        else:
            sid = min(dists, key=dists.get)
        d0 = dists[sid]
        seg = next(s for s in segments if s["id"] == sid)
        if d0 > cfg.assign_max_km:
            sid, seg, trench, mech = "UNASSIGNED", None, "none", "crustal"
        else:
            trench, mech = seg["trench"], seg["mech"]
        depth = float(e["depth"]) if ("depth" in e and pd.notna(e.get("depth"))) else cfg.depth_default_km
        if depth >= cfg.intraslab_depth_km:
            mech = "intraplate"           # slab event (e.g. 1993 Kushiro-oki, ~100 km): d = 0.22
        multi = []
        if seg is not None and mag >= cfg.multiseg_mmin and trench in TRENCH_ORDER:
            order = TRENCH_ORDER[trench]
            i = order.index(sid)
            # a great event is allowed to span its immediate neighbours within 150 km
            for j in (i - 1, i + 1):
                if 0 <= j < len(order):
                    nb = next(s for s in segments if s["id"] == order[j])
                    if dist_point_to_polygon_km(lon, lat, nb["corners"]) <= 150.0:
                        multi.append(order[j])
        month = int(e["month"]) if ("month" in e and pd.notna(e.get("month"))) else 6
        eff_year = int(e["year"]) + (1 if month >= cfg.season_end_month else 0)
        rows.append({"event_id": f"E{int(e['year']):04d}_{k}", "year": int(e["year"]), "month": month,
                     "eff_year": eff_year,
                     "lat": lat, "lon": lon, "mag": mag, "name": str(e.get("name", "")),
                     "era": str(e.get("era", "")), "segment": sid, "trench": trench, "mech": mech,
                     "dist_to_segment_km": float(d0) if seg is not None else np.nan,
                     "multi_segment": "+".join([sid] + multi) if multi else sid,
                     "depth_km": depth})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Nodes from 002 (merged sites + detection-usable spans)
# ---------------------------------------------------------------------------
def build_nodes(sites: pd.DataFrame, chron: pd.DataFrame | None, sel: dict | None) -> pd.DataFrame:
    nodes = sites.dropna(subset=["lat", "lon"]).copy()
    if "site_name" not in nodes.columns:
        nodes["site_name"] = nodes["site_id"]
    nodes = nodes[["site_id", "site_name", "lat", "lon"] +
                  [c for c in ("species", "species_code", "n_series") if c in nodes.columns]]
    if chron is not None and sel is not None:
        method = sel["ring_width"]["selected"]
        c = chron[(chron["proxy"] == "ring_width") & (chron["method"] == method)]
        # merged sites exist only in chronologies -> add them as nodes
        extra = []
        for s in set(c["site_id"]) - set(nodes["site_id"]):
            extra.append({"site_id": s, "site_name": s, "lat": np.nan, "lon": np.nan})
        if extra:
            nodes = pd.concat([nodes, pd.DataFrame(extra)], ignore_index=True)
        g = c.groupby("site_id")
        span = pd.DataFrame({
            "usable_depth_first": g.apply(lambda d: int(d.loc[d["usable_depth"], "year"].min()) if d["usable_depth"].any() else -1),
            "usable_depth_last": g.apply(lambda d: int(d.loc[d["usable_depth"], "year"].max()) if d["usable_depth"].any() else -1),
            "usable_depth_years": g["usable_depth"].sum().astype(int),
            "eps_first": g.apply(lambda d: int(d.loc[d["usable"], "year"].min()) if d["usable"].any() else -1),
        }).reset_index()
        nodes = nodes.merge(span, on="site_id", how="left")
    return nodes


def fill_merged_coords(nodes: pd.DataFrame, prov: dict) -> pd.DataFrame:
    """Merged sites (from 002) get the mean coordinates of their members."""
    merged = {}
    mp = DATA_DIR / "002_merged_sites.json"
    if mp.exists():
        merged = json.load(open(mp))
    if not merged and prov:
        merged = prov.get("002", {}).get("merged_sites", {})
    for mid, members in merged.items():
        m = nodes["site_id"].isin(members)
        if m.any() and mid in set(nodes["site_id"]):
            nodes.loc[nodes["site_id"] == mid, ["lat", "lon"]] = [nodes.loc[m, "lat"].mean(), nodes.loc[m, "lon"].mean()]
            nodes.loc[nodes["site_id"] == mid, "site_name"] = f"Yakushima (merged)" if mid.startswith("YAKU") else f"{mid} (merged)"
            nodes.loc[nodes["site_id"] == mid, "n_series"] = nodes.loc[m, "n_series"].sum() if "n_series" in nodes else np.nan
    nodes["cluster"] = np.where(nodes["lat"] > 41.5, "Hokkaido",
                        np.where(nodes["lat"] < 31.5, "Yakushima",
                        np.where(nodes["lon"] < 134, "Setouchi/Shikoku", "Honshu")))
    return nodes.dropna(subset=["lat", "lon"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_map(nodes, segments, ev, cfg: Config, out: Path):
    try:
        import os, glob
        if not os.environ.get("GMT_LIBRARY_PATH"):
            for d in ["/opt/homebrew/lib", "/usr/local/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib/aarch64-linux-gnu",
                      "C:/programs/gmt6/bin", "C:/Program Files/GMT6/bin"] + glob.glob("/opt/homebrew/Cellar/gmt/*/lib"):
                if glob.glob(os.path.join(d, "libgmt*")) or glob.glob(os.path.join(d, "gmt*.dll")):
                    os.environ["GMT_LIBRARY_PATH"] = d; break
        import pygmt
        _fig_map_gmt(nodes, segments, ev, cfg, out, pygmt)
        print("  [figure] 003a rendered with PyGMT")
    except Exception as e:
        print(f"  [figure] PyGMT unavailable ({type(e).__name__}); matplotlib fallback")
        _fig_map_mpl(nodes, segments, ev, cfg, out)


TRENCH_COLORS = {"Nankai": "#d62728", "Kuril": "#1f77b4", "Japan": "#7f7f7f", "Sagami": "#9467bd",
                 "Hokkaido": "#17becf", "Inland": "#ff7f0e"}


def _fig_map_gmt(nodes, segments, ev, cfg, out, pygmt):
    lon0, lon1, lat0, lat1 = cfg.map_region
    fig = pygmt.Figure()
    pygmt.config(FONT_ANNOT_PRIMARY="9p", FONT_LABEL="10p", FONT_TITLE="12p", MAP_FRAME_TYPE="plain")
    fig.basemap(region=[lon0, lon1, lat0, lat1], projection="M14c",
                frame=["WSen+tDRYAS - Sensor Array, Fault Segments and Assigned M>=7 Events",
                       "xaf+lLongitude (deg E)", "yaf+lLatitude (deg N)"])
    fig.coast(land="#ececec", water="white", shorelines="0.35p,#555555", resolution="i")
    for s in segments:
        xs = [c[0] for c in s["corners"]] + [s["corners"][0][0]]
        ys = [c[1] for c in s["corners"]] + [s["corners"][0][1]]
        col = TRENCH_COLORS.get(s["trench"], "#333333")
        fig.plot(x=xs, y=ys, pen=f"0.9p,{col}", fill=f"{col}@85")
        cx, cy = polygon_centroid(s["corners"])
        fig.text(x=cx, y=cy, text=s["id"], font=f"6p,Helvetica-Bold,{col}", justify="CM")
    big = ev[(ev["mag"] >= 7.0) & (ev["segment"] != "UNASSIGNED")]
    if len(big):
        fig.plot(x=big["lon"], y=big["lat"], style="cc", size=0.045 * (big["mag"] - 4) ** 2,
                 pen="0.8p,#333333", fill="#ffffff@40")
    fig.plot(x=nodes["lon"], y=nodes["lat"], style="t0.36c", fill="#2e7d32", pen="0.4p,black")
    lab = nodes.sort_values("lat", ascending=False).reset_index(drop=True)
    fig.text(x=lab["lon"], y=lab["lat"], text=[str(i + 1) for i in range(len(lab))],
             font="7.5p,Helvetica-Bold,#1b4d1e", justify="LM", offset="0.22c/0.12c", fill="white@25")
    key = [f"L 7p,Helvetica L {i + 1}  {r['site_name']}" for i, r in lab.iterrows()]
    spec = "\n".join([
        "S 0.35c t 0.32c #2e7d32 0.4p,black 0.8c Tree-ring site (numbered)",
        "S 0.35c c 0.32c #ffffff 0.8p,#333333 0.8c Assigned M>=7 event (size ~ M)",
        "S 0.35c r 0.5c/0.25c #d62728@85 0.9p,#d62728 0.8c Nankai Trough segments",
        "S 0.35c r 0.5c/0.25c #1f77b4@85 0.9p,#1f77b4 0.8c Kuril Trench segments",
        "S 0.35c r 0.5c/0.25c #7f7f7f@85 0.9p,#7f7f7f 0.8c Japan Trench segments",
        "S 0.35c r 0.5c/0.25c #ff7f0e@85 0.9p,#ff7f0e 0.8c Inland fault zones",
        "G 0.15c", "D 0.1c 0.4p,#888888", "G 0.1c", "L 7.5p,Helvetica-Bold L Site key", *key]) + "\n"
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(spec); sp = fh.name
    fig.legend(spec=sp, position="JMR+jML+o0.4c+w5.6c", box=False)
    fig.savefig(out)
    Path(sp).unlink(missing_ok=True)


def _fig_map_mpl(nodes, segments, ev, cfg, out):
    fig, ax = plt.subplots(figsize=(9, 8.5), facecolor="white"); ax.set_facecolor("white")
    for s in segments:
        col = TRENCH_COLORS.get(s["trench"], "#333333")
        xs = [c[0] for c in s["corners"]] + [s["corners"][0][0]]; ys = [c[1] for c in s["corners"]] + [s["corners"][0][1]]
        ax.fill(xs, ys, color=col, alpha=0.15); ax.plot(xs, ys, color=col, linewidth=0.9)
        cx, cy = polygon_centroid(s["corners"]); ax.text(cx, cy, s["id"], fontsize=6, ha="center", color=col)
    big = ev[(ev["mag"] >= 7.0) & (ev["segment"] != "UNASSIGNED")]
    ax.scatter(big["lon"], big["lat"], s=(big["mag"] - 4) ** 2 * 6, facecolor="none", edgecolor="#333", zorder=3)
    ax.scatter(nodes["lon"], nodes["lat"], marker="^", s=45, color="#2e7d32", edgecolor="black", zorder=4)
    for i, r in nodes.sort_values("lat", ascending=False).reset_index(drop=True).iterrows():
        ax.annotate(str(i + 1), (r["lon"], r["lat"]), xytext=(5, 3), textcoords="offset points", fontsize=7)
    ax.set_xlim(cfg.map_region[0], cfg.map_region[1]); ax.set_ylim(cfg.map_region[2], cfg.map_region[3])
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean(cfg.map_region[2:]))))
    ax.set_xlabel("Longitude (deg E)"); ax.set_ylabel("Latitude (deg N)")
    ax.set_title("DRYAS - Sensor Array, Fault Segments and Assigned M>=7 Events")
    handles = [Patch(facecolor=c, alpha=0.4, label=f"{t} segments") for t, c in TRENCH_COLORS.items()]
    handles.append(Line2D([0], [0], marker="^", color="w", markerfacecolor="#2e7d32", markersize=9, label="Tree-ring site"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_response_matrix(site_event: pd.DataFrame, nodes: pd.DataFrame, ev: pd.DataFrame, out: Path):
    big = ev[ev["mag"] >= 7.5].sort_values("year")
    if len(big) == 0:
        return
    se = site_event[site_event["event_id"].isin(big["event_id"])]
    mat = se.pivot(index="site_id", columns="event_id", values="p_response").reindex(columns=big["event_id"])
    order = nodes.sort_values("lat", ascending=False)["site_id"].tolist()
    mat = mat.reindex([s for s in order if s in mat.index])
    names = dict(zip(nodes["site_id"], nodes["site_name"]))
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(big) + 3), 0.42 * len(mat) + 2.5), facecolor="white")
    im = ax.imshow(mat.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(len(mat))); ax.set_yticklabels([names.get(s, s) for s in mat.index], fontsize=8)
    ax.set_xticks(range(len(big)))
    ax.set_xticklabels([f"{r['year']} {r['segment']} M{r['mag']:.1f}" for _, r in big.iterrows()],
                       rotation=90, fontsize=6.5)
    ax.set_title("DRYAS - Prior Expected Tree Response p(PGA) per Site and M>=7.5 Event")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); cb.set_label("p_response (prior, to be refit in 006)")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


def fig_attenuation(cfg: Config, out: Path):
    x = np.linspace(1, 600, 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="white")
    for mw, col in [(6.5, "#9ecae1"), (7.0, "#6baed6"), (7.5, "#3182bd"), (8.0, "#08519c"), (8.5, "#08306b")]:
        pga = pga_si_midorikawa_1999(mw, x, cfg.depth_default_km, "interplate")
        axes[0].plot(x, pga, color=col, label=f"Mw {mw}")
        axes[1].plot(x, p_response(pga, cfg), color=col, label=f"Mw {mw}")
    axes[0].set_yscale("log"); axes[0].set_xlabel("Fault distance (km)"); axes[0].set_ylabel("PGA (gal), Si & Midorikawa 1999, interplate, D=20 km")
    axes[0].axhline(10 ** cfg.resp_x0_log10gal, color="#999", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Fault distance (km)"); axes[1].set_ylabel("Prior p(tree response)")
    axes[1].axhline(0.5, color="#999", linestyle="--", linewidth=0.8)
    for a in axes:
        a.grid(True, linestyle=":", linewidth=0.5); a.set_facecolor("white")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8.5)
    fig.suptitle("DRYAS - Ground-Motion Prior used to Link Events to Tree Sites", y=1.02)
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def save_table(df, name, records):
    p = DATA_DIR / name
    df.to_parquet(p, index=False)
    records[name] = {"rows": int(len(df)), "sha256": _sha256_of_file(p), "path": str(p)}
    print(f"  wrote {p}  ({len(df):,} rows)")


def main():
    ap = argparse.ArgumentParser(description="DRYAS 003 — spatial graph & fault geometry")
    ap.add_argument("--segments", type=str, default=None, help="JSON file with a SEGMENTS-like list")
    args = ap.parse_args()
    cfg = Config()
    segments = json.load(open(args.segments)) if args.segments else SEGMENTS
    for s in segments:
        s["corners"] = [tuple(c) for c in s["corners"]]

    sites = pd.read_parquet(DATA_DIR / "site_metadata.parquet")
    eq = pd.read_parquet(DATA_DIR / "eq_catalog.parquet")
    chron_p, sel_p, prov_p = DATA_DIR / "chronologies.parquet", DATA_DIR / "002_selection.json", DATA_DIR / "provenance.json"
    chron = pd.read_parquet(chron_p) if chron_p.exists() else None
    sel = json.load(open(sel_p)) if sel_p.exists() else None
    prov = json.load(open(prov_p)) if prov_p.exists() else {}
    print(f"DRYAS 003 — {len(sites)} sites, {len(eq):,} catalog events, {len(segments)} segments")

    nodes = fill_merged_coords(build_nodes(sites, chron, sel), prov)
    nodes = nodes[nodes.get("usable_depth_years", pd.Series(1, index=nodes.index)).fillna(1) > 0].reset_index(drop=True)
    print(f"  nodes: {len(nodes)} (sites with detection-usable years)")

    # catalog -> segments
    ev = assign_segments(eq, segments, cfg)
    n_un = int((ev["segment"] == "UNASSIGNED").sum())
    print(f"  events M>={cfg.assign_mmin}: {len(ev)} assigned, {n_un} unassigned (> {cfg.assign_max_km:.0f} km from any segment)")

    # site x event PGA prior
    rows = []
    for _, e in ev.iterrows():
        seg = next((s for s in segments if s["id"] == e["segment"]), None)
        for _, n in nodes.iterrows():
            if seg is not None and e["mech"] != "intraplate":
                xs = dist_point_to_polygon_km(n["lon"], n["lat"], seg["corners"])
                xs = max(xs, 10.0) if xs == 0 else xs
                # shallow interplate/crustal rupture: fault distance ~ sqrt(surface^2 + depth^2)
                x = float(np.hypot(xs, min(e["depth_km"], 30.0)))
            else:
                # point source (unassigned or slab event): hypocentral distance
                x = float(np.hypot(haversine_km(n["lat"], n["lon"], e["lat"], e["lon"]), e["depth_km"]))
            pga = float(pga_si_midorikawa_1999(e["mag"], np.array([x]), e["depth_km"], e["mech"])[0])
            rows.append({"site_id": n["site_id"], "event_id": e["event_id"], "year": e["year"],
                         "eff_year": e["eff_year"], "month": e["month"],
                         "mag": e["mag"], "segment": e["segment"], "dist_km": float(x),
                         "pga_gal": pga, "p_response": float(p_response(np.array([pga]), cfg)[0]),
                         "in_usable_span": bool(n.get("usable_depth_first", -1) <= e["eff_year"] <= n.get("usable_depth_last", 9999))})
    site_event = pd.DataFrame(rows)

    # graph edges
    ss = []
    for i, a in nodes.iterrows():
        for j, b in nodes.iterrows():
            if j <= i:
                continue
            ss.append({"src": a["site_id"], "dst": b["site_id"],
                       "dist_km": float(haversine_km(a["lat"], a["lon"], b["lat"], b["lon"])),
                       "same_cluster": a["cluster"] == b["cluster"]})
    edges_ss = pd.DataFrame(ss)
    sg = []
    for _, n in nodes.iterrows():
        for s in segments:
            x = dist_point_to_polygon_km(n["lon"], n["lat"], s["corners"])
            x_eff = max(x, 10.0)
            pga = float(pga_si_midorikawa_1999(cfg.reference_mag, np.array([x_eff]), cfg.depth_default_km, s["mech"])[0])
            sg.append({"site_id": n["site_id"], "segment": s["id"], "trench": s["trench"],
                       "dist_km": float(x), "pga_ref_gal": pga,
                       "p_response_ref": float(p_response(np.array([pga]), cfg)[0])})
    edges_sg = pd.DataFrame(sg)

    # persist
    records = {}
    geo = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {k: v for k, v in s.items() if k != "corners"},
         "geometry": {"type": "Polygon", "coordinates": [[list(c) for c in s["corners"]] + [list(s["corners"][0])]]}}
        for s in segments]}
    gp = DATA_DIR / "fault_segments.geojson"
    json.dump(geo, open(gp, "w"), indent=1)
    records["fault_segments.geojson"] = {"rows": len(segments), "sha256": _sha256_of_file(gp), "path": str(gp)}
    print(f"  wrote {gp}")
    save_table(nodes, "graph_nodes.parquet", records)
    save_table(edges_ss, "graph_edges_site_site.parquet", records)
    save_table(edges_sg, "graph_edges_site_segment.parquet", records)
    save_table(ev, "eq_segment_assignment.parquet", records)
    save_table(site_event, "site_event_pga.parquet", records)
    prov["003"] = {"script": "003_spatial_graph.py", "generated_utc": datetime.now(timezone.utc).isoformat(),
                   "config": asdict(cfg), "artifacts": records,
                   "note": "segment rectangles are approximate textbook geometry; see script header"}
    json.dump(prov, open(prov_p, "w"), indent=2, default=str)

    print("rendering figures ...")
    fig_map(nodes, segments, ev, cfg, PDF_DIR / "003a_array_and_segments.pdf")
    fig_response_matrix(site_event, nodes, ev, PDF_DIR / "003b_expected_response_matrix.pdf")
    fig_attenuation(cfg, PDF_DIR / "003c_attenuation_curves.pdf")
    for f in ("003a_array_and_segments", "003b_expected_response_matrix", "003c_attenuation_curves"):
        print(f"  wrote {PDF_DIR / (f + '.pdf')}")

    print("\n=== SUMMARY ===")
    print("  segment counts (M>=7, assigned):")
    big = ev[ev["mag"] >= 7.0]
    print(big.groupby("segment").size().sort_values(ascending=False).to_string())
    print("  events with prior p_response >= 0.5 at >= 1 site inside its usable span:")
    hit = site_event[(site_event["p_response"] >= 0.5) & site_event["in_usable_span"]]
    tab = hit.groupby(["event_id", "year", "segment", "mag"])["site_id"].apply(lambda s: ",".join(sorted(s))).reset_index()
    print(tab.sort_values("year").to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
