#!/usr/bin/env python3
# =============================================================================
# DRYAS  |  Script 001  —  DATA ACQUISITION & INTEGRATION
# =============================================================================
# WHAT THIS SCRIPT DOES
# ---------------------------------------------------------------------------
# DRYAS treats a spatially-distributed network of fault-proximal trees as a
# passive sensor array, in order to reconstruct multi-century earthquake-cascade
# history from tree-ring archives. This first script builds the two raw data
# layers that the whole pipeline stands on and harmonises them into a single,
# fully-provenanced bundle:
#
#   LAYER A — Tree-ring network (the "sensor array")
#       * Source (real): ITRDB / NOAA NCEI Paleoclimatology. Every Japanese
#         measurement file that actually exists in the NOAA "asia" directory
#         (verified 2026-09-04): japa1, japa008, japa010-japa020. For each site
#         the NOAA Template file "<code>-rwl-noaa.txt" is fetched first: its
#         header carries lat/lon/elevation/species/first-last year and its body
#         is a tab-separated year x tree matrix already in millimetres. If the
#         template is unavailable the Tucson ".rwl" is fetched and parsed instead
#         (then site coordinates stay NaN and are flagged).
#       * Optional multi-proxy: the "<code>x" maximum-latewood-density files
#         (Schweingruber-type) are fetched into a separate density table.
#       * Downloads are cached under data/raw/ so re-runs are offline-capable
#         (TALOS-style: real files can also be dropped there by hand).
#       * Tidy long table: (site_id, tree_id, year, ring_width_mm) plus a
#         site_metadata table (site_id, site_name, lat, lon, elev_m, species,
#         first_year, last_year, n_series, investigators, url, coords_source).
#       * NOTE on Yakushima (JAPA020): Cryptomeria japonica, 1-1999 CE. It is
#         the only two-millennium record in the array and sits 100-200 km from
#         the Nankai/Hyuga-nada seismogenic zone; sample depth before ~450 CE is
#         only two trees, so 002 must gate usable years by sample depth / EPS.
#
#   LAYER B — Earthquake catalog (the "ground truth" for calibration)
#       * Source (real, instrumental era ~1900+): USGS FDSN event service,
#         restricted to the Japan region and M >= MMIN.
#       * Source (historical era, pre-instrumental): a bundled table of major
#         Japanese earthquakes with textbook date/lat/lon/magnitude parameters
#         (used for validation & as the long-baseline reconstruction target).
#
# Calibration anchor for DRYAS = the Japanese historical earthquake record
# (decision "A"): Japan has an unusually rich multi-century written record,
# which is ideal ground truth for calibrating the tree-ring response.
#
# NETWORK NOTE
#   The real hosts (ncei.noaa.gov, earthquake.usgs.gov) are reachable on a
#   normal machine/Colab but NOT inside the restricted sandbox. The script
#   therefore supports three modes:
#       --mode real       : download from the real public sources (use on your machine)
#       --mode synthetic  : build a controlled synthetic ground-truth testbed
#       --mode auto       : try real, fall back to synthetic on failure (default)
#   The synthetic generator is NOT a throwaway: it embeds known earthquake
#   suppression signatures into the ring-width series, giving every downstream
#   script (separation, detection, cascade analysis) a controlled ground truth.
#
# OUTPUTS (written to ../data and ../PDF)
#   data/treering_network.parquet   tidy ring-width long table
#   data/eq_catalog.parquet         merged earthquake catalog
#   data/site_metadata.parquet      one row per tree site
#   data/provenance.json            source, mode, sha256, UTC timestamp per artifact
#   PDF/001a_spatial_coverage.pdf   map: tree sites + earthquake epicentres
#   PDF/001b_temporal_coverage.pdf  chronology spans + earthquake timeline
#
# FIGURE CONVENTIONS (project standard)
#   White background; all text in English; legends placed in the margin so they
#   never overlap the data.
#   001a is rendered with PyGMT (coastlines, numbered sites + site key, M>=7
#   circles, M5-7 as faint dots) when GMT is installed:
#       macOS   : brew install gmt      ; uv sync --extra gmt
#       Windows : install GMT 6 from github.com/GenericMappingTools/gmt/releases,
#                 then uv sync --extra gmt
#   Without GMT the script falls back to an equivalent matplotlib map (no coastline).
#
# USAGE
#   python 001_data_acquisition.py                 # auto mode
#   python 001_data_acquisition.py --mode real     # real download (your machine)
#   python 001_data_acquisition.py --mode synthetic --n-sites 60 --seed 7
# =============================================================================

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless / file output only
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths (resolve relative to this file so the script is location-independent)
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
DATA_DIR = ROOT / "data"
PDF_DIR = ROOT / "PDF"
for _d in (DATA_DIR, PDF_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # Geographic window for Japan (lon_min, lon_max, lat_min, lat_max)
    bbox: tuple = (128.0, 146.5, 30.0, 46.0)
    # Map extent for figures only (slightly wider so Yakushima's response radius is not clipped)
    map_region: tuple = (127.0, 147.0, 29.0, 46.5)
    mmin: float = 5.0            # minimum magnitude kept (tree response threshold)
    # Catalog / chronology window. In SYNTHETIC mode this is fixed at 1500 so the
    # testbed stays reproducible. In REAL mode main() lowers year_min to the
    # first year actually covered by the downloaded rings (Yakushima -> 1 CE),
    # so pre-1500 historical earthquakes are kept for calibration.
    year_min: int = 1500
    year_max: int = 2026
    fault_response_radius_km: float = 80.0  # radius within which a tree "feels" a quake
    # ITRDB site codes to fetch in real mode. These are the COMPLETE set of
    # Japanese ring-width measurement files present in the NOAA NCEI "asia"
    # directory as of 2026-09-04 (japa001-007 and japa009 do not exist there).
    # Known sites: japa008 Mt Asahidake (Hokkaido, Picea glehnii, 1532-1997),
    # japa011 Shiretoko, japa012 Teshio, japa013 Moshiri, japa014 Tokachi
    # (all Hokkaido, P. glehnii, Davi et al.), japa020 Yakushima (Cryptomeria
    # japonica, 1-1999 CE), japa1 Miyajima/Hiroshima Bay (Pinus densiflora, short).
    # Coordinates/species/years for every site are read from the NOAA header.
    itrdb_base: str = "https://www.ncei.noaa.gov/pub/data/paleo/treering/measurements/asia/"
    itrdb_sites: tuple = (
        "japa008", "japa010", "japa011", "japa012", "japa013", "japa014",
        "japa015", "japa016", "japa017", "japa018", "japa019", "japa020",
        "japa1",
    )
    # Optional multi-proxy layer: maximum-latewood-density files ("x" suffix).
    itrdb_density_sites: tuple = (
        "japa008x", "japa012x", "japa013x", "japa014x", "japa015x",
        "japa016x", "japa017x", "japa018x",
    )
    raw_dir: str = "raw"         # data/raw/ local cache of downloaded source files
    usgs_fdsn: str = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    # Synthetic generator controls
    n_sites: int = 60
    trees_per_site: tuple = (8, 18)   # min,max cores per site
    seed: int = 13
    mode: str = "auto"


# ---------------------------------------------------------------------------
# Bundled major Japanese historical earthquakes.
# Date / lat / lon / Mw are well-established textbook values (public-domain
# facts). Used for the historical (pre-instrumental) era and as validation
# targets. "instrumental" events are also kept so synthetic mode is self-contained.
# ---------------------------------------------------------------------------
HISTORICAL_JP_EQ = [
    # year, month, day, lat,   lon,    mag,  name,                       era
    # Pre-1500 events (kept only when year_min is lowered in REAL mode; the
    # synthetic testbed filters at 1500 and is unaffected). Historical
    # epicentres are textbook mid-range values (Usami-type catalogues) and are
    # inherently approximate (~0.5 deg); magnitudes are catalogue M estimates.
    (684, 11, 29, 33.00, 134.50, 8.3, "Hakuho Nankai",            "historical"),
    (869,  7, 13, 38.50, 143.80, 8.4, "Jogan Sanriku",            "historical"),
    (887,  8, 26, 33.00, 135.00, 8.3, "Ninna Nankai",             "historical"),
    (1096,12, 17, 34.00, 137.50, 8.3, "Eicho Tokai",              "historical"),
    (1099, 2, 22, 33.00, 135.50, 8.2, "Kowa Nankai",              "historical"),
    (1185, 8, 13, 35.00, 135.80, 7.4, "Bunji Kyoto",              "historical"),
    (1293, 5, 27, 35.20, 139.50, 7.2, "Einin Kamakura",           "historical"),
    (1361, 8,  3, 33.00, 135.00, 8.4, "Shohei Nankai",            "historical"),
    (1498, 9, 20, 34.00, 138.00, 8.3, "Meio Tokai",               "historical"),
    (1586,11,29, 36.00, 136.90, 7.9, "Tensho",                   "historical"),
    (1605, 2, 3, 33.50, 138.50, 7.9, "Keicho Nankaido",          "historical"),
    (1611,12, 2, 39.00, 144.40, 8.1, "Keicho Sanriku",           "historical"),
    (1703,12,31, 34.70, 139.80, 8.2, "Genroku Kanto",            "historical"),
    (1707,10,28, 33.20, 135.90, 8.6, "Hoei",                     "historical"),
    (1751, 5,21, 37.10, 138.20, 7.0, "Echigo-Takada",            "historical"),
    (1828,12,18, 37.60, 138.90, 6.9, "Sanjo",                    "historical"),
    (1847, 5, 8, 36.70, 138.20, 7.4, "Zenkoji",                  "historical"),
    (1854,12,23, 34.00, 137.80, 8.4, "Ansei Tokai",              "historical"),
    (1854,12,24, 33.00, 135.00, 8.4, "Ansei Nankai",             "historical"),
    (1855,11,11, 35.65, 139.80, 7.0, "Ansei Edo",                "historical"),
    (1891,10,28, 35.60, 136.30, 8.0, "Nobi",                     "historical"),
    (1896, 6,15, 39.50, 144.00, 8.3, "Meiji Sanriku",            "historical"),
    (1923, 9, 1, 35.30, 139.10, 7.9, "Taisho Kanto",             "instrumental"),
    (1927, 3, 7, 35.60, 134.90, 7.3, "Kita-Tango",               "instrumental"),
    (1933, 3, 3, 39.20, 144.70, 8.1, "Showa Sanriku",            "instrumental"),
    (1943, 9,10, 35.50, 134.20, 7.2, "Tottori",                  "instrumental"),
    (1944,12, 7, 33.60, 136.20, 7.9, "Tonankai",                 "instrumental"),
    (1945, 1,13, 34.70, 137.10, 6.8, "Mikawa",                   "instrumental"),
    (1946,12,21, 33.00, 135.60, 8.0, "Showa Nankai",             "instrumental"),
    (1948, 6,28, 36.10, 136.20, 7.1, "Fukui",                    "instrumental"),
    (1995, 1,17, 34.60, 135.00, 6.9, "Hyogo-ken Nanbu (Kobe)",   "instrumental"),
    (2004,10,23, 37.30, 138.80, 6.8, "Niigata Chuetsu",          "instrumental"),
    (2007, 7,16, 37.50, 138.60, 6.8, "Chuetsu-oki",              "instrumental"),
    (2008, 6,14, 39.00, 140.90, 7.2, "Iwate-Miyagi Nairiku",     "instrumental"),
    (2011, 3,11, 38.10, 142.90, 9.0, "Tohoku-oki",               "instrumental"),
    (2016, 4,16, 32.80, 130.80, 7.0, "Kumamoto",                 "instrumental"),
    (2024, 1, 1, 37.50, 137.30, 7.6, "Noto Peninsula",           "instrumental"),
    (2026, 4,20, 39.20, 144.00, 7.7, "Sanriku-oki (2026)",       "instrumental"),
]
HIST_COLS = ["year", "month", "day", "lat", "lon", "mag", "name", "era"]


# ---------------------------------------------------------------------------
# Provenance helpers (MNEMOSYNE-style: every artifact tagged with source + hash)
# ---------------------------------------------------------------------------
def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Geodesy helper
# ---------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km. Arrays broadcast."""
    R = 6371.0088
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Tucson (.rwl) ring-width parser  — dependency-light, format-robust
# ---------------------------------------------------------------------------
def parse_rwl(text: str, site_id: str) -> pd.DataFrame:
    """
    Parse a Tucson decadal ring-width (.rwl) file into a tidy long DataFrame:
    columns = [site_id, tree_id, year, ring_width_mm].

    The Tucson format stores, per line: a series id, a decade-start year, then up
    to 10 ring values (typically in 0.01 mm or 0.001 mm units). Terminators are
    999 (0.01 mm files) or -9999 (0.001 mm files). We auto-detect the unit by the
    terminator family and convert everything to millimetres.
    """
    rows = []
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        # series id = first token; the rest should be numeric (decade + values)
        sid = line[:8].strip()
        rest = line[8:].split()
        if not sid or len(rest) < 2:
            continue
        try:
            decade = int(rest[0])
        except ValueError:
            continue  # header / non-data line
        try:
            vals = [int(v) for v in rest[1:]]
        except ValueError:
            continue
        if not (-3000 < decade < 3000):
            continue
        for i, v in enumerate(vals):
            if v in (999, -9999):       # series terminator markers
                continue
            if v < 0:                    # stray negative -> skip
                continue
            rows.append((sid, decade + i, v))
    if not rows:
        return pd.DataFrame(columns=["site_id", "tree_id", "year", "ring_width_mm"])
    df = pd.DataFrame(rows, columns=["tree_id", "year", "raw"])
    # Unit heuristic: if many large values (>100) it is almost certainly 0.01 mm.
    scale = 0.01 if df["raw"].median() > 50 else 0.001
    df["ring_width_mm"] = df["raw"] * scale
    df.insert(0, "site_id", site_id)
    return df.drop(columns="raw")


# ---------------------------------------------------------------------------
# NOAA Template ("<code>-rwl-noaa.txt") parser
#   Header lines start with '#'. Site metadata lives in "#   Key: value" lines.
#   Variable definitions start with '##'. The data block is tab-separated with
#   a header row (age_CE, <series>_raw, ...) and "NA" for missing values;
#   ring widths are already in millimetres. A literal 0 is a locally absent
#   ring and is KEPT (it is the strongest possible suppression signal).
# ---------------------------------------------------------------------------
_NOAA_META_KEYS = {
    "Site_Name": "site_name", "Location": "location",
    "Northernmost_Latitude": "lat", "Easternmost_Longitude": "lon",
    "Elevation_m": "elev_m", "First_Year": "first_year", "Last_Year": "last_year",
    "Species_Name": "species", "Tree_Species_Code": "species_code",
    "Investigators": "investigators", "Parameter_Keywords": "parameter",
    "Collection_Name": "collection", "NOAA_Landing_Page": "landing_page",
}


def parse_noaa_template(text: str, site_id: str):
    """
    Parse a NOAA paleo Template v4 tree-ring measurement file.
    Returns (meta: dict, tidy: DataFrame[site_id, tree_id, year, value]).
    """
    meta = {"site_id": site_id}
    data_lines = []
    for raw in text.splitlines():
        line = raw.rstrip("\r\n")
        if line.startswith("##"):
            continue                       # per-variable definitions
        if line.startswith("#"):
            body = line.lstrip("#").strip()
            if ":" in body:
                key, _, val = body.partition(":")
                key = key.strip()
                if key in _NOAA_META_KEYS:
                    # keep the first occurrence only (Location appears once,
                    # NOAA_Landing_Page has a URL containing ':' -> use partition)
                    meta.setdefault(_NOAA_META_KEYS[key], val.strip())
            continue
        if line.strip():
            data_lines.append(line)
    if not data_lines:
        return meta, pd.DataFrame(columns=["site_id", "tree_id", "year", "value"])
    wide = pd.read_csv(io.StringIO("\n".join(data_lines)), sep="\t",
                       na_values=["NA"], dtype=str)
    year_col = wide.columns[0]
    wide[year_col] = pd.to_numeric(wide[year_col], errors="coerce")
    wide = wide.dropna(subset=[year_col])
    tidy = wide.melt(id_vars=[year_col], var_name="tree_id", value_name="value")
    tidy["value"] = pd.to_numeric(tidy["value"], errors="coerce")
    tidy = tidy.dropna(subset=["value"])
    tidy["tree_id"] = tidy["tree_id"].str.replace(r"_raw$", "", regex=True).str.strip()
    tidy["year"] = tidy[year_col].astype(int)
    tidy.insert(0, "site_id", site_id)
    tidy = tidy[["site_id", "tree_id", "year", "value"]].reset_index(drop=True)
    # numeric coercion of header fields
    for k in ("lat", "lon", "elev_m", "first_year", "last_year"):
        if k in meta:
            try:
                meta[k] = float(meta[k]) if k in ("lat", "lon", "elev_m") else int(meta[k])
            except ValueError:
                meta[k] = np.nan
    return meta, tidy


def _load_source_text(cfg: Config, fname: str) -> tuple[str, str]:
    """
    Return (text, origin) for a NOAA file. Order: local cache data/raw/<fname>
    (offline / hand-placed real data), then HTTP download (cached afterwards).
    """
    raw_dir = DATA_DIR / cfg.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)
    local = raw_dir / fname
    if local.exists() and local.stat().st_size > 0:
        return local.read_text(encoding="utf-8", errors="replace"), f"cache:{local}"
    import requests
    url = cfg.itrdb_base + fname
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    local.write_text(r.text, encoding="utf-8")
    return r.text, url


def _acquire_site(cfg: Config, code: str):
    """
    Acquire one ITRDB site: NOAA template first (metadata + data), Tucson .rwl
    as fallback (data only). Returns (meta, tidy) or raises.
    """
    try:
        text, origin = _load_source_text(cfg, f"{code}-rwl-noaa.txt")
        meta, tidy = parse_noaa_template(text, site_id=code)
        if tidy.empty:
            raise ValueError("template parsed but no data rows")
        meta["url"] = origin
        meta["coords_source"] = "noaa-template-header"
        meta["format"] = "noaa-template"
        return meta, tidy
    except Exception as e_tpl:
        text, origin = _load_source_text(cfg, f"{code}.rwl")
        df = parse_rwl(text, site_id=code).rename(columns={"ring_width_mm": "value"})
        if df.empty:
            raise RuntimeError(f"template failed ({e_tpl}); .rwl parsed empty")
        meta = {"site_id": code, "url": origin, "lat": np.nan, "lon": np.nan,
                "species": "unknown", "coords_source": "MISSING-fill-manually",
                "format": "tucson-rwl", "parameter": "ring width"}
        return meta, df


def fetch_itrdb_real(cfg: Config):
    """
    Download the tree-ring sensor array. Returns (rings, sites, density):
      rings   : tidy ring widths [site_id, tree_id, year, ring_width_mm]
      sites   : one row per site with header metadata (lat/lon/species/...)
      density : tidy max-density proxies [site_id, tree_id, year, density]
                (empty DataFrame if none fetched)
    """
    frames, meta_rows, dens_frames = [], [], []
    for code in cfg.itrdb_sites:
        try:
            meta, tidy = _acquire_site(cfg, code)
            tidy = tidy.rename(columns={"value": "ring_width_mm"})
            meta["n_series"] = int(tidy["tree_id"].nunique())
            meta["first_year"] = int(tidy["year"].min())
            meta["last_year"] = int(tidy["year"].max())
            frames.append(tidy)
            meta_rows.append(meta)
            print(f"  [ITRDB] {code}: {meta.get('site_name', '?')} "
                  f"({meta.get('lat', np.nan)}, {meta.get('lon', np.nan)}) "
                  f"{meta['n_series']} series, {meta['first_year']}-{meta['last_year']} "
                  f"[{meta['format']}]")
        except Exception as e:
            print(f"  [ITRDB] {code}: FAILED ({e})")
    for code in cfg.itrdb_density_sites:
        try:
            meta, tidy = _acquire_site(cfg, code)
            tidy = tidy.rename(columns={"value": "density"})
            tidy["parameter"] = meta.get("parameter", "density")
            dens_frames.append(tidy)
            print(f"  [ITRDB-density] {code}: {tidy['tree_id'].nunique()} series, "
                  f"{tidy['year'].min()}-{tidy['year'].max()} "
                  f"({meta.get('parameter', '?')})")
        except Exception as e:
            print(f"  [ITRDB-density] {code}: skipped ({e})")
    if not frames:
        raise RuntimeError("No ITRDB site downloaded.")
    rings = pd.concat(frames, ignore_index=True)
    sites = pd.DataFrame(meta_rows)
    keep = ["site_id", "site_name", "lat", "lon", "elev_m", "species", "species_code",
            "first_year", "last_year", "n_series", "investigators", "parameter",
            "url", "coords_source", "format", "landing_page"]
    for k in keep:
        if k not in sites.columns:
            sites[k] = np.nan
    sites = sites[keep]
    missing = sites["lat"].isna().sum()
    if missing:
        print(f"  [ITRDB] WARNING: {missing} site(s) without coordinates "
              f"(fallback .rwl) — fill lat/lon manually before 003.")
    density = (pd.concat(dens_frames, ignore_index=True) if dens_frames
               else pd.DataFrame(columns=["site_id", "tree_id", "year", "density", "parameter"]))
    return rings, sites, density


def fetch_eq_real(cfg: Config) -> pd.DataFrame:
    import requests
    lon0, lon1, lat0, lat1 = cfg.bbox
    params = {
        "format": "csv", "starttime": "1900-01-01", "endtime": "2026-12-31",
        "minmagnitude": cfg.mmin,
        "minlatitude": lat0, "maxlatitude": lat1,
        "minlongitude": lon0, "maxlongitude": lon1,
        "orderby": "time",
    }
    r = requests.get(cfg.usgs_fdsn, params=params, timeout=60)
    r.raise_for_status()
    usgs = pd.read_csv(io.StringIO(r.text))
    usgs = usgs.rename(columns={"latitude": "lat", "longitude": "lon", "mag": "mag"})
    usgs["time"] = pd.to_datetime(usgs["time"], errors="coerce", utc=True)
    usgs["year"] = usgs["time"].dt.year
    usgs["month"] = usgs["time"].dt.month
    usgs["name"] = usgs.get("place", "")
    usgs["era"] = "instrumental"
    usgs["source"] = "USGS-FDSN"
    if "depth" not in usgs.columns:
        usgs["depth"] = np.nan
    inst = usgs[["year", "month", "lat", "lon", "depth", "mag", "name", "era", "source"]].dropna(subset=["mag"])
    hist = bundled_historical(cfg)
    hist = hist[hist["year"] < int(inst["year"].min() if len(inst) else 1900)]
    return pd.concat([hist, inst], ignore_index=True).sort_values("year").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Bundled historical catalog (used in all modes)
# ---------------------------------------------------------------------------
def bundled_historical(cfg: Config) -> pd.DataFrame:
    df = pd.DataFrame(HISTORICAL_JP_EQ, columns=HIST_COLS)
    df = df[(df["mag"] >= cfg.mmin) & (df["year"] >= cfg.year_min)].copy()
    df["source"] = "bundled-historical"
    df["depth"] = np.nan          # historical events: depth unknown (003 uses a default)
    return df[["year", "month", "lat", "lon", "depth", "mag", "name", "era", "source"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# SYNTHETIC ground-truth generator
#   Builds a fault-proximal tree network whose ring widths = climate (AR1) +
#   long-term age trend + KNOWN earthquake suppression + noise. Because the
#   embedded earthquakes are known, this is a controlled testbed for every
#   downstream separation / detection / cascade script.
# ---------------------------------------------------------------------------
# Approximate traces of major Japanese active fault zones, as (lat, lon) anchors,
# used to scatter synthetic tree sites near tectonically active ground.
FAULT_ANCHORS = [
    ("ISTL (Itoigawa-Shizuoka)", 36.2, 138.0),
    ("MTL (Median Tectonic Line)", 34.0, 134.5),
    ("Atotsugawa", 36.4, 137.2),
    ("Neodani (Nobi)", 35.6, 136.4),
    ("Atera", 35.6, 137.4),
    ("Yamasaki", 35.0, 134.4),
    ("Tottori zone", 35.4, 134.0),
    ("Hidaka (Hokkaido)", 42.7, 142.6),
    ("Sanriku coast", 39.3, 141.9),
    ("Kumamoto (Futagawa)", 32.8, 130.9),
    ("Noto", 37.4, 137.2),
]
SPECIES = ["Cryptomeria japonica", "Chamaecyparis obtusa", "Tsuga sieboldii",
           "Quercus crispula", "Larix kaempferi", "Pinus densiflora"]


def make_synthetic(cfg: Config):
    rng = np.random.default_rng(cfg.seed)
    eq = bundled_historical(cfg)
    eq["source"] = "bundled-historical"  # synthetic catalog == bundled historical

    # --- scatter sites near fault anchors -----------------------------------
    site_rows = []
    for i in range(cfg.n_sites):
        fa = FAULT_ANCHORS[i % len(FAULT_ANCHORS)]
        lat = fa[1] + rng.normal(0, 0.35)
        lon = fa[2] + rng.normal(0, 0.45)
        site_rows.append({
            "site_id": f"SYN{i:03d}",
            "fault_zone": fa[0],
            "lat": float(np.clip(lat, cfg.bbox[2], cfg.bbox[3])),
            "lon": float(np.clip(lon, cfg.bbox[0], cfg.bbox[1])),
            "species": SPECIES[i % len(SPECIES)],
            "url": "synthetic",
        })
    sites = pd.DataFrame(site_rows)

    # --- shared regional climate signal (AR1) so climate is spatially smooth -
    years = np.arange(cfg.year_min, cfg.year_max + 1)
    n_yr = len(years)
    climate = np.zeros(n_yr)
    for t in range(1, n_yr):
        climate[t] = 0.72 * climate[t - 1] + rng.normal(0, 0.45)
    climate = (climate - climate.mean()) / climate.std()

    # precompute per-(site,eq) felt intensity: suppression scales with M and 1/dist
    eq_arr = eq.reset_index(drop=True)
    ring_rows = []
    for _, s in sites.iterrows():
        n_tree = rng.integers(cfg.trees_per_site[0], cfg.trees_per_site[1] + 1)
        # distance from this site to every earthquake
        d = haversine_km(s.lat, s.lon, eq_arr["lat"].values, eq_arr["lon"].values)
        felt = np.zeros(n_yr)
        for j, row in eq_arr.iterrows():
            if d[j] > cfg.fault_response_radius_km:
                continue
            if row.year < cfg.year_min:
                continue
            # intensity proxy: grows with M, decays with distance
            amp = (0.35 * (row.mag - cfg.mmin + 1.0)
                   * np.exp(-d[j] / cfg.fault_response_radius_km))
            yi = int(row.year - cfg.year_min)
            # suppression in the event year + multi-year recovery (1-5 yr)
            for k, frac in enumerate([1.0, 0.6, 0.35, 0.2, 0.1]):
                if 0 <= yi + k < n_yr:
                    felt[yi + k] += amp * frac
        for tt in range(n_tree):
            # local climate sensitivity & age trend differ per tree
            sens = rng.uniform(0.5, 1.2)
            start = int(rng.integers(cfg.year_min, cfg.year_max - 60))
            age_axis = years - start
            age_trend = np.where(age_axis > 0,
                                 1.6 * np.exp(-age_axis / 220.0) + 0.4, np.nan)
            base = age_trend  # mm-scale baseline growth
            signal = (base
                      * (1.0 + 0.18 * sens * climate)   # climate modulation
                      * (1.0 - np.clip(felt, 0, 0.9))   # earthquake SUPPRESSION
                      + rng.normal(0, 0.05, n_yr))      # measurement noise
            valid = ~np.isnan(signal) & (years >= start)
            for y, w in zip(years[valid], signal[valid]):
                ring_rows.append((s.site_id, f"{s.site_id}-{tt:02d}",
                                  int(y), max(0.02, float(w))))
    rings = pd.DataFrame(ring_rows,
                         columns=["site_id", "tree_id", "year", "ring_width_mm"])
    return rings, sites, eq_arr


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
# Figure helpers -------------------------------------------------------------
MAG_EMPH = 7.0     # instrumental events >= this are drawn as sized circles; smaller ones as faint dots
MAG_LINE = 7.0     # instrumental events >= this get a timeline tick in the temporal figure


def _short_site_labels(sites: pd.DataFrame, cluster_km: float = 15.0) -> pd.DataFrame:
    """
    One label per spatial cluster of sites (sites closer than cluster_km are
    merged, e.g. the three Yakushima collections), so labels never overlap.
    Returns DataFrame [lat, lon, label].
    """
    s = sites.dropna(subset=["lat", "lon"]).copy()
    s["short"] = (s.get("site_name", s["site_id"]).fillna(s["site_id"]).astype(str)
                  .str.split(",").str[0].str.strip())
    used = np.zeros(len(s), dtype=bool)
    rows = []
    lat, lon = s["lat"].to_numpy(), s["lon"].to_numpy()
    for i in range(len(s)):
        if used[i]:
            continue
        d = haversine_km(lat[i], lon[i], lat, lon)
        members = np.where((d < cluster_km) & (~used))[0]
        used[members] = True
        names = list(dict.fromkeys(s["short"].iloc[members]))
        rows.append({"lat": float(lat[members].mean()), "lon": float(lon[members].mean()),
                     "label": " / ".join(names) if len(names) <= 3 else f"{names[0]} (+{len(names)-1})"})
    return pd.DataFrame(rows)


def _mag_size_cm(mag: np.ndarray, cfg: Config) -> np.ndarray:
    """Circle diameter (cm) proportional to magnitude above the catalog floor."""
    return 0.042 * (np.asarray(mag, dtype=float) - cfg.mmin + 1.0) ** 2.0


def fig_spatial_gmt(sites: pd.DataFrame, eq: pd.DataFrame, cfg: Config, out: Path):
    """
    Publication map with PyGMT/GMT: coastlines, tree sites (+ response radius),
    earthquakes split into faint small events and magnitude-scaled large events.
    Legend sits outside the frame on the right.
    """
    import pygmt
    lon0, lon1, lat0, lat1 = cfg.map_region
    fig = pygmt.Figure()
    pygmt.config(FONT_ANNOT_PRIMARY="9p", FONT_LABEL="10p", FONT_TITLE="12p",
                 MAP_FRAME_TYPE="plain", MAP_TITLE_OFFSET="4p")
    fig.basemap(region=[lon0, lon1, lat0, lat1], projection="M14c",
                frame=["WSen+tDRYAS - Tree-Ring Sensor Array and Earthquake Catalog (Japan)",
                       "xaf+lLongitude (deg E)", "yaf+lLatitude (deg N)"])
    fig.coast(land="#ececec", water="white", shorelines="0.35p,#555555", resolution="i")

    inst = eq[eq["era"] == "instrumental"]
    hist = eq[eq["era"] == "historical"]
    small = inst[inst["mag"] < MAG_EMPH]
    large = inst[inst["mag"] >= MAG_EMPH]
    if len(small):
        fig.plot(x=small["lon"], y=small["lat"], style="c0.045c", fill="#1f5fb0",
                 transparency=75)
    if len(large):
        fig.plot(x=large["lon"], y=large["lat"], style="cc",
                 size=_mag_size_cm(large["mag"], cfg), pen="1.0p,#1f5fb0", fill=None,
                 transparency=15)
    if len(hist):
        fig.plot(x=hist["lon"], y=hist["lat"], style="cc",
                 size=_mag_size_cm(hist["mag"], cfg), pen="1.2p,#B5651D", fill=None)
    ok = sites.dropna(subset=["lat", "lon"])
    if len(ok):
        # response radius (diameter in km, -SE- style) then the site symbol on top
        fig.plot(x=ok["lon"], y=ok["lat"], style=f"E-{2*cfg.fault_response_radius_km:.0f}k",
                 pen="0.5p,#2e7d32,-")
        fig.plot(x=ok["lon"], y=ok["lat"], style="t0.34c", fill="#2e7d32", pen="0.4p,black")
        lab = _short_site_labels(ok)
        lab = lab.sort_values("lat", ascending=False).reset_index(drop=True)
        lab["num"] = np.arange(1, len(lab) + 1)
        fig.text(x=lab["lon"], y=lab["lat"], text=lab["num"].astype(str),
                 font="7.5p,Helvetica-Bold,#1b4d1e", justify="LM", offset="0.22c/0.12c",
                 fill="white@25", clearance="0.5p")
    else:
        lab = pd.DataFrame(columns=["num", "label"])
    # explicit legend spec so symbol sizes in the legend are fixed (the -SE- km-sized
    # radius circles would otherwise explode the legend width); numbered site key below
    key_lines = [f"L 7p,Helvetica L {int(r['num'])}  {r['label']}" for _, r in lab.iterrows()]
    spec = "\n".join([
        "S 0.35c t 0.32c #2e7d32 0.4p,black 0.8c Tree-ring site (numbered)",
        f"S 0.35c - 0.6c - 0.5p,#2e7d32,- 0.8c Tree response radius ({cfg.fault_response_radius_km:.0f} km)",
        "S 0.35c c 0.32c - 1.2p,#B5651D 0.8c Historical EQ (pre-instrumental)",
        f"S 0.35c c 0.32c - 1.0p,#1f5fb0 0.8c Instrumental EQ M>={MAG_EMPH:g}",
        f"S 0.35c c 0.09c #1f5fb0 - 0.8c Instrumental EQ M{cfg.mmin:g}-{MAG_EMPH:g}",
        "L 8p,Helvetica-Oblique L (circle size proportional to magnitude)",
        "G 0.15c", "D 0.1c 0.4p,#888888", "G 0.1c",
        "L 7.5p,Helvetica-Bold L Site key",
        *key_lines,
    ]) + "\n"
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(spec); spec_path = fh.name
    fig.legend(spec=spec_path, position="JMR+jML+o0.4c+w5.4c", box=False)
    fig.savefig(out)
    Path(spec_path).unlink(missing_ok=True)


def fig_spatial_mpl(sites: pd.DataFrame, eq: pd.DataFrame, cfg: Config, out: Path):
    """Matplotlib fallback (no coastline): same layering rules as the GMT map."""
    fig, ax = plt.subplots(figsize=(8.2, 8.6), facecolor="white")
    ax.set_facecolor("white")
    inst = eq[eq["era"] == "instrumental"]
    hist = eq[eq["era"] == "historical"]
    small = inst[inst["mag"] < MAG_EMPH]
    large = inst[inst["mag"] >= MAG_EMPH]
    if len(small):
        ax.scatter(small["lon"], small["lat"], s=3, color="#1f5fb0", alpha=0.2, linewidths=0, zorder=2)
    for sub, col in ((large, "#1f5fb0"), (hist, "#B5651D")):
        if len(sub):
            ax.scatter(sub["lon"], sub["lat"], s=(sub["mag"] - cfg.mmin + 1) ** 2.6 * 8,
                       facecolor="none", edgecolor=col, linewidth=1.3, alpha=0.9, zorder=3)
    ok = sites.dropna(subset=["lat", "lon"])
    ax.scatter(ok["lon"], ok["lat"], s=38, marker="^", color="#2e7d32",
               edgecolor="black", linewidth=0.4, zorder=5)
    lab = _short_site_labels(ok).sort_values("lat", ascending=False).reset_index(drop=True)
    key = []
    for i, r in lab.iterrows():
        ax.annotate(str(i + 1), (r["lon"], r["lat"]), xytext=(5, 3), textcoords="offset points",
                    fontsize=7, fontweight="bold", color="#1b4d1e", zorder=6)
        key.append(f"{i + 1}  {r['label']}")
    if key:
        ax.text(1.02, 0.55, "Site key\n" + "\n".join(key), transform=ax.transAxes,
                fontsize=7, va="top", ha="left")
    ax.set_xlim(cfg.map_region[0], cfg.map_region[1]); ax.set_ylim(cfg.map_region[2], cfg.map_region[3])
    ax.set_aspect(1.0 / np.cos(np.radians(np.mean(cfg.map_region[2:]))))
    ax.set_xlabel("Longitude (deg E)"); ax.set_ylabel("Latitude (deg N)")
    ax.set_title("DRYAS - Tree-Ring Sensor Array and Earthquake Catalog (Japan)")
    ax.grid(True, linestyle=":", linewidth=0.5, color="#cccccc")
    handles = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2e7d32", markersize=9, label="Tree-ring site"),
        Line2D([0], [0], marker="o", color="w", markeredgecolor="#B5651D", markerfacecolor="none",
               markersize=10, label="Historical EQ (pre-instrumental)"),
        Line2D([0], [0], marker="o", color="w", markeredgecolor="#1f5fb0", markerfacecolor="none",
               markersize=10, label=f"Instrumental EQ M>={MAG_EMPH:g}"),
        Line2D([0], [0], marker=".", color="w", markerfacecolor="#1f5fb0", markersize=6,
               label=f"Instrumental EQ M{cfg.mmin:g}-{MAG_EMPH:g}"),
        Line2D([0], [0], marker="o", color="w", markeredgecolor="#555", markerfacecolor="none",
               markersize=7, label="(circle size \u221d magnitude)"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _hint_gmt_library_path():
    """
    Help PyGMT locate libgmt when `gmt --show-library` is unavailable (e.g. a
    broken Homebrew dependency) by pointing GMT_LIBRARY_PATH at common install
    directories. Cross-platform: macOS (Homebrew arm64/x86), Linux, Windows.
    """
    import os, glob
    if os.environ.get("GMT_LIBRARY_PATH"):
        return
    candidates = ["/opt/homebrew/lib", "/usr/local/lib", "/usr/lib/x86_64-linux-gnu",
                  "/usr/lib/aarch64-linux-gnu", "/usr/lib", "C:/programs/gmt6/bin",
                  "C:/Program Files/GMT6/bin"] + glob.glob("/opt/homebrew/Cellar/gmt/*/lib")
    for d in candidates:
        if glob.glob(os.path.join(d, "libgmt*.dylib")) or glob.glob(os.path.join(d, "libgmt*.so*")) \
                or glob.glob(os.path.join(d, "gmt*.dll")):
            os.environ["GMT_LIBRARY_PATH"] = d
            return


def fig_spatial(sites: pd.DataFrame, eq: pd.DataFrame, cfg: Config, out: Path):
    """PyGMT map when GMT is available (uv sync --extra gmt + system GMT), else matplotlib."""
    try:
        _hint_gmt_library_path()
        import pygmt  # noqa: F401
        fig_spatial_gmt(sites, eq, cfg, out)
        print("  [figure] 001a rendered with PyGMT")
    except Exception as e:  # ImportError or GMT library not found
        print(f"  [figure] PyGMT unavailable ({type(e).__name__}: {e}); using matplotlib fallback")
        fig_spatial_mpl(sites, eq, cfg, out)


def fig_temporal(rings: pd.DataFrame, eq: pd.DataFrame, cfg: Config, out: Path,
                 sites: pd.DataFrame | None = None):
    """
    Chronology spans (one row per site, labelled by site name) with per-year
    sample depth shown as line darkness, against the earthquake record.
    Instrumental events are only ticked at M >= MAG_LINE to keep the panel legible.
    """
    depth = (rings.groupby(["site_id", "year"])["tree_id"].nunique()
             .rename("n").reset_index())
    span = (depth.groupby("site_id")["year"].agg(["min", "max"])
            .sort_values("min").reset_index())
    names = {}
    if sites is not None and "site_name" in sites.columns:
        names = dict(zip(sites["site_id"], sites["site_name"].fillna(sites["site_id"])))
    fig, ax = plt.subplots(figsize=(11.0, min(0.5 * len(span) + 3.0, 14.0)), facecolor="white")
    ax.set_facecolor("white")
    cmap = plt.get_cmap("Greens")
    nmax = max(depth["n"].max(), 1)
    for i, r in span.iterrows():
        d = depth[depth["site_id"] == r["site_id"]].sort_values("year")
        # draw contiguous runs coloured by sample depth (log scale for legibility)
        yrs, n = d["year"].to_numpy(), d["n"].to_numpy()
        cols = cmap(0.25 + 0.75 * np.log1p(n) / np.log1p(nmax))
        ax.scatter(yrs, np.full_like(yrs, i), c=cols, s=6, marker="|", linewidths=1.6, zorder=2)
    for _, e in eq.iterrows():
        if e["era"] == "historical":
            ax.axvline(e["year"], color="#B5651D", linewidth=0.8, alpha=0.6, zorder=1)
        elif e["mag"] >= MAG_LINE:
            ax.axvline(e["year"], color="#1f5fb0", linewidth=0.6, alpha=0.35, zorder=1)
    ax.set_xlim(cfg.year_min, cfg.year_max)
    ax.set_ylim(-0.8, len(span) - 0.2)
    ax.set_yticks(range(len(span)))
    ax.set_yticklabels([f"{names.get(s, s)} [{s}]" for s in span["site_id"]], fontsize=8)
    ax.set_xlabel("Year (CE)")
    ax.set_title("DRYAS - Temporal Coverage of Chronologies vs Earthquake Record")
    ax.grid(True, axis="x", linestyle=":", linewidth=0.5, color="#cccccc")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    handles = [
        Line2D([0], [0], color=cmap(0.35), linewidth=3, label="Chronology (few trees)"),
        Line2D([0], [0], color=cmap(0.95), linewidth=3, label=f"Chronology (up to {nmax} trees)"),
        Line2D([0], [0], color="#B5651D", linewidth=1.5, label="Historical EQ"),
        Line2D([0], [0], color="#1f5fb0", linewidth=1.5, label=f"Instrumental EQ M>={MAG_LINE:g}"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def write_provenance(records: dict, cfg: Config, mode_used: str):
    """Write the 001 section; other scripts' sections in provenance.json are preserved."""
    p = DATA_DIR / "provenance.json"
    prov = {}
    if p.exists():
        try:
            prov = json.load(open(p))
        except Exception:
            prov = {}
    # legacy layout (top-level 001 keys) -> keep only script sections
    prov = {k: v for k, v in prov.items() if k.isdigit() or k.startswith("00")}
    prov["001"] = {
        "script": "001_data_acquisition.py",
        "project": "DRYAS",
        "generated_utc": _utc_now(),
        "mode_requested": cfg.mode,
        "mode_used": mode_used,
        "config": asdict(cfg),
        "artifacts": records,
    }
    with open(p, "w") as f:
        json.dump(prov, f, indent=2, default=str)
    print(f"  wrote {p}")


def save_table(df: pd.DataFrame, name: str, records: dict):
    p = DATA_DIR / name
    df.to_parquet(p, index=False)
    records[name] = {"rows": int(len(df)), "sha256": _sha256_of_file(p),
                     "path": str(p)}
    print(f"  wrote {p}  ({len(df):,} rows)")


def main():
    ap = argparse.ArgumentParser(description="DRYAS 001 — data acquisition")
    ap.add_argument("--mode", choices=["auto", "real", "synthetic"], default="auto")
    ap.add_argument("--n-sites", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = Config()
    cfg.mode = args.mode
    if args.n_sites is not None:
        cfg.n_sites = args.n_sites
    if args.seed is not None:
        cfg.seed = args.seed

    print(f"DRYAS 001 — data acquisition (mode={cfg.mode})")
    rings = sites = eq = None
    density = pd.DataFrame(columns=["site_id", "tree_id", "year", "density", "parameter"])
    mode_used = None

    def _do_real():
        print("[real] fetching ITRDB + USGS ...")
        r, s, d = fetch_itrdb_real(cfg)
        # Extend the catalog window back to the oldest ring actually acquired
        # (Yakushima reaches 1 CE) so pre-1500 historical events are retained.
        cfg.year_min = int(min(cfg.year_min, r["year"].min()))
        e = fetch_eq_real(cfg)
        return r, s, e, d

    def _do_synth():
        print("[synthetic] generating controlled ground-truth testbed ...")
        r, s, e = make_synthetic(cfg)
        return r, s, e, density

    if cfg.mode == "real":
        rings, sites, eq, density = _do_real(); mode_used = "real"
    elif cfg.mode == "synthetic":
        rings, sites, eq, density = _do_synth(); mode_used = "synthetic"
    else:  # auto
        try:
            rings, sites, eq, density = _do_real(); mode_used = "real"
        except Exception as e:
            print(f"[auto] real acquisition unavailable ({e}); "
                  f"falling back to synthetic.")
            cfg.year_min = Config.year_min   # restore the fixed synthetic window
            rings, sites, eq, density = _do_synth(); mode_used = "synthetic"

    # filter to magnitude / window
    eq = eq[(eq["mag"] >= cfg.mmin) & (eq["year"] >= cfg.year_min)
            & (eq["year"] <= cfg.year_max)].reset_index(drop=True)

    # persist
    records: dict = {}
    save_table(rings, "treering_network.parquet", records)
    save_table(sites, "site_metadata.parquet", records)
    save_table(eq, "eq_catalog.parquet", records)
    if len(density):
        save_table(density, "treering_density.parquet", records)
    write_provenance(records, cfg, mode_used)

    # figures
    print("rendering figures ...")
    fig_spatial(sites, eq, cfg, PDF_DIR / "001a_spatial_coverage.pdf")
    fig_temporal(rings, eq, cfg, PDF_DIR / "001b_temporal_coverage.pdf", sites=sites)
    print(f"  wrote {PDF_DIR/'001a_spatial_coverage.pdf'}")
    print(f"  wrote {PDF_DIR/'001b_temporal_coverage.pdf'}")

    # summary
    print("\n=== SUMMARY ===")
    print(f"  mode used        : {mode_used}")
    print(f"  tree sites       : {sites['site_id'].nunique()}")
    print(f"  tree series      : {rings['tree_id'].nunique():,}")
    print(f"  ring measurements: {len(rings):,}")
    print(f"  year range       : {rings['year'].min()}-{rings['year'].max()}")
    print(f"  earthquakes (M>={cfg.mmin}): {len(eq)} "
          f"({(eq['era']=='historical').sum()} hist / "
          f"{(eq['era']=='instrumental').sum()} instr)")
    if len(density):
        print(f"  density proxies  : {density['site_id'].nunique()} sites, "
              f"{len(density):,} measurements")
    if mode_used == "real" and sites["lat"].isna().any():
        print("  NOTE: some sites lack coordinates — see WARNING above.")
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
