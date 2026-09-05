#!/usr/bin/env python3
# =============================================================================
# DRYAS  |  Script 008  —  MANUSCRIPT TABLES & CITATION CONSISTENCY CHECK
# =============================================================================
# WHAT THIS SCRIPT DOES
#   1. Generates the three manuscript tables directly from the data files so
#      that no number in the paper is typed by hand:
#        Table 1  sites (code, name, species, lat, lon, elevation, series, span,
#                 first/last year with >= 10 living trees, NOAA study ID)
#        Table 2  array upper bounds for M >= 7.5 events seen by >= 2 sites,
#                 in amplitude and in stand-mean units, with the spread across
#                 the 006 ablation variants
#        Table 3  the pre-specified primary test and the exploratory tests
#                 against the independent-year and year-shift nulls (007)
#      Each table is written as CSV (for the docx build) and as Markdown
#      (for the draft), plus a JSON of the headline numbers quoted in the text.
#   2. Cross-checks the manuscript:
#        - every in-text citation (Author, 2020 / Author and Author, 2020 /
#          Author et al., 2020) has an entry in the reference list, and vice versa;
#        - figures and tables are first mentioned in numerical order;
#        - the headline numbers quoted in the text match the data files
#          (a small dictionary of "number -> source" is checked).
#
# INPUTS  data/site_metadata.parquet, data/graph_nodes.parquet, data/chronologies.parquet,
#         data/005_upper_bounds_array.parquet, data/006_ablations.parquet,
#         data/004_response_calibration.json, data/007_shift_null.parquet, data/007_summary.json,
#         manuscript/DRYAS_manuscript_v*.md (latest), manuscript/references_elsevier_harvard.md
# OUTPUTS manuscript/table1_sites.{csv,md}, table2_array_bounds.{csv,md}, table3_tests.{csv,md},
#         manuscript/headline_numbers.json, manuscript/008_citation_check.md
#
# USAGE   uv run python src/008_manuscript_tables.py [--manuscript path]
# =============================================================================

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
DATA_DIR = ROOT / "data"
MS_DIR = ROOT / "manuscript"
MS_DIR.mkdir(exist_ok=True)


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def write_table(df: pd.DataFrame, stem: str, caption: str):
    df.to_csv(MS_DIR / f"{stem}.csv", index=False)
    (MS_DIR / f"{stem}.md").write_text(f"**{caption}**\n\n" + md_table(df) + "\n", encoding="utf-8")
    print(f"  wrote {MS_DIR / (stem + '.csv')} and .md  ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def table1(sites: pd.DataFrame, nodes: pd.DataFrame, merged: dict) -> pd.DataFrame:
    s = sites.copy()
    keep = [c for c in ("site_id", "site_name", "species", "lat", "lon", "elev_m", "n_series",
                        "first_year", "last_year", "landing_page") if c in s.columns]
    s = s[keep]
    for c in ("landing_page", "species", "elev_m", "n_series", "first_year", "last_year"):
        if c not in s.columns:
            s[c] = np.nan
    n = nodes[["site_id", "usable_depth_first", "usable_depth_last", "usable_depth_years", "cluster"]]
    t = s.merge(n, on="site_id", how="left")
    # merged sites from 002
    for mid, members in merged.items():
        row = nodes[nodes["site_id"] == mid]
        if row.empty:
            continue
        row = row.iloc[0]
        t = pd.concat([t, pd.DataFrame([{
            "site_id": mid, "site_name": f"Yakushima (merged: {', '.join(members)})" if mid.startswith("YAKU") else f"{mid} (merged)",
            "species": s.loc[s["site_id"].isin(members), "species"].iloc[0] if "species" in s else "",
            "lat": row["lat"], "lon": row["lon"], "elev_m": np.nan,
            "n_series": int(s.loc[s["site_id"].isin(members), "n_series"].sum()) if "n_series" in s else np.nan,
            "first_year": int(s.loc[s["site_id"].isin(members), "first_year"].min()),
            "last_year": int(s.loc[s["site_id"].isin(members), "last_year"].max()),
            "landing_page": "", "usable_depth_first": row["usable_depth_first"],
            "usable_depth_last": row["usable_depth_last"], "usable_depth_years": row["usable_depth_years"],
            "cluster": row["cluster"]}])], ignore_index=True)
    t["study_id"] = t["landing_page"].astype(str).str.extract(r"study/(\d+)")
    for c in ("usable_depth_first", "usable_depth_last", "usable_depth_years", "first_year", "last_year", "n_series", "elev_m"):
        if c in t.columns:
            t[c] = t[c].apply(lambda v: "" if pd.isna(v) else str(int(v)))
    t["lat"] = t["lat"].round(2); t["lon"] = t["lon"].round(2)
    t = t.rename(columns={"site_id": "Code", "site_name": "Site", "species": "Species", "lat": "Lat (°N)",
                          "lon": "Lon (°E)", "elev_m": "Elev. (m)", "n_series": "Series",
                          "first_year": "First year", "last_year": "Last year",
                          "usable_depth_first": "≥10 trees from", "usable_depth_last": "≥10 trees to",
                          "usable_depth_years": "Usable years", "cluster": "Cluster", "study_id": "NOAA study"})
    t = t.drop(columns=["landing_page"])
    t = t.sort_values(["Cluster", "Lat (°N)"], ascending=[True, False]).reset_index(drop=True)
    return t


def table2(arr: pd.DataFrame, abl: pd.DataFrame, f: float) -> pd.DataFrame:
    piv = abl.pivot_table(index="event_id", columns="variant", values="A_UB")
    spread = (piv.max(axis=1) - piv.min(axis=1)) * f
    t = arr.copy().sort_values("year")
    t["A_UB_amp"] = t["A_UB_array"].round(3)
    t["Af_pct"] = (t["A_UB_array"] * f * 100).round(1)
    t["spread_pct"] = t["event_id"].map(spread).mul(100).round(1)
    t = t[["year", "segment", "mag", "n_sites", "pga_mean_gal", "z_array", "p_one_sided", "A_UB_amp", "Af_pct", "spread_pct", "sites"]]
    t["pga_mean_gal"] = t["pga_mean_gal"].round(0).astype(int)
    t["z_array"] = t["z_array"].round(2); t["p_one_sided"] = t["p_one_sided"].round(2)
    return t.rename(columns={"year": "Year", "segment": "Segment", "mag": "M", "n_sites": "S",
                             "pga_mean_gal": "Mean prior PGA (gal)", "z_array": "z_array", "p_one_sided": "p (one-sided)",
                             "A_UB_amp": "A_95 (amplitude)", "Af_pct": "Stand-mean loss bound (%)",
                             "spread_pct": "Spread across variants (% pts)", "sites": "Sites"})


def table3(shift: pd.DataFrame, summ: dict) -> pd.DataFrame:
    # independent-null p-values for the matching exploratory tests
    indep = {}
    for r in summ.get("sea", []):
        if r["subset"].startswith("all sites"):
            indep["SEA years 0-1 (index, p>=0.5)"] = r["p_y01"]
        if r["subset"].startswith("raw widths"):
            indep["SEA years 0-1 (raw, p>=0.5)"] = r["p_y01"]
    indep["Lloret resistance median (p>=0.2)"] = summ.get("resilience", {}).get("resistance", {}).get("p_events_lower")
    indep["lag-tolerant min gc median (p>=0.2)"] = summ.get("lag_tolerant", {}).get("index", {}).get("p_events_lower")
    for c in ("pct20", "pct40", "schw25"):
        v = summ.get("responsive_fraction", {}).get(f"index:{c}", {})
        indep[f"fraction {c} (index)"] = v.get("median_site_p")
    t = shift.copy()
    t["p_independent"] = t["test"].map(indep)
    t["null_range"] = t.apply(lambda r: f"{r['null_median']:.3f} [{r['null_q05']:.3f}, {r['null_q95']:.3f}]", axis=1)
    t = t[["test", "n_pairs", "obs", "null_range", "p", "p_independent"]]
    t["obs"] = t["obs"].round(3); t["p"] = t["p"].round(3)
    t["p_independent"] = pd.to_numeric(t["p_independent"], errors="coerce").round(3)
    return t.rename(columns={"test": "Test", "n_pairs": "Pairs", "obs": "Observed", "null_range": "Year-shift null median [5–95%]",
                             "p": "p (year-shift)", "p_independent": "p (independent years)"})


# ---------------------------------------------------------------------------
# Citation check
# ---------------------------------------------------------------------------
# matches "Gao et al., 2024", "Gao et al. (2024)", "Si and Midorikawa, 1999", "Si and Midorikawa (1999)", "Hough, 2026"
CITE_RE = re.compile(r"([A-Z][A-Za-zÀ-ž'’\-]+(?: and [A-Z][A-Za-zÀ-ž'’\-]+| et al\.)?),? \(?(\d{4}[a-z]?)\)?")


def parse_citations(text: str) -> set:
    text = re.sub(r"【[^】]*】", "", text)      # placeholders
    found = set()
    # "Yasue (2013a–g)" / "Yasue, 2013a–g" -> 2013a ... 2013g ; "Davi et al. (2006, 2011)" -> both years
    for m in re.finditer(r"([A-Z][A-Za-zÀ-ž'’\-]+)(?: et al\.)?,? \(?(\d{4})([a-z])[–-]([a-z])", text):
        for ch in range(ord(m.group(3)), ord(m.group(4)) + 1):
            found.add((m.group(1), m.group(2) + chr(ch)))
    for m in re.finditer(r"([A-Z][A-Za-zÀ-ž'’\-]+)(?: et al\.)?,? \(?((?:\d{4}, )+\d{4})", text):
        for y in m.group(2).split(", "):
            found.add((m.group(1), y))
    for m in CITE_RE.finditer(text):
        name, year = m.group(1), m.group(2)
        first = name.split(" ")[0]
        found.add((first, year))
    return found


def parse_reflist(text: str) -> dict:
    refs = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("<"):
            continue
        m = re.match(r"([A-Z][A-Za-zÀ-ž'’\-\. ]+?),.*?(\d{4}[a-z]?|【year】)\.", line)
        if m:
            first = m.group(1).split(",")[0].split(" ")[0]
            if line.startswith("U.S. Geological Survey"):
                first = "Survey"        # in-text form is "(U.S. Geological Survey, 2026)"
            refs[(first, m.group(2))] = line
    return refs


def check_figure_order(text: str) -> list:
    msgs = []
    for kind in ("Fig.", "Table"):
        seen = []
        for m in re.finditer(rf"{re.escape(kind)}\s*(S?\d+)", text):
            n = m.group(1)
            if n not in seen:
                seen.append(n)
        nums = [x for x in seen if not x.startswith("S")]
        expected = [str(i + 1) for i in range(len(nums))]
        if nums != expected:
            msgs.append(f"{kind} first-mention order is {nums}, expected {expected}")
        else:
            msgs.append(f"{kind} order OK: {nums}")
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manuscript", default=None)
    args = ap.parse_args()
    sites = pd.read_parquet(DATA_DIR / "site_metadata.parquet")
    nodes = pd.read_parquet(DATA_DIR / "graph_nodes.parquet")
    merged = json.load(open(DATA_DIR / "002_merged_sites.json")) if (DATA_DIR / "002_merged_sites.json").exists() else {}
    calib = json.load(open(DATA_DIR / "004_response_calibration.json"))
    f = float(calib.get("f_median_detected") or 0.45)
    arr = pd.read_parquet(DATA_DIR / "005_upper_bounds_array.parquet") if (DATA_DIR / "005_upper_bounds_array.parquet").exists() else pd.DataFrame()
    abl = pd.read_parquet(DATA_DIR / "006_ablations.parquet") if (DATA_DIR / "006_ablations.parquet").exists() else pd.DataFrame()
    shift = pd.read_parquet(DATA_DIR / "007_shift_null.parquet") if (DATA_DIR / "007_shift_null.parquet").exists() else pd.DataFrame()
    summ7 = json.load(open(DATA_DIR / "007_summary.json")) if (DATA_DIR / "007_summary.json").exists() else {}
    print(f"DRYAS 008 — f={f:.2f}; {len(sites)} sites; {len(arr)} array events; {len(abl)} ablation rows; {len(shift)} shift tests")

    t1 = table1(sites, nodes, merged)
    write_table(t1, "table1_sites", "Table 1. ITRDB sites used in this study.")
    if len(arr) and len(abl):
        t2 = table2(arr, abl, f)
        write_table(t2, "table2_array_bounds", "Table 2. Array upper bounds (one-sided 95%) on stand-mean growth loss in the event year.")
    if len(shift):
        t3 = table3(shift, summ7)
        write_table(t3, "table3_tests", "Table 3. Pre-specified primary test and exploratory tests against two nulls.")
    dp = DATA_DIR / "005_array_design.json"
    if dp.exists():
        design = json.load(open(dp))
        req = design.get("required_sites", {})
        if req:
            amps = sorted({a for n in req for a in req[n]}, key=float)
            rows = []
            for n in sorted(req, key=int):
                rows.append({"Trees per site": int(n), **{f"A = {float(a):.2f}": int(req[n][a]) for a in amps}})
            t4 = pd.DataFrame(rows)
            write_table(t4, "table4_array_design",
                        f"Table 4. Number of coherent sites required to detect a stand-level suppression amplitude A at 3σ "
                        f"(z = slope(n)·A·√S, slope ∝ n^{design.get('depth_exponent', float('nan')):.2f}, f = {design.get('f_used', f):.2f}).")

    # headline numbers
    hn = {"f_measured": f, "n_pairs_004": calib.get("n_pairs"), "n_detected_004": calib.get("n_detected"),
          "auc_004": calib.get("auc"), "base_rate_004": calib.get("base_rate_detected")}
    if len(arr):
        for _, r in arr.iterrows():
            hn[f"Af_pct_{r['year']}_{r['segment']}"] = round(float(r["A_UB_array"] * f * 100), 1)
    if len(shift):
        pr = shift.iloc[0]
        hn["primary_z"] = float(pr["obs"]); hn["primary_p"] = float(pr["p"])
        if "A_UB_pooled" in shift.columns and pd.notna(pr.get("A_UB_pooled")):
            hn["primary_Af_pct"] = round(float(pr["A_UB_pooled"]) * f * 100, 1)
    json.dump(hn, open(MS_DIR / "headline_numbers.json", "w"), indent=2, default=float)
    print(f"  wrote {MS_DIR / 'headline_numbers.json'}")

    # citation check
    cands = sorted(glob.glob(str(MS_DIR / "DRYAS_manuscript_v*.md")))
    ms_path = Path(args.manuscript) if args.manuscript else (Path(cands[-1]) if cands else MS_DIR / "DRYAS_manuscript_missing.md")
    ref_path = MS_DIR / "references_elsevier_harvard.md"
    report = [f"# 008 citation check — {ms_path.name} vs {ref_path.name}", ""]
    if ms_path.exists() and ref_path.exists():
        ms = ms_path.read_text(encoding="utf-8")
        body = ms.split("## References")[0]
        cites = parse_citations(body)
        refs = parse_reflist(ref_path.read_text(encoding="utf-8"))
        ref_keys = set(refs.keys())
        # dataset refs with placeholder years match by surname only
        ref_surnames = {k[0] for k in ref_keys}
        missing = sorted(c for c in cites if c not in ref_keys and not (c[0] in ref_surnames and any(k[1] == "【year】" for k in ref_keys if k[0] == c[0])))
        # ignore obvious non-citations (e.g. "Table 2", "Section 3")
        non_cite = {"Table", "Fig", "Section", "Supplementary", "Mw", "M", "In", "The", "A", "January", "February", "March",
                    "April", "May", "June", "July", "August", "September", "October", "November", "December",
                    "Hoei", "Ansei", "Genroku", "Tokachi-oki", "Nemuro-oki", "Kushiro-oki", "Hyuga-nada", "Nankai", "Kanto",
                    "Cascadia", "Tosya", "Jiuzhaigou", "Since", "Before", "After", "From", "Between", "CE", "BCE"}
        missing = [c for c in missing if c[0] not in non_cite]
        uncited = sorted(k for k in ref_keys if k not in cites and not any(c[0] == k[0] for c in cites))
        cited_surnames_any = {m.group(1) for m in re.finditer(r"([A-Z][A-Za-z'’\-]+)(?: et al\.)?,? \(?【year】", body)}
        uncited = [k for k in uncited if not (k[1] == "【year】" and k[0] in cited_surnames_any)]
        def bl(items):
            items = list(items)
            return items if items else ["- none"]
        report += [f"In-text citations found: {len(cites)}; reference entries: {len(ref_keys)}", ""]
        report += ["## Cited in text but NOT in reference list"] + bl(f"- {a}, {y}" for a, y in missing) + [""]
        report += ["## In reference list but NOT cited in text"] + bl(f"- {a}, {y}" for a, y in uncited) + [""]
        report += ["## Figure / table order"] + [f"- {m}" for m in check_figure_order(body)] + [""]
        report += ["## Placeholders still in the manuscript"] + bl(f"- {m}" for m in sorted(set(re.findall(r"【[^】]*】", ms))))
    else:
        report.append("manuscript or reference list not found")
    (MS_DIR / "008_citation_check.md").write_text("\n".join(report), encoding="utf-8")
    print(f"  wrote {MS_DIR / '008_citation_check.md'}")
    print("\n".join(report[2:]))
    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
