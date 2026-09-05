# DRYAS

Started as a multi-century earthquake-cascade reconstruction from a tree-ring sensor array. After a pre-registered calibration test (004) found no far-field earthquake response in the ITRDB Japan array, the project became a quantitative detection-limit study: how much growth suppression great subduction earthquakes could have caused unseen at 100-250 km, and what array would be needed to see it.

## Pipeline

| # | script | status |
|---|---|---|
| 001 | data acquisition (13 real ITRDB Japan sites incl. Yakushima 1-1999 CE, 8 density proxies, Japanese historical catalogue from 684 CE) | done |
| 002 | chronology building: 5 detrending methods, EPS gating, injection-recovery test of suppression-signal preservation, automatic method selection | done (synthetic-verified) |
| 003 | spatial graph, fault-segment geometry (GeoJSON), PGA prior (Si & Midorikawa 1999), catalog-to-segment assignment | done (synthetic-verified) |
| 004 | SEA, per-event response, responding fraction, empirical calibration of the PGA prior, MXD alignment, feature table | done (synthetic-verified) |
| 005 | detection limits: injection power curves, amplitude upper bounds (site & array), MXD test, array design | done (synthetic-verified) |
| 006 | growing-season timing, five ablations, headline figure in A x f units, pre-registration record | done (synthetic-verified) |
| 007 | replication of published far-field criteria (responsive-tree fractions, Rao-2019 double-bootstrap SEA, lag-tolerant stats, Lloret resilience) with random-year nulls | done (synthetic positive control) |
| 008 | manuscript (Dendrochronologia primary); frozen modules (GNN separation, cascade, forecast test) kept as future work for a near-fault array | next |

Vocabulary rule: the project produces long-term conditional probabilities (*forecast*), never *prediction*.

## Setup

```bash
uv sync            # base environment
uv sync --extra ml # add torch for the separation model (script 005+)
```

## Run

```bash
uv run python src/001_data_acquisition.py            # auto mode
uv run python src/001_data_acquisition.py --mode real # real ITRDB (NOAA template headers) + USGS; downloads cached in data/raw/
```

## Layout

- `src/`    numbered pipeline scripts
- `data/`   acquired / harmonised data (parquet) + provenance.json
- `models/` best model checkpoints
- `PDF/`    all figures (white background, English, legends in margin)
- `HANDOFF.md` handoff notes for continuing in a new chat
