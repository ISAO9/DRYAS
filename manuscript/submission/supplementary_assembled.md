# Supplementary Material

**Detection limits for far-field earthquake signals in tree-ring networks: a pre-registered injection–recovery test with the Japanese ITRDB array**

Isao Kurosawa (IVXA, Japan)

## Text S1. Pre-registration record

The record below is generated automatically by script 006 from the project files and lists each decision rule together with the point at which it was declared, the outcome and the evidence file. The corresponding rule numbers are cited in the main text (Section 2.9).



Generated: 2026-09-05T00:44:44.134377+00:00

### Declared rules and outcomes

| # | Rule (declared before the result) | Declared in | Outcome | Evidence |
|---|---|---|---|---|
| R1 | Standardization methods with median injected-signal preservation (A=0.35) < 0.70 are rejected; among survivors pick highest median SNR | 002 design (HANDOFF §7), before real run | all 5 methods passed; spline100 selected (SNR 1.08) | 002_selection.json (2026-09-04T14:18:13.857487+00:00) |
| R2 | Usable years for event detection = living-tree depth >= 10 (EPS >= 0.85 gate kept only as climate-quality flag) | 002 v2, after seeing EPS cut Yakushima | YAKU-M usable 644-2005 | chronologies.parquet |
| R3 | Tree-response prior: logistic in log10 PGA (x0=2.0, k=5.5) declared as placeholder to be re-fitted | 003 design | fitted x0 at bound (4.5), AUC 0.46 -> no relation | 004_response_calibration.json (2026-09-05T00:42:34.017458+00:00) |
| R4 | If AUC < 0.6 and Hokkaido SEA (incl. 1952) p > 0.05 after bug fixes, the project becomes a detection-limit study | HANDOFF §7d, before 004 rerun | condition met (AUC 0.46, all SEA p >= 0.15) | 004 SUMMARY, 004a/004b |
| R5 | Upper bound A_UB = (z_obs + 1.645)/slope with slope from injection at the measured f; array stack equal-weight | 005 design | reported in 005/006 | 005_upper_bounds_array.parquet (2026-09-05T00:43:48.636701+00:00) |
| R6 | Growing-season effective year (month >= 8 -> next ring) adopted as baseline before seeing its effect | 006 design (HANDOFF §7f) | see ablation 'timing' below | 006_ablations.parquet |

### Ablation summary (array bounds on site-mean growth loss A x f)

| event | baseline | geometry: +25 km | geometry: +50 km | geometry: -25 km | geometry: -50 km | method: negexp | method: spline67 | stack: PGA-weighted | timing: calendar year |
|---|---|---|---|---|---|---|---|---|---|
| 1909 BEPPU M7.5 | 0.036 | 0.036 | 0.036 | 0.036 | 0.036 | 0.040 | 0.048 | 0.053 | 0.000 |
| 1931 HYUGA M7.9 | 0.257 | 0.257 | 0.257 | 0.257 | 0.257 | 0.261 | 0.265 | 0.250 | 0.231 |
| 1941 HYUGA M8.0 | 0.093 | 0.093 | 0.093 | 0.093 | 0.093 | 0.116 | 0.105 | 0.095 | 0.049 |
| 1950 UNASSIGNED M7.7 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 1952 KT-TOK M8.1 | 0.071 | 0.071 | 0.027 | 0.071 | 0.071 | 0.074 | 0.070 | 0.070 | 0.071 |
| 1961 HYUGA M7.5 | 0.238 | 0.238 | 0.238 | 0.238 |  | 0.247 | 0.238 | 0.207 | 0.238 |
| 1968 KT-TOK M7.9 | 0.191 | 0.191 | 0.191 | 0.156 | 0.156 | 0.215 | 0.193 | 0.186 | 0.191 |
| 1968 HYUGA M7.5 | 0.230 | 0.230 | 0.230 | 0.230 |  | 0.220 | 0.225 | 0.229 | 0.230 |
| 1973 KT-NEM M7.7 | 0.152 | 0.152 | 0.152 | 0.178 | 0.178 | 0.164 | 0.153 | 0.174 | 0.152 |
| 1993 HIDAKA M7.6 | 0.028 | 0.028 | 0.028 | 0.028 | 0.028 | 0.034 | 0.025 | 0.026 | 0.028 |

### Key numbers

```
{
  "f_measured": 0.4666666666666667,
  "baseline_bounds_A": {
    "E1909_58": 0.0766,
    "E1931_289": 0.5502,
    "E1941_541": 0.1987,
    "E1950_643": 0.0,
    "E1952_674": 0.1518,
    "E1961_844": 0.5104,
    "E1968_1058": 0.4083,
    "E1968_1069": 0.4938,
    "E1973_1321": 0.3258,
    "E1993_2789": 0.0599
  },
  "baseline_bounds_Af": {
    "E1909_58": 0.0358,
    "E1931_289": 0.2568,
    "E1941_541": 0.0927,
    "E1950_643": 0.0,
    "E1952_674": 0.0708,
    "E1961_844": 0.2382,
    "E1968_1058": 0.1905,
    "E1968_1069": 0.2304,
    "E1973_1321": 0.152,
    "E1993_2789": 0.0279
  },
  "max_spread_across_variants_A": {
    "E1909_58": 0.1134,
    "E1931_289": 0.0728,
    "E1941_541": 0.1449,
    "E1950_643": 0.0,
    "E1952_674": 0.1023,
    "E1961_844": 0.0856,
    "E1968_1058": 0.1255,
    "E1968_1069": 0.0227,
    "E1973_1321": 0.0549,
    "E1993_2789": 0.0206
  },
  "A_for_f_grid": {
    "0.3": {
      "E1909_58": 0.11918515032671886,
      "E1931_289": 0.8559238753023339,
      "E1941_541": 0.3090990283548283,
      "E1950_643": 0.0,
      "E1952_674": 0.23611164490947978,
      "E1961_844": 0.7939711975801006,
      "E1968_1058": 0.6350991212690992,
      "E1968_1069": 0.7681301787076349,
      "E1973_1321": 0.5067525001048455,
      "E1993_2789": 0.09315490791465807
    },
    "0.45": {
      "E1909_58": 0.07945676688447924,
      "E1931_289": 0.5706159168682226,
      "E1941_541": 0.20606601890321882,
      "E1950_643": 0.0,
      "E1952_674": 0.1574077632729865,
      "E1961_844": 0.529314131720067,
      "E1968_1058": 0.42339941417939947,
      "E1968_1069": 0.5120867858050898,
      "E1973_1321": 0.33783500006989703,
      "E1993_2789": 0.06210327194310537
    },
    "0.6": {
      "E1909_58": 0.05959257516335943,
      "E1931_289": 0.42796193765116697,
      "E1941_541": 0.15454951417741414,
      "E1950_643": 0.0,
      "E1952_674": 0.11805582245473989,
      "E1961_844": 0.3969855987900503,
      "E1968_1058": 0.3175495606345496,
      "E1968_1069": 0.38406508935381745,
      "E1973_1321": 0.25337625005242276,
      "E1993_2789": 0.04657745395732903
    },
    "1.0": {
      "E1909_58": 0.03575554509801566,
      "E1931_289": 0.25677716259070016,
      "E1941_541": 0.09272970850644847,
      "E1950_643": 0.0,
      "E1952_674": 0.07083349347284393,
      "E1961_844": 0.23819135927403018,
      "E1968_1058": 0.19052973638072976,
      "E1968_1069": 0.23043905361229045,
      "E1973_1321": 0.15202575003145366,
      "E1993_2789": 0.02794647237439742
    }
  },
  "timing_changed_events": [
    "E1909_58",
    "E1941_541",
    "E1931_289"
  ]
}
```

## Text S2. Synthetic positive controls

Every statistic reported in the main text was first run on a synthetic testbed generated by script 001 (`--mode synthetic`): 60 sites and 770 series (1501–2026) with ring width = age trend × (1 + 0.18·sensitivity·AR(1) climate) × (1 − earthquake suppression) + noise, in which the 29 catalogued M ≥ 5 earthquakes of the bundled catalogue are embedded as suppressions of magnitude-dependent amplitude, decaying over five years, within an 80 km response radius. On this testbed, with the same code and thresholds as on the real data:

- the injection–recovery test (002) recovers preservation 0.94–1.06 for the five standardisation methods and selects a method by the same rule;
- the ground-motion prior has skill (004: AUC 0.93, fitted logistic midpoint 503 gal reflecting the embedded response rule), and the per-site SEA is significant at sites with several embedded events;
- the array upper bounds (005) are large where events were embedded and near zero where they were not (e.g. 1946 and 2011, which lie outside the embedded response radius of the synthetic sites);
- the pooled double-bootstrap SEA, the responsive-tree fractions under the 30 % and 40 % criteria and the prior-weighted primary statistic (007) are all significant against the year-shift null (*p* < 0.001 for SEA and the primary statistic, *z*_w = 10.7).

These runs establish that a signal of the assumed shape, had it been present in the real data at the amplitudes considered, would have been detected by the same pipeline. Exact values for each run are printed by the scripts and archived with the code (Zenodo, https://doi.org/10.5281/zenodo.22349501).

## Supplementary figures

- **Fig. S1.** Temporal coverage of the chronologies (site rows, sample depth as line darkness) against the historical and instrumental (M ≥ 7) earthquake record. (PDF/001b_temporal_coverage.pdf)
- **Fig. S2.** Site chronologies for the selected standardisation with sample depth, climate-quality and detection-usable years, and nearby M ≥ 7 events. (PDF/002a_chronologies_selected.pdf)
- **Fig. S3.** Running EPS per site and the two usable-year gates. (PDF/002c_eps_sample_depth.pdf)
- **Fig. S4.** Worked example of one injection trial (clean versus injected chronology and the recovered profile). (PDF/002d_injection_example.pdf)
- **Fig. S5.** Prior expected tree-response probability per site and M ≥ 7.5 event. (PDF/003b_expected_response_matrix.pdf)
- **Fig. S6.** Ground-motion prior: PGA and response probability versus fault distance. (PDF/003c_attenuation_curves.pdf)
- **Fig. S7.** Superposed epoch analysis of maximum latewood density per site. (PDF/005c_mxd_sea.pdf)
- **Fig. S8.** Array upper bounds under all ablation variants. (PDF/006b_ablation_summary.pdf)
- **Fig. S9.** Sensitivity of the array bounds to source-rectangle perturbation. (PDF/006c_geometry_sensitivity.pdf)
- **Fig. S10.** Pooled double-bootstrap SEA by subset, distance bin and trench. (PDF/007b_pooled_sea_double_bootstrap.pdf)
- **Fig. S11.** Lag-tolerant statistic and Lloret resilience indices at event versus random years. (PDF/007c_lag_tolerant_and_resilience.pdf)
