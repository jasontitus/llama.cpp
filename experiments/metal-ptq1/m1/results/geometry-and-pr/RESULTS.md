# M1 Ultra ABBA results

A/B configurations are recorded in studies.json. Values below are full-request aggregate tok/s for server cells and standard llama-bench tok/s for pp/tg cells. Timing is complete after three quartets pass the spread gate; server results also require matching token IDs across and within arms for acceptance. Bench cells do not test generated output. Paired speedup is the geometric mean of the available quartet ratios; range is the observed quartet range, not a confidence interval.

| Study | Cell | Quartets | A tok/s | B tok/s | Paired gain | Range | Cross-arm mismatches | Within-arm mismatches | Rejected | Status |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| m1-geometry | pp2 | 3/3 | 38.96 | 47.93 | +23.0% | 1.211-1.246x | 0 | 0 | 0 | accepted |
| m1-geometry | pp3 | 3/3 | 39.01 | 49.03 | +25.7% | 1.236-1.268x | 0 | 0 | 0 | accepted |
| m1-geometry | s1c1 | 3/3 | 27.33 | 34.76 | +27.2% | 1.256-1.291x | 0 | 0 | 0 | accepted |
| previous-pr | pp2 | 3/3 | 44.02 | 48.54 | +10.3% | 1.092-1.117x | 0 | 0 | 0 | accepted |
| previous-pr | pp3 | 3/3 | 50.42 | 49.10 | -2.6% | 0.959-0.988x | 0 | 0 | 0 | accepted |
| previous-pr | s0c1 | 3/3 | 27.42 | 29.48 | +7.5% | 1.074-1.076x | 0 | 0 | 0 | accepted |
| previous-pr | s1c1 | 3/3 | 35.45 | 34.71 | -2.1% | 0.962-0.995x | 0 | 0 | 0 | accepted |
| headline-ptq1-same-mode | tg128 | 0/3 | - | - | - | - | - | - | - | incomplete |
| headline-ptq1-same-mode | s0c1 | 0/3 | - | - | - | - | - | - | - | incomplete |
| headline-ptq1-same-mode | s1c1 | 0/3 | - | - | - | - | - | - | - | incomplete |
| headline-ptq1-total | s1c1 | 0/3 | - | - | - | - | - | - | - | incomplete |

## Single-user native generation

These means exclude prompt/request overhead. Only timing-complete cells with matching generated output are shown. Plain and MTP are separate cells; their ratio is descriptive, not a paired ABBA estimate. A/B means all flags off / M5-recommended flags under test. Bonsai 1 has no MTP head in this experiment.

| Model | Original plain | Original MTP | Optimized plain | Optimized MTP | MTP vs optimized plain |
|---|---:|---:|---:|---:|---:|
