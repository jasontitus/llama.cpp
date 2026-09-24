# M1 Ultra ABBA results

A/B configurations are recorded in studies.json. Values below are full-request aggregate tok/s for server cells and standard llama-bench tok/s for pp/tg cells. Timing is complete after three quartets pass the spread gate; server results also require matching token IDs across and within arms for acceptance. Bench cells do not test generated output. Paired speedup is the geometric mean of the available quartet ratios; range is the observed quartet range, not a confidence interval.

| Study | Cell | Quartets | A tok/s | B tok/s | Paired gain | Range | Cross-arm mismatches | Within-arm mismatches | Rejected | Status |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| previous-pr-mc-only | pp2 | 3/3 | 44.28 | 44.08 | -0.5% | 0.978-1.007x | 0 | 0 | 0 | accepted |
| previous-pr-mc-only | pp3 | 3/3 | 50.55 | 49.02 | -3.0% | 0.959-0.981x | 0 | 0 | 0 | accepted |
| previous-pr-mc-only | s1c1 | 3/3 | 35.60 | 35.38 | -0.6% | 0.990-0.996x | 0 | 0 | 0 | accepted |
| m1-no-stage-prefill | pp2 | 3/3 | 44.62 | 48.42 | +8.5% | 1.064-1.098x | 0 | 0 | 0 | accepted |
| m1-no-stage-prefill | pp3 | 3/3 | 50.27 | 51.02 | +1.5% | 1.002-1.032x | 0 | 0 | 0 | accepted |

## Single-user native generation

These means exclude prompt/request overhead. Only timing-complete cells with matching generated output are shown. Plain and MTP are separate cells; their ratio is descriptive, not a paired ABBA estimate. A/B means all flags off / M5-recommended flags under test. Bonsai 1 has no MTP head in this experiment.

| Model | Original plain | Original MTP | Optimized plain | Optimized MTP | MTP vs optimized plain |
|---|---:|---:|---:|---:|---:|
