# M1 Ultra ABBA results

A/B configurations are recorded in studies.json. Values below are full-request aggregate tok/s for server cells and standard llama-bench tok/s for pp/tg cells. Timing is complete after three quartets pass the spread gate; server results also require matching token IDs across and within arms for acceptance. Bench cells do not test generated output. Paired speedup is the geometric mean of the available quartet ratios; range is the observed quartet range, not a confidence interval.

| Study | Cell | Quartets | A tok/s | B tok/s | Paired gain | Range | Cross-arm mismatches | Within-arm mismatches | Rejected | Status |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| full-column-specialization | pp2 | 3/3 | 44.17 | 44.68 | +1.2% | 1.002-1.018x | 0 | 0 | 0 | accepted |
| full-column-specialization | pp3 | 3/3 | 48.88 | 50.26 | +2.8% | 1.016-1.040x | 0 | 0 | 0 | accepted |
| full-column-specialization | s1c1 | 3/3 | 35.14 | 35.39 | +0.7% | 1.005-1.012x | 0 | 0 | 0 | accepted |
| previous-pr-m1 | pp2 | 3/3 | 44.52 | 48.39 | +8.7% | 1.082-1.094x | 0 | 0 | 0 | accepted |
| previous-pr-m1 | pp3 | 3/3 | 50.55 | 52.33 | +3.5% | 1.012-1.047x | 0 | 0 | 0 | accepted |
| previous-pr-m1 | s0c1 | 3/3 | 27.38 | 29.71 | +8.5% | 1.078-1.093x | 0 | 0 | 0 | accepted |
| previous-pr-m1 | s1c1 | 3/3 | 35.42 | 35.53 | +0.4% | 0.994-1.019x | 0 | 0 | 0 | accepted |

## Single-user native generation

These means exclude prompt/request overhead. Only timing-complete cells with matching generated output are shown. Plain and MTP are separate cells; their ratio is descriptive, not a paired ABBA estimate. A/B means all flags off / M5-recommended flags under test. Bonsai 1 has no MTP head in this experiment.

| Model | Original plain | Original MTP | Optimized plain | Optimized MTP | MTP vs optimized plain |
|---|---:|---:|---:|---:|---:|
