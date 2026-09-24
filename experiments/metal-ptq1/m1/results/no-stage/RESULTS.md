# M1 Ultra ABBA results

A/B configurations are recorded in studies.json. Values below are full-request aggregate tok/s for server cells and standard llama-bench tok/s for pp/tg cells. Timing is complete after three quartets pass the spread gate; server results also require matching token IDs across and within arms for acceptance. Bench cells do not test generated output. Paired speedup is the geometric mean of the available quartet ratios; range is the observed quartet range, not a confidence interval.

| Study | Cell | Quartets | A tok/s | B tok/s | Paired gain | Range | Cross-arm mismatches | Within-arm mismatches | Rejected | Status |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| previous-pr-no-stage | s1c1 | 3/3 | 35.60 | 35.29 | -0.9% | 0.985-1.000x | 0 | 0 | 0 | accepted |

## Single-user native generation

These means exclude prompt/request overhead. Only timing-complete cells with matching generated output are shown. Plain and MTP are separate cells; their ratio is descriptive, not a paired ABBA estimate. A/B means all flags off / M5-recommended flags under test. Bonsai 1 has no MTP head in this experiment.

| Model | Original plain | Original MTP | Optimized plain | Optimized MTP | MTP vs optimized plain |
|---|---:|---:|---:|---:|---:|
