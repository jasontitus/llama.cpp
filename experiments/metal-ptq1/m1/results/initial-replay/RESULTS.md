# M1 Ultra ABBA results

A/B configurations are recorded in studies.json. Values below are full-request aggregate tok/s for server cells and standard llama-bench tok/s for pp/tg cells. Timing is complete after three quartets pass the spread gate; server results also require matching token IDs across and within arms for acceptance. Bench cells do not test generated output. Paired speedup is the geometric mean of the available quartet ratios; range is the observed quartet range, not a confidence interval.

| Study | Cell | Quartets | A tok/s | B tok/s | Paired gain | Range | Cross-arm mismatches | Within-arm mismatches | Rejected | Status |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| headline-ptq1-same-mode | tg128 | 3/3 | 29.74 | 32.21 | +8.3% | 1.078-1.087x | 0 | 0 | 0 | accepted |
| headline-ptq1-same-mode | s0c1 | 3/3 | 27.26 | 29.72 | +9.0% | 1.082-1.100x | 0 | 0 | 0 | accepted |
| headline-ptq1-same-mode | s1c1 | 3/3 | 13.33 | 27.44 | +105.9% | 2.056-2.065x | 0 | 0 | 0 | accepted |
| headline-ptq1-total | s1c1 | 3/3 | 27.19 | 27.53 | +1.1% | 0.941-1.067x | 0 | 0 | 0 | accepted |
| headline-pq2-total | s1c1 | 3/3 | 30.02 | 27.19 | -9.6% | 0.843-0.944x | 0 | 0 | 0 | accepted |
| abba3-b2-pq2 | tg128 | 1/3 | 32.28 | 35.92 | +11.3% | 1.113-1.113x | 0 | 0 | 0 | incomplete |
| abba3-b2-pq2 | pp2 | 1/3 | 35.14 | 37.05 | +5.4% | 1.054-1.054x | 0 | 0 | 0 | incomplete |
| abba3-b2-pq2 | pp3 | 1/3 | 44.52 | 46.63 | +4.7% | 1.047-1.047x | 0 | 0 | 0 | incomplete |
| abba3-b2-pq2 | pp32 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba3-b2-pq2 | pp512 | 1/3 | 283.63 | 289.30 | +2.0% | 1.020-1.020x | 0 | 0 | 0 | incomplete |
| abba3-b2-pq2 | s0c1 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba3-b2-pq2 | s1c1 | 1/3 | 29.57 | 28.64 | -3.2% | 0.968-0.968x | 0 | 0 | 0 | incomplete |
| abba3-b2-pq2 | s0c2 | 1/3 | 31.44 | 32.66 | +3.9% | 1.039-1.039x | 0 | 0 | 0 | incomplete |
| abba3-b2-pq2 | s1c2 | 1/3 | 32.96 | 32.66 | -0.9% | 0.991-0.991x | 0 | 0 | 0 | incomplete |
| abba3-b1-ternary | tg128 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba3-b1-ternary | pp2 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba3-b1-ternary | pp32 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba3-b1-ternary | pp512 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba3-b1-ternary | s0c1 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba3-b1-ternary | s0c2 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba3-b1-binary | tg128 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba3-b1-binary | pp2 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba3-b1-binary | pp32 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba3-b1-binary | pp512 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba3-b1-binary | s0c1 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba3-b1-binary | s0c2 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba3-b2-ptq1-smallm | pp16 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba3-b2-ptq1-smallm | pp32 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba3-b2-ptq1-smallm | pp128 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba3-b2-ptq1-smallm | pp512 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba3-b2-ptq1-smallm | s0c1 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba3-b2-ptq1-smallm | s1c1 | 0/3 | - | - | - | - | - | - | - | incomplete |
| bitexact-ptq1 | tg128 | 0/3 | - | - | - | - | - | - | - | incomplete |
| bitexact-ptq1 | s0c1 | 0/3 | - | - | - | - | - | - | - | incomplete |
| bitexact-pq2 | tg128 | 0/3 | - | - | - | - | - | - | - | incomplete |
| bitexact-pq2 | s0c1 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba5-invariant-cost | tg128 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba5-invariant-cost | pp2 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba5-invariant-cost | pp4 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba5-invariant-cost | pp8 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba5-invariant-cost | s0c1 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba5-invariant-cost | s1c1 | 0/3 | - | - | - | - | - | - | - | incomplete |
| abba5-invariant-cost | s1c2 | 0/3 | - | - | - | - | - | - | - | incomplete |

## Single-user native generation

These means exclude prompt/request overhead. Only timing-complete cells with matching generated output are shown. Plain and MTP are separate cells; their ratio is descriptive, not a paired ABBA estimate. A/B means all flags off / M5-recommended flags under test. Bonsai 1 has no MTP head in this experiment.

| Model | Original plain | Original MTP | Optimized plain | Optimized MTP | MTP vs optimized plain |
|---|---:|---:|---:|---:|---:|
| Bonsai 2 PTQ1_0 | 29.57 | 13.80 | 32.12 | 29.49 | -8.2% |
