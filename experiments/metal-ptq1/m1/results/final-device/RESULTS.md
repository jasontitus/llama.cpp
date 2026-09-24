# M1 Ultra ABBA results

A/B configurations are recorded in studies.json. Values below are full-request aggregate tok/s for server cells and standard llama-bench tok/s for pp/tg cells. Timing is complete after three quartets pass the spread gate; server results also require matching token IDs across and within arms for acceptance. Bench cells do not test generated output. Paired speedup is the geometric mean of the available quartet ratios; range is the observed quartet range, not a confidence interval.

| Study | Cell | Quartets | A tok/s | B tok/s | Paired gain | Range | Cross-arm mismatches | Within-arm mismatches | Rejected | Status |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| ptq1-same-mode | tg128 | 3/3 | 29.67 | 32.29 | +8.8% | 1.086-1.090x | 0 | 0 | 0 | accepted |
| ptq1-same-mode | pp2 | 3/3 | 16.32 | 48.44 | +196.7% | 2.965-2.970x | 0 | 0 | 0 | accepted |
| ptq1-same-mode | pp4 | 3/3 | 26.12 | 39.54 | +51.4% | 1.497-1.524x | 0 | 0 | 0 | accepted |
| ptq1-same-mode | pp8 | 3/3 | 30.00 | 43.73 | +45.8% | 1.451-1.463x | 0 | 0 | 0 | accepted |
| ptq1-same-mode | s0c1 | 3/3 | 27.04 | 29.56 | +9.3% | 1.087-1.105x | 0 | 0 | 0 | accepted |
| ptq1-same-mode | s1c1 | 3/3 | 13.26 | 34.59 | +160.9% | 2.576-2.650x | 0 | 0 | 0 | accepted |
| ptq1-same-mode | s0c2 | 3/3 | 15.26 | 41.68 | +173.1% | 2.694-2.756x | 0 | 0 | 0 | accepted |
| ptq1-bitexact | tg128 | 3/3 | 29.58 | 32.06 | +8.4% | 1.073-1.100x | 0 | 0 | 0 | accepted |
| ptq1-bitexact | s0c1 | 3/3 | 26.94 | 29.04 | +7.8% | 1.065-1.087x | 0 | 0 | 0 | accepted |
| ptq1-total | s1c1 | 3/3 | 27.06 | 34.69 | +28.1% | 1.210-1.334x | 0 | 0 | 0 | accepted |
| pq2-same-mode | tg128 | 3/3 | 32.11 | 35.96 | +12.0% | 1.112-1.126x | 0 | 0 | 0 | accepted |
| pq2-same-mode | pp2 | 3/3 | 34.66 | 37.09 | +7.0% | 1.062-1.082x | 0 | 0 | 0 | accepted |
| pq2-same-mode | s0c1 | 3/3 | 29.94 | 33.37 | +11.5% | 1.104-1.123x | 0 | 0 | 0 | accepted |
| pq2-same-mode | s0c2 | 3/3 | 31.27 | 32.94 | +5.3% | 1.037-1.066x | 0 | 0 | 0 | accepted |
| pq2-total | s1c1 | 3/3 | 29.98 | 27.14 | -9.6% | 0.838-0.954x | 0 | 0 | 0 | accepted |
| b1-ternary | tg128 | 3/3 | 34.85 | 39.39 | +13.0% | 1.125-1.134x | 0 | 0 | 0 | accepted |
| b1-ternary | s0c1 | 3/3 | 32.12 | 36.26 | +12.9% | 1.118-1.138x | 0 | 0 | 0 | accepted |
| b1-ternary | s0c2 | 3/3 | 32.56 | 36.52 | +12.2% | 1.108-1.135x | 0 | 0 | 0 | accepted |
| b1-binary | tg128 | 3/3 | 39.09 | 43.86 | +12.2% | 1.118-1.126x | 0 | 0 | 0 | accepted |
| b1-binary | s0c1 | 3/3 | 36.43 | 40.76 | +11.9% | 1.111-1.123x | 0 | 0 | 0 | accepted |
| b1-binary | s0c2 | 3/3 | 45.30 | 52.59 | +16.1% | 1.156-1.166x | 0 | 0 | 0 | accepted |
| m1-single-user | s0c1 | 3/3 | 27.04 | 29.66 | +9.7% | 1.085-1.111x | 0 | 0 | 0 | accepted |
| m1-single-user | s1c1 | 3/3 | 13.22 | 35.28 | +167.0% | 2.655-2.680x | 0 | 0 | 0 | accepted |
| m1-total | s1c1 | 3/3 | 27.08 | 35.39 | +30.5% | 1.214-1.370x | 0 | 0 | 0 | accepted |
| previous-pr-m1 | pp2 | 3/3 | 44.18 | 49.00 | +10.9% | 1.084-1.124x | 0 | 0 | 0 | accepted |
| previous-pr-m1 | pp3 | 3/3 | 49.87 | 53.55 | +7.4% | 1.062-1.082x | 0 | 0 | 0 | accepted |
| previous-pr-m1 | s0c1 | 3/3 | 27.37 | 29.64 | +8.3% | 1.072-1.096x | 0 | 0 | 0 | accepted |
| previous-pr-m1 | s1c1 | 3/3 | 35.28 | 35.74 | +1.3% | 1.010-1.019x | 0 | 0 | 0 | accepted |
| reload-baseline | s0c1 | 3/3 | 26.97 | 27.02 | +0.2% | 0.992-1.010x | 0 | 0 | 0 | accepted |
| reload-baseline | s1c1 | 3/3 | 13.19 | 13.26 | +0.6% | 0.990-1.018x | 0 | 0 | 0 | accepted |

## Single-user native generation

These means exclude prompt/request overhead. Only timing-complete cells with matching generated output are shown. Plain and MTP are separate cells; their ratio is descriptive, not a paired ABBA estimate. A/B means all flags off / flags under test (see studies.json). The M1 profile disables PTQ1 staging. Bonsai 1 has no MTP head in this experiment. When only the PQ2 total-benefit study provides optimized MTP, original MTP remains unmeasured.

| Model | Original plain | Original MTP | Optimized plain | Optimized MTP | MTP vs optimized plain |
|---|---:|---:|---:|---:|---:|
| Bonsai 2 PTQ1_0 | 29.34 | 13.74 | 31.95 | 38.02 | +19.0% |
| Bonsai 2 PTQ1_0 (M1 profile) | 29.33 | 13.69 | 31.91 | 38.63 | +21.1% |
| Bonsai 2 PQ2_0 | 31.97 | - | 35.81 | 28.73 | -19.8% |
| Bonsai 1 ternary | 34.44 | - | 39.18 | - | - |
| Bonsai 1 binary | 38.79 | - | 43.69 | - | - |
