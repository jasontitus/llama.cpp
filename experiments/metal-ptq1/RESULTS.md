# M1 Ultra PTQ1_0 multi-column results

Implemented on local branch `downstream/metal-ptq1-current`, based on PrismML default branch `prism` at `3b19c377d18157bfec39ec71bad193e9ef000cf2`, fetched September 23, 2026. Original user checkouts are untouched. Changes are uncommitted; nothing was pushed or submitted upstream.

The opt-in candidate extends PrismML’s PTQ1-specific single-vector byte ownership and activation-coefficient calculation to columns 2, 3 and 4. The generic baseline already shares dequantized values across columns; this candidate shares the cheaper PTQ1-specific decode. Rows per SIMD group and groups per threadgroup remain parameterized. The measured configuration is four rows, one group (R4/S1). Enable with `GGML_METAL_PTQ1_MULTICOL=1`; default dispatch is unchanged.

## Standard end-to-end llama-bench

Official Bonsai 2 27B PTQ1_0, M1 Ultra 64-core GPU, 128 GB RAM, macOS 26.7, Release build, GPU offload 99, flash attention on, 16 CPU threads. Each test has three independent A-B-B-A quartets, five internal repetitions per invocation, and its own model load/warmup. A and B are the same final binary with the candidate flag off/on. Six invocation means per condition. Throughput below averages those means; speedups compare paired mean latency. Ranges are the three observed quartet ratios, not confidence intervals. No sample exclusion.

| Test | Baseline tokens/s | Candidate tokens/s | Paired speedup | Paired range | A/B latency CV |
|---|---:|---:|---:|---:|---:|
| pp1 | 28.17 | 28.33 | 1.007× | 0.988–1.034× | 3.1% / 1.6% |
| pp2 | 16.32 | 44.43 | 2.721× | 2.708–2.743× | 0.5% / 1.8% |
| pp3 | 21.99 | 50.80 | 2.310× | 2.277–2.336× | 0.5% / 1.7% |
| pp4 | 26.00 | 44.32 | 1.704× | 1.688–1.721× | 0.4% / 1.9% |
| pp8 | 29.81 | 29.79 | 0.999× | 0.994–1.003× | 0.8% / 0.4% |
| pp16 | 63.74 | 63.64 | 0.998× | 0.996–1.001× | 0.2% / 0.3% |
| tg32 | 29.10 | 29.32 | 1.008× | 1.003–1.013× | 1.1% / 0.7% |

`ppN` measures prompt processing with N tokens; `tg32` measures ordinary single-token generation for 32 tokens. Small-prompt improvements do not establish MTP/speculative acceptance rates or serving throughput. Single-token decode is outside this candidate’s scope.

## Projection measurements

Four actual PTQ1 weight shapes from model metadata, n=1,2,3,4,8. Each gets three A-B-B-A cycles, six samples per condition, reproducibly shuffled shape order between cycles, fixed input seed, serialized GPU work, and warmup in every process. The table reports mean latency and geometric mean of three paired quartet speedups. Each backend sample replays its graph for at least one second. Internal replays are not independent samples. These reuse the same tensors and include amortized backend overhead; they are not cold-cache memory bandwidth measurements.

| K x M | n | A (µs) | B (µs) | Paired speedup | Paired range | A/B sample CV |
|---|---:|---:|---:|---:|---:|---:|
| 5120 x 10240 | 1 | 50.47 | 49.92 | 1.011× | 1.004–1.016× | 5.3% / 4.2% |
| 5120 x 10240 | 2 | 241.86 | 64.95 | 3.728× | 3.602–3.895× | 6.7% / 7.9% |
| 5120 x 10240 | 3 | 271.67 | 92.56 | 2.932× | 2.887–3.012× | 7.9% / 6.0% |
| 5120 x 10240 | 4 | 285.64 | 149.53 | 1.910× | 1.875–1.963× | 3.2% / 1.2% |
| 5120 x 10240 | 8 | 590.79 | 585.92 | 1.008× | 1.000–1.024× | 8.4% / 7.8% |
| 5120 x 17408 | 1 | 72.18 | 74.04 | 0.977× | 0.944–0.998× | 7.3% / 9.6% |
| 5120 x 17408 | 2 | 409.92 | 103.51 | 3.959× | 3.921–3.989× | 11.8% / 11.7% |
| 5120 x 17408 | 3 | 445.60 | 151.03 | 2.947× | 2.922–2.993× | 13.3% / 12.0% |
| 5120 x 17408 | 4 | 469.38 | 235.62 | 1.990× | 1.924–2.114× | 6.5% / 1.3% |
| 5120 x 17408 | 8 | 952.52 | 949.31 | 1.003× | 1.000–1.008× | 12.3% / 11.8% |
| 17408 x 5120 | 1 | 85.10 | 83.50 | 1.019× | 0.999–1.051× | 3.5% / 1.2% |
| 17408 x 5120 | 2 | 440.42 | 119.47 | 3.681× | 3.627–3.774× | 10.7% / 8.7% |
| 17408 x 5120 | 3 | 494.52 | 186.59 | 2.643× | 2.534–2.854× | 9.2% / 3.2% |
| 17408 x 5120 | 4 | 568.57 | 306.76 | 1.853× | 1.792–1.933× | 5.7% / 1.7% |
| 17408 x 5120 | 8 | 1033.88 | 1052.20 | 0.984× | 0.965–1.003× | 13.1% / 14.7% |
| 5120 x 6144 | 1 | 33.45 | 32.90 | 1.016× | 0.998–1.048× | 4.1% / 1.9% |
| 5120 x 6144 | 2 | 149.21 | 45.55 | 3.272× | 3.211–3.355× | 8.3% / 6.0% |
| 5120 x 6144 | 3 | 190.31 | 67.05 | 2.830× | 2.684–3.041× | 9.4% / 3.5% |
| 5120 x 6144 | 4 | 198.15 | 101.63 | 1.949× | 1.904–2.011× | 4.8% / 1.2% |
| 5120 x 6144 | 8 | 364.89 | 364.52 | 1.001× | 0.999–1.002× | 10.9% / 10.9% |

Targeted paired speedups range from 1.85× to 3.96×. Minimum targeted quartet speedup: 1.79×. Unchanged n=1/n=8 controls range from 0.977× to 1.019×.

Maximum absolute-timing sample CV is 14.7%. ABBA reduces linear drift; it does not eliminate background load, thermal variation, or nonlinear effects. GPU jobs from this experiment never overlap; other system activity is not locked out. Read-only thermal diagnostics reported no recorded warnings, but temperature and power were not measured. These data support large effects, not sub-percent tuning claims.

The final 17408 x 5120 n=4 gain is lower than the preceding binary’s run (1.85× versus 2.48×). The source artifacts and run times differ, so the cause is unresolved; a compiler/register-allocation investigation is a follow-up, not a reason to substitute older numbers. All tables here use the corrected final binary.

## Is the output the same?

Generated text matched in the targeted tests. Floating-point logits are **not bitwise identical**. The candidate follows the single-vector arithmetic geometry, while the replaced generic path accumulates differently. Rounding differences can change stochastic sampling or a near-tied argmax on other inputs.

- Before the final 64-bit output-index correction: all 195 greedy decisions and decoded outputs matched across nine prompt/microbatch combinations (microbatches 2, 3, 4). Compared 48,422,400 logits; maximum absolute difference 0.0018577576, maximum logit NMSE 4.5893e-10. Three arithmetic-prompt cases emitted immediate EOS and contribute one decision each; the other six contribute 32 each.
- After that correction: repeated microbatch-4 checks match all 65 greedy decisions and text, comparing 16,140,800 logits. Maximum absolute difference 0.0015320778; maximum NMSE 3.8678e-10. The corrected indexing does not change arithmetic for these shapes.
- Standard `llama-perplexity`, first two 512-token WikiText-2 test chunks, batch/microbatch 4: baseline **8.7487 ± 1.12295**, candidate **8.7486 ± 1.12294** (as reported by the tool). This is a bounded smoke test, not full-corpus quality evaluation.
- Final build: 20/20 full-size CPU-reference projection checks pass with candidate off and on; 48/48 F32 broadcast/padding/column edge cases pass each way; 48 existing upstream PTQ1 F32 cases pass with the candidate.
- All nine row/group configurations pass 42 deterministic checks each (378 checks), with CPU-dequantized, double-accumulated reference dots and a 1e-8 NMSE bound. Worst reference NMSE 5.82e-11 and worst cross-batch NMSE 1.79e-11. Candidate-off standalone checks also pass.
- 48 F16 fixture cases are explicitly skipped because the CPU reference does not support them. They are not counted as numerical coverage.
- n=1, n=8, other types, and MUL_MAT_ID implementations remain unchanged. New row-tail loads are clamped; output-column indexing uses 64-bit multiplication. Very large overflow-boundary allocations were not exercised on hardware.

## Experiment decisions and M5 follow-up

R4/S1 is the first promising portable candidate, not a proven global optimum. R2/S1 was generally slower in exploratory sweeps but had one isolated shape win; retain it for paired follow-up. R8 and additional group counts pass correctness but are not performance-selected. The experiment log retains exploratory results, rejected ideas and measurement corrections.

M5 remains phase two: replay the same model, pinned baseline, correctness tests and ABBA protocol on actual M5, starting with R4/S1 and R2/S1. Explore other row/group choices only as data justifies. Add Apple10-specific variants or dispatch only after measured M5 evidence. No M5 optimization, simulation or speed claim is included.

## Reviewable downstream work

The core patch changes four Metal files plus the optional deterministic seed in the existing backend test runner (166 insertions, two deletions). The research patch adds standalone helpers, fixtures, reproduction instructions and experiment documentation under `experiments/metal-ptq1`. It adds no new file under `tests/` or CMake test target. Both patches are checked against the pristine pinned baseline.

Existing [PrismML PR #225](https://github.com/PrismML-Eng/llama.cpp/pull/225) changes single-vector dense row count. This candidate preserves that path and should be coordinated with that work before submission. The exported patch is prepared for local review; no PR was created, and broader device/CI coverage is still needed before enabling it by default.

Checkout: the original M1 work checkout (local path omitted)
Branch: `downstream/metal-ptq1-current`
Model SHA-256: `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3`

The archive includes the core/research patches, reproduction guide, experiment log, final commands/raw measurements, historical comparison runs, validation logs, model metadata and provenance checksums. It excludes model weights, binaries and large logit dumps. Full local logit dumps remain in `work/current-model-check` and `work/final-model-check`.

Apply either the core patch or the full research patch, not both, to the pinned PrismML revision. Follow `experiments/metal-ptq1/README.md` for reproduction.
