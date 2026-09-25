# Bonsai on Apple silicon: Metal decoding experiments and results

Downstream research on PrismML's llama.cpp fork (base `0324c66`). It ports the CUDA small-batch work from
[PrismML PR #218](https://github.com/PrismML-Eng/llama.cpp/pull/218) to Metal, plus findings from tuning
on Apple silicon. Research kernels remain opt-in. Flags-off kernel dispatch and arithmetic are retained, but safety fixes also affect scratch reservation and flag-cache handling.

| Device | Status |
|---|---|
| Apple M5 Max (40-core GPU, Apple10) | **Measured**: results below |
| Apple M1 Ultra (64-core GPU, Apple7) | **Measured**: 64 GPU cores, 128 GB, macOS 26.7; [M1 results and profile](../m1/README.md) |
| iPhone 17 Pro Max (A19 Pro, Apple10, 12 GB, iOS 27) | **Measured** with the [`BonsaiBench`](../ios/BonsaiBench) app: an overnight suite, 3 quartets per cell (2 for the popcount context), every run starting at nominal temperature; MTP is generation only |

Desktop columns use the same model-specific flags and paired A-B-B-A procedure. The M1 column below uses the M5 flag set, including PTQ1 staging; the preferred M1 single-user profile disables staging and is reported separately. The iPhone uses an adapted in-app protocol, not the desktop tools. Compare paired gains within each study; absolute rates also reflect device, software revision, and run conditions.

## Summary

Every speedup below is ours alone: our flags against the flags-off build of the same code on the same device.
PrismML's own popcount option is never counted.

**Text generation, one user (Bonsai 2 PTQ1_0 unless noted):**

| | M5 Max | M1 Ultra | iPhone 17 Pro Max |
|---|---|---|---|
| Plain decoding | 1.10x | 1.09x | 1.09x (tg128), 1.08x (chat128) |
| With MTP, 1 draft token (server) | **1.34x** | 1.28x | **1.29x** (generation only; M5 generation only: 1.35x) |
| Plain decoding, other models (tg128) | 1.11-1.14x | 1.12x (PQ2_0) | 1.07-1.08x (PQ2_0), **1.18x** (Q1_0) |

- The generated text is identical between the arms in every server comparison.
- A bit-identical mode (in-place delta-net state only) gives 1.07x on the M5 and 1.08x on the M1.

**Batches of 2-8 tokens** (MTP verification, concurrent users). This is where the CUDA PR #218 port pays
most: upstream's PTQ1_0 multi-token path is slow on Apple GPUs.

| PTQ1_0 | M5 Max | M1 Ultra | iPhone 17 Pro Max |
|---|---|---|---|
| pp2 / pp4 / pp8 | 3.85x / 2.12x / 2.29x | 2.97x / 1.51x / 1.46x | **4.11x / 2.41x / 2.31x** |
| 2 concurrent requests (server) | 3.43x | 2.73x | – |
| MTP vs upstream's own MTP (server) | 3.34x | 2.61x | **4.0x** (generation only) |

Upstream's MTP is slower than its own plain decoding on these GPUs; only with the multi-column kernels
does MTP pay off.

**Prompt processing, Bonsai 1 binary (Q1_0), M5 Max:** a new K32 tensor prefill kernel with grid swizzle
(`GGML_METAL_Q1_SWIZZLE_LOG=1`, ported from our earlier Bonsai 1 tuning) makes pp512 1.07x and pp128 1.08x
faster on top of the Q1 flags, with bitwise-identical logits on the full model; the full Q1 set is 1.11x
(pp512), 1.14x (pp128) and 1.13x (tg128) over flags off. It applies to micro-batches whose size is a multiple of
128 tokens. On the iPhone it gives **1.12x at pp512** and 1.03-1.07x at pp128 (two studies), with the app
confirming the K32 kernel ran.

**Quality is unchanged.** On the M5, all four models with every flag on (PTQ1_0 including the tensor path)
against upstream:

- the same top token at 100% of positions;
- mean KL divergence 0;
- perplexity within its error of upstream;
- identical HellaSwag and Winogrande scores.

The strict kernel checkers pass. After the M1 changes the correctness matrix is identical and M5 speed is
within 1.1%, so the M5 numbers stand for the current branch.

**By device:**

- **M5 Max:** the results table is complete.
- **M1 Ultra:** the same pattern, somewhat smaller. MTP helps PTQ1_0 but not PQ2_0 (0.90x): see the M1
  profile.
- **iPhone 17 Pro Max:**
  - Small-batch gains are the largest of any device, because the phone GPU is compute-bound on these kernels.
  - Memory is not a limit: weights are memory-mapped, with a 0.4 GB app footprint and about 6 GB still
    available.
  - Heat is the main measurement problem, so the app gates on thermal state and uses 40-60 s cooldowns.
  - MTP works on the phone: 85.5% of drafts accepted, text identical to upstream, 1.29x over upstream plain
    decoding and 4.0x over upstream's own MTP (overnight suite, 3 quartets each).
  - The overnight suite starts every run at nominal temperature and waits up to an hour for it: all nine
    studies completed with no quartet lost to heat. An evening run that allowed fair starts lost most of
    its generation quartets to heat.
  - Both 7.2 GB PQ2_0 models fit (memory-mapped, 0.4 GB app footprint).
  - **Correction:** the "PTQ1_0 pp512 is ~30% slower with the flags on the phone" reported earlier is not
    established. It came only from two quartets rejected because the phone reached serious during the stack
    runs, whose calls slowed from 7.3 s to 10.8 s within the run (heat). Measured cleanly, the two flags that
    act at 512 tokens are neutral on the phone (`SMALLM_MM` 0.975x, rows mode 0.988x), and on the M5 the stack
    is 1.046x. The app's diagnostics now measure the full stack at pp512 under the overnight thermal gate.
  - 512-token prompt runs sometimes fail on the phone with an iOS GPU error, in every arm including plain
    upstream (evening: about 1 in 4 runs; overnight, starting cool: 4 PTQ1 runs, no Q1 runs); see the app's
    README.

## Results by device

Paired speedup against the flags-off research build **on the same device**, based on PrismML `0324c66`, not current PrismML HEAD. Desktop values are geometric means of three A-B-B-A quartets, with flags-off -> enabled tokens/s in parentheses. Server rates include prompt/request overhead and are aggregate rates for concurrent requests. Server rows: llama-server, 128 greedy tokens, short prompt;
generated text was identical between arms unless noted. Flags per model are listed under "Recommended
flags" below; the bit-identical row uses `GGML_GDN_ROWS_PLAIN=1` only. The bit-identical label follows historical M5 numerical checks; the M1 GDN-only timing study checks generated tokens, not whole-model bitwise logits. M1 batch-invariant-mode validation is recorded separately.

| Result | Apple M5 Max | Apple M1 Ultra | iPhone 17 Pro Max |
|---|---|---|---|
| PTQ1: upstream plain -> flags + MTP (server, 1 request) | **1.34x** (41.4 -> 55.5) | 1.28x (27.1 -> 34.7) | n/a (server row; see "generation only" below) |
| PQ2: upstream plain -> flags + MTP (server, 1 request) | **1.22x** (42.9 -> 52.5) | 0.90x (30.0 -> 27.1) | n/a (PQ2 MTP file not on the phone) |
| PTQ1 plain decoding (server, 1 request) | 1.10x (41.3 -> 45.7) | 1.09x (27.0 -> 29.6) | n/a |
| PTQ1 plain decoding, bit-identical subset (server, 1 request) | 1.07x (41.2 -> 44.3) | 1.08x (26.9 -> 29.0) | n/a |
| PTQ1 tg128 | 1.08x (43.4 -> 47.0) | 1.09x (29.7 -> 32.3) | 1.09x (5.94 -> 6.45) |
| PTQ1 MTP -> MTP (server, 1 request) | 3.34x (16.6 -> 55.4) | 2.61x (13.3 -> 34.6) | n/a (server row; see "generation only" below) |
| PTQ1 generation only: upstream plain -> flags + MTP | 1.35x (45.4 -> 61.4) | – | **1.29x** (6.03 -> 7.81; 1.14-1.45), identical text, 85.5% of drafts accepted |
| PTQ1 generation only: MTP -> MTP | not measured | – | **4.0x** (1.97 -> 7.86; 3.78-4.29), identical text |
| PTQ1 2 requests (server) | **3.43x** (17.9 -> 62.2) | 2.73x (15.3 -> 41.7) | n/a |
| PTQ1 pp2 / pp4 / pp8 | **3.85x / 2.12x / 2.29x** (20.6 -> 79.3, 37.0 -> 78.4, 40.5 -> 92.8) | 2.97x / 1.51x / 1.46x (16.3 -> 48.4, 26.1 -> 39.5, 30.0 -> 43.7) | **4.11x / 2.41x / 2.31x** (2.88 -> 11.85, 4.82 -> 11.61, 4.98 -> 11.51) |
| PQ2 tg128 | 1.11x (45.5 -> 50.4) | 1.12x (32.1 -> 36.0) | 1.08x (7.01 -> 7.61); pp2 1.34x |
| PQ2 2 requests (server) | 1.33x (48.1 -> 64.1) | 1.05x (31.3 -> 32.9) | n/a |
| Bonsai 1 ternary tg128 | 1.12x (48.0 -> 53.6) | 1.13x (34.9 -> 39.4) | 1.07x (7.23 -> 7.73); pp2 1.37x |
| Bonsai 1 ternary 2 requests (server) | 1.33x (50.1 -> 66.6) | 1.12x (32.6 -> 36.5) | n/a |
| Bonsai 1 binary tg128 | 1.14x (66.7 -> 75.7) | 1.12x (39.1 -> 43.9) | **1.18x** (10.9 -> 12.8) |
| Bonsai 1 binary 2 requests (server) | 1.17x (79.8 -> 93.2) | 1.16x (45.3 -> 52.6) | n/a |
| Bonsai 1 binary pp512 / pp128, flags + K32 prefill (swizzle 1) | 1.11x / 1.14x (856.7 -> 950.3, 613.1 -> 697.3) | – | K32 over the Q1 flags: **1.12x** (78.3 -> 87.4) / 1.03x (overnight; 1.065x in an evening study) |

### M1 Ultra: single-user generation rates

Native generation tok/s, excluding prompt/request overhead. Plain and MTP are separate randomized cells; the final percentage is the descriptive ratio of their displayed optimized means, not a paired ABBA gain. MTP uses one draft token and the pinned grafted head. Bonsai 1 has no MTP head in this study.

| Model | Original plain | Original MTP | Optimized plain | Optimized MTP | MTP vs optimized plain |
|---|---:|---:|---:|---:|---:|
| Bonsai 2 PTQ1 (M1 profile) | 29.33 | 13.69 | 31.91 | 38.63 | +21.1% |
| Bonsai 2 PQ2 | 31.97 | unmeasured | 35.81 | 28.73 | -19.8% |
| Bonsai 1 ternary | 34.44 | n/a | 39.18 | n/a | n/a |
| Bonsai 1 one-bit | 38.79 | n/a | 43.69 | n/a | n/a |

For PTQ1 on this M1 Ultra, use the listed PTQ1 flags with `GGML_METAL_PTQ1_STAGE=0`. The direct prior-PR comparison measured native plain 29.28 -> 31.87 tok/s and MTP 38.65 -> 39.16, with paired full-request gains of 8.3% and 1.3%, respectively. This restores MTP while improving plain decoding. The separate same-mode means in the table above retain their own measured values. For PQ2, optimized plain is faster on this measured workload; do not inherit the M5 MTP recommendation.

The direct baseline-plain -> M1-profile-MTP comparison improves full-request throughput by 30.5% (27.08 -> 35.39 tok/s). PQ2's corresponding paired result loses 9.6% (29.98 -> 27.14); the loss is consistent with its weaker two-token speedup failing to offset speculative overhead, but exact per-operation costs were not profiled. Draft acceptance matches PTQ1 on the tested prompts, so lower acceptance is not the explanation for the format difference. No historical PQ2 MTP regression is established by these cells.

See [M1 results](../m1/results/final-device/RESULTS.md) and [experiment log](../m1/EXPERIMENTS.md) for prior-PR controls, numerical checks, source/binary hashes, raw observations, and limitations. Codex remained open; thermal samples were nominal, but per-process GPU attribution was unavailable. Ordinary optimized logits are not claimed bitwise equal to baseline.

### Filling in a device column

On the device, from a checkout of this branch built with Metal:

```sh
zsh experiments/metal-ptq1/m5/tools/run-device-study.sh build/bin /path/to/models /path/to/mtp-models out-<device>
python3 experiments/metal-ptq1/m5/tools/device-table.py out-<device>
```

The first command runs every study behind the table (about 1.5 hours; nothing else on the GPU, AC
power). The second prints one line per row; paste the values into that device's column, add the
device's details (chip, GPU cores, memory, macOS) to the device list above, copy `out-<device>/*/summary.json`
into `results/<device>/`, and commit.

The iPhone column comes from the BonsaiBench app's exported JSON, saved in [`../ios/results`](../ios/results).
- The app measures tg128, ppK and gen128. gen128 is greedy generation, plain or MTP, through llama.cpp's
  speculative-decoding loop, and it excludes the prompt.
- The app has no server, so its MTP numbers go in the "generation only" rows. Compare them with the M5's
  generation-only rate, not with the server rows, which include the prompt.
- The app's tg128 is measured like llama-bench's (128 single-token decodes from an empty context, no
  sampling), with 1 repetition per observation instead of 3.
- Phone observations reuse one app/model with fresh contexts, 1024 context, and thermal-gated cooldowns (40 s for Q1, which stayed nominal; 50-60 s in the reported PTQ1 studies, which ran mostly at fair). Desktop observations use fresh processes and an 8 s cooldown.
- The phone column comes from the overnight suite of 2026-09-25 (`../ios/results/overnight-2026-09-25/`): three A-B-B-A quartets per cell (two for the popcount context), every run starting at nominal temperature after a 40-90 s cooldown, waiting up to an hour for it, and a 1.2 spread gate. The Q1 tg128 figure is from an earlier complete study. The phone runs a build of `e610dc1` plus the then-uncommitted changes published in `3faac35`.
- Rebuild `llama.xcframework` after pulling core changes before collecting new phone results; existing phone archives retain their original measured revision.

## Apple M5 Max: headline (paired against upstream on the same machine)

Bonsai 2 27B, one request, 128 greedy tokens, llama-server, tokens/s including the short prompt:

| | Upstream plain decoding | M5 flags + MTP (1 draft) | Speedup | Output |
|---|---:|---:|---:|---|
| Bonsai 2 PTQ1_0 | 41.4 | **55.5** | **1.34x** (1.24-1.40) | identical text |
| Bonsai 2 PQ2_0 | 42.9 | **52.5** | **1.22x** (1.14-1.28) | identical text |

Generation rate alone (excluding prompt processing): PTQ1 45.4 -> 61.4 tok/s (1.35x), PQ2 46.0 -> 57.2 (1.24x).
Upstream's own MTP is slower than its plain decoding on Metal (16.6 tok/s for PTQ1) because two-token
verification falls back to a generic kernel; the fast multi-column kernels are what make MTP pay off.

### Pick your guarantee

The following modes and numerical evidence describe the historical M5 snapshots. They are not new quality or bitwise guarantees for the final merged M1 code.

Re-checked on the M5 Max after the M1 changes (`results/m1-change-check/`):

- **Correctness is unchanged.** The correctness matrix (test-backend-ops suites, fixtures and the strict
  1e-8 checkers for PTQ1_0, PQ2_0 and Q1_0) gives the same results as before, 12 of 12 configurations,
  with the same kernel variants.
- **Speed is unchanged.** A paired A-B-B-A of the pre-M1 build (`754d1fb`) against the current build, same
  PTQ1 flags, stays within 1.1% on every cell, with identical tokens in the server cells:

  | Cell | Paired (current vs pre-M1) |
  |---|---:|
  | tg128 | 1.005x |
  | pp2 | 0.989x |
  | pp4 | 0.994x |
  | pp8 | 1.000x |
  | 2 requests | 0.992x |
  | MTP, 1 request | 0.998x |

- **The M5 numbers above therefore stand for the current branch.** On the M5 the M1 changes are
  memory-layout and M1-only tuning; the 4-row PTQ1 default applies to GPU family 7 only.

| Guarantee | Flags | Single-request speedup vs upstream |
|---|---|---|
| **Bit-identical to upstream** | `GGML_GDN_ROWS_PLAIN=1` | plain decoding +7-9% (PTQ1 1.07-1.08x, PQ2 1.09x); Bonsai 1 binary +14% with `GGML_METAL_SMALLM=1` |
| **Identical text in every test** | recommended set below + MTP | 1.34x PTQ1, 1.22x PQ2 |
| **MTP bit-identical to plain decoding** | recommended set + `GGML_METAL_BATCH_INVARIANT=1` | ~1.2x PTQ1 (the mode costs ~11% on multi-token steps) |

"Identical text in every test": every paired study compared the generated token IDs of both arms; none
differed. Logits differ from upstream in the last bits (NMSE ~1e-10), as with any kernel change;
upstream itself is not bitwise stable across batch sizes.

### Recommended flags per model

These are the measured M5 selections. Use the separate M1 single-user guidance above on M1 Ultra.

| Model | Flags | Measured vs upstream |
|---|---|---|
| Bonsai 2 PTQ1_0 | `GGML_METAL_PTQ1_MULTICOL=1 GGML_METAL_PTQ1_MULTICOL_MAX=8 GGML_METAL_PTQ1_GLU=1 GGML_METAL_PTQ1_STAGE=1 GGML_GDN_ROWS_PLAIN=1 GGML_METAL_SMALLM_MM=1` (+ `GGML_METAL_PTQ1_TENSOR=1` for 4 concurrent MTP requests) | plain +10.5%, with MTP 1.34x |
| Bonsai 2 PQ2_0 | `GGML_METAL_PQ2_MULTICOL=1 GGML_METAL_PQ2_GLU=1 GGML_GDN_ROWS_PLAIN=1 GGML_METAL_SMALLM=1 GGML_METAL_SMALLM_MM=1` | plain +11%, with MTP 1.22x, 2 requests +33% |
| Bonsai 1 ternary (PQ2_0) | same as Bonsai 2 PQ2_0 | decode +11.6%, 2 requests +33% |
| Bonsai 1 binary (Q1_0) | `GGML_GDN_ROWS_PLAIN=1 GGML_METAL_SMALLM=1 GGML_METAL_SMALLM_MM=1 GGML_METAL_Q1_SWIZZLE_LOG=1` | decode +13.6% (tg128, where only the two bit-identical flags act); server single request +10.9%; prompts pp512 +10.9%, pp128 +13.7% (the swizzle flag alone: +7.1%, +8.2%, bitwise equal) |

`GGML_METAL_SMALLM_MM` changes prompt processing of the 48-row projections: it is not bit-identical to
upstream, but closer to exact (NMSE vs a double-precision reference ~1e-14 instead of ~1e-7).

MTP: `llama-server --spec-type draft-mtp --spec-draft-n-max 1` with a model that carries an MTP head
(the Bonsai 2 models were grafted with the recipe from
[sudoingX/bonsai2-small-gpu](https://github.com/sudoingX/bonsai2-small-gpu/tree/eb52d9d7363cda2d910146f4e37f4b8c64c30c46/graft)).
One draft token is best on M5; 2 and 3 drafts measured 0.90x and 0.75x.

Not ours, for context: `GGML_METAL_Q1_0_POPCNT=1` is PrismML's own Q1_0 bit-plane option (in their code, off
by default), not part of these changes and not counted in any speedup above. Measured on top of our flags
it gave Bonsai 1 binary +17-28% on 4-8-token batches and +14% at 4 concurrent requests, with unchanged
perplexity, but its int8 activations change greedy text under concurrency.

## What changed and why (M5 findings)

- **Multi-token steps were ALU-bound on trit decoding.** Single-token decode is close to memory-bound on
  M5, but 2-8-token steps (MTP verify, concurrency) spent their time unpacking base-3 weights. Integer
  multiply-add is half rate on this GPU, so CUDA's dp4a/SIMD-in-register decode does not port; the
  existing float-floor decode is the right primitive, and the work is in sharing it.
- **Ported from CUDA #218:** multi-column kernels extended to 5-8 columns as tiles; fused gate/up/SWIGLU;
  a one-time activation layout pass (the planar-activation idea); tiny-projection routing (the 48-row
  delta-net gate projections were running as 1-4 threadgroups); batch-invariant mode.
- **Not from CUDA:** the delta-net recurrent state was copied out and back (3 MB per layer per token);
  PrismML's existing in-place rows mode is now also used for plain decoding. This alone is most of the
  single-request plain-decoding gain and is bitwise identical.
- **Q1_0 prefill (from our earlier Bonsai 1 work, not CUDA):** a copy of the tensor prefill kernel with a
  static K=32 `matmul2d` and no bounds handling for products made only of full tiles, plus a grid swizzle
  that runs pairs of row tiles on the same activation columns: +7% pp512. It lives in an optional Metal
  library, so a device that cannot build it loses only these kernels.
- **M5 tensor units:** `matmul2d` runs half at ~33 T MAC/s and int8 at ~64 T. A tensor-unit matvec with
  hi/lo-split activations (NMSE 4e-13) wins only at 8 columns (+9% for 4 concurrent MTP requests).
  Upstream's tensor prefill already runs at ~88% of the best arrangement measured; CUDA's prefill ideas
  (#214) and an int8 path were tested and rejected with measurements.

## Method

- Apple M5 Max (40-core GPU), 128 GB, macOS 26.6.2, AC power, High Power mode.
- A-B-B-A quartets, 3 per cell, reproducibly shuffled order, 8 s cooldown, fresh process per observation,
  one GPU process at a time, 20% within-arm spread gate; rejected quartets kept (none in the final studies).
- Historical M5 timing runs used frozen snapshots, including `snapshots/snap-final.diff` (SHA-256 4f0e4599...) for the headline. The added M5 pp2/4/8 and concurrency results use `9f7a364` with an empty patch. M1 final results use production code at `8a1bdf7`; subsequent merged app/documentation changes do not change that code. The merged branch is not byte-identical to the historical M5/phone snapshots. Revalidate the M1 kernel/safety changes on M5 before claiming preserved M5 performance.
- Short prompts (~30 tokens) and 128 generated tokens. Long-context throughput is not claimed.

## Correctness evidence

This section records the historical M5 snapshots and their quality runs. Final M1 checks are listed in [the M1 experiment log](../m1/EXPERIMENTS.md); no new M1 quality-benchmark suite is claimed.

- Strict checker (NMSE <= 1e-8 vs a double-precision reference, plus batch invariance) for PTQ1_0, PQ2_0 and
  Q1_0, including 4163-row tails, n = 1..8 and K = 33024; test-backend-ops MUL_MAT, fusion and flash
  attention suites; every new kernel verified to execute in the configuration meant to exercise it
  (`results/correctness-matrix.txt`).
- Full-model teacher-forced logits vs upstream on all four models, ubatch 1..8: 0 token mismatches, NMSE <=
  4.2e-10 (the M1-validated candidate measured 4.6e-10).
- Batch invariance: 64/64 positions bitwise identical at batch sizes 2/3/4, also across 1134- and
  2011-token contexts.
- Three adversarial code reviews; all findings fixed (see EXPERIMENTS.md).
- Q1_0 K32 prefill (`GGML_METAL_Q1_SWIZZLE_LOG=1`, added after the quality runs below): full-model float
  logits bitwise equal with and without it on M5 Max (700-token prompt, 173.8M logits), and bitwise equal on
  17 isolated products with the kernel identified per product (`results/q1-k32/`). Two further adversarial
  reviews; findings fixed. The Q1_0 quality row below therefore also holds with it on M5; other devices need
  their own check.
- Standard quality benchmarks, every model with all of its flags on (PTQ1_0 including the tensor path)
  against upstream (logs in `results/quality/`, script `tools/run-quality.sh`):

  | Model | PPL ratio (WikiText-2, 20 x 512) | Mean KLD (max) | Same top token | HellaSwag 400 | Winogrande 1267 |
  |---|---|---|---|---|---|
  | Bonsai 2 PTQ1_0, batch 1 / 4 | 1.00006 +/- 0.00006 | 0.000000 (5.5e-5) | 100% | 75.25 = 75.25 | 73.32 = 73.32 |
  | Bonsai 2 PQ2_0, batch 1 / 2 | 1.00006 +/- 0.00006 | 0.000000 (5.6e-5) | 100% | 75.25 = 75.25 | 73.32 = 73.32 |
  | Bonsai 1 ternary PQ2_0, batch 1 / 2 | 1.00048 +/- 0.00038 | 0.000000 (5.7e-5) | 100% | 74.50 = 74.50 | 71.67 = 71.67 |
  | Bonsai 1 binary Q1_0, batch 1 / 4 | 0.999999 | 0.000000 (6.4e-5) | 100% | 67.50 = 67.50 | 68.90 = 68.90 |

  Every accuracy score is identical, and every perplexity ratio is within its error of 1. The largest
  per-token KL divergence is at the level of float rounding.
  - For context, not ours: PrismML's popcount option on Q1_0 gives a PPL ratio of 1.00037 +/- 0.00046,
    mean KLD 0.00040 (max 0.034) and the same top token 99.06% of the time, with identical HellaSwag and
    Winogrande.

## Reproduce on your Mac

1. Check out this branch and build with Metal: `cmake -B build -DGGML_METAL=ON -DLLAMA_BUILD_SERVER=ON && cmake --build build -j`.
2. Run any tool with a flag set, e.g. `GGML_GDN_ROWS_PLAIN=1 GGML_METAL_PTQ1_MULTICOL=1 ... build/bin/llama-server -m model.gguf`.
3. For paired numbers comparable to the tables above, use `tools/abba-m5.py` (A-B-B-A, spread gate, token
   comparison): `python3 tools/abba-m5.py --output out --bin build/bin --src . --model M.gguf --mtp-model M-mtp.gguf
   --bench tg128 --server s0c1 s1c1 --env-a '{}' --env-b '{"GGML_GDN_ROWS_PLAIN":"1", ...}'`, and
   `tools/abba-m5-draft.py --draft-a 0 --draft-b 1` for upstream plain decoding vs MTP.
4. Correctness on a new device: `tools/correctness-matrix.sh` (test-backend-ops + strict checkers per flag
   set) and `tools/check-model-m5.py` (full-model logits vs upstream).

On Apple7/8/9 GPUs (M1-M4), `GGML_METAL_PTQ1_TENSOR` has no effect (no tensor units); everything else is
portable. Tuned defaults (rows per simdgroup, tile widths, the PQ2 width limit) were chosen on the M5 Max
and may not be optimal elsewhere. M1 measurements select STAGE=0 for PTQ1 single-user operation; enabled staging uses the measured family7 R4 fallback. Other family7 devices remain unmeasured. M5/A19/A20 need device-specific correctness, memory and ABBA gates before promoting defaults.

## Files

- `EXPERIMENTS.md`: full experiment log, including rejected ideas and the measurement corrections.
- `TODO.md`: open items.
- `results/`: per-study summaries and progress logs (paired ratios, ranges, token checks).
- `snapshots/`: the exact diff behind each benchmarked build.
- `tools/`: ABBA harness, full-model gate, invariance probes, strict checkers, standalone kernel bench,
  per-op profiler summary. Scripts assume the layout of the M5 handoff package (models under
  `workspace/work`), so adjust paths for other setups.
