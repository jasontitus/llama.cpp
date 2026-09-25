# M5 Max experiment log (2026-09-23)

Device: MacBook Pro Mac17,7, Apple M5 Max (18 CPU cores, 40 GPU cores), 128 GB, macOS 26.6.2 (25G83),
Metal 4 / MTLGPUFamilyApple10, tensor API available. AC power, High Power mode. Source: PrismML
0324c66 + the M1 multi-column patch, in the editable worktree `development/llama.cpp` (uncommitted;
no commits, pushes or PRs). All new work is opt-in behind environment flags so baseline and candidates
share one binary. Timed studies use a frozen snapshot build (`snap-abba1`, diff SHA-256 `2515cc65…`).

## What the M5 looks like (measured, before changes)

- **Single-token decode is not DRAM-bound, but it is close.** llama-bench tg128 = 45 tok/s (22 ms/token,
  ~150 GB/s of weight traffic). With weights streamed from DRAM (16 rotating copies in the standalone
  bench), the PTQ1 n=1 kernel takes 49.7 µs on ffn_up; removing only the base-3 decode gives 44.6 µs.
- **Multi-column (n >= 2) is ALU-bound on M5.** n=4 costs 151 µs vs 89 µs with decode removed
  (memory floor ~45 µs). Every extra column costs ~1 FMA + ~0.7 op of activation staging per weight.
- **ALU microbenchmarks** (lane-ops/s): fp32 FMA 7.6 T; fp16 8.8 T; packed half2 ~12.6 T half-flops;
  floor ~full rate; int32 mul-add 3.6 T (half rate), ushort2 1.9 T. CUDA's integer SIMD-in-register
  trit decode + dp4a therefore does not port; the existing float-floor decode is the right primitive.
- **Tensor API (per-core neural accelerators) via `matmul2d`:** half 33 T MAC/s at 64x32x32 (16 T at
  8 columns), int8 64 T, float 8 T (no gain). simdgroup_matrix 8x8 = 6.5 T (no gain over scalar).
- **Per-op GPU profile of one decode token** (new `GGML_METAL_PROFILE_OPS`, serialized so small ops are
  inflated): PTQ1 n=1 matvecs ~65-70%; recurrent-state copies (GET_ROWS gather + CPY write-back of the
  3 MB delta-net state per layer) 1.6 + 0.8 ms; everything else small.
- **Prefill:** the upstream tensor-API mul_mm already gives 3.2x on M5 (pp512 724 vs 224 tok/s with
  `GGML_METAL_TENSOR_DISABLE=1`). Keep it on.
- **Family-study absolute numbers were depressed.** The packaged run measured 27-31 tok/s plain C1;
  the same reference binary with the same flags measures 45 tok/s in isolation. Use its off/on ratios,
  not its absolute rates. The laptop also drifts ~15% under back-to-back load, so only ABBA counts.

## Candidates built (all opt-in)

| Flag | CUDA idea it ports | What it does |
|---|---|---|
| `GGML_METAL_PTQ1_MULTICOL_MAX=8` | ">4 columns take a different path" | n=5..8 run the PTQ1 multi-column kernel as 3+3 / 4+4 column tiles instead of the generic mul_mv_ext |
| `GGML_METAL_PTQ1_GLU=1` | mmvq gate fusion (`has_gate`) | ffn_gate + ffn_up + SWIGLU in one kernel for n=1..8; activations staged once for both; 2 rows/simdgroup |
| `GGML_METAL_PTQ1_STAGE=1` | planar activation re-layout pass | a pre-pass writes each column's collapse coefficients once; kernels load them as float4 (projections with >= 4096 rows) |
| `GGML_GDN_ROWS_PLAIN=1` | (not CUDA) | plain decode reads/writes the delta-net state in place (existing PrismML rows mode), removing the per-layer 3 MB gather + copy-back; single-sequence batches only |
| `GGML_METAL_PTQ1_TENSOR=1` | MMQ (tensor cores) for wider batches | n=5..8 on the M5 tensor units; exact half weights, hi/lo-split activations (built, not yet timed) |

## Kernel-level screens (single interleaved passes, µs; confirmation is the ABBA below)

- Tiles n=5..8 vs generic path: 1.3-2.6x faster on every Bonsai shape (ffn_up n=8 677 -> 362).
- Fused GLU per layer (gate+up+swiglu): n=1 114.7 -> 102.7, n=2 168.4 -> 152.9, n=4 359.6 -> 323.1.
  With 4 rows/simdgroup it spills: n=2 211, n=4 678. Default is 2 rows.
- Staged activations: 8-24% faster on every shape except the 1024-row attn_k/v at n=2 (-15%); gated to
  >= 4096 rows. Staged n=3 prefers 2 rows/simdgroup (up to 29% faster); 4-column tiles prefer 4.
- Rows-mode state: serialized GPU time per token 24.43 -> 22.79 ms; bit-identical logits.
- Tensor path v2 (standalone bench, ffn_up): n=8 138 µs vs 305 µs staged scalar; n=4 ~134 vs ~136 (tie).

## Rejected / not pursued (with evidence)

- **Threadgroup lookup table for trit decode**: 20-25% slower at every n (random tg loads cost more than
  10 float ops per byte).
- **Tensor path v1** (threadgroup-wide tiles, per-threadgroup activation staging): 2-4x slower than scalar.
- **Tensor path v3** (collapse coefficients through the tensor op, per-block epilogue on cooperative
  tensor elements): correct (NMSE 2e-11) but 220 µs vs v2's 138 µs.
- **Padding the tensor A slab** against bank conflicts: slower (threadgroup memory 16 -> 20 KB cut occupancy).
- **Half-only activations** for the tensor path: NMSE 3.4e-8, fails the 1e-8 gate; hi/lo split kept.
- **Existing multicol geometry sweep** (rows 2/4/8 x groups 1/2/4) could not fix n >= 3 scaling.
- **n=1 row count 5 or 8 (PrismML PR #225 idea)**: 1-2% in the standalone bench, within noise; not pursued.

## Correctness evidence

- test-backend-ops (MTL0): 72 Bonsai decode-shape fixtures (n=1..8, 16, incl. lm_head), 48 broadcast/
  padding edges, 101 PTQ1 MUL_MAT cases, 44 MUL_MAT_VEC_FUSION cases, across baseline, M1 candidate
  and the full stack. Added research cases with 4163 rows (row tails) and 6467-row fusion shapes so the
  staged and fused kernels (incl. partial 3-column tiles) actually execute.
- Strict standalone checker (NMSE <= 1e-8 vs double reference, plus batch invariance), extended to
  n=1..8 and a 4163-row case: 52/52 in every configuration.
- Full-model logits (Bonsai 2 PTQ1, teacher-forced, full vocabulary), baseline vs full stack, ubatch
  1..8 x 3 prompts: 24 cases, 129,126,400 logits, 0 token mismatches, identical text, max NMSE 4.2e-10,
  max |diff| 0.0017 (M1 candidate reference: 4.6e-10 / 0.0019).

## Bugs found and fixed during the work

1. **Fused GLU read/write race.** The allocator may place the SWIGLU output in memory freed by the FFN
   input (its last reader was ffn_up). Fused, the kernel read the input while writing the output there.
   Showed up only at ubatch 1 as NMSE 1.4e-4 with identical tokens. Fix: refuse fusion when the output
   overlaps the input, and refuse staging into the gate buffer when the output overlaps it.
2. **Rows-mode multi-sequence hazard (adversarial review).** With several sequences and a cell reorder,
   the extra-cell relocation runs before the in-place read and can give a sequence another's state.
   Fix: plain rows mode only when there are no extra cells (n_rs == n_seqs). The same pre-existing
   hazard exists in PrismML's MTP rows path; reported, not changed here.
3. **Test seeding** gave same-type/same-shape tensors identical data (gate == up), hiding swaps.
   Fixed with a per-tensor sequence number.

## ABBA 1: M1 candidate (A) vs M5 stack (B)

Frozen snapshot `snap-abba1` (diff SHA-256 2515cc65…), build `build-abba1`. A = `GGML_METAL_PTQ1_MULTICOL=1`
(the M1-validated R4/S1 candidate). B = A + `MULTICOL_MAX=8 GLU=1 STAGE=1 GDN_ROWS_PLAIN=1`. Three
A-B-B-A quartets per cell, reproducibly shuffled cell order per cycle, 8 s cooldown before every
observation, fresh process per observation, 20% within-arm spread gate. 36/36 quartets accepted on the
first attempt, 0 rejected, 0 A/B token mismatches in any server slot. Server: one wave of C requests,
128 greedy tokens each, 4096 context per slot, prompts rotated per cycle; aggregate = generated tokens /
wave makespan (includes prefill). llama-bench: 3 internal repetitions, 16 threads, FA on.

| Cell | A tok/s | B tok/s | Paired speedup | Quartet range | CV A/B |
|---|---:|---:|---:|---:|---:|
| tg128 | 42.93 | 46.68 | 1.087x | 1.079-1.093 | 1.3% / 0.8% |
| pp2 | 64.64 | 78.78 | 1.219x | 1.207-1.241 | 0.7% / 1.2% |
| pp3 | 56.87 | 81.80 | 1.439x | 1.433-1.444 | 0.6% / 0.2% |
| pp4 | 65.60 | 78.54 | 1.197x | 1.194-1.202 | 0.0% / 0.4% |
| pp8 | 40.46 | 92.65 | 2.290x | 2.285-2.293 | 0.0% / 0.2% |
| pp512 | 724.78 | 755.44 | 1.042x | 1.041-1.044 | 0.1% / 0.1% |
| server plain C1 | 41.57 | 45.62 | 1.097x | 1.088-1.116 | 2.0% / 0.1% |
| server MTP C1 (d=1) | 49.86 | 55.50 | 1.113x | 1.111-1.115 | 5.7% / 5.8% |
| server plain C2 | 52.85 | 67.53 | 1.278x | 1.246-1.317 | 1.6% / 2.9% |
| server MTP C2 | 47.25 | 55.20 | 1.168x | 1.158-1.180 | 1.3% / 1.5% |
| server plain C4 | 49.18 | 64.07 | 1.302x | 1.288-1.320 | 2.5% / 1.9% |
| server MTP C4 | 29.51 | 57.40 | 1.945x | 1.924-1.959 | 2.1% / 3.3% |

Per-request native generation rate (server timings, excludes prefill): plain C1 44.8 -> 49.4;
MTP C1 54.7 -> 61.4; MTP C4 7.9 -> 16.5 per request. MTP C1 CV is high in both arms because prompt
rotation changes acceptance per cycle; the paired ratio is stable (1.111-1.115).

Reading: single-stream plain decode +8.7-9.7% (FFN fusion + in-place recurrent state), single-stream
MTP +11%, and the multi-column cases gain most. MTP at C4 (8-token verify) nearly doubles and is no
longer slower than plain C4 (57.4 vs 64.1 aggregate; it was 29.5 vs 49.2).

## ABBA 2: M5 stack (A) vs M5 stack + tensor path (B)

Frozen snapshot `snap-abba2` (diff SHA-256 91045fe8…). Same protocol as ABBA 1; 18/18 quartets accepted
first attempt, 0 token mismatches. Tensor path took every n=5..8 PTQ1 product (and displaced the fused
FFN at those widths).

| Cell | A | B | Paired speedup | Range |
|---|---:|---:|---:|---:|
| pp4 (control, n=4 unaffected) | 78.75 | 78.67 | 0.999x | 0.995-1.002 |
| pp5 | 78.24 | 60.83 | 0.777x | 0.763-0.786 |
| pp6 | 87.85 | 72.52 | 0.825x | 0.818-0.831 |
| pp7 | 83.32 | 84.61 | 1.015x | 1.015-1.016 |
| pp8 | 92.88 | 96.27 | 1.036x | 1.033-1.039 |
| server MTP C4 (8-column verify) | 58.13 | 63.47 | 1.092x | 1.067-1.136 |

Reading: the tensor kernel costs about the same at any width (flat ~138 µs on ffn_up in isolation) while
the scalar 3+3 tiles are cheap at 5-6 columns, so it only wins at 8. The 2.2x kernel-level gain at n=8
shrinks to 1.04-1.09x end to end: it replaces the fused FFN kernel, pays a hi/lo pre-pass and two
barriers per projection, and the matvecs are only part of an 8-token step. Default minimum width
changed to 8 (`GGML_METAL_PTQ1_TENSOR_MIN`). Kept as an M5/A19 opt-in; not part of the portable stack.

## Round 2: other Bonsai models and prefill

### Which ideas apply to which model

| Model (weights) | Multi-column kernel | Fused FFN | Activation staging | Rows-mode state | Small projections |
|---|---|---|---|---|---|
| Bonsai 2 PTQ1_0 | done (round 1) | done | done | done | alpha/beta are BF16, 4.2 µs, not needed |
| Bonsai 2 PQ2_0 | new, n=2 only | new | not ported | applies unchanged | BF16, not needed |
| Bonsai 1 ternary (PQ2_0) | new, n=2 only | new | not ported | applies unchanged | new (PQ2_0 alpha/beta) |
| Bonsai 1 binary (Q1_0) | already in PrismML (nr1 2..4, M5-Pro-tuned) | not ported | n/a | applies unchanged | new (Q1_0 alpha/beta) |

All four are `qwen35`, so the delta-net state change needs no kernel work. Per-op profiles of one decode
token (serialized): the state gather + copy-back is 10.4% of Bonsai 2 PTQ1, 10.6% of Bonsai 1 ternary and
14% of Bonsai 1 binary. The 48-row alpha/beta projections are 5% of both Bonsai 1 models (3 threadgroups).

### PQ2_0 port

`kernel_mul_mv_pq2_0_mc<nr0,nr1,GLU>` reuses PQ2's base-4 collapse, with the three floors per byte shared
across columns. Unlike PTQ1, PQ2's generic mul_mv_ext path is already competitive from 3 columns (the
2-bit decode is cheap). Kernel sweep: 1.15-1.54x at n=2, mixed at n=3, 0.76-0.95x at n>=4. The model-level
profile decided it: Bonsai 2 PQ2 decode graph -4.5% at n=1 (fused FFN 120.5 -> 104.6 µs/layer), -15% at
n=2, but +6..10% at n=3. Default range is therefore n<=2 (`GGML_METAL_PQ2_MC_MAX`), 1 simdgroup, 2 rows.
Correctness: strict 1e-8 checker 52/52 (50 cases verifiably take the new kernel), 72 decode-shape
fixtures, 85 upstream PQ2 cases, 44 fusion cases; full-model logits on Bonsai 2 PQ2 and Bonsai 1
ternary: 0 token mismatches, NMSE <= 3e-10.

### Small projections (`GGML_METAL_SMALLM`)

One row per simdgroup for single-column products with <= 256 rows. Q1_0 5120x48: 6.28 -> 4.31 µs;
PQ2_0: 7.35 -> 4.46 µs. Bit-identical on Bonsai 1 binary (NMSE 0).

### Prefill: the CUDA ideas do not transfer

PTQ1 pp512: 82% of GPU time is the tensor-API mul_mm (ffn_up/gate 2.09 ms each, ~22 T MAC/s).
- Whole-block, byte-once trit decode (`GGML_METAL_PTQ1_MM_B128`, CUDA PR #214's "uniform trit unpacking"):
  correct (48/48 prefill fixtures incl. odd widths, 101/101 PTQ1 cases) but 3-8% slower on every shape
  (16 KB threadgroup tile costs occupancy; upstream's decode already overlaps the matmul). Rejected.
- Tensor-unit ceiling for this tile (64x128x32, A half in threadgroup memory): B float from device 24.9 T,
  B half from device 19.8 T, B staged in threadgroup memory 19.9-21.1 T. Upstream already uses the fastest
  arrangement and runs at ~88% of it. Wider tiles / half activations would not help.
- Remaining prefill headroom is outside the matmuls: chunked delta-net 6%, the 48-row BF16 alpha/beta via
  mul_mm 3% (4 threadgroups at n=512), SWIGLU + CONT ~4%.

### Small-row routing at prompt lengths (`GGML_METAL_SMALLM_MM`)

The 48-row ssm_alpha/ssm_beta projections reach mul_mm once a batch exceeds 8 tokens (16 for Q1_0), and
mul_mm's 64x128 tiles give them ceil(n/128) threadgroups. Keeping them on the mat-vec kernels (which
tile over tokens): BF16 at 9 tokens 147.7 -> 6.0 µs, 64 tokens 152.6 -> 22.2, 512 tokens 195.8 -> 141.3;
PQ2_0 at 9 tokens 135.0 -> 7.9, 512 tokens 176.2 -> 51.0; Q1_0 at 17 tokens 124.2 -> 10.3, 512 tokens
162.4 -> 36.0. Restricted to 2-D weights with K >= 1024.

**Gate exception, documented rather than loosened.** Full-model logits at ubatch 32 differ from the
baseline by NMSE 1.3e-8 (Bonsai 2 PTQ1 / PQ2) and 2.3e-8 (Bonsai 1 ternary), above the 1e-8 research
bound; tokens and text identical, max |diff| <= 0.0088; Bonsai 1 binary unaffected (NMSE 0). Against a
double-precision reference on the same 48x5120 product (`check-smallrows`), the baseline tensor mul_mm
has NMSE 1.3-1.4e-7 while the mat-vec path has ~1e-14 (BF16) / ~2e-11 (PQ2_0). The logit shift is the
baseline's own error being removed.

### Adversarial review round 2

One confirmed defect, fixed: the PQ2 staging helper took the activation offset as `short`, so any PQ2
product with K >= 32896 would read activations from the wrong place (not reachable with Bonsai's K <=
17408). Regression: strict checker case K=33024 fails with the bug restored (NMSE 8e47) and passes with
the fix (2.2e-11). Note: test-backend-ops' MUL_MAT tolerance (NMSE 5e-4) passed the corrupted kernel;
the strict checker and the full-model logit gate are the guards that matter. The reviewer confirmed the
PQ2 arithmetic is identical to the single-vector kernel, every requested variant is instantiated, and
the b128 prefill kernel mirrors the generic tensor kernel's batch/broadcast handling.

## ABBA 3: the other models (baseline A vs applicable M5 changes B)

Frozen snapshot `snap-abba3` (diff SHA-256 edcc0661…). 21 cells x 3 quartets + PTQ1 small-row cells; every
quartet accepted on the first attempt, 0 rejected, 0 A/B token mismatches in any server slot.

| Cell | Bonsai 2 PQ2 | Bonsai 1 ternary (PQ2) | Bonsai 1 binary (Q1_0) |
|---|---:|---:|---:|
| tg128 | 1.109x | 1.116x | 1.136x |
| pp2 | 1.252x | 1.250x | 1.104x |
| pp3 | 1.038x | - | - |
| pp32 | 1.046x | 1.054x | 1.053x |
| pp512 | 1.042x | 1.038x | 1.035x |
| server plain C1 | 1.107x | 1.112x | 1.109x |
| server MTP C1 | 1.157x | n/a | n/a |
| server plain C2 | 1.334x | 1.329x | 1.168x |
| server MTP C2 | 1.023x | n/a | n/a |

B = `PQ2_MULTICOL PQ2_GLU GDN_ROWS_PLAIN SMALLM SMALLM_MM` for the PQ2 models, `GDN_ROWS_PLAIN SMALLM SMALLM_MM`
for Q1_0. Bonsai 2 PQ2 MTP C1 is now 52.3 tok/s vs 47.7 plain (on M1 MTP was slower than plain for PQ2).
PTQ1 small-row routing on top of the round-1 stack: pp16 1.040x, pp32 1.018x, pp128 1.020x, pp512 1.002x,
server C1 plain/MTP 1.004x/1.001x.

## int8 tensor path: rejected on evidence (no flag built)

Prefill-shaped tensor tile (64 rows x 128 columns, A threadgroup, B device): half x float (current)
28.1 T MAC/s; int8 x int8 without scaling 32.1 T (1.14x ceiling); int8 with the per-128-block rescale that
Bonsai's per-block weight scales require 19.0 T (0.68x); per-64 9.1 T, per-32 (CUDA Q8_1 granularity)
4.3 T. Rescaling int32 partials every block stalls the accelerator. Avoiding it would need re-quantized
weight scales on top of int8 activations for at most a 14% matmul gain. Decode is bound by trit
unpacking, which int8 does not reduce.

## Bonsai 1 binary (Q1_0) follow-ups

- **Fused FFN** (`GGML_METAL_Q1_GLU`): decode graph -3.3% (n=1), -6.0% (n=2), -6.8% (n=3), +4.5% (n=4) against
  PrismML's multi-column kernels; default n<=3. Full-model logits: 0 token mismatches, NMSE 1.5e-10.
- **PrismML's existing Q1_0 options on M5 Max** (llama-bench, two interleaved passes): popcount bit-plane
  path (`GGML_METAL_Q1_0_POPCNT=1`, off by default upstream) +16% pp4, +26% pp8, +34% pp16, neutral at pp2.
  Forced 3/4-column tiles and the extended mv_ext path are 20-60% slower; defaults are otherwise right.
  Popcount stores activations as int8 bit-planes: strict checker 2/52 (per-product NMSE up to 1.6e-4),
  full-model logits NMSE up to 3.1e-5 with identical tokens; WikiText-2 (8 x 512, batch 8) PPL ratio
  0.99992 +/- 0.00067, mean KLD 0.00037, same top token 99.07%. Quality-neutral opt-in.
- Popcount and the fused FFN compose (64/64 layers fuse with both on).

## Testing correction (important)

zsh does not word-split an unquoted `$cfg`, so loops of the form `env $cfg ...` with multi-flag strings set
only the first flag (to a value like "1 GGML_...", which atoi reads as 1). Affected: several "full stack"
test-backend-ops suites and strict-checker runs reported earlier, and the first popcount/fusion composition
profile (which wrongly showed fusion disabled). Not affected: every full-model logit gate and every ABBA
study (environments passed as JSON), kernel sweeps and profiles (arguments passed separately), single-flag
runs. The whole matrix was re-run in bash with arrays (`correctness-matrix.sh`, output
`correctness-matrix.txt`), with the loaded-kernel list checked per configuration: all 12 configurations
pass (PTQ1 101/44/72 + strict 52/52; PQ2 100/48/72 + strict 54/54; Q1_0 157/88 + strict 52/52; BF16
152), and every new kernel appears under the configuration meant to exercise it.

## ABBA 4: MTP draft length and Bonsai 1 binary follow-ups

Frozen snapshot `snap-abba4` (cb66810e…). Every quartet accepted first attempt.

**Draft length (single request, server MTP):** one draft stays best on M5.

| Model | A: 1 draft | B | Paired | Range | Acceptance A / B |
|---|---:|---:|---:|---:|---|
| Bonsai 2 PTQ1, B = 2 drafts | 55.66 | 50.34 | 0.897x | 0.801-0.982 | 76.4% / 67.3% |
| Bonsai 2 PTQ1, B = 3 drafts | 55.74 | 42.40 | 0.749x | 0.626-0.862 | 76.4% / 55.5% |
| Bonsai 2 PQ2, B = 2 drafts | 52.50 | 48.20 | 0.911x | 0.812-1.000 | 76.4% / 67.3% |

Cheaper multi-column verification does not outrun the falling per-draft acceptance. (PTQ1 and PQ2 are the
same ternary weights in two packings, so their outputs and acceptance counts match.)

**Bonsai 1 binary: A = rows + small-row routing, B = A + Q1_0 fused FFN + PrismML popcount path.**

| Cell | A | B | Paired | Note |
|---|---:|---:|---:|---|
| tg128 | 75.81 | 75.59 | 0.997x | fused FFN only: no gain |
| pp2 | 105.55 | 104.41 | 0.989x | fused FFN only: no gain |
| pp4 | 130.58 | 152.62 | 1.169x | popcount |
| pp8 | 152.38 | 194.34 | 1.275x | popcount |
| pp32 | 205.46 | 205.28 | 0.999x | >16 columns: neither path |
| server C1 | 67.71 | 68.81 | 1.016x | |
| server C2 | 93.30 | 91.80 | 0.984x | 16 slots with different tokens |
| server C4 | 104.14 | 119.03 | 1.143x | 20 slots with different tokens |

- Q1_0 fused FFN: rejected. Its per-op profile gain (-3..-7% GPU time) does not appear end to end.
- Popcount: real gains at 4+ columns, but it changes greedy output under concurrency (int8 bit-plane
  activations; ~99% per-step top-token agreement compounds over 128 tokens). Opt-in with that caveat only.

## Batch-invariant mode (`GGML_METAL_BATCH_INVARIANT=1`, the CUDA PR #218 flag)

- Stock PrismML is not batch-invariant: 0/32 positions bitwise identical between one-token decode and
  2..4-token batches (NMSE ~3e-11), so MTP verification sees slightly different logits than plain decode.
- `check-divergence` (per-node capture through the eval callback) found only two sources on Bonsai 2
  PTQ1: the PTQ1 matvecs (separate single-vector kernel at n=1) and the 48-row BF16 ssm_alpha/beta
  (mul_mv_ext at 2..8 columns). The delta-net, attention, norms, rotations and conv are already
  invariant once their inputs are (the chunked and autoregressive delta-net kernels agree bit for bit).
- The mode routes n=1 PTQ1 through the same multi-column template, keeps 1..4-column float products on
  the per-column mat-vec kernel, and turns off activation staging.
- Result: **whole-model bitwise invariance** on Bonsai 2 PTQ1, 64/64 positions, full vocabulary, max
  |diff| exactly 0, at batch sizes 2, 3 and 4, for two prompts, both for MULTICOL alone and for the full
  stack (tiles, fused FFN, rows mode, small-row routing). With the mode off: 0/64. Kernel level: batch
  error 0 for every n=1..8 with 5..8 tiles. This goes beyond the CUDA flag, which covers 1..4 columns
  and is documented as not whole-model.
- Scope: PTQ1_0 + float weights. PQ2_0, Q1_0 and the tensor path are not covered.
- Bug found on the way: with the mode on, the staging predicate matched n=1 and requested a staged
  one-column kernel that does not exist (segfault). Fixed: staging requires n>=2 and is off in this mode.

### Invariance: cost, review round 3, and long contexts

**Cost (ABBA 5, snapshot `snap-abba5` 62712823…; A = full PTQ1 stack, B = A + invariant mode; all
quartets first attempt, 0 token mismatches):** tg128 0.994x, server plain C1 0.990x, pp2/pp4/pp8
0.879x/0.882x/0.887x, server MTP C1 0.895x, MTP C2 0.871x. Mostly staging being off. MTP with invariance
(49.8 tok/s) still beats plain decoding (45.1).

**Adversarial review round 3:** no crash/bounds/dispatch/allocation defects; every requested pipeline is
instantiated; Q1_0 fused kernel, small-row routing and the PQ2 offset fix verified. Scope findings, all fixed:
- S1: flash attention picked per-query-count tuned configs (M5 table) once the KV length reached 1024, and
  its simdgroup count grows with KV length (thresholds at 2048/4096), so the short-context proof did not
  cover long contexts. Invariant mode now uses the baseline attention config and a fixed simdgroup count.
- S2: the mode did nothing for PTQ1 unless `GGML_METAL_PTQ1_MULTICOL=1` was also set; it now implies it.
- S3: the tensor path was not disabled in the mode; now it is.

**Long contexts after the fixes:** 1134-token prefix and a 2011-token prefix decoding across the 2048
threshold, 40 positions each: bitwise identical at batch sizes 2 and 4 (40/40). Mode off: 0/40 (NMSE up
to 1.2e-9 at 1134 tokens, where the tuned attention configs differ). Invariance correctness matrix
re-run: PTQ1 101, BF16 152, F16 427, fusion 44, fixtures 72, strict 52/52 (batch error 0 for n=1..8 with
tiles), flash attention 4809/4809.

Remaining scope limits: PTQ1_0 and float weights only (not PQ2_0, Q1_0, other quant types, MUL_MAT_ID);
float products above 4 columns; the result depends on the Metal compiler emitting identical per-column
code for the <nr0,1..4> instances under fast math, so keep `correctness-invariant.sh` and
`check-invariance` as regression gates after any toolchain or OS update.

## Headline: upstream PrismML (all flags off) vs the recommended M5 configuration

Frozen snapshot `snap-final` (diff SHA-256 4f0e4599…), build `build-final`. Bonsai 2 27B, one request,
128 greedy tokens, fresh server per observation, 3 A-B-B-A quartets per cell, all accepted first attempt,
0 token mismatches in every cell, including upstream plain decoding vs M5 + MTP (identical text).
PTQ1 config: MULTICOL, MULTICOL_MAX=8, PTQ1_GLU, PTQ1_STAGE, GDN_ROWS_PLAIN, SMALLM_MM.
PQ2 config: PQ2_MULTICOL, PQ2_GLU, GDN_ROWS_PLAIN, SMALLM, SMALLM_MM.

| Comparison | Upstream | M5 | Paired | Range | Generation rate (excl. prefill) |
|---|---:|---:|---:|---:|---|
| PTQ1 upstream plain -> M5 + MTP | 41.44 | 55.51 | 1.338x | 1.241-1.404 | 45.44 -> 61.38 (1.351x) |
| PQ2 upstream plain -> M5 + MTP | 42.89 | 52.46 | 1.221x | 1.138-1.277 | 45.95 -> 57.16 (1.244x) |
| PTQ1 plain -> plain (server) | 41.34 | 45.67 | 1.105x | 1.104-1.107 | 45.31 -> 49.44 (1.091x) |
| PTQ1 plain -> plain (llama-bench tg128) | 43.38 | 47.03 | 1.084x | 1.072-1.102 | |
| PTQ1 upstream MTP -> M5 MTP | 16.58 | 55.43 | 3.345x | 3.292-3.386 | 17.09 -> 61.27 |

Upstream MTP (16.6 tok/s) is slower than upstream plain decoding (41.3), so the MTP-vs-MTP ratio is not a
user-facing claim; upstream plain -> M5 + MTP is.

## Bit-identical subset: upstream vs in-place delta-net state only (`GGML_GDN_ROWS_PLAIN=1`)

Same frozen `snap-final` build; all quartets first attempt, 0 token mismatches; the flag's logits are
bitwise identical to upstream (NMSE 0 on all four models).

| Model / cell | Upstream | Rows mode | Paired | Range |
|---|---:|---:|---:|---:|
| Bonsai 2 PTQ1 tg128 | 43.18 | 46.67 | 1.081x | 1.079-1.082 |
| Bonsai 2 PTQ1 server C1 | 41.24 | 44.28 | 1.074x | 1.071-1.078 |
| Bonsai 2 PQ2 tg128 | 45.44 | 49.58 | 1.091x | 1.084-1.099 |
| Bonsai 2 PQ2 server C1 | 43.05 | 46.85 | 1.088x | 1.087-1.091 |

Most of the single-request plain-decoding gain is bit-exact; the kernel changes add ~3 points there and
matter mainly for MTP (1.34x / 1.22x total) and concurrency.

## Quality benchmarks (M5 Max): upstream vs every model's full flag set

`tools/run-quality.sh`; logs in `results/quality/`.

- **KL divergence and perplexity:** WikiText-2 test, 20 chunks of 512 tokens, against the baseline's own
  logits (`--kl-divergence-base`). Run at batch 1 (decode kernels) and at batch 4 (PTQ1_0, Q1_0) or 2 (PQ2_0
  models), for the multi-column kernels.
- **Task accuracy:** HellaSwag (first 400 tasks) and Winogrande (1267, debiased) at default batching.
- **Flag sets:**
  - PTQ1_0 ran the maximum configuration, including the tensor path (`GGML_METAL_PTQ1_TENSOR=1`).
  - The PQ2_0 models ran the PQ2 stack.
  - Q1_0 ran its stack, plus PrismML's popcount option as a separate arm for context.

| Model | ub | PPL(Q)/PPL(base) | Mean KLD | Max KLD | Same top p | HellaSwag base / opt | Winogrande base / opt |
|---|---|---|---|---|---|---|---|
| Bonsai 2 PTQ1_0 | 1 | 1.000060 +/- 0.000060 | 0.000000 | 0.000052 | 100.000% | 75.25 / 75.25 | 73.32 / 73.32 |
| Bonsai 2 PTQ1_0 | 4 | 1.000059 +/- 0.000060 | 0.000000 | 0.000055 | 100.000% | | |
| Bonsai 2 PQ2_0 | 1 | 1.000060 +/- 0.000060 | 0.000000 | 0.000055 | 100.000% | 75.25 / 75.25 | 73.32 / 73.32 |
| Bonsai 2 PQ2_0 | 2 | 1.000061 +/- 0.000060 | 0.000000 | 0.000056 | 100.000% | | |
| Bonsai 1 ternary PQ2_0 | 1 | 1.000480 +/- 0.000383 | 0.000000 | 0.000057 | 100.000% | 74.50 / 74.50 | 71.67 / 71.67 |
| Bonsai 1 ternary PQ2_0 | 2 | 1.000480 +/- 0.000383 | 0.000000 | 0.000054 | 100.000% | | |
| Bonsai 1 binary Q1_0 | 1 | 0.999999 | 0.000000 | 0.000064 | 100.000% | 67.50 / 67.50 | 68.90 / 68.90 |
| Bonsai 1 binary Q1_0 | 4 | 0.999999 | 0.000000 | 0.000064 | 100.000% | | |
| *Q1_0 + PrismML popcount (theirs)* | 4 | *1.000374 +/- 0.000462* | *0.000403* | *0.033525* | *99.059%* | *67.50* | *68.90* |

- The same top token in 100% of positions on every model and batch size, and identical task scores: the
  flags do not change what the models compute beyond float rounding.
- The PPL(base) of the two Bonsai 2 files agrees to 4 decimals (9.4936 / 9.4935): PTQ1_0 and PQ2_0 are the
  same ternary weights in two packings.
- The Bonsai 1 ternary ratio (1.00048 +/- 0.00038) is within 1.3 standard errors of 1.


## PTQ1 small batches and 2 requests vs upstream (M5 Max)

Upstream (no flags) vs the recommended PTQ1 flags, same protocol as the headline (3 A-B-B-A quartets, 8 s
cooldown, AC power, no thermal warnings), revision `9f7a364` with a clean library tree. Results are in
`results/fill-ptq1-ppk-2requests/`. Every quartet was accepted on the first attempt.

| Cell | Upstream | Flags | Paired | Range |
|---|---:|---:|---:|---|
| pp2 | 20.6 | 79.3 | 3.85x | 3.83-3.87 |
| pp4 | 37.0 | 78.4 | 2.12x | 2.12-2.13 |
| pp8 | 40.5 | 92.8 | 2.29x | 2.29-2.29 |
| llama-server, 2 concurrent requests | 17.9 | 62.2 | 3.43x | 2.94-3.71 |

Upstream's PTQ1_0 multi-token path is slower than its own single-token decode. Two concurrent requests
total 17.9 tok/s, less than half of one request (41); the multi-column kernels are what fix it. The
generated text was identical in every slot. The iPhone 17 Pro Max shows the same pattern: 4.19x / 2.44x /
2.35x at pp2 / pp4 / pp8, preliminary, 2 quartets.

## M1 regression repair and final device study

The M1 Ultra investigation restored the MTP performance lost with M5-selected staged geometry, specialized complete PTQ1 tiles, and fixed scratch alignment and research-profile lifetime hazards. See [the M1 experiment log](../m1/EXPERIMENTS.md) for controlled/rejected candidates and [the final 30-cell results](../m1/results/final-device/RESULTS.md). The shared chart preserves historical M5 and preliminary phone evidence and adds the M1 column; its separate M1 profile uses STAGE=0. The final merged kernels/safety fixes still require M5/phone revalidation before their historical gains can be attributed to this revision.

## M1 changes re-checked on M5 Max (no change)

After the M1 session's library commits (`7414230`, `dad539c`, `6f068ec`): the correctness matrix is
identical to the earlier one (12/12 configurations, same kernel variants; the M1 4-row default is
GPU-family-7 only), and a paired A-B-B-A of the pre-M1 build (`754d1fb`) against the current build with
the PTQ1 flags in both arms gives tg128 1.005x, pp2 0.989x, pp4 0.994x, pp8 1.000x, 2 requests 0.992x,
MTP 1 request 0.998x (3 quartets each, all accepted first attempt, identical tokens in the server cells).
Results in `results/m1-change-check/`.

## Q1_0 K32 prefill (`GGML_METAL_Q1_MM_K32_ALIGNED`, `GGML_METAL_Q1_SWIZZLE_LOG`)

Ported from the earlier Bonsai 1 kernel-tuning work (`~/experiments/ktune`, historically -4.8% cold prefill,
bitwise; TODO item 23). `kernel_mul_mm_q1_0_f32_k32` is the tensor-API `kernel_mul_mm` specialised for Q1_0
products made only of full tiles: a static K=32 `matmul2d` descriptor, fixed 32 x 128 operand views and a
whole-tile store, no bounds handling. The same threads dequantize the same 16-weight chunks in the same K
order, so the result can be (and on M5 is) bitwise equal to the generic kernel. `SWIZZLE_LOG=L` (1-3,
implies K32) also groups 2^L adjacent row tiles on the grid's x axis so they run on the same activation
columns.

**Eligibility, per product:** Q1_0 x F32 -> F32 on a device with tensor units, all contiguous, and
K % 32 == 0, M % 64 == 0, N % 128 == 0; otherwise the whole product stays on the generic kernel (swizzle
also needs M/64 % 2^L == 0). In Bonsai-27B-Q1_0 every Q1_0 projection qualifies at 128- and 512-token
micro-batches (FFN, attention, delta-net in/out, and the output head when every position has logits) except
the 48-row `ssm_alpha`/`ssm_beta` (small-row path). A micro-batch whose size is not a multiple of 128 (the last
one of most prompts, continuous-batching steps) gets nothing; see TODO 23.

**Safety:** the kernels live in their own Metal library (`kernels/mul_mm_q1.metal`), marked optional: if a
device compiler rejects it, the backend loads without it and a selected K32 kernel falls back to the generic
one with a warning, remembered for the process (tested by injecting a compile error; `results/q1-k32/
library-and-fallback.txt`). Cold build of that library on M5: 0.13 s, in parallel with the 0.74 s `mul_mm`
library, so startup is unchanged; its pipelines are created only when a flag selects them.

**Correctness (M5 Max, final build):**

- `tools/check-q1-mm.cpp`: 17 products x 4 flag settings, each in a fresh process (so the pipeline log names
  the kernel that ran: `loaded kernel_mul_mm_q1_0_f32_k32...`), bitwise equal to flags off, guard buffer after
  the output untouched. Shapes: all Bonsai-27B projection shapes including the 248320-row output head, a
  one-column-tile product, M/64 = 3, batched, broadcast (dims 2 and 3), and four tail shapes that must stay
  generic.
- `tools/check-q1-model.cpp`: full model, 700 WikiText tokens as one prompt with logits at every position (a
  512-token micro-batch on K32, a 188-token one on the generic kernel): Q1 stack vs + K32, + swizzle 1 and a
  repeat of the stack, and flags off vs swizzle 1 alone: 0 of 173,824,000 float logits differ in every arm,
  with the kernel each arm introduced identified.
- Correctness matrix unchanged; its test-backend-ops shapes are too small to reach K32, so it only shows
  that the flags leave everything else alone.
- Bitwise equality is a property of this device's compiler, not a guarantee: re-run both checkers on other
  devices and OS versions. (An earlier llama-perplexity KL check sits at that tool's 16-bit storage floor,
  max KLD 5.2e-5 for the same configuration run twice as for swizzle 1: `results/q1-k32/kld-prefix-build.txt`.)

**Speed (M5 Max, Bonsai-27B-Q1_0, llama-bench `-fa on`, ubatch 512, depth 0, A-B-B-A, 3 quartets, all
accepted first attempt; tokens/s A -> B, per-quartet range):**

| Comparison | pp512 | pp128 | tg128 |
|---|---|---|---|
| Q1 stack -> + swizzle 1 | **1.071x** (887.7 -> 950.5; 1.069-1.072) | **1.082x** (643.5 -> 696.2; 1.079-1.084) | 0.999x (0.995-1.007) |
| + swizzle 1 -> + swizzle 2 | 0.989x | 0.999x | |
| + swizzle 1 -> + swizzle 3 | 0.978x | 1.001x | |
| flags off -> Q1 stack + swizzle 1 | **1.109x** (856.7 -> 950.3) | **1.137x** (613.1 -> 697.3) | 1.132x (63.7 -> 72.2) |
| Q1 stack -> + K32 (no swizzle; build before the review fixes) | 1.056x | 1.085x | discarded |
| same, kernels in the optional library (final source) | **1.070x** (888.1 -> 950.1; 1.069-1.071) | **1.081x** (643.8 -> 696.2; 1.080-1.082) | |

- Swizzle 1 is the best grouping; it adds about 1.4% at pp512 over K32 alone (two separate studies, not a
  paired comparison) and nothing at pp128, where there is one column tile.
- Decode never reaches `mul_mm` (one token per step), so tg128 is unaffected by construction; the K32-only
  study's tg128 was disturbed by drift within its `-r 3` invocations (72 -> 54 tok/s) and is not reported.
- Results in `results/q1-k32/`.
