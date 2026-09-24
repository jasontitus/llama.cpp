# Bonsai on Apple M5: faster Metal decoding with identical output

Downstream research on PrismML's llama.cpp fork (base `0324c66`), measured on an Apple M5 Max.
It ports the CUDA small-batch work from [PrismML PR #218](https://github.com/PrismML-Eng/llama.cpp/pull/218)
to Metal and adds M5-specific findings. **Every change is an opt-in environment flag**; with no flags set,
behaviour is upstream's.

## Headline (paired against upstream on the same machine)

Bonsai 2 27B, one request, 128 greedy tokens, llama-server, tokens/s including the short prompt:

| | Upstream plain decoding | M5 flags + MTP (1 draft) | Speedup | Output |
|---|---:|---:|---:|---|
| Bonsai 2 PTQ1_0 | 41.4 | **55.5** | **1.34x** (1.24-1.40) | identical text |
| Bonsai 2 PQ2_0 | 42.9 | **52.5** | **1.22x** (1.14-1.28) | identical text |

Generation rate alone (excluding prompt processing): PTQ1 45.4 -> 61.4 tok/s (1.35x), PQ2 46.0 -> 57.2 (1.24x).
Upstream's own MTP is slower than its plain decoding on Metal (16.6 tok/s for PTQ1) because two-token
verification falls back to a generic kernel; the fast multi-column kernels are what make MTP pay off.

### Pick your guarantee

| Guarantee | Flags | Single-request speedup vs upstream |
|---|---|---|
| **Bit-identical to upstream** | `GGML_GDN_ROWS_PLAIN=1` | plain decoding +7-9% (PTQ1 1.07-1.08x, PQ2 1.09x); Bonsai 1 binary +14% with `GGML_METAL_SMALLM=1` |
| **Identical text in every test** | recommended set below + MTP | 1.34x PTQ1, 1.22x PQ2 |
| **MTP bit-identical to plain decoding** | recommended set + `GGML_METAL_BATCH_INVARIANT=1` | ~1.2x PTQ1 (the mode costs ~11% on multi-token steps) |

"Identical text in every test": every paired study compared the generated token IDs of both arms; none
differed. Logits differ from upstream in the last bits (NMSE ~1e-10), as with any kernel change;
upstream itself is not bitwise stable across batch sizes.

### Recommended flags per model

| Model | Flags | Measured vs upstream |
|---|---|---|
| Bonsai 2 PTQ1_0 | `GGML_METAL_PTQ1_MULTICOL=1 GGML_METAL_PTQ1_MULTICOL_MAX=8 GGML_METAL_PTQ1_GLU=1 GGML_METAL_PTQ1_STAGE=1 GGML_GDN_ROWS_PLAIN=1 GGML_METAL_SMALLM_MM=1` (+ `GGML_METAL_PTQ1_TENSOR=1` for 4 concurrent MTP requests) | plain +10.5%, with MTP 1.34x |
| Bonsai 2 PQ2_0 | `GGML_METAL_PQ2_MULTICOL=1 GGML_METAL_PQ2_GLU=1 GGML_GDN_ROWS_PLAIN=1 GGML_METAL_SMALLM=1 GGML_METAL_SMALLM_MM=1` | plain +11%, with MTP 1.22x, 2 requests +33% |
| Bonsai 1 ternary (PQ2_0) | same as Bonsai 2 PQ2_0 | decode +11.6%, 2 requests +33% |
| Bonsai 1 binary (Q1_0) | `GGML_GDN_ROWS_PLAIN=1 GGML_METAL_SMALLM=1 GGML_METAL_SMALLM_MM=1` | decode +13.6% (tg128, where only the two bit-identical flags act); server single request +10.9% |

`GGML_METAL_SMALLM_MM` changes prompt processing of the 48-row projections: it is not bit-identical to
upstream, but closer to exact (NMSE vs a double-precision reference ~1e-14 instead of ~1e-7).

MTP: `llama-server --spec-type draft-mtp --spec-draft-n-max 1` with a model that carries an MTP head
(the Bonsai 2 models were grafted with the recipe from
[sudoingX/bonsai2-small-gpu](https://github.com/sudoingX/bonsai2-small-gpu/tree/eb52d9d7363cda2d910146f4e37f4b8c64c30c46/graft)).
One draft token is best on M5; 2 and 3 drafts measured 0.90x and 0.75x.

Optional, changes output: `GGML_METAL_Q1_0_POPCNT=1` (PrismML's existing Q1_0 bit-plane path) gives Bonsai 1
binary +17-28% on 4-8-token batches and +14% at 4 concurrent requests with unchanged perplexity, but
int8 activations change greedy text under concurrency.

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
- **M5 tensor units:** `matmul2d` runs half at ~33 T MAC/s and int8 at ~64 T. A tensor-unit matvec with
  hi/lo-split activations (NMSE 4e-13) wins only at 8 columns (+9% for 4 concurrent MTP requests).
  Upstream's tensor prefill already runs at ~88% of the best arrangement measured; CUDA's prefill ideas
  (#214) and an int8 path were tested and rejected with measurements.

## Method

- Apple M5 Max (40-core GPU), 128 GB, macOS 26.6.2, AC power, High Power mode.
- A-B-B-A quartets, 3 per cell, reproducibly shuffled order, 8 s cooldown, fresh process per observation,
  one GPU process at a time, 20% within-arm spread gate; rejected quartets kept (none in the final studies).
- Every timing run used a frozen snapshot build; the committed code is byte-identical to the snapshot
  behind the headline (`snapshots/snap-final.diff`, SHA-256 4f0e4599...).
- Short prompts (~30 tokens) and 128 generated tokens. Long-context throughput is not claimed.

## Correctness evidence

- Strict checker (NMSE <= 1e-8 vs a double-precision reference, plus batch invariance) for PTQ1_0, PQ2_0 and
  Q1_0, including 4163-row tails, n = 1..8 and K = 33024; test-backend-ops MUL_MAT, fusion and flash
  attention suites; every new kernel verified to execute in the configuration meant to exercise it
  (`results/correctness-matrix.txt`).
- Full-model teacher-forced logits vs upstream on all four models, ubatch 1..8: 0 token mismatches, NMSE <=
  4.2e-10 (the M1-validated candidate measured 4.6e-10).
- Batch invariance: 64/64 positions bitwise identical at batch sizes 2/3/4, also across 1134- and
  2011-token contexts.
- Three adversarial code reviews; all findings fixed (see EXPERIMENTS.md).
- Quality benchmarks (KL divergence on WikiText-2 at batch 1 and 2/4, HellaSwag, Winogrande): **pending**.

## Files

- `EXPERIMENTS.md`: full experiment log, including rejected ideas and the measurement corrections.
- `TODO.md`: open items.
- `results/`: per-study summaries and progress logs (paired ratios, ranges, token checks).
- `snapshots/`: the exact diff behind each benchmarked build.
- `tools/`: ABBA harness, full-model gate, invariance probes, strict checkers, standalone kernel bench,
  per-op profiler summary. Scripts assume the layout of the M5 handoff package (models under
  `workspace/work`), so adjust paths for other setups.
