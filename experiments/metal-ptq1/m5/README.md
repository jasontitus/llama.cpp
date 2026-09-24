# Bonsai on Apple silicon: faster Metal decoding with identical output

Downstream research on PrismML's llama.cpp fork (base `0324c66`). It ports the CUDA small-batch work from
[PrismML PR #218](https://github.com/PrismML-Eng/llama.cpp/pull/218) to Metal, plus findings from tuning
on Apple silicon. **Every change is an opt-in environment flag**; with no flags set, behaviour is upstream's.

| Device | Status |
|---|---|
| Apple M5 Max (40-core GPU, Apple10) | **Measured**: results below |
| Apple M1 Ultra (64-core GPU, Apple7) | In progress (the portable flags; no tensor units on Apple7) |
| iPhone 17 Pro Max (A19 Pro, Apple10, 12 GB, iOS 27) | First results, from the [`BonsaiBench`](../ios/BonsaiBench) app |

All devices use the same flags, the same paired A-B-B-A protocol and the same tools (see
"Reproduce on your Mac" below), so results are comparable as paired speedups. Absolute tokens/s
differ by device.

## Results by device

Paired speedup vs upstream PrismML **on the same device** (geometric mean of three A-B-B-A quartets),
with upstream -> flags tokens/s in parentheses. Server rows: llama-server, 128 greedy tokens, short prompt;
generated text was identical between arms unless noted. Flags per model are listed under "Recommended
flags" below; the bit-identical row uses `GGML_GDN_ROWS_PLAIN=1` only.

| Result | Apple M5 Max | Apple M1 Ultra | iPhone 17 Pro Max |
|---|---|---|---|
| PTQ1: upstream plain -> flags + MTP (server, 1 request) | **1.34x** (41.4 -> 55.5) | _pending_ | n/a (no MTP in app) |
| PQ2: upstream plain -> flags + MTP (server, 1 request) | **1.22x** (42.9 -> 52.5) | _pending_ | n/a (no MTP in app) |
| PTQ1 plain decoding (server, 1 request) | 1.10x (41.3 -> 45.7) | _pending_ | n/a |
| PTQ1 plain decoding, bit-identical subset (server, 1 request) | 1.07x (41.2 -> 44.3) | _pending_ | n/a |
| PTQ1 tg128 | 1.08x (43.4 -> 47.0) | _pending_ | _pending_ |
| PTQ1 MTP -> MTP (server, 1 request) | 3.34x (16.6 -> 55.4) | _pending_ | n/a |
| PTQ1 2 requests (server) | _pending_ | _pending_ | n/a |
| PTQ1 pp2 / pp4 / pp8 | _pending_ | _pending_ | preliminary, 1 quartet each: 4.27x / 2.35x / 2.30x (2.8 -> 11.9, 4.5 -> 10.6, 5.1 -> 11.8) |
| PQ2 tg128 | 1.11x (45.5 -> 50.4) | _pending_ | _pending_ |
| PQ2 2 requests (server) | 1.33x (48.1 -> 64.1) | _pending_ | n/a |
| Bonsai 1 ternary tg128 | 1.12x (48.0 -> 53.6) | _pending_ | _pending_ |
| Bonsai 1 ternary 2 requests (server) | 1.33x (50.1 -> 66.6) | _pending_ | n/a |
| Bonsai 1 binary tg128 | 1.14x (66.7 -> 75.7) | _pending_ | **1.18x** (10.9 -> 12.8) |
| Bonsai 1 binary 2 requests (server) | 1.17x (79.8 -> 93.2) | _pending_ | n/a |

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

The iPhone column comes from the BonsaiBench app's exported JSON (tg128 and ppK cells; the app has no
server or MTP), saved in [`../ios/results`](../ios/results).
- The app's tg128 is measured like llama-bench's (128 single-token decodes from an empty context, no
  sampling), with 1 repetition per observation instead of 3.
- It used a 40 s cooldown so that the phone stayed at a nominal thermal state.

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
and may not be optimal elsewhere; the paired study will show it.

## Files

- `EXPERIMENTS.md`: full experiment log, including rejected ideas and the measurement corrections.
- `TODO.md`: open items.
- `results/`: per-study summaries and progress logs (paired ratios, ranges, token checks).
- `snapshots/`: the exact diff behind each benchmarked build.
- `tools/`: ABBA harness, full-model gate, invariance probes, strict checkers, standalone kernel bench,
  per-op profiler summary. Scripts assume the layout of the M5 handoff package (models under
  `workspace/work`), so adjust paths for other setups.
