# Downstream PTQ1_0 Metal multi-column experiment

This is local downstream research on PrismML baseline
`3b19c377d18157bfec39ec71bad193e9ef000cf2`, on branch
`downstream/metal-ptq1-current`. No upstream submission, commits, or push.
The original local checkouts are untouched. The baseline has the working
Hadamard/Bonsai 2 implementation; stock upstream llama.cpp is not equivalent.

## Scope and dispatch

The candidate is opt-in with `GGML_METAL_PTQ1_MULTICOL=1`. It only handles
PTQ1_0 weights, F32 activations with contiguous elements within each vector,
and exactly 2, 3, or 4 columns. All other dispatch remains the baseline.
Unset or set the flag to 0 to compare baseline dispatch in the same binary.
This is intentionally not an M1-only or Apple10-specific dispatch rule.

The kernel retains the baseline eight-lane ownership of a 128-weight PTQ1
block and its activation coefficient transform. It decodes each packed byte
once per row/block, then reuses the decoded coefficients across columns.
It reuses the activation coefficients across rows. Per-column block traversal,
accumulation order, and SIMD reduction follow the single-vector algorithm.
That does not imply bitwise equality: compiler contraction/register decisions
and the generic fallback's summation geometry can change rounding.

The candidate has explicit row-tail load clamping and supports input tensor
strides and broadcasting in dimensions 2 and 3. The existing single-vector
and MUL_MAT_ID kernels are unchanged.

Parameters, read once per process:

| Variable | Values | Default |
|---|---|---|
| `GGML_METAL_PTQ1_MULTICOL` | 0 / 1 | 0 (off) |
| `GGML_METAL_PTQ1_NR0` | 2 / 4 / 8 rows per SIMD group | 4 |
| `GGML_METAL_PTQ1_NSG` | 1 / 2 / 4 SIMD groups per threadgroup | 1 |

Other row/group values fall back to 4/1. Column count selects the matching
compile-time specialization. Kernel and pipeline names include rows, columns,
and SIMD groups, avoiding configuration cache collisions. Eight lanes per
block is the inherited packed-byte mapping, not a detected M1 property.
Changing it requires a new mapping plus numerical tests; it is not exposed
as a knob that would silently change reduction semantics.

## Reproduce

Use separate build directories to preserve the original baseline binaries.
A Release build with `GGML_METAL=ON`, `GGML_CCACHE=OFF`, and `LLAMA_BUILD_TESTS=ON` supplies the standard `test-backend-ops` and `llama-bench` tools. Compile the optional standalone numerical check as shown below; it is not added to the upstream test suite. Run on a macOS host with Metal access; a sandbox can hide the GPU.
The exact device name here is `MTL0`, not `Metal`.

```sh
cmake -S . -B build-ptq1 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_METAL=ON -DGGML_CCACHE=OFF -DLLAMA_BUILD_TESTS=ON
cmake --build build-ptq1 --target test-backend-ops llama-bench llama-perplexity -j 12
c++ -std=c++17 -O2 experiments/metal-ptq1/check-numerical.cpp \
  -I ggml/include -L build-ptq1/bin -lggml -lggml-base \
  -Wl,-rpath,"$PWD/build-ptq1/bin" -o build-ptq1/bin/test-metal-ptq1
python3 experiments/metal-ptq1/generate-cases.py
python3 experiments/metal-ptq1/run.py --build build-ptq1 \
  --output results/baseline --model /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf
python3 experiments/metal-ptq1/run.py --build build-ptq1 \
  --output results/r4-s1 --candidate --rows 4 --simdgroups 1 \
  --model /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf
```

Output directories must be new. The runner serializes all GPU work, records
commands and environment, requires 20 timing rows, and rejects empty test
passes. Each projection is warmed up, then repeatedly executed through the
backend graph runner for at least one second. Timings include amortized backend
overhead and reuse the same tensors; they are not isolated GPU timestamps or
cold-cache whole-model memory bandwidth. Three separate sweeps provide a
median per shape. A one-sweep exploration is preliminary.

`projections.txt` covers K x M = 5120 x 10240, 5120 x 17408, 17408 x 5120,
5120 x 6144, at n=1,2,3,4,8. These shapes were verified in model metadata.
`edges.txt` defines 96 broadcast/padding/type/column combinations. The 48 F32
cases run against CPU, including n=5..8 fallbacks; the 48 F16 cases are reported
as unsupported by the CPU reference, so they are not numerical coverage. `test-metal-ptq1` uses a fixed seed and identical inputs
across batch sizes, compares CPU-dequantized weights with double-accumulated
dots, tests K=128,384,512,5120,17408, M=1,3,7,8, padded strides, zero and
alternating-sign inputs. Its NMSE bound is 1e-8, stricter than the generic
backend test's 5e-4 bound. Single-column reference errors are pooled across
all eight vectors to avoid an unstable relative error on one near-zero dot.
It reports maximum absolute error too. Tests fail if Metal is unavailable.

End-to-end runs use the official PTQ1_0 GGUF, prompt sizes 1,2,3,4,8,16,
32-token generation, five repetitions, full GPU offload, flash attention on,
and 16 CPU threads. Prompt processing is a small-batch proxy; it is not an
MTP/speculative acceptance-rate benchmark. No MTP speedup is claimed.

## Final measurement protocol

The initial sequential sweeps are exploratory. The final measurements use:

```sh
python3 experiments/metal-ptq1/abba.py --build build-ptq1 \
  --output results/abba-r4-s1 --cycles 3 --rows 4 --simdgroups 1 \
  --model /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf
```

A is the unchanged baseline dispatch; B enables the candidate in the same
binary. Each shape gets A-B-B-A, repeated over three cycles (six samples per
condition). Shape order is shuffled reproducibly between cycles. Backend
inputs use `GGML_TEST_SEED=20260923`; that optional test-runner setting fixes
the uniform data generator while retaining its original behavior when unset.
Each subprocess warms its pipeline before timing. Processes never overlap.
Each llama-bench case separately gets three A-B-B-A cycles, five internal repetitions per invocation, with its own model load and warmup. This avoids carrying work from pp2/3/4 into a later pp8 control in the same process. Its process-local default random-token sequence repeats
across A/B invocations. The model is already cached before these runs.

For each quartet, compare the mean A latency to the mean B latency. Report
the geometric mean and range of those three paired speedups, plus sample
CV for each condition. Do not count internal operation replays as independent
measurements. All samples are retained; no outlier exclusion. n=1 and n=8
are unchanged dispatch controls. ABBA reduces linear drift but cannot prove
absence of background load or thermal effects. These are one-machine results,
not a cross-device default selection or proof of speculative-decode gain.

## Second phase: M5, after M1 evidence

1. Keep the same baseline revision, model checksum, compiler options, fixtures,
   and numerical checks. Record M5 OS/toolchain/device-family metadata.
2. Repeat paired baseline/candidate runs on actual M5 hardware for the portable
   row/group candidates retained in EXPERIMENTS.md, including n=1 and n=8 controls.
3. Add whole-model small-batch and actual speculative verification measurements;
   include repeated sustained runs to separate startup, thermal, and variance effects.
4. Only if measurements show a repeatable need, introduce Apple10-specific
   tuning or thresholds, querying device capabilities rather than model names.
5. Keep a generic fallback, repeat correctness checks on both generations, and
   avoid promoting an M1 optimum to the universal default based only on M1 data.

No M5 benchmark, tuning, Apple10 variant, or architecture-specific dispatch has
been introduced in this phase. Phone-class devices need their own later runs.

## Full-model output comparison

`check-model.cpp` and `check-model.py` are standalone research checks. They save full-vocabulary F32 logits, greedy token IDs and decoded text. B is teacher-forced with A's tokens so logits are compared on the exact same prefix, even if an argmax were to differ. They test three fixed prompts at microbatch sizes 2, 3 and 4; up to 32 generated tokens per prompt. This is a targeted equivalence check, not a broad quality evaluation or a promise of bitwise determinism.

```sh
c++ -std=c++17 -O2 experiments/metal-ptq1/check-model.cpp \
  -I include -I ggml/include -L build-ptq1/bin -lllama -lggml -lggml-base \
  -Wl,-rpath,"$PWD/build-ptq1/bin" -o build-ptq1/bin/check-model
python3 experiments/metal-ptq1/check-model.py --binary build-ptq1/bin/check-model \
  --model /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf --output results/model-check
```

The isolated core patch only changes four Metal implementation files and adds an optional seed to the existing backend test runner. Fixtures, standalone helpers, logs and this experiment documentation are research material, not a proposed new test subsystem.

## Standard perplexity smoke check

Download the WikiText-2 archive referenced by `scripts/get-wikitext-2.sh` and use its `wiki.test.raw` file. Run sequentially with the feature flag 0, then 1:

```sh
GGML_METAL_PTQ1_MULTICOL=0 build-ptq1/bin/llama-perplexity \
  -m /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf -f /path/to/wiki.test.raw \
  -c 512 -b 4 -ub 4 --chunks 2 -ngl 99 -fa on -t 16
GGML_METAL_PTQ1_MULTICOL=1 build-ptq1/bin/llama-perplexity \
  -m /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf -f /path/to/wiki.test.raw \
  -c 512 -b 4 -ub 4 --chunks 2 -ngl 99 -fa on -t 16
```

This bounded check exercises the four-column candidate across longer context. The numerical and full-model helpers cover columns two and three. Do not treat two chunks as a full quality evaluation, or use these sequential runs as performance evidence. `results/manifest.json` in the archive records model/corpus checksums; final ABBA metadata records the exact binary and dynamic-library checksums.
