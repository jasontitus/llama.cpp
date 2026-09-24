# From the original CUDA work to Apple GPU experiments

Prepared September 23, 2026. This is an experiment map, not a claim that NVIDIA launch geometry or thresholds transfer to Apple GPUs. M1 results and rejected ideas are preserved in experiment/EXPERIMENTS.md and m1-reference/.

## Source anchors

The motivating CUDA series is [PrismML PR #218](https://github.com/PrismML-Eng/llama.cpp/pull/218). It reorganizes Q8 activations, uses a dedicated PTQ1 small-column kernel with better work distribution and row reuse, holds reduction geometry fixed, provides an optional batch-invariance mode, changes small-BF16 routing, and sends batches above four columns to MMQ. Its reported RTX 3060 decode gain is about 1.5x; that is an NVIDIA measurement, not a forecast for Apple.

A pinned code reference for its four-column crossover is [285542d98d37d0f07f491cd206aefa31f1848f33](https://github.com/sudoingX/llama.cpp/commit/285542d98d37d0f07f491cd206aefa31f1848f33). Inspect ggml/src/ggml-cuda/mmvq-ptq1_0.cuh, mmvq.cu, quantize.cu and the attention/BF16 dispatch changes in that revision and PR. Fetch this reference into a separate research checkout if source inspection is needed; do not merge CUDA changes into the measured Metal checkout merely to obtain them.

The complementary large-batch work is [PR #214](https://github.com/PrismML-Eng/llama.cpp/pull/214): wider MMQ tile choices and uniform trit unpacking reduced prefill cost while leaving decode unchanged in its reported tests. Reference implementation commit: b16ac95b7eb6714899ebacda96493ae9a1fd960f. This suggests investigating decode/unpack cost and tiling separately on Metal; it does not establish the same tile widths or speedup there.

The MTP preparation recipe is pinned at [sudoingX/bonsai2-small-gpu eb52d9d](https://github.com/sudoingX/bonsai2-small-gpu/tree/eb52d9d7363cda2d910146f4e37f4b8c64c30c46/graft). The bundled graft tools and exact head/trunk/merged hashes reproduce the models already measured on M1. MTP is an end-to-end use of multi-column verification, not an automatic improvement to plain single-token decode.

Related Metal single-vector work: [PrismML PR #225](https://github.com/PrismML-Eng/llama.cpp/pull/225). It changes dense row tiling and reports numerical failures for some alternatives. Coordinate scope and recheck its current status before ordinary-decode work.

## Our proposed translation and measurements

Everything in this table is a local hypothesis or description of our existing Metal patch, rather than an Apple performance claim from the CUDA sources.

| Idea | Existing Metal position | Next bounded experiment | Evidence required |
|---|---|---|---|
| Better work distribution | PTQ1 single-vector specialization already exists; our patch adds 2-4-column specializations | Replay R4/S1 and R2/S1, then R2/4/8 with S1/2/4 if useful | Per-shape ABBA, GPU occupancy/register/spill observations, unchanged n=1 and n=8 controls |
| Share weight decoding across columns | Current candidate reuses decoded coefficients and activation transforms | Profile decode instructions and loads before changing arithmetic | Equal inputs, per-pipeline GPU trace, strict numerical/output checks |
| Share activation data across rows | Rows are an explicit parameter | Measure cache/load savings versus register pressure | Whole-model throughput and physical-memory evidence, not microbench alone |
| Rearrange activations | Metal path consumes F32, unlike the motivating CUDA Q8 layout | Treat any new packing/quantization pass as a separate change | Include transform cost, allocation lifetime, numerical error and total latency |
| Fixed reduction geometry | Candidate inherits the single-vector ordering, but compilation still changes rounding | Test identical prefixes across batch sizes and MTP verification sizes | Full-vocabulary logits, finite/bounded error, tokens, perplexity and acceptance |
| Find matvec/matrix crossover | Candidate currently handles n=2,3,4 only | Sweep n=1..8 and 16/32/64 as a separate extended study | Shape-specific crossover on each device, including fallback correctness |
| Avoid tiny-matrix routing overhead | No Metal BF16 routing change is included | First trace the small BF16/gating shapes and their share of total time | Standard tg/pp and attention controls; do not infer bottlenecks from CUDA |
| Reduce uniform unpack/tiling cost | Larger prefill is still a separate path | Profile pp512/2048 and packed PTQ1 loads | Compare tiled alternatives including conversion/storage overhead |
| Accelerated tensor operations | Not implemented by our patch | Only after portable measurements: evaluate tiled tensor math on supported hardware | End-to-end win after dequantization and staging, capability gating, M1 fallback |

Start with measured Bonsai projections K x M = 5120 x 10240, 5120 x 17408, 17408 x 5120 and 5120 x 6144. The bundled first-pass fixtures cover n=1,2,3,4,8; larger crossover sweeps must be added and recorded explicitly rather than relabeling current coverage. Run tails, padding, broadcasting and unsupported-path controls. A rows-per-group win is not evidence that eight-lane block ownership should change; a different mapping needs its own unpack/address/reduction proof and tests.

## Separate user-visible outcomes

1. Ordinary one-request generation: profile n=1 and the rest of the graph. Current multi-column kernel leaves this path unchanged. Any new single-vector comparison needs an unchanged baseline implementation.
2. One-request MTP: measure draft/verify time and acceptance together. Start at draft length one, then evaluate two/three on matched prompts in a separately paired plain/MTP experiment. A faster verify kernel can still lose after drafting, copying and acceptance costs.
3. Concurrent requests: keep concurrency 1/2/4 distinct from the number of columns processed in one kernel. Record aggregate throughput and per-request latency independently.
4. Prefill: short pp2-4, fixed pp512, long prompts and filled contexts are different workloads. Do not collapse them into a single tokens/sec number.

Keep the original CUDA comparison, our M1 evidence, and new Apple-device measurements labeled separately. Do not import CUDA batch-invariance guarantees or its four-column cutoff as universal properties of Metal.
