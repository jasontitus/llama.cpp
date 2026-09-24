# Experiment log

## Baseline and provenance

- Date: 2026-09-23. Machine: Apple M1 Ultra, 64 GPU cores, 128 GB RAM.
- OS: macOS 26.7 (25G229). Metal reports Apple7, SIMD reductions and matrix operations.
- Base: PrismML `d8f26eec76da6d09bb708bcba51ef64b8cd868a3`.
- Local source: `experiments/ktune-owned-m1-studio-32a893e05ee2/bonsai2-standard-pq2-1c4999f077ef/source`.
- Dedicated branch: `downstream/metal-ptq1-multicol`, isolated local clone.
- Model: official `prism-ml/Ternary-Bonsai-2-27B-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf`,
  matching the small-gpu repository's README download command. No MTP graft.
- Model URL: https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf/tree/main
- Metadata verifies the requested weight shapes. Full tensor metadata is saved with baseline results.
- Release build, Metal enabled, full GPU offload, FA on, 16 CPU threads for llama-bench.

## Baseline recorded before candidate runs

Three sequential sweeps of unchanged kernels (exploratory medians, microseconds):

| K x M | n=1 | n=2 | n=3 | n=4 | n=8 |
|---|---:|---:|---:|---:|---:|
| 5120 x 10240 | 48.99 | 227.55 | 257.96 | 280.26 | 527.21 |
| 5120 x 17408 | 69.09 | 378.65 | 407.47 | 452.64 | 877.27 |
| 17408 x 5120 | 84.05 | 410.19 | 467.60 | 554.82 | 957.06 |
| 5120 x 6144 | 32.48 | 141.66 | 176.82 | 193.91 | 339.25 |

Baseline llama-bench: pp1 28.58, pp2 16.31, pp3 22.03, pp4 25.88,
pp8 29.82, pp16 63.64, tg32 29.08 tokens/s (five internal repeats).
The sequential results motivate an experiment; final claims use ABBA below.
All 20 baseline projection CPU-reference checks passed.

### Measurement setup corrections

- The sandbox initially hid Metal. An empty-device run exited zero, but produced
  no result rows and was discarded. All reported data explicitly selects MTL0.
- The baseline CSV printer omits timing fields. Those initial CSV sweeps were
  not used as timing evidence. Console runs were repeated, saved, and parsed.
- The strict numerical test initially normalized each single scalar dot
  independently. Near-zero dots produced misleading relative errors. The test
  now pools the eight single-vector reference runs, retains absolute errors,
  and applies the same 1e-8 NMSE threshold to baseline and candidates. No kernel
  tolerance was relaxed to admit a candidate.
- F16 edge cases are unsupported by the CPU reference. They are recorded as
  skips, not successful numerical checks. The 48 F32 edge cases all execute.

## C1: reuse decoded PTQ1 weights across 2/3/4 columns

The single-vector byte mapping, collapse coefficients, and SIMD reduction
are preserved in a separate opt-in multi-column implementation. Rows per
SIMD group are 2/4/8; groups per threadgroup are 1/2/4. Pipeline cache keys
include the parameters. No architecture detection or Apple10 tuning is added.

- R4/S1: first candidate, inherited row/group geometry. Three exploratory
  sweeps plus llama-bench completed. Final ABBA evaluation targets this candidate.
- R2/S1: one exploratory sweep completed. It was generally slower than R4/S1,
  but faster on the 17408 x 5120 n=3 sample. That is insufficient to select a
  shape-specific dispatch threshold; retain it as a portable follow-up candidate.
- R8 and additional group counts: numerical validation complete, performance
  not measured in this phase. Do not infer a winner from occupancy limits alone.

Numerical validation of all nine row/group combinations: 42 deterministic
cases each, including every 2/3/4-column specialization, n=1 and n=8 controls,
K-block tails, M-row tails, and padded activation strides. All 378 checks pass.
Worst reported CPU-reference NMSE is 5.82e-11; worst cross-batch NMSE 1.79e-11.
These include baseline n=1 and n=8 paths and are not claims of bitwise equality.
R4/S1 also passes 20 full-size projection comparisons and the 48 supported
broadcast/padded edge cases; candidate-off edge cases also pass.

## Final ABBA protocol

Requested explicitly by the user after exploratory runs. A and B use one
binary and identical seeded input generation. Only the opt-in dispatch flag
changes. Per shape: A-B-B-A, repeated three times; shape order is shuffled
reproducibly between cycles. Exactly one GPU process runs at a time. Each
backend sample includes its own warmup. There are six A and six B samples
per shape. All samples are retained and n=1/n=8 are control paths.

The end-to-end suite uses the same A-B-B-A schedule, three cycles, five
internal repetitions per test case. The official model has already been
loaded before measurement. This measures small prompt batches and ordinary
generation, not MTP speculative acceptance or full application serving.

Final paired measurements and variability are recorded in RESULTS.md and
raw result directories. Initial sequential figures above are not substituted
for missing ABBA results.

## Rejected or deferred ideas

- Do not port CUDA's planar Q8 activation representation in this first candidate:
  Metal consumes F32 activations, and introducing another quantization/layout
  transform would conflate numerical changes with packed-weight reuse.
- Do not change the single-vector kernel or route n=8 into the new kernel in
  this patch. Both are controls outside the requested n=2..4 specialization.
- Do not promote R2 based on its isolated n=3 win. It needs paired confirmation.
- Do not enable a global default or create device-family thresholds from one
  M1 Ultra. The feature stays opt-in pending broader validation.
- Do not treat the early sequential performance ratios as final evidence;
  replace those claims with repeated ABBA results.

## Explicit M5 second phase

Replay the exact model, baseline, tests, and ABBA protocol on actual M5.
Start with portable R4/S1 and R2/S1, then benchmark the already parameterized
row/group choices as justified by M5 results. Preserve correctness gates and
n=1/n=8 controls. Only after measurements support it, add Apple10-specific
variants or dispatch. Nothing in this phase estimates or optimizes M5 speed.

## Current upstream HEAD follow-up

The user requested alignment with current PrismML HEAD and a clean future PR. Fetched the current default `prism` branch at `3b19c377d18157bfec39ec71bad193e9ef000cf2` on 2026-09-23. Created a new isolated downstream branch, `downstream/metal-ptq1-current`. The kernel/dispatch diff applies without conflicts. Relevant PTQ1 Metal kernels did not change between the two revisions, but model execution did, so fresh baseline/candidate measurements are required and are saved separately.

Current contributor guidance asks not to add standalone files under `tests/` without maintainer approval. The new branch therefore uses existing `test-backend-ops` fixtures and keeps standalone numerical/model helpers under the experiment directory; no new CMake test target is introduced. Nothing is submitted, committed or pushed.

Existing related PR: https://github.com/PrismML-Eng/llama.cpp/pull/225 changes single-vector dense row count to five while preserving four for MUL_MAT_ID. Our change does not alter those paths or constants, and adds a separate 2..4-column kernel. Coordinate scope with this work before submission; it is not evidence for our performance claims.

Full-model output check on current HEAD: all 195 greedy decisions and decoded outputs matched across nine prompt/microbatch combinations. Compared 48,422,400 full-vocabulary logits. Maximum absolute difference: 0.0018577576; maximum logit NMSE: 4.5893e-10. Three arithmetic-prompt cases emitted EOS immediately, so each contributes only one decision; the other six produce 32 tokens each. This is not bitwise equality and does not guarantee unchanged sampling or near-tied argmax choices for every prompt.

The older whole-suite ABBA showed movement in pp8 despite unchanged projection dispatch. To avoid cross-case thermal/work-order effects, current-HEAD llama-bench measurements use independent per-case ABBA blocks and fresh process warmup. Do not mix those results with older whole-suite timings.

## Path investigation and final review

Current PrismML already has a specialized byte-owning PTQ1 single-vector kernel: eight lanes cover each 128-weight block, activation collapse coefficients are staged once and reused across rows, and base-3 decoding uses float floors. The n=2..8 baseline instead enters the generic `kernel_mul_mv_ext_q4_f32_impl` family. That path already reuses each dequantized float4 across columns; the candidate does not invent column reuse. It extends the more efficient PTQ1-specific byte ownership and activation-coefficient calculation to several columns, sharing its decoded coefficients across them, and retains the single-vector reduction geometry. This also explains why floating-point results need not match the generic path bitwise.

Private code review found that the new output-column stride multiplication needed a 64-bit cast before multiplication. Corrected it before the final export. Existing measured shapes were far below the overflow boundary, but all final performance data and repeated correctness checks are generated from the corrected binary. No large allocation overflow regression was exercised on hardware; the type correction is verified by inspection.

Final validation reruns both baseline/candidate projection and edge fixtures, all nine parameter combinations, and 48 existing upstream PTQ1 F32 cases. Full-model logits/text are rechecked at microbatch 4 after the indexing correction; the earlier nine-case comparison covers microbatches 2, 3 and 4. Standard `llama-perplexity` compares baseline and candidate on the first two 512-token chunks of the repository-referenced WikiText-2 test corpus with batch/microbatch 4. This is a bounded quality smoke test, not a full-corpus evaluation.


Final standard perplexity smoke check: baseline 8.7487 +/- 1.12295, candidate 8.7486 +/- 1.12294. After the 64-bit indexing correction, all 65 repeated microbatch-4 greedy decisions/text match, with maximum logit absolute difference 0.0015320778 and maximum NMSE 3.8678e-10. The current upstream remote was checked again during the final ABBA run and still pointed to `3b19c377d18157bfec39ec71bad193e9ef000cf2`.


The final binary's projection speedups are 1.85–3.96x. The 17408 x 5120 n=4 result is lower than the preceding binary's run (1.85x versus 2.48x); its candidate mean increased from 236.90 to 306.76 microseconds while baseline mean decreased from 587.82 to 568.57. Pipeline-reported maximum thread count remains 384. The runs used different compiled kernel artifacts and were separated in time, so do not attribute the difference solely to the indexing cast or solely to system drift without a dedicated paired-binary investigation. This remains a useful follow-up on register allocation/code generation. All final claims use the corrected binary, with no selection of faster numbers from the older run.

## Actual MTP follow-up

The user asked to test whether the multi-column improvement speeds up one conversation through speculative decoding. Added ordinary single-token decode optimization as a separate TODO, then prepared the CUDA fork's exact MTP graft. The head and merged model match the recipe's SHA-256 values. The existing PrismML MTP Hadamard support is sufficient, so no new production-code changes were needed.

The 0/1/2/3-draft screen uses the standard llama-server with native token IDs and timings. One draft token was selected for repeated confirmation across code, prose and shell prompts. All 24 screening outputs matched ordinary greedy decoding for 128 tokens. A longer original-model versus merged-MTP check matched all 1,106 token IDs and decoded text. This does not establish universal batch invariance or identical stochastic sampling.

The first ABBA run was interrupted after a severe timing disturbance; all completed samples are retained. A fresh total-benefit run completed with every paired quartet favoring MTP, but some timing drift persisted. The second paired study isolates the kernel contribution while MTP stays enabled on both arms. See SPECULATIVE.md for the protocol and SPECULATIVE-RESULTS.md for final measured results and limitations. The M5 second-phase plan remains intact.

## Four-model standalone benchmark preparation (2026-09-23)

Advanced the downstream branch to PrismML 0324c66521960d67aa7da8687fb1453a79a6565c (CPU_REPACK rotation-tensor fix). The Metal kernel diff is unchanged. Downloaded and SHA256-verified the two older official models and their Q4_1 DSpark drafters; verified the local Bonsai 2 PQ2_0 file and grafted the same validated MTP head. All Bonsai 2 ordinary/MTP concurrency 1,2,4 smoke cases pass. Older DSpark server cases fail: ternary legacy tensor type label; binary missing target hidden-state capture, documented by upstream test-dspark-real-eval.cpp. No DSpark core fix was mixed into the patch.

An in-app ABBA attempt encountered severe transient host drift and is retained as preliminary. At the user request, stopped it and prepared family-benchmark/ plus the runnable outputs/bonsai-benchmark package. Full independent measurements are pending a Terminal launch with Codex closed. Includes three ABBA quartets, 20% within-variant spread gate, complete rejected/interrupted-run retention, token checks, concurrency 1/2/4, standard llama-bench batch tests, memory/acceptance data, basic GPU monitoring and optional administrator powermetrics. Short packaged ABBA/concurrency-two validation had equal tokens and correctly detected Codex (Service). Plot/report generation was schema-tested with synthetic fixtures stored only under work/, not used as performance evidence. M5 remains a separate measured phase.

## Completed independent four-model run review

Run 20260923-133649 completed in 90.8 minutes: 216 accepted server observations, 48 accepted llama-bench observations, no process overlap. All 378 within-quartet and 168 ordinary/MTP output comparisons matched. Codex Service absent in 2,596 samples. PTQ1_0 ordinary C2/C4 gains 2.585x/1.817x; MTP C1/C2/C4 gains 3.216x/2.189x/1.152x against upstream MTP; pp2/pp3/pp4 gains 2.776x/2.415x/1.915x. Thermal Heavy in 1,836/2,071 power samples; 10 unstable quartets retained and repeated. All per-process GPU counters were zero, even for llama, so no process-level GPU attribution is valid. Final output parity does not imply identical acceptance counts or floating-point results. Charts now disclose thermal conditions. Preserve the pp8 result as an observation requiring dispatch/order investigation; do not claim optimization of the untouched formats. Source/kernel unchanged, M5 still deferred.
