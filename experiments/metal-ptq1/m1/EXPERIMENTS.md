# M1 Ultra regression investigation, 2026-09-24

## What changed

The M5 research flags made PTQ1 MTP slower on M1 Ultra than the earlier multi-column PR. The main cause was the staged product kernel's M5-selected two-row geometry for narrow tiles. Forcing four rows in the same source restored most of the lost speed; a separate frozen-build comparison confirmed the correction across three prompts. Draft acceptance, generated length, and output tokens were unchanged.

Commit `7414230` makes the staged product default four rows on Apple family 7, retains the previous defaults on other families, and preserves explicit row/group overrides. It also aligns staged float4 scratch to 16 bytes and reserves the matching padding. The alignment fix matters for odd output sizes; the measured Bonsai projection sizes were already aligned.

## Controlled results

Three ABBA quartets per cell; six observations per arm. Native rates exclude prompt/request overhead. Paired changes use the geometric mean of full-request B/A ratios, not the ratio of the displayed native means.

| Comparison | Native A tok/s | Native B tok/s | Paired full-request change |
|---|---:|---:|---:|
| Pre-fix research -> fixed research, full flags, MTP | 29.37 | 38.20 | +27.24% |
| Literal previous PR -> fixed research, full flags, plain | 29.37 | 31.86 | +7.54% |
| Literal previous PR -> fixed research, full flags, MTP | 38.86 | 38.15 | -2.15% |
| Literal previous PR -> fixed research, staging disabled, MTP | 39.04 | 38.69 | -0.92% |

The pre-fix/fixed MTP quartet gains were +25.59% to +29.05%. All measured server output tokens match across and within arms. Each observation generated 128 tokens; accepted/drafted counts were 59/68, 49/78, and 57/70 for the three prompts in both arms. Changed acceptance does not explain the regression.

With full flags, standard llama-bench pp2 improved 23.02% and pp3 25.68% against the pre-fix build. Against the literal previous PR, pp2 improved 10.27% but pp3 declined 2.62%. The full stack is therefore not a universal M1 improvement. Disabling staging removes both the standalone activation prepass and fused-GLU staging; fusion remains enabled.

## Exploratory ablations

These used one quartet and one prompt on the pre-fix source. They select candidates; they are not final performance estimates.

| B configuration, A = MULTICOL only | Paired full-request change |
|---|---:|
| Full M5 flags | -21.98% |
| Staging only | -31.23% |
| Full flags, force four rows | -0.71% |
| Full flags, no staging | +0.86% |
| Fusion only | +0.27% |
| Full flags, no fusion | -30.25% |
| Full flags, neither staging nor fusion | +0.16% |

The early global-R2 control was rejected: its override also changes nonstaged kernels and cannot isolate the staged default. The accepted geometry comparison uses frozen `7ff4358` versus `7414230`, with identical full flags.

## Revisions and comparability

- Pre-fix research: `7ff43583b29278b44d770220fc1c60ff9c47c78e`, PrismML base `0324c66521960d67aa7da8687fb1453a79a6565c`.
- Fixed research: `7414230`, same base; manifests include exact research-harness diffs and binary hashes.
- Literal prior PR: `d4ca15de27a8502218832ee85caf0fdf4222db3b`, PrismML base `0781925904391351963d499cb32cd735849b06a5`.
- Flags-off results are the research-build baseline, not current PrismML HEAD.

The two PrismML bases differ in CPU warning cleanup, SYCL support, and tied-output Hadamard support. No Metal, server, speculative, or Qwen35 graph changes were found between them. These pinned PTQ models have Hadamard version 1 and an explicit output.weight, so the tied-output version-2 change is inactive. The literal PR comparison nevertheless changes revision and flags; the frozen pre-fix/fixed comparison isolates the repair.

## Review and measurement limits

Independent adversarial reviews checked the dispatch, alignment, numerical coverage, binary identity, accepted/rejected quartets, and result arithmetic. They found and corrected the invalid R2 control, missing exact-configuration validation, separate-binary resume mislabeling, omitted staged diffs, missing untracked tool provenance, and headline acceptance of mismatching output. Historical metadata has not been rewritten to claim newer instrumentation.

M1 Ultra: 64 GPU cores, 128 GB. Full Metal offload, Flash Attention, F16 KV, 16 CPU threads, 4096 context per server slot, batch/ubatch 512, one MTP draft token. Prompts are short, roughly 30 tokens. Each fresh server warms 32 tokens per slot before timing. Long contexts, phone memory limits, and broader model quality are not established by these measurements.

Codex remained open. Telemetry records device counters, relevant process CPU activity, and thermal state every two seconds. GPU counters cover only the first AGXAccelerator statistics entry; no per-process GPU attribution was available. All 484 thermal samples in the geometry/prior-PR confirmation were nominal. Passing the 20% within-arm spread gate does not prove a quiet GPU. Quartet ranges are observed ranges, not confidence intervals.

No code ran on M5 during this investigation. M5's existing row selection is retained, but final changes require M5 validation before claiming preserved M5 performance. A19/A20 tuning remains a separate device-measured phase; no unmeasured mobile defaults are introduced.

## Residual shared-kernel regression

A separate literal-PR comparison using MULTICOL only on both sides measured pp2 -0.49%, pp3 -3.03%, and MTP -0.62%. It isolated the common path from staging and GLU. Source comparison found unchanged dot arithmetic but new runtime column-count/clamp/store guards needed for partial tiles at n=5 and n=7.

A reviewed function-constant specialization removes that work for complete tiles, retaining the partial-tile path and distinguishing both variants in pipeline cache keys. Against a clean, independently rebuilt `7414230` control with identical MULTICOL/MAX=8 flags, three ABBA quartets measured pp2 +1.16%, pp3 +2.82%, and MTP +0.71%; every quartet ratio was above one. MTP native rate was 38.51 -> 38.78 tok/s. No server token IDs differed.

Exact-config strict, matrix, and fusion checks passed. Logs confirm both full=0 and full=1 pipelines execute, including n=3/5/6 and n=4/7/8 cache cases in one process. The full-model invariant mode produced bitwise-identical logits at all 64 checked positions for batch sizes 2, 3, and 4. This is evidence for the invariant mode, not a bitwise claim for ordinary optimized decoding.

The frozen control was rebuilt separately before timing. A planned binary copy was rejected before any measurements because its absolute runtime-library paths would have loaded the candidate libraries. An independent audit verified the rebuilt control's library paths and every recorded binary/source hash.

## Combined M1 configuration versus the prior PR

With the complete-tile specialization and staging disabled, the literal-PR comparison completed with all output IDs matching:

| Mode | Prior PR native tok/s | M1 candidate native tok/s | Paired full-request change | Quartet ratio range |
|---|---:|---:|---:|---|
| Plain | 29.30 | 31.96 | +8.54% | 1.078-1.093x |
| MTP | 38.83 | 38.91 | +0.35% | 0.994-1.019x |

Standard llama-bench pp2 improved 8.70% (44.52 -> 48.39 tok/s); pp3 improved 3.51% (50.55 -> 52.33). This restores MTP to roughly previous-PR performance while improving plain decoding and prefill. It does not establish a meaningful MTP speedup over that PR. The full M5 configuration's large MTP regression is avoided by the measured M1 profile.

The M1 profile uses MULTICOL=1, MULTICOL_MAX=8, GLU=1, STAGE=0, GDN_ROWS_PLAIN=1, and SMALLM_MM=1, with the normal GGML_METAL_ / GGML_ prefixes. This recommendation is scoped to the tested single-user, short-context PTQ1 model; larger concurrency still needs its own profile comparison.

These measurements used the source captured under `complete-tiles/provenance/tracked.diff` in its raw archive (kernel change later committed as `dad539c`). They predate integration of the remote flag-reload/iOS/device-table commits. Final merged-source results are recorded separately; historical measurements are not relabeled.

## Integration review of the newer remote branch

During final integration the remote advanced from `7ff4358` to `1b8d593`, adding an iOS benchmark app, context-triggered research-flag reload, and the 21-cell device-table runner. The flag reload had two independent correctness hazards: mutable unsynchronized flag caches, and a process-wide generation change that could change the flags seen by already-live contexts. A first proposed context-count/snapshot fix was rejected by independent review because graph allocations can precede or outlive a backend context. Scratch reservation must remain sufficient across those lifetimes too.

The original 55-cell replay is retained as an optional tool mode. The final all-model comparison follows the newly committed 21-cell device-table suite, plus the focused M1 single-user, total-benefit, and previous-PR studies and the flags-off regression control (30 cells total). This is a deliberate update to follow the current benchmark instructions, not a claim that the interrupted older replay finished.

## Safe context reload and allocation

Commit `6f068ec` closes the new reload hazards. Research flags are captured as an immutable profile while any Metal backend context is live; a conflicting new context is rejected. Thread-local caches follow the profile generation. GDN rows mode is captured per llama context. Scratch capacity depends on tensor shape and immutable device capabilities, not transient flags, so allocations can precede a backend or survive a profile change. Overlapping scratch requirements use their maximum, not their sum. M1 reserves no unsupported tensor scratch.

The extended existing reload checker passed retained-allocation sentinel checks for PTQ1 and Q1, same-profile multiple contexts, rejection of conflicting live profiles, unset-versus-zero semantics, and GDN state across graph rebuilding. A baseline/M1-profile/baseline/M5-profile/baseline sequence produced bitwise-equal logits in repeated baseline arms. Separate full-model invariant-mode checks again matched all 64 positions at widths 2, 3, and 4. Ordinary optimized logits are not claimed bitwise equal to baseline.

Before/after server startup checks used frozen `7414230` and the merged candidate, flags off and the M1 profile, with plain and MTP. All reported compute-buffer sizes were unchanged: main Metal 160.13 MiB / CPU 24.02 MiB, plus the MTP head Metal 136.02 MiB / CPU 14.02 MiB. These are graph allocation sizes, not peak process memory; single startup RSS samples vary and cannot establish peak-memory equivalence. The small odd-shape test explicitly reserves 1,932 scratch bytes after 49,956 PTQ1 output bytes and 432 after 84 Q1 output bytes.

Independent review found no remaining blocker in `7414230`, `dad539c`, or `6f068ec`, including their merged interactions and default flag state. M5 tensor execution cannot be validated on M1. The flags-off ABBA control measures the combined differences between `7414230` and the merged candidate, not scratch allocation in isolation.

## Final source freeze

The final 30-cell run uses `8a1bdf7` with no tracked source patch. It follows the current 21-cell M5 device suite plus seven M1 configuration/prior-PR cells and two flags-off regression cells. Initial archive `initial-replay` actually planned 43 cells and completed five; the later optional replay tool plans 55, which must not be confused with that historical run.

## PQ2 single-user interpretation

The final PQ2 study measures baseline plain at 31.97 native tok/s, optimized plain at 35.81, and optimized MTP at 28.73. Optimized plain and optimized MTP are separate randomized cells; their mean-rate comparison is descriptive. The directly paired baseline-plain -> optimized-MTP study loses 9.64%, with all three ratios below one (0.954, 0.838, 0.923), matching output tokens. Use optimized plain for this tested PQ2/M1 workload.

This is distinct from PTQ staging: PQ2 uses separate dispatch and does not enable the PTQ stage flag. Both formats have the same MTP acceptance counts on these prompts (59/68, 49/78, 57/70). PQ2 pp2 gains only 7.02% (34.66 -> 37.09 tok/s), with optimized plain tg128 at 35.96; it offers much less two-token amortization than PTQ. The result is consistent with draft-head and verification work exceeding the savings. No per-operation profile attributes exact shares, and the current suite has no flags-off PQ2 MTP cell; it does not establish a historical PQ2 MTP regression. This interpretation received independent source-and-results review.

The final binaries retain build label `4676829`: they were compiled from that revision plus the safety patch before it was committed as `6f068ec`. Commit `8a1bdf7` then changes only research scripts. The measured production code equals `8a1bdf7`; validation provenance retains the compiled production patch and binary hashes. A rebuild after committing would change the version stamp, so no rebuild was performed during timing.

## Final merged-source results (2026-09-24)

All 30 cells completed: 90 accepted ABBA quartets, no stability retries, and zero cross-arm or within-arm server-token mismatches. All 1,983 selected-configuration numerical/matrix/fusion checks passed. The 2,745 thermal samples were nominal, with no monitor errors and a maximum sampling gap of 2.132 seconds. Codex remained open and per-process GPU attribution was unavailable; this is not proof of a quiet GPU. A read-only Git fetch was performed during late measurements at the user's request, without changing the measured source or binaries.

Compared directly with the literal prior PR (`d4ca15d`):

| Workload | Prior PR | M1 profile | Paired change | Quartet ratio range |
|---|---:|---:|---:|---|
| Plain native generation tok/s | 29.28 | 31.87 | +8.29% full-request | 1.072-1.096x |
| MTP native generation tok/s | 38.65 | 39.16 | +1.32% full-request | 1.010-1.019x |
| llama-bench pp2 tok/s | 44.18 | 49.00 | +10.92% | 1.084-1.124x |
| llama-bench pp3 tok/s | 49.87 | 53.55 | +7.36% | 1.062-1.082x |

This restores MTP and shows a small positive result against the previous PR, not the large gain suggested by comparing with the inefficient flags-off MTP path. The flags-off revision control was effectively flat: plain +0.18% (range 0.992-1.010x), MTP +0.59% (0.990-1.018x). These observed ranges are not confidence intervals.

The M1-profile same-mode study measured native plain 29.33 -> 31.91 tok/s and MTP 13.69 -> 38.63. Optimized MTP is 21.1% above optimized plain by the ratio of separate-cell means. The directly paired baseline-plain -> M1-profile-MTP study improves full-request throughput 30.50% (27.08 -> 35.39); native generation is 29.37 -> 38.78. Independent repeated studies have different absolute means; retain each study's values and pairing instead of selecting the largest number.

The shared device chart uses the same M5 flags on M1, including STAGE=1, and the measured family7 R4 default. The separate M1 single-user table uses STAGE=0. PQ2 should use optimized plain for this workload; older Bonsai models have no MTP head in this experiment. The 21-cell shared device suite is complete; it does not include fresh concurrency-four or long-prefill measurements.

Latest remote `754d1fb` was merged as `bf9a24e` after timing. It changes app/research documentation and results, with no changes to the measured ggml, src, common, or tests trees. Newly recorded M5 and phone results retain their own measured revisions. No M5 code execution occurred in this task. Rebuild the phone framework before testing the merged core.

## Evidence index

- `results/final-device/`: final manifest, selected studies, acceptance analysis, correctness counts, readable results, and full raw logs in `raw.tar.gz`.
- `results/safe-reload-validation/`: retained-allocation/lifetime/GDN checks, invariant-mode output comparison, memory logs, review record, compiled patch, and binary hashes. Full-vocabulary binary files are omitted; their hashes are retained in `validation.json`.
- `results/initial-replay/`: interrupted 5/43-cell initial replay.
- `results/exploratory-ablations/`: one-quartet hypothesis selection, not final estimates.
- `results/geometry-and-pr/`: controlled geometry repair and literal-PR comparison, 7/11 cells before interruption.
- `results/no-stage/`, `results/common-path/`, `results/complete-tiles/`: later controlled diagnostic steps, with original source/patch and raw records.

Each directory includes SHA256SUMS. Historical metadata is preserved rather than rewritten to claim newer instrumentation.
