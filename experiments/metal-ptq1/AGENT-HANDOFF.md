# Continuation context

The user requested a handoff package for M5 Max but explicitly said the M5 is busy. Do not connect to it, build there, or start measurements until the user chooses to run this package or authorizes that work. Package preparation on M1 is not an M5 benchmark.

Objective: independently validate the existing downstream Metal PTQ1_0 multi-column candidate on actual M5 Max with careful ABBA measurements and numerical/output checks. Keep it downstream on the dedicated branch; preserve a clean potential PrismML PR. Do not tune for Apple10 before portable-candidate results support it. Read the target repository's AGENTS.md and CONTRIBUTING.md before any proposed submission.

Source: PrismML-Eng/llama.cpp, pinned 0324c66521960d67aa7da8687fb1453a79a6565c. The uncommitted core patch changes four Metal implementation files and adds an optional deterministic seed to the existing backend test runner. No new upstream test files. The bundled experiment helpers are local research material. No commit, push, PR or external messages have been authorized by preparing this package.

The kernel inherits eight-lane PTQ1 block ownership, shares packed-weight decoding and activation transforms across 2-4 columns, and parameterizes rows/group geometry. It is opt-in via GGML_METAL_PTQ1_MULTICOL=1. GGML_METAL_PTQ1_NR0=2/4/8 and GGML_METAL_PTQ1_NSG=1/2/4 are supported. M1 preferred R4/S1; R2/S1 deserves paired M5 evaluation. Do not assume hardcoded device-name dispatch is necessary.

M1 findings: major PTQ1 small-column and MTP improvements, ordinary single-token throughput effectively unchanged. PQ2_0 and Q1_0 are controls. Native generation at single concurrency with candidate enabled: PTQ1 plain 27.26 tok/s versus MTP 37.18; Bonsai 2 PQ2 plain 31.36 versus MTP 24.62; older ternary plain 30.64; older binary plain 37.01. Plain and MTP cells are not directly ABBA paired, unlike kernel off/on. See the included source data and M1 report instead of treating these rounded values as new evidence.

The M1 independent run completed 216 server observations and 48 standard llama-bench observations. All 378 within-quartet output comparisons and 168 ordinary/MTP comparisons matched. Earlier full-vocabulary tests found small floating-point logit changes (maximum absolute difference 0.0018577576, maximum NMSE 4.5893e-10); the implementation is not universally bitwise invariant. Do not promise identical stochastic samples or tied argmax behavior for all prompts.

M1 timing caveats: 10 unstable quartets retained/repeated, Heavy thermal pressure in 88.7% of power samples, no Codex Service in 2596 basic samples, unusable all-zero per-process GPU accounting. Large paired gains were consistent, but small control shifts and unexpected PTQ1 pp8 gain need investigation. Never overwrite or silently omit these limitations in the M5 comparison.

MTP recipe: pinned Qwen3.8 donor head from the CUDA experiment, not a new trained head. Both Bonsai 2 merged models match pinned SHA256. Older models use DSpark; ternary has a legacy tensor-format loader failure and binary lacks server hidden-state capture. Keep that compatibility work separate from this patch.

Output handoff: whole family/projection run directories, native timings and full-request rates, output/logit checks, model/build/source pins, toolchain/device metadata, telemetry, accepted/rejected quartets, and an updated experiment log. Check current PrismML head separately before a future PR; do not silently rebase the comparison checkout mid-study. Any later rebase needs fresh checks.

Historical experiment documents in experiment/ preserve earlier stages, including older baseline references and once-pending TODOs. This handoff README and the audited m1-reference report describe the current packaged state.


## Subsequent user requests and current package behavior

The user explicitly requested fixes for all adversarial-review findings, an editable M5 optimization environment, and the information needed to extend the original CUDA ideas to A19/A20. Read REVIEW-FIXES.md, DEVELOPMENT.md, CUDA-TO-APPLE.md and DEVICE-PLAN.md before continuing.

Run-All.command now uses a persistent screen session and a combined job supervisor. Setup and testing share one inherited lock; existing validation inputs are not rebuilt on resume. Full-model numerical and output gates fail the job; telemetry failures are surfaced; copied executable helpers are pinned; interrupted preparation uses verified temporary outputs. Candidate snapshots preserve exact source/build/model identity while the separate development checkout remains editable.

The initial validation job automatically creates development/llama.cpp after successful comparison. develop.py prepare freezes staged and unstaged source changes plus experiment helpers into a new named snapshot and compiles it. It does not benchmark until the user runs the candidate job. Geometry parameters are rows 2/4/8 and SIMD groups 1/2/4. Do not confuse these available choices with measured winners.

There is no confirmed A20 capability claim in this package. The inspected Apple table establishes Apple10 for M5/A19 but does not list A20. Query actual hardware and current SDK support before making a device-specific change. No M5/iPhone optimization or performance run occurred during handoff preparation.
