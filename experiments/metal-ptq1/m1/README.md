# M1 Ultra regression investigation and device benchmarks

The M5-selected staged PTQ1 geometry regressed M1 MTP performance. The fix restores four rows on Apple7, aligns staged scratch, and specializes complete column tiles. For the measured single-user M1 profile, staging is disabled; M5 keeps its existing geometry and needs a fresh device run before these changes can be called performance-neutral there. All research kernels remain opt-in.

See [EXPERIMENTS.md](EXPERIMENTS.md) for the diagnosis, rejected ideas, controlled comparisons, and review findings. The initial replay was deliberately interrupted when the regression appeared; partial results remain labeled partial. Final merged-source results are recorded separately from earlier diagnostic snapshots.

## Completed results

All 30 selected cells passed: 90 ABBA quartets, matching server tokens across and within arms, no stability retries, and 1,983 numerical/matrix/fusion checks. All 2,745 thermal samples were nominal. Final measured production source is `8a1bdf7`; its binary build label remains `4676829` because the safety patch was built before it was committed. The compiled patch and binary hashes are retained in validation provenance.

Against the literal previous PR, native plain generation is 29.28 -> 31.87 tok/s and MTP is 38.65 -> 39.16. Paired full-request gains are +8.29% plain and +1.32% MTP. Standard llama-bench pp2/pp3 improve 10.92%/7.36%. The flags-off revision control is effectively flat (+0.18% plain, +0.59% MTP).

For a single user, the M1 PTQ1 profile measured 31.91 tok/s plain and 38.63 with MTP in the same-mode study. For PQ2, choose plain: 35.81 tok/s versus 28.73 with MTP. These plain/MTP mean comparisons use separate randomized cells. See [all results](results/final-device/RESULTS.md), [experiment log](EXPERIMENTS.md), and [shared device chart](../m5/README.md#results-by-device).

## Method

- Apple M1 Ultra, 64 GPU cores, 128 GB. OS, compiler, source revision, binary hashes, and verified model hashes are in the run manifest.
- Use the M5 ABBA measurement procedure. `abba-m5.py` adds optional separate arm-A build/source paths for revision comparisons; these require fresh output directories and record both builds. Both harnesses sanitize inherited research flags. Three A-B-B-A quartets per cell, eight-second cooldown before each observation, reproducibly shuffled cell order, fresh process per observation, and the same 20% within-arm spread limit. Rejected quartets are retained; three failed attempts leave the cell incomplete.
- The default `device` suite matches the current M5 `run-device-study.sh`: 21 cells across seven studies and all four models. The final run adds seven M1 profile/prior-PR cells and two flags-off regression controls, for 30 cells. The first interrupted archive planned 43 cells and completed five. The later 55-cell plan remains available with `--suite replay`; it was not completed. The device suite includes concurrency one and two, not four; earlier concurrency-four studies are not represented as fresh measurements.

- One GPU workload at a time. Full Metal offload, Flash Attention, 16 CPU threads, server batch/ubatch 512, 4096 context per slot, greedy generation of up to 128 tokens, and a 32-token warmup per slot. MTP uses one draft token and the pinned grafted head.
- Sanitize all inherited `GGML_*` and `LLAMA_ARG_*` settings before launching the M5 harness. Explicit A/B flags and exact commands are recorded. The same-mode studies compare flags off/on within a decoding mode. The total-benefit studies compare original plain against optimized MTP using the same merged model in both arms.
- Validate all three quantization formats before timing: strict double-reference numerical checks and test-backend-ops matrix/fusion checks with nonempty pass counts. Server token IDs are compared across arms and within repeated arms. These comparisons do not imply bitwise-equal logits or a broad quality evaluation.
- The report distinguishes `timing_complete` (three quartets passing the spread gate), `output_match` (server token IDs matching across and within arms), and `accepted` (complete timings plus matching server output). Bench cells do not test generated output. Mismatching server cells retain their measured rates and both mismatch counts with an explicit status in the detailed table, and are excluded from the single-user headline table.

## Measurement limits

Codex remained open during this run. Device GPU utilization, relevant process CPU activity, and macOS thermal state were sampled every two seconds. Administrator authentication was unavailable, so there is no per-process GPU attribution. Passing the spread gate does not establish absence of background GPU interference. These are monitored, in-app measurements, not a claimed quiet-machine replacement for the earlier standalone run.

The GPU counter scope is only the first `AGXAccelerator` `PerformanceStatistics` entry returned by `ioreg`; it is not an aggregate across devices or chips. Older telemetry, including the initial replay and any already-running diagnostic process, has this same scope but lacks an explicit scope field. Updated runs label the scope and record how many statistics entries were present. Editing the monitor does not change telemetry already collected or code already loaded by a running process.

The M5 committed studies used several frozen snapshots; the initial M1 replay used `7ff4358`; diagnostic runs identify their exact source and patch. The final run uses `8a1bdf7` with no tracked patch. Compare paired gains within each device. Absolute M1/M5 throughput also reflects hardware, OS/toolchain, and run conditions. The M5 tensor path is capability-gated and unavailable on M1; it is not silently included in the flags under test. The original broad replay was interrupted to prioritize the user-requested regression fix; incomplete cells remain incomplete and are not headline results.

The bitexact-labeled studies match the M5 study names. They denote the GDN-only configuration, not an independent proof of bitwise whole-model equality on M1. Similarly, invariant-cost timings alone do not prove batch invariance.

## Reproduce

Use a workspace with `work/llama-tuning` containing this checkout, `work/models` containing the four files in `models.json`, and `work/mtp` containing the two grafted models. Use the pinned graft recipe linked under MTP in [the M5 README](../m5/README.md#recommended-flags-per-model) to construct the merged models; verify their hashes against `models.json`. Run from the workspace root, outside other GPU-heavy work. The output directory must not exist.

```sh
cmake -S work/llama-tuning -B work/build-tuning -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DLLAMA_BUILD_TESTS=ON -DGGML_CCACHE=OFF
cmake --build work/build-tuning --target llama-server llama-bench test-backend-ops -j 8
for kind in m5 pq2 q1; do
  c++ -O2 -std=c++17 "work/llama-tuning/experiments/metal-ptq1/m5/tools/check-numerical-$kind.cpp" \
    -I work/llama-tuning/ggml/include -L work/build-tuning/bin -lggml -lggml-base \
    -Wl,-rpath,"$PWD/work/build-tuning/bin" -o "work/build-tuning/bin/check-numerical-$kind"
done
clang -framework Foundation work/llama-tuning/experiments/metal-ptq1/m1/tools/thermal-state.m \
  -o work/build-tuning/bin/thermal-state
caffeinate -dimsu python3 work/llama-tuning/experiments/metal-ptq1/m1/tools/run.py \
  --workspace . --output outputs/m1-device
python3 work/llama-tuning/experiments/metal-ptq1/m1/tools/summarize.py outputs/m1-device
```

The report's acceptance check can be verified without GPU work using `python3 work/llama-tuning/experiments/metal-ptq1/m1/tools/summarize.py --self-test`. This synthetic fixture first accepts matching ABBA outputs, then introduces a cross-arm and within-arm mismatch and verifies that the cell is rejected and omitted from the headline table.

Read every study's status and rejected quartets before using the results. A failed study does not prevent independent later studies from running. The raw logs retain actual Metal device initialization. Raw observation JSON includes commands, timing samples, and server token IDs; the M5 harness retains only the last server log for each repeated arm, so those logs alone are not the observation record.

## Adversarial review

Independent reviews covered the Metal paths, measurement method, and final fixes. Findings addressed: staged scratch alignment, an invalid global-R2 control, separate-build resume mislabeling, omitted staged diffs, missing tool/checker provenance, and acceptance logic that previously could present mismatching output as complete. Reviews also rejected an unsafe first flag-reload fix because graph allocations can precede or outlive a backend context. The final fix freezes flags across live contexts, rejects conflicting profiles, snapshots GDN per context, and reserves sufficient scratch independent of research flags. The report distinguishes timing completion from matching output, shows cross-arm and within-arm mismatches, and excludes incorrect cells from the headline table. `summarize.py --self-test` exercises matching and mismatching records without GPU work.

Final runs snapshot research source files, the tracked patch, model pins, and binary/checker hashes. Historical manifests are not retroactively rewritten to claim newer instrumentation. No automatic cross-device defaults beyond the measured family-7 row correction were introduced; M5 performance must not be inferred from M1 results.

## M1 single-user profile

```sh
export GGML_METAL_PTQ1_MULTICOL=1
export GGML_METAL_PTQ1_MULTICOL_MAX=8
export GGML_METAL_PTQ1_GLU=1
export GGML_METAL_PTQ1_STAGE=0
export GGML_GDN_ROWS_PLAIN=1
export GGML_METAL_SMALLM_MM=1
```

This profile is scoped to Bonsai 2 PTQ1, the tested short prompts, and single-user operation. Do not infer optimal flags for larger contexts, concurrency, or other devices. The device-comparison table deliberately uses the M5 flags, including STAGE=1, to compare the same configuration; the M1 profile has separate results.

For the final 30-cell selection, replace the run command above with:

```sh
caffeinate -dimsu python3 work/llama-tuning/experiments/metal-ptq1/m1/tools/run.py \
  --workspace . --output outputs/m1-final \
  --studies ptq1-same-mode ptq1-bitexact ptq1-total pq2-same-mode pq2-total \
  b1-ternary b1-binary m1-single-user m1-total previous-pr-m1 reload-baseline
```

The revision comparisons additionally require clean, separately built checkouts at `work/llama-pr` (`d4ca15d`, build `work/build-pr`) and `work/llama-tuning-r4` (`7414230`, build `work/build-tuning-r4`). Do not copy binaries whose runtime library paths point at the candidate build. Their source and library hashes are recorded in the manifest. The flags-off control measures all changes between the frozen revision and candidate, not scratch allocation alone.

## Further device work

Keep the portable candidates parameterized and opt-in. Retest the final kernels, scratch layout, and in-process reload on M5, including n=1..8, partial tiles, staged geometry, tensor allocation, and single-user/concurrent server cells. Only add Apple10-specific defaults after paired measurements support them. A19 requires its own correctness, memory, thermal, and throughput run; A20 capabilities and tuning remain unverified. M1 cannot execute or validate the tensor path.
