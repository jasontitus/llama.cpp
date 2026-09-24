# Continue optimization on the M5 Max

The package is both a reproducible validation study and a development workspace. The complete validation job creates the editable checkout automatically after the family/projection studies and comparison pass. If you only need to inspect source before completing that study, `python3 develop.py init` creates it without benchmarking.

## Locations and boundaries

- `workspace/work/llama-current`: pinned validation checkout on `downstream/metal-ptq1-m5-validation`. Preserve this reference and its build/results.
- `development/llama.cpp`: editable worktree on `downstream/metal-ptq1-m5-tuning`, starting at the same PrismML revision with the existing patch applied.
- `candidates/<name>/`: a separate, detached source snapshot, source patch, Release build, configuration, candidate notes and results. Candidate names cannot be reused, and later edits to the development worktree do not change them.
- `development/EXPERIMENTS.jsonl`: candidate preparation index. Add outcome and rejected-idea notes to the development checkout's experiment log after each run.
- `jobs/`: persistent job status, logs and run receipts. The same launcher reattaches or resumes without rebuilding an already prepared experiment.

The model files are shared by verified paths; source and builds are separate. No automatic commits, push, PR or rebase. The intended PR destination remains PrismML's prism branch. Inspect new upstream changes separately and revalidate any rebase as a new experiment.

## A complete candidate cycle

Read CUDA-TO-APPLE.md, DEVICE-PLAN.md, and the experiment log first. Finish portable R4/S1 and R2/S1 validation before introducing Apple10-specific changes.

Edit source in `development/llama.cpp`. From the package directory, freeze and build a candidate:

```sh
python3 develop.py prepare r4-s2 --rows 4 --simdgroups 2 \
  --note 'Hypothesis: two SIMD groups improve occupancy; preserve arithmetic and compare n=1/n=8 controls.'
zsh Run-All.command --candidate r4-s2
```

The first command compiles a source snapshot; the second performs correctness checks, the four-model ABBA matrix and a projection ABBA study for that geometry, then generates reports. These are development commands for later use on the M5, not work performed while this package was prepared.

New production files must be staged with git add before snapshotting; no commit is required. Staged and unstaged tracked changes are captured together. Files under experiments/ are research material and copied separately. A failed preparation remains available for diagnosis; choose a new name when retrying it. This intentionally favors preserving evidence over overwriting a candidate.

The existing experiment flag is the A/B boundary: A disables the multi-column candidate, B enables it. If you change a shared path or ordinary single-vector code, the flag no longer supplies an unchanged A implementation. Before timing that kind of change, add a separate baseline/candidate binary or explicit dispatch selector and record the new comparison protocol. Do not use same-binary flag results to claim a shared-path optimization.

## Promotion gates

Projection, edge, full-model, output-token and numerical checks must pass. Full-model outputs must be nonempty/finite, token/text identical, NMSE <=1e-8 and maximum absolute logit difference <=0.01. Those are explicit research bounds selected above the measured M1 rounding differences, not proof of universal sampling equivalence. Do not loosen a bound just to admit a candidate.

Both timing suites retain all rejected quartets, apply the 20% within-A/within-B spread gate and stop after three failed attempts. Telemetry errors/missing coverage fail timing rather than count as an idle device. Compare sustained behavior and all control paths before selecting a default. The older M1 projection study retained all samples without this gate, so disclose that protocol difference when comparing its historical data.

Keep family tuning, ordinary single-token work, tensor-accelerator exploration, and iPhone deployment as separate hypotheses and patches. Record compiler flags, actual device capabilities, profile traces, numerical errors, model hashes, candidate geometry, acceptance rates, prefill, decode, full-request throughput, memory and rejection counts for every conclusion. Report failed candidates as well as winners.
