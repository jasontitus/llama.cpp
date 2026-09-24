# Follow-up work

- [ ] Optimize ordinary single-token Bonsai 2 decode on Metal. Inspect the current PTQ1 single-vector kernel and whole-model bottlenecks; compare portable candidates with ABBA, numerical checks, and standard llama-bench tg measurements. The multi-column results do not satisfy this goal. Coordinate with PrismML PR #225 before proposing overlapping changes.
- [x] Measure actual speculative decoding on the M1 Ultra with the multi-column candidate off and on. Start with the Bonsai 2 MTP setup used by the CUDA fork, draft lengths 1, 2, and 3; record acceptance, generated-token throughput, and output comparisons against ordinary decoding. Preserve any required MTP compatibility changes separately from the Metal kernel patch.
- [ ] Investigate the final build's lower n=4 long-input projection performance using paired compiled variants; do not substitute older measurements.
- [ ] Run portable R4/S1 and R2/S1 candidates on actual M5 with the same model, correctness checks, and ABBA protocol. Add Apple10-specific tuning only if those measurements support it.

- [ ] Extend MTP validation to more prompts, longer filled contexts and other sampling modes. Greedy equality on the tested prompts is not universal batch invariance.

- [x] Completed and reviewed the independent four-model ABBA run (20260923-133649): all supported cells complete, token comparisons pass, charts include the heavy-thermal-pressure caveat. Older DSpark server compatibility remains separate work.

- [ ] Investigate the standalone run's thermal/frequency variation, small PQ2_0 batch-control differences, and unexpected PTQ1_0 pp8 gain before claiming broader improvements or zero small regressions.
