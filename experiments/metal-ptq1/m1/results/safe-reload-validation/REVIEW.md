# Independent review record

Reviewed final source: 8a1bdf7.

- Kernel review: family7 R4 is used only for enabled staging; STAGE=0 disables ordinary and fused-GLU staging. Full-column specialization retains partial tiles and separate cache keys. No optimization becomes enabled by default.
- Reload review: the first snapshot/refcount proposal was rejected because allocation can precede or outlive a context. The final shape-only reservation closes that lifetime gap. Maximum scratch size is sufficient because paths are mutually exclusive and include their own padding. No remaining concrete correctness blocker was found.
- Runtime evidence: retained allocations and guards passed; conflicting held-context profiles were rejected; GDN remained frozen across graph rebuilds; repeated baseline logits were bitwise identical. Invariant mode matched all 64 positions at widths 2, 3, and 4.
- Memory evidence: reported server compute-buffer sizes are unchanged for tested before/after plain and MTP configurations. Single startup RSS observations cannot establish peak-memory equivalence.
- Measurement review: the 21-cell device suite matches the current M5 script. Seven M1 supplements plus two flags-off controls bring the final run to 30 cells. The latter measures all revision differences, not scratch alone.
- Historical evidence: all 77 checksum entries across six archives verified; ratios and native rates recomputed; partial runs remain partial.

Limits: M5 tensor execution requires M5 validation. Flags-off keeps kernel dispatch/math but can change allocation and CPU overhead. Q1 extra allocation covers n=2..16. Other family7 hardware is unmeasured. Portable profile choices still need per-device testing; GPU family alone does not establish optimal settings. The current iOS harness has no actual MTP or server-concurrency test.

Reviews were read-only during final timing. No M5 remote execution occurred.
