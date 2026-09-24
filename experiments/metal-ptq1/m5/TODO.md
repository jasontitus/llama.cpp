# Open items after the first M5 round (2026-09-23)

Status legend: **open** = not started; **partial** = started, evidence incomplete; **blocked** = needs a
decision or another device. Each item lists the evidence that would close it.

## Measurement and validation

1. **Tensor path end-to-end gap** (partial). Standalone bench: 2.2x at n=8 (138 vs 305 µs). In the model the
   first quartets show about +4% on pp8 and about +13% on MTP C4. Suspects: it displaces the fused FFN at
   n=5..8 (two tensor products + SWIGLU instead of one fused kernel), and each projection pays an extra
   hi/lo pass plus two barriers. Close with: per-op profile of an n=8 graph with and without the flag; a
   fused gate/up tensor variant; folding the hi/lo split into the staging pass shared by projections
   that read the same activation (qkv/gate, gate/up).
2. **MTP draft length on M5** (closed: 1 draft stays best; 2 = 0.90x, 3 = 0.75x). n=3 now costs ~1.2x n=2, so 2-3 drafts may pay off. Harness is ready
   (`abba-m5-draft.py --draft-a/--draft-b`). Close with ABBA d1 vs d2 vs d3 at C1, all prompts, token check.
3. **Headline total-benefit study** (open): stock PrismML plain decode vs M5 stack + MTP, same merged model
   (`--draft-a 0`), for the single-request case users feel.
4. **Longer contexts and filled KV** (open). Every study so far starts from ~30-token prompts. The GDN rows
   mode, FA and staging costs change with context. Close with ABBA at 4K/16K filled contexts.
5. **Sustained / thermal behaviour** (open). This laptop drifts ~15% under back-to-back load. Record
   powermetrics-free sustained runs (10+ minutes) for plain and MTP, per arm.
6. **Upstream fusion test coverage** (partial). Research cases were added locally to test-backend-ops; a PR
   would need maintainer agreement (contributor guidance discourages new standalone tests).
7. **Strict checker for fused GLU** (open). The 1e-8 standalone checker covers MUL_MAT only; fused GLU is
   covered by test-backend-ops (1e-7) and the full-model logit gate. Add a strict fused-GLU check.

## Correctness follow-ups

8. **PrismML MTP rows-mode multi-sequence hazard** (open, upstream). `build_rs_cache_view` relocates extra
   cells before the rows-mode GDN read; with a cell reorder a sequence can read another's state. Exists on
   the pinned PrismML MTP path (`n_rs_seq > 0`), independent of this work. Needs a reproducer (server
   `-np 4`, idle slot + three active slots) and an upstream report; our plain-decode flag avoids it by
   requiring `n_rs == n_seqs`.
9. **Fusion vs allocator aliasing on other backends** (open). The fused-GLU race (output placed over the
   FFN input) is generic; check whether PrismML's CUDA gate fusion relies on a guard for it.

## Performance ideas not yet tried

10. **n=1 decode** (open). The kernel is near memory-bound (49.7 vs 44.6 µs floor on ffn_up). Remaining
    single-stream time is small ops: 48 FWHT rotations, norms, adds, conv, GDN, FA. Candidates: fold the
    Hadamard rotation into the staging pass; fuse ssm_alpha/ssm_beta (two 5120x48 BF16 matvecs on the same
    input, CUDA PR #218 re-routed these); fuse the GDN state scatter into the op epilogue (the counted
    SET_ROWS went 32 -> 80 in rows mode).
11. **Staging shared across sibling projections** (open). attn_qkv and attn_gate (and q/k/v in attention
    layers) read the same normalized activation; stage it once.
12. **Tensor path for n=3..4** (open). Currently a tie at n=4; a cheaper decode (fewer ops per weight) or
    two-row-slab occupancy tuning could make it win.
13. **Batch invariance mode** (done for PTQ1_0 + float weights, whole model, bitwise, verified to a 2011-token
    context; costs ~11% on multi-token work). Extending to PQ2_0/Q1_0 would need the same one-template routing.
    Original note: CUDA offers a mode where results are bitwise independent of column
    count. Metal candidates share the single-vector reduction order but are not bitwise invariant across
    paths (scalar n<=4 vs tensor n>=5). Needed only if bitwise sampling reproducibility is required.

## Added in round 2

16. **Prefill beyond the matmuls** (open). Chunked delta-net is 6% of a PTQ1 pp512 ubatch; SWIGLU + CONT
    ~4%. A gate/up fusion inside the tensor mul_mm (two A tiles, SWIGLU epilogue) would also read x once.
17. **Activation staging for PQ2_0** (open). Not ported; PQ2 multi-column is only used at n=2 where the
    per-column staging share is small.
18. **Q1_0 fused FFN** (open). Bonsai 1 binary's FFN is 38% of its decode token; a fused gate/up/SWIGLU
    for the existing Q1_0 multi-column kernel mirrors the PTQ1/PQ2 ones.
19. **test-backend-ops tolerance** (note). Its MUL_MAT NMSE bound (5e-4) passed a kernel that read
    activations from the wrong place for 2 of 258 blocks; keep the strict checker as the gate.

20. **Split the research diff into reviewable patches** (open). The worktree holds winners, opt-in
    trade-offs and rejected experiments (b128 prefill kernel, Q1_0 fused FFN, profiler) in one diff. A PR
    needs a clean patch per hypothesis.
21. **Invariance without losing staging** (open). Staging all column counts (including n=1) might keep
    invariance at lower cost than turning staging off; measure the n=1 pre-pass cost.

## Other devices

14. **A19 / A20 (iPhone)** (blocked on a signed iOS harness). Portable pieces (tiles, GLU fusion, staging,
    rows mode) need no Apple10-only features; the tensor path needs `has_tensor` and is capability-gated.
    A19 is Apple10; A20 family and limits are unverified.
15. **M1 regression check** (blocked on the M1). Replay the portable flags on the M1 Ultra before any
    cross-device default.
