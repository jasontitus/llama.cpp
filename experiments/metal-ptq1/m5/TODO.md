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

14. **A19 / A20 (iPhone)** (partial A19 app results recorded; full device validation remains open). Portable pieces (tiles, GLU fusion, staging,
    rows mode) need no Apple10-only features; the tensor path needs `has_tensor` and is capability-gated.
    A19 is Apple10; A20 family and limits are unverified.
15. **M1 regression check** (completed for the recorded workload). The 21-cell device suite plus nine profile/revision controls passed; see [M1 results](../m1/README.md). PTQ1 selects STAGE=0 for single-user use; enabled staging uses family7 R4. PQ2 plain is faster than MTP on M1. Concurrency four, long-context behavior, and other family7 hardware remain unmeasured by this final suite.

22. **Revalidate the merged M1 fixes on M5 and phones** (open). Use the exact merged revision and rebuild the phone framework. Cover flags-off performance/allocations, n=1..8 full and partial tiles, odd rows, padded inputs, retained/pre-backend allocations, sequential profile changes and held-context rejection. Exercise tensor scratch on supported hardware; M1 cannot do so. Measure plain/MTP C1 and C2/C4 where supported, larger prefill, actual memory footprint and sustained thermal behavior before any Apple10-specific dispatch. The current phone app does not establish real MTP/server-concurrency performance; A20 capabilities remain unverified.

## From the earlier Bonsai 1 kernel-tuning work (ktune / mcpzim), 2026-09-24

A scan of `~/experiments/ktune` and `~/experiments/mcpzim/tools/bonsai-ab` for Bonsai 1 changes still missing
here. That work measured cold **prefill** on a 14.4k-token prompt against upstream ggml `df03399`; decode was
flat in every pair, and its decode experiments (R4 mat-vec, packed16, 0/1-FMA, delta-net tweaks) found
nothing, which our multi-column, small-row and rows-mode flags now cover.

23. **Q1_0 K32-aligned tensor prefill with grid swizzle** (done on M5; phone open). Ported as
    `GGML_METAL_Q1_MM_K32_ALIGNED` / `GGML_METAL_Q1_SWIZZLE_LOG` (EXPERIMENTS.md, "Q1_0 K32 prefill"): with
    swizzle 1, pp512 1.07x and pp128 1.08x over the Q1 stack on M5 Max, decode unaffected, float logits bitwise
    equal on the full model. Open:
    - **Phone:** measured on iPhone 17 Pro Max (iOS 27), overnight suite: pp512 1.116x (3 quartets),
      pp128 1.03x (1.065x in an evening study); iOS builds the optional library and the app logged
      `kernel_mul_mm_q1_0_f32_k32_swizzle1`.
    - **Ragged batches:** a product is K32 only when N % 128 == 0 (and M % 64 == 0), so a prompt's last
      micro-batch usually stays generic (a 300-token chat prompt gets nothing) and so do continuous-batching
      steps. Next step: run the aligned column prefix on K32 and the tail on the generic kernel (two dispatches),
      or a K32 variant that keeps the N bounds.
24. **Flash-attention query promotion + tensor QK, 16 queries per threadgroup** (open). Historical isolated
    attention -10%, full-model -2 to -4.5% (screened). Only the non-vec FA path at 512 queries: ~0 at short
    contexts (attention is ~1% of a 512-token chunk at depth 0), helps long prompts; applies to Q1_0, PQ2_0
    and PTQ1_0 alike. Ports exist that apply cleanly to our `fa.metal`:
    `~/experiments/ktune/runs/bonsai2-release/variants/old7/candidate.diff`. Risks: F16 KV never
    qualified (only Q4_0 KV), tensor-API headers in the fa library add runtime-compile time at startup
    (+12 s historically; measure on a phone). Test with a depth cell (llama-bench `-p 512 -d 8192`).
25. **"old9" attention on top of 24** (open): smaller Q shared memory on the tensor path, compact score
    planes, V fragments reused across two query groups. Isolated -9% to -17% vs old7; confirmed full-model
    -0.53% vs old7 on Bonsai 1; on Bonsai 2 PQ2 (our FA source, 24 + 25 together) -4.63% prefill at 13.6k
    tokens but not overall-qualified (one decode guard at 1.051). Cumulative diff:
    `variants/old9/candidate.diff`. Alternative arm: `bonsai-deep-05/0001` (-1.56% vs old7).
26. **Conv-native: fused conv + conv-history write** (low priority). Screened -3.9% prefill (3 pairs, one
    favouring control), decode within noise; the SiLU fusion part is already here (PrismML `fuse_silu`), the
    CONCAT/CPY history removal is not. Needs allocator and graph grouping changes.
27. **Half-open range overlap** (`ggml-metal-common.cpp`, `>=` -> `>`) (low): measured 1.2% slower prefill
    historically; a one-line decode probe at most.
28. **Ahead-of-time metallib packaging** (startup only): 40 s -> 28 s launch-to-answer historically when the
    tensor library was compiled at runtime; no tokens/s effect.
