# Upstream PR series (PrismML-Eng/llama.cpp)

PrismML merged #262 (2026-09-25, `df7c49e`): the opt-in PTQ1_0 mat-vec for 2-4 columns
(`GGML_METAL_PTQ1_MULTICOL=1`). This series adds the rest of the research branch
(`downstream/metal-ptq1-m5-tuning`), one self-contained PR at a time, each branched from current upstream
`prism`, in upstream style (plain `getenv` switches, no research-flag registry, no benchmark tooling), with focused
test-backend-ops cases and a short evidence table. At most two open at once, in files that do not overlap.

Every change is behind an opt-in switch, so each PR's evidence is a same-binary A-B-B-A on its own branch
(upstream + that PR, switch off vs on), plus a check that with the switch off it matches upstream. Evidence marked
_research_ is from the research branch with other flags on, and is replaced as each PR is measured.

| # | Branch (on the fork) | What | Files | Evidence | Status |
|---|---|---|---|---|---|
| 1 | `downstream/metal-ios-command-buffers` | 4 command buffers per graph on iPhone-class OSes; abort-callback clamp; free all command buffers | ggml-metal.cpp, ggml-metal-context.m | iPhone 17 Pro Max: 10/50 pp512 runs failed with 1, 0/10 with 4; bitwise-identical output (M5, 1/4/8) | ready (`b899e2d`); waiting for the phone's decode-cost measurement (1 vs 4 at 1 and 8 tokens) |
| 2 | `downstream/qwen35-gdn-rows-plain` | qwen35: in-place delta-net state rows for plain decode (`GGML_GDN_ROWS_PLAIN`) | src/models/qwen35.cpp, delta-net-base.cpp, llama-cparams.h, llama-context.cpp | per-PR, M5: tg128 1.076x, pp512 1.046x, 1 request 1.070x; bitwise | ready (`2787e12`) |
| 3 | `downstream/metal-ptq1-multicol-8` | PTQ1_0 multi-column 5-8 columns: partial tiles on #262's kernel, `GGML_METAL_PTQ1_MULTICOL_MAX` | mul_mv.metal, ggml-metal-device.cpp, test-backend-ops | per-PR, M5: _A-B-B-A running_; 2-4 columns bitwise = #262 | ready (`9f69b28`) pending the A-B-B-A |
| 4 | – | PTQ1_0 fused gate/up + SwiGLU mat-vec (`GGML_METAL_PTQ1_GLU`) and activation staging (`GGML_METAL_PTQ1_STAGE`), possibly two PRs | same | _research_: plain decode and 2-4-token steps (ABBA 1 in EXPERIMENTS.md) | to port |
| 5 | – | PQ2_0 multi-column + fused GLU | same | _research_: PQ2 decode +11%, 2 requests +33% | to port |
| 6 | – | small-row routing for the 48-row projections (`GGML_METAL_SMALLM`, `GGML_METAL_SMALLM_MM` with its width cap) | ggml-metal-ops.cpp, mul_mv.metal | _research_: Bonsai 1 decode +14% with rows mode; A19 needs the cap at 512 columns | to port |
| 7 | – | Q1_0 K32 tensor prefill in an optional Metal library | kernels/mul_mm_q1.metal, CMakeLists, ggml-metal-device.* | _research_: M5 pp512 1.07x over the other Q1 flags, bitwise | to port |
| later | – | batch-invariant mode; PTQ1 tensor mat-vec (M5/A19 opt-in) | | | undecided |

Ported code differs from the research branch only in how switches are read and in test scaffolding; each PR
states the research-branch commit its kernel arithmetic matches.

## PR 1 description (draft)

**metal : split graphs into 4 command buffers on iOS**

### Overview

On iOS, a Metal command buffer that runs for about 5 s of GPU time while another GPU client (such as the display
compositor) is waiting gets discarded, and the graph fails with
`Discarded (victim of GPU error/recovery) (00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)`
(`llama_decode` returns -3). The Metal backend encodes a graph as a small first command buffer plus
`n_cb` = 1 more holding ~90% of the nodes, so a long prefill ubatch on a phone is one multi-second command buffer.

This PR makes the default `n_cb` 4 on iPhone-class OSes (iOS, iPadOS, visionOS, tvOS; not Mac Catalyst) and keeps
1 elsewhere; `GGML_METAL_N_CB=1..8` overrides. Output is unchanged: splitting only moves command-buffer boundaries.

It also fixes two latent issues that a default above 1 exposes:

- with an abort callback set, `graph_compute` commits only command buffers 0 and 1 (later ones only when capturing);
  the main thread's buffer is index `n_cb`, so `n_cb > 1` hung in `synchronize`. `n_cb` is now clamped to 1 while an
  abort callback is set.
- `ggml_metal_free` released `GGML_METAL_MAX_COMMAND_BUFFERS` of the `+ 1` command buffers.

### Evidence

iPhone 17 Pro Max (A19 Pro, iOS 27), Ternary Bonsai 2 27B PTQ1_0, 512-token prefill, every run starting at nominal
temperature (BonsaiBench diagnostics):

| Command buffers | Runs | Failed | Longest command buffer |
|---|---:|---:|---:|
| default (1) | 50 | 10 | 5-8 s (each failure: discarded after 5.0 s) |
| 4 | 10 | 0 | 1.6-2.3 s |
| 1, 256-token ubatches | 10 | 0 | 3.0-4.3 s |

pp512 with 4: 71.7 vs 71.4 tok/s (no measurable cost). Decode and 8-token steps: _pending (being measured)_.

M5 Max (macOS): full-model logits bitwise identical with `GGML_METAL_N_CB` 1, 4 and 8 (700 tokens, 173.8M logits);
test-backend-ops MUL_MAT (1482) and FLASH_ATTN_EXT (4809) pass with 1 and 4; the abort-callback reproducer completes
with 1, 4 and 8 (hung with 4 before the clamp). iOS device build checked.

### Requirements

- [x] I have read and agree with the contributing guidelines
- AI usage disclosure: YES. Claude (Anthropic) assisted with the implementation, testing, measurements and this
  description; the contributor reviewed and owns the change.

## PR 2 description (draft)

**qwen35 : in-place delta-net state rows for plain decode (`GGML_GDN_ROWS_PLAIN=1`)**

### Overview

The fused gated delta net has a rows mode that reads and writes each sequence's state rows in the recurrent cache
in place. Upstream uses it only when `n_rs_seq > 0` (speculative contexts that keep snapshots). Plain decode instead
gathers every layer's state into a contiguous tensor, runs the recurrence, and copies the state back; on the Bonsai 2
models that gather and copy-back is 10-14% of a decode token (M5 Max per-op profile).

With `GGML_GDN_ROWS_PLAIN=1`, plain decode takes the rows path too, with one snapshot slot (K = 1). Output is
bitwise identical.

- Only when no extra cells are relocated (`n_rs == n_seqs`): the relocation in `build_rs_cache_view` runs before the
  GDN read and, after a cell reorder, could overwrite a row another sequence reads (the gathered path reads first).
  Graph reuse compares the `s_copy_extra` size, so such a batch rebuilds and takes the gathered path.
- `GGML_GDN_ROWS_PLAIN_MAX_TOKENS=N` limits it to batches of at most N tokens per sequence. On the A19 the in-place
  recurrence op is ~18% slower at 512-token prefill (per-op profile; end to end 0.988x), while decode gains.
  Contexts with `n_rs_seq > 0` keep rows mode at every width, as today.
- Off by default; read once per context. `delta-net-base.cpp` now accepts rows mode without snapshots (the assert
  that required `n_rs_seq > 0` becomes the `keep` condition).

### Evidence

M5 Max (macOS), Ternary Bonsai 2 27B PTQ1_0, this branch, switch off (A) vs on (B), three A-B-B-A quartets with
8 s cooldowns (tok/s; llama-server rates are aggregate, 128 greedy tokens):

| Case | Off | On | Speedup [quartet range] |
|---|---:|---:|---:|
| tg128 (llama-bench) | 38.95 | 41.96 | 1.076 [1.068-1.091] |
| pp512 (llama-bench) | 722.8 | 755.9 | 1.046 [1.042-1.052] |
| llama-server, 1 request | 37.49 | 40.13 | 1.070 [1.065-1.073] |
| llama-server, 2 requests | 18.64 | 19.25 | 1.033 [1.021-1.040] |

Generated tokens identical in every pair. The last cycle ran ~30% slower in both arms (machine state); the mirrored
quartets cancel it, so the ratios stay tight while the absolute means carry it.

Correctness: every logit bitwise equal, off vs on, for 96 single-token decodes plus 12 batches of 4 on PTQ1_0, Q1_0
and PQ2_0 Bonsai models.

### Requirements

- [x] I have read and agree with the contributing guidelines
- AI usage disclosure: YES. Claude (Anthropic) assisted with the implementation, testing, measurements and this
  description; the contributor reviewed and owns the change.

## PR 3 description (draft)

**metal : PTQ1_0 multi-column mat-vec for 5-8 columns**

### Overview

#262 added `GGML_METAL_PTQ1_MULTICOL=1`, a PTQ1_0 mat-vec that serves 2-4 columns; 5-8 columns still take the
generic `mul_mv_ext` path. Batched decode, MTP verification and multi-request steps regularly produce 5-8 columns
(4 requests with MTP draft 1 are 8).

This PR splits 5-8 columns into two tiles of at most four (5 -> 3+2, 6 -> 3+3, 7 -> 4+3, 8 -> 4+4) on the existing
c2/c3/c4 kernels:

- a partial last tile clamps its column reads to the last valid column and skips the writes past it;
- a bool function constant (`FC_MUL_MV + 5`) removes those checks when every tile is complete, so 2-4 columns
  compile to exactly #262's kernel;
- `GGML_METAL_PTQ1_MULTICOL_MAX` (clamped to 4..8, default 8) caps the widths the kernel takes; 4 restores #262.

### Evidence

Correctness (M5 Max):
- test-backend-ops `-o MUL_MAT`: 1493/1493 with the switch off, on with max 4, on with max 8 (the partial-tile
  pipelines compile and run), and with an out-of-range max (clamped to 4). New cases: 5-7 columns with row tails
  and broadcast, 5 and 7 with a strided B.
- Full model, batches of 1-8 tokens (108 positions, 26.8M logits): switch off, bitwise equal to upstream; max 4,
  bitwise equal to #262; max 8 first differs at the first 5-token batch (a different summation order than
  `mul_mv_ext`), max relative difference 3.7e-5, same top token at every position.

Speed: _A-B-B-A running (max 4 vs max 8: pp4 control, pp5-pp8, tg128 control, MTP with 3 and 4 requests)_.

### Requirements

- [x] I have read and agree with the contributing guidelines
- AI usage disclosure: YES. Claude (Anthropic) assisted with the implementation, testing, measurements and this
  description; the contributor reviewed and owns the change.
