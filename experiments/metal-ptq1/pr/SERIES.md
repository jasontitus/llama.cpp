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
| 1 | `downstream/metal-ios-command-buffers` | 4 command buffers per graph on iPhone-class OSes; abort-callback clamp; free all command buffers | ggml-metal.cpp, ggml-metal-context.m | iPhone 17 Pro Max: 10/50 pp512 runs failed with 1, 0/10 with 4; bitwise-identical output (M5, 1/4/8) | open: PrismML-Eng/llama.cpp#275 (`0ffb0c7`) |
| 2 | `downstream/qwen35-gdn-rows-plain` | qwen35: in-place delta-net state rows for plain decode (`GGML_GDN_ROWS_PLAIN`) | src/models/qwen35.cpp, delta-net-base.cpp, llama-cparams.h, llama-context.cpp | per-PR, M5: tg128 1.076x, pp512 1.046x, 1 request 1.070x; bitwise | open: #276 (`1ad814a`); see #207 (CPU in-place GDN, same lines) |
| 3 | `downstream/metal-ptq1-multicol-8` | PTQ1_0 multi-column 5-8 columns: partial tiles on #262's kernel, `GGML_METAL_PTQ1_MULTICOL_MAX` | mul_mv.metal, ggml-metal-device.cpp, test-backend-ops | per-PR, M5: pp6-pp8 2.1-2.4x, MTP at 3/4 requests 2.13x/1.93x vs #262; 2-4 columns bitwise = #262 | open: #277 (`9f69b28`) |
| 4 | `downstream/metal-ptq1-glu` (on 3) | PTQ1_0 fused gate/up + SwiGLU mat-vec (`GGML_METAL_PTQ1_GLU`), without staging | mul_mv.metal, ggml-metal-device.*, ggml-metal-ops.cpp, test-backend-ops | per-PR, M5: 0.98-1.01x (no gain without staging) | **not proposed** in this form (`db4f5df` kept on the fork) |
| 4' | – | PTQ1_0 activation staging (`GGML_METAL_PTQ1_STAGE`) with the staged fused GLU | same + ggml-metal.cpp (scratch size) | research build, M5: staging +5% at 2-8 columns, +3% MTP; fused GLU on top +1-3%; M1 Ultra: about neutral with its four-row fix, profile keeps it off | optional, low priority |
| 5 | `downstream/metal-pq2-multicol` | PQ2_0 mat-vec for 2 columns (`GGML_METAL_PQ2_MULTICOL`) | mul_mv.metal, ggml-metal-device.*, ggml-metal-ops.cpp, test-backend-ops | per-PR, M5: pp2 1.175x, 2 requests 1.154x, MTP 1 request 1.165x, controls flat; off = upstream bitwise | ready to open (`9ae2fda`); hold until 1-3 are reviewed |
| – | – | PQ2_0 fused GLU (`GGML_METAL_PQ2_GLU`) | | screen, M5: 0.997-1.011x on top of multi-column | dropped |
| – | – | small-row mat-vec for one column (`GGML_METAL_SMALLM`) | | screen, M5: Q1_0 0.992-1.004x, PTQ1_0 0.998-1.002x | dropped |
| 6 | – | keep the 48-row projections on mat-vec at prefill (`GGML_METAL_SMALLM_MM` + width cap) | ggml-metal-ops.cpp | screen, M5 Q1_0: pp64 1.033x, pp128 1.028x, pp512 1.007x; A19 needs the cap at 512 | low priority |
| 7 | – | Q1_0 K32 tensor prefill (`GGML_METAL_Q1_SWIZZLE_LOG`) | kernels/mul_mm_q1.metal, CMakeLists, ggml-metal-device.* | screen, M5: pp128 1.076x, pp512 1.067x, tg128 1.000x; bitwise | to port |
| later | – | batch-invariant mode; PTQ1 tensor mat-vec (M5/A19 opt-in) | | | undecided |

Ported code differs from the research branch only in how switches are read and in test scaffolding; each PR
states the research-branch commit its kernel arithmetic matches.

Upstream's AGENTS.md forbids PRs opened by AI agents (penalty: a project ban), so the contributor opens each PR
from a prefilled GitHub compare link, reviews it, and answers review comments personally. Final PR texts are
unwrapped (GitHub keeps single line breaks) and ASCII-only; the drafts below are the working copies.

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

Speed: M5 Max, Ternary Bonsai 2 27B PTQ1_0, this branch, `GGML_METAL_PTQ1_MULTICOL=1` in both arms, max 4 (#262)
vs max 8 (this PR), three A-B-B-A quartets with 8 s cooldowns (tok/s; MTP rows are llama-server with draft-mtp,
draft 1, aggregate over the requests, 128 greedy tokens each):

| Case | Columns per step | #262 | This PR | Speedup [quartet range] |
|---|---:|---:|---:|---:|
| pp4 (control) | 4 | 69.86 | 70.34 | 1.007 [1.000-1.013] |
| pp5 | 5 | 45.14 | 61.82 | 1.370 [1.368-1.373] |
| pp6 | 6 | 30.76 | 74.78 | 2.433 [2.373-2.552] |
| pp7 | 7 | 34.91 | 73.20 | 2.097 [2.059-2.164] |
| pp8 | 8 | 40.45 | 86.82 | 2.146 [2.138-2.161] |
| tg128 (control) | 1 | 43.67 | 43.39 | 0.994 [0.987-1.003] |
| MTP, 3 requests | 6 | 24.48 | 52.23 | 2.134 [2.123-2.148] |
| MTP, 4 requests | 8 | 28.53 | 55.07 | 1.932 [1.888-2.023] |

Generated tokens identical in every pair; no quartet rejected. 5 columns gain less because the 3+2 split costs
the same as 3+3 (80.9 vs 80.2 ms per step).

### Requirements

- [x] I have read and agree with the contributing guidelines
- AI usage disclosure: YES. Claude (Anthropic) assisted with the implementation, testing, measurements and this
  description; the contributor reviewed and owns the change.

## PR 4 findings: fusion needs staging

Measured on the M5 with `GGML_METAL_PTQ1_MULTICOL=1` (max 8) in both arms, three A-B-B-A quartets each, no
quartet rejected, identical tokens:

| Case | Fused GLU, unstaged (PR 4 branch) | Staging alone (research build) | Fused GLU on top of staging (research build) |
|---|---:|---:|---:|
| tg128 | 1.009 | 0.998 | 1.016 |
| pp2 | 0.998 | 1.055 | 1.028 |
| pp4 | 0.982 | 1.053 | 1.023 |
| pp8 | 0.986 | 1.049 | 1.006 |
| server, 1 request (plain / MTP) | 1.002 / 0.990 | – / 1.032 | – / 1.028 |
| server MTP, 4 requests | 0.983 | 1.036 | 1.012 |

The unstaged fused kernel reads each activation block once for gate and up but loses what the unfused
multi-column kernel gets from four rows per simdgroup; only with staged activations does fusion pay (+1-3%).
Staging plus fusion together: about +8% at 2-4 columns and +5-6% for MTP on the M5. On the M1 Ultra, staging
with the M5's two-row tiles cost 31% on MTP (one exploratory quartet); with the family-7 four-row default
(`7414230`) the full stack was -0.7%, and the M1 profile keeps staging off. Worth one optional PR at most, after 1-3,
and it would need the per-family row choice.

On Bonsai 2 27B, 14 of 64 FFN layers never fuse: the allocator places the GLU output over the FFN input
(the in-place guard refuses it). Copying the input into the gate projection's unused output buffer first
would let those layers fuse too (follow-up idea, unmeasured).

## Screens of the remaining changes (2026-09-25)

Research build (`build-dev`), same binary, one switch off (A) vs on (B), three A-B-B-A quartets per cell with 8 s
cooldowns, no quartet rejected, identical tokens in every pair. Logs: development/m5/screen-*.log.

| Change | Model | Cells (speedup [quartet range]) |
|---|---|---|
| PQ2_0 multi-column vs off | Bonsai 2 PQ2_0 | tg128 0.989 [0.977-1.000], pp2 1.167 [1.139-1.187], pp4 1.001, pp8 1.001, 2 requests 1.154, MTP 1 request 1.159 |
| PQ2_0 fused GLU, on top of multi-column | Bonsai 2 PQ2_0 | tg128 1.011, pp2 1.009, pp4 0.997, pp8 1.002, 2 requests 0.998, MTP 1 request 0.998 |
| small-row mat-vec (`SMALLM`) | Bonsai 1 Q1_0 | tg128 0.992, pp2 1.004, 1 request 1.000 |
| small-row mat-vec (`SMALLM`) | Bonsai 2 PTQ1_0 | tg128 1.002, 1 request 0.998 |
| 48-row projections kept on mat-vec (`SMALLM_MM`) | Bonsai 1 Q1_0 | pp16 1.000, pp64 1.033, pp128 1.028, pp512 1.007 |
| Q1_0 K32 prefill | Bonsai 1 Q1_0 | tg128 1.000, pp128 1.076 [1.075-1.076], pp512 1.067 [1.067-1.068] |

The earlier "Bonsai 1 decode +14%" for small-row routing was measured together with rows mode (PR 2); alone it does
nothing on the M5.
