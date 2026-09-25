# Upstream PR series (PrismML-Eng/llama.cpp)

PrismML merged #262 (2026-09-25, `df7c49e`): the opt-in PTQ1_0 mat-vec for 2-4 columns
(`GGML_METAL_PTQ1_MULTICOL=1`). This series adds the rest of the research branch
(`downstream/metal-ptq1-m5-tuning`), one self-contained PR at a time, each branched from current upstream
`prism`, in upstream style (plain `getenv` switches, no research-flag registry, no benchmark tooling), with focused
test-backend-ops cases and a short evidence table. At most two open at once, in files that do not overlap.

| # | Branch (on the fork) | What | Files | Evidence | Status |
|---|---|---|---|---|---|
| 1 | `downstream/metal-ios-command-buffers` | 4 command buffers per graph on iPhone-class OSes; abort-callback clamp; free all command buffers | ggml-metal.cpp, ggml-metal-context.m | iPhone 17 Pro Max: 10/50 pp512 runs failed with 1, 0/10 with 4; bitwise-identical output (M5, 1/4/8) | ready (`b899e2d`); waiting for the phone's decode-cost measurement (1 vs 4 at 1 and 8 tokens) |
| 2 | – | qwen35: in-place delta-net state rows for plain decode (`GGML_GDN_ROWS_PLAIN`) | src/models/qwen35.cpp, llama-cparams.h, llama-context.cpp | bit-identical; plain decode +7% (M5), +8% (M1) | to port |
| 3 | – | PTQ1_0 multi-column 5-8 columns: partial tiles on #262's kernel, `GGML_METAL_PTQ1_MULTICOL_MAX` | mul_mv.metal, ggml-metal-device.cpp/.h, ggml-metal-ops.cpp | M5: pp8 2.29x, MTP at 4 requests 1.95x | to port |
| 4 | – | PTQ1_0 fused gate/up + SwiGLU mat-vec (`GGML_METAL_PTQ1_GLU`) and activation staging (`GGML_METAL_PTQ1_STAGE`), possibly two PRs | same | M5: plain decode and 2-4-token steps (ABBA 1 in EXPERIMENTS.md) | to port |
| 5 | – | PQ2_0 multi-column + fused GLU | same | M5: PQ2 decode +11%, 2 requests +33% | to port |
| 6 | – | small-row routing for the 48-row projections (`GGML_METAL_SMALLM`, `GGML_METAL_SMALLM_MM` with its width cap) | ggml-metal-ops.cpp, mul_mv.metal | Bonsai 1 decode +14% with rows mode; A19 needs the cap at 512 columns | to port |
| 7 | – | Q1_0 K32 tensor prefill in an optional Metal library | kernels/mul_mm_q1.metal, CMakeLists, ggml-metal-device.* | M5: pp512 1.07x, bitwise | to port |
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
