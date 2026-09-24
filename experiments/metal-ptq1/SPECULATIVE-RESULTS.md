# Bonsai 2 speculative decoding on M1 Ultra

The existing opt-in Metal kernel makes actual single-stream MTP decoding faster on these short-context tests. Across code, prose and shell prompts, the geometric mean paired gain over ordinary decoding is 1.345x. This is an equal-weight summary of these three prompts, not a workload-wide estimate. Ordinary single-token kernel optimization is still a separate TODO.

## How it works

The added MTP prediction head proposes one next token. Bonsai verifies it together with the current token using a two-token batch. Accepted proposals advance the response without another full single-token target pass. Our multi-column Metal kernel makes this verification cheap enough to pay for drafting. The main model still checks the proposal; the prediction head does not get to accept its own answer.

The model is a separate 7.01 GB file made with the [CUDA fork's graft recipe](https://github.com/sudoingX/bonsai2-small-gpu/tree/eb52d9d7363cda2d910146f4e37f4b8c64c30c46/graft). Its full SHA-256 exactly matches that recipe: `83a396ee218c36e5ed88205eccb940a71549d9a72a3d020cc2713f94a78f70f0`. The original PTQ1 model is unchanged. Current PrismML already supplies the needed MTP/Hadamard support; no extra production-code patch was required.

## ABBA: benefit over ordinary decoding

A: ordinary decoding, Metal candidate off. B: MTP, one draft token, Metal candidate on. Same merged model on both arms; unused MTP weights are skipped by ordinary decoding. Same downstream baseline `3b19c377d18157bfec39ec71bad193e9ef000cf2` and kernel as the earlier study.

| Prompt | A generation tok/s | B generation tok/s | Paired speedup | Quartet range | A/B full-request tok/s | B draft acceptance |
|---|---:|---:|---:|---:|---:|---:|
| code | 27.79 | 39.51 | 1.427x | 1.323-1.592x | 25.58 / 35.72 | 86.8% |
| prose | 27.27 | 34.20 | 1.259x | 1.201-1.346x | 25.13 / 31.32 | 62.8% |
| bash | 28.97 | 39.17 | 1.353x | 1.311-1.396x | 26.58 / 35.47 | 81.4% |

Each prompt has three A-B-B-A quartets, six observations per arm. Each observation starts a fresh localhost-only llama-server, warms up, then generates 128 tokens. Prompt order is shuffled between cycles. All samples are retained, and GPU processes from this experiment never overlap. Ranges are observed quartet ranges, not confidence intervals.

The server reports generation throughput excluding prompt processing and the first token from its timed generation denominator. Full-request throughput uses the client-observed latency, including prompt processing, and all predicted tokens. Server startup and model loading are excluded from both. Settings: one slot, MTL0 GPU offload 99, flash attention on, F16 KV, context capacity 4096, batch/microbatch 512, 16 CPU threads, thinking off, greedy sampling, repeat penalty 1, seed 20260923, no prompt-cache reuse. These prompts start at short context; this is not a 4K-filled-context or long-conversation benchmark.

## ABBA: contribution of the Metal kernel

Both arms use one-token MTP. A disables the Metal candidate; B enables it. This separates the kernel improvement from simply turning on speculative decoding.

| Prompt | A generation tok/s | B generation tok/s | Paired speedup | Quartet range | A/B full-request tok/s | B draft acceptance |
|---|---:|---:|---:|---:|---:|---:|
| code | 11.55 | 39.91 | 3.460x | 3.311-3.633x | 11.15 / 36.04 | 86.8% |
| prose | 10.46 | 35.56 | 3.402x | 3.382-3.441x | 10.16 / 32.48 | 62.8% |
| bash | 11.21 | 37.60 | 3.348x | 3.323-3.386x | 10.84 / 34.16 | 81.4% |

Without the optimized verification kernel, MTP can be slower than ordinary decoding on this Metal baseline. The larger kernel-only ratio must not be presented as the overall speedup over ordinary generation.

## Draft-length screening

One exploratory sample per prompt/setting, 128 output tokens. These sequential measurements selected one draft token for the paired confirmation; they are not the final performance evidence.

| Kernel | Draft tokens | Code tok/s | Prose tok/s | Shell tok/s |
|---|---:|---:|---:|---:|
| off | 0 | 29.26 | 29.51 | 29.63 |
| off | 1 | 14.57 | 12.38 | 13.54 |
| off | 2 | 18.42 | 13.47 | 17.38 |
| off | 3 | 19.93 | 12.49 | 17.56 |
| on | 0 | 29.73 | 29.32 | 29.72 |
| on | 1 | 40.70 | 35.69 | 39.69 |
| on | 2 | 43.06 | 29.97 | 38.21 |
| on | 3 | 35.19 | 20.97 | 29.76 |

One draft token was the most useful starting configuration across these prompts. Two improved code somewhat more but lost most of the prose gain. Three was slower overall. No universal draft-length optimum is claimed.

## Output correctness

- All 24 screening outputs matched ordinary greedy decoding for 128 tokens each.
- All completed ABBA outputs match within each study: total-benefit study True; kernel-only study True.
- A separate longer check compared the original, ungrafted model using ordinary decoding against the merged model using optimized one-token MTP. All 1,106 returned token IDs and decoded text matched: 423 code tokens (natural end), 171 prose tokens (natural end), 512 shell tokens (configured limit).
- This is tested greedy-output agreement, not a guarantee of bitwise batch invariance, identical stochastic sampling, or unchanged output on every prompt. The earlier numerical checks found small logit rounding differences. No Metal batch-invariance mode was added.

## Timing disturbance and limits

The first total-benefit ABBA attempt was interrupted after a sudden slowdown: prose fell from about 36 tok/s to 4.67 and 6.49 tok/s; baseline also slowed. A CPU-active background Python process was present, but GPU contention was not established. Thermal diagnostics reported no recorded warnings. No other workload was stopped. Every completed sample from that attempt is retained in `abba-total-d1`, with an interruption note.

After output checks showed recovered timings, the entire paired study was restarted into a fresh directory, followed by the kernel-only study. Tables above use those complete runs, with no per-sample outlier deletion. The repeat also shows drift: ordinary generation fell from about 29 to 23 tokens/s in its last cycle. Every paired quartet still favored optimized MTP, but the ratios should not be treated as precision tuning results. This is still a shared-machine benchmark: no control of other applications, direct GPU power/temperature measurement, or cross-device claim.

## Try the tested configuration

The supplied `run-bonsai-mtp.sh` launcher uses the existing local build and prepared model, enables R4/S1 and one-token MTP, and serves on `http://127.0.0.1:8899`. It defaults to greedy sampling, thinking off, one slot and a 4096-token context capacity. Clients can override sampling settings, which would go beyond the tested output comparison. Stop it with Ctrl-C. The benchmark servers have all been terminated.

Key settings are `GGML_METAL_PTQ1_MULTICOL=1`, `--spec-type draft-mtp`, and `--spec-draft-n-max 1`. The original model file has no grafted MTP head, so use the prepared `Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf`.

## Remaining work

- Optimize ordinary single-token Metal decode separately, using ABBA and correctness checks; coordinate with PrismML PR #225.
- Qualify MTP on more prompts, longer filled contexts, longer outputs and other sampling settings before general deployment.
- Investigate the prior long-input n=4 compiler/performance change separately.
- Keep phase two on actual M5: replay portable candidates and MTP tests before adding Apple10-specific tuning. No M5 tuning was done here.

The archive contains the benchmark runner, reproduction notes, raw responses/tokens, server logs, commands, checksums, model-preparation tools and the interrupted run. It excludes model weights and compiled binaries. The original core Metal patch remains unchanged and no PR has been submitted.
