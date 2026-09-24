# Bonsai 2 MTP follow-up

## Purpose

Measure whether the PTQ1 n=2..4 Metal kernel improves real single-stream speculative decoding. Ordinary single-token kernel optimization remains a separate TODO.

MTP uses a small prediction head to propose tokens. With one draft token, the target checks a two-token batch. Accepted drafts advance generation without a separate full target pass per token. A speedup requires the accepted work to exceed drafting and verification overhead.

## Model and code

- Target: PrismML Bonsai 2 27B PTQ1_0; its original weights are unchanged.
- Graft recipe: https://github.com/sudoingX/bonsai2-small-gpu/tree/eb52d9d7363cda2d910146f4e37f4b8c64c30c46/graft
- Donor: unsloth/Qwen3.8-27B-GGUF, Qwen3.8-27B-UD-Q4_K_M.gguf.
- Downloaded the header and required head/embedding tensor ranges from a pinned donor revision, rather than the full 16.5 GB file. The partial donor must not be used for inference.
- Extracted head SHA-256: 89a3144aff7a71cf4c12e089bb6174800851fa4d8c3ee32c3e67192f0e294956.
- Merged 7,012,820,512-byte model SHA-256: 83a396ee218c36e5ed88205eccb940a71549d9a72a3d020cc2713f94a78f70f0.
- Both full-file hashes match the CUDA fork recipe exactly.
- Same downstream PrismML baseline 3b19c377d18157bfec39ec71bad193e9ef000cf2 and reviewed Metal kernel as the projection study. Current PrismML already contains the MTP Hadamard inverse fix. No new production-code changes were needed for this experiment.

## Protocol

Use standard llama-server and its native completion endpoint. Apply the model's chat template with thinking disabled. Sample greedily with temperature 0, repeat penalty 1, fixed seed, no prompt cache reuse, one sequence, full Metal offload, flash attention on, F16 KV, context capacity 4096, batch/microbatch 512 and 16 CPU threads. Prompts cover Python code, explanatory prose, and a shell script. Screening generates 128 tokens per prompt.

Screen both kernel settings at draft lengths 0, 1, 2 and 3. One draft token had the best geometric mean generation throughput across the three screening prompts. Two helped code somewhat more but lost most of the prose gain. Three was slower overall. These sequential screening runs choose a candidate; they are not the final performance evidence.

For confirmation, run two separate paired studies:

1. Total benefit: ordinary decoding with kernel off (A), MTP with one draft token and kernel on (B).
2. Kernel contribution: MTP with one draft token, kernel off (A) versus on (B).

Each prompt gets three A-B-B-A cycles, six observations per arm. Every observation uses a fresh server, warmup of up to 32 tokens, then the measured request. Prompt order is shuffled between cycles. GPU processes never overlap. Keep all observations. Report each prompt separately and geometric mean paired ratios; do not treat generated tokens as independent samples.

Count actual returned token IDs, not streaming events. Record server generation throughput, client request wall time including prefill, accepted/generated drafts, full output tokens/text, commands and binary/library hashes. Compare all greedy token sequences. Generation throughput uses llama-server's reported timing, which excludes the first token from its timed generation denominator. Client throughput includes all returned predicted tokens and complete request latency.

Exact agreement on tested greedy outputs is not a guarantee of bitwise batch invariance or identical stochastic sampling. Longer contexts, concurrent requests, other prompts and other sampling settings require separate validation. No Metal equivalent of GGML_CUDA_BATCH_INVARIANT was added or claimed. The M5 plan remains unchanged.

## Reproduction

Build the standard server target in the same Release Metal build. Use speculative.py with --mode screen, then --mode abba --pair total or --pair kernel, --draft 1, --cycles 3. Supply --binary, --model and a new --output directory. Raw result directories record exact commands and environment selection. The parity mode compares ordinary decoding and the optimized MTP path over longer outputs.
