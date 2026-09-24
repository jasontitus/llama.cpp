#!/bin/zsh
set -e
cd "$(dirname "$0")"
M=../../workspace/work/models
MTP=../../workspace/work/mtp
STACK='"GGML_METAL_PTQ1_MULTICOL":"1","GGML_METAL_PTQ1_MULTICOL_MAX":"8","GGML_METAL_PTQ1_GLU":"1","GGML_METAL_PTQ1_STAGE":"1","GGML_GDN_ROWS_PLAIN":"1","GGML_METAL_SMALLM_MM":"1"'
python3 abba-m5.py --output abba5-invariant-cost --bin ../build-abba5/bin --src ../snap-abba5 --cycles 3 --cooldown 8 \
  --model $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --mtp-model $MTP/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf \
  --bench tg128 pp2 pp4 pp8 --server s0c1 s1c1 s1c2 \
  --env-a "{$STACK}" --env-b "{$STACK,\"GGML_METAL_BATCH_INVARIANT\":\"1\"}"
