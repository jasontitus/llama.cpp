#!/bin/zsh
# ABBA 3: per-model baseline (A, all flags off) vs the M5 changes applicable to that model (B).
# Runs sequentially from the frozen snapshot build; one GPU process at a time.
set -e
cd "$(dirname "$0")"
M=../../workspace/work/models
MTP=../../workspace/work/mtp
BIN=../build-abba3/bin
SRC=../snap-abba3

python3 abba-m5.py --output abba3-b2-pq2 --bin $BIN --src $SRC --cycles 3 --cooldown 8 \
  --model $M/Ternary-Bonsai-2-27B-PQ2_0.gguf --mtp-model $MTP/Ternary-Bonsai-2-27B-PQ2_0-mtp.gguf \
  --bench tg128 pp2 pp3 pp32 pp512 --server s0c1 s1c1 s0c2 s1c2 \
  --env-a '{}' \
  --env-b '{"GGML_METAL_PQ2_MULTICOL":"1","GGML_METAL_PQ2_GLU":"1","GGML_GDN_ROWS_PLAIN":"1","GGML_METAL_SMALLM":"1","GGML_METAL_SMALLM_MM":"1"}'

python3 abba-m5.py --output abba3-b1-ternary --bin $BIN --src $SRC --cycles 3 --cooldown 8 \
  --model $M/Ternary-Bonsai-27B-PQ2_0.gguf \
  --bench tg128 pp2 pp32 pp512 --server s0c1 s0c2 \
  --env-a '{}' \
  --env-b '{"GGML_METAL_PQ2_MULTICOL":"1","GGML_METAL_PQ2_GLU":"1","GGML_GDN_ROWS_PLAIN":"1","GGML_METAL_SMALLM":"1","GGML_METAL_SMALLM_MM":"1"}'

python3 abba-m5.py --output abba3-b1-binary --bin $BIN --src $SRC --cycles 3 --cooldown 8 \
  --model $M/Bonsai-27B-Q1_0.gguf \
  --bench tg128 pp2 pp32 pp512 --server s0c1 s0c2 \
  --env-a '{}' \
  --env-b '{"GGML_GDN_ROWS_PLAIN":"1","GGML_METAL_SMALLM":"1","GGML_METAL_SMALLM_MM":"1"}'

STACK='"GGML_METAL_PTQ1_MULTICOL":"1","GGML_METAL_PTQ1_MULTICOL_MAX":"8","GGML_METAL_PTQ1_GLU":"1","GGML_METAL_PTQ1_STAGE":"1","GGML_GDN_ROWS_PLAIN":"1"'
python3 abba-m5.py --output abba3-b2-ptq1-smallm --bin $BIN --src $SRC --cycles 3 --cooldown 8 \
  --model $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --mtp-model $MTP/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf \
  --bench pp16 pp32 pp128 pp512 --server s0c1 s1c1 \
  --env-a "{$STACK}" --env-b "{$STACK,\"GGML_METAL_SMALLM_MM\":\"1\"}"
