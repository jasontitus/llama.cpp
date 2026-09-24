#!/bin/zsh
set -e
cd "$(dirname "$0")"
M=../../workspace/work/models
MTP=../../workspace/work/mtp
BIN=../build-abba4/bin
SRC=../snap-abba4
PTQ='{"GGML_METAL_PTQ1_MULTICOL":"1","GGML_METAL_PTQ1_MULTICOL_MAX":"8","GGML_METAL_PTQ1_GLU":"1","GGML_METAL_PTQ1_STAGE":"1","GGML_GDN_ROWS_PLAIN":"1","GGML_METAL_SMALLM_MM":"1"}'
PQ2='{"GGML_METAL_PQ2_MULTICOL":"1","GGML_METAL_PQ2_GLU":"1","GGML_GDN_ROWS_PLAIN":"1","GGML_METAL_SMALLM":"1","GGML_METAL_SMALLM_MM":"1"}'
Q1A='{"GGML_GDN_ROWS_PLAIN":"1","GGML_METAL_SMALLM":"1","GGML_METAL_SMALLM_MM":"1"}'
Q1B='{"GGML_GDN_ROWS_PLAIN":"1","GGML_METAL_SMALLM":"1","GGML_METAL_SMALLM_MM":"1","GGML_METAL_Q1_GLU":"1","GGML_METAL_Q1_0_POPCNT":"1"}'

python3 abba-m5-draft.py --output abba4-ptq1-d1-vs-d2 --bin $BIN --src $SRC --cycles 3 --cooldown 8 \
  --model $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --mtp-model $MTP/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf \
  --bench --server s1c1 --env-a "$PTQ" --env-b "$PTQ" --draft-a 1 --draft-b 2
python3 abba-m5-draft.py --output abba4-ptq1-d1-vs-d3 --bin $BIN --src $SRC --cycles 3 --cooldown 8 \
  --model $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --mtp-model $MTP/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf \
  --bench --server s1c1 --env-a "$PTQ" --env-b "$PTQ" --draft-a 1 --draft-b 3
python3 abba-m5-draft.py --output abba4-pq2-d1-vs-d2 --bin $BIN --src $SRC --cycles 3 --cooldown 8 \
  --model $M/Ternary-Bonsai-2-27B-PQ2_0.gguf --mtp-model $MTP/Ternary-Bonsai-2-27B-PQ2_0-mtp.gguf \
  --bench --server s1c1 --env-a "$PQ2" --env-b "$PQ2" --draft-a 1 --draft-b 2
python3 abba-m5.py --output abba4-q1-glu-popcnt --bin $BIN --src $SRC --cycles 3 --cooldown 8 \
  --model $M/Bonsai-27B-Q1_0.gguf \
  --bench tg128 pp2 pp4 pp8 pp32 --server s0c1 s0c2 s0c4 --env-a "$Q1A" --env-b "$Q1B"
