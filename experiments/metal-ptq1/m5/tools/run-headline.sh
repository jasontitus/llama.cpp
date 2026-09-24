#!/bin/zsh
# Headline: upstream PrismML (every flag off) vs the recommended M5 configuration, paired ABBA.
set -e
cd "$(dirname "$0")"
M=../../workspace/work/models
MTP=../../workspace/work/mtp
BIN=../build-final/bin
SRC=../snap-final
PTQ='{"GGML_METAL_PTQ1_MULTICOL":"1","GGML_METAL_PTQ1_MULTICOL_MAX":"8","GGML_METAL_PTQ1_GLU":"1","GGML_METAL_PTQ1_STAGE":"1","GGML_GDN_ROWS_PLAIN":"1","GGML_METAL_SMALLM_MM":"1"}'
PQ2='{"GGML_METAL_PQ2_MULTICOL":"1","GGML_METAL_PQ2_GLU":"1","GGML_GDN_ROWS_PLAIN":"1","GGML_METAL_SMALLM":"1","GGML_METAL_SMALLM_MM":"1"}'
# 1) same decoding mode on both arms: plain vs plain, MTP vs MTP
python3 abba-m5.py --output headline-ptq1-same-mode --bin $BIN --src $SRC --cycles 3 --cooldown 8 \
  --model $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --mtp-model $MTP/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf \
  --bench tg128 --server s0c1 s1c1 --env-a '{}' --env-b "$PTQ"
# 2) what a user gets: upstream plain decoding vs M5 + MTP (same merged model on both arms)
python3 abba-m5-draft.py --output headline-ptq1-total --bin $BIN --src $SRC --cycles 3 --cooldown 8 \
  --model $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --mtp-model $MTP/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf \
  --bench --server s1c1 --env-a '{}' --env-b "$PTQ" --draft-a 0 --draft-b 1
python3 abba-m5-draft.py --output headline-pq2-total --bin $BIN --src $SRC --cycles 3 --cooldown 8 \
  --model $M/Ternary-Bonsai-2-27B-PQ2_0.gguf --mtp-model $MTP/Ternary-Bonsai-2-27B-PQ2_0-mtp.gguf \
  --bench --server s1c1 --env-a '{}' --env-b "$PQ2" --draft-a 0 --draft-b 1
