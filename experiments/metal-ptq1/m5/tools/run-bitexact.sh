#!/bin/zsh
# Bit-identical-to-upstream subset: only in-place delta-net state (NMSE 0 on every model).
set -e
cd "$(dirname "$0")"
while [ ! -f headline-pq2-total/summary.json ]; do sleep 20; done
M=../../workspace/work/models
MTP=../../workspace/work/mtp
BIN=../build-final/bin
SRC=../snap-final
python3 abba-m5.py --output bitexact-ptq1 --bin $BIN --src $SRC --cycles 3 --cooldown 8 \
  --model $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf --mtp-model $MTP/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf \
  --bench tg128 --server s0c1 --env-a '{}' --env-b '{"GGML_GDN_ROWS_PLAIN":"1"}'
python3 abba-m5.py --output bitexact-pq2 --bin $BIN --src $SRC --cycles 3 --cooldown 8 \
  --model $M/Ternary-Bonsai-2-27B-PQ2_0.gguf --mtp-model $MTP/Ternary-Bonsai-2-27B-PQ2_0-mtp.gguf \
  --bench tg128 --server s0c1 --env-a '{}' --env-b '{"GGML_GDN_ROWS_PLAIN":"1"}'
