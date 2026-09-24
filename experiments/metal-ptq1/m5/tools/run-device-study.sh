#!/bin/zsh
# Run the paired studies behind the "Results by device" table on this Mac, with the same flags and
# protocol as the M5 Max column. Afterwards: python3 device-table.py <out> prints this device's column.
#
# usage: run-device-study.sh <llama build bin dir> <models dir> <mtp dir> <out dir>
#   models dir: Ternary-Bonsai-2-27B-PTQ1_0.gguf, Ternary-Bonsai-2-27B-PQ2_0.gguf,
#               Ternary-Bonsai-27B-PQ2_0.gguf, Bonsai-27B-Q1_0.gguf (missing models are skipped)
#   mtp dir:    Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf, Ternary-Bonsai-2-27B-PQ2_0-mtp.gguf
# Build this branch first (cmake -B build -DGGML_METAL=ON -DLLAMA_BUILD_SERVER=ON; cmake --build build -j).
# Takes about 1.5 hours; run with nothing else using the GPU, on AC power.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${1:?build bin dir}; M=${2:?models dir}; MTP=${3:?mtp dir}; OUT=${4:?output dir}
SRC=$(cd "$HERE/../../../.." && pwd)
mkdir -p "$OUT"

PTQ='{"GGML_METAL_PTQ1_MULTICOL":"1","GGML_METAL_PTQ1_MULTICOL_MAX":"8","GGML_METAL_PTQ1_GLU":"1","GGML_METAL_PTQ1_STAGE":"1","GGML_GDN_ROWS_PLAIN":"1","GGML_METAL_SMALLM_MM":"1"}'
PQ2='{"GGML_METAL_PQ2_MULTICOL":"1","GGML_METAL_PQ2_GLU":"1","GGML_GDN_ROWS_PLAIN":"1","GGML_METAL_SMALLM":"1","GGML_METAL_SMALLM_MM":"1"}'
Q1='{"GGML_GDN_ROWS_PLAIN":"1","GGML_METAL_SMALLM":"1","GGML_METAL_SMALLM_MM":"1"}'
ROWS='{"GGML_GDN_ROWS_PLAIN":"1"}'
ABBA=(python3 "$HERE/abba-m5.py" --bin "$BIN" --src "$SRC" --cycles 3 --cooldown 8)
DRAFT=(python3 "$HERE/abba-m5-draft.py" --bin "$BIN" --src "$SRC" --cycles 3 --cooldown 8)

have() { [ -f "$1" ] || { echo "skip: $1 not found"; return 1; } }

if have "$M/Ternary-Bonsai-2-27B-PTQ1_0.gguf"; then
  "${ABBA[@]}" --output "$OUT/ptq1-same-mode" --model "$M/Ternary-Bonsai-2-27B-PTQ1_0.gguf" \
    --mtp-model "$MTP/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf" --bench tg128 pp2 pp4 pp8 --server s0c1 s1c1 s0c2 \
    --env-a '{}' --env-b "$PTQ"
  "${ABBA[@]}" --output "$OUT/ptq1-bitexact" --model "$M/Ternary-Bonsai-2-27B-PTQ1_0.gguf" \
    --bench tg128 --server s0c1 --env-a '{}' --env-b "$ROWS"
  [ -f "$MTP/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf" ] && "${DRAFT[@]}" --output "$OUT/ptq1-total" \
    --model "$M/Ternary-Bonsai-2-27B-PTQ1_0.gguf" --mtp-model "$MTP/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf" \
    --bench --server s1c1 --env-a '{}' --env-b "$PTQ" --draft-a 0 --draft-b 1
fi
if have "$M/Ternary-Bonsai-2-27B-PQ2_0.gguf"; then
  "${ABBA[@]}" --output "$OUT/pq2-same-mode" --model "$M/Ternary-Bonsai-2-27B-PQ2_0.gguf" \
    --bench tg128 pp2 --server s0c1 s0c2 --env-a '{}' --env-b "$PQ2"
  [ -f "$MTP/Ternary-Bonsai-2-27B-PQ2_0-mtp.gguf" ] && "${DRAFT[@]}" --output "$OUT/pq2-total" \
    --model "$M/Ternary-Bonsai-2-27B-PQ2_0.gguf" --mtp-model "$MTP/Ternary-Bonsai-2-27B-PQ2_0-mtp.gguf" \
    --bench --server s1c1 --env-a '{}' --env-b "$PQ2" --draft-a 0 --draft-b 1
fi
if have "$M/Ternary-Bonsai-27B-PQ2_0.gguf"; then
  "${ABBA[@]}" --output "$OUT/b1-ternary" --model "$M/Ternary-Bonsai-27B-PQ2_0.gguf" \
    --bench tg128 --server s0c1 s0c2 --env-a '{}' --env-b "$PQ2"
fi
if have "$M/Bonsai-27B-Q1_0.gguf"; then
  "${ABBA[@]}" --output "$OUT/b1-binary" --model "$M/Bonsai-27B-Q1_0.gguf" \
    --bench tg128 --server s0c1 s0c2 --env-a '{}' --env-b "$Q1"
fi
echo "done; now run: python3 $HERE/device-table.py $OUT"
