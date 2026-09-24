#!/opt/homebrew/bin/bash
# Quality check of each Bonsai model: upstream (all flags off) vs the maximum-optimization configuration.
#  - KL divergence + perplexity on WikiText-2 (20 x 512-token chunks) against the baseline's own logits,
#    at batch 1 (decode kernels) and batch 4 or 2 (multi-token kernels)
#  - HellaSwag (400 tasks) and Winogrande (1267) accuracy at default batching
# Environments are bash arrays so every flag is applied. Waits for the running timing studies.
set -u
cd "$(dirname "$0")"
while [ ! -f bitexact-pq2/summary.json ]; do sleep 30; done
cmake --build ../build-final --target llama-perplexity -j 16 > quality-build.log 2>&1 || { echo "build failed"; exit 1; }
PPL=../build-final/bin/llama-perplexity
M=../../workspace/work/models
D=data
OUT=quality
mkdir -p $OUT

PTQ=(GGML_METAL_PTQ1_MULTICOL=1 GGML_METAL_PTQ1_MULTICOL_MAX=8 GGML_METAL_PTQ1_GLU=1 GGML_METAL_PTQ1_STAGE=1 GGML_GDN_ROWS_PLAIN=1 GGML_METAL_SMALLM_MM=1 GGML_METAL_PTQ1_TENSOR=1)
PQ2=(GGML_METAL_PQ2_MULTICOL=1 GGML_METAL_PQ2_GLU=1 GGML_GDN_ROWS_PLAIN=1 GGML_METAL_SMALLM=1 GGML_METAL_SMALLM_MM=1)
Q1=(GGML_GDN_ROWS_PLAIN=1 GGML_METAL_SMALLM=1 GGML_METAL_SMALLM_MM=1)
Q1PC=(GGML_GDN_ROWS_PLAIN=1 GGML_METAL_SMALLM=1 GGML_METAL_SMALLM_MM=1 GGML_METAL_Q1_0_POPCNT=1)
COMMON=(-ngl 99 -fa on -t 16)

kld() {  # model tag ub arm-name env...
  local model=$1 tag=$2 ub=$3 arm=$4; shift 4
  local base=$OUT/kld-base-$tag-ub$ub.bin
  if [ ! -f $OUT/$tag-kld-ub$ub-base.log ]; then
    env X=0 $PPL -m $model -f $D/wikitext-2-raw/wiki.test.raw -c 512 --chunks 20 -b $ub -ub $ub "${COMMON[@]}" \
      --kl-divergence-base $base > $OUT/$tag-kld-ub$ub-base.log 2>&1
  fi
  env "$@" $PPL -m $model -f $D/wikitext-2-raw/wiki.test.raw -c 512 --chunks 20 -b $ub -ub $ub "${COMMON[@]}" \
    --kl-divergence-base $base --kl-divergence > $OUT/$tag-kld-ub$ub-$arm.log 2>&1
}

tasks() {  # model tag arm env...
  local model=$1 tag=$2 arm=$3; shift 3
  env "$@" $PPL -m $model -f $D/hellaswag_val_full.txt --hellaswag --hellaswag-tasks 400 "${COMMON[@]}" \
    > $OUT/$tag-hellaswag-$arm.log 2>&1
  env "$@" $PPL -m $model -f $D/winogrande-debiased-eval.csv --winogrande "${COMMON[@]}" \
    > $OUT/$tag-winogrande-$arm.log 2>&1
}

run_model() {  # model tag multi-ub opt-array-name [extra-arm-name extra-array-name]
  local model=$1 tag=$2 mub=$3 optname=$4
  local -n opt=$optname
  echo "$(date +%T) $tag"
  kld $model $tag 1 opt "${opt[@]}"
  kld $model $tag $mub opt "${opt[@]}"
  tasks $model $tag base X=0
  tasks $model $tag opt "${opt[@]}"
  if [ $# -ge 6 ]; then
    local -n extra=$6
    kld $model $tag $mub $5 "${extra[@]}"
    tasks $model $tag $5 "${extra[@]}"
  fi
  rm -f $OUT/kld-base-$tag-*.bin
}

run_model $M/Ternary-Bonsai-2-27B-PTQ1_0.gguf b2-ptq1    4 PTQ
run_model $M/Ternary-Bonsai-2-27B-PQ2_0.gguf  b2-pq2     2 PQ2
run_model $M/Ternary-Bonsai-27B-PQ2_0.gguf    b1-ternary 2 PQ2
run_model $M/Bonsai-27B-Q1_0.gguf             b1-binary  4 Q1 popcnt Q1PC
echo "$(date +%T) done"
