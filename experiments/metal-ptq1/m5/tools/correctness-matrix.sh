#!/bin/bash
# Correctness matrix with properly split environments (bash arrays). For each config: test-backend-ops
# suites (+ which new kernels loaded) and the strict 1e-8 checker for the config's weight type.
set -u
T=build-dev/bin/test-backend-ops
run_cfg() {
  local label="$1" type="$2" strict="$3"; shift 3
  local envs=("$@")
  echo "=== $label [${envs[*]}]"
  local out
  for suite in "MUL_MAT -p ${type}" "MUL_MAT_VEC_FUSION -p ${type}"; do
    out=$(env "${envs[@]}" GGML_TEST_SEED=20260923 $T test -b MTL0 -o $suite 2>&1)
    echo "  $(echo "$suite" | cut -d' ' -f1): $(echo "$out" | grep -E 'tests passed' | tr -s ' ') $(echo "$out" | grep -c FAIL) FAIL; new kernels: $(echo "$out" | grep -oE 'loaded kernel_(mul_mv_(ptq1_0|pq2_0|q1_0)_f32_(mc|mcs|glu|glus|tmv|small)[a-z0-9_]*|ptq1_0_(stage|hilo)|mul_mm_ptq1_0_f32_b128|mul_mm_q1_0_f32_k32[a-z0-9_]*)' | sed 's/loaded kernel_//' | sed -E 's/_nsg=.*//' | sort -u | tr '\n' ' ')"
  done
  for f in "${@:0:0}"; do :; done
  if [ -n "$FIX" ]; then
    out=$(env "${envs[@]}" GGML_TEST_SEED=20260923 $T test -b MTL0 -o MUL_MAT --test-file $FIX 2>&1)
    echo "  fixtures $FIX: $(echo "$out" | grep -E 'tests passed' | tr -s ' ') $(echo "$out" | grep -c FAIL) FAIL"
  fi
  if [ -n "$strict" ]; then
    out=$(env "${envs[@]}" $strict 2>/dev/null)
    echo "  strict $(basename $strict): $(echo "$out" | grep -c PASS) PASS / $(echo "$out" | grep -c FAIL) FAIL"
  fi
}
FIX=m5/decode-shapes.txt
run_cfg "PTQ1 baseline"       ptq1_0 build-dev/bin/test-metal-ptq1-m5 X=0
run_cfg "PTQ1 M5 stack"       ptq1_0 build-dev/bin/test-metal-ptq1-m5 GGML_METAL_PTQ1_MULTICOL=1 GGML_METAL_PTQ1_MULTICOL_MAX=8 GGML_METAL_PTQ1_GLU=1 GGML_METAL_PTQ1_STAGE=1
run_cfg "PTQ1 stack+tensor"   ptq1_0 build-dev/bin/test-metal-ptq1-m5 GGML_METAL_PTQ1_MULTICOL=1 GGML_METAL_PTQ1_MULTICOL_MAX=8 GGML_METAL_PTQ1_GLU=1 GGML_METAL_PTQ1_STAGE=1 GGML_METAL_PTQ1_TENSOR=1
run_cfg "PTQ1 tensor n>=5"    ptq1_0 build-dev/bin/test-metal-ptq1-m5 GGML_METAL_PTQ1_TENSOR=1 GGML_METAL_PTQ1_TENSOR_MIN=5
run_cfg "PTQ1 prefill b128"   ptq1_0 "" GGML_METAL_PTQ1_MM_B128=1
FIX=m5/pq2-shapes.txt
run_cfg "PQ2 baseline"        pq2_0  build-dev/bin/test-metal-pq2-m5 X=0
run_cfg "PQ2 stack"           pq2_0  build-dev/bin/test-metal-pq2-m5 GGML_METAL_PQ2_MULTICOL=1 GGML_METAL_PQ2_GLU=1 GGML_METAL_SMALLM=1 GGML_METAL_SMALLM_MM=1
run_cfg "PQ2 stack, max 8"    pq2_0  build-dev/bin/test-metal-pq2-m5 GGML_METAL_PQ2_MULTICOL=1 GGML_METAL_PQ2_GLU=1 GGML_METAL_PQ2_MC_MAX=8
FIX=""
run_cfg "Q1 baseline"         q1_0   build-dev/bin/test-metal-q1-m5 X=0
run_cfg "Q1 stack"            q1_0   build-dev/bin/test-metal-q1-m5 GGML_METAL_Q1_GLU=1 GGML_METAL_SMALLM=1 GGML_METAL_SMALLM_MM=1
run_cfg "Q1 stack, glu max 4" q1_0   build-dev/bin/test-metal-q1-m5 GGML_METAL_Q1_GLU=1 GGML_METAL_Q1_GLU_MAX=4
run_cfg "Q1 prefill K32"      q1_0   build-dev/bin/test-metal-q1-m5 GGML_METAL_Q1_MM_K32_ALIGNED=1
run_cfg "Q1 recommended + swz 1" q1_0 build-dev/bin/test-metal-q1-m5 GGML_GDN_ROWS_PLAIN=1 GGML_METAL_SMALLM=1 GGML_METAL_SMALLM_MM=1 GGML_METAL_Q1_SWIZZLE_LOG=1
run_cfg "BF16 small rows"     bf16   "" GGML_METAL_SMALLM_MM=1
