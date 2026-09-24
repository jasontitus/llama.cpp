#!/bin/bash
T=build-dev/bin/test-backend-ops
for cfg in "GGML_METAL_BATCH_INVARIANT=1" "GGML_METAL_PTQ1_MULTICOL=1 GGML_METAL_BATCH_INVARIANT=1 GGML_METAL_PTQ1_MULTICOL_MAX=8 GGML_METAL_PTQ1_GLU=1 GGML_METAL_PTQ1_STAGE=1 GGML_METAL_SMALLM_MM=1"; do
  read -ra E <<< "$cfg"
  echo "=== [$cfg]"
  for suite in "MUL_MAT -p ptq1_0" "MUL_MAT -p bf16" "MUL_MAT -p f16" "MUL_MAT_VEC_FUSION -p ptq1_0"; do
    out=$(env "${E[@]}" GGML_TEST_SEED=20260923 $T test -b MTL0 -o $suite 2>&1)
    echo "  $suite: $(echo "$out" | grep -E 'tests passed' | tr -s ' ') $(echo "$out" | grep -c FAIL) FAIL; kernels: $(echo "$out" | grep -oE 'loaded kernel_mul_mv_ptq1_0_f32_(mc|glu|mcs|glus)_r[0-9]_c[0-9]' | sed 's/loaded kernel_mul_mv_ptq1_0_f32_//' | sort -u | tr '\n' ' ')"
  done
  out=$(env "${E[@]}" GGML_TEST_SEED=20260923 $T test -b MTL0 -o MUL_MAT --test-file m5/decode-shapes.txt 2>&1); echo "  fixtures: $(echo "$out" | grep -E 'tests passed' | tr -s ' ')"
  out=$(env "${E[@]}" PER_N=1 build-dev/bin/test-metal-ptq1-pern 2>/dev/null)
  echo "  strict: $(echo "$out" | grep -c ' PASS') PASS $(echo "$out" | grep -c ' FAIL') FAIL; nonzero batch error by n: $(echo "$out" | grep -E '^   n=' | awk '{split($1,a,"="); split($2,b,"="); if (b[2]+0>0) nz[a[2]]++} END {for (n in nz) printf "n%s=%d ", n, nz[n]}')"
done
