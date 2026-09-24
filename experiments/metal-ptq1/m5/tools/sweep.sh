#!/bin/zsh
# usage: sweep.sh fixture label [ENV=VAL ...]  -> prints "label name us"
f=$1; label=$2; shift 2
env GGML_TEST_SEED=20260923 "$@" ../build-dev/bin/test-backend-ops perf -b MTL0 -o MUL_MAT --test-file $f 2>/dev/null | grep -oE 'name=k[0-9_mn]+.* ([0-9.]+) us/run' | sed -E "s/name=(k[0-9_mn]+).* ([0-9.]+) us\/run/$label \1 \2/"
