#!/usr/bin/env python3
"""Generate native test-backend-ops fixtures from this checkout's enum values."""
from pathlib import Path
import re
root = Path(__file__).resolve().parents[2]
header = (root / 'ggml/include/ggml.h').read_text()
ops = re.findall(r'^\s*(GGML_OP_\w+)', header.split('enum ggml_op {')[1].split('};')[0], re.M)
op = ops.index('GGML_OP_MUL_MAT')
ptq = int(re.search(r'GGML_TYPE_PTQ1_0\s*=\s*(\d+)', header)[1])
f16 = int(re.search(r'GGML_TYPE_F16\s*=\s*(\d+)', header)[1])
def source(typ, shape, strides=None):
    return ' '.join(map(str, [typ, *shape, *(strides or [0]*4)]))
def case(k, m, n, name, broadcast=False, padded=False, typ=0):
    a = source(ptq, [k, m, 2 if broadcast else 1, 1])
    shape = [k, n, 4 if broadcast else 1, 2 if broadcast else 1]
    elem = 2 if typ == f16 else 4
    row = elem*(k + (32 if padded else 0))
    b = source(typ, shape, [elem, row, row*n, row*n*shape[2]])
    return f'{op} 0 {m} {n} {shape[2]} {shape[3]} 0 2 {a} {b} {name}'
out = Path(__file__).resolve().parent
(out / 'projections.txt').write_text('\n'.join(
    case(k,m,n,f'k{k}_m{m}_n{n}')
    for k,m in [(5120,10240),(5120,17408),(17408,5120),(5120,6144)]
    for n in [1,2,3,4,8])+'\n')
(out / 'edges.txt').write_text('\n'.join(
    case(k,m,n,f'edge_k{k}_m{m}_n{n}_pad{int(pad)}_t{typ}', True, pad, typ)
    for k,m in [(128,3),(384,7),(5120,9)]
    for n in [1,2,3,4,5,6,7,8]
    for pad in [False, True]
    for typ in [0, f16])+'\n')
