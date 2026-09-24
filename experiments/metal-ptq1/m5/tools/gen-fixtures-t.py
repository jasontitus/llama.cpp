#!/usr/bin/env python3
"""M5 fixtures: every PTQ1_0 decode projection in Bonsai 2 27B, n=1..8 and 16."""
from pathlib import Path
import re, sys
root = Path(__file__).resolve().parents[1] / 'llama.cpp'
header = (root / 'ggml/include/ggml.h').read_text()
ops = re.findall(r'^\s*(GGML_OP_\w+)', header.split('enum ggml_op {')[1].split('};')[0], re.M)
op = ops.index('GGML_OP_MUL_MAT')
ptq = int(re.search(r'GGML_TYPE_' + (sys.argv[3] if len(sys.argv) > 3 else 'PTQ1_0') + r'\s*=\s*(\d+)', header)[1])
def case(k, m, n, name):
    a = ' '.join(map(str, [ptq, k, m, 1, 1, 0, 0, 0, 0]))
    row = 4*k
    b = ' '.join(map(str, [0, k, n, 1, 1, 4, row, row*n, row*n]))
    return f'{op} 0 {m} {n} 1 1 0 2 {a} {b} {name}'
# (K, M, count per token)
SHAPES = {'ffn_up_gate': (5120, 17408), 'ffn_down': (17408, 5120), 'attn_qkv': (5120, 10240),
          'attn_gate': (5120, 6144), 'ssm_out': (6144, 5120), 'attn_q': (5120, 12288),
          'attn_out': (6144, 5120), 'attn_kv': (5120, 1024), 'lm_head': (5120, 248320)}
ns = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else [1,2,3,4,5,6,7,8,16]
seen = set(); lines = []
for key, (k, m) in SHAPES.items():
    if (k, m) in seen: continue
    seen.add((k, m))
    for n in ns: lines.append(case(k, m, n, f'k{k}_m{m}_n{n}'))
Path(sys.argv[2] if len(sys.argv) > 2 else 'decode-shapes.txt').write_text('\n'.join(lines) + '\n')
print(len(lines), 'cases')
