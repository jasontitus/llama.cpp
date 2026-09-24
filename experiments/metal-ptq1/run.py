#!/usr/bin/env python3
"""Run serialized PTQ1 Metal experiments; refuse missing devices/results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--model', type=Path)
p.add_argument('--repeats', type=int, default=3)
p.add_argument('--rows', type=int, choices=[2, 4, 8], default=4)
p.add_argument('--simdgroups', type=int, choices=[1, 2, 4], default=1)
p.add_argument('--candidate', action='store_true')
a = p.parse_args()
if a.repeats < 1: p.error('--repeats must be positive')
a.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[2]
bin_dir = a.build.resolve() / 'bin'
env = os.environ.copy()
for key in list(env):
    if key.startswith('GGML_METAL_PTQ1_'): del env[key]
env.update(GGML_METAL_PTQ1_MULTICOL=str(int(a.candidate)),
           GGML_METAL_PTQ1_NR0=str(a.rows), GGML_METAL_PTQ1_NSG=str(a.simdgroups))

def run(name, command):
    (a.output / (name + '.command.json')).write_text(json.dumps([str(c) for c in command]))
    with (a.output / (name + '.txt')).open('w') as out, (a.output / (name + '.log')).open('w') as err:
        subprocess.run(command, env=env, stdout=out, stderr=err, check=True)
    return (a.output / (name + '.txt')).read_text()

metadata = dict(platform=platform.platform(), processor=platform.processor(),
                revision=subprocess.check_output(['git', '-C', root, 'rev-parse', 'HEAD'], text=True).strip(),
                diff_sha256=hashlib.sha256(subprocess.check_output(['git', '-C', root, 'diff'])).hexdigest(),
                environment={k:v for k,v in env.items() if k.startswith(('GGML_', 'MTL_'))},
                model=str(a.model.resolve()) if a.model else None)
(a.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
args = [bin_dir / 'test-backend-ops', 'perf', '-b', 'MTL0', '-o', 'MUL_MAT',
        '--test-file', root / 'experiments/metal-ptq1/projections.txt']
results = {}
for rep in range(a.repeats):
    text = run(f'perf-{rep+1}', args)
    rows = re.findall(r'name=(k\d+_m\d+_n\d+).*?([\d.]+) us/run', text)
    if len(rows) != 20: raise RuntimeError(f'Expected 20 timing rows; got {len(rows)}')
    for name, us in rows: results.setdefault(name, []).append(float(us))
    print(f'Completed sweep {rep+1}/{a.repeats}', flush=True)
(a.output / 'timings.json').write_text(json.dumps({k:dict(samples_us=v, median_us=statistics.median(v)) for k,v in results.items()}, indent=2))
args[1] = 'test'
text = run('correctness', args)
if '20/20 tests passed' not in text: raise RuntimeError('Missing projection correctness results')
args[-1] = root / 'experiments/metal-ptq1/edges.txt'
text = run('edges', args)
if '48/48 tests passed' not in text: raise RuntimeError('Missing edge correctness results')
text = run('numerical', [bin_dir / 'test-metal-ptq1'])
if text.count('PASS') != 42 or 'FAIL' in text: raise RuntimeError('Missing numerical results')
if a.model:
    text = run('llama-bench', [bin_dir / 'llama-bench', '-m', a.model.resolve(),
               '-p', '1,2,3,4,8,16', '-n', '32', '-r', '5', '-ngl', '99', '-fa', 'on', '-t', '16', '-o', 'json'])
    data = json.loads(text)
    if len(data) != 7: raise RuntimeError('Expected 7 llama-bench results')
print('Experiment complete', flush=True)
