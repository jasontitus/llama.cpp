#!/usr/bin/env python3
"""Paired ABBA projection and llama-bench measurements, one GPU process at a time."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--model', type=Path)
p.add_argument('--cycles', type=int, default=3)
p.add_argument('--rows', type=int, choices=[2,4,8], default=4)
p.add_argument('--simdgroups', type=int, choices=[1,2,4], default=1)
p.add_argument('--shape', help='single fixture name, e.g. k5120_m10240_n2')
a = p.parse_args()
if a.cycles < 2: p.error('use at least two ABBA cycles')
a.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parents[2]
bins = a.build.resolve() / 'bin'
base_env = os.environ.copy()
for key in list(base_env):
    if key.startswith('GGML_METAL_PTQ1_'): del base_env[key]
base_env.update(GGML_TEST_SEED='20260923', GGML_METAL_PTQ1_NR0=str(a.rows),
                GGML_METAL_PTQ1_NSG=str(a.simdgroups))
record = dict(platform=platform.platform(), start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              revision=subprocess.check_output(['git','-C',root,'rev-parse','HEAD'],text=True).strip(),
              binary_sha256=hashlib.sha256((bins/'test-backend-ops').read_bytes()).hexdigest(),
              artifact_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in sorted(bins.iterdir())
                               if f.is_file() and not f.is_symlink() and
                               (f.suffix in ['.dylib', '.metallib'] or f.name in ['test-backend-ops', 'llama-bench', 'llama-perplexity'])},
              diff_sha256=hashlib.sha256(subprocess.check_output(['git','-C',root,'diff'])).hexdigest(),
              environment={k:v for k,v in base_env.items() if k.startswith(('GGML_','MTL_'))},
              cycles=a.cycles, sequence='ABBA', rows=a.rows, simdgroups=a.simdgroups,
              A='same binary, GGML_METAL_PTQ1_MULTICOL=0', B='same binary, GGML_METAL_PTQ1_MULTICOL=1')
(a.output/'metadata.json').write_text(json.dumps(record,indent=2))

def run(name, command, variant):
    env=base_env.copy(); env['GGML_METAL_PTQ1_MULTICOL']=str(int(variant=='B'))
    start=time.time()
    with (a.output/(name+'.txt')).open('w') as out, (a.output/(name+'.log')).open('w') as err:
        subprocess.run(command,env=env,stdout=out,stderr=err,check=True)
    with (a.output/'commands.jsonl').open('a') as f:
        f.write(json.dumps(dict(name=name,variant=variant,start=start,end=time.time(),command=[str(c) for c in command]))+'\n')
    telemetry=os.environ.get('BONSAI_BENCH_TELEMETRY')
    if telemetry and os.environ.get('BONSAI_ALLOW_CODEX')!='1':
        for line in Path(telemetry).read_text().splitlines():
            try: sample=json.loads(line)
            except json.JSONDecodeError: continue
            if start <= sample['time'] <= time.time() and any(x['name']=='Codex (Service)' for x in sample.get('processes',[])):
                raise RuntimeError('Codex Service appeared; retain this partial run and restart projections after quitting it')
    return (a.output/(name+'.txt')).read_text()

def summarize(samples):
    A=[s['us'] for s in samples if s['variant']=='A']; B=[s['us'] for s in samples if s['variant']=='B']
    ratios=[]
    for cycle in range(a.cycles):
        x=[s for s in samples if s['cycle']==cycle]
        aa=statistics.mean(s['us'] for s in x if s['variant']=='A')
        bb=statistics.mean(s['us'] for s in x if s['variant']=='B')
        ratios.append(aa/bb)
    return dict(A_mean_us=statistics.mean(A), B_mean_us=statistics.mean(B),
                A_cv_pct=100*statistics.stdev(A)/statistics.mean(A),
                B_cv_pct=100*statistics.stdev(B)/statistics.mean(B),
                paired_speedups=ratios, paired_geomean_speedup=math.exp(statistics.mean(map(math.log,ratios))),
                speedup_min=min(ratios),speedup_max=max(ratios), samples=samples)

names=[line.split()[-1] for line in (root/'experiments/metal-ptq1/projections.txt').read_text().splitlines()]
if a.shape:
    if a.shape not in names: p.error('unknown shape')
    names=[a.shape]
# Different shape order each cycle; identical A/B shape and data within each quartet.
samples={name:[] for name in names}
for cycle in range(a.cycles):
    order=names.copy(); random.Random(20260923+cycle).shuffle(order)
    for shape in order:
        for attempt in range(1,4):
            pending=[];begin=time.time()
            for slot,variant in enumerate('ABBA'):
                name=f'{shape}-c{cycle+1}-try{attempt}-{slot+1}{variant}'
                text=run(name,[bins/'test-backend-ops','perf','-b','MTL0','-o','MUL_MAT',
                              '--test-file',root/'experiments/metal-ptq1/projections.txt','-p',f'^name={shape},'],variant)
                rows=re.findall(r'name=(k\d+_m\d+_n\d+).*?([\d.]+) us/run',text)
                if len(rows)!=1 or rows[0][0]!=shape:raise RuntimeError('missing/unexpected timing row')
                log=(a.output/(name+'.log')).read_text()
                if variant=='B' and shape.endswith(('_n2','_n3','_n4')) and f'_mc_r{a.rows}_' not in log:
                    raise RuntimeError('candidate dispatch not observed')
                pending.append(dict(cycle=cycle,slot=slot,variant=variant,us=float(rows[0][1])))
            telemetry=os.environ.get('BONSAI_BENCH_TELEMETRY')
            if telemetry:
                from monitor import validate_samples
                validate_samples(telemetry,begin,time.time(),allow_codex=os.environ.get('BONSAI_ALLOW_CODEX')=='1')
            spread=max(max(x['us'] for x in pending if x['variant']==v)/min(x['us'] for x in pending if x['variant']==v) for v in 'AB')
            accepted=spread<=1.20
            with (a.output/'quality-gates.jsonl').open('a') as f:
                f.write(json.dumps(dict(shape=shape,cycle=cycle,attempt=attempt,spread=spread,accepted=accepted,samples=pending))+'\n')
            if accepted:
                samples[shape].extend(pending);break
            if attempt==3:raise RuntimeError('Three unstable projection quartets; all raw samples retained')
            time.sleep(15)
        (a.output/'samples.json').write_text(json.dumps(samples,indent=2))
        print(f'cycle {cycle+1}/{a.cycles}: {shape} ABBA complete',flush=True)
summary={name:summarize(values) for name,values in samples.items()}
(a.output/'summary.json').write_text(json.dumps(summary,indent=2))
if a.model:
    e2e={}
    cases=[('pp1',1,0),('pp2',2,0),('pp3',3,0),('pp4',4,0),('pp8',8,0),('pp16',16,0),('tg32',0,32)]
    for cycle in range(a.cycles):
        order=cases.copy(); random.Random(20260923+cycle).shuffle(order)
        for key,pp,tg in order:
            for slot,variant in enumerate('ABBA'):
                name=f'llama-{key}-c{cycle+1}-{slot+1}{variant}'
                text=run(name,[bins/'llama-bench','-m',a.model.resolve(),'-p',str(pp),'-n',str(tg),
                              '-r','5','-ngl','99','-fa','on','-t','16','-o','json'],variant)
                rows=json.loads(text)
                if len(rows)!=1: raise RuntimeError('expected one llama-bench result')
                row=rows[0]
                if 'MTL' not in row.get('backends',''): raise RuntimeError('Metal backend missing')
                e2e.setdefault(key,[]).append(dict(cycle=cycle,slot=slot,variant=variant,us=row['avg_ns']/1000,
                                                  tokens_per_second=row['avg_ts'],stddev_ts=row['stddev_ts']))
            (a.output/'llama-samples.json').write_text(json.dumps(e2e,indent=2))
            print(f'llama cycle {cycle+1}/{a.cycles}: {key} ABBA complete',flush=True)
    (a.output/'llama-summary.json').write_text(json.dumps({key:summarize(values) for key,values in e2e.items()},indent=2))
print('ABBA measurements complete',flush=True)
