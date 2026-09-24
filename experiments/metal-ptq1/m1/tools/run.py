#!/usr/bin/env python3
"""Replay the portable M5 studies and selected M1 comparisons with the M5 ABBA harness."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--workspace', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--studies', nargs='+', help='run named studies in this order; omit for the original full suite')
a = p.parse_args()
root = a.workspace.resolve()
out = a.output.resolve()
src = Path(__file__).resolve().parents[4]
bin_dir = root / 'work/build-tuning/bin'
tools = src / 'experiments/metal-ptq1/m5/tools'
env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_', 'LLAMA_ARG_'))}
env['GGML_TEST_SEED'] = '20260923'
ptq = dict(GGML_METAL_PTQ1_MULTICOL='1', GGML_METAL_PTQ1_MULTICOL_MAX='8', GGML_METAL_PTQ1_GLU='1', GGML_METAL_PTQ1_STAGE='1', GGML_GDN_ROWS_PLAIN='1', GGML_METAL_SMALLM_MM='1')
pq2 = dict(GGML_METAL_PQ2_MULTICOL='1', GGML_METAL_PQ2_GLU='1', GGML_GDN_ROWS_PLAIN='1', GGML_METAL_SMALLM='1', GGML_METAL_SMALLM_MM='1')
q1 = dict(GGML_GDN_ROWS_PLAIN='1', GGML_METAL_SMALLM='1', GGML_METAL_SMALLM_MM='1')

# Same cells, flags, prompts, and ABBA settings as the committed M5 scripts.
studies=[
 ('headline-ptq1-same-mode','b2-ptq1',{},ptq,['tg128'],['s0c1','s1c1'],None),
 ('headline-ptq1-total','b2-ptq1',{},ptq,[],['s1c1'],[0,1]),
 ('headline-pq2-total','b2-pq2',{},pq2,[],['s1c1'],[0,1]),
 ('abba3-b2-pq2','b2-pq2',{},pq2,['tg128','pp2','pp3','pp32','pp512'],['s0c1','s1c1','s0c2','s1c2'],None),
 ('abba3-b1-ternary','b1-ternary',{},pq2,['tg128','pp2','pp32','pp512'],['s0c1','s0c2'],None),
 ('abba3-b1-binary','b1-binary',{},q1,['tg128','pp2','pp32','pp512'],['s0c1','s0c2'],None),
 ('abba3-b2-ptq1-smallm','b2-ptq1',{k:v for k,v in ptq.items() if k!='GGML_METAL_SMALLM_MM'},ptq,['pp16','pp32','pp128','pp512'],['s0c1','s1c1'],None),
 ('bitexact-ptq1','b2-ptq1',{},dict(GGML_GDN_ROWS_PLAIN='1'),['tg128'],['s0c1'],None),
 ('bitexact-pq2','b2-pq2',{},dict(GGML_GDN_ROWS_PLAIN='1'),['tg128'],['s0c1'],None),
 ('abba5-invariant-cost','b2-ptq1',ptq,dict(ptq,GGML_METAL_BATCH_INVARIANT='1'),['tg128','pp2','pp4','pp8'],['s0c1','s1c1','s1c2'],None),
 ('abba1-m1cand-vs-m5stack','b2-ptq1',dict(GGML_METAL_PTQ1_MULTICOL='1'),{k:v for k,v in ptq.items() if k!='GGML_METAL_SMALLM_MM'},['tg128','pp2','pp3','pp4','pp8','pp512'],['s0c1','s1c1','s0c2','s1c2','s0c4','s1c4'],None),
]
# These targeted comparisons run only when explicitly selected.
optional_studies=[
 ('m1-geometry','b2-ptq1',ptq,ptq,['pp2','pp3'],['s1c1'],None),
 ('previous-pr','b2-ptq1',dict(GGML_METAL_PTQ1_MULTICOL='1'),ptq,['pp2','pp3'],['s0c1','s1c1'],None),
 ('previous-pr-no-stage','b2-ptq1',dict(GGML_METAL_PTQ1_MULTICOL='1'),dict(ptq,GGML_METAL_PTQ1_STAGE='0'),[],['s1c1'],None),
 ('previous-pr-mc-only','b2-ptq1',dict(GGML_METAL_PTQ1_MULTICOL='1'),dict(GGML_METAL_PTQ1_MULTICOL='1'),['pp2','pp3'],['s1c1'],None),
 ('m1-no-stage-prefill','b2-ptq1',dict(GGML_METAL_PTQ1_MULTICOL='1'),dict(ptq,GGML_METAL_PTQ1_STAGE='0'),['pp2','pp3'],[],None),
 ('full-column-specialization','b2-ptq1',dict(GGML_METAL_PTQ1_MULTICOL='1',GGML_METAL_PTQ1_MULTICOL_MAX='8'),dict(GGML_METAL_PTQ1_MULTICOL='1',GGML_METAL_PTQ1_MULTICOL_MAX='8'),['pp2','pp3'],['s1c1'],None),
 ('previous-pr-m1','b2-ptq1',dict(GGML_METAL_PTQ1_MULTICOL='1'),dict(ptq,GGML_METAL_PTQ1_STAGE='0'),['pp2','pp3'],['s0c1','s1c1'],None),
]
comparison_builds={
    'm1-geometry':(root/'work/llama-tuning-before',root/'work/build-tuning-before/bin'),
    'previous-pr':(root/'work/llama-pr',root/'work/build-pr/bin'),
    'previous-pr-no-stage':(root/'work/llama-pr',root/'work/build-pr/bin'),
    'previous-pr-mc-only':(root/'work/llama-pr',root/'work/build-pr/bin'),
    'm1-no-stage-prefill':(root/'work/llama-pr',root/'work/build-pr/bin'),
    'full-column-specialization':(root/'work/llama-tuning-r4',root/'work/build-tuning-r4/bin'),
    'previous-pr-m1':(root/'work/llama-pr',root/'work/build-pr/bin'),
}
if a.studies is not None:
    available={study[0]:study for study in studies+optional_studies}
    unknown=[name for name in a.studies if name not in available]
    if unknown:
        p.error('unknown studies: '+', '.join(unknown))
    if len(set(a.studies))!=len(a.studies):
        p.error('--studies cannot contain duplicate names')
    studies=[available[name] for name in a.studies]
out.mkdir(parents=True, exist_ok=False)

model_checks={
    'b2-ptq1':('ptq1_0','m5'),
    'b2-pq2':('pq2_0','pq2'),
    'b1-ternary':('pq2_0','pq2'),
    'b1-binary':('q1_0','q1'),
}
selected_models={study[1] for study in studies}
validation_configs={}
for name,model,A,B,bench,server,draft in studies:
    typ,checker=model_checks[model]
    for arm,config in [('A',A),('B',B)]:
        key=(typ,json.dumps(config,sort_keys=True))
        if key not in validation_configs:
            label='config-'+hashlib.sha256(key[1].encode()).hexdigest()[:12]
            validation_configs[key]=dict(type=typ,checker=checker,label=label,flags=config,uses=[])
        validation_configs[key]['uses'].append(dict(study=name,model=model,arm=arm,
            benchmark_build='comparison_a' if arm=='A' and name in comparison_builds else 'current'))


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def capture(cmd):
    return subprocess.check_output(cmd, text=True).strip()

def snapshot_sources(paths):
    records={}
    for path in paths:
        relative=path.relative_to(src)
        snapshot=Path('provenance/sources')/relative
        data=path.read_bytes()
        (out/snapshot).parent.mkdir(parents=True,exist_ok=True)
        (out/snapshot).write_bytes(data)
        records[str(relative)]=dict(sha256=hashlib.sha256(data).hexdigest(),snapshot=str(snapshot))
    return records

# Preserve research files that may be untracked, plus the exact tracked patch.
m1_tools=Path(__file__).resolve().parent
sources=snapshot_sources([
    m1_tools/'run.py',m1_tools/'summarize.py',m1_tools/'thermal-state.m',
    m1_tools.parent/'models.json',tools/'abba-m5.py',tools/'abba-m5-draft.py',
    src/'tests/test-backend-ops.cpp',
    *[tools/f'check-numerical-{kind}.cpp' for kind in ['m5','pq2','q1']],
])
source_diff=subprocess.check_output(['git','-C',str(src),'diff','--no-ext-diff','--binary','HEAD'])
(out/'provenance/tracked.diff').write_bytes(source_diff)
required_binaries=['llama-bench','llama-server','test-backend-ops','thermal-state',
                   *[f'check-numerical-{kind}' for kind in sorted({model_checks[model][1] for model in selected_models})]]
binary_hashes={name:digest(bin_dir/name) for name in required_binaries}
binary_hashes.update({f.name:digest(f) for f in bin_dir.iterdir() if f.suffix=='.dylib'})

comparison_a={}
for study in studies:
    name=study[0]
    if name not in comparison_builds:
        continue
    old_src,old_bin=comparison_builds[name]
    old_head=capture(['git','-C',str(old_src),'rev-parse','HEAD'])
    old_diff=subprocess.check_output(['git','-C',str(old_src),'diff','--no-ext-diff','--binary','HEAD'])
    old_provenance=out/'provenance'/name
    old_provenance.mkdir()
    (old_provenance/'HEAD.txt').write_text(old_head+'\n')
    (old_provenance/'tracked.diff').write_bytes(old_diff)
    old_hashes={name:digest(old_bin/name) for name in ['llama-bench','llama-server']}
    old_hashes.update({f.name:digest(f) for f in old_bin.iterdir() if f.suffix=='.dylib'})
    comparison_a[name]=dict(source=old_head,src=str(old_src),bin=str(old_bin),binaries=old_hashes,
        source_snapshot=f'provenance/{name}/HEAD.txt',
        source_diff_sha256=hashlib.sha256(old_diff).hexdigest(),
        source_diff_snapshot=f'provenance/{name}/tracked.diff')

pins = json.loads((Path(__file__).resolve().parents[1] / 'models.json').read_text())
pins = [pin for pin in pins if pin['key'] in selected_models]
models = {}
for pin in pins:
    models[pin['key']] = root / 'work/models' / pin['file']
    if 'merged_mtp' in pin:
        models[pin['key'] + '-mtp'] = root / 'work/mtp' / pin['merged_mtp']['file']
    for key, expected in [(pin['key'], pin)] + ([(pin['key'] + '-mtp', pin['merged_mtp'])] if 'merged_mtp' in pin else []):
        actual = digest(models[key])
        if actual != expected['sha256']:
            raise RuntimeError(f'model hash mismatch: {key}')
        print(f'Verified {key}: {actual}', flush=True)
manifest = dict(started=time.time(), source=capture(['git','-C',str(src),'rev-parse','HEAD']),
    source_diff_sha256=hashlib.sha256(source_diff).hexdigest(),source_diff_snapshot='provenance/tracked.diff',
    source_snapshots=sources,
    hardware={k:capture(['sysctl','-n',k]) for k in ['hw.model','hw.memsize','hw.ncpu','machdep.cpu.brand_string']},
    os=capture(['sw_vers']), compiler=capture(['clang','--version']), models=pins,
    binaries=binary_hashes,comparison_a=comparison_a,studies=[study[0] for study in studies],
    telemetry_note='Codex remains open. GPU counters cover only the first AGXAccelerator PerformanceStatistics entry returned by ioreg, not an aggregate across entries. Process CPU activity is sampled; per-process GPU attribution unavailable because sudo is not authenticated. Thermal state: 0 nominal, 1 fair, 2 serious, 3 critical.',
    protocol=dict(cycles=3,cooldown_seconds=8,spread_limit=1.20,attempts=3,tokens=128),
    flags=dict(ptq1=ptq,pq2=pq2,q1=q1),
    correctness_scope=dict(build='current',source=capture(['git','-C',str(src),'rev-parse','HEAD']),bin=str(bin_dir),
        note='Every selected A/B flag configuration is checked on the current build. Separate comparison A binaries are not revalidated by these checks; their correctness needs separate evidence.'),
    correctness_configs=list(validation_configs.values()))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))

# Validation is complete before any timed observation.
checks=[]
for validation in validation_configs.values():
    typ,checker,config=validation['type'],validation['checker'],validation['flags']
    for suite, command in [('strict',[str(bin_dir/f'check-numerical-{checker}')]),
        ('MUL_MAT',[str(bin_dir/'test-backend-ops'),'test','-b','MTL0','-o','MUL_MAT','-p',typ]),
        ('MUL_MAT_VEC_FUSION',[str(bin_dir/'test-backend-ops'),'test','-b','MTL0','-o','MUL_MAT_VEC_FUSION','-p',typ])]:
        name=f"{typ}-{validation['label']}-{suite}"
        with (out/f'{name}.log').open('w') as log:
            r=subprocess.run(command,env=dict(env,**config),stdout=log,stderr=subprocess.STDOUT)
        text=(out/f'{name}.log').read_text()
        counts=re.findall(r'(\d+)/(\d+) tests passed',text)
        passed=text.count(' PASS') if suite=='strict' else sum(int(x) for x,y in counts)
        if r.returncode or 'FAIL' in text or not passed:
            raise RuntimeError(f'correctness failed or empty: {name}; see log')
        checks.append(dict(name=name,passed=passed,counts=counts,command=command,flags=config,config_label=validation['label'],uses=validation['uses'],
            build='current',bin=str(bin_dir)))
        print(f'Correctness {name}: {passed} passed',flush=True)
(out/'correctness.json').write_text(json.dumps(checks,indent=2))

stop=threading.Event()
def monitor():
    with (out/'telemetry.jsonl').open('w',buffering=1) as f:
        while not stop.is_set():
            row=dict(time=time.time())
            try:
                raw=capture(['ioreg','-r','-c','AGXAccelerator','-d','1'])
                entries=[x for x in raw.splitlines() if '"PerformanceStatistics"' in x]
                row['gpu_scope']='first AGXAccelerator PerformanceStatistics entry in ioreg output'
                row['gpu_statistics_entries']=len(entries)
                line=entries[0]
                row['gpu']={k:int(v) for k,v in re.findall(r'"([^"\n]+)"=(\d+)',line)}
                row['thermal']=int(capture([str(bin_dir/'thermal-state')]))
                processes=capture(['ps','-axo','pid=,%cpu=,comm='])
                row['processes']=[x.strip() for x in processes.splitlines() if any(n in x for n in ['Codex','llama-server','llama-bench','WindowServer'])]
            except Exception as e:
                row['error']=str(e)
            f.write(json.dumps(row)+'\n')
            stop.wait(2)
thread=threading.Thread(target=monitor,daemon=True)
thread.start()

(out/'studies.json').write_text(json.dumps(studies,indent=2))
status=[]
try:
    for name,model,A,B,bench,server,draft in studies:
        script='abba-m5-draft.py' if draft else 'abba-m5.py'
        cmd=[sys.executable,str(tools/script),'--output',str(out/name),'--bin',str(bin_dir),'--src',str(src),
            '--model',str(models[model]),'--env-a',json.dumps(A),'--env-b',json.dumps(B),
            '--cycles','3','--attempts','3','--cooldown','8','--tokens','128','--spread','1.20',
            '--bench',*bench,'--server',*server]
        if model+'-mtp' in models:
            cmd+=['--mtp-model',str(models[model+'-mtp'])]
        if draft:
            cmd+=['--draft-a',str(draft[0]),'--draft-b',str(draft[1])]
        if name in comparison_builds:
            old_src,old_bin=comparison_builds[name]
            cmd+=['--bin-a',str(old_bin),'--src-a',str(old_src)]
        print(f'Starting {name}',flush=True)
        started=time.time()
        r=subprocess.run(cmd,env=env)
        status.append(dict(study=name,returncode=r.returncode,started=started,ended=time.time(),command=cmd))
        (out/'status.json').write_text(json.dumps(status,indent=2))
finally:
    stop.set();thread.join()
print('Suite finished; inspect status, rejected quartets, and token comparisons before accepting results.',flush=True)
sys.exit(any(x['returncode'] for x in status))
