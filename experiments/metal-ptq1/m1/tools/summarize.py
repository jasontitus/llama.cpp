#!/usr/bin/env python3
"""Validate raw ABBA records and write tables, retaining incomplete and rejected cells."""
import argparse
import json
import math
from pathlib import Path
import statistics as st
import subprocess
import sys
import tempfile

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',type=Path,nargs='?')
p.add_argument('--self-test',action='store_true',help='check output acceptance with synthetic records; no GPU required')
a=p.parse_args()
if a.self_test:
    with tempfile.TemporaryDirectory() as tmp:
        fixture=Path(tmp)
        study='headline-ptq1-same-mode'
        (fixture/study).mkdir()
        (fixture/'studies.json').write_text(json.dumps([(study,'b2-ptq1',{},{},[],['s0c1'],None)]))
        for cycle in range(3):
            obs=[dict(arm=arm,tps=10.0,gen_tps_mean=10.0,requests=[dict(slot=0,tokens=[1])]) for arm in 'ABBA']
            (fixture/study/f's0c1-cycle{cycle}.json').write_text(json.dumps(dict(cycle=cycle,obs=obs,ratio=1.0,token_mismatch_slots=0)))
        def summarize_fixture():
            subprocess.run([sys.executable,str(Path(__file__).resolve()),str(fixture)],check=True,capture_output=True)
            return json.loads((fixture/'analysis.json').read_text())[0],(fixture/'RESULTS.md').read_text()
        row,report=summarize_fixture()
        assert row['timing_complete'] and row['output_match'] and row['accepted']
        assert '| Bonsai 2 PTQ1_0 |' in report
        path=fixture/study/'s0c1-cycle0.json'
        q=json.loads(path.read_text())
        q['obs'][1]['requests'][0]['tokens']=[2]
        q['token_mismatch_slots']=2
        path.write_text(json.dumps(q))
        row,report=summarize_fixture()
        assert row['timing_complete'] and not row['output_match'] and not row['accepted']
        assert row['cross_arm_token_mismatches']==2 and row['within_arm_token_mismatches']==1
        assert '| Bonsai 2 PTQ1_0 |' not in report
        assert 'output mismatch' in report
    print('Synthetic matching/mismatching output acceptance checks passed.')
    sys.exit(0)
if a.run is None:
    p.error('run is required unless --self-test is selected')
run=a.run
studies=json.loads((run/'studies.json').read_text())
results=[]
for name,model,A,B,bench,server,draft in studies:
    for cell in bench+server:
        qs=[json.loads(f.read_text()) for f in sorted((run/name).glob(f'{cell}-cycle*.json'))]
        row=dict(study=name,model=model,cell=cell,quartets=len(qs),timing_complete=len(qs)==3,
                 output_match=None,accepted=False,within_arm_token_mismatches=0)
        if qs:
            assert len({q['cycle'] for q in qs})==len(qs)
            for q in qs:
                assert ''.join(o['arm'] for o in q['obs'])=='ABBA'
                values={arm:[o['tps'] for o in q['obs'] if o['arm']==arm] for arm in 'AB'}
                assert all(math.isfinite(x) and x>0 for v in values.values() for x in v)
                assert max(max(v)/min(v) for v in values.values())<=1.20
                assert math.isclose(q['ratio'],st.mean(values['B'])/st.mean(values['A']),rel_tol=1e-12)
                if 'requests' in q['obs'][0]:
                    expected_slots=list(range(int(cell.split('c')[1])))
                    assert all([r['slot'] for r in o['requests']]==expected_slots for o in q['obs'])
                    assert all(r['tokens'] for o in q['obs'] for r in o['requests'])
                    mismatches=sum(ra['tokens']!=rb['tokens'] for oa in q['obs'] if oa['arm']=='A' for ob in q['obs'] if ob['arm']=='B' for ra,rb in zip(oa['requests'],ob['requests']))
                    assert mismatches==q['token_mismatch_slots']
            all_obs=[o for q in qs for o in q['obs']]
            row.update(A_tps=st.mean(o['tps'] for o in all_obs if o['arm']=='A'),B_tps=st.mean(o['tps'] for o in all_obs if o['arm']=='B'),
                speedup=math.exp(st.mean(math.log(q['ratio']) for q in qs)),min_ratio=min(q['ratio'] for q in qs),max_ratio=max(q['ratio'] for q in qs),
                cross_arm_token_mismatches=sum(q.get('token_mismatch_slots',0) for q in qs),rejected=len(list((run/name/'rejected').glob(f'{cell}-*.json'))))
            if server and cell.startswith('s'):
                row.update(A_gen_tps=st.mean(o['gen_tps_mean'] for o in all_obs if o['arm']=='A'),B_gen_tps=st.mean(o['gen_tps_mean'] for o in all_obs if o['arm']=='B'))
                row['within_arm_token_mismatches']=sum(
                    r1['tokens']!=r2['tokens'] for q in qs for arm in 'AB'
                    for r1,r2 in zip(*[o['requests'] for o in q['obs'] if o['arm']==arm]))
                row['output_match']=row['cross_arm_token_mismatches']==0 and row['within_arm_token_mismatches']==0
            row['accepted']=row['timing_complete'] and (not cell.startswith('s') or row['output_match'])
        results.append(row)
(run/'analysis.json').write_text(json.dumps(results,indent=2))
lines=['# M1 Ultra ABBA results','',
       'A/B configurations are recorded in studies.json. Values below are full-request aggregate tok/s for server cells and standard llama-bench tok/s for pp/tg cells. Timing is complete after three quartets pass the spread gate; server results also require matching token IDs across and within arms for acceptance. Bench cells do not test generated output. Paired speedup is the geometric mean of the available quartet ratios; range is the observed quartet range, not a confidence interval.','',
       '| Study | Cell | Quartets | A tok/s | B tok/s | Paired gain | Range | Cross-arm mismatches | Within-arm mismatches | Rejected | Status |',
       '|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|']
for r in results:
    if not r['quartets']:
        lines.append(f"| {r['study']} | {r['cell']} | 0/3 | - | - | - | - | - | - | - | incomplete |")
        continue
    status='output mismatch' if r['output_match'] is False else ('accepted' if r['accepted'] else 'incomplete')
    lines.append(f"| {r['study']} | {r['cell']} | {r['quartets']}/3 | {r['A_tps']:.2f} | {r['B_tps']:.2f} | {(r['speedup']-1)*100:+.1f}% | {r['min_ratio']:.3f}-{r['max_ratio']:.3f}x | {r['cross_arm_token_mismatches']} | {r['within_arm_token_mismatches']} | {r['rejected']} | {status} |")
lines+=['','## Single-user native generation','',
    'These means exclude prompt/request overhead. Only timing-complete cells with matching generated output are shown. Plain and MTP are separate cells; their ratio is descriptive, not a paired ABBA estimate. A/B means all flags off / M5-recommended flags under test. Bonsai 1 has no MTP head in this experiment.','',
    '| Model | Original plain | Original MTP | Optimized plain | Optimized MTP | MTP vs optimized plain |',
    '|---|---:|---:|---:|---:|---:|']
for name,study in [('Bonsai 2 PTQ1_0','headline-ptq1-same-mode'),('Bonsai 2 PQ2_0','abba3-b2-pq2'),('Bonsai 1 ternary','abba3-b1-ternary'),('Bonsai 1 binary','abba3-b1-binary')]:
    plain=next((r for r in results if r['study']==study and r['cell']=='s0c1' and r['accepted']),None)
    mtp=next((r for r in results if r['study']==study and r['cell']=='s1c1' and r['accepted']),None)
    if not plain:
        continue
    oldmtp=f"{mtp['A_gen_tps']:.2f}" if mtp else '-'
    newmtp=f"{mtp['B_gen_tps']:.2f}" if mtp else '-'
    gain=f"{(mtp['B_gen_tps']/plain['B_gen_tps']-1)*100:+.1f}%" if mtp else '-'
    lines.append(f"| {name} | {plain['A_gen_tps']:.2f} | {oldmtp} | {plain['B_gen_tps']:.2f} | {newmtp} | {gain} |")
(run/'RESULTS.md').write_text('\n'.join(lines)+'\n')
print(f'{sum(r["timing_complete"] for r in results)}/{len(results)} timing-complete cells; {sum(r["accepted"] for r in results)} accepted')
