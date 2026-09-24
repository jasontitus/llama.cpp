#!/usr/bin/env python3
"""M5 copy of experiment/check-model.py with arbitrary A/B environments."""
"""Compare full-vocabulary logits and greedy tokens on identical model prefixes."""
import argparse,array,json,math,os
from pathlib import Path
import subprocess
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--env-a',default='{}',help='JSON env for arm A')
p.add_argument('--env-b',default='{}',help='JSON env for arm B')
p.add_argument('--max-nmse',type=float,default=1e-8)
p.add_argument('--max-absolute-error',type=float,default=0.01)
p.add_argument('--ubatches',type=int,nargs='+',choices=list(range(1,9))+[16,32,64],default=[2,3,4])
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
prompts=[
 'The capital of France is',
 'Question: What is 17 times 23? Answer:',
 'def fibonacci(n):\n    """Return the nth Fibonacci number."""\n',
]
result=[]
for ubatch in a.ubatches:
 for case,prompt in enumerate(prompts):
  prefix={v:a.output/f'u{ubatch}-p{case}-{v}' for v in ['A','B']}
  for v in ['A','B']:
   env={k:x for k,x in os.environ.items() if not k.startswith(('GGML_METAL_PTQ1','GGML_GDN'))};env.update(json.loads(a.env_a if v=='A' else a.env_b))
   cmd=[str(a.binary.resolve()),str(a.model.resolve()),str(ubatch),str(prefix[v]),prompt]
   if v=='B':cmd.append(str(prefix['A'])+'.tokens')
   with Path(str(prefix[v])+'.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=log,check=True)
  aa=array.array('f');aa.frombytes(Path(str(prefix['A'])+'.logits').read_bytes())
  bb=array.array('f');bb.frombytes(Path(str(prefix['B'])+'.logits').read_bytes())
  if len(aa)!=len(bb):raise RuntimeError('logit size mismatch')
  ta=Path(str(prefix['A'])+'.tokens').read_text().splitlines();tb=Path(str(prefix['B'])+'.tokens').read_text().splitlines()
  if len(ta)!=len(tb):raise RuntimeError('token count mismatch')
  if not ta or not aa or len(aa)%len(ta) or len(aa)<=len(ta):raise RuntimeError('Empty or inconsistent logit/token records')
  if not all(math.isfinite(x) for x in aa) or not all(math.isfinite(x) for x in bb):raise RuntimeError('Nonfinite logits')
  err=norm=0.;max_abs=0.;changed=0
  for x,y in zip(aa,bb):
   delta=float(x)-float(y);err+=delta*delta;norm+=float(x)*x;max_abs=max(max_abs,abs(delta));changed+=x!=y
  r=dict(ubatch=ubatch,prompt=prompt,steps=len(ta),token_mismatches=sum(x!=y for x,y in zip(ta,tb)),
         logit_count=len(aa),changed_logits=changed,max_abs_logit_difference=max_abs,nmse=err/max(norm,1e-30),
         text_identical=Path(str(prefix['A'])+'.text').read_bytes()==Path(str(prefix['B'])+'.text').read_bytes())
  result.append(r);(a.output/'summary.json').write_text(json.dumps(result,indent=2))
  print(json.dumps(r),flush=True)

if any(r['token_mismatches'] or not r['text_identical'] or r['nmse']>a.max_nmse or r['max_abs_logit_difference']>a.max_absolute_error for r in result):
 raise SystemExit('Output or logit tolerance failed; inspect summary.json before promoting this candidate')
