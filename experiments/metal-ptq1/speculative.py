#!/usr/bin/env python3
"""Serial local-server checks of real MTP decoding, with tokens and timings retained."""
import argparse,hashlib,json,math,os,random,socket,statistics,subprocess,time,urllib.request
from pathlib import Path
PROMPTS=[
 ('code','Write a Python function that merges two sorted lists into one sorted list, with a docstring.'),
 ('prose','Explain the difference between mmap and read for loading large files, in one paragraph.'),
 ('bash','Write a bash script that watches a directory and prints new files as they appear.'),
]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
p.add_argument('--baseline-model',type=Path)
p.add_argument('--output',type=Path,required=True);p.add_argument('--mode',choices=['smoke','screen','parity','abba'],required=True)
p.add_argument('--draft',type=int,choices=[1,2,3],default=1);p.add_argument('--pair',choices=['total','kernel'],default='total')
p.add_argument('--tokens',type=int,default=128);p.add_argument('--cycles',type=int,default=3)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[2]
meta=dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},revision=subprocess.check_output(['git','-C',root,'rev-parse','HEAD'],text=True).strip(),
 diff_sha256=hashlib.sha256(subprocess.check_output(['git','-C',root,'diff'])).hexdigest(),start=time.time(),
 artifact_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in a.binary.resolve().parent.iterdir() if f.is_file() and not f.is_symlink() and (f.suffix=='.dylib' or f.name=='llama-server')})
(a.output/'metadata.json').write_text(json.dumps(meta,indent=2))
results=[]
def run(name,kernel,draft,prompts):
 with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
 url=f'http://127.0.0.1:{port}'
 cmd=[str(a.binary.resolve()),'-m',str((a.baseline_model if a.baseline_model and not draft else a.model).resolve()),'-c','4096','-b','512','-ub','512','-ngl','99','--device','MTL0','-fa','on','-np','1','-t','16','--jinja','--metrics','--host','127.0.0.1','--port',str(port),'--no-context-shift','--spec-type','draft-mtp' if draft else 'none']
 if draft:cmd+=['--spec-draft-n-max',str(draft),'--spec-draft-n-min','0','--spec-draft-p-min','0']
 env=os.environ.copy()
 for k in list(env):
  if k.startswith('GGML_METAL_PTQ1_'):del env[k]
 env.update(GGML_METAL_PTQ1_MULTICOL=str(kernel),GGML_METAL_PTQ1_NR0='4',GGML_METAL_PTQ1_NSG='1')
 (a.output/(name+'.command.json')).write_text(json.dumps(dict(command=cmd,kernel=kernel,draft=draft),indent=2))
 def request(path,data=None):
  req=urllib.request.Request(url+path,data=json.dumps(data).encode() if data is not None else None,headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=180) as res:return json.load(res)
 def complete(prompt,tokens):
  formatted=request('/apply-template',dict(messages=[dict(role='user',content=prompt)],chat_template_kwargs={'enable_thinking':False}))['prompt']
  body=dict(prompt=formatted,n_predict=tokens,temperature=0,seed=20260923,repeat_penalty=1.0,cache_prompt=False,return_tokens=True,stream=False)
  start=time.perf_counter();res=request('/completion',body);elapsed=time.perf_counter()-start
  if not isinstance(res.get('tokens'),list) or not res['tokens']:raise RuntimeError('Missing token IDs')
  if res['timings']['predicted_n']<=0:raise RuntimeError('Missing generation timing')
  return dict(response=res,elapsed_s=elapsed,prompt=formatted)
 with (a.output/(name+'.log')).open('w') as log:
  proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=log)
  try:
   process_started=time.time()
   deadline=time.monotonic()+180
   while True:
    if proc.poll() is not None:raise RuntimeError(name+' server exited; inspect its log')
    try:
     request('/health');break
    except Exception:
     if time.monotonic()>deadline:raise RuntimeError('Server startup timeout')
     time.sleep(.25)
   warm=complete('Write a short Python function that adds two numbers.',32)
   (a.output/(name+'-warmup.json')).write_text(json.dumps(warm,indent=2))
   rows=[]
   for key,prompt in prompts:
    row=complete(prompt,a.tokens);row.update(name=name,kernel=kernel,draft=draft,case=key)
    (a.output/(name+'-'+key+'.json')).write_text(json.dumps(row,indent=2))
    rows.append(row);t=row['response']['timings'];print(name,key,round(t['predicted_per_second'],2),'tokens/s','accept',t.get('draft_n_accepted',0),'/',t.get('draft_n',0),flush=True)
   with urllib.request.urlopen(url+'/metrics') as res:(a.output/(name+'.metrics')).write_bytes(res.read())
   return rows
  finally:
   proc.terminate()
   try:proc.wait(timeout=15)
   except subprocess.TimeoutExpired:proc.kill();proc.wait()
   with (a.output/'processes.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,start=process_started,end=time.time(),exit_code=proc.returncode))+'\n')
if a.mode in ['smoke','screen','parity']:
 configs=([(1,a.draft)] if a.mode=='smoke' else [(0,0),(1,a.draft)] if a.mode=='parity' else [(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3)])
 for kernel,draft in configs:
  results+=run(f'k{kernel}-d{draft}',kernel,draft,PROMPTS if a.mode!='smoke' else PROMPTS[:1])
  (a.output/'samples.json').write_text(json.dumps(results,indent=2))
else:
 for cycle in range(a.cycles):
  prompts=PROMPTS.copy();random.Random(20260923+cycle).shuffle(prompts)
  for prompt in prompts:
   for slot,variant in enumerate('ABBA'):
    kernel=int(variant=='B');draft=a.draft if variant=='B' or a.pair=='kernel' else 0
    rows=run(f'c{cycle+1}-{prompt[0]}-{slot+1}{variant}',kernel,draft,[prompt])
    for row in rows:row.update(cycle=cycle,slot=slot,variant=variant)
    results+=rows;(a.output/'samples.json').write_text(json.dumps(results,indent=2))
 summary={}
 for key,_ in PROMPTS:
  rows=[r for r in results if r['case']==key];ratios=[]
  for cycle in range(a.cycles):
   aa=[r['response']['timings']['predicted_per_token_ms'] for r in rows if r['cycle']==cycle and r['variant']=='A']
   bb=[r['response']['timings']['predicted_per_token_ms'] for r in rows if r['cycle']==cycle and r['variant']=='B']
   ratios.append(statistics.mean(aa)/statistics.mean(bb))
  base=next(r['response']['tokens'] for r in rows if r['variant']=='A')
  summary[key]=dict(paired_speedups=ratios,paired_geomean=math.exp(statistics.mean(map(math.log,ratios))),all_tokens_equal=all(r['response']['tokens']==base for r in rows))
  for v in ['A','B']:
   rr=[r for r in rows if r['variant']==v];tt=[r['response']['timings'] for r in rr]
   summary[key][v]=dict(mean_tps=statistics.mean(t['predicted_per_second'] for t in tt),mean_wall_tps=statistics.mean(r['response']['timings']['predicted_n']/r['elapsed_s'] for r in rr),tokens=[t['predicted_n'] for t in tt],acceptance=sum(t.get('draft_n_accepted',0) for t in tt)/max(1,sum(t.get('draft_n',0) for t in tt)))
 (a.output/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
