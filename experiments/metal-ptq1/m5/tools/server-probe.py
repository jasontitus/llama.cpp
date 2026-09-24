#!/usr/bin/env python3
"""Start llama-server, run N sequential identical completions, print native timings. Localhost only."""
import argparse, json, os, subprocess, time, urllib.request, socket, sys
p = argparse.ArgumentParser()
p.add_argument('--bin', default=os.path.join(os.path.dirname(__file__), '../build-dev/bin/llama-server'))
p.add_argument('--model', default=os.path.join(os.path.dirname(__file__), '../../workspace/work/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf'))
p.add_argument('--n', type=int, default=128)
p.add_argument('--reps', type=int, default=3)
p.add_argument('--ctx', default='4096')
p.add_argument('--extra', default='')
p.add_argument('--body', default='{}')
p.add_argument('--parallel', type=int, default=1)
a = p.parse_args()
s = socket.socket(); s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]; s.close()
cmd = [a.bin, '-m', a.model, '-c', a.ctx, '-b', '512', '-ub', '512', '-ngl', '99', '--device', 'MTL0', '-fa', 'on',
       '-np', str(a.parallel), '-t', '16', '--jinja', '--host', '127.0.0.1', '--port', str(port), '--no-context-shift',
       '--cache-ram', '0'] + a.extra.split()
srv = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=open('/tmp/claude-probe-server.log' if False else os.devnull, 'w'))
def req(path, body=None):
    r = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(body).encode() if body else None,
                               headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(r, timeout=600))
try:
    for _ in range(600):
        try:
            if req('/health').get('status') == 'ok': break
        except Exception: pass
        time.sleep(0.5)
    prompt = '<|im_start|>user\nWrite a Python function that merges two sorted lists and explain it.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    body = dict(prompt=prompt, n_predict=a.n, temperature=0, seed=20260923, repeat_penalty=1, cache_prompt=False, return_tokens=True)
    body.update(json.loads(a.body))
    req('/completion', dict(body, n_predict=16))
    for i in range(a.reps):
        t = time.perf_counter(); r = req('/completion', body); wall = time.perf_counter() - t
        tm = r['timings']
        print(json.dumps(dict(gen_tps=round(tm['predicted_per_second'], 2), ms_tok=round(tm['predicted_per_token_ms'], 2),
              pp_ms=round(tm['prompt_ms'], 1), n=tm['predicted_n'], wall=round(wall, 3),
              drafts=[tm.get('draft_n'), tm.get('draft_n_accepted')], tok_hash=hash(tuple(r.get('tokens', []))) & 0xffffffff)), flush=True)
finally:
    srv.terminate(); srv.wait()
