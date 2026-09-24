#!/usr/bin/env python3
"""ABBA study of two environment arms on one binary: llama-bench cases and fresh-server cells.

Every cell gets --cycles A-B-B-A quartets (cell order shuffled reproducibly per cycle), one GPU
process at a time, a cooldown between observations, and the handoff's 20% within-arm spread gate
(larger/smaller throughput of the two A runs, and of the two B runs, must each be <= 1.20); a failed
quartet is kept under rejected/ and repeated, at most --attempts times. Server cells compare the
returned token IDs of A and B per slot. Localhost only; no data leaves the machine.
"""
import argparse, concurrent.futures, hashlib, json, math, os, random, signal, socket, statistics, subprocess, threading, time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROMPTS = [
    'Write a Python function that merges two sorted lists into one sorted list, with a docstring.',
    'Explain the difference between mmap and read for loading large files, in one paragraph.',
    'Write a bash script that watches a directory and prints new files as they appear.',
    'Explain how a hash table handles collisions, with a worked example and pseudocode.',
]

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--bin', type=Path, default=HERE.parent / 'build-dev/bin')
p.add_argument('--src', type=Path, default=HERE.parent / 'llama.cpp', help='source tree the binaries were built from (provenance)')
p.add_argument('--model', type=Path, default=ROOT / 'workspace/work/models/Ternary-Bonsai-2-27B-PTQ1_0.gguf')
p.add_argument('--mtp-model', type=Path, default=ROOT / 'workspace/work/mtp/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf')
p.add_argument('--env-a', default='{}', help='JSON environment for arm A')
p.add_argument('--env-b', default='{}', help='JSON environment for arm B')
p.add_argument('--bench', nargs='*', default=['tg128', 'pp2', 'pp3', 'pp4', 'pp8', 'pp512'])
p.add_argument('--server', nargs='*', default=['s0c1', 's1c1', 's0c2', 's1c2', 's0c4', 's1c4'],
               help='s<spec 0|1>c<concurrency>')
p.add_argument('--cycles', type=int, default=3)
p.add_argument('--attempts', type=int, default=3)
p.add_argument('--cooldown', type=float, default=8.0)
p.add_argument('--tokens', type=int, default=128)
p.add_argument('--spread', type=float, default=1.20)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
(a.output / 'rejected').mkdir(exist_ok=True)

base_env = {k: v for k, v in os.environ.items() if not k.startswith(('GGML_METAL_', 'GGML_GDN_', 'LLAMA_ARG_'))}
ARMS = {'A': dict(base_env, **json.loads(a.env_a)), 'B': dict(base_env, **json.loads(a.env_b))}

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

meta = dict(created=time.time(), env_a=json.loads(a.env_a), env_b=json.loads(a.env_b), args={k: str(v) for k, v in vars(a).items()},
            binaries={f.name: digest(f) for f in sorted(a.bin.iterdir()) if not f.is_symlink() and f.suffix == '.dylib' or f.name in ('llama-bench', 'llama-server')},
            revision=subprocess.check_output(['git', '-C', a.src, 'rev-parse', 'HEAD'], text=True).strip(), src=str(a.src),
            diff_sha256=hashlib.sha256(subprocess.check_output(['git', '-C', a.src, 'diff'])).hexdigest(),
            thermal=subprocess.run(['pmset', '-g', 'therm'], capture_output=True, text=True).stdout,
            power=subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True).stdout)
(a.output / 'meta.json').write_text(json.dumps(meta, indent=2))
log = open(a.output / 'progress.log', 'a')

def say(msg):
    line = time.strftime('%H:%M:%S ') + msg
    print(line, flush=True)
    log.write(line + '\n'); log.flush()

def bench_obs(case, arm):
    kind, n = case[:2], int(case[2:])
    cmd = [str(a.bin / 'llama-bench'), '-m', str(a.model), '-ngl', '99', '-fa', 'on', '-t', '16', '-r', '3', '-o', 'json',
           '-p', str(n) if kind == 'pp' else '0', '-n', str(n) if kind == 'tg' else '0']
    out = subprocess.run(cmd, env=ARMS[arm], capture_output=True, text=True, check=True).stdout
    rows = json.loads(out)
    if len(rows) != 1:
        raise RuntimeError(f'expected one llama-bench row for {case}')
    r = rows[0]
    return dict(tps=r['avg_ts'], samples_ts=r.get('samples_ts'), command=cmd)

def server_obs(cell, arm, cycle):
    spec, c = int(cell[1]), int(cell[3:])
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
    cmd = [str(a.bin / 'llama-server'), '-m', str(a.mtp_model if spec else a.model), '-c', str(4096 * c), '-b', '512', '-ub', '512',
           '-ngl', '99', '--device', 'MTL0', '-fa', 'on', '-np', str(c), '-t', '16', '--jinja', '--host', '127.0.0.1',
           '--port', str(port), '--no-context-shift', '--cache-ram', '0']
    cmd += ['--spec-type', 'draft-mtp', '--spec-draft-n-max', '1', '--spec-draft-n-min', '0', '--spec-draft-p-min', '0'] if spec else ['--spec-type', 'none']
    def request(path, body=None):
        req = urllib.request.Request(f'http://127.0.0.1:{port}{path}', data=json.dumps(body).encode() if body is not None else None,
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)
    logf = open(a.output / f'server-{cell}-c{cycle}-{arm}.log', 'w')
    proc = subprocess.Popen(cmd, env=ARMS[arm], stdout=logf, stderr=logf, start_new_session=True)
    try:
        deadline = time.monotonic() + 300
        while True:
            try:
                if request('/health').get('status') == 'ok':
                    break
            except Exception:
                pass
            if proc.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f'server failed to start for {cell} {arm}')
            time.sleep(0.5)
        texts = [PROMPTS[(cycle + i) % len(PROMPTS)] for i in range(c)]
        formatted = [request('/apply-template', dict(messages=[dict(role='user', content=t)], chat_template_kwargs={'enable_thinking': False}))['prompt'] for t in texts]
        def wave(n):
            barrier = threading.Barrier(c); t0 = time.perf_counter()
            def one(i):
                body = dict(prompt=formatted[i], n_predict=n, temperature=0, seed=20260923, repeat_penalty=1, cache_prompt=False, return_tokens=True, id_slot=i)
                barrier.wait(); t = time.perf_counter(); r = request('/completion', body); e = time.perf_counter()
                if not r.get('tokens'):
                    raise RuntimeError('missing token ids')
                return dict(slot=i, start=t - t0, end=e - t0, tokens=r['tokens'], timings=r['timings'])
            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as pool:
                rows = list(pool.map(one, range(c)))
            span = max(x['end'] for x in rows) - min(x['start'] for x in rows)
            return rows, span
        wave(32)  # warm every slot
        rows, span = wave(a.tokens)
        agg = sum(r['timings']['predicted_n'] for r in rows) / span
        gen = statistics.mean(r['timings']['predicted_per_second'] for r in rows)
        return dict(tps=agg, gen_tps_mean=gen, wall_s=span, requests=rows, command=cmd)
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        logf.close()

def quartet(cell, cycle):
    dest = a.output / f'{cell}-cycle{cycle}.json'
    if dest.exists():
        return json.loads(dest.read_text())
    for attempt in range(1, a.attempts + 1):
        obs = []
        for pos, arm in enumerate('ABBA'):
            time.sleep(a.cooldown)
            t = time.time()
            r = bench_obs(cell, arm) if cell.startswith(('tg', 'pp')) else server_obs(cell, arm, cycle)
            r.update(arm=arm, pos=pos, started=t)
            obs.append(r)
            say(f'{cell} cycle{cycle} att{attempt} {pos+1}{arm} {r["tps"]:.2f} tok/s')
        A = [o['tps'] for o in obs if o['arm'] == 'A']; B = [o['tps'] for o in obs if o['arm'] == 'B']
        spread = max(max(A) / min(A), max(B) / min(B))
        q = dict(cell=cell, cycle=cycle, attempt=attempt, obs=obs, spread=spread, ratio=statistics.mean(B) / statistics.mean(A))
        if 'requests' in obs[0]:
            q['token_mismatch_slots'] = sum(ra['tokens'] != rb['tokens'] for oa in obs if oa['arm'] == 'A' for ob in obs if ob['arm'] == 'B'
                                            for ra, rb in zip(oa['requests'], ob['requests']))
        if spread <= a.spread:
            dest.write_text(json.dumps(q, indent=1))
            return q
        (a.output / 'rejected' / f'{cell}-cycle{cycle}-att{attempt}.json').write_text(json.dumps(q, indent=1))
        say(f'REJECTED {cell} cycle{cycle} att{attempt} spread {spread:.3f}')
    raise RuntimeError(f'{cell} cycle{cycle}: timing unstable for {a.attempts} quartets; all retained')

cells = list(a.bench) + list(a.server)
results = {}
for cycle in range(a.cycles):
    order = cells[:]; random.Random(20260923 + cycle).shuffle(order)
    for cell in order:
        results.setdefault(cell, []).append(quartet(cell, cycle))

summary = []
for cell in cells:
    qs = results[cell]
    ratios = [q['ratio'] for q in qs]
    A = [o['tps'] for q in qs for o in q['obs'] if o['arm'] == 'A']; B = [o['tps'] for q in qs for o in q['obs'] if o['arm'] == 'B']
    row = dict(cell=cell, A_tps=statistics.mean(A), B_tps=statistics.mean(B), speedup_geomean=math.exp(statistics.mean(math.log(r) for r in ratios)),
               speedup_min=min(ratios), speedup_max=max(ratios), cv_A=statistics.stdev(A) / statistics.mean(A), cv_B=statistics.stdev(B) / statistics.mean(B),
               attempts=[q['attempt'] for q in qs], token_mismatch_slots=sum(q.get('token_mismatch_slots', 0) for q in qs))
    if 'requests' in qs[0]['obs'][0]:
        row['A_gen_tps'] = statistics.mean(o['gen_tps_mean'] for q in qs for o in q['obs'] if o['arm'] == 'A')
        row['B_gen_tps'] = statistics.mean(o['gen_tps_mean'] for q in qs for o in q['obs'] if o['arm'] == 'B')
    summary.append(row)
(a.output / 'summary.json').write_text(json.dumps(summary, indent=2))
for r in summary:
    say(f"{r['cell']:7s} A {r['A_tps']:7.2f}  B {r['B_tps']:7.2f}  x{r['speedup_geomean']:.3f} [{r['speedup_min']:.3f}-{r['speedup_max']:.3f}] "
        f"cv {r['cv_A']:.1%}/{r['cv_B']:.1%} attempts {r['attempts']} tokmis {r['token_mismatch_slots']}")
say('done')
