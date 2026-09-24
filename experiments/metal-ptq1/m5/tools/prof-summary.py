#!/usr/bin/env python3
import json, sys, collections
lines=[json.loads(l) for l in open(sys.argv[1])]
sel = int(sys.argv[2]) if len(sys.argv)>2 else -1
g=lines[sel]
tot=sum(o[8] for o in g['ops'])
agg=collections.defaultdict(lambda:[0,0.0])
for op,res,t0,t1,k,m,k1,n1,us,name in g['ops']:
    key=f"{op}{'(+'+str(res-1)+')' if res>1 else ''} {t0} {k}x{m} n={n1}" if op in('MUL_MAT','MUL_MAT_ID') else f"{op}{'(+'+str(res-1)+')' if res>1 else ''} {t0}"
    agg[key][0]+=1; agg[key][1]+=us
print(f"graphs={len(lines)} using #{sel}: nodes={g['n_nodes']} ops={len(g['ops'])} sum_gpu={tot/1000:.2f} ms")
for k,(c,us) in sorted(agg.items(), key=lambda x:-x[1][1])[:int(sys.argv[3]) if len(sys.argv)>3 else 40]:
    print(f"{us/1000:8.3f} ms {100*us/tot:5.1f}% {c:5d}x {us/c:8.2f} us  {k}")
