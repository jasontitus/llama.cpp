#!/usr/bin/env python3
"""Print one device's column of the "Results by device" table from run-device-study.sh output.

usage: device-table.py <out dir>
Each value is the paired speedup (geometric mean of 3 quartets) with A -> B tokens/s; "n/a" when the
study was skipped. Replace the device's "pending" cells in README.md with these values.
"""
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])

# README row label -> (study directory, cell)
ROWS = [
    ("PTQ1: upstream plain -> flags + MTP (server, 1 request)", "ptq1-total", "s1c1"),
    ("PQ2: upstream plain -> flags + MTP (server, 1 request)", "pq2-total", "s1c1"),
    ("PTQ1 plain decoding (server, 1 request)", "ptq1-same-mode", "s0c1"),
    ("PTQ1 plain decoding, bit-identical subset (server, 1 request)", "ptq1-bitexact", "s0c1"),
    ("PTQ1 tg128", "ptq1-same-mode", "tg128"),
    ("PTQ1 MTP -> MTP (server, 1 request)", "ptq1-same-mode", "s1c1"),
    ("PTQ1 2 requests (server)", "ptq1-same-mode", "s0c2"),
    ("PTQ1 pp2 / pp4 / pp8", "ptq1-same-mode", ["pp2", "pp4", "pp8"]),
    ("PQ2 tg128", "pq2-same-mode", "tg128"),
    ("PQ2 2 requests (server)", "pq2-same-mode", "s0c2"),
    ("Bonsai 1 ternary tg128", "b1-ternary", "tg128"),
    ("Bonsai 1 ternary 2 requests (server)", "b1-ternary", "s0c2"),
    ("Bonsai 1 binary tg128", "b1-binary", "tg128"),
    ("Bonsai 1 binary 2 requests (server)", "b1-binary", "s0c2"),
]


def load(study):
    p = out / study / "summary.json"
    return {r["cell"]: r for r in json.loads(p.read_text())} if p.exists() else {}


def fmt(r, short=False):
    if r is None:
        return "n/a"
    s = f"{r['speedup_geomean']:.2f}x"
    if not short:
        s += f" ({r['A_tps']:.1f} -> {r['B_tps']:.1f})"
    if r.get("token_mismatch_slots"):
        s += " tokens differ"
    return s


meta = next((json.loads(p.read_text()) for p in out.glob("*/meta.json")), {})
print(f"source revision: {meta.get('revision', '?')}  diff sha256: {meta.get('diff_sha256', '?')[:12]}")
for label, study, cell in ROWS:
    s = load(study)
    if isinstance(cell, list):
        vals = " / ".join(fmt(s.get(c), short=True) for c in cell)
    else:
        vals = fmt(s.get(cell))
    print(f"| {label} | {vals} |")
