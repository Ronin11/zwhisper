#!/usr/bin/env python3
"""Side-by-side segments from two JSONL files (ours vs reference) plus token agreement."""
import json, sys, difflib

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]

ours, ref = load(sys.argv[1]), load(sys.argv[2])
print(f"segments: ours {len(ours)}  ref {len(ref)}")
for i in range(max(len(ours), len(ref))):
    a = ours[i] if i < len(ours) else None
    b = ref[i] if i < len(ref) else None
    fa = f"[{a['t0']:5d}-{a['t1']:5d}] {a['text']}" if a else "-"
    fb = f"[{b['t0']:5d}-{b['t1']:5d}] {b['text']}" if b else "-"
    mark = "  " if a and b and a["text"] == b["text"] and a["t0"] == b["t0"] and a["t1"] == b["t1"] else "!!"
    print(f"{mark} ours {fa}\n   ref  {fb}")
ta = [t for s in ours for t in s["tokens"]]
tb = [t for s in ref for t in s["tokens"]]
sm = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
print(f"token stream: ours {len(ta)} ref {len(tb)} similarity {sm.ratio():.4f}")
wa = " ".join(s["text"] for s in ours).split()
wb = " ".join(s["text"] for s in ref).split()
sm = difflib.SequenceMatcher(a=wa, b=wb, autojunk=False)
print(f"words: ours {len(wa)} ref {len(wb)} similarity {sm.ratio():.4f}")
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag != "equal":
        print(f"  {tag}: ours {' '.join(wa[i1:i2])!r} vs ref {' '.join(wb[j1:j2])!r}")
