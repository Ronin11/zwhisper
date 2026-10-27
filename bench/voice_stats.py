#!/usr/bin/env python3
"""Summarize voice tags from zwhisper --json output: voices by count, pitch, quoted share."""
import json, sys, collections
segs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = collections.defaultdict(list)
for s in segs: by[s["voice"]].append(s)
print(f"{len(segs)} spans, {len(by)} voices")
for v, ss in sorted(by.items(), key=lambda kv: -len(kv[1])):
    p = [s["pitch"] for s in ss if s["pitch"] > 0]
    q = sum(1 for s in ss if s["quoted"])
    dur = sum(s["t1"] - s["t0"] for s in ss) / 100
    med = sorted(p)[len(p)//2] if p else 0
    print(f"  v{v:<3} spans {len(ss):3d}  {dur:6.1f} s  quoted {q:3d}  pitch median {med:4.0f} Hz  e.g. {ss[0]['text'].strip()[:60]!r}")
