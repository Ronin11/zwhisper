#!/usr/bin/env python3
"""Cut labeled windows into one transcript snippet per scene, so every training example's
text is exactly the stretch its label describes.
usage: snippets.py <labels.json> <transcript_dir> <out_dir> <out_labels.json>"""
import json, sys, os
labels, tdir, odir, out_labels = sys.argv[1:5]
labels = json.load(open(labels))
os.makedirs(odir, exist_ok=True)
out = {}
for name, scenes in labels.items():
    path = os.path.join(tdir, name + ".jsonl")
    if not os.path.exists(path): continue
    segs = [json.loads(l) for l in open(path) if l.strip()]
    for k, s in enumerate(scenes):
        a = s["start"]; b = scenes[k + 1]["start"] if k + 1 < len(scenes) else 300
        lines = [g for g in segs if a <= g["t0"] / 100 < b]
        if sum(len(g["text"].split()) for g in lines) < 25: continue
        sid = f"{name}_{k}"
        with open(os.path.join(odir, sid + ".jsonl"), "w") as f:
            for g in lines: f.write(json.dumps(g, separators=(",", ":")) + "\n")
        out[sid] = [dict(s, start=0)]
json.dump(out, open(out_labels, "w"), indent=1)
print(len(out), "snippets from", len(labels), "windows")
