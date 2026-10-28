#!/usr/bin/env python3
"""Score `zwhisper scenes` outputs against bench/eval/labels.json.
usage: score.py <label> <out_dir_glob_prefix>   e.g. score.py qwen1.5b bench/eval/out_1.5b_
For every reference scene, the predicted scene active at its midpoint is compared."""
import json, re, sys, glob, os
labels = json.load(open(os.path.join(os.path.dirname(__file__), "labels.json")))
tag, prefix = sys.argv[1], sys.argv[2]
tot = dict(n=0, music=0, mood=0, inten=0.0, bnd_hit=0, bnd_ref=0, bnd_pred=0, nscenes=0, win=0, secs=0.0)
rows = []
for name, refs in labels.items():
    path = f"{prefix}{name}.txt"
    if not os.path.exists(path):
        continue
    preds, cur = [], None
    secs = 0.0
    for line in open(path):
        m = re.match(r"\s*(\d+)s\s+(\w+)\s+(\d)\s+(.*)", line)
        if m:
            cur = dict(start=int(m[1]), music=m[2], intensity=int(m[3]), summary=m[4].strip(), mood="")
            preds.append(cur)
        elif line.strip().startswith("mood:") and cur:
            cur["mood"] = line.split("mood:", 1)[1].strip()
        elif "scenes in" in line:
            mm = re.search(r"in ([\d.]+) s", line)
            if mm:
                secs = float(mm[1])
    if not preds:
        continue
    preds.sort(key=lambda p: p["start"])
    tot["win"] += 1; tot["nscenes"] += len(preds); tot["secs"] += secs
    ends = [r["start"] for r in refs[1:]] + [300]
    for r, end in zip(refs, ends):
        mid = (r["start"] + end) / 2
        active = [p for p in preds if p["start"] <= mid]
        p = active[-1] if active else preds[0]
        tot["n"] += 1
        tot["music"] += p["music"] == r["music"]
        tot["mood"] += (p["mood"] == r["mood"]) if r["music"] == "score" else (p["music"] == r["music"])
        tot["inten"] += abs(p["intensity"] - r["intensity"])
        rows.append((name, r["start"], r["music"], r["mood"][:22], r["intensity"], p["music"], p["mood"][:22], p["intensity"]))
    # boundaries (excluding the window start)
    rb = [r["start"] for r in refs[1:]]
    pb = [p["start"] for p in preds[1:]]
    tot["bnd_ref"] += len(rb); tot["bnd_pred"] += len(pb)
    tot["bnd_hit"] += sum(1 for b in rb if any(abs(b - q) <= 30 for q in pb))
n = max(tot["n"], 1)
prec = tot["bnd_hit"] / max(tot["bnd_pred"], 1); rec = tot["bnd_hit"] / max(tot["bnd_ref"], 1)
print(f"{tag:10s} windows {tot['win']}  scenes/window {tot['nscenes']/max(tot['win'],1):.1f}  "
      f"music acc {tot['music']/n:.2f}  mood acc {tot['mood']/n:.2f}  intensity MAE {tot['inten']/n:.2f}  "
      f"boundary P {prec:.2f} R {rec:.2f}  {tot['secs']/max(tot['win'],1):.0f} s/window")
if "-v" in sys.argv:
    for r in rows:
        print("   %-10s %4ds ref %-7s %-22s %d | got %-7s %-22s %d" % r)
