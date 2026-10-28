#!/usr/bin/env python3
"""How consistent is the 30B labeler with itself? Re-label each scene snippet on its own
and compare to the label it gave that scene inside its window."""
import json, sys, os, random, urllib.request
sys.path.insert(0, 'bench/eval')
import importlib.util
spec = importlib.util.spec_from_file_location("lab", "bench/eval/label_ollama.py")
# reuse PROMPT/SCHEMA/MOODS/ask/lines_of without running its main
src = open("bench/eval/label_ollama.py").read().split("out_path, items =")[0]
ns = {}; exec(src, ns)
FAMILY = json.load(open("bench/train/family.json"))
labels = json.load(open("bench/train/snip_labels.json"))
names = [n for n in labels if labels[n][0]["music"] == "score"]
random.seed(1); random.shuffle(names); names = names[:int(sys.argv[1]) if len(sys.argv) > 1 else 120]
same_mood = same_fam = same_music = n = 0; inten = 0
for nm in names:
    ref = labels[nm][0]
    scenes = ns["ask"](ns["lines_of"](f"bench/train/snips/{nm}.jsonl"))
    if not scenes: continue
    # the scene covering the most of the snippet
    scenes.sort(key=lambda s: s["start"]); best = max(range(len(scenes)), key=lambda i: (scenes[i+1]["start"] if i+1 < len(scenes) else 300) - scenes[i]["start"])
    got = scenes[best]
    if got["music"] == "song" and not (got.get("song_title") or "").strip(): got["music"] = "score"
    n += 1
    same_music += got["music"] == ref["music"]
    if got["music"] == "score":
        same_mood += got.get("mood") == ref["mood"]
        same_fam += FAMILY.get(got.get("mood"), "?") == FAMILY.get(ref["mood"], "!")
    inten += abs(int(got["intensity"]) - int(ref["intensity"]))
    print(f"{nm:14s} ref {ref['mood'][:24]:24s} {ref['intensity']} | again {got.get('mood','')[:24]:24s} {got['intensity']} {got['music']}", flush=True)
print(f"\n30B self-agreement on {n} scenes: music {same_music/n:.2f}  mood {same_mood/n:.2f}  family {same_fam/n:.2f}  intensity MAE {inten/n:.2f}")
