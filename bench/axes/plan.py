#!/usr/bin/env python3
"""Pick long windows from the books of bench/train/plan.json for the axis experiment.
Each window is WIN seconds, never in the first or last SKIP seconds of the book."""
import json, random, subprocess, sys
WIN, SKIP, PER_BOOK = 15 * 60, 6 * 60, 4
books = sorted({x["book"] for x in json.load(open("../train/plan.json"))})
rng = random.Random(7)
plan = []
for b in books:
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", b],
                               capture_output=True, text=True).stdout.strip() or 0)
    span = dur - 2 * SKIP - WIN
    if span <= 0:
        print("skip (short)", b, file=sys.stderr); continue
    # spread the windows over the book, with jitter, so a long book gets early and late scenes
    for k in range(PER_BOOK):
        lo = SKIP + span * k / PER_BOOK
        plan.append({"book": b, "offset": int(lo + rng.random() * span / PER_BOOK), "len": WIN})
for i, w in enumerate(plan): w["id"] = f"x{i:03d}"
json.dump(plan, open("plan.json", "w"), indent=0)
print(len(plan), "windows from", len(books), "books")
