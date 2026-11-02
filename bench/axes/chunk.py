#!/usr/bin/env python3
"""Cut each transcript into chunks of about CHUNK seconds at segment boundaries.
Writes chunks.jsonl: {"id","window","idx","t0","t1","words","text"}."""
import json, glob, os
CHUNK = 60.0
out = open("chunks.jsonl", "w")
n = 0
for path in sorted(glob.glob("tx/x*.jsonl")):
    win = os.path.basename(path)[:-6]
    segs = [json.loads(l) for l in open(path) if l.strip()]
    cur, t0 = [], None
    def flush(t1):
        global n
        text = " ".join(s["text"].strip() for s in cur)
        if not text: return
        out.write(json.dumps({"id": f"{win}c{len(done):02d}", "window": win, "idx": len(done), "t0": t0 / 100, "t1": t1 / 100,
                              "words": len(text.split()), "text": text}) + "\n")
        done.append(1); n += 1
    done = []
    for s in segs:
        if not cur: t0 = s["t0"]
        cur.append(s)
        if (s["t1"] - t0) / 100 >= CHUNK:
            flush(s["t1"]); cur = []
    if cur and (cur[-1]["t1"] - t0) / 100 >= CHUNK * 0.5: flush(cur[-1]["t1"])
print(n, "chunks")
