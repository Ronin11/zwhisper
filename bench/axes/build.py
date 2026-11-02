#!/usr/bin/env python3
"""Merge teacher labels by consensus and write indecis datasets for spans of 1, 2, 4 and 8 chunks.
usage: build.py labels_a.jsonl [labels_b.jsonl ...]
Writes data/L{n}_{train,test}.jsonl, chunk_truth.json, and prints teacher agreement."""
import json, sys, os, random, collections
chunks = {c["id"]: c for c in map(json.loads, open("chunks.jsonl"))}
plan = {w["id"]: w for w in json.load(open("plan.json"))}
teachers = [{r["id"]: r for r in map(json.loads, open(p))} for p in sys.argv[1:]]
VAL = {"dark": -1, "neutral": 0, "light": 1}
VNAME = {-1: "dark", 0: "neutral", 1: "light"}

def consensus(cid):
    rs = [t[cid] for t in teachers if cid in t]
    if not rs: return None
    if len(rs) == 1: return dict(rs[0])
    a, b = rs[0], rs[1]
    if a["silence"] != b["silence"] or a["valence"] != b["valence"] or abs(a["energy"] - b["energy"]) > 1: return None
    return {"silence": a["silence"], "energy": min(a["energy"], b["energy"]), "valence": a["valence"]}

if len(teachers) >= 2:
    both = [c for c in chunks if all(c in t for t in teachers)]
    a, b = teachers[0], teachers[1]
    print(f"teacher agreement on {len(both)} chunks: silence {sum(a[c]['silence']==b[c]['silence'] for c in both)/len(both):.2f}  "
          f"energy exact {sum(a[c]['energy']==b[c]['energy'] for c in both)/len(both):.2f}  within1 {sum(abs(a[c]['energy']-b[c]['energy'])<=1 for c in both)/len(both):.2f}  "
          f"MAE {sum(abs(a[c]['energy']-b[c]['energy']) for c in both)/len(both):.2f}  valence {sum(a[c]['valence']==b[c]['valence'] for c in both)/len(both):.2f}")
truth = {c: consensus(c) for c in chunks}
truth = {c: v for c, v in truth.items() if v}
print(f"{len(truth)} of {len(chunks)} chunks labeled by consensus")
print("energy", sorted(collections.Counter(v["energy"] for v in truth.values()).items()),
      "valence", dict(collections.Counter(v["valence"] for v in truth.values())),
      "silence", sum(v["silence"] for v in truth.values()))
json.dump(truth, open("chunk_truth.json", "w"))

# held-out books, the same for every span length
books = sorted({plan[c["window"]]["book"] for c in chunks.values()})
rng = random.Random(3); rng.shuffle(books)
test_books = set(books[:8])
print("test books:", [b.split("/")[4][:30] for b in sorted(test_books)])

def agg(ids):
    vs = [truth[i] for i in ids]
    e = sum(v["energy"] for v in vs) / len(vs)
    val = sum(VAL[v["valence"]] for v in vs) / len(vs)
    return {"silence": sum(v["silence"] for v in vs) * 2 > len(vs), "energy": str(int(e + 0.5)),
            "valence": VNAME[-1 if val < -0.34 else 1 if val > 0.34 else 0]}

by_win = collections.defaultdict(list)
for c in chunks.values(): by_win[c["window"]].append(c)
os.makedirs("data", exist_ok=True)
for L in (1, 2, 4, 8):
    n = {"train": 0, "test": 0}
    with open(f"data/L{L}_train.jsonl", "w") as ftr, open(f"data/L{L}_test.jsonl", "w") as fte:
        for win, cs in by_win.items():
            cs.sort(key=lambda c: c["idx"])
            book = plan[win]["book"]
            is_test = book in test_books
            step = L if (is_test or L == 1) else max(1, L // 2)   # training spans may overlap by half
            for s in range(0, len(cs) - L + 1, step):
                ids = [c["id"] for c in cs[s:s + L]]
                if any(i not in truth for i in ids): continue
                ex = {"text": " ".join(chunks[i]["text"] for i in ids), "labels": agg(ids), "family": book,
                      "meta": {"window": win, "start": str(s), "len": str(L), "ids": ",".join(ids)}}
                (fte if is_test else ftr).write(json.dumps(ex) + "\n"); n["test" if is_test else "train"] += 1
    print(f"L{L}: train {n['train']}  test {n['test']}")
