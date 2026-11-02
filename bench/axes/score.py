#!/usr/bin/env python3
"""Score a model at chunk resolution: predictions on spans are broadcast to their chunks and
compared with the chunk-level consensus labels. usage: score.py <model_dir> <test.jsonl> [--smooth]"""
import json, sys, subprocess, collections, statistics
model, test = sys.argv[1], sys.argv[2]
truth = json.load(open("chunk_truth.json"))
VAL = {"dark": -1, "neutral": 0, "light": 1}; VNAME = {-1: "dark", 0: "neutral", 1: "light"}
exs = [json.loads(l) for l in open(test)]
raw = subprocess.run([sys.argv[0].replace("score.py", "") + "../../../indecis/bin/indecis", "predict", "-model", model, "-in", test],
                     capture_output=True, text=True, check=True).stdout
preds = [json.loads(l)["answers"] for l in raw.splitlines() if l.startswith("{")]
assert len(preds) == len(exs), (len(preds), len(exs))

def metrics(rows):
    """rows: (truth, pred) per chunk, each {"silence","energy","valence"}."""
    n = len(rows)
    return {"n": n,
            "sil_acc": sum(t["silence"] == p["silence"] for t, p in rows) / n,
            "en_acc": sum(t["energy"] == p["energy"] for t, p in rows) / n,
            "en_mae": sum(abs(t["energy"] - p["energy"]) for t, p in rows) / n,
            "en_w1": sum(abs(t["energy"] - p["energy"]) <= 1 for t, p in rows) / n,
            "val_acc": sum(t["valence"] == p["valence"] for t, p in rows) / n,
            "dl_acc": sum((t["valence"] == p["valence"]) for t, p in rows if t["valence"] != "neutral") / max(1, sum(t["valence"] != "neutral" for t, _ in rows))}

def fmt(tag, m):
    return (f"{tag:22s} n={m['n']:4d}  silence {m['sil_acc']:.2f}  energy acc {m['en_acc']:.2f} MAE {m['en_mae']:.2f} ±1 {m['en_w1']:.2f}"
            f"  valence {m['val_acc']:.2f}  dark/light {m['dl_acc']:.2f}")

# per-chunk predictions, in window order
by_win = collections.defaultdict(dict)
span_rows = []
for ex, p in zip(exs, preds):
    pred = {"silence": p["silence"]["p"] > 0.5, "energy": int(p["energy"]["choice"]), "valence": p["valence"]["choice"],
            "en_score": p["energy"]["score"] + 1, "val_score": p["valence"].get("score", VAL[p["valence"]["choice"]] + 1) - 1}
    lab = ex["labels"]
    span_rows.append(({"silence": lab["silence"], "energy": int(lab["energy"]), "valence": lab["valence"]}, pred))
    for cid in ex["meta"]["ids"].split(","):
        by_win[ex["meta"]["window"]][cid] = pred
chunk_rows, ceiling_rows, maj_rows = [], [], []
maj = {"silence": False, "energy": statistics.mode(v["energy"] for v in truth.values()), "valence": statistics.mode(v["valence"] for v in truth.values())}
for ex, p in zip(exs, preds):
    lab = {"silence": ex["labels"]["silence"], "energy": int(ex["labels"]["energy"]), "valence": ex["labels"]["valence"]}
    for cid in ex["meta"]["ids"].split(","):
        chunk_rows.append((truth[cid], by_win[ex["meta"]["window"]][cid]))
        ceiling_rows.append((truth[cid], lab))
        maj_rows.append((truth[cid], maj))
print(fmt("majority", metrics(maj_rows)))
print(fmt("span-label ceiling", metrics(ceiling_rows)))
print(fmt("model @span", metrics(span_rows)))
print(fmt("model @chunk", metrics(chunk_rows)))
if "--smooth" in sys.argv:
    for k in (3, 5):
        rows = []
        for win, cs in by_win.items():
            ids = sorted(cs, key=lambda i: int(i.split("c")[1]))
            for j, cid in enumerate(ids):
                lo, hi = max(0, j - k // 2), min(len(ids), j + k // 2 + 1)
                nb = [cs[i] for i in ids[lo:hi]]
                sm = {"silence": sum(x["silence"] for x in nb) * 2 > len(nb),
                      "energy": int(statistics.median(x["en_score"] for x in nb) + 0.5),
                      "valence": VNAME[round(statistics.median(x["val_score"] for x in nb))]}
                rows.append((truth[cid], sm))
        print(fmt(f"model @chunk median{k}", metrics(rows)))
