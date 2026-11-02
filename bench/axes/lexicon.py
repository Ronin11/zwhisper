#!/usr/bin/env python3
"""NRC VAD lexicon baseline on the consensus split. usage: lexicon.py <NRC-VAD-Lexicon.txt>
Per chunk: mean valence/arousal/dominance over covered words plus a few tail statistics.
(a) unsupervised: quantile thresholds on the mean alone; (b) ridge on the ~12 lexicon features."""
import sys, json, re, collections, numpy as np
lex = {}
for line in open(sys.argv[1], encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) == 4:
        try: lex[p[0]] = (float(p[1]), float(p[2]), float(p[3]))
        except ValueError: pass
print(f"lexicon {len(lex)} words")
tr = [json.loads(l) for l in open("data/L1_train.jsonl")]; te = [json.loads(l) for l in open("data/L1_test.jsonl")]
VAL = {"dark": -1, "neutral": 0, "light": 1}
def feats(text):
    words = re.findall(r"[a-z']+", text.lower())
    hits = np.array([lex[w] for w in words if w in lex])
    if len(hits) == 0: return np.zeros(13)
    v, a, d = hits[:, 0], hits[:, 1], hits[:, 2]
    return np.array([v.mean(), a.mean(), d.mean(), v.std(), a.std(),
                     np.mean(v < 0.3), np.mean(v > 0.7), np.mean(a > 0.7), np.mean(a < 0.3),
                     np.mean(np.abs(v - 0.5)), len(hits) / len(words), np.percentile(a, 90), np.percentile(v, 10)])
Xtr = np.array([feats(e["text"]) for e in tr]); Xte = np.array([feats(e["text"]) for e in te])
ytr_e = np.array([int(e["labels"]["energy"]) for e in tr]); yte_e = np.array([int(e["labels"]["energy"]) for e in te])
ytr_v = np.array([VAL[e["labels"]["valence"]] for e in tr]); yte_v = np.array([VAL[e["labels"]["valence"]] for e in te])
print(f"coverage: {Xtr[:, 10].mean():.2f} of words are in the lexicon")
def corr(x, y): return np.corrcoef(x, y)[0, 1]
print(f"correlation with labels (train): mean arousal~energy {corr(Xtr[:,1], ytr_e):+.2f}  p90 arousal~energy {corr(Xtr[:,11], ytr_e):+.2f}  "
      f"frac high-arousal~energy {corr(Xtr[:,7], ytr_e):+.2f}\n"
      f"                                 mean valence~valence {corr(Xtr[:,0], ytr_v):+.2f}  frac low-valence~valence {corr(Xtr[:,5], ytr_v):+.2f}  p10 valence~valence {corr(Xtr[:,12], ytr_v):+.2f}")
def report(tag, pe, pv):
    nn = yte_v != 0
    print(f"{tag:26s} energy acc {np.mean(pe == yte_e):.2f} MAE {np.mean(np.abs(pe - yte_e)):.2f}   valence acc {np.mean(pv == yte_v):.2f}  "
          f"dark/light {np.mean(pv[nn] == yte_v[nn]):.2f}  wrong-sign {int(np.sum(pv * yte_v < 0))}")
maj_e = collections.Counter(ytr_e).most_common(1)[0][0]; maj_v = collections.Counter(ytr_v).most_common(1)[0][0]
report("majority", np.full(len(te), maj_e), np.full(len(te), maj_v))
# (a) one feature, thresholds at the training label quantiles
def quant(x_tr, y_tr, x_te, levels):
    cuts = [np.quantile(x_tr, np.mean(y_tr < lv)) for lv in levels[1:]]
    return np.array(levels)[np.searchsorted(cuts, x_te)]
report("lexicon: mean A / mean V", quant(Xtr[:,1], ytr_e, Xte[:,1], [1,2,3,4,5]), quant(Xtr[:,0], ytr_v, Xte[:,0], [-1,0,1]))
report("lexicon: p90 A / p10 V", quant(Xtr[:,11], ytr_e, Xte[:,11], [1,2,3,4,5]), quant(Xtr[:,12], ytr_v, Xte[:,12], [-1,0,1]))
# (b) ridge on all lexicon features
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
Z = lambda X: np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
def ridge(X, y, l2=3.0): return np.linalg.solve(X.T @ X + l2 * np.eye(X.shape[1]), X.T @ y)
pe = np.clip(np.rint(Z(Xte) @ ridge(Z(Xtr), ytr_e)), 1, 5)
S = np.stack([Z(Xte) @ ridge(Z(Xtr), (ytr_v == k).astype(float)) for k in (-1, 0, 1)], 1)
report("lexicon: ridge 13 feats", pe, np.array([-1, 0, 1])[S.argmax(1)])
# which teacher does the lexicon side with when they disagree on valence?
t1 = {r["id"]: r for r in map(json.loads, open("labels_qwen3_30b-a3b.jsonl"))}
t2 = {r["id"]: r for r in map(json.loads, open("labels_qwen36_35b-a3b.jsonl"))}
tx = {c["id"]: c["text"] for c in map(json.loads, open("chunks.jsonl"))}
dis = [i for i in t1 if i in t2 and t1[i]["valence"] != t2[i]["valence"] and "neutral" not in (t1[i]["valence"], t2[i]["valence"])]
side = collections.Counter()
for i in dis:
    v = feats(tx[i])[0]
    side["t1" if (v - 0.5) * VAL[t1[i]["valence"]] > 0 else "t2"] += 1
print(f"dark-vs-light teacher disagreements: {len(dis)}; lexicon sides with teacher1 {side['t1']}, teacher2 {side['t2']}")
