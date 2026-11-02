#!/usr/bin/env python3
"""Bag-of-words baseline on the same consensus split: log-count unigram+bigram features,
ridge for energy (rounded), one-vs-rest ridge for valence. No pretrained model."""
import json, re, collections, numpy as np
tr = [json.loads(l) for l in open("data/L1_train.jsonl")]; te = [json.loads(l) for l in open("data/L1_test.jsonl")]
VAL = {"dark": -1, "neutral": 0, "light": 1}
def toks(t):
    w = re.findall(r"[a-z']+", t.lower()); return w + [a + "_" + b for a, b in zip(w, w[1:])]
df = collections.Counter()
for e in tr: df.update(set(toks(e["text"])))
vocab = {w: i for i, w in enumerate(w for w, n in df.items() if n >= 3)}
def feats(exs):
    X = np.zeros((len(exs), len(vocab) + 1))
    for r, e in enumerate(exs):
        c = collections.Counter(toks(e["text"]))
        for w, n in c.items():
            if w in vocab: X[r, vocab[w]] = np.log1p(n) * np.log(len(tr) / df[w])
        X[r, -1] = 1.0
    X[:, :-1] /= (np.linalg.norm(X[:, :-1], axis=1, keepdims=True) + 1e-9)
    return X
Xtr, Xte = feats(tr), feats(te)
print(f"vocab {len(vocab)}  train {len(tr)}  test {len(te)}")
def ridge(X, y, l2):  # dual form: n examples << features
    return X.T @ np.linalg.solve(X @ X.T + l2 * np.eye(X.shape[0]), y)
ytr_e = np.array([int(e["labels"]["energy"]) for e in tr]); yte_e = np.array([int(e["labels"]["energy"]) for e in te])
ytr_v = np.array([VAL[e["labels"]["valence"]] for e in tr]); yte_v = np.array([VAL[e["labels"]["valence"]] for e in te])
maj_e = collections.Counter(ytr_e).most_common(1)[0][0]; maj_v = collections.Counter(ytr_v).most_common(1)[0][0]
print(f"majority         energy acc {np.mean(yte_e == maj_e):.2f} MAE {np.mean(np.abs(yte_e - maj_e)):.2f}   valence acc {np.mean(yte_v == maj_v):.2f}")
for l2 in (0.3, 1.0, 3.0):
    pe = np.clip(np.rint(Xte @ ridge(Xtr, ytr_e, l2)), 1, 5)
    # valence: one-vs-rest scores, argmax
    S = np.stack([Xte @ ridge(Xtr, (ytr_v == k).astype(float), l2) for k in (-1, 0, 1)], 1)
    pv = np.array([-1, 0, 1])[S.argmax(1)]
    nn = yte_v != 0
    print(f"bow ridge l2={l2:<4} energy acc {np.mean(pe == yte_e):.2f} MAE {np.mean(np.abs(pe - yte_e)):.2f}   valence acc {np.mean(pv == yte_v):.2f}  dark/light {np.mean(pv[nn] == yte_v[nn]):.2f}  wrong-sign {int(np.sum(pv * yte_v < 0))}")
# what words carry each axis
w_e = ridge(Xtr, ytr_e, 1.0)[:-1]; w_v = ridge(Xtr, ytr_v, 1.0)[:-1]
inv = {i: w for w, i in vocab.items()}
top = lambda w, s: ", ".join(inv[i] for i in (np.argsort(w) if s < 0 else np.argsort(-w))[:12])
print("energy up:  ", top(w_e, +1)); print("energy down:", top(w_e, -1))
print("light:      ", top(w_v, +1)); print("dark:       ", top(w_v, -1))
