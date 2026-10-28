#!/usr/bin/env python3
"""Train linear probes on frozen window features and score them on the evaluation set.

usage: probe.py <features.txt> [--labels=bench/train/labels.json] [--eval-labels=bench/eval/labels.json]
features.txt: one line per window, "<path> f0 f1 ...", from `zwhisper features` or `zwhisper embed`.
Each window's target is its dominant scene (the one covering the most time)."""
import json, sys, os, re
import numpy as np

MOODS = ["warm solo piano, slow, tender", "driving percussion, fast, urgent", "tense low strings, slow pulse, ominous",
         "soft ambient pad, still, curious", "bright acoustic guitar, light, hopeful", "dark drones, sparse, dread",
         "playful pizzicato, quick, mischievous", "epic brass and drums, big, triumphant", "gentle harp and flute, calm, wonder",
         "pulsing synth, steady, focused", "melancholy cello, slow, grief", "shimmering strings, rising, awe",
         "quiet guitar, unhurried, nostalgic", "rhythmic marching drums, determined", "eerie high strings, uneasy, mystery",
         "lively folk fiddle, upbeat, festive"]
FAMILY = {"warm solo piano, slow, tender": "calm", "quiet guitar, unhurried, nostalgic": "calm", "gentle harp and flute, calm, wonder": "calm",
          "soft ambient pad, still, curious": "calm", "bright acoustic guitar, light, hopeful": "bright", "playful pizzicato, quick, mischievous": "bright",
          "lively folk fiddle, upbeat, festive": "bright", "tense low strings, slow pulse, ominous": "tense", "dark drones, sparse, dread": "tense",
          "eerie high strings, uneasy, mystery": "tense", "driving percussion, fast, urgent": "action", "rhythmic marching drums, determined": "action",
          "epic brass and drums, big, triumphant": "action", "pulsing synth, steady, focused": "focus", "melancholy cello, slow, grief": "sad",
          "shimmering strings, rising, awe": "awe"}
FAMILIES = sorted(set(FAMILY.values()))

def arg(name, default):
    for a in sys.argv[2:]:
        if a.startswith(f"--{name}="): return a.split("=", 1)[1]
    return default

feat_path = sys.argv[1]
train_labels = json.load(open(arg("labels", "bench/train/labels.json")))
eval_labels = json.load(open(arg("eval-labels", "bench/eval/labels.json")))

feats = {}
for line in open(feat_path):
    parts = line.split()
    if len(parts) < 10: continue
    name = os.path.basename(parts[0]).replace(".jsonl", "")
    feats[name] = np.array([float(x) for x in parts[1:]], dtype=np.float64)

def dominant(scenes, length=300):
    best, best_len = None, -1
    for i, s in enumerate(scenes):
        end = scenes[i + 1]["start"] if i + 1 < len(scenes) else length
        if end - s["start"] > best_len:
            best, best_len = s, end - s["start"]
    return best

def targets(labels):
    X, fam, mood, inten, music, names = [], [], [], [], [], []
    for name, scenes in labels.items():
        if name not in feats or not scenes: continue
        d = dominant(scenes)
        X.append(feats[name]); names.append(name)
        music.append(["score", "silence", "song"].index(d["music"]))
        m = d["mood"] if d["music"] == "score" and d["mood"] in MOODS else None
        mood.append(MOODS.index(m) if m else -1)
        fam.append(FAMILIES.index(FAMILY[m]) if m else -1)
        inten.append(d["intensity"])
    return np.array(X), np.array(fam), np.array(mood), np.array(inten, dtype=np.float64), np.array(music), names

Xtr, ftr, mtr, itr, mutr, _ = targets(train_labels)
Xev, fev, mev, iev, muev, ev_names = targets(eval_labels)
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
Z = lambda X: (X - mu) / sd

def softmax_fit(X, y, k, l2=1.0, steps=600, lr=0.05):
    keep = y >= 0
    X, y = X[keep], y[keep]
    n, d = X.shape
    W = np.zeros((d, k)); b = np.zeros(k)
    Y = np.eye(k)[y]
    for _ in range(steps):
        logits = X @ W + b
        logits -= logits.max(1, keepdims=True)
        P = np.exp(logits); P /= P.sum(1, keepdims=True)
        G = P - Y
        W -= lr * (X.T @ G / n + l2 * W / n)
        b -= lr * G.mean(0)
    return W, b

def softmax_predict(X, W, b):
    return np.argmax(X @ W + b, 1)

def ridge_fit(X, y, l2=10.0):
    d = X.shape[1]
    A = X.T @ X + l2 * np.eye(d)
    w = np.linalg.solve(A, X.T @ (y - y.mean()))
    return w, y.mean()

def report(tag, Xtr, Xev):
    Wf, bf = softmax_fit(Xtr, ftr, len(FAMILIES))
    Wm, bm = softmax_fit(Xtr, mtr, len(MOODS))
    Wmu, bmu = softmax_fit(Xtr, mutr, 3)
    wi, bi = ridge_fit(Xtr, itr)
    pf = softmax_predict(Xev, Wf, bf); pm = softmax_predict(Xev, Wm, bm); pmu = softmax_predict(Xev, Wmu, bmu)
    pi = np.clip(np.rint(Xev @ wi + bi), 1, 5)
    keepf = fev >= 0
    fam_acc = (pf[keepf] == fev[keepf]).mean() if keepf.any() else float("nan")
    mood_acc = (pm[keepf] == mev[keepf]).mean() if keepf.any() else float("nan")
    print(f"{tag:14s} n_train {len(Xtr)}  eval {len(Xev)}:  family acc {fam_acc:.2f}  mood acc {mood_acc:.2f}  "
          f"intensity MAE {np.abs(pi - iev).mean():.2f}  music acc {(pmu == muev).mean():.2f}")
    return pf, pm, pi

# majority baseline
maj_f = np.bincount(ftr[ftr >= 0]).argmax(); maj_m = np.bincount(mtr[mtr >= 0]).argmax()
keepf = fev >= 0
print(f"{'majority':14s} family acc {(fev[keepf] == maj_f).mean():.2f}  mood acc {(mev[keepf] == maj_m).mean():.2f}  "
      f"intensity MAE {np.abs(np.rint(itr.mean()) - iev).mean():.2f}")
pf, pm, pi = report("probe", Z(Xtr), Z(Xev))
for name, f, m, i, pf1, pm1, pi1 in zip(ev_names, fev, mev, iev, pf, pm, pi):
    print(f"   {name:10s} ref {FAMILIES[f] if f >= 0 else 'none':7s} {MOODS[m][:22] if m >= 0 else '':22s} {int(i)} | got {FAMILIES[pf1]:7s} {MOODS[pm1][:22]:22s} {int(pi1)}")

# 5-fold cross-validation on the training set against its own labels
rng = np.random.default_rng(0)
idx = rng.permutation(len(Xtr)); folds = np.array_split(idx, 5)
accs = []
for k in range(5):
    te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
    mu2, sd2 = Xtr[tr].mean(0), Xtr[tr].std(0) + 1e-6
    Wf, bf = softmax_fit((Xtr[tr] - mu2) / sd2, ftr[tr], len(FAMILIES))
    p = softmax_predict((Xtr[te] - mu2) / sd2, Wf, bf)
    keep = ftr[te] >= 0
    accs.append((p[keep] == ftr[te][keep]).mean())
print(f"5-fold CV family acc on training labels: {np.mean(accs):.2f} (majority {np.mean(ftr[ftr>=0] == maj_f):.2f})")
