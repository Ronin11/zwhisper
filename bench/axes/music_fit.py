#!/usr/bin/env python3
"""Fit DSP features -> CLAP-anchored energy/valence, 5-fold CV. Prints correlations and weights."""
import json, numpy as np
F = json.load(open("song_feats.json")); names = F["names"]
axes = {s["id"]: s for s in json.load(open("song_axes.json"))}
X = np.array([s["feats"] for s in F["songs"]]); ids = [s["id"] for s in F["songs"]]
Y = {k: np.array([axes[i][k] for i in ids]) for k in ("energy_raw", "valence_raw", "voice_raw")}
X = np.nan_to_num(X)
mu, sd = X.mean(0), X.std(0) + 1e-9
Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
rng = np.random.default_rng(0); idx = rng.permutation(len(X)); folds = np.array_split(idx, 5)
def ridge(A, y, l2): return np.linalg.solve(A.T @ A + l2 * np.eye(A.shape[1]), A.T @ y)
print(f"{len(X)} songs, {X.shape[1]} features")
for target, y in Y.items():
    for l2 in (1.0, 10.0, 50.0):
        pred = np.zeros(len(y))
        for k in range(5):
            te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
            pred[te] = Z[te] @ ridge(Z[tr], y[tr], l2)
        r = np.corrcoef(pred, y)[0, 1]
        print(f"{target:12s} l2={l2:<5} CV r = {r:.3f}   (rank corr {np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(y)))[0,1]:.3f})")
    w = ridge(Z, y, 10.0)
    order = np.argsort(-np.abs(w[:-1]))[:8]
    print("   top weights:", ", ".join(f"{names[i]} {w[i]:+.3f}" for i in order))
# single-feature correlations, for intuition
print("\nsingle-feature |r| with energy / valence:")
for i in np.argsort(-np.abs([np.corrcoef(X[:, i], Y['energy_raw'])[0, 1] for i in range(X.shape[1])]))[:8]:
    print(f"   {names[i]:18s} energy {np.corrcoef(X[:, i], Y['energy_raw'])[0,1]:+.2f}  valence {np.corrcoef(X[:, i], Y['valence_raw'])[0,1]:+.2f}")
