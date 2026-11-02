#!/usr/bin/env python3
"""Cheap DSP features per song (the ones a phone can compute from decoded PCM), from three
10-second windows at a quarter, half and three quarters, 16 kHz mono. Writes song_feats.json."""
import json, os, subprocess, sys, numpy as np
from concurrent.futures import ThreadPoolExecutor
RATE, WIN, NFFT, HOP = 16000, 10.0, 1024, 160
ROOTS = ["/mnt/nas-media/Music", "/mnt/nas-media/Music/Purchased", "/mnt/nas-media/Music/Game Soundtracks"]
songs = json.load(open("song_axes.json"))
KR_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KR_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
freqs = np.fft.rfftfreq(NFFT, 1 / RATE)
# chroma map: bin -> pitch class, ignoring bins below 60 Hz
midi = 69 + 12 * np.log2(np.maximum(freqs, 1) / 440.0)
pc = np.rint(midi).astype(int) % 12
chroma_ok = freqs >= 60
window = np.hanning(NFFT).astype(np.float32)
# 32 mel bands, 60 Hz .. 7.6 kHz
def hz2mel(f): return 2595 * np.log10(1 + f / 700)
def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
NMEL = 32
mel_pts = mel2hz(np.linspace(hz2mel(60), hz2mel(7600), NMEL + 2))
MEL = np.zeros((NMEL, len(freqs)))
for b in range(NMEL):
    lo, c, hi = mel_pts[b], mel_pts[b + 1], mel_pts[b + 2]
    MEL[b] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)

def decode(path, start):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{WIN}", "-i", path, "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=120).stdout
    return np.frombuffer(raw, dtype=np.float32)

def feats(x):
    if len(x) < RATE: return None
    n = (len(x) - NFFT) // HOP
    frames = np.lib.stride_tricks.as_strided(x, (n, NFFT), (x.strides[0] * HOP, x.strides[0])) * window
    spec = np.abs(np.fft.rfft(frames, axis=1))  # [n][513]
    power = spec ** 2
    frame_rms = np.sqrt(power.sum(1) / NFFT + 1e-12)
    rms_db = 20 * np.log10(frame_rms + 1e-9)
    tot = power.sum(1) + 1e-12
    centroid = (power * freqs).sum(1) / tot
    cum = np.cumsum(power, 1) / tot[:, None]
    rolloff = freqs[np.argmax(cum >= 0.85, axis=1)]
    flat = np.exp(np.log(spec + 1e-9).mean(1)) / (spec.mean(1) + 1e-9)
    low = power[:, freqs < 250].sum(1) / tot
    high = power[:, freqs > 4000].sum(1) / tot
    zcr = np.mean(np.abs(np.diff(np.sign(frames), axis=1)) > 0, axis=1)
    # onsets: half-wave rectified log-spectral flux
    ls = np.log1p(spec)
    flux = np.maximum(ls[1:] - ls[:-1], 0).sum(1)
    flux = flux - np.convolve(flux, np.ones(21) / 21, mode="same")  # local mean removed
    thr = flux.mean() + flux.std()
    peaks = (flux[1:-1] > thr) & (flux[1:-1] >= flux[:-2]) & (flux[1:-1] > flux[2:])
    onset_rate = peaks.sum() / (len(flux) * HOP / RATE)
    # tempo: autocorrelation of the onset envelope, lags 0.3..1.0 s
    env = np.maximum(flux, 0); env = env - env.mean()
    ac = np.correlate(env, env, mode="full")[len(env) - 1:]
    ac = ac / (ac[0] + 1e-9)
    lo, hi = int(0.3 * RATE / HOP), int(1.0 * RATE / HOP)
    lag = lo + int(np.argmax(ac[lo:hi]))
    bpm = 60.0 / (lag * HOP / RATE)
    tempo_strength = float(ac[lag])
    # mode: chroma vs Krumhansl profiles over 24 keys
    chroma = np.zeros(12)
    for k in range(12): chroma[k] = power[:, chroma_ok & (pc == k)].sum()
    chroma = chroma / (chroma.sum() + 1e-9)
    maj = max(np.corrcoef(chroma, np.roll(KR_MAJ, k))[0, 1] for k in range(12))
    mnr = max(np.corrcoef(chroma, np.roll(KR_MIN, k))[0, 1] for k in range(12))
    melspec = np.log(power @ MEL.T + 1e-6)  # [n][32]
    melspec -= melspec.mean()  # level-free timbre
    mel_mean = melspec.mean(0); mel_sd = melspec.std(0)
    return list(mel_mean) + list(mel_sd) + [rms_db.mean(), rms_db.std(), np.percentile(rms_db, 95) - np.percentile(rms_db, 20), centroid.mean(), np.log(centroid.mean() + 1),
            rolloff.mean(), flat.mean(), low.mean(), high.mean(), zcr.mean(), flux.mean(), onset_rate, bpm, tempo_strength, maj, mnr, maj - mnr]

NAMES = [f"mel{b}" for b in range(NMEL)] + [f"melsd{b}" for b in range(NMEL)] + ["rms_db", "rms_sd", "dyn_range", "centroid", "log_centroid", "rolloff", "flatness", "low_ratio", "high_ratio", "zcr", "flux", "onset_rate", "bpm", "tempo_strength", "major", "minor", "mode"]

def one(s):
    path = next((os.path.join(r, s["path"]) for r in ROOTS if os.path.isfile(os.path.join(r, s["path"]))), None)
    if not path: return None
    total = s["seconds"] or 60
    rows = []
    for at in (0.25, 0.5, 0.75):
        start = max(0.0, min(total * at, total - WIN))
        try:
            f = feats(decode(path, start))
        except Exception:
            f = None
        if f is not None: rows.append(f)
    if not rows: return None
    rows = np.array(rows)
    return {"id": s["id"], "feats": np.concatenate([rows.mean(0), rows.std(0)]).tolist()}

out = []
with ThreadPoolExecutor(8) as ex:
    for i, r in enumerate(ex.map(one, songs)):
        if r: out.append(r)
        if (i + 1) % 200 == 0: print(i + 1, "songs", file=sys.stderr, flush=True)
json.dump({"names": NAMES + [n + "_sd" for n in NAMES], "songs": out}, open("song_feats.json", "w"))
print(len(out), "songs with features", file=sys.stderr)
