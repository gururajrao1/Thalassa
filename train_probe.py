"""Supervised probe head: frozen AST embeddings + logistic regression.

Frozen Audio Spectrogram Transformer (MIT/ast-finetuned-audioset, AudioSet
pretraining) embeds each clip; a multinomial logistic probe maps mean-pooled
embeddings to species. Evaluated with leave-one-out CV because the labeled
set is still tiny (4 synthetic probes + 4 wild DCLDE clips + 4 assumed clips).
This is the supervised counterpart to the rule-based match_species baseline:
same clip set, honest small-N numbers, confusion matrix included.
Usage: python train_probe.py
"""
import os

import numpy as np

CLIPS = [
    ("outputs_eval/probe_humpback.wav", "humpback-like", "synthetic"),
    ("outputs_eval/probe_blue.wav", "blue-like", "synthetic"),
    ("outputs_eval/probe_fin.wav", "fin-like", "synthetic"),
    ("outputs_eval/probe_noise.wav", "unassigned", "synthetic"),
    ("wild_dclde/wild_huwh.wav", "humpback-like", "DCLDE2013 HUWH"),
    ("wild_dclde/wild_huwh2.wav", "humpback-like", "DCLDE2013 HUWH2"),
    ("wild_dclde/wild_fiwh.wav", "fin-like", "DCLDE2013 FIWH"),
    ("wild_dclde/wild_riwh.wav", "right-whale", "DCLDE2013 RIWH confuser"),
    ("samples/mbari_2152.wav", "humpback-like", "assumed MBARI"),
    ("samples/mbari_2133.wav", "humpback-like", "assumed MBARI"),
    ("samples/mbari_2143.wav", "humpback-like", "assumed MBARI"),
    ("samples/bluewhale.wav", "blue-like", "assumed archive.org"),
]

LABELS = ["humpback-like", "blue-like", "fin-like", "unassigned",
          "right-whale"]


def load16(path, sr=16000, cap_s=60.0):
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from math import gcd
    rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float64)
    if data.dtype != np.float64 and np.issubdtype(
            getattr(data, "dtype", np.float64), np.integer):
        data = data / float(np.iinfo(data.dtype).max)
    if rate != sr:
        g = gcd(rate, sr)
        data = resample_poly(data, sr // g, rate // g)
    n = int(sr * cap_s)
    return data[:n].astype(np.float32)


AST_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"

# 10.24 s windows / 5 s hop: matches the shipped C-head cadence so clip
# and segment embeddings stay comparable. Short inputs are padded.
WIN, HOP, MINW = 16000 * 10, 16000 * 5, 16000


def _windows(x):
    return ([x[i:i + WIN] for i in range(0, max(1, len(x) - WIN + 1), HOP)]
            or [x])


def _as_float_audio(x):
    """Match load16: int PCM -> [-1, 1] float32 (avoids int overflow in
    filterbanks when callers pass raw wavfile arrays)."""
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
    return np.asarray(x, dtype=np.float32)


def ast_embed_arrays(arrays, sr=16000):
    """Frozen AST encoder, mean-pooled over time then windows (needs weights)."""
    import torch
    from transformers import ASTFeatureExtractor, ASTModel
    fe = ASTFeatureExtractor.from_pretrained(AST_ID)
    model = ASTModel.from_pretrained(AST_ID)
    model.eval()
    vecs = []
    with torch.no_grad():
        for x in arrays:
            embs = []
            for w in _windows(_as_float_audio(x)):
                if len(w) < MINW:
                    w = np.pad(w, (0, MINW - len(w)))
                inp = fe(w, sampling_rate=sr, return_tensors="pt")
                h = model(**inp).last_hidden_state.mean(dim=1)
                embs.append(h.squeeze(0).numpy())
            vecs.append(np.stack(embs).mean(axis=0))
    return np.stack(vecs).astype(np.float32)


def mel_pool_embed_arrays(arrays, sr=16000):
    """Deterministic fallback embedding: mean-pooled log-mel, no weights.

    Same windowing as the AST path so backends stay interchangeable; the
    caller records ``used_fallback=True`` (SSLEmbeddingUnits convention).
    """
    from whale_song_syntax_analyzer import log_mel
    vecs = []
    for x in arrays:
        x = _as_float_audio(x)
        parts = []
        for w in _windows(x):
            if len(w) < MINW:
                w = np.pad(w, (0, MINW - len(w)))
            m = log_mel(w, sr)
            parts.append(m.reshape(m.shape[0], -1).mean(axis=1))
        vecs.append(np.stack(parts).mean(axis=0))
    return np.stack(vecs).astype(np.float32)


def embed_arrays(arrays, sr=16000, backend="auto"):
    """Embed raw-audio arrays with a frozen foundation encoder.

    backend: "auto" (AST, else mel-pool fallback), "ast", or "mel".
    Returns (X, backend_used, used_fallback). Never silently swaps: the
    used backend is always reported so benchmarks stay honest.
    """
    if backend == "mel":
        return mel_pool_embed_arrays(arrays, sr), "mel", True
    if backend == "ast":
        return ast_embed_arrays(arrays, sr), "ast", False
    try:
        return ast_embed_arrays(arrays, sr), "ast", False
    except Exception:
        return mel_pool_embed_arrays(arrays, sr), "mel", True


def embed(paths, backend="auto"):
    """Embed wav files. Returns (X, backend_used, used_fallback)."""
    return embed_arrays([load16(p) for p in paths],
                        backend=backend)


def embed_segments(wav_path, bounds_s, sr=16000, backend="ast", cap=12):
    """Embed (start_s, end_s) slices of one clip for per-call scoring.

    Uniformly subsamples to ``cap`` segments so long files stay bounded.
    Returns (X_seg, kept_index). Raises only on unreadable audio; empty
    bounds give an empty (0, D) array with no indices.
    """
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from math import gcd
    rate, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = np.asarray(data, dtype=np.float64)
    if rate != sr:
        g = gcd(rate, sr)
        data = resample_poly(data, sr // g, rate // g)
    n = len(bounds_s)
    idx = [int(i * (n - 1) / (cap - 1)) for i in range(min(n, cap))] \
        if n > cap else list(range(n))
    arrs = []
    for i in idx:
        t0, t1 = bounds_s[i]
        s0, s1 = max(0, int(t0 * sr)), max(1, int(t1 * sr))
        w = data[s0:s1].astype(np.float32)
        arrs.append(w if len(w) else np.zeros(1, dtype=np.float32))
    if not arrs:
        return np.zeros((0, 128), dtype=np.float32), []
    X, _, _ = embed_arrays(arrs, sr, backend=backend)
    return X, idx


def centroid_scores(X, centroids, labels, temperature=0.5):
    """Cosine-to-centroid call scoring with softmax confidence.

    centroids: {label: vector}; labels: ordered class list. Returns
    (pred_idx, confidence) arrays. Pure numpy, no training involved.
    """
    C = np.stack([np.asarray(centroids[lab], dtype=float) for lab in labels])
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
    Xn = np.asarray(X, dtype=float)
    Xn = Xn / (np.linalg.norm(Xn, axis=1, keepdims=True) + 1e-12)
    sims = Xn @ Cn.T / max(temperature, 1e-6)
    sims = sims - sims.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(sims, -50, 50))
    proba = exp / exp.sum(axis=1, keepdims=True)
    pred = proba.argmax(axis=1)
    return pred, proba.max(axis=1)


def main(backend="auto"):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler
    paths = [c[0] for c in CLIPS]
    y = np.array([LABELS.index(c[1]) for c in CLIPS])
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"missing clips: {missing}")
    print(f"embedding 12 clips (backend={backend}; auto falls back "
          f"AST -> mel-pool, flagged)...")
    X, used, fallback = embed(paths, backend=backend)
    print(f"embeddings: {X.shape} backend={used} fallback={fallback}")
    Xs = StandardScaler().fit_transform(X)
    loo = LeaveOneOut()
    pred = np.zeros_like(y)
    for tr, te in loo.split(Xs):
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xs[tr], y[tr])
        pred[te] = clf.predict(Xs[te])
    acc = float((pred == y).mean())
    print(f"leave-one-out accuracy: {acc:.3f} ({int((pred == y).sum())}/12)")
    print("true -> pred:")
    for (p, t, note), ph in zip(CLIPS, pred):
        mark = "OK " if ph == LABELS.index(t) else "MISS"
        print(f"  [{mark}] {os.path.basename(p)}: {t} -> {LABELS[ph]} ({note})")
    print("confusion (rows true, cols pred):")
    print("  " + " ".join(f"{lab[:5]:>5}" for lab in LABELS))
    for i, lab in enumerate(LABELS):
        row = [(pred[y == i] == j).sum() for j in range(len(LABELS))]
        print(f"  {lab[:5]:>5} " + " ".join(f"{v:>5}" for v in row))
    clf = LogisticRegression(max_iter=2000).fit(Xs, y)
    print(f"full-fit train accuracy (optimistic): {clf.score(Xs, y):.3f}")
    print("NOTE: n=12; treat as protocol proven, not as a species claim. "
          "Production needs 30+ wild clips per class.")
    return acc


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "ast", "mel"])
    main(**vars(ap.parse_args()))
