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


def embed(paths):
    import torch
    from transformers import ASTFeatureExtractor, ASTModel
    fe = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593")
    model = ASTModel.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593")
    model.eval()
    vecs = []
    with torch.no_grad():
        for path in paths:
            x = load16(path)
            # 10.24 s windows, mean-pool time then windows (frozen encoder).
            win, hop = 16000 * 10, 16000 * 5
            parts = [x[i:i + win] for i in range(0, max(1, len(x) - win + 1), hop)] or [x]
            embs = []
            for w in parts:
                if len(w) < 16000:
                    w = np.pad(w, (0, 16000 - len(w)))
                inp = fe(w, sampling_rate=16000,
                         return_tensors="pt")
                h = model(**inp).last_hidden_state.mean(dim=1)
                embs.append(h.squeeze(0).numpy())
            vecs.append(np.stack(embs).mean(axis=0))
    return np.stack(vecs)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler
    paths = [c[0] for c in CLIPS]
    y = np.array([LABELS.index(c[1]) for c in CLIPS])
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"missing clips: {missing}")
    print("embedding 12 clips with frozen AST (first run downloads weights)...")
    X = embed(paths)
    print(f"embeddings: {X.shape}")
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
    main()
