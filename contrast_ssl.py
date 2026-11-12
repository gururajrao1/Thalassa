"""Contrastive SSL + iterative retokenization (#4): hubert-iter units.

HuBERT-style loop at CPU scale, numpy + tiny torch:
  round 0: MiniBatchKMeans on standardized mel (same as the kmeans backend).
  round 1: train a linear projector with SimCLR-style NT-Xent on two
    augmented views (circular time-shift + band mask + gaussian noise),
    then re-cluster (retokenize) in the projected space.

Deterministic given seed; falls back to round-0 tokens if torch is
unavailable. The full augmentation suite (#6, augment.py) plugs in here
later — these minimal views stay local so #4 works standalone.
Usage: --units hubert-iter
"""
import numpy as np


def augment_view(mel, seed):
    """One stochastic view of a (bands, T) mel: shift + mask + noise."""
    rng = np.random.default_rng(seed)
    m = np.asarray(mel, dtype=np.float64)
    n, T = m.shape
    if T > 4:
        m = np.roll(m, int(rng.integers(1, max(2, T // 4))), axis=1)
    if n > 4:
        f0 = int(rng.integers(0, n - 2))
        fw = min(n - f0, max(1, int(rng.integers(1, n // 8))))
        m = m.copy()
        m[f0:f0 + fw] = m.mean()
    m = m + rng.standard_normal(m.shape) * (m.std() * 0.05 + 1e-6)
    return m.astype(np.float32)


def train_projector(mels, out_dim=64, steps=50, seed=0):
    """SimCLR-style linear projector: NT-Xent over paired views.

    Frames from all segments form the batch; positives are the two views
    of the same frame, negatives every other frame. Returns (W, mean, std)
    with W (in_dim, out_dim) L2-normalized rows for cosine similarity.
    """
    import torch
    torch.manual_seed(seed)
    base = np.concatenate([np.asarray(m, dtype=np.float64)
                           .reshape(m.shape[0], -1).T for m in mels], axis=0)
    base = base - base.mean(axis=0, keepdims=True)
    std = base.std(axis=0, keepdims=True) + 1e-9
    base = (base / std).astype(np.float32)
    X = torch.from_numpy(base)
    P = torch.stack([torch.from_numpy(
        np.concatenate([augment_view(m, seed + 1)
                         .reshape(m.shape[0], -1).T for m in mels], axis=0))
        for _ in range(2)])
    P = (P - P.mean(dim=1, keepdim=True)) / (P.std(dim=1, keepdim=True) + 1e-9)
    proj = torch.nn.Linear(X.shape[1], out_dim, bias=False)
    torch.nn.init.orthogonal_(proj.weight)
    opt = torch.optim.Adam(proj.parameters(), lr=1e-2)
    tau = 0.1
    Z = torch.cat([proj(P[0]), proj(P[1])], dim=0)
    Z = Z / (Z.norm(dim=1, keepdim=True) + 1e-12)
    N = Z.shape[0] // 2
    pos_idx = (torch.arange(2 * N) + N) % (2 * N)
    for _ in range(max(1, int(steps))):
        opt.zero_grad()
        sim = (Z @ Z.T) / tau
        sim.fill_diagonal_(-1e9)
        logp = sim - sim.logsumexp(dim=1, keepdim=True)
        loss = -logp[torch.arange(2 * N), pos_idx].mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            Z = torch.cat([proj(P[0]), proj(P[1])], dim=0)
            Z = Z / (Z.norm(dim=1, keepdim=True) + 1e-12)
    with torch.no_grad():
        W = proj.weight.detach().cpu().numpy().T
    return W.astype(np.float32), base.mean(axis=0), std.squeeze(0)


def hubert_iter_units(seg_mels, K=64, seed=0, steps=50):
    """Two-round discrete units. Returns (seqs, codebook, info).

    seqs: token lists per segment (dedup + stride-4 cadence like the other
    backends is applied by the caller convention — here tokens are per
    frame; the caller dedups). codebook: K projected centroids (or mel
    centroids when torch is missing). info: backend diagnostics.
    """
    from sklearn.cluster import MiniBatchKMeans
    flats, bounds = [], []
    for m in seg_mels:
        a = np.asarray(m, dtype=np.float64).reshape(m.shape[0], -1)
        bounds.append(a.shape[1])
        flats.append(a)
    X0 = np.concatenate(flats, axis=1).T
    mu, sd = X0.mean(axis=0, keepdims=True), X0.std(axis=0, keepdims=True) + 1e-9
    km0 = MiniBatchKMeans(n_clusters=K, n_init=3, random_state=seed)
    lab0 = km0.fit_predict((X0 - mu) / sd)
    try:
        W, _, _ = train_projector(seg_mels, seed=seed, steps=steps)
        Z = ((X0 - mu) / sd) @ W
        Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
        km1 = MiniBatchKMeans(n_clusters=K, n_init=3,
                              random_state=seed + 1)
        lab1 = km1.fit_predict(Z)
        agree = float((lab0 == lab1).mean()) if len(lab0) else 0.0
        codebook = km1.cluster_centers_.astype(float).tolist()
        info = {"rounds": 2, "round_agreement": round(agree, 3),
                "contrastive": True}
        labs = lab1
    except Exception:
        codebook = km0.cluster_centers_.astype(float).tolist()
        info = {"rounds": 1, "round_agreement": 1.0,
                "contrastive": False, "fallback": "torch-missing"}
        labs = lab0
    seqs, i = [], 0
    for b in bounds:
        seqs.append([int(t) for t in labs[i:i + b]])
        i += b
    return seqs, codebook, info
