"""Learned U-Net spectral-mask denoiser (#5).

Tiny Conv1d U-Net predicting a [0, 1] magnitude-STFT mask from log-magnitude
frames: 2 down blocks (stride-2) + bottleneck + 2 up blocks with skip
concatenation, ~20k params, CPU-trainable in minutes on synthetic
noisy-clean pairs. Ships to models/unet_denoiser.pt; the dispatcher falls
back to the hand-rolled stationary gate (flagged) when weights are absent,
mirroring the ssl-kmeans/noisereduce convention.

Usage:
  python -m unet_denoise --train          # builds pairs, trains, ships
  python whale_song_syntax_analyzer.py --denoise-backend unet ...
"""
import os

import numpy as np

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "unet_denoiser.pt")

N_FFT, HOP = 1024, 256


def stft_mag(x, sr=44100):
    from scipy.signal import stft
    _, _, Z = stft(x, fs=sr, window="hann", nperseg=N_FFT,
                   noverlap=N_FFT - HOP, nfft=N_FFT, boundary="zeros")
    return np.abs(Z).astype(np.float32), np.angle(Z).astype(np.float32)


def istft_back(mag, phase, sr=44100):
    from scipy.signal import istft
    _, x = istft(mag * np.exp(1j * phase), fs=sr, window="hann",
                 nperseg=N_FFT, noverlap=N_FFT - HOP, nfft=N_FFT)
    return np.asarray(x, dtype=np.float32)


class TinyMaskUNet:
    """Two-level 1D U-Net over frequency bins (channels), frames = length.

    enc1 -> down(stride 2) -> enc2 -> up -> concat with enc1 (skip) ->
    dec -> out(sigmoid mask). ~20k params at width=32.
    """

    def __init__(self, n_bins=N_FFT // 2 + 1, width=32, seed=0):
        import torch
        torch.manual_seed(seed)
        self.enc1 = torch.nn.Conv1d(n_bins, width, 5, padding=2)
        self.down = torch.nn.Conv1d(width, width, 5, stride=2, padding=2)
        self.enc2 = torch.nn.Conv1d(width, width * 2, 5, padding=2)
        self.up = torch.nn.ConvTranspose1d(width * 2, width, 4, stride=2,
                                           padding=1)
        self.dec = torch.nn.Conv1d(width * 2, width, 5, padding=2)
        self.out = torch.nn.Conv1d(width, n_bins, 5, padding=2)
        self.relu = torch.nn.ReLU()
        self.n_bins = n_bins
        self._mods = (self.enc1, self.down, self.enc2, self.up, self.dec,
                      self.out)

    def parameters(self):
        return [p for m in self._mods for p in m.parameters()]

    def state_dict(self):
        return {f"m{i}": m.state_dict() for i, m in enumerate(self._mods)}

    def load_state_dict(self, st):
        for i, m in enumerate(self._mods):
            m.load_state_dict(st[f"m{i}"])

    def eval(self):
        for m in self._mods:
            m.eval()

    def __call__(self, x):
        """x: (1, bins, F) normalized log-mag. Returns (1, bins, F) mask."""
        import torch
        F = x.shape[2]
        pad = (-F) % 2
        if pad:
            x = torch.nn.functional.pad(x, (0, pad))
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(self.relu(self.down(e1))))
        u = self.relu(self.up(e2))[:, :, :e1.shape[2]]
        y = self.out(self.relu(self.dec(
            torch.cat([u, e1], dim=1))))[:, :, :F]
        return torch.sigmoid(y)


def train_unet(pairs, epochs=5, seed=0):
    """pairs: [(noisy_mag, clean_mag)] log-scaled. Returns state dict."""
    import torch
    torch.manual_seed(seed)
    net = TinyMaskUNet(seed=seed)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    Xs = torch.from_numpy(np.concatenate(
        [np.log1p(p[0]).T for p in pairs], axis=0))
    Ys = torch.from_numpy(np.concatenate(
        [(p[1] / (p[0] + 1e-9)).clip(0, 1).T for p in pairs], axis=0))
    mu, sd = Xs.mean(), Xs.std() + 1e-9
    Xn = ((Xs - mu) / sd).unsqueeze(0).transpose(1, 2)
    Yn = Ys.unsqueeze(0).transpose(1, 2)
    for _ in range(max(1, int(epochs))):
        opt.zero_grad()
        mask = net(Xn)
        loss = ((mask - Yn) ** 2).mean()
        loss.backward()
        opt.step()
    return {"state": net.state_dict(), "mu": float(mu),
            "sd": float(sd), "n_fft": N_FFT, "hop": HOP}


def save_model(state, path=MODEL_PATH):
    import torch
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)
    return path


def load_model(path=MODEL_PATH):
    import torch
    st = torch.load(path, map_location="cpu", weights_only=False)
    net = TinyMaskUNet(n_bins=st.get("n_fft", N_FFT) // 2 + 1)
    net.load_state_dict(st["state"])
    net.eval()
    return net, st


def unet_denoise(x, sr=44100, weights_path=None):
    """Masked denoise. Returns (audio, used_fallback_bool)."""
    wp = weights_path or MODEL_PATH
    if not os.path.isfile(wp):
        from whale_song_syntax_analyzer import spectral_gate_denoise
        return spectral_gate_denoise(x, sr), True
    import torch
    net, st = load_model(wp)
    mag, phase = stft_mag(np.asarray(x, dtype=np.float32), sr)
    Lm = torch.from_numpy(np.log1p(mag).T).unsqueeze(0)
    Xn = ((Lm - st["mu"]) / st["sd"]).transpose(1, 2)
    with torch.no_grad():
        mask = net(Xn).transpose(1, 2).squeeze(0).numpy().T
    y = istft_back((mag * np.clip(mask, 0, 1)).astype(np.float32), phase, sr)
    n = len(np.asarray(x))
    y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    return y.astype(np.float32), False


def make_pairs(outdir="outputs_unet", n_clean=4, snrs=(-5.0, 0.0, 5.0),
               seed=0, sr=44100):
    """Synthetic noisy-clean pairs: chirp fixtures + noise-bed mixtures."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from whale_song_syntax_analyzer import generate_synthetic_wav
    from scipy.io import wavfile
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)
    pairs = []
    for i in range(n_clean):
        cp = os.path.join(outdir, f"clean_{i}.wav")
        generate_synthetic_wav(cp, duration_s=6.0, seed=seed + i)
        _, clean = wavfile.read(cp)
        clean = (clean.astype(np.float64) / 32768.0)
        for snr in snrs:
            noise = rng.standard_normal(len(clean))
            scale = np.sqrt((clean ** 2).mean() /
                            max(noise.var(), 1e-12) / (10.0 ** (snr / 10.0)))
            noisy = (clean + scale * noise).astype(np.float32)
            pairs.append((stft_mag(noisy, sr)[0],
                          stft_mag(clean.astype(np.float32), sr)[0]))
    return pairs


def snr_db(clean, est):
    clean = np.asarray(clean, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)[:len(clean)]
    return float(10 * np.log10((clean ** 2).sum() /
                               max(((clean - est) ** 2).sum(), 1e-12)))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if not a.train:
        ap.error("--train is required (inference lives in the pipeline)")
    pairs = make_pairs(seed=a.seed)
    st = train_unet(pairs, epochs=a.epochs, seed=a.seed)
    path = save_model(st)
    print(f"trained on {len(pairs)} pairs, saved {path}")


if __name__ == "__main__":
    main()
