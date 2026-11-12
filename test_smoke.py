"""Fast CPU smoke test: tiny synthetic fixture, 1 epoch each, schema asserts."""

import json
import os

from whale_song_syntax_analyzer import run_pipeline


def test_smoke_pipeline(tmp_path):
    outdir = str(tmp_path / "out")
    res = run_pipeline(
        input_path=str(tmp_path / "missing.wav"),
        outdir=outdir,
        epochs_vq=1,
        epochs_lm=1,
        K=64,
        seed=0,
        gpt_layers=1,
        gpt_dim=32,
        block_size=8,
        synthetic_if_missing=True,
        synth_duration_s=6.0,
    )
    for key in ("json", "png", "pdf"):
        assert os.path.isfile(res[key]), key
    with open(res["json"]) as f:
        payload = json.load(f)
    assert "segments" in payload and len(payload["segments"]) >= 1
    s0 = payload["segments"][0]
    assert {"start_s", "end_s", "tokens"} <= set(s0)
    assert all(0 <= t < 64 for t in s0["tokens"])
    assert len(payload["codebook_vectors"]) == 64
    assert len(payload["lm"]["transition_matrix_KxK"]) == 64
    assert payload["lm"]["perplexity"] > 0
    m = res["metrics"]
    assert m["bigram_ppl"] > 0 and m["gpt_ppl"] > 0
    assert m["codes_used"] >= 1


def test_unit_backends_fast(tmp_path):
    """Upgrade-pack backends run without training: kmeans, fsq+pcen."""
    from whale_song_syntax_analyzer import run_pipeline
    for units, kwargs in (
        ("kmeans", {}),
        ("fsq", {"use_pcen": True}),
    ):
        outdir = str(tmp_path / f"out_{units}")
        res = run_pipeline(
            input_path=str(tmp_path / "missing.wav"),
            outdir=outdir,
            epochs_vq=1,
            epochs_lm=1,
            K=64,
            seed=0,
            gpt_layers=1,
            gpt_dim=32,
            block_size=8,
            synthetic_if_missing=True,
            synth_duration_s=3.0,
            units=units,
            **kwargs,
        )
        with open(res["json"]) as f:
            payload = json.load(f)
        assert payload["meta"]["units"] == units
        assert len(payload["segments"]) >= 1
        assert len(payload["codebook_vectors"]) == 64
        assert res["metrics"]["gpt_ppl"] > 0


def test_foundation_transfer_fast(tmp_path):
    """Frozen-embedding transfer (#1): mel-pool fallback is deterministic,
    interchangeable in shape with AST, and centroid call-scoring works."""
    import numpy as np
    from scipy.io import wavfile
    from train_probe import (centroid_scores, embed, embed_arrays,
                             embed_segments)
    from whale_song_syntax_analyzer import generate_synthetic_wav
    wa = str(tmp_path / "a.wav")
    wb = str(tmp_path / "b.wav")
    generate_synthetic_wav(wa, duration_s=4.0, seed=0)
    generate_synthetic_wav(wb, duration_s=4.0, seed=1)
    from train_probe import load16
    Xa, used_a, fb_a = embed([wa, wb], backend="mel")
    assert used_a == "mel" and fb_a is True
    assert Xa.shape == (2, 128) and np.isfinite(Xa).all()
    Xb, _, _ = embed_arrays([load16(wa)], sr=16000, backend="mel")
    assert np.allclose(Xa[0], Xb[0], atol=1e-6)  # file == array path
    sr_a, d_a = wavfile.read(wa)
    # Raw int16 arrays are normalized like load16 (no int overflow).
    Xi, _, _ = embed_arrays([d_a], sr=sr_a, backend="mel")
    assert np.isfinite(Xi).all() and (np.abs(Xi) < 1e6).all()
    Xa2, _, _ = embed([wa, wb], backend="mel")
    assert np.allclose(Xa, Xa2)  # deterministic
    assert not np.allclose(Xa[0], Xa[1])  # different seeds differ
    cents = {"humpback": Xa[0], "blue": Xa[1]}
    pred, conf = centroid_scores(Xa, cents, ["humpback", "blue"])
    assert list(pred) == [0, 1] and float(conf.min()) > 0.5
    bounds = [(0.0, 2.0), (2.0, 4.0)]
    Xs, kept = embed_segments(wa, bounds, backend="mel")
    assert len(Xs) == 2 and kept == [0, 1] and Xs.shape[1] == 128
    Xs_empty, kept_empty = embed_segments(wa, [], backend="mel")
    assert len(Xs_empty) == 0 and kept_empty == []


def test_box_detector_fast(tmp_path):
    """Spectrogram box detector (#2): chirps yield valid boxes, tiny/empty
    inputs yield none, and the pipeline persists them with timings."""
    import json
    import numpy as np
    from whale_song_syntax_analyzer import (detect_boxes, generate_synthetic_wav,
                                            log_mel, run_pipeline)
    from scipy.io import wavfile
    assert detect_boxes(np.zeros((0, 0))) == []
    assert detect_boxes(np.zeros((128, 1))) == []
    wp = str(tmp_path / "chirp.wav")
    generate_synthetic_wav(wp, duration_s=6.0, seed=0)
    sr, d = wavfile.read(wp)
    if d.ndim > 1:
        d = d.mean(axis=1)
    m = log_mel(d.astype(np.float32), sr)
    boxes = detect_boxes(m, sr)
    assert len(boxes) >= 1
    for b in boxes:
        assert b["t1_s"] > b["t0_s"] >= 0.0
        assert b["f1_hz"] > b["f0_hz"] >= 0.0
        assert b["score"] > 0
    scores = [b["score"] for b in boxes]
    assert scores == sorted(scores, reverse=True)
    assert boxes[0]["f0_hz"] < 1000.0  # synthetic chirps live low
    res = run_pipeline(
        input_path=str(tmp_path / "missing.wav"),
        outdir=str(tmp_path / "out_det"),
        epochs_vq=1, epochs_lm=1, K=64, seed=0,
        gpt_layers=1, gpt_dim=32, block_size=8,
        synthetic_if_missing=True, synth_duration_s=3.0, units="fsq",
    )
    with open(res["json"]) as f:
        payload = json.load(f)
    assert isinstance(payload.get("detections"), list)
    assert payload["meta"]["n_boxes"] == len(payload["detections"])
    assert payload["meta"]["timings_s"]["detect_s"] >= 0


def test_dual_path_fast(tmp_path):
    """Dual-path classifier (#3): fixed-length, deterministic, finite,
    and seed-sensitive features."""
    import numpy as np
    from whale_song_syntax_analyzer import (dual_path_features,
                                            generate_synthetic_wav,
                                            waveform_stats)
    from scipy.io import wavfile
    wa = str(tmp_path / "a.wav")
    wb = str(tmp_path / "b.wav")
    generate_synthetic_wav(wa, duration_s=4.0, seed=0)
    generate_synthetic_wav(wb, duration_s=4.0, seed=1)
    _, da = wavfile.read(wa)
    _, db = wavfile.read(wb)
    fa = dual_path_features(da.astype(np.float32), 44100)
    fb = dual_path_features(db.astype(np.float32), 44100)
    assert len(fa) == 8 + 2 * 128
    assert np.isfinite(np.asarray(fa)).all()
    assert dual_path_features(da.astype(np.float32), 44100) == fa
    assert not np.allclose(fa, fb)
    w = waveform_stats(da.astype(np.float32), 44100)
    assert len(w) == 8 and np.isfinite(np.asarray(w)).all()
    assert 0.0 <= w[1] <= 1.0  # zero-crossing rate is a fraction


def test_hubert_iter_fast(tmp_path):
    """Contrastive SSL (#4): two-round retokenization is deterministic,
    yields K valid codes, and records its diagnostics."""
    import json
    import numpy as np
    from contrast_ssl import augment_view, hubert_iter_units
    from whale_song_syntax_analyzer import log_mel, run_pipeline
    from scipy.io import wavfile
    from whale_song_syntax_analyzer import generate_synthetic_wav
    wp = str(tmp_path / "c.wav")
    generate_synthetic_wav(wp, duration_s=4.0, seed=0)
    _, d = wavfile.read(wp)
    m = log_mel((d / 32768.0).astype(np.float32), 44100)
    v1 = augment_view(m, 0)
    assert v1.shape == m.shape and np.isfinite(v1).all()
    assert not np.allclose(v1, m)  # the view actually changes
    s1, cb1, info1 = hubert_iter_units([m, m], K=64, seed=0)
    s2, _, info2 = hubert_iter_units([m, m], K=64, seed=0)
    assert [list(s) for s in s1] == [list(s) for s in s2]  # deterministic
    assert len(cb1) == 64 and len(s1) == 2
    assert all(0 <= t < 64 for s in s1 for t in s)
    assert info1["rounds"] in (1, 2)
    res = run_pipeline(
        input_path=str(tmp_path / "missing.wav"),
        outdir=str(tmp_path / "out_hub"),
        epochs_vq=1, epochs_lm=1, K=64, seed=0,
        gpt_layers=1, gpt_dim=32, block_size=8,
        synthetic_if_missing=True, synth_duration_s=3.0, units="hubert-iter",
    )
    with open(res["json"]) as f:
        payload = json.load(f)
    assert payload["meta"]["units"] == "hubert-iter"
    assert payload["meta"]["hubert_iter"]["rounds"] in (1, 2)
    assert len(payload["codebook_vectors"]) == 64


def test_unet_denoise_fast(tmp_path):
    """Learned mask denoiser (#5): mask in [0,1], length preserved,
    deterministic, and missing weights fall back flagged."""
    import numpy as np
    import torch
    from unet_denoise import TinyMaskUNet, unet_denoise
    from whale_song_syntax_analyzer import (denoise,
                                            denoise_backend_available,
                                            generate_synthetic_wav)
    from scipy.io import wavfile
    wp = str(tmp_path / "n.wav")
    generate_synthetic_wav(wp, duration_s=2.0, seed=0)
    _, d = wavfile.read(wp)
    x = (d.astype(np.float32) / 32768.0)
    net = TinyMaskUNet(seed=0)
    net.eval()
    with torch.no_grad():
        m1 = net(torch.zeros(1, net.n_bins, 17))
    assert m1.shape == (1, net.n_bins, 17)
    assert float(m1.min()) >= 0.0 and float(m1.max()) <= 1.0
    y1, fb1 = unet_denoise(x, 44100,
                           weights_path=str(tmp_path / "nope.pt"))
    assert fb1 is True and len(y1) == len(x)  # fallback, flagged
    assert denoise_backend_available("unet") in (True, False)
    assert denoise_backend_available(
        "unet", unet_weights=str(tmp_path / "nope.pt")) is False
    y2, _ = unet_denoise(x, 44100, weights_path=str(tmp_path / "nope.pt"))
    assert np.allclose(np.asarray(y1), np.asarray(y2))  # deterministic


def test_augment_fast(tmp_path):
    """Augmentations (#6): deterministic, shape-preserving, ship noise hits
    its SNR, and the pipeline records augmented training copies."""
    import json
    import numpy as np
    from augment import augment_clip, augment_mel, mixup, ship_noise
    from whale_song_syntax_analyzer import (generate_synthetic_wav, log_mel,
                                            run_pipeline)
    from scipy.io import wavfile
    wp = str(tmp_path / "s.wav")
    generate_synthetic_wav(wp, duration_s=4.0, seed=0)
    _, d = wavfile.read(wp)
    m = log_mel((d / 32768.0).astype(np.float32), 44100)
    a1 = augment_mel(m, 7, use_spec=True)
    a2 = augment_mel(m, 7, use_spec=True)
    assert a1.shape == m.shape and np.allclose(a1, a2)  # deterministic
    assert not np.allclose(a1, m)  # masks change the view
    assert np.allclose(augment_mel(m, 7, use_spec=False), m)  # off = clean
    mx, lam = mixup(m, m[::-1], lam=None, seed=3)
    assert mx.shape == m.shape and 0.2 <= lam <= 0.6
    x = (d.astype(np.float64) / 32768.0).astype(np.float32)
    ref = float(np.sqrt((x.astype(np.float64) ** 2).mean()))
    nz = ship_noise(len(x), 44100, seed=11, snr_db=5.0, ref_rms=ref)
    got = 20 * np.log10(ref / (np.sqrt((nz.astype(np.float64) ** 2).mean())
                               + 1e-12))
    assert abs(got - 5.0) < 1.0  # hits target SNR within 1 dB
    xa = augment_clip(x, 44100, seed=5, ship_snr=5.0)
    assert xa.shape == x.shape and not np.allclose(xa, x)
    assert np.allclose(augment_clip(x, 44100, seed=5, ship_snr=None), x)
    res = run_pipeline(
        input_path=str(tmp_path / "missing.wav"),
        outdir=str(tmp_path / "out_aug"),
        epochs_vq=1, epochs_lm=1, K=64, seed=0,
        gpt_layers=1, gpt_dim=32, block_size=8,
        synthetic_if_missing=True, synth_duration_s=3.0, units="fsq",
        aug_spec=True, aug_ship_snr=5.0, aug_mixup=True,
    )
    with open(res["json"]) as f:
        payload = json.load(f)
    assert payload["meta"]["aug_spec"] is True
    assert payload["meta"]["aug_ship_snr"] == 5.0
    assert payload["meta"]["n_fit_extra"] >= 2 * len(payload["segments"])


def test_event_f1_fast():
    """Event metrics (#7): collar matching on hand-built cases."""
    from eval_events import event_f1, sweep
    pr, rc, f1, tp, fp, fn = event_f1([(1.0, 2.0, 0.9)], [(1.1, 1.9)],
                                      collar_s=0.5)
    assert (tp, fp, fn) == (1, 0, 0) and f1 == 1.0
    pr, rc, f1, tp, fp, fn = event_f1([(5.0, 6.0, 0.9)], [(1.0, 2.0)],
                                      collar_s=0.5)
    assert (tp, fp, fn) == (0, 1, 1) and f1 == 0.0
    # Two preds, one true: one TP + one FP; empty preds: all FN.
    pr, rc, f1, tp, fp, fn = event_f1(
        [(1.0, 2.0, 0.9), (1.2, 2.2, 0.4)], [(1.1, 1.9)], collar_s=0.5)
    assert (tp, fp, fn) == (1, 1, 0)
    pr, rc, f1, tp, fp, fn = event_f1([], [(1.0, 2.0)], collar_s=0.5)
    assert (tp, fp, fn) == (0, 0, 1)
    _, best = sweep([(1.0, 2.0, 0.9), (9.0, 9.5, 0.1)], [(1.1, 1.9)])
    assert best["f1"] == 1.0 and best["thresh"] > 0.1


def test_literature_upgrades_fast(tmp_path):
    """ssl-kmeans fallback, PCEN flags, phrase/zipf/gap, selection table."""
    outdir = str(tmp_path / "out_ssl")
    res = run_pipeline(
        input_path=str(tmp_path / "missing.wav"),
        outdir=outdir,
        epochs_vq=1,
        epochs_lm=1,
        K=64,
        seed=0,
        gpt_layers=1,
        gpt_dim=32,
        block_size=8,
        synthetic_if_missing=True,
        synth_duration_s=6.0,
        units="ssl-kmeans",
        use_pcen=True,
        pcen_tau=0.08,
        pcen_s=0.04,
        pcen_alpha=0.95,
        denoise_backend="noisereduce",  # falls back to gating offline
    )
    assert os.path.isfile(res["table"])
    with open(res["json"]) as f:
        payload = json.load(f)
    assert payload["meta"]["units"] == "ssl-kmeans"
    assert payload["meta"]["ssl_fallback"] is True
    assert payload["meta"]["denoise_backend"] == "noisereduce"
    assert payload["meta"]["pcen_tau"] == 0.08
    assert "phrases" in payload and isinstance(payload["phrases"], list)
    lm = payload["lm"]
    assert "trigram_gap_nats" in lm and "zipf_slope" in lm
    assert res["metrics"]["n_phrases"] == len(payload["phrases"])
    assert "theme_syntax" in payload and "sequence" in payload["theme_syntax"]
    assert "timings_s" in payload["meta"] and payload["meta"]["timings_s"]["total_s"] > 0
    with open(res["table"]) as f:
        header = f.readline()
    assert "Begin Time (s)" in header
