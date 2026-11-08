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
