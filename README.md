# Thalassa — Whale-Song Decoder

End-to-end bioacoustic language pipeline: hydrophone audio in, species,
syntax, and population insight out.

## Console UI (full-fledged)

```bash
pip install -r requirements.txt   # torch CPU wheel recommended
uvicorn server:app --port 8000
```

Open http://localhost:8000 in a browser: upload hydrophone audio, tune
units/denoise/PCEN/VAD, review detections, phrases, themes, Species and
Meaning panels, the sonar report, and batch comparison. Results persist
under `jobs/<id>/`. The old Streamlit entry (`app.py`) remains as a
fallback.

## Pipeline

Continuous hydrophone WAV (`underwater_hydrophone.wav`, 44.1 kHz) goes through:

1. **Denoise** — 4th-order Butterworth SOS bandpass 50–4000 Hz (zero-phase
   `sosfiltfilt`) + stationary spectral-gating denoise (noise profile from
   the quietest ~0.75 s, `prop_decrease=0.9`).
2. **Energy VAD segmentation** — 46 ms RMS frames, adaptive floor
   (30th percentile + 7 dB), median filter, 250 ms hangover, min-unit 0.25 s,
   merge gaps < 0.20 s.
3. **128-bin log-mel** (25 ms window / 10 ms hop, 50–4000 Hz) per segment.
   Uses `torchaudio` when installed, else `librosa`, else a scipy +
   hand-rolled triangular filterbank fallback (no hard dependency).
4. **VQ-VAE unit discovery (K=64)** — Conv1d encoder/decoder (stride-2 x2,
   time downsample x4), nearest-codebook lookup, straight-through estimator,
   commitment loss (beta 0.25), EMA dictionary updates (decay 0.99, disable
   with `--no-vq-ema`). Each segment becomes a token sequence (consecutive
   duplicates removed, GSLM-style).
5. **Syntax LM** — required bigram baseline (64×64 counts + Laplace α=1,
   held-out perplexity) plus a hand-rolled nanoGPT-style causal transformer
   (default 2 layers / 128-d, configurable). Reports GPT perplexity
   `exp(mean NLL)`, unigram Shannon entropy (bits), bigram perplexity,
   codebook utilization / perplexity.

## Outputs (in `--outdir`)

- `song_tokens.json` — `{meta, segments: [{id, start_s, end_s, duration_s,
  rms_db, tokens}], codebook_vectors, codebook, lm: {perplexity,
  bits_per_token, entropy_bits_per_token, bigram_perplexity,
  transition_matrix_KxK, ...}}`
- `syntax_analysis_report.png` / `.pdf` — 3 panels, matplotlib only:
  codebook PCA scatter (KMeans colors, top-5 frequent codes annotated,
  UMAP inset when `umap-learn` is installed) | bigram transition heatmap
  (log scale) | VQ-VAE + GPT loss curves with bigram-NLL baseline.

## Usage

```bash
pip install -r requirements.txt   # torch CPU wheel recommended
python whale_song_syntax_analyzer.py --input underwater_hydrophone.wav --outdir outputs
python whale_song_syntax_analyzer.py --input missing.wav --outdir outputs --synthetic-if-missing
python whale_song_syntax_analyzer.py --input in.wav --outdir out --epochs-vq 10 --epochs-lm 10
pytest test_smoke.py -q
```

Key flags: `--epochs-vq`, `--epochs-lm`, `--K` (default 64), `--seed`,
`--vq-ema` / `--no-vq-ema`, `--gpt-layers`, `--gpt-dim`, `--block-size`,
`--synthetic-if-missing`, `--synth-duration-s`,
`--units vqvae|kmeans|fsq|ssl-kmeans|encodec-rvq`,
`--use-pcen`, `--use-deltas`, `--fmax`, `--input-dir`, `--config`,
`--denoise-backend gating|noisereduce`, `--pcen-tau/s/alpha`,
`--pcen-preset quiet|noisy|harsh`, `--ssl-weights`,
`--vad-offset-db/hangover/min-unit/merge-gap`, `--raven-import table.txt`.

- `--units kmeans`: MiniBatchKMeans-64 on standardized mel, zero training.
- `--units fsq`: FSQ-lite 8x8 fixed grid (exactly K=64), zero training.
- `--units ssl-kmeans`: pretrained bioacoustics frontend + k-means when
  `--ssl-weights` points at local AVES/Animal2Vec-style weights; offline
  mel-pool fallback otherwise (flagged in `meta.ssl_fallback`).
- `--use-pcen` / `--use-deltas` / `--fmax 8000`: PCEN frontend, delta
  channels, wider mel band for harmonic-retention experiments.
- `--denoise-backend noisereduce`: non-stationary spectral gating when the
  `noisereduce` package is installed, hand-rolled gate otherwise.
- `--input-dir wavs/ --outdir batch_out`: per-file subdirs plus
  `summary_by_file.json` (per-file PPL/entropy/Zipf/gap for year/region tracking).

Outputs add `selection_table.txt` (Raven-compatible Begin/End times plus
Low/High Freq, Theme columns), `phrases` (repeat-sequence phrase layer),
`themes` (transition-graph communities, provisional), and
`lm.trigram_gap_nats` / `lm.shuffle_gap_nats` / `lm.zipf_slope`
deep-syntax diagnostics. Batch adds `trend_report.png/.pdf`
(entropy/complexity tracker with revolution-dip flags) and
`population_matrix.json` (Levenshtein similarity).

No venv creation required; everything runs on CPU with float32 and fixed seeds.

## Benchmarking (measured, `python bench_models.py`)

Seven backends pitted head-to-head on the same 4 ground-truth probes
plus the short real blue clip. Efficiency = accuracy ÷ mean probe
seconds (`outputs_bench/bench_ranking.json`):

| Rank | Config | Probes | Mean probe s | Real clip s | Efficiency |
| --- | --- | --- | --- | --- | --- |
| 1 | encodec-rvq/mel | 4/4 | 1.27 | 1.33 | 0.7845 |
| 2 | fsq/mel | 4/4 | 1.45 | 1.41 | 0.6908 |
| 3 | fsq/gamma+pcen | 3/4 (humpback FAIL) | 1.46 | 1.52 | 0.5152 |
| 4 | vqvae/mel-1ep | 3/4 (humpback FAIL) | 1.55 | 1.41 | 0.4840 |
| 5 | kmeans/gamma+pcen | 4/4 | 2.11 | 2.24 | 0.4739 |
| 6 | ssl-kmeans/mel | 3/4 (humpback FAIL) | 1.84 | 1.92 | 0.4082 |
| 7 | kmeans/mel | 4/4 | 9.20 (cold start) | 2.28 | 0.1087 |

Physics frontends cost accuracy: both gamma+pcen variants lost the
humpback probe. Real-song tiebreak (MBARI 2152, 4.5 min): encodec
humpback-like 1.0, 50 phrases, gap +2.52 nats, 38.5 s vs fsq
humpback-like 1.0, 31 phrases, gap +1.53 nats, 28.4 s. Winner
`Champion (encodec-rvq, efficient)` is the default preset in
`app.py`, `server.py`, and `web/app.js`. Note: encodec-rvq here is a
numpy-only experimental backend (not a real Encodec checkpoint) —
promoted on measured wins, not paper claims.

Batch parallelism: `run_batch(..., max_workers=2)` gives 2.79x on 4
probes (49.5 s → 17.7 s) with bit-identical output to serial, via a
training lock that keeps torch RNG streams from crossing threads.

## Upgrades log

- Harmonic-gated VAD (autocorr voicing + loud bypass) rejects boat/wind hiss.
- Sea-profile spectral gating (Wenz-like weighting) + lowcut 10 Hz mode.
- Gammatone-ERB frontend with per-channel PCEN tau for baleen moans.
- Louvain themes (modularity) replacing quantile connected components.
- Needleman-Wunsch population similarity + PELT-style change-points.
- Bayesian log-odds species evidence with hard no-structure gate.
- VQ-VAE dead-code revival (EMA reseed + jitter) fixing collapse.
- Filterbank cache, per-stage `timings_s`, theme-level syntax layer.
- `eval_matrix.py`: confusion matrix + P/R/F1 + threshold sweep +
  max-F1 operating point; Platt scaling refused until ≥30 wild clips.

## Limits (honest)

- Synthetic-probe-anchored; wild precision/recall needs ~30+ labeled
  clips (DCLDE 2013 first).
- Single hydrophone, no TDOA/localization, no streaming yet.
- `ssl-kmeans` always falls back without real AVES/Perch weights.
- Minutes-per-file CPU on long recordings; not buoy-ready.
