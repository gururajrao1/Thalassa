# Thalassa — Deep-Dive Handbook (future-you README)

Hydrophone WAV in → species / syntax / population insight out.
This file is the return-to-project guide: what Thalassa is, how we built it,
how every piece benchmarks, where the gaps are, and what to try next.
The operator-facing [README](./README.md) stays short;
this file keeps the reasoning.

## 1. What this is

**Thalassa whale-song decoder.** An end-to-end bioacoustics pipeline in
[`whale_song_syntax_analyzer.py`](./whale_song_syntax_analyzer.py)
(~2,400 lines, `run_pipeline` ~line 1819): load WAV at 44.1 kHz →
Butterworth SOS bandpass 50–4000 Hz (zero-phase `sosfiltfilt`) + stationary
spectral-gating denoise → adaptive energy VAD (30th-percentile + 7 dB floor,
harmonic gating) → 128-bin log-mel (25 ms / 10 ms) → discrete acoustic units
(K=64) → bigram + nanoGPT syntax LM → song tokens, phrases, themes, species,
and a Raven-compatible report (`song_tokens.json`, `syntax_analysis_report.png/.pdf`,
`selection_table.txt`).

Two consoles serve it: FastAPI dashboard
([`server.py`](./server.py) + `web/index.html`/`app.js`/`styles.css`,
served at `/`) with job persistence under `jobs/<id>/`, batch workers slider
(1–4), and 25 MB / 30 s streaming slices; Streamlit fallback in
[`app.py`](./app.py). Presets: Champion / Balanced / Fast / Full
(`encodec-rvq` Champion is default).

## 2. What happened (repo + sessions)

~10 commits Nov 8–9, 2026 (branch `master` at `bad4eab`, in sync with origin):

- base decoder + Champion preset + eval matrix + parallel batch
- Thalassa rename → dashboard workers → console redesign + first wild DCLDE labels + AST probe
- fragment-tolerant song gate → wild-calibrated gates + Platt + 32-clip library
- streaming ingest → multi-whale A/B/C recognizer + shipped species head
- latest: wild blue clips, 43-row matrix, fitted Platt, retrained head, library ×39

Uncommitted work (Nov 11): `whale_song_syntax_analyzer.py` +28/−8 —
`CODEC_VERSION` 14→15, `_frame_harmonicity` now blends autocorrelation peak
with YIN (cumulative-mean-normalized difference, de Cheveigné & Kawahara 2002;
stronger witness wins; helps low-SNR 20 Hz fin pulses), plus a blue B/A-call
duration prior (B 10–20 s, A ~8 s, McDonald). Commit this plus a README touch-up
on return. Untracked by design: `jobs/`, `outputs*/`, `samples/`, `wild_dclde/`,
`__pycache__/` — decide commit-vs-`gitignore` before sharing.

Convention (per project notes): every model change ships a benchmark table,
an upgrades-log line, and honest limits — no unmeasured claims.

## 3. How we approach it (pipeline + file map)

1. **Denoise** (`denoise()`): Butterworth + spectral gate from quietest ~0.75 s
   (`prop_decrease=0.9`); `--denoise-backend noisereduce` selects non-stationary path.
2. **VAD** (`vad_segments`, `_frame_harmonicity`): 46 ms RMS frames, adaptive floor,
   median filter, 250 ms hangover, min-unit 0.25 s, merge < 0.20 s; harmonic gate
   (autocorr+YIN voicing + loud bypass) rejects boat/wind hiss.
3. **Features**: log-mel or gammatone-ERB + PCEN (`--pcen-preset quiet|noisy|harsh`),
   optional deltas, `--fmax`, lowcut 10 vs 50 Hz; backend order
   torchaudio → librosa → scipy hand-rolled fallback; filterbank cache + `timings_s`.
4. **Units** (K=64, dedup stride-4): VQ-VAE Conv1d ×4 + EMA + dead-code revival;
   KMeans (MiniBatchKMeans); FSQ 8×8 grid; SSL-kmeans (AVES weights or mel-pool
   fallback flagged `ssl_fallback`); EncodecRVQ 2-stage residual FSQ (numpy-only,
   experimental — not a real Encodec checkpoint).
5. **Syntax LM**: bigram 64×64 Laplace α=1 + held-out PPL; nanoGPT causal
   transformer (default 2×128); unigram entropy, codebook utilization/perplexity,
   `trigram_gap` / `shuffle_gap` nats, `zipf_slope`, repeat-n-gram phrases,
   Louvain theme communities + theme syntax, Needleman–Wunsch similarity,
   PELT-style change-points, trend report, 7-d `file_vector` + library cosine.
6. **Species**: rule-based Bayesian log-odds (humpback/blue/fin + fragment,
   singleton, no-structure gates) **plus** learned head
   `models/species_head.pkl` (winner Model C below, ≤12 uniform 10 s windows).
7. **Data**: `samples/` (bluewhale + 3 MBARI clips), `wild_dclde/manifest.csv`
   (35 clips: 12 fin FIWH lowcut10, 12 humpback HUWH lowcut50, 8 right-whale RIWH
   confusers expect-unassigned, 3 blue BLWH) + 81k-annotation DCLDE CSVs
   (DOI 10.25921/zaea-1s39) via range-fetch [`fetch_wild.py`](./fetch_wild.py);
   synthetic fallback `generate_synthetic_wav`; `library.json` ~26–39 entries with
   DOI provenance ([`build_library.py`](./build_library.py),
   [`append_library.py`](./append_library.py)).

## 4. Benchmarking — every piece, measured

**Unit backends** (`python bench_models.py`, efficiency = accuracy ÷ mean probe s,
`outputs_bench/bench_ranking.json`; 4 synthetic probes + real blue clip):

| Rank | Config | Probes | Mean probe s | Real clip s | Efficiency |
| --- | --- | --- | --- | --- | --- |
| 1 | encodec-rvq / mel | 4/4 | 1.27 | 1.33 | 0.7845 |
| 2 | fsq / mel | 4/4 | 1.45 | 1.41 | 0.6908 |
| 3 | fsq / gamma+pcen | 3/4 (humpback FAIL) | 1.46 | 1.52 | 0.5152 |
| 4 | vqvae / mel-1ep | 3/4 (humpback FAIL) | 1.55 | 1.41 | 0.4840 |
| 5 | kmeans / gamma+pcen | 4/4 | 2.11 | 2.24 | 0.4739 |
| 6 | ssl-kmeans / mel | 3/4 (humpback FAIL) | 1.84 | 1.92 | 0.4082 |
| 7 | kmeans / mel | 4/4 | 9.20 (cold start) | 2.28 | 0.1087 |

Physics frontends cost accuracy: both gamma+PCEN variants lost the humpback probe.
Real-song tiebreak (MBARI 2152, 4.5 min): encodec-rvq humpback-like 1.0, 50 phrases,
gap +2.52 nats, 38.5 s vs FSQ humpback-like 1.0, 31 phrases, gap +1.53 nats, 28.4 s.
Caveat: encodec-rvq here is numpy-only experimental, promoted on measured wins.

**Wild + probes** ([`eval_matrix.py`](./eval_matrix.py): confusion + P/R/F1 +
threshold sweep 0.1–0.9 + max-F1 point; Platt refused until ≥30 wild clips):

| Slice | Acc | Macro-F1 | n |
| --- | --- | --- | --- |
| Probes (smoke only) | 1.00 | 1.00 | 4 |
| Wild DCLDE 2013 | 0.657 | 0.551 | 35 |
| All rows | 0.721 | 0.69 | 43 |

Wild detail: fin 12/12 fin-like; right-whale confusers abstain 8/8; humpback 2/12
(fragments abstain via `n_phrases>=5` gate); blue 1/3 (blwh1 blue-like 0.88, blwh2 too
short, blwh3 fin-like). Synthetic gate ([`eval_accuracy.py`](./eval_accuracy.py)):
hum→humpback-like, noise→unassigned, moans/pulses never humpback.

**Species heads** ([`train_species.py`](./train_species.py), LOO over 43 clips):

| Model | LOO acc |
| --- | --- |
| A file-vector (7 syntax numbers) + logreg | 0.535 |
| B token-histogram (64 code counts) + logreg | 0.465 |
| C frozen AST + logreg | **0.744** |

C ships as `models/species_head.pkl` (`models/species_head.json`).
Confusion: humpback 12/16, fin 10/13, other 8/8, blue 2/5 — blue is thinnest.
**Supervised probe** ([`train_probe.py`](./train_probe.py), frozen MIT AST
`ast-finetuned-audioset-10-10-0.4593`, LOO over 12): 5/12 — all 5 real humpbacks
(2 wild + 3 MBARI) correct across sites, all 4 synthetic probes miss (OOD signal:
argument for wild data). Full-fit trains to 1.00 (memorization). Production needs
30+ wild clips/class.
**Calibration**: Platt slope +0.669, intercept +0.45, fit acc 0.692 (n=39) —
higher confidence measurably predicts correctness. Isochrony tested and rejected
(onset-interval CV does not separate fin trains from song fragments).
**Latency / memory / robustness**: batch `run_batch(max_workers=2)` 49.5 s → 17.7 s
(2.79×, bit-identical via training lock); matplotlib serialized under the same lock;
per-stage `timings_s` + filterbank cache on/off; AST capped ≤12×10 s windows;
grid VQ-VAE/GPT epochs×layers/dim/block (1ep-1×32-8 … 5ep-2×128-64); PCEN
quiet/noisy/harsh sweep; lowcut 10 vs 50; VAD grid; seed-variance + serial/parallel
parity in [`test_app.py`](./test_app.py) / [`test_smoke.py`](./test_smoke.py).

## 4b. Upgrade round Nov 11–12 (uncommitted; all 7 built, 10/10 smoke green)

1. **Frozen-embedding transfer** (`train_probe.py`: `embed`/`embed_arrays`/`embed_segments`,
   `centroid_scores`; per-call `_segment_species_predict` in analyzer, ≤12 segments):
   mel-pool fallback deterministic and shape-interchangeable with AST; raw int16
   arrays normalized like `load16` (no overflow). Shipped head retrained with centroids
   (`models/species_head.json`: `has_centroids: true`, LOO **0.744**, n=43 — unchanged).
2. **Box detector** (`detect_boxes`, persisted as `det_boxes` + `n_boxes` in meta):
   6 s synthetic → 64 boxes. Event eval (`eval_events.py`, collar 0.5 s,
   `outputs_events/event_eval.json`): blue-like precision **0.036**, recall 1.0,
   F1 **0.07** (83 preds vs 3 true; operating point F1 0.092) — massively over-fires.
3. **Dual-path features** (`dual_path_features`: waveform stats + mel) — unit-tested,
   not yet benchmarked against single-path.
4. **Contrastive SSL + retokenization** (`contrast_ssl.py`: `train_projector`,
   `hubert_iter_units`; `units=hubert-iter` backend) — unit-tested, no full-bench row yet.
5. **U-Net denoiser** (`unet_denoise.py`, `models/unet_denoiser.pt` trained;
   `eval_denoise.py`, `outputs_denoise/denoise_eval.json`, 6 synthetic ship-noise pairs):
   mean lift gating **+5.34 dB** vs U-Net **+4.91 dB** — stationary gate still wins.
6. **Augmentation** (`augment.py`: `augment_mel`/`mixup`/`augment_clip`/`ship_noise`;
   `--aug-spec/--aug-ship-snr/--aug-mixup` train units only, clean segments for LM):
   aug-on 6 s run → 2 segments, `n_fit_extra` 5. No ablation vs clean yet.
7. **Event metrics** (`eval_events.py`: collar F1 + threshold sweep + per-class PR) —
   code done, numbers above; MBARI OOD soundscape eval still open.

**Regression fixed this session:** `seg_rms` was appended only inside the augmentation
branch, so default runs zipped `bounds × seg_rms × seqs` to **0 segments**
(`test_smoke_pipeline`, `test_unit_backends_fast` failed). Moved into the per-segment
loop; full suite 10/10 green.

## 5. Gaps (honest, tracked in `known_gaps`)

Blue class thinnest (2/5, wild 1/3); humpback fragments abstain (2/12);
single-unit abstention by design; `ssl-kmeans` always falls back without real
AVES/Perch weights; `encodec-rvq` not a real checkpoint; gamma+PCEN loses humpback
probe — ablate before trusting; single hydrophone, no TDOA/localization;
minutes-per-file CPU, not buoy-ready; 30+ wild clips/class still needed for
per-class thresholds and a production head.
New from this round: box detector precision unusable as-is (needs operating-point
tuning or box-label training); U-Net adds no lift over the gate on synthetic pairs
(needs real ship-noise pairs); dual-path/hubert-iter/augment have no benchmark
deltas yet — `bench_models.py` still ranks only the 7 pre-round configs.

## 6. Research-backed upgrades (ranked)

1. **Frozen foundation-embedding transfer** — Perch 2.0 / BirdNET embeddings +
   linear probe for frame/call scoring. Cheapest win, fits the existing
   `species_head.pkl` pattern, cuts label need. Supported by Perch 2.0 (millions
   of recordings, 5-s embeddings, marine/whale eval sets) and Hexeberg et al.
   arXiv:2502.13440 (contrastive SSL +18.8% OOD). Cost: weight dependency + fallback.
2. **Spectrogram object detector** — RF-DETR / DINOv2-ViT or HTS-AT / AST emitting
   time-frequency boxes (BioDCASE 2025/2026 Task 2 Antarctic blue/fin winners).
   Fixes multi-call-per-window ambiguity. Cost: box labels, heavier infer.
3. **Dual-path waveform + mel classifier** — wav2vec2/HuBERT raw branch +
   mel EfficientNet (2026 underwater-species result), or Dolph2Vec domain SSL;
   keep ssl-kmeans/encodec-rvq as baselines. Cost: compute/data.
4. **Contrastive SSL + iterative retokenization** — SimCLR/CPC-pretrain the unit
   encoder, HuBERT-style SSL-kmeans retokenization before nanoGPT; scale depth via
   the existing perplexity-vs-bigram-NLL report. Cost: longer training.
5. **Learned U-Net spectral-mask denoiser** (or enable stubbed `noisereduce`
   non-stationary path) to replace the stationary gate for ship noise.
   Cost: noisy-clean pairs or unsupervised mask training.
6. **Frontend + augmentation** — keep 128-bin log-mel + PCEN/gammatone, add
   SpecAugment masks + mixup + SNR-varied ship-noise injection, turn on
   `use_deltas`. Cost: tuning; re-ablate gamma+PCEN (lost humpback probe).
7. **Event-based metrics** — mAP/F1 with temporal collars (`sed_eval` style) +
   per-class PR + held-out MBARI OOD soundscape eval. Cost: eval code + collar
   convention. (Perch-quant whale scores and BioDCASE report PDFs were not
   fully fetched — snippet-level evidence; re-verify before citing numbers.)

## 7. Roadmap / resume checklist

Next: commit the YIN + blue-prior edit **plus this 7-upgrade round** (8 files modified,
7 new modules, retrained `species_head.pkl`, `unet_denoiser.pt`); tune box-detector
operating point or train on box labels (precision 0.036 blocks use); re-bench
`hubert-iter`/dual-path/augment in `bench_models.py`; U-Net on real ship-noise pairs;
add ≥30 wild clips/class (blue first); per-class thresholds; real AVES/Perch +
Encodec weights; TDOA/localization; buoy-ready speed; streaming full pipeline.
Decide `gitignore` for `jobs/`/`outputs*/`/`samples/`/`wild_dclde/`.

## 8. Reproduce

```bash
pip install -r requirements.txt   # torch CPU wheel recommended
python whale_song_syntax_analyzer.py --input in.wav --outdir outputs
python bench_models.py            # outputs_bench/bench_ranking.json
python eval_accuracy.py && python eval_matrix.py   # confusion, Platt, operating point
python train_species.py && python train_probe.py   # LOO A/B/C, probe, species_head.pkl
python eval_denoise.py && python eval_events.py    # gating vs U-Net lift, collar F1
python fetch_wild.py && python build_library.py   # DCLDE range-fetch + library.json
pytest test_smoke.py test_app.py -q
uvicorn server:app --port 8000    # dashboard at http://localhost:8000
```

Seeds fixed, CPU float32; `CODEC_VERSION` busts stale job caches; DOI
10.25921/zaea-1s39 + `wild_dclde/manifest.csv` + `library.json` provenance.
Sources: Hexeberg et al. arXiv:2502.13440; Google Perch 2.0 (IEEE Spectrum);
BioDCASE 2025/2026 Task 2; MIT AST `ast-finetuned-audioset-10-10-0.4593`;
NOAA NCEI DCLDE 2013.

