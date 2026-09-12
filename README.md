# Drone3D

Two tracks on one codebase:

- **Product Track** — a **deployable single-entrypoint pipeline** that turns a drone flight **video into a 3D reconstruction** (`scripts/reconstruct.py`). One module of a larger drone SaaS (alongside person-following, video recording, etc.). Ship-first deliverable.
- **Research Track** — **online, telemetry-supervised adaptation of streaming 3D foundation models for drone video** (the paper). Layers on top of the same reconstruction code once Pixhawk capture data arrives.

> Restored 2026-09-12 from `sank6112/drone3d` after local loss. `checkpoints/` and `third_party/` are gitignored — re-run `SETUP.md` (or the minimal MASt3R stack) to repopulate them.

## Quick start — Product pipeline

```bash
conda activate drone3d
# from a flight video (extracts frames, filters blur/duplicates, reconstructs):
python scripts/reconstruct.py --video flight.mp4 --out outputs/run1
# or from a folder of already-extracted frames:
python scripts/reconstruct.py --frames data/test_real/ --out outputs/run1
```
Outputs `reconstruction.ply` (colored, metric — meters), `poses.npz` (cam-to-world), and `result.json`. Runs on the 6 GB RTX 4050 with MASt3R.

## Status (as of 2026-09-12)

| Phase | What | Status |
|---|---|---|
| P | **Product pipeline** `reconstruct.py` (video → frames → MASt3R → .ply/poses) | ✅ built; extraction verified, full run pending MASt3R ckpt re-download |
| 0 | Env + scaffolding | ✅ done |
| 1 | Ground-level baselines (DUSt3R, MASt3R, MonST3R) on NLE tower | ✅ done |
| 1 | VGGT / CUT3R / Point3R / StreamVGGT baselines | ⏸ deferred to cloud |
| 2 | Aerial-MASt3R + 4-backbone matrix on AerialMegaDepth | ✅ done |
| 3 | Eval framework (Chamfer-L1/L2, F-score, ATE, RPE, Sim(3)-Umeyama) | ✅ done |
| 3 | ETH3D-courtyard ground-truth pose evaluation | ✅ done |
| 4 | Full baseline matrix incl. cloud-only backbones | 🟡 partial |
| 5 | Drone capture pipeline (frame extract + telemetry parse + sync) | ⏳ next — drone setup in progress |
| 6 | Telemetry-supervised online adaptation (photometric + geometric + telemetry + dyn-mask losses) | ❌ blocked on Phase 5 + cloud |
| 7 | Synthetic Blender flights for controlled ablation | ❌ not started |
| 8 | Ablations + paper | ❌ not started |

## Headline empirical findings so far

1. **MASt3R is metric, DUSt3R/MonST3R are not.** NLE tower bbox: MASt3R 3.55 × 2.33 × 6.58 m; DUSt3R 0.42 × 0.24 × 0.54 (arbitrary). → telemetry-derived scale loss is only meaningful on metric backbones.
2. **Aerial fine-tuning materially changes geometry.** On AerialMegaDepth fine-arts museum, standard MASt3R reconstructs 6.81 × 4.30 × 3.03 m; Aerial-MASt3R reconstructs 1.21 × 1.79 × 1.87 m. Same architecture, same images, only weights differ. Chamfer-L1 between them = 0.65 — the difference is geometric, not a uniform rescale.
3. **Bidirectional domain-adaptation effect (ETH3D courtyard, ground-level, with real GT poses):**

   | Model | ATE RMSE ↓ | RPE trans ↓ | RPE rot ↓ |
   |---|---:|---:|---:|
   | **MASt3R** | **0.12 m** | **0.10 m** | **0.85°** |
   | DUSt3R | 0.36 m | 0.24 m | 1.66° |
   | Aerial-MASt3R | 0.33 m | 0.16 m | 2.23° |

   Aerial-MASt3R is ~3× worse than standard MASt3R on ground-level data. The paper will tell the symmetric story on drone data: standard MASt3R loses; our telemetry-supervised variant wins. The asymmetry is the empirical motivation for the contribution.

See `ANALYSIS_REPORT.md` for the full interpretation, including how to read each metric.

## How to read the metrics (quick reference)

- **Chamfer-L1 / L2** — average distance between the two point clouds. **Lower is better.** L1 is robust; L2 punishes outliers. Units = reconstruction units (meters for metric models). Use only to compare predictions to a single ground-truth point cloud, or two predictions against each other.
- **F-score @ τ** — fraction of predicted points within τ of GT (and vice versa), F1-combined. **Higher is better, in [0, 1].** τ = {0.01, 0.05, 0.1} m. F@0.05 ≈ "within 5 cm" is the most-cited threshold.
- **Accuracy** — average distance from each predicted point to nearest GT point. **Lower = the prediction is clean.**
- **Completeness** — average distance from each GT point to nearest predicted point. **Lower = the prediction covers the scene.**
- **ATE (Absolute Trajectory Error) RMSE** — after Sim(3)-Umeyama alignment of predicted camera centers to GT, RMS distance per camera. **Lower is better.** Best single number for "are the camera poses correct overall?".
- **RPE translation / rotation** — error per consecutive pose pair. **Lower is better.** Decouples local motion error from global drift; useful for streaming/online models.
- **Sim(3) scale** — the uniform scale factor that aligns prediction to GT. ≈1 means the model is metric; large values (e.g. 28.95 for DUSt3R) mean the model is arbitrary-unit and a single scale absorbs all the magnitude error.

## Repository layout

```
PLAN.md                research plan (read first)
CAPTURE_PROTOCOL.md    instructions for the field operator with the drone
ANALYSIS_REPORT.md     deep interpretation of every result produced so far
KNOWN_ISSUES.md        upstream model quirks + workarounds
SETUP.md               environment install
src/drone3d/           sys.path injector for the colliding CroCo backbones
scripts/               reconstruct (product pipeline), 00_smoke_test, 01_baseline_inference, 02_compare, 03_evaluate, util_colmap_to_npz
checkpoints/           ~25 GB across 7 foundation-model weights (gitignored)
third_party/           cloned foundation-model repos (gitignored)
data/                  test_real (NLE tower), aerial_finearts, aerial_mall, eth3d, eth3d_courtyard
outputs/               per-model reconstructions + comparison stats + eval JSON
```

GitHub: `sank6112/drone3d`.

## Next steps

0. **Product Track (ship first):** finish MASt3R checkpoint re-download, run `scripts/reconstruct.py` end-to-end on the first real frames/video, then package it as a callable component for the SaaS backend. Telemetry fusion plugs into the marked hook in `reconstruct_frames()`.
1. **Phase 5 pipeline build-out** (now unblocked since the drone is being set up locally with telemetry logging):
   - `scripts/extract_frames.py` — FPS sampling, Laplacian-blur filter, SSIM duplicate filter.
   - `scripts/parse_telemetry.py` — `.bin` (ArduPilot via `pymavlink`) / `.ulg` (PX4 via `pyulog`) → per-frame CSV of (lat, lon, alt, roll, pitch, yaw, ax/ay/az, gx/gy/gz).
   - `scripts/sync_video_telemetry.py` — align video frames to telemetry rows using the chosen sync method.
2. **First test flight** following `CAPTURE_PROTOCOL.md` (one ~90 s orbital pattern, one full Pixhawk log). Goal: validate sync to <100 ms before committing to the telemetry loss in Phase 6.
3. **Cloud GPU** (~1 A100 hour, ~$3) — run VGGT, CUT3R-512, StreamVGGT to complete the baseline matrix.
4. **ETH3D laser-scan GT** (`multi_view_training_dslr_scan_eval.7z`, ~5 GB) — turns pose-only eval into full Chamfer/F-score numbers for Table 1.
5. **Public aerial dataset ingest** — ClaraVid, UrbanScene3D LiDAR, DroneSplat sequences via the standard loader interface (Phase 2 finish).
6. **Phase 6 losses** — only after Phase 5 sync is validated.
