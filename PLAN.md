# Drone3D — Research Plan

Supersedes `Resources and documentation/Drone3D_Research_Execution_Guide.pdf`. The PDF is kept as historical context only.

---

## Thesis

**Online, telemetry-supervised adaptation of streaming 3D foundation models for drone video.**

We adapt a pretrained streaming 3D reconstruction model (Point3R / CUT3R / StreamVGGT) to drone imagery using a combination of:
- Photometric multi-view consistency (no GT depth/pose needed)
- Geometric forward–backward depth consistency
- **Drone telemetry as weak metric supervision** — GPS, IMU, barometric altitude from Pixhawk logs
- Dynamic-object masking (DroneSplat-style)

Adaptation runs **during a flight**, not as an offline pre-training pass.

## Why this is publishable (and the existing prior art is not the same paper)

| Prior work | What they did | What we do differently |
|---|---|---|
| LoRA3D (ICLR 2025) | Offline LoRA self-calibration on Replica/TUM/Waymo, photometric pseudo-labels only | Online, drone domain, telemetry-fused, streaming backbone |
| Fin3R (2025) | LoRA + monocular distillation, offline | Online adaptation, telemetry signal |
| AerialMegaDepth (CVPR 2025) | Supervised aerial fine-tuning of DUSt3R using Google Earth pseudo-GT | Self-supervised, no external pseudo-GT, uses drone's own telemetry |
| MonST3R / St4RTrack | Self-supervised on dynamic video, generic | Drone-specific signals (GPS scale, IMU rotation) |
| DroneSplat (CVPR 2025) | 3DGS for drone with dynamic masking | Same masking idea applied inside a foundation-model adaptation loss |

The contribution is **the training signal + when it runs + which backbone**. LoRA is the parameter-efficient training mechanism; we will ablate it against adapter layers and decoder-only fine-tuning.

---

## Two tracks: Product and Research

This project now runs on **two parallel tracks**. They share the same codebase and backbones but have different deadlines and success criteria.

### Product Track (near-term deliverable — ship first)

A **deployable pipeline** that takes a drone flight **video** and produces a **3D reconstruction** (point cloud / mesh) as a **single entrypoint** (`scripts/reconstruct.py`). This is the promised deliverable and is one module of a larger **SaaS** that also bundles other drone applications (person-following, video recording, etc.), so it must be **importable/callable as a component**, not just a research script.

- **Success criterion:** `python scripts/reconstruct.py --video flight.mp4 --out outputs/run1` runs end to end and writes a colored `.ply` + camera poses, on the local 6 GB card, with no manual frame prep.
- **Backbone:** MASt3R (metric, fits 6 GB). Telemetry/streaming novelty is *not* required for the product path.
- **Stages:** video → frame extraction (FPS sample + Laplacian-variance blur filter + SSIM duplicate filter) → MASt3R inference + global alignment → point-cloud/mesh export → (optional) telemetry alignment hook for later.
- **On-drone note:** the 6 GB laptop path is the reference implementation; the "placed on drone" target is an embedded/edge or ground-station deployment of the same entrypoint. Keep the CLI + a `reconstruct(video, out, ...)` function so it can be wrapped by the SaaS backend.

### Research Track (the paper — layers on top)

Everything below (Phases 0–8) — telemetry-supervised online adaptation on a streaming backbone. This is unblocked the moment drone capture data (video + Pixhawk `.bin`/`.ulg`) arrives from the collaborator, and reuses the Product Track's frame-extraction + reconstruction code.

**Sequencing:** ship the Product Track reconstruction now with public/existing images and any first drone video; keep building the Research Track (losses, telemetry, ablations) in parallel and fold results back into both the paper and the SaaS.

## Target venue and timeline

We are not locked to a 6-month sprint. Two tracks, decided at week ~16 based on results:

- **Track A — Conference (preferred if competitive at week 16):** CVPR 2027 (~Nov 2026 deadline) main conference. ~6 month total runway.
- **Track B — Journal (if results are exceptional or we want to extend scope):** IJCV / ISPRS J. Photogrammetry, ~9 month runway. Allows deeper ablations, more datasets, a hardware-deployment section tied to the encoder module.
- **Floor (guaranteed):** CVPR/ICCV workshop on UAVs / EarthVision — submission window in early 2027.
- **arXiv preprint:** posted as soon as Phase 6 has any non-trivial positive result, to establish priority against scoop risk.

## Phased plan

### Phase 0 — Project setup *(week 1)*
- Clean project layout under `/home/rudra/projects/drone3d/` (renamed from `Drone 3d` to drop the space; restored from GitHub 2026-09-12 after local loss).
- Conda env `drone3d` (Python 3.11, PyTorch 2.5+cu121).
- Clone & install: DUSt3R, MASt3R, VGGT, StreamVGGT, Point3R, CUT3R, MonST3R, DroneSplat (as reference).
- Pin all dependency versions in `environment.yml` and `requirements.txt`.
- Deliverable: `python -c "import dust3r, mast3r, vggt, point3r"` all succeed.

### Phase 1 — Baseline reproduction *(week 1–2)*
- Run DUSt3R, MASt3R, **VGGT** on existing 7 NLE tower images.
- Visual + quantitative comparison.
- Deliverable: 3 point clouds + a comparison figure + a notes file on per-model VRAM/time.

### Phase 2 — Public aerial data ingestion *(week 2)*
- Download: **ClaraVid (primary)**, AerialMegaDepth, ETH3D outdoor (~25 scenes), UrbanScene3D (LiDAR GT), DroneSplat sequences.
- Standard loader interface — every dataset exposes `(images, intrinsics?, gt_pointcloud?, gt_poses?, gt_depths?)`.
- Deliverable: `data/` populated; one notebook iterating a sample from each.

### Phase 3 — Evaluation framework *(week 2–3)*
- `scripts/evaluate.py` — Chamfer (L1/L2), F-score @ {0.01, 0.05, 0.1}, Accuracy, Completeness.
- Pose metrics: ATE (Procrustes-aligned), RPE for consecutive frames.
- **New:** telemetry-vs-predicted-pose comparison (rotation error in deg, translation error in m).
- Saves JSON + markdown table per run.
- Deliverable: runs against any (pred, gt) pair; ablations later read from these JSONs.

### Phase 4 — Baseline comparison study *(week 3)*
- Matrix: {DUSt3R, MASt3R, MonST3R, VGGT, StreamVGGT, Point3R, CUT3R} × {ClaraVid, AerialMegaDepth, ETH3D, UrbanScene3D}.
- Produces the paper's Table 1.
- Deliverable: `outputs/benchmark/results.csv` + bar charts.

### Phase 5 — Drone capture pipeline *(week 3–4, parallel with collaborator capturing)*
- See `CAPTURE_PROTOCOL.md` — ship to collaborator now so capture starts week 1.
- `scripts/extract_frames.py` — FPS sampling, blur filter (Laplacian variance), duplicate filter (SSIM).
- `scripts/parse_telemetry.py` — `.bin`/`.ulg` → frame-aligned (lat, lon, alt, roll, pitch, yaw, ax, ay, az, gx, gy, gz) at frame timestamps. Use `pymavlink` for ArduPilot, `pyulog` for PX4.
- `scripts/sync_video_telemetry.py` — align camera frames to telemetry via either (a) Pixhawk camera-trigger PWM event, (b) clap/LED sync at flight start, or (c) GPS-UTC timestamp matching.
- Deliverable: one short test flight from collaborator processed end-to-end.

### Phase 6 — Novelty: telemetry-supervised online adaptation *(week 4–8)*
1. `scripts/losses/photometric.py` — multi-scale SSIM + L1, auto-masked, confidence-weighted.
2. `scripts/losses/geometric.py` — forward–backward depth consistency.
3. `scripts/losses/telemetry.py` — **the new bit**:
   - **Scale loss**: predicted relative translation magnitude ↔ GPS-derived baseline.
   - **Rotation loss**: predicted relative rotation ↔ IMU-integrated rotation between frames.
   - **Altitude consistency**: predicted up-axis vs gravity from IMU.
4. `scripts/losses/dynamic_mask.py` — DroneSplat-style dynamic-region masking that gates photometric loss.
5. `scripts/finetune_online.py` — picks a streaming backbone (default Point3R), wraps with LoRA-rank-8 (or alternative: adapter / decoder-only — configurable), trains during sliding window over a flight.
6. Cloud run: A100 / 4090 rental for ablation cells.

### Phase 7 — Synthetic data + controlled experiments *(week 6–8, parallel)*
- Blender drone flights over procedural urban scenes → RGB + GT depth + GT poses + simulated noisy GPS/IMU.
- Used for: (a) clean controlled ablation of each loss term, (b) sanity-check telemetry noise robustness.

### Phase 8 — Ablations, paper, figures *(week 8–16 conference / week 8–28 journal)*
- Ablation matrix: each loss term × LoRA rank × frozen-vs-unfrozen × backbone choice.
- Failure-case analysis.
- Publication figures (300dpi PNG + vector PDF).
- arXiv preprint at the earliest credible result (week ~12–14).
- Conference track: LaTeX draft week 12–16, submit week 16.
- Journal track (optional extension): additional datasets, deployment section on the encoder/RC module, extra ablations, larger compute runs, weeks 16–36.

## Compute plan

- Local 6GB RTX 4050: inference, debugging, dry-run training (≤100 steps), eval.
- Cloud: 1–2× A100 or 4090 for actual training runs and ablations. Estimated total: $50–150.
- Cache all pretrained checkpoints locally to avoid re-download on cloud spin-ups.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Telemetry-image sync is poor → telemetry signal noisy → losses don't help | Phase 5 explicitly tests sync quality; if bad, fall back to Phase 6 without telemetry loss (still publishable as photometric + geometric + dynamic mask on streaming backbone) |
| Capture in India is delayed | Phases 1–4 + 6 + 7 work fully on public data; drone-specific experiments become one paper section |
| Another group publishes the same angle before us | Sub-direction A: focus on the encoder/RC-mounted real-time inference module (engineering paper, ICRA/IROS-style). Sub-direction B: emphasize the streaming backbone choice + ablate other backbones less explored. |
| 6GB VRAM blocks even debugging | Use Colab T4 (16GB free) for any local cloud run that exceeds RTX 4050 |

## Roles

- **User**: method design, loss formulation, paper writing, ablation design, cloud training runs.
- **Collaborator (India)**: drone capture (per `CAPTURE_PROTOCOL.md`), evaluation runs on captured data, assists with training scripts as a learning exercise. Initial concrete tasks: (1) run capture protocol once, (2) parse `.bin` logs into CSV using `pymavlink`, (3) verify visual/telemetry sync.
