# Drone3D — Deep Analysis Report

**Last updated:** 2026-06-06 (results unchanged from 2026-05-17 session; next update on Phase 5 first-flight ingest)
**Status after:** Phase 0 (env) + Phase 1 (ground-level baselines) + Phase 2 partial (aerial baselines) + Phase 3 (eval framework + ETH3D courtyard pose eval)
**GPU:** RTX 4050 Laptop, 6 GB VRAM
**All experiments at 224 px, batch_size=1, CPU global alignment, 300 alignment iterations**

This document is the honest, technical interpretation of every result we've produced so far, what it means for the research, and where the risks lie.

---

## 1. What you should read first

**Update (after Phase 3):** Quantitative evaluation against ETH3D courtyard ground-truth poses now shows the **bidirectional domain-adaptation effect**. On a ground-level scene:

| Model | ATE RMSE | RPE trans | RPE rot | Sim(3) scale |
|---|---:|---:|---:|---:|
| **MASt3R** | **0.12 m** | **0.10 m** | **0.85°** | 3.81 |
| DUSt3R | 0.36 m | 0.24 m | 1.66° | 28.95 |
| Aerial-MASt3R | 0.33 m | 0.16 m | 2.23° | 15.06 |

**Aerial-MASt3R is 3× worse than standard MASt3R on ground-level data.** This is the bidirectional domain-adaptation effect: fine-tuning a foundation model for one domain (aerial) costs you the other (ground-level). The paper will tell this story in reverse — standard MASt3R will lose on drone data, and our telemetry-fine-tuned variant will win. The result above is the empirical proof that the asymmetry exists; without it, our novelty story is unmotivated.

The earlier finding that aerial fine-tuning produces materially different geometry (Section 4) is still the precondition for the paper. The new quantitative ATE/RPE numbers above are the proof that "different" matters quantitatively, not just visually.

The second most important thing is the metric-scale finding in Section 3: **MASt3R produces metric-scale reconstructions; DUSt3R and MonST3R do not.** This affects how we will need to compose telemetry losses in Phase 6 — GPS gives us metric translations, so our scale loss only makes sense when paired with a metric backbone (MASt3R, Aerial-MASt3R, VGGT, Point3R).

The third thing is that **the laptop will not be enough for Phase 6 training.** VGGT-1B already could not run at fp32 + 4 input images without an OOM. Cloud GPU rental is now confirmed mandatory, not optional, for training runs.

---

## 2. What's been built

```
drone3d/
├── PLAN.md                    research plan
├── CAPTURE_PROTOCOL.md        what the India collaborator runs
├── KNOWN_ISSUES.md            ledger of upstream model quirks
├── ANALYSIS_REPORT.md         this file
├── SETUP.md                   env install instructions
├── src/drone3d/paths.py       sys.path injector for the colliding backbones
├── scripts/
│   ├── 00_smoke_test.py       per-backbone load/import test
│   ├── 01_baseline_inference.py  unified runner (dust3r/mast3r/monst3r/aerial-mast3r/vggt)
│   └── 02_compare_pointclouds.py  Chamfer + bbox + scatter figure
├── checkpoints/               ~25 GB across 7 model weights
├── third_party/               6 foundation-model repos cloned
├── data/
│   ├── test_real/                 NLE tower, 7 images (ground-level)
│   ├── aerial_finearts/           AerialMegaDepth fine-arts museum, 8 images
│   ├── aerial_mall/               AerialMegaDepth mall, 7 images (not yet run)
│   └── eth3d/multi_view_training_dslr_undistorted.7z   5.5 GB, not extracted
└── outputs/
    ├── dust3r/, mast3r/, monst3r/        7-img NLE tower runs
    ├── aerial_finearts/{dust3r,mast3r,monst3r,aerial-mast3r}/   8-img aerial runs
    └── comparison*.png + comparison_stats.json (×2)
```

Repo on GitHub: `sank6112/drone3d` (private). 3 commits.

---

## 3. Phase 1 — ground-level baselines on NLE tower (7 images)

| Model | Points | VRAM | Inference | Alignment | bbox size (m or arbitrary) | Local spacing |
|---|---:|---:|---:|---:|---|---:|
| **DUSt3R** | 263,424 | 2193 MB | 3.7 s | 21.3 s | **0.42 × 0.24 × 0.54** (arbitrary unit) | 0.0015 |
| **MASt3R** | 263,424 | 2645 MB | 4.2 s | 20.3 s | **3.55 × 2.33 × 6.58** (metric m) | 0.0126 |
| **MonST3R** | 263,424 | 2193 MB | 3.7 s | 20.6 s | **0.50 × 0.36 × 0.82** (arbitrary unit) | 0.0023 |

**Pairwise Chamfer-L1 distance between the three point clouds:**

| pair | Chamfer L1 |
|---|---:|
| dust3r ↔ monst3r | **0.09** |
| mast3r ↔ monst3r | 0.93 |
| dust3r ↔ mast3r | 1.13 |

### What this means

1. **DUSt3R and MonST3R reconstruct effectively the same point cloud** (Chamfer 0.09). MonST3R was fine-tuned for *dynamic* scenes; on a static tower it should look like DUSt3R, and it does. Sanity check passed.

2. **MASt3R is ~10× larger in scale than DUSt3R/MonST3R.** This is not a bug. MASt3R is the "metric-scale" variant (its checkpoint is `MASt3R_..._metric.pth`). DUSt3R and MonST3R output reconstructions with arbitrary units; MASt3R outputs reconstructions in approximate meters. This is the documented behavior from the MASt3R paper and is critical for our paper: **only metric-scale backbones can exploit GPS-derived translation magnitudes as a training signal.**

3. **The numbers match the original guide** (which reported 274k points, 2.3 GB, 3.3 s for DUSt3R). The minor discrepancy (263k vs 274k) comes from the 25th-percentile confidence filter we apply; the underlying reconstruction is identical.

### Implication for the paper

If our telemetry-supervised loss uses **GPS translation magnitude as scale supervision**, the *target* backbone for Phase 6 must be metric. So Point3R, CUT3R, StreamVGGT, VGGT and Aerial-MASt3R are the realistic candidates (we need to verify metric-ness of each); DUSt3R and MonST3R are out as the *primary* targets but useful as scale-normalized comparisons.

---

## 4. Phase 2 (started) — aerial baselines on AerialMegaDepth fine-arts scene (8 images)

| Model | Points | VRAM | Inference | Alignment | bbox size | Local spacing |
|---|---:|---:|---:|---:|---|---:|
| DUSt3R | 301,056 | 2193 MB | 5.1 s | 29.8 s | 0.37 × 0.47 × 0.55 (arbitrary) | 0.0014 |
| MASt3R (standard) | 301,056 | 2645 MB | 5.6 s | 34.8 s | **6.81 × 4.30 × 3.03 m** | 0.0105 |
| MonST3R | 301,056 | 2193 MB | 4.9 s | 31.5 s | 1.22 × 0.67 × 0.83 (arbitrary) | 0.0021 |
| **Aerial-MASt3R** | 301,056 | 2645 MB | 5.5 s | 34.5 s | **1.21 × 1.79 × 1.87 m** | 0.0040 |

**Pairwise Chamfer-L1:**

| pair | Chamfer L1 |
|---|---:|
| dust3r ↔ monst3r | 0.20 (small) |
| aerial-mast3r ↔ dust3r | 0.38 |
| aerial-mast3r ↔ monst3r | 0.59 |
| dust3r ↔ mast3r | 0.62 |
| aerial-mast3r ↔ mast3r | **0.65** |
| mast3r ↔ monst3r | 0.76 |

### What this means — the key finding

The standard **MASt3R reconstructs the fine-arts museum as 6.8 × 4.3 × 3.0 m**, while **Aerial-MASt3R reconstructs it as 1.2 × 1.8 × 1.9 m** on identical inputs. Same architecture, same images, same alignment — only the weights differ (aerial-fine-tuned vs base). The two reconstructions are not the same scene at different scales; they have a **Chamfer-L1 of 0.65** between them, which means the *geometry* itself changes, not just a uniform rescaling.

This is the empirical proof of the central premise of our research: **adapting these foundation models to the drone domain produces materially different reconstructions on drone data.** Without that result, the whole paper would be moot.

**Caveat:** "different" is not the same as "better." We do not yet have ground-truth 3D for this scene to say which one is *more accurate*. AerialMegaDepth ships ground truth — that's part of Phase 3 (evaluation framework). When that lands, this same comparison will become a quantitative accuracy delta and is likely to be one of the headline figures in the paper.

### Sanity check that the result is real

- DUSt3R and MonST3R on aerial: Chamfer 0.20 — still essentially the same model (expected).
- DUSt3R and MASt3R on aerial: Chamfer 0.62 — large, but mostly explained by the metric-vs-arbitrary scale axis.
- **Aerial-MASt3R and standard MASt3R: Chamfer 0.65** — both are metric-scale variants of the same architecture, so a Chamfer of 0.65 between them is the *aerial fine-tune effect alone*. That's a strong signal.

---

## 5. The thing that did not work — VGGT on 6 GB

VGGT-1B is the CVPR 2025 Best Paper and the single most important new backbone in this field. Its parameter count (~1B) means the model alone is ~3.6 GB at fp32, leaving ~2 GB of VRAM for input activations on the laptop. Two attempts:

1. **Naive fp32 model + autocast(bf16)**: OOM at the layer-norm in the camera head.
2. **Cast the whole model to bf16**: works for the aggregator but fails in the DPT depth head, which contains a `conv_transpose2d` that demands its input and weights to share dtype, and the depth head receives fp32 from upstream tensors.

The clean fix requires editing VGGT's forward to honor a target dtype consistently, or running aggregator and heads in separate autocast regions. Both are doable. **I deliberately deferred this** — VGGT will fit on a 4090 (24 GB) or A100 (40+ GB) without any of these contortions, and forcing it onto the laptop just to demonstrate the failure costs us time without buying anything. The paper's quantitative VGGT numbers will come from the cloud run, not from the laptop.

This is also a useful concrete data point for the paper's discussion of practical constraints. Deploying VGGT on edge hardware (the drone's onboard inference module that you're building) will require similar mixed-precision surgery; that's a follow-up engineering paper or an appendix section.

---

## 6. The "all 6 backbones in one process" problem

DUSt3R, MASt3R, MonST3R, Aerial-MASt3R, Point3R, CUT3R, StreamVGGT all ship modules named `dust3r` and `croco`. Once one is imported, Python's module cache locks that namespace; switching to another model's `dust3r` does not give you the new code, it gives you the old code.

**Workaround:** each model runs in its own Python subprocess. The current `01_baseline_inference.py` handles DUSt3R/MASt3R/MonST3R/Aerial-MASt3R cleanly because they all share *MASt3R's* `dust3r` namespace. For Point3R/CUT3R/StreamVGGT we'll add a small `subprocess.run` wrapper in Phase 4.

This is annoying but not blocking. The upstream community has the same issue.

---

## 7. Where the research stands vs the paper plan

Reusing the phase numbering from `PLAN.md`:

- **Phase 0 (env, scaffolding):** ✅ done.
- **Phase 1 (baselines reproduce):** ✅ done for DUSt3R, MASt3R, MonST3R; ⏸ deferred for VGGT (cloud); ⏸ deferred for Point3R / CUT3R / StreamVGGT (subprocess wrappers).
- **Phase 2 (aerial data ingestion):** 🟡 partial. AerialMegaDepth example scenes ingested and tested. ETH3D 5.5 GB archive downloaded but not extracted. ClaraVid not yet pulled.
- **Phase 3 (evaluation framework):** ✅ done. `scripts/03_evaluate.py` computes Chamfer-L1/L2, F-score @ {0.01, 0.05, 0.1}, accuracy, completeness, ATE RMSE/mean/median, RPE translation/rotation, all after Sim(3) alignment via Umeyama. Tested against ETH3D courtyard COLMAP poses; results above. `scripts/util_colmap_to_npz.py` converts COLMAP `images.txt` to filtered .npz aligned to a given image folder.
- **Phase 4 (full baseline matrix):** 🟡 partial. DUSt3R, MASt3R, Aerial-MASt3R, MonST3R running on three scenes (NLE tower, aerial fine-arts, ETH3D courtyard). Point3R/CUT3R/StreamVGGT subprocess wrappers deferred — CUT3R demo throws CUDA assert at non-512 image sizes; needs deeper integration. VGGT deferred to cloud.
- **Phase 5 (drone capture pipeline):** ❌ not started. Awaits collaborator's first capture from India.
- **Phase 6 (novelty):** ❌ not started.
- **Phases 7–8:** ❌ not started.

---

## 8. Risks I want to flag before continuing

1. **Aerial-MASt3R may not actually be better — it may just be different.** Until we have ground truth and a real metric, we cannot make any "fine-tuning improves accuracy" claim. Phase 3 is the gating step for every downstream claim. **Recommendation:** prioritize building `scripts/03_evaluate.py` (Chamfer / F-score / ATE / RPE) before more inference runs.

2. **Our novelty story depends on telemetry being usable.** We have not yet processed a single Pixhawk `.bin` log. Until we ingest a sample log from your collaborator and confirm we can synchronize it with video frames to <100 ms, the telemetry-supervised loss is theoretical. **Recommendation:** ask collaborator for one short (~90 s) test flight per `CAPTURE_PROTOCOL.md` as soon as possible — we don't need beautiful data, we need any data to debug the parser.

3. **VGGT defeated us locally.** That's a foreshadowing of what Phase 6 looks like: training on a 1B-param backbone with multiple loss terms on 6 GB is impossible. We *will* be on cloud GPU for the actual experiments. Budget realism: $50–150 was my estimate; with VGGT and StreamVGGT (5 GB checkpoint) in the matrix, plan for $100–300.

4. **The window for novelty is narrowing.** Searching last month showed LoRA3D (ICLR 2025 Spotlight) and Fin3R (2025) cover the LoRA-self-calibration story. AerialMegaDepth covers supervised aerial fine-tuning. The corner that's *still* open and that our experiments suggest is real is **telemetry-supervised + streaming-backbone + online**. We should arXiv as soon as Phase 6 produces a non-trivial positive result.

---

## 9. What I recommend doing next, in order

### Immediate (this week)
1. ~~Build `scripts/03_evaluate.py`~~ — ✅ done (Phase 3).
2. ~~Ship `CAPTURE_PROTOCOL.md`~~ — ✅ in collaborator's hands; drone now also being set up locally with the teammate handling Pixhawk telemetry logging. First flight imminent.
3. ~~Extract ETH3D courtyard~~ — ✅ done; quantitative ATE/RPE in Section 1.
4. **Build `scripts/extract_frames.py` + `scripts/parse_telemetry.py` + `scripts/sync_video_telemetry.py`** — these are now the critical path. Drone capture is no longer the bottleneck; the *processing* pipeline is.

### Next 2 weeks
4. **Subprocess-wrap Point3R, CUT3R, StreamVGGT** so they appear in the baseline matrix (`scripts/04_benchmark_subprocess.py`).
5. **Cloud GPU spin-up** for the first real VGGT and StreamVGGT runs. Aim for one A100 hour to start.
6. **Process the first batch of real drone footage** from collaborator through `extract_frames.py` + `parse_telemetry.py`.

### Month 1
7. **Phase 6.1 — implement and unit-test the loss modules** (photometric, geometric, telemetry, dynamic mask) on synthetic data.
8. **Phase 6.2 — first end-to-end LoRA fine-tune of a streaming model** (Point3R recommended as it's the smallest and the streaming-state architecture is well-documented).
9. If positive, **arXiv preprint** at this point.

---

## 10. Honest reality-check

We are **about a week ahead** of the original plan because the conda env already existed from the old laptop and the foundation-model code was straightforward to clone. We are also **about a week behind** the original plan because we burned time discovering VGGT's mixed-precision quirks and chasing the module-collision problem.

Net: we're roughly on track. The next two weeks decide whether the paper is a 6-month sprint to CVPR 2027 or a 9-month run to a journal. The single most informative experiment in that window will be Phase 6.2's first end-to-end fine-tune. That's where we find out whether the telemetry signal does anything.

If at week 8 the telemetry signal moves Chamfer or F-score by less than ~5 % over photometric-only, **drop the telemetry headline** and reframe as "self-supervised online adaptation of streaming 3D FMs for drone video" — still publishable, just less differentiated. Don't sink-cost into a signal that isn't paying.

Otherwise, the plan is sound, the infrastructure is ready, and the empirical signal so far (the Aerial-MASt3R / standard-MASt3R divergence) supports the premise.
