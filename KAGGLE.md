# Running drone3d on Kaggle

Goal: use Kaggle's free **16 GB** GPU to push reconstruction quality well past the 6 GB laptop, train on the datasets we have, and measure which techniques help — before renting an A100.

## One-time Kaggle setup
1. Create a new **Notebook** at kaggle.com → **File → Import Notebook** and upload `notebooks/drone3d_kaggle.ipynb`
   (or make a blank notebook and paste the cells).
2. In the right-hand **Settings** panel:
   - **Accelerator → GPU P100** (single 16 GB; simplest). `GPU T4 x2` also works but needs multi-GPU flags.
   - **Internet → ON** (required — the notebook downloads code, checkpoints, and data).
   - **Persistence → Variables and Files** (optional, keeps `/kaggle/working` between sessions).
3. Run cells top to bottom. For long jobs use **Save Version → Save & Run All (Commit)** — it runs up to **9 h** in the background after you close the tab.

## What the notebook does
| Step | Action |
|---|---|
| 1–2 | Clone `sank6112/drone3d` (branch `kaggle-setup`) and run `scripts/setup_kaggle.sh` (installs gsplat + MASt3R, downloads checkpoints from HuggingFace/Naver, DroneSplat via gdown, Rubble via wget). |
| 3 | **Ablation** on Sculpture (7k @ factor 2): baseline → +antialiased → +app_opt → +bilateral → +pose_opt, with a PSNR/SSIM/LPIPS table. |
| 4 | **Final** full-res Sculpture (factor 1, 30k, 2.5 M Gaussians) with the winning knobs. |
| 5 | **Product path**: MASt3R poses from raw images (20 frames — Kaggle RAM allows more than the laptop) → splat with `--pose_opt`. |
| 6 | **Rubble** big aerial scene (set `RUBBLE=1` in step 2 first). |
| 7 | Copies `.splat`s to `/kaggle/working/deliverables/` + writes `outputs/SUMMARY.md`. |

Download the `.splat` files from the **Output** panel and drag them into <https://supersplat.com> to fly through.

## Quality knobs (why each matters for drone footage)
- `--antialiased` — alias-free rendering across the huge depth range of aerial views.
- `--app_opt` — per-image appearance embedding; absorbs **auto-exposure / white-balance drift** across a flight (a main cause of the muddy look).
- `--use_bilateral_grid` — photometric harmonization across views (experimental; the ablation tells us if it helps here).
- `--pose_opt` — refines camera poses during training; biggest win for the **MASt3R** path.
- 16 GB enables `--data_factor 1` (full res) and `--strategy.cap-max 2–3 M` Gaussians.

## Quota / budget
- **30 h GPU / week**, sessions up to 9 h. The ablation (~5×8 min) + a couple of 30k runs (~30–60 min each) fit comfortably in a few hours.
- Fresh sessions re-download data (~15 min). To avoid that, later save the downloads as a private **Kaggle Dataset** and mount it read-only (needs your `kaggle.json`).

## Next: A100
Once the config is dialed in here, the same `scripts/train_quality.sh` runs on an A100 with more frames, higher caps, and (stretch) Scaffold-GS / Octree-GS for large aerial scenes.
