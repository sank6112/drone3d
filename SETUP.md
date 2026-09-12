# Setup

## 1. Conda environment

```bash
conda env create -f environment.yml
conda activate drone3d
```

Verify GPU is visible:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Foundation model repos

Cloned into `third_party/` (gitignored). **Only VGGT is pip-installable.** The CroCo-family repos (mast3r, StreamVGGT, Point3R, monst3r, cut3r) ship no `setup.py` and are used as plain folders via `src/drone3d/paths.py` — do NOT `pip install -e` them (it errors; see KNOWN_ISSUES.md).

```bash
mkdir -p third_party && cd third_party

# DUSt3R + MASt3R (MASt3R has DUSt3R as a submodule) — used via paths.py, no pip install
git clone --recursive https://github.com/naver/mast3r.git

# VGGT (CVPR 2025 Best Paper) — the one repo that IS pip-installable
git clone https://github.com/facebookresearch/vggt.git
pip install -e vggt

# StreamVGGT (ICLR 2026) — used via paths.py
git clone https://github.com/wzzheng/StreamVGGT.git

# Point3R (NeurIPS 2025) — used via paths.py
git clone https://github.com/YkiWu/Point3R.git

# CUT3R (CVPR 2025) — used via paths.py
git clone https://github.com/CUT3R/CUT3R.git || git clone https://github.com/cvg/cut3r.git
# (URL may shift; check the awesome-DUST3R list if cloning fails)

# MonST3R — dynamic-scene baseline only — used via paths.py
git clone https://github.com/junyi42/monst3r.git

cd ..
```

> **Minimal deployable stack:** for the Product Track (`scripts/reconstruct.py`) you only need `mast3r` cloned + the DUSt3R and MASt3R-metric checkpoints below. The rest are for the research baseline matrix.

## 3. Checkpoints

```bash
mkdir -p checkpoints && cd checkpoints

# DUSt3R 512
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# MASt3R 512 metric
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

# VGGT — pulled via huggingface_hub at runtime (see scripts/00_smoke_test.py)

cd ..
```

## 4. Smoke test

```bash
python scripts/00_smoke_test.py
```

Should print, for each of DUSt3R/MASt3R/VGGT: load time, VRAM after load, a 1-image dummy forward (if it fits in 6GB), and "OK". If any model fails to load on the 6GB laptop, the smoke test logs that — the cloud node will run the failing ones.

## 5. Run baselines on the existing test images

```bash
python scripts/01_baseline_inference.py --model dust3r --input data/test_real/
python scripts/01_baseline_inference.py --model mast3r --input data/test_real/
python scripts/01_baseline_inference.py --model vggt   --input data/test_real/
```

(Scripts will be added in Phase 1.)
