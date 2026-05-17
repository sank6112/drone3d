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

Cloned into `third_party/` (gitignored). Each is a separate upstream repo; install each in editable mode so we can patch and trace.

```bash
mkdir -p third_party && cd third_party

# DUSt3R + MASt3R (MASt3R has DUSt3R as a submodule)
git clone --recursive https://github.com/naver/mast3r.git
pip install -e mast3r

# VGGT (CVPR 2025 Best Paper)
git clone https://github.com/facebookresearch/vggt.git
pip install -e vggt

# StreamVGGT (ICLR 2026)
git clone https://github.com/wzzheng/StreamVGGT.git
pip install -e StreamVGGT

# Point3R (NeurIPS 2025)
git clone https://github.com/YkiWu/Point3R.git
pip install -e Point3R

# CUT3R (CVPR 2025)
git clone https://github.com/CUT3R/CUT3R.git || git clone https://github.com/cvg/cut3r.git
# (URL may shift; check the awesome-DUST3R list if cloning fails)

# MonST3R — only needed for dynamic-scene baseline
git clone https://github.com/junyi42/monst3r.git
pip install -e monst3r

cd ..
```

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
