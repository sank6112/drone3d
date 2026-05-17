#!/usr/bin/env python3
"""
Phase 1 test: Load DUSt3R on 2 synthetic 224px images, run inference, print VRAM usage.
Tuned for RTX 4050 Laptop GPU (5.6 GB VRAM): batch_size=1, 224px resolution.
"""
import sys
import os
import numpy as np
from PIL import Image

PROJECT_DIR = os.path.expanduser("~/projects/drone3d")
MAST3R_DIR = os.path.join(PROJECT_DIR, "mast3r")
DUST3R_DIR = os.path.join(MAST3R_DIR, "dust3r")

sys.path.insert(0, MAST3R_DIR)
sys.path.insert(0, DUST3R_DIR)

print("=" * 55)
print("  DUSt3R Phase 1 Test  |  224px  |  batch_size=1")
print("=" * 55)

# --- 1. GPU check ---
import torch

print("\n[1/5] GPU info")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory / 1024**3          # total_memory (not total_mem)
    print(f"  GPU  : {props.name}")
    print(f"  VRAM : {total_vram:.2f} GB")
    DEVICE = "cuda"
else:
    print("  No GPU — using CPU (will be slow)")
    DEVICE = "cpu"
    total_vram = 0

# --- 2. Imports ---
print("\n[2/5] Importing DUSt3R modules")
try:
    from dust3r.model import AsymmetricCroCo3DStereo
    from dust3r.inference import inference
    from dust3r.utils.image import load_images
    from dust3r.image_pairs import make_pairs
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
    print("  OK — all DUSt3R modules imported")
except ImportError as e:
    print(f"  FAIL: {e}")
    sys.exit(1)

# --- 3. Create synthetic test images ---
print("\n[3/5] Creating synthetic test images (224 x 224)")
test_dir = os.path.join(PROJECT_DIR, "data", "test_synthetic")
os.makedirs(test_dir, exist_ok=True)

img_a = np.zeros((224, 224, 3), dtype=np.uint8)
img_a[30:100, 30:120]  = [200, 60, 60]   # red rect
img_a[80:160, 90:190]  = [60, 200, 60]   # green rect
img_a[120:200, 50:150] = [60, 60, 200]   # blue rect

# Slight lateral shift to simulate a second camera viewpoint
img_b = np.zeros((224, 224, 3), dtype=np.uint8)
img_b[30:100, 40:130]  = [200, 60, 60]
img_b[80:160, 100:200] = [60, 200, 60]
img_b[120:200, 60:160] = [60, 60, 200]

path_a = os.path.join(test_dir, "img_A.jpg")
path_b = os.path.join(test_dir, "img_B.jpg")
Image.fromarray(img_a).save(path_a)
Image.fromarray(img_b).save(path_b)
print(f"  Saved: {path_a}")
print(f"  Saved: {path_b}")

# --- 4. Load model ---
ckpt = os.path.join(PROJECT_DIR, "checkpoints", "DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
if not os.path.exists(ckpt):
    print(f"\n  FAIL: checkpoint not found at {ckpt}")
    sys.exit(1)

print(f"\n[4/5] Loading DUSt3R from checkpoint")
print(f"  Path: {ckpt}")

torch.cuda.reset_peak_memory_stats()
model = AsymmetricCroCo3DStereo.from_pretrained(ckpt).to(DEVICE)
model.eval()

if DEVICE == "cuda":
    after_load = torch.cuda.memory_allocated() / 1024**3
    print(f"  VRAM after model load : {after_load:.2f} GB / {total_vram:.2f} GB")

# --- 5. Inference ---
print("\n[5/5] Running inference  (batch_size=1, image_size=224)")
images = load_images([path_a, path_b], size=224, verbose=False)
pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
print(f"  Pairs to process: {len(pairs)}")

with torch.no_grad():
    output = inference(pairs, model, DEVICE, batch_size=1, verbose=True)

# PairViewer mode: lightweight, no iterative optimisation — safe on 6 GB
scene = global_aligner(output, device=DEVICE, mode=GlobalAlignerMode.PairViewer)

pts3d = scene.get_pts3d()
print(f"\n  3D pointmaps shape: {[p.shape for p in pts3d]}")

# --- VRAM summary ---
if DEVICE == "cuda":
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    cur_vram  = torch.cuda.memory_allocated() / 1024**3
    print(f"\n  Peak VRAM usage    : {peak_vram:.2f} GB")
    print(f"  Current VRAM usage : {cur_vram:.2f} GB")
    print(f"  Total VRAM         : {total_vram:.2f} GB")
    print(f"  Headroom remaining : {total_vram - peak_vram:.2f} GB")

print("\n" + "=" * 55)
print("  PASSED — DUSt3R inference completed successfully")
print("=" * 55)
