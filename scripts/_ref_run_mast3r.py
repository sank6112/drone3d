#!/usr/bin/env python3
"""Run MASt3R on NLE tower images: GPU inference -> CPU global alignment -> PLY + poses."""

import sys
import os
import time
import tempfile

import numpy as np
import torch

# Path setup
PROJECT_ROOT = os.path.expanduser("~/projects/drone3d")
MAST3R_ROOT  = os.path.join(PROJECT_ROOT, "mast3r")
DUST3R_ROOT  = os.path.join(MAST3R_ROOT, "dust3r")

sys.path.insert(0, MAST3R_ROOT)
sys.path.insert(0, DUST3R_ROOT)

# ── constants ─────────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(PROJECT_ROOT, "checkpoints",
                          "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
IMAGE_DIR  = os.path.join(PROJECT_ROOT, "data", "test_real")
OUT_PLY    = os.path.join(PROJECT_ROOT, "outputs", "mast3r_tower.ply")
OUT_POSES  = os.path.join(PROJECT_ROOT, "outputs", "mast3r_tower_poses.npz")
IMG_SIZE   = 224
NITER      = 300   # global-alignment iterations


def save_ply(path, pts3d_list, colors_list):
    """Write a binary PLY from per-image (N,3) point/colour arrays."""
    pts  = np.concatenate([p.reshape(-1, 3) for p in pts3d_list], axis=0)
    cols = np.concatenate([c.reshape(-1, 3) for c in colors_list], axis=0)
    cols = (cols * 255).clip(0, 255).astype(np.uint8)

    n = len(pts)
    header = (
        f"ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        f"property float x\nproperty float y\nproperty float z\n"
        f"property uchar red\nproperty uchar green\nproperty uchar blue\n"
        f"end_header\n"
    ).encode()

    data = np.zeros(n, dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    data['x'], data['y'], data['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    data['red'], data['green'], data['blue'] = cols[:, 0], cols[:, 1], cols[:, 2]

    with open(path, 'wb') as f:
        f.write(header)
        f.write(data.tobytes())
    return n


def peak_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2
    return 0.0


def main():
    # ── device selection ─────────────────────────────────────────────────────
    if torch.cuda.is_available():
        inf_device = 'cuda'
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  total_memory={props.total_memory/1024**2:.0f} MB")
        torch.cuda.reset_peak_memory_stats()
    else:
        inf_device = 'cpu'
        print("No CUDA — running fully on CPU")

    # ── imports ───────────────────────────────────────────────────────────────
    from mast3r.model import AsymmetricMASt3R
    from mast3r.image_pairs import make_pairs
    from mast3r.cloud_opt.sparse_ga import (
        sparse_global_alignment,
        forward_mast3r,
        convert_dust3r_pairs_naming,
    )
    import mast3r.utils.path_to_dust3r  # noqa
    from dust3r.utils.image import load_images

    # ── load model on GPU ────────────────────────────────────────────────────
    print(f"\nLoading MASt3R checkpoint …")
    model = AsymmetricMASt3R.from_pretrained(CHECKPOINT).to(inf_device).eval()

    # ── load images ──────────────────────────────────────────────────────────
    img_paths = sorted([
        os.path.join(IMAGE_DIR, f)
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    print(f"\nFound {len(img_paths)} images")

    images = load_images(img_paths, size=IMG_SIZE, verbose=True)
    pairs  = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    print(f"Image pairs: {len(pairs)}")

    # ── two-phase reconstruction ──────────────────────────────────────────────
    t0 = time.time()

    # Use a persistent temp dir so the GPU-populated cache survives into the
    # CPU alignment phase.
    with tempfile.TemporaryDirectory(suffix='_mast3r') as cache_dir:

        # ── phase 1: GPU inference (caches per-pair predictions to disk) ────
        print(f"\n[Phase 1] GPU inference — populating cache …")
        pairs_named = convert_dust3r_pairs_naming(img_paths, pairs)

        try:
            forward_mast3r(pairs_named, model, cache_path=cache_dir, device=inf_device)
        except torch.cuda.OutOfMemoryError:
            print("  CUDA OOM on forward pass — retrying on CPU …")
            torch.cuda.empty_cache()
            model = model.to('cpu')
            inf_device = 'cpu'
            forward_mast3r(pairs_named, model, cache_path=cache_dir, device='cpu')

        # Free GPU memory before the alignment phase
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ── phase 2: CPU global alignment (uses the on-disk cache) ─────────
        print(f"\n[Phase 2] CPU global alignment ({NITER} iterations) …")

        # Reload model on CPU; forward_mast3r will skip inference (cache hit)
        model_cpu = AsymmetricMASt3R.from_pretrained(CHECKPOINT).to('cpu').eval()

        try:
            scene = sparse_global_alignment(
                img_paths,
                pairs,
                cache_dir,
                model_cpu,
                device='cpu',
                dtype=torch.float32,
                niter1=NITER,
                niter2=NITER,
                shared_intrinsics=False,
            )
        except Exception as e:
            print(f"  Alignment failed ({e}), check VRAM/RAM and retry.")
            raise

    elapsed  = time.time() - t0
    vram_mb  = peak_vram_mb()

    # ── extract results ───────────────────────────────────────────────────────
    pts3d_list  = [p.detach().cpu().numpy() for p in scene.pts3d]
    colors_list = scene.pts3d_colors   # already numpy [0,1]
    cam2w       = scene.cam2w.detach().cpu().numpy()   # (N,4,4)

    # ── save outputs ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(OUT_PLY), exist_ok=True)
    n_pts = save_ply(OUT_PLY, pts3d_list, colors_list)
    print(f"\nSaved PLY  → {OUT_PLY}")

    np.savez(OUT_POSES, cam2w=cam2w, img_paths=img_paths)
    print(f"Saved poses → {OUT_POSES}")

    # ── summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print(f"3D points      : {n_pts:,}")
    print(f"Peak VRAM (MB) : {vram_mb:.1f}")
    print(f"Inference time : {elapsed:.1f}s")
    print("="*50)


if __name__ == '__main__':
    main()
