"""Phase 1: run a baseline foundation model on a folder of images and dump a point cloud.

Currently supports DUSt3R and MASt3R cleanly (they share namespace). VGGT runs
via its own pip-installed package. Point3R, CUT3R, StreamVGGT will be added
via subprocess isolation in a follow-up because they collide with mast3r's
`dust3r` module name.

Usage:
    python scripts/01_baseline_inference.py --model dust3r --input data/test_real/
    python scripts/01_baseline_inference.py --model mast3r --input data/test_real/ --out outputs/mast3r_tower
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from drone3d import paths as P


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_mb() -> float:
    return torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0


def list_images(folder: Path) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}
    files = sorted(str(p) for p in folder.iterdir() if p.suffix in exts)
    if not files:
        raise FileNotFoundError(f"no images found under {folder}")
    return files


def save_ply(path: Path, pts: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Minimal text PLY writer; avoids open3d dependency at this stage."""
    n = len(pts)
    header = ["ply", "format ascii 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if colors is not None:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += ["end_header"]
    with path.open("w") as f:
        f.write("\n".join(header) + "\n")
        if colors is None:
            for p in pts:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
        else:
            for p, c in zip(pts, colors):
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def run_dust3r_like(model_name: str, input_dir: Path, out_dir: Path, image_size: int, niter: int) -> dict:
    """Shared loop for DUSt3R and MASt3R (same API surface)."""
    P.inject("mast3r")
    from dust3r.image_pairs import make_pairs  # type: ignore
    from dust3r.inference import inference  # type: ignore
    from dust3r.utils.image import load_images  # type: ignore
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode  # type: ignore

    if model_name == "dust3r":
        from dust3r.model import AsymmetricCroCo3DStereo  # type: ignore
        ckpt = P.checkpoint("DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
        model = AsymmetricCroCo3DStereo.from_pretrained(str(ckpt))
    elif model_name == "mast3r":
        from mast3r.model import AsymmetricMASt3R  # type: ignore
        ckpt = P.checkpoint("MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
        model = AsymmetricMASt3R.from_pretrained(str(ckpt))
    else:
        raise ValueError(model_name)
    model = model.to(device()).eval()

    images = list_images(input_dir)
    print(f"[{model_name}] {len(images)} images @ {image_size}px")
    imgs = load_images(images, size=image_size)
    # Choose scene_graph based on count (per CLAUDE.md / VRAM rules)
    sg = "complete" if len(images) <= 15 else "swin-5"
    pairs = make_pairs(imgs, scene_graph=sg, prefilter=None, symmetrize=True)
    print(f"[{model_name}] pairs: {len(pairs)}  scene_graph={sg}")

    t0 = time.time()
    with torch.no_grad():
        out = inference(pairs, model, device(), batch_size=1)
    t_inf = time.time() - t0
    print(f"[{model_name}] inference: {t_inf:.1f}s  VRAM after fwd: {gpu_mb():.0f} MB")

    # Run global alignment on CPU to keep VRAM usable
    scene = global_aligner(out, device="cpu", mode=GlobalAlignerMode.PointCloudOptimizer)
    t0 = time.time()
    _ = scene.compute_global_alignment(init="mst", niter=niter, schedule="cosine", lr=0.01)
    t_align = time.time() - t0
    print(f"[{model_name}] global alignment: {t_align:.1f}s  ({niter} iters, CPU)")

    # Gather point cloud + colors (scene.imgs are already numpy; pts3d are tensors)
    def _np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    pts3d = [_np(pc).reshape(-1, 3) for pc in scene.get_pts3d()]
    cols = [_np(im).reshape(-1, 3) for im in scene.imgs]
    pts = np.concatenate(pts3d, axis=0)
    cols = (np.concatenate(cols, axis=0) * 255).clip(0, 255).astype(np.uint8)

    # Optional confidence-based filter
    conf = scene.get_conf()
    conf_flat = np.concatenate([_np(c).reshape(-1) for c in conf], axis=0)
    keep = conf_flat > np.percentile(conf_flat, 25)  # drop bottom 25 %
    pts, cols = pts[keep], cols[keep]
    print(f"[{model_name}] points (filtered): {len(pts):,}")

    out_dir.mkdir(parents=True, exist_ok=True)
    ply = out_dir / f"{model_name}_pointcloud.ply"
    save_ply(ply, pts, cols)
    print(f"[{model_name}] wrote {ply}")

    # Save poses (cam-to-world 4x4) for downstream evaluation
    poses = np.stack([_np(m) for m in scene.get_im_poses()], axis=0)
    np.savez(out_dir / f"{model_name}_poses.npz", cam2world=poses)

    return {
        "model": model_name,
        "n_images": len(images),
        "n_pairs": len(pairs),
        "scene_graph": sg,
        "image_size": image_size,
        "n_points": int(len(pts)),
        "vram_mb": round(gpu_mb(), 1),
        "t_inference_s": round(t_inf, 2),
        "t_alignment_s": round(t_align, 2),
        "ply": str(ply),
    }


def run_vggt(input_dir: Path, out_dir: Path, image_size: int) -> dict:
    """VGGT inference on a 6 GB GPU: load fp16, single batch, ≤4 images at 224."""
    from vggt.models.vggt import VGGT  # type: ignore
    from vggt.utils.load_fn import load_and_preprocess_images  # type: ignore

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device()).to(dtype).eval()
    print(f"[vggt] loaded  VRAM: {gpu_mb():.0f} MB  dtype: {dtype}")

    images = list_images(input_dir)
    if torch.cuda.is_available() and len(images) > 4:
        print(f"[vggt] capping to 4 images (6GB VRAM); have {len(images)}")
        images = images[:4]

    imgs = load_and_preprocess_images(images, mode="crop")
    imgs = imgs.to(device()).to(dtype).unsqueeze(0)  # [B,N,3,H,W]
    print(f"[vggt] input batch shape: {tuple(imgs.shape)}")

    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        preds = model(imgs)
    t_inf = time.time() - t0
    print(f"[vggt] inference: {t_inf:.1f}s  VRAM peak: {torch.cuda.max_memory_allocated()/1024/1024:.0f} MB" if torch.cuda.is_available() else f"[vggt] inference: {t_inf:.1f}s")

    pts = preds["world_points"][0].reshape(-1, 3).float().cpu().numpy()
    # color from input
    cols_t = (imgs[0].float().cpu().permute(0, 2, 3, 1).numpy() * 255).clip(0, 255).astype(np.uint8)
    cols = cols_t.reshape(-1, 3)
    n = min(len(pts), len(cols))
    pts, cols = pts[:n], cols[:n]

    out_dir.mkdir(parents=True, exist_ok=True)
    ply = out_dir / "vggt_pointcloud.ply"
    save_ply(ply, pts, cols)
    print(f"[vggt] wrote {ply}  points: {len(pts):,}")

    return {
        "model": "vggt",
        "n_images": len(images),
        "image_size": image_size,
        "n_points": int(len(pts)),
        "t_inference_s": round(t_inf, 2),
        "ply": str(ply),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["dust3r", "mast3r", "vggt"])
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("outputs"))
    ap.add_argument("--size", type=int, default=224, help="image side (px); 224 for 6GB GPU")
    ap.add_argument("--niter", type=int, default=300, help="global-alignment iterations")
    args = ap.parse_args()

    if args.model in ("dust3r", "mast3r"):
        stats = run_dust3r_like(args.model, args.input, args.out / args.model, args.size, args.niter)
    else:
        stats = run_vggt(args.input, args.out / args.model, args.size)

    print("\n--- summary ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
