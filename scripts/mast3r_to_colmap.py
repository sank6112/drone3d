"""MASt3R -> COLMAP sparse model, so gsplat can splat directly from raw images
(no COLMAP/known poses). This is the real product path: video frames -> MASt3R
metric poses+points -> gsplat. InstantSplat-style.

Writes, under --out:
    images/                 the MASt3R-resized RGB frames (what the poses match)
    sparse/0/cameras.txt    PINHOLE intrinsics per image
    sparse/0/images.txt     world->cam poses (COLMAP convention)
    sparse/0/points3D.txt   MASt3R point cloud (init for gsplat)

    python scripts/mast3r_to_colmap.py --frames data/dronesplat/Sculpture/images \
        --out outputs/sculpture_mast3r_colmap --max-frames 15
"""

from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from drone3d import paths as P  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    """COLMAP's rotation-matrix -> quaternion (qw, qx, qy, qz)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    qvec = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=15, help="all-pairs OOMs >~15-20 on 6GB")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--niter", type=int, default=300)
    ap.add_argument("--conf-pct", type=float, default=30.0)
    args = ap.parse_args()

    frames = sorted(p for p in Path(args.frames).iterdir() if p.suffix in IMG_EXTS)
    if args.max_frames and len(frames) > args.max_frames:
        idx = np.linspace(0, len(frames) - 1, args.max_frames).round().astype(int)
        frames = [frames[i] for i in idx]
    frames = [str(p) for p in frames]

    P.inject("mast3r")
    from dust3r.image_pairs import make_pairs            # type: ignore
    from dust3r.inference import inference               # type: ignore
    from dust3r.utils.image import load_images           # type: ignore
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode  # type: ignore
    from mast3r.model import AsymmetricMASt3R            # type: ignore

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = P.checkpoint("MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    model = AsymmetricMASt3R.from_pretrained(str(ckpt)).to(dev).eval()

    imgs = load_images(frames, size=args.image_size)
    sg = "complete" if len(frames) <= 15 else "swin-5"
    pairs = make_pairs(imgs, scene_graph=sg, prefilter=None, symmetrize=True)
    print(f"[m2c] {len(frames)} frames @ {args.image_size}px  pairs={len(pairs)}  sg={sg}")

    t0 = time.time()
    with torch.no_grad():
        out = inference(pairs, model, dev, batch_size=1)
    scene = global_aligner(out, device="cpu", mode=GlobalAlignerMode.PointCloudOptimizer)
    scene.compute_global_alignment(init="mst", niter=args.niter, schedule="cosine", lr=0.01)
    print(f"[m2c] alignment done in {time.time()-t0:.1f}s")

    def _np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    poses = _np(torch.stack(list(scene.get_im_poses())))          # cam2world [N,4,4]
    focals = _np(scene.get_focals()).reshape(-1)                  # [N] pixels
    try:
        pps = _np(scene.get_principal_points())                  # [N,2]
    except Exception:
        pps = None
    rgb_imgs = [(_np(im) * 255).clip(0, 255).astype(np.uint8) for im in scene.imgs]

    out = Path(args.out)
    img_dir = out / "images"
    sparse = out / "sparse" / "0"
    img_dir.mkdir(parents=True, exist_ok=True)
    sparse.mkdir(parents=True, exist_ok=True)

    names = [f"frame_{i:04d}.png" for i in range(len(frames))]
    for name, im in zip(names, rgb_imgs):
        Image.fromarray(im).save(img_dir / name)

    # --- COLMAP *binary* model (the fork's .txt reader is py2-broken) --------- #
    with open(sparse / "cameras.bin", "wb") as f:
        f.write(struct.pack("L", len(names)))
        for i, im in enumerate(rgb_imgs, start=1):
            H, W = im.shape[:2]
            fc = float(focals[i - 1])
            cx, cy = (float(pps[i - 1][0]), float(pps[i - 1][1])) if pps is not None else (W / 2, H / 2)
            f.write(struct.pack("IiLL", i, 1, W, H))          # model 1 = PINHOLE
            f.write(struct.pack("dddd", fc, fc, cx, cy))

    with open(sparse / "images.bin", "wb") as f:
        f.write(struct.pack("L", len(names)))
        istruct = struct.Struct("<I4d3dI")
        for i, name in enumerate(names, start=1):
            w2c = np.linalg.inv(poses[i - 1])
            q = rotmat2qvec(w2c[:3, :3])
            t = w2c[:3, 3]
            f.write(istruct.pack(i, q[0], q[1], q[2], q[3], t[0], t[1], t[2], i))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("Q", 0))                      # num_points2D = 0

    # points3D from the MASt3R cloud (init for gsplat), subsampled + conf-filtered
    pts = np.concatenate([_np(pc).reshape(-1, 3) for pc in scene.get_pts3d()], 0)
    cols = np.concatenate([_np(im).reshape(-1, 3) for im in scene.imgs], 0)
    cols = (cols * 255).clip(0, 255).astype(np.uint8)
    conf = np.concatenate([_np(c).reshape(-1) for c in scene.get_conf()], 0)
    keep = conf > np.percentile(conf, args.conf_pct)
    pts, cols = pts[keep], cols[keep]
    if len(pts) > 200000:
        sel = np.random.choice(len(pts), 200000, replace=False)
        pts, cols = pts[sel], cols[sel]
    with open(sparse / "points3D.bin", "wb") as f:
        f.write(struct.pack("L", len(pts)))
        pstruct = struct.Struct("<Q3d3BdQ")
        for j, (p, c) in enumerate(zip(pts, cols), start=1):
            f.write(pstruct.pack(j, float(p[0]), float(p[1]), float(p[2]),
                                 int(c[0]), int(c[1]), int(c[2]), 0.0, 0))

    print(f"[m2c] wrote COLMAP model: {len(frames)} cams, {len(pts)} points -> {out}")


if __name__ == "__main__":
    main()
