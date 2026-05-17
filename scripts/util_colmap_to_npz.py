"""Convert COLMAP images.txt → Nx4x4 cam-to-world .npz, filtered & ordered by an image folder.

Usage:
    python scripts/util_colmap_to_npz.py \
        --images-txt path/to/images.txt \
        --image-dir data/eth3d_courtyard/ \
        --out data/eth3d_courtyard/gt_poses.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_colmap(path: Path) -> dict[str, np.ndarray]:
    """Return name -> 4x4 cam-to-world (OpenCV-style: x right, y down, z forward)."""
    out: dict[str, np.ndarray] = {}
    with path.open() as f:
        lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        qw, qx, qy, qz = (float(parts[k]) for k in (1, 2, 3, 4))
        tx, ty, tz = (float(parts[k]) for k in (5, 6, 7))
        name = parts[9]
        n = qw * qw + qx * qx + qy * qy + qz * qz
        s = 0.0 if n < 1e-9 else 2.0 / n
        wx, wy, wz = s * qw * qx, s * qw * qy, s * qw * qz
        xx, xy, xz = s * qx * qx, s * qx * qy, s * qx * qz
        yy, yz, zz = s * qy * qy, s * qy * qz, s * qz * qz
        R = np.array([
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ])
        # COLMAP stores world-to-camera; invert to cam-to-world
        Rcw, tcw = R.T, -R.T @ np.array([tx, ty, tz])
        T = np.eye(4)
        T[:3, :3] = Rcw
        T[:3, 3] = tcw
        # COLMAP image names may use forward slashes (e.g. "dslr_images_undistorted/DSC_0286.JPG")
        base = name.rsplit("/", 1)[-1]
        out[base] = T
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-txt", type=Path, required=True)
    ap.add_argument("--image-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    db = read_colmap(args.images_txt)
    in_dir = sorted([p.name for p in args.image_dir.iterdir()
                     if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    missing = [n for n in in_dir if n not in db]
    if missing:
        print(f"warning: {len(missing)} images have no COLMAP pose: {missing[:3]}")
    in_dir = [n for n in in_dir if n in db]
    poses = np.stack([db[n] for n in in_dir])
    np.savez(args.out, cam2world=poses, names=np.array(in_dir))
    print(f"wrote {args.out}: {len(poses)} poses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
