"""Mega-NeRF (Mill-19 / UrbanScene3D) -> COLMAP binary, downscaled for 6 GB.

Each train/metadata/<name>.pt is {H, W, c2w(3x4), intrinsics[fx,fy,cx,cy]}.
We subsample frames, downscale images to --long px (scaling intrinsics), and
write a COLMAP binary model with a random point init (no SfM points ship with
Mega-NeRF). Axis convention is assumed OpenCV; Stage C self-validates via
train-view PSNR, so a wrong convention is detected, not silently trusted.

    python scripts/meganerf_to_colmap.py --src data/rubble/rubble-pixsfm/train \
        --out data/rubble_sub --n 60 --long 768
"""
from __future__ import annotations
import argparse, struct
from pathlib import Path
import numpy as np, torch
from PIL import Image


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([[Rxx-Ryy-Rzz,0,0,0],[Ryx+Rxy,Ryy-Rxx-Rzz,0,0],
                  [Rzx+Rxz,Rzy+Ryz,Rzz-Rxx-Ryy,0],
                  [Ryz-Rzy,Rzx-Rxz,Rxy-Ryx,Rxx+Ryy+Rzz]]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3,0,1,2], np.argmax(vals)]
    return q if q[0] >= 0 else -q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="mega-nerf .../train dir (rgbs/ + metadata/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--long", type=int, default=768, help="downscaled long side")
    args = ap.parse_args()

    src = Path(args.src)
    rgbs = sorted((src / "rgbs").glob("*.jpg")) + sorted((src / "rgbs").glob("*.png"))
    if not rgbs:
        raise SystemExit(f"no rgbs under {src/'rgbs'}")
    idx = np.linspace(0, len(rgbs) - 1, min(args.n, len(rgbs))).round().astype(int)
    rgbs = [rgbs[i] for i in idx]

    out = Path(args.out)
    img_dir = out / "images"; sparse = out / "sparse" / "0"
    img_dir.mkdir(parents=True, exist_ok=True); sparse.mkdir(parents=True, exist_ok=True)

    names, sizes, focs, pps, poses = [], [], [], [], []
    for rp in rgbs:
        md = torch.load(src / "metadata" / (rp.stem + ".pt"), map_location="cpu", weights_only=False)
        H, W = int(md["H"]), int(md["W"])
        fx, fy, cx, cy = [float(x) for x in md["intrinsics"].tolist()]
        c2w = np.eye(4); c2w[:3, :4] = md["c2w"].numpy()
        im = Image.open(rp).convert("RGB")
        scale = args.long / max(W, H)
        nW, nH = int(round(W * scale)), int(round(H * scale))
        im = im.resize((nW, nH), Image.BICUBIC)
        name = rp.stem + ".png"
        im.save(img_dir / name)
        names.append(name); sizes.append((nW, nH))
        focs.append((fx * scale, fy * scale)); pps.append((cx * scale, cy * scale))
        poses.append(c2w)

    # cameras.bin
    with open(sparse / "cameras.bin", "wb") as f:
        f.write(struct.pack("L", len(names)))
        for i, (W, H) in enumerate(sizes, 1):
            f.write(struct.pack("IiLL", i, 1, W, H))
            f.write(struct.pack("dddd", focs[i-1][0], focs[i-1][1], pps[i-1][0], pps[i-1][1]))
    # images.bin
    with open(sparse / "images.bin", "wb") as f:
        f.write(struct.pack("L", len(names)))
        st = struct.Struct("<I4d3dI")
        for i, name in enumerate(names, 1):
            w2c = np.linalg.inv(poses[i-1]); q = rotmat2qvec(w2c[:3, :3]); t = w2c[:3, 3]
            f.write(st.pack(i, q[0], q[1], q[2], q[3], t[0], t[1], t[2], i))
            f.write(name.encode() + b"\x00"); f.write(struct.pack("Q", 0))
    # points3D.bin — random init in the camera bbox (Mega-NeRF ships no SfM points)
    cams = np.array([p[:3, 3] for p in poses])
    lo, hi = cams.min(0), cams.max(0); ctr = (lo + hi) / 2; rad = np.linalg.norm(hi - lo) / 2 + 1e-3
    npts = 80000
    pts = ctr + (np.random.rand(npts, 3) - 0.5) * 2 * rad
    cols = np.random.randint(80, 200, (npts, 3), dtype=np.uint8)
    with open(sparse / "points3D.bin", "wb") as f:
        f.write(struct.pack("L", npts))
        st = struct.Struct("<Q3d3BdQ")
        for j in range(npts):
            p, c = pts[j], cols[j]
            f.write(st.pack(j+1, float(p[0]), float(p[1]), float(p[2]),
                            int(c[0]), int(c[1]), int(c[2]), 0.0, 0))
    print(f"[mn2c] wrote {len(names)} cams @ ~{args.long}px + {npts} rand points -> {out}")


if __name__ == "__main__":
    main()
