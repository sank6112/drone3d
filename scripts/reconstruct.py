"""Product Track deliverable: drone video -> 3D reconstruction in one entrypoint.

Pipeline:
    video --> frame extraction (FPS sample + blur filter + duplicate filter)
          --> MASt3R metric inference + global alignment
          --> colored point cloud (.ply) + camera poses (.npz)

Designed to be both a CLI and an importable component of the larger SaaS:

    from scripts.reconstruct import reconstruct
    result = reconstruct("flight.mp4", "outputs/run1")

CLI:
    python scripts/reconstruct.py --video flight.mp4 --out outputs/run1
    python scripts/reconstruct.py --frames data/test_real/ --out outputs/run1   # skip extraction
    python scripts/reconstruct.py --video flight.mp4 --out outputs/run1 --fps 2 --max-frames 20

Runs on the 6 GB RTX 4050. MASt3R is metric, so distances/scale are in meters.
Telemetry fusion (GPS/IMU scale + rotation supervision) is the Research Track and
plugs in at the marked hook below once Pixhawk logs arrive.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from drone3d import paths as P

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}


# --------------------------------------------------------------------------- #
# Stage 1: frame extraction (FPS sample + blur filter + duplicate filter)
# --------------------------------------------------------------------------- #
def extract_frames(
    video: Path,
    out_dir: Path,
    fps: float = 2.0,
    max_frames: int = 30,
    min_frames: int = 6,
    blur_pct: float = 25.0,
    dup_thresh: float = 0.985,
) -> list[Path]:
    """Sample frames from a video, drop the blurriest, drop near-duplicates.

    - fps:        target sampling rate (frames per second of source video).
    - max_frames: hard cap (VRAM: MASt3R does all-pairs; keep <= ~15-20 for 6 GB).
    - min_frames: floor; if the duplicate filter prunes below this, fall back to
                  evenly sampling the sharp set (aerial footage overlaps heavily,
                  so an aggressive dup filter can otherwise collapse to ~1 frame).
    - blur_pct:   drop frames whose Laplacian variance is in the bottom percentile.
    - dup_thresh: drop a frame if its normalized correlation to the last kept
                  frame exceeds this (near-static / hovering segments).
    """
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / max(fps, 1e-3))))

    # First pass: sample candidates + record blur score.
    candidates: list[tuple[np.ndarray, float]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            candidates.append((frame, blur))
        idx += 1
    cap.release()

    if not candidates:
        raise RuntimeError(f"no frames sampled from {video}")

    # Blur filter: drop the bottom percentile of sharpness.
    blur_scores = np.array([b for _, b in candidates])
    cutoff = np.percentile(blur_scores, blur_pct)
    kept_sharp = [f for f, b in candidates if b >= cutoff]

    # Duplicate filter: skip frames too similar to the previous kept one.
    def _sig(img: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (64, 64)).astype(np.float32)
        g -= g.mean()
        n = np.linalg.norm(g) + 1e-8
        return g / n

    kept: list[np.ndarray] = []
    last_sig = None
    for f in kept_sharp:
        s = _sig(f)
        if last_sig is not None and float((s * last_sig).sum()) > dup_thresh:
            continue
        kept.append(f)
        last_sig = s

    # Floor: never let the dup filter collapse the set (aerial frames overlap
    # heavily). Fall back to evenly sampling the sharp set.
    floor = min(min_frames, len(kept_sharp))
    if len(kept) < floor:
        sel = np.linspace(0, len(kept_sharp) - 1, floor).round().astype(int)
        kept = [kept_sharp[i] for i in sel]
        print(f"[extract] dup filter under floor; fell back to {len(kept)} even samples")

    # Even subsample down to max_frames (preserve temporal spread).
    if len(kept) > max_frames:
        sel = np.linspace(0, len(kept) - 1, max_frames).round().astype(int)
        kept = [kept[i] for i in sel]

    paths: list[Path] = []
    for i, f in enumerate(kept):
        p = out_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(p), f)
        paths.append(p)

    print(f"[extract] {len(candidates)} sampled -> {len(kept_sharp)} sharp "
          f"-> {len(paths)} kept (fps~{fps}, cap {max_frames})")
    return paths


# --------------------------------------------------------------------------- #
# Stage 2: MASt3R reconstruction (metric)
# --------------------------------------------------------------------------- #
def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _save_ply(path: Path, pts: np.ndarray, cols: np.ndarray) -> None:
    """Binary-free colored PLY writer (no open3d dependency required)."""
    header = ["ply", "format ascii 1.0", f"element vertex {len(pts)}",
              "property float x", "property float y", "property float z",
              "property uchar red", "property uchar green", "property uchar blue",
              "end_header"]
    with path.open("w") as f:
        f.write("\n".join(header) + "\n")
        for p, c in zip(pts, cols):
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def reconstruct_frames(
    frames: list[str],
    out_dir: Path,
    image_size: int = 512,
    niter: int = 300,
    conf_pct: float = 25.0,
) -> dict:
    """Run MASt3R on a list of image paths -> point cloud + poses. Metric scale."""
    P.inject("mast3r")
    from dust3r.image_pairs import make_pairs          # type: ignore
    from dust3r.inference import inference             # type: ignore
    from dust3r.utils.image import load_images         # type: ignore
    from dust3r.cloud_opt import global_aligner, GlobalAlignerMode  # type: ignore
    from mast3r.model import AsymmetricMASt3R          # type: ignore

    dev = _device()
    ckpt = P.checkpoint("MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    model = AsymmetricMASt3R.from_pretrained(str(ckpt)).to(dev).eval()

    imgs = load_images(list(frames), size=image_size)
    sg = "complete" if len(frames) <= 15 else "swin-5"
    pairs = make_pairs(imgs, scene_graph=sg, prefilter=None, symmetrize=True)
    print(f"[mast3r] {len(frames)} frames @ {image_size}px  pairs={len(pairs)}  sg={sg}")

    t0 = time.time()
    with torch.no_grad():
        out = inference(pairs, model, dev, batch_size=1)
    t_inf = time.time() - t0

    # Global alignment on CPU to keep VRAM usable on 6 GB.
    scene = global_aligner(out, device="cpu", mode=GlobalAlignerMode.PointCloudOptimizer)
    t0 = time.time()
    scene.compute_global_alignment(init="mst", niter=niter, schedule="cosine", lr=0.01)
    t_align = time.time() - t0

    # ---- Research Track hook -------------------------------------------------
    # When Pixhawk telemetry is available, align/rescale `scene` here using
    # GPS-derived baselines (metric scale) and IMU-integrated rotation before
    # extracting points. See PLAN.md Phase 6 (scripts/losses/telemetry.py).
    # -------------------------------------------------------------------------

    def _np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    pts = np.concatenate([_np(pc).reshape(-1, 3) for pc in scene.get_pts3d()], axis=0)
    cols = np.concatenate([_np(im).reshape(-1, 3) for im in scene.imgs], axis=0)
    cols = (cols * 255).clip(0, 255).astype(np.uint8)

    conf = np.concatenate([_np(c).reshape(-1) for c in scene.get_conf()], axis=0)
    keep = conf > np.percentile(conf, conf_pct)
    pts, cols = pts[keep], cols[keep]

    out_dir.mkdir(parents=True, exist_ok=True)
    ply = out_dir / "reconstruction.ply"
    _save_ply(ply, pts, cols)
    poses = np.stack([_np(m) for m in scene.get_im_poses()], axis=0)
    np.savez(out_dir / "poses.npz", cam2world=poses)

    result = {
        "n_frames": len(frames),
        "n_pairs": len(pairs),
        "image_size": image_size,
        "n_points": int(len(pts)),
        "t_inference_s": round(t_inf, 2),
        "t_alignment_s": round(t_align, 2),
        "ply": str(ply),
        "poses": str(out_dir / "poses.npz"),
        "metric": True,
    }
    print(f"[mast3r] inference {t_inf:.1f}s  align {t_align:.1f}s  "
          f"points={len(pts):,}  -> {ply}")
    return result


# --------------------------------------------------------------------------- #
# End-to-end entrypoint
# --------------------------------------------------------------------------- #
def reconstruct(
    video: str | Path | None = None,
    out: str | Path = "outputs/reconstruct",
    frames_dir: str | Path | None = None,
    fps: float = 2.0,
    max_frames: int = 30,
    image_size: int = 512,
    niter: int = 300,
) -> dict:
    """Video (or a folder of frames) -> 3D reconstruction. SaaS-callable.

    Provide exactly one of `video` or `frames_dir`.
    Returns a JSON-serializable result dict (also written to <out>/result.json).
    """
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if video is not None:
        frame_paths = extract_frames(Path(video), out_dir / "frames",
                                     fps=fps, max_frames=max_frames)
        frames = [str(p) for p in frame_paths]
    elif frames_dir is not None:
        fdir = Path(frames_dir)
        frames = sorted(str(p) for p in fdir.iterdir() if p.suffix in IMG_EXTS)
        if not frames:
            raise FileNotFoundError(f"no images under {fdir}")
    else:
        raise ValueError("provide either video= or frames_dir=")

    result = reconstruct_frames(frames, out_dir, image_size=image_size, niter=niter)
    result["source"] = str(video) if video else str(frames_dir)
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Drone video -> 3D reconstruction (MASt3R, metric).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="input flight video (mp4/mov/...)")
    src.add_argument("--frames", type=str, help="folder of pre-extracted frames (skip extraction)")
    ap.add_argument("--out", type=str, default="outputs/reconstruct", help="output directory")
    ap.add_argument("--fps", type=float, default=2.0, help="frame sampling rate from video")
    ap.add_argument("--max-frames", type=int, default=30, help="cap on frames fed to MASt3R")
    ap.add_argument("--image-size", type=int, default=512, help="512 (quality) or 224 (fast/low-VRAM)")
    ap.add_argument("--niter", type=int, default=300, help="global-alignment iterations")
    args = ap.parse_args()

    result = reconstruct(
        video=args.video, frames_dir=args.frames, out=args.out,
        fps=args.fps, max_frames=args.max_frames,
        image_size=args.image_size, niter=args.niter,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
