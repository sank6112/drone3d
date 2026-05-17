"""Compare multiple model outputs from Phase 1.

For each model in outputs/<model>/, load the .ply, compute statistics, and
produce a side-by-side visualization saved as PNG.

Usage:
    python scripts/02_compare_pointclouds.py
    python scripts/02_compare_pointclouds.py --models dust3r mast3r monst3r
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import open3d as o3d  # noqa: E402


def load_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None
    return pts, cols


def stats(pts: np.ndarray) -> dict:
    bb_min = pts.min(0).tolist()
    bb_max = pts.max(0).tolist()
    size = [b - a for a, b in zip(bb_min, bb_max)]
    centroid = pts.mean(0).tolist()
    # Local density via mean distance to 16 nearest neighbours
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    kdt = o3d.geometry.KDTreeFlann(pcd)
    sample = np.random.RandomState(0).choice(len(pts), size=min(5000, len(pts)), replace=False)
    dists = []
    for i in sample:
        _, _, dd = kdt.search_knn_vector_3d(pcd.points[i], 16)
        if len(dd) > 1:
            dists.append(np.sqrt(np.mean(dd[1:])))
    return {
        "n_points": int(len(pts)),
        "bbox_min": [round(v, 3) for v in bb_min],
        "bbox_max": [round(v, 3) for v in bb_max],
        "bbox_size": [round(v, 3) for v in size],
        "centroid": [round(v, 3) for v in centroid],
        "mean_local_spacing": round(float(np.mean(dists)), 4) if dists else None,
    }


def chamfer_l1(a: np.ndarray, b: np.ndarray, sample: int = 20000) -> float:
    """Bidirectional Chamfer-L1 between two point clouds, on a random subsample."""
    rng = np.random.RandomState(0)
    A = a[rng.choice(len(a), size=min(sample, len(a)), replace=False)]
    B = b[rng.choice(len(b), size=min(sample, len(b)), replace=False)]
    pA = o3d.geometry.PointCloud(); pA.points = o3d.utility.Vector3dVector(A)
    pB = o3d.geometry.PointCloud(); pB.points = o3d.utility.Vector3dVector(B)
    d_ab = np.asarray(pA.compute_point_cloud_distance(pB))
    d_ba = np.asarray(pB.compute_point_cloud_distance(pA))
    return float(0.5 * (d_ab.mean() + d_ba.mean()))


def render_scatter(pts: np.ndarray, cols: np.ndarray | None, ax) -> None:
    if len(pts) > 30000:
        idx = np.random.RandomState(0).choice(len(pts), 30000, replace=False)
        pts, cols = pts[idx], (cols[idx] if cols is not None else None)
    c = cols if cols is not None else np.full((len(pts), 3), 0.3)
    ax.scatter(pts[:, 0], pts[:, 2], -pts[:, 1], c=np.clip(c, 0, 1), s=0.5, alpha=0.7)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("-Y (up)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--models", nargs="+", default=None,
                    help="Auto-detect by default; pass model names to restrict.")
    ap.add_argument("--save", type=Path, default=Path("outputs/comparison.png"))
    ap.add_argument("--stats-json", type=Path, default=Path("outputs/comparison_stats.json"))
    args = ap.parse_args()

    plys: dict[str, Path] = {}
    if args.models:
        for m in args.models:
            cand = args.outputs / m / f"{m}_pointcloud.ply"
            if cand.exists():
                plys[m] = cand
    else:
        for d in sorted(args.outputs.iterdir()):
            if not d.is_dir():
                continue
            cand = d / f"{d.name}_pointcloud.ply"
            if cand.exists():
                plys[d.name] = cand
    if not plys:
        raise SystemExit("no point clouds found")
    print(f"comparing: {list(plys)}")

    loaded = {n: load_ply(p) for n, p in plys.items()}
    stat_table: dict[str, dict] = {}
    for n, (pts, _cols) in loaded.items():
        stat_table[n] = stats(pts)
        print(f"[{n}] n={stat_table[n]['n_points']:,}  bbox={stat_table[n]['bbox_size']}  "
              f"spacing={stat_table[n]['mean_local_spacing']}")

    # Pairwise Chamfer
    names = list(loaded)
    chamfer = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = chamfer_l1(loaded[a][0], loaded[b][0])
            chamfer[f"{a}_vs_{b}"] = round(d, 4)
            print(f"  chamfer L1  {a:8s} ↔ {b:8s} = {d:.4f}")
    stat_table["__chamfer__"] = chamfer

    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    args.stats_json.write_text(json.dumps(stat_table, indent=2))
    print(f"wrote {args.stats_json}")

    # Figure: 1 row × N columns of 3D scatters
    n = len(loaded)
    fig = plt.figure(figsize=(5 * n, 5))
    for i, (name, (pts, cols)) in enumerate(loaded.items()):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        render_scatter(pts, cols, ax)
        ax.set_title(f"{name}\n{len(pts):,} pts")
    fig.tight_layout()
    fig.savefig(args.save, dpi=150)
    print(f"wrote {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
