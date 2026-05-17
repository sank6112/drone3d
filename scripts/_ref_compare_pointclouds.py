#!/usr/bin/env python3
"""Compare DUSt3R vs MASt3R point clouds: stats, Chamfer distance, side-by-side PNG."""

import os
import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

PROJECT_ROOT = os.path.expanduser("~/projects/drone3d")
DUST3R_PLY   = os.path.join(PROJECT_ROOT, "outputs", "dust3r_poster.ply")
MAST3R_PLY   = os.path.join(PROJECT_ROOT, "outputs", "mast3r_poster.ply")
OUT_PNG      = os.path.join(PROJECT_ROOT, "outputs", "comparison_dust3r_vs_mast3r_poster.png")
SAMPLE_N     = 50_000


# ── helpers ───────────────────────────────────────────────────────────────────

def load_ply(path):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else None
    return pts, cols


def bbox_dims(pts):
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return hi - lo


def point_density(pts):
    """Mean nearest-neighbour distance as a density proxy (lower = denser)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    dists = pcd.compute_nearest_neighbor_distance()
    return float(np.mean(dists))


def chamfer_distance(a, b, n=SAMPLE_N, seed=42):
    rng = np.random.default_rng(seed)
    a_s = a[rng.choice(len(a), min(n, len(a)), replace=False)]
    b_s = b[rng.choice(len(b), min(n, len(b)), replace=False)]

    pcd_a = o3d.geometry.PointCloud()
    pcd_a.points = o3d.utility.Vector3dVector(a_s)
    pcd_b = o3d.geometry.PointCloud()
    pcd_b.points = o3d.utility.Vector3dVector(b_s)

    d_ab = np.asarray(pcd_a.compute_point_cloud_distance(pcd_b))
    d_ba = np.asarray(pcd_b.compute_point_cloud_distance(pcd_a))

    return float(np.mean(d_ab**2) + np.mean(d_ba**2))


def print_stats(label, pts):
    dims = bbox_dims(pts)
    vol  = float(np.prod(dims))
    dens = point_density(pts[:min(20_000, len(pts))])  # subsample for speed
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  Points           : {len(pts):,}")
    print(f"  Bounding box (m) : X={dims[0]:.3f}  Y={dims[1]:.3f}  Z={dims[2]:.3f}")
    print(f"  BBox volume      : {vol:.4f}")
    print(f"  Mean NN dist     : {dens:.5f}  (lower = denser)")


# ── scatter plot for one cloud on an Axes3D ───────────────────────────────────

def scatter3d(ax, pts, cols, title, n=30_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pts), min(n, len(pts)), replace=False)
    p   = pts[idx]

    if cols is not None:
        c = cols[idx].clip(0, 1)
    else:
        # colour by height
        z = p[:, 2]
        z_norm = (z - z.min()) / (z.ptp() + 1e-9)
        c = plt.cm.viridis(z_norm)[:, :3]

    ax.scatter(p[:, 0], p[:, 1], p[:, 2],
               c=c, s=0.3, linewidths=0, alpha=0.7)

    # nice viewing angle
    ax.view_init(elev=25, azim=45)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel("X", fontsize=8); ax.set_ylabel("Y", fontsize=8); ax.set_zlabel("Z", fontsize=8)
    ax.tick_params(labelsize=6)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading point clouds …")
    dust3r_pts, dust3r_cols = load_ply(DUST3R_PLY)
    mast3r_pts, mast3r_cols = load_ply(MAST3R_PLY)

    print_stats("DUSt3R", dust3r_pts)
    print_stats("MASt3R", mast3r_pts)

    print(f"\nComputing Chamfer distance (sampling {SAMPLE_N:,} pts each) …")
    cd = chamfer_distance(dust3r_pts, mast3r_pts)
    print(f"\n{'═'*50}")
    print(f"  Chamfer distance : {cd:.6f}")
    print(f"  (mean-squared bidirectional nearest-neighbour)")
    print(f"{'═'*50}")

    # ── side-by-side visualization ────────────────────────────────────────────
    print("\nRendering side-by-side PNG …")
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor('#1a1a2e')

    ax1 = fig.add_subplot(121, projection='3d', facecolor='#16213e')
    ax2 = fig.add_subplot(122, projection='3d', facecolor='#16213e')

    scatter3d(ax1, dust3r_pts, dust3r_cols,
              f"DUSt3R  ({len(dust3r_pts):,} pts)")
    scatter3d(ax2, mast3r_pts, mast3r_cols,
              f"MASt3R  ({len(mast3r_pts):,} pts)")

    for ax in (ax1, ax2):
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#334')
        ax.yaxis.pane.set_edgecolor('#334')
        ax.zaxis.pane.set_edgecolor('#334')
        ax.tick_params(colors='#aaa')
        ax.xaxis.label.set_color('#aaa')
        ax.yaxis.label.set_color('#aaa')
        ax.zaxis.label.set_color('#aaa')
        ax.title.set_color('white')

    bbox_d = bbox_dims(dust3r_pts)
    bbox_m = bbox_dims(mast3r_pts)
    fig.suptitle(
        f"Point Cloud Comparison — Mip-NeRF 360 Garden (15 images)\n"
        f"Chamfer distance: {cd:.5f}  |  "
        f"DUSt3R bbox: {bbox_d[0]:.2f}×{bbox_d[1]:.2f}×{bbox_d[2]:.2f}  |  "
        f"MASt3R bbox: {bbox_m[0]:.2f}×{bbox_m[1]:.2f}×{bbox_m[2]:.2f}",
        color='white', fontsize=10, y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {OUT_PNG}")


if __name__ == '__main__':
    main()
