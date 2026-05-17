"""Phase 3: quantitative evaluation framework.

Reads predicted and ground-truth point clouds and/or pose sequences, computes
the standard 3D-reconstruction metrics, prints a markdown table, and writes
JSON for downstream ablation aggregation.

Inputs supported (any combination):
  --pred         path to predicted .ply
  --gt           path to ground-truth .ply
  --pred-poses   path to predicted *_poses.npz (key 'cam2world')
  --gt-poses     path to ground-truth poses (.npz with 'cam2world', or COLMAP images.txt)

If both pred and gt point clouds are supplied → geometry metrics.
If both pose sets are supplied → trajectory metrics (ATE + RPE) after Sim(3) alignment.

Usage:
    # geometry only
    python scripts/03_evaluate.py --pred outputs/dust3r/dust3r_pointcloud.ply \
                                  --gt   gt.ply
    # no GT — just print stats of a single point cloud
    python scripts/03_evaluate.py --pred outputs/mast3r/mast3r_pointcloud.ply
    # poses only
    python scripts/03_evaluate.py --pred-poses pred.npz --gt-poses gt.npz
    # everything
    python scripts/03_evaluate.py --pred a.ply --gt b.ply --pred-poses ap.npz --gt-poses bp.npz \
                                  --json outputs/eval/run.json --tag dust3r_finearts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d


# ----------------------------- I/O -----------------------------------------

def load_ply(path: Path) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(str(path))
    if not pcd.has_points():
        raise ValueError(f"empty point cloud: {path}")
    return np.asarray(pcd.points)


def load_poses(path: Path) -> np.ndarray:
    """Load Nx4x4 cam-to-world poses from .npz (key 'cam2world') or COLMAP images.txt."""
    if path.suffix == ".npz":
        d = np.load(path)
        for key in ("cam2world", "poses", "extrinsics"):
            if key in d:
                P = d[key]
                if P.ndim == 3 and P.shape[-2:] == (4, 4):
                    return P
        raise ValueError(f"{path}: no Nx4x4 pose array under known keys")
    if path.name.endswith("images.txt"):
        return _read_colmap_images(path)
    raise ValueError(f"unknown pose format: {path}")


def _read_colmap_images(path: Path) -> np.ndarray:
    """COLMAP images.txt → Nx4x4 cam-to-world poses."""
    poses: list[np.ndarray] = []
    with path.open() as f:
        lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
    for i in range(0, len(lines), 2):  # COLMAP stores 2 lines per image
        parts = lines[i].split()
        qw, qx, qy, qz = (float(parts[k]) for k in (1, 2, 3, 4))
        tx, ty, tz = (float(parts[k]) for k in (5, 6, 7))
        # quaternion (w,x,y,z) → rotation matrix
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
        poses.append(T)
    return np.stack(poses)


# ----------------------------- Geometry metrics ----------------------------

def _to_pcd(arr: np.ndarray, voxel: float | None = None) -> o3d.geometry.PointCloud:
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(arr)
    if voxel is not None:
        p = p.voxel_down_sample(voxel)
    return p


def chamfer(pred: np.ndarray, gt: np.ndarray, sample: int = 50_000) -> dict:
    rng = np.random.RandomState(0)
    if len(pred) > sample:
        pred = pred[rng.choice(len(pred), sample, replace=False)]
    if len(gt) > sample:
        gt = gt[rng.choice(len(gt), sample, replace=False)]
    pP, pG = _to_pcd(pred), _to_pcd(gt)
    d_pg = np.asarray(pP.compute_point_cloud_distance(pG))
    d_gp = np.asarray(pG.compute_point_cloud_distance(pP))
    return {
        "chamfer_l1": float(0.5 * (d_pg.mean() + d_gp.mean())),
        "chamfer_l2": float(0.5 * ((d_pg ** 2).mean() + (d_gp ** 2).mean()) ** 0.5),
        "accuracy_mean": float(d_pg.mean()),    # pred → gt
        "accuracy_median": float(np.median(d_pg)),
        "completeness_mean": float(d_gp.mean()),  # gt → pred
        "completeness_median": float(np.median(d_gp)),
    }


def fscore(pred: np.ndarray, gt: np.ndarray, thresholds: tuple[float, ...], sample: int = 50_000) -> dict:
    rng = np.random.RandomState(0)
    if len(pred) > sample:
        pred = pred[rng.choice(len(pred), sample, replace=False)]
    if len(gt) > sample:
        gt = gt[rng.choice(len(gt), sample, replace=False)]
    pP, pG = _to_pcd(pred), _to_pcd(gt)
    d_pg = np.asarray(pP.compute_point_cloud_distance(pG))
    d_gp = np.asarray(pG.compute_point_cloud_distance(pP))
    out: dict[str, float] = {}
    for t in thresholds:
        precision = float((d_pg < t).mean())
        recall = float((d_gp < t).mean())
        f = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        out[f"P@{t}"] = precision
        out[f"R@{t}"] = recall
        out[f"F@{t}"] = f
    return out


def pcd_stats(pts: np.ndarray) -> dict:
    return {
        "n_points": int(len(pts)),
        "bbox_size": [round(float(v), 4) for v in (pts.max(0) - pts.min(0))],
        "centroid": [round(float(v), 4) for v in pts.mean(0)],
    }


# ----------------------------- Pose metrics --------------------------------

def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Sim(3) alignment: find scale s, rotation R, translation t s.t. dst ≈ s R src + t.

    src, dst: Nx3.
    """
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    var_s = (s_c ** 2).sum() / len(src)
    cov = d_c.T @ s_c / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[-1, -1] = -1
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / max(var_s, 1e-12)
    t = mu_d - s * R @ mu_s
    return float(s), R, t


def trajectory_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """ATE (after Sim(3) alignment) + RPE on consecutive frames."""
    n = min(len(pred), len(gt))
    pred, gt = pred[:n], gt[:n]
    src = pred[:, :3, 3]
    dst = gt[:, :3, 3]
    s, R, t = umeyama_sim3(src, dst)
    src_aligned = (s * (R @ src.T)).T + t
    ate_per_frame = np.linalg.norm(src_aligned - dst, axis=1)

    # RPE: relative pose between frame i and i+1, then compare pred vs gt deltas
    def rel(T0, T1):
        return np.linalg.inv(T0) @ T1
    rpe_t, rpe_r = [], []
    for i in range(n - 1):
        rp = rel(pred[i], pred[i + 1])
        rg = rel(gt[i], gt[i + 1])
        # translation delta length, scaled to align
        rpe_t.append(abs(s * np.linalg.norm(rp[:3, 3]) - np.linalg.norm(rg[:3, 3])))
        # rotation error: angle of R_pred R_gt^T
        Rdiff = rp[:3, :3] @ rg[:3, :3].T
        cos = max(-1.0, min(1.0, 0.5 * (np.trace(Rdiff) - 1.0)))
        rpe_r.append(np.degrees(np.arccos(cos)))
    rpe_t = np.array(rpe_t) if rpe_t else np.array([0.0])
    rpe_r = np.array(rpe_r) if rpe_r else np.array([0.0])
    return {
        "n_frames": int(n),
        "sim3_scale": s,
        "ate_rmse_m": float(np.sqrt((ate_per_frame ** 2).mean())),
        "ate_mean_m": float(ate_per_frame.mean()),
        "ate_median_m": float(np.median(ate_per_frame)),
        "rpe_trans_rmse_m": float(np.sqrt((rpe_t ** 2).mean())),
        "rpe_rot_rmse_deg": float(np.sqrt((rpe_r ** 2).mean())),
    }


# ----------------------------- Top-level -----------------------------------

def evaluate(
    pred_ply: Path | None = None,
    gt_ply: Path | None = None,
    pred_poses: Path | None = None,
    gt_poses: Path | None = None,
    thresholds: tuple[float, ...] = (0.01, 0.05, 0.1),
) -> dict:
    out: dict = {}
    if pred_ply is not None:
        pred = load_ply(pred_ply)
        out["pred_stats"] = pcd_stats(pred)
        if gt_ply is not None:
            gt = load_ply(gt_ply)
            out["gt_stats"] = pcd_stats(gt)
            out["chamfer"] = chamfer(pred, gt)
            out["fscore"] = fscore(pred, gt, thresholds=thresholds)
    if pred_poses is not None and gt_poses is not None:
        out["trajectory"] = trajectory_metrics(load_poses(pred_poses), load_poses(gt_poses))
    return out


def format_markdown(tag: str, res: dict) -> str:
    lines = [f"# Evaluation — {tag}", ""]
    if "pred_stats" in res:
        s = res["pred_stats"]
        lines.append(f"**Predicted:** {s['n_points']:,} points, bbox {s['bbox_size']}, centroid {s['centroid']}.")
    if "gt_stats" in res:
        s = res["gt_stats"]
        lines.append(f"**Ground truth:** {s['n_points']:,} points, bbox {s['bbox_size']}.")
    if "chamfer" in res:
        c = res["chamfer"]
        lines += ["", "## Geometry", "",
                  "| metric | value |", "|---|---:|",
                  f"| Chamfer-L1 | {c['chamfer_l1']:.4f} |",
                  f"| Chamfer-L2 | {c['chamfer_l2']:.4f} |",
                  f"| Accuracy (mean) | {c['accuracy_mean']:.4f} |",
                  f"| Accuracy (median) | {c['accuracy_median']:.4f} |",
                  f"| Completeness (mean) | {c['completeness_mean']:.4f} |",
                  f"| Completeness (median) | {c['completeness_median']:.4f} |"]
    if "fscore" in res:
        lines += ["", "## F-score", "",
                  "| threshold | precision | recall | F |", "|---:|---:|---:|---:|"]
        keys = sorted({k.split("@")[1] for k in res["fscore"]}, key=float)
        for k in keys:
            p = res["fscore"][f"P@{k}"]
            r = res["fscore"][f"R@{k}"]
            f = res["fscore"][f"F@{k}"]
            lines.append(f"| {k} | {p:.3f} | {r:.3f} | {f:.3f} |")
    if "trajectory" in res:
        t = res["trajectory"]
        lines += ["", "## Trajectory (after Sim(3) alignment)", "",
                  f"- N frames: {t['n_frames']}",
                  f"- Sim(3) scale: {t['sim3_scale']:.4f}",
                  f"- ATE RMSE: {t['ate_rmse_m']:.4f} m",
                  f"- ATE mean: {t['ate_mean_m']:.4f} m",
                  f"- ATE median: {t['ate_median_m']:.4f} m",
                  f"- RPE trans RMSE: {t['rpe_trans_rmse_m']:.4f} m",
                  f"- RPE rot RMSE: {t['rpe_rot_rmse_deg']:.3f} deg"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path)
    ap.add_argument("--gt", type=Path)
    ap.add_argument("--pred-poses", type=Path)
    ap.add_argument("--gt-poses", type=Path)
    ap.add_argument("--thresholds", nargs="+", type=float, default=[0.01, 0.05, 0.1])
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--tag", type=str, default="eval")
    args = ap.parse_args()

    if not any([args.pred, args.pred_poses]):
        ap.error("supply --pred and/or --pred-poses")

    res = evaluate(
        pred_ply=args.pred,
        gt_ply=args.gt,
        pred_poses=args.pred_poses,
        gt_poses=args.gt_poses,
        thresholds=tuple(args.thresholds),
    )
    print(format_markdown(args.tag, res))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"tag": args.tag, **res}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
