"""Option B: extract a clean textured mesh from a trained 2DGS checkpoint.

Renders per-training-view RGB + median (surface) depth from the 2D Gaussians,
then TSDF-fuses them with Open3D into a watertight, surface-aligned mesh.

    python scripts/mesh_from_2dgs.py \
        --ckpt outputs/sculpture_2dgs/ckpts/ckpt_29999_rank0.pt \
        --data_dir data/dronesplat/Sculpture --data_factor 4 \
        --out outputs/sculpture_2dgs/mesh.ply

Runs on the 6 GB RTX 4050 (one view rendered at a time). The 2DGS surfels give
flat, plane-aligned surfaces — that's the "not dots, real surfaces" deliverable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import open3d as o3d

# reuse the gsplat example's COLMAP loader + 2DGS rasterizer
EX = Path(__file__).resolve().parents[1] / "third_party" / "gsplat" / "examples"
sys.path.insert(0, str(EX))
from datasets.colmap import Parser, Dataset  # noqa: E402
from gsplat.rendering import rasterization_2dgs  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--data_factor", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--voxels", type=int, default=512, help="grid resolution across the scene extent")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- load gaussians ---------------------------------------------------- #
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    s = ckpt["splats"]
    means = s["means"].to(device)
    quats = s["quats"].to(device)
    scales = torch.exp(s["scales"]).to(device)
    opacities = torch.sigmoid(s["opacities"]).to(device)
    colors = torch.cat([s["sh0"], s["shN"]], dim=1).to(device)  # [N, K, 3]

    # --- cameras from COLMAP ---------------------------------------------- #
    # IMPORTANT: match the trainer's default (normalize_world_space=True), otherwise
    # cameras and the trained gaussians live in different coordinate frames.
    parser = Parser(data_dir=args.data_dir, factor=args.data_factor, normalize=True)
    dataset = Dataset(parser, split="train")

    cam_centers = parser.camtoworlds[:, :3, 3]
    extent = float(np.linalg.norm(cam_centers.max(0) - cam_centers.min(0)))
    voxel_length = extent / args.voxels
    sdf_trunc = 5.0 * voxel_length
    depth_trunc = extent * 2.0
    print(f"[mesh] scene extent={extent:.2f}  voxel={voxel_length:.4f}  sdf_trunc={sdf_trunc:.4f}")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for i in range(len(dataset)):
        data = dataset[i]
        K = data["K"].to(device).float()
        camtoworld = data["camtoworld"].to(device).float()
        H, W = int(data["image"].shape[0]), int(data["image"].shape[1])
        viewmat = torch.linalg.inv(camtoworld)[None]  # [1,4,4] world->cam

        with torch.no_grad():
            rgb, _, _, _, _, median, _ = rasterization_2dgs(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=colors, viewmats=viewmat, Ks=K[None], width=W, height=H,
                sh_degree=args.sh_degree, render_mode="RGB+D",
                near_plane=0.01, far_plane=depth_trunc,
            )
        color_np = (rgb[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        depth_np = median.squeeze().cpu().numpy().astype(np.float32)  # [H,W], world units

        o3d_color = o3d.geometry.Image(np.ascontiguousarray(color_np))
        o3d_depth = o3d.geometry.Image(np.ascontiguousarray(depth_np))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth, depth_scale=1.0, depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        intr = o3d.camera.PinholeCameraIntrinsic(
            W, H, float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        )
        volume.integrate(rgbd, intr, viewmat[0].cpu().numpy())
        print(f"[mesh] integrated view {i+1}/{len(dataset)}", end="\r")

    print("\n[mesh] extracting triangle mesh ...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(out), mesh)
    # also drop a .obj for CAD/Blender import
    o3d.io.write_triangle_mesh(str(out.with_suffix(".obj")), mesh)
    print(f"[mesh] wrote {out}  ({len(mesh.vertices)} verts, {len(mesh.triangles)} tris)")


if __name__ == "__main__":
    main()
