"""Mean train-view PSNR for a gsplat checkpoint against a COLMAP model.
Prints a single line 'TRAIN_PSNR=<val>'. Used to validate poses (a wrong pose
convention can't fit the training views, so low train PSNR == bad conversion).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch

EX = Path(__file__).resolve().parents[1] / "third_party" / "gsplat" / "examples"
sys.path.insert(0, str(EX))
from datasets.colmap import Parser, Dataset          # noqa: E402
from gsplat.rendering import rasterization           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--data_factor", type=int, default=1)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    s = torch.load(args.ckpt, map_location=dev, weights_only=False)["splats"]
    means, quats = s["means"], s["quats"]
    scales, opac = torch.exp(s["scales"]), torch.sigmoid(s["opacities"])
    colors = torch.cat([s["sh0"], s["shN"]], 1)
    p = Parser(data_dir=args.data_dir, factor=args.data_factor, normalize=True)
    ds = Dataset(p, split="train")
    ps = []
    for i in range(len(ds)):
        d = ds[i]; K = d["K"].to(dev).float(); c2w = d["camtoworld"].to(dev).float()
        gt = d["image"].to(dev).float() / 255.0; H, W = gt.shape[:2]
        with torch.no_grad():
            r, _, _ = rasterization(means, quats, scales, opac, colors,
                                    torch.linalg.inv(c2w)[None], K[None], W, H, sh_degree=3)
        mse = ((r[0, ..., :3].clamp(0, 1) - gt) ** 2).mean()
        ps.append(float(-10 * torch.log10(mse)))
    print(f"TRAIN_PSNR={np.mean(ps):.2f}")


if __name__ == "__main__":
    main()
