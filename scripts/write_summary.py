"""Aggregate overnight results into outputs/SUMMARY.md."""
from __future__ import annotations
import glob, json, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def latest_val(d: Path):
    js = sorted(glob.glob(str(d / "stats" / "val_step*.json")))
    if not js:
        return None
    return json.load(open(js[-1]))


def main():
    lines = ["# drone3d overnight summary", "", f"_generated {time.ctime()}_", ""]
    lines += ["| result dir | PSNR | SSIM | LPIPS | #GS | splat | mesh |",
              "|---|---|---|---|---|---|---|"]
    for d in sorted(OUT.iterdir()):
        if not d.is_dir():
            continue
        v = latest_val(d)
        has_splat = (d / "model.splat").exists() or bool(glob.glob(str(d / "*.splat")))
        has_mesh = (d / "mesh.ply").exists()
        if v is None and not (has_splat or has_mesh):
            continue
        psnr = f"{v['psnr']:.2f}" if v else "-"
        ssim = f"{v['ssim']:.3f}" if v else "-"
        lpips = f"{v['lpips']:.3f}" if v else "-"
        ngs = f"{v['num_GS']:,}" if v else "-"
        lines.append(f"| {d.name} | {psnr} | {ssim} | {lpips} | {ngs} | "
                     f"{'✅' if has_splat else '—'} | {'✅' if has_mesh else '—'} |")

    lines += ["", "## Artifacts", ""]
    for pat in ["*/model.splat", "*/model.ply", "*/mesh.ply", "*/mesh.obj", "*/videos/*.mp4"]:
        for f in sorted(glob.glob(str(OUT / pat))):
            sz = os.path.getsize(f) / 1e6
            lines.append(f"- `{os.path.relpath(f, ROOT)}`  ({sz:.1f} MB)")

    (OUT / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("wrote", OUT / "SUMMARY.md")


if __name__ == "__main__":
    main()
