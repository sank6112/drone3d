"""Phase 0 smoke test.

For each backbone: try to import, optionally load the checkpoint, report VRAM
and time. Failures don't abort the whole test — we report per-backbone and
exit non-zero only if ALL fail.

Usage:
    python scripts/00_smoke_test.py
    python scripts/00_smoke_test.py --models dust3r mast3r vggt
    python scripts/00_smoke_test.py --no-load   # skip weight load, imports only
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

# Make src/ importable without installing the project
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from drone3d import paths as p  # noqa: E402


def gpu_mb() -> float:
    return torch.cuda.memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0


def env_report() -> None:
    print("=" * 60)
    print(f"PyTorch:    {torch.__version__}")
    print(f"CUDA:       {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        d = torch.cuda.get_device_properties(0)
        print(f"Device:     {d.name}  ({d.total_memory / 1024**3:.2f} GB)")
    print("=" * 60)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def smoke_dust3r(load: bool) -> None:
    p.inject("mast3r")  # mast3r bundles dust3r
    from dust3r.model import AsymmetricCroCo3DStereo  # type: ignore

    print("  import: ok")
    if not load:
        return
    ckpt = p.checkpoint("DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth")
    t0 = time.time()
    model = AsymmetricCroCo3DStereo.from_pretrained(str(ckpt)).to(_device())
    model.eval()
    print(f"  load:   {time.time() - t0:.1f}s   VRAM: {gpu_mb():.0f} MB")


def smoke_mast3r(load: bool) -> None:
    p.inject("mast3r")
    from mast3r.model import AsymmetricMASt3R  # type: ignore

    print("  import: ok")
    if not load:
        return
    ckpt = p.checkpoint("MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
    t0 = time.time()
    model = AsymmetricMASt3R.from_pretrained(str(ckpt)).to(_device())
    model.eval()
    print(f"  load:   {time.time() - t0:.1f}s   VRAM: {gpu_mb():.0f} MB")


def smoke_vggt(load: bool) -> None:
    from vggt.models.vggt import VGGT  # type: ignore

    print("  import: ok")
    if not load:
        return
    # VGGT-1B is ~4 GB at fp32. On a 6 GB GPU, load to CPU first and let user
    # decide later whether to move to GPU; here we just confirm load works.
    t0 = time.time()
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model.eval()
    print(f"  load:   {time.time() - t0:.1f}s   (kept on CPU; ~4 GB at fp32)")


def smoke_point3r(load: bool) -> None:
    # Point3R reuses the `dust3r` module name. Inject Point3R BEFORE mast3r
    # in sys.path so its version wins. To avoid module-cache collisions we
    # also pop any pre-imported `dust3r*` modules.
    import sys as _sys

    for k in [k for k in _sys.modules if k == "dust3r" or k.startswith("dust3r.")]:
        del _sys.modules[k]
    p.inject("Point3R")
    from dust3r.point3r import Point3R  # type: ignore

    print("  import: ok")
    if not load:
        return
    print("  load:   skipped (no local checkpoint yet; download in Phase 1)")


def smoke_streamvggt(load: bool) -> None:
    p.inject("StreamVGGT")
    import importlib

    # Try common module locations until one works
    found = None
    for modname in ("streamvggt.models", "src.streamvggt.models", "lib.streamvggt", "src.models"):
        try:
            importlib.import_module(modname)
            found = modname
            break
        except Exception:
            continue
    if not found:
        raise ImportError("could not locate StreamVGGT module; inspect third_party/StreamVGGT")
    print(f"  import: ok ({found})")
    if not load:
        return
    print("  load:   skipped (no local checkpoint yet; download in Phase 1)")


def smoke_cut3r(load: bool) -> None:
    p.inject("cut3r")
    # CUT3R extends DUSt3R; its top-level demo.py imports from src/.
    import importlib

    for modname in ("src.dust3r.model", "dust3r.model"):
        try:
            importlib.import_module(modname)
            print(f"  import: ok ({modname})")
            break
        except Exception:
            continue
    else:
        raise ImportError("could not locate CUT3R model module")
    if not load:
        return
    print("  load:   skipped (no local checkpoint yet; download in Phase 1)")


SMOKE: dict[str, callable] = {
    "dust3r": smoke_dust3r,
    "mast3r": smoke_mast3r,
    "vggt": smoke_vggt,
    "point3r": smoke_point3r,
    "streamvggt": smoke_streamvggt,
    "cut3r": smoke_cut3r,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(SMOKE.keys()), choices=list(SMOKE.keys()))
    ap.add_argument("--no-load", action="store_true", help="import-only, don't load weights")
    args = ap.parse_args()

    env_report()
    fails: list[str] = []
    for name in args.models:
        print(f"[{name}]")
        try:
            SMOKE[name](load=not args.no_load)
        except Exception:
            traceback.print_exc(limit=2)
            fails.append(name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()

    if len(fails) == len(args.models):
        print(f"ALL FAILED: {fails}")
        return 1
    if fails:
        print(f"PARTIAL: {len(args.models) - len(fails)}/{len(args.models)} passed; failed: {fails}")
        return 0
    print(f"All {len(args.models)} smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
