"""Phase 0 smoke test.

Confirms the conda env has CUDA, can load each foundation model, and reports
memory/time per model. Run after `conda env create -f environment.yml` and the
clones in SETUP.md.

Usage:
    python scripts/00_smoke_test.py
    python scripts/00_smoke_test.py --models dust3r mast3r vggt
"""

from __future__ import annotations

import argparse
import importlib
import time
import traceback

import torch


def gpu_mem_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 * 1024)


def report_env() -> None:
    print("=" * 60)
    print("Environment")
    print("=" * 60)
    print(f"PyTorch:    {torch.__version__}")
    print(f"CUDA avail: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device:     {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"VRAM:       {props.total_memory / (1024**3):.2f} GB")
    print()


def try_import(modname: str) -> tuple[bool, str]:
    try:
        importlib.import_module(modname)
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def smoke_dust3r() -> None:
    from dust3r.model import AsymmetricCroCo3DStereo

    t0 = time.time()
    ckpt = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    model = AsymmetricCroCo3DStereo.from_pretrained(ckpt)
    model = model.cuda() if torch.cuda.is_available() else model
    print(f"  load: {time.time() - t0:.1f}s   VRAM: {gpu_mem_mb():.0f} MB")


def smoke_mast3r() -> None:
    from mast3r.model import AsymmetricMASt3R

    t0 = time.time()
    ckpt = "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
    model = AsymmetricMASt3R.from_pretrained(ckpt)
    model = model.cuda() if torch.cuda.is_available() else model
    print(f"  load: {time.time() - t0:.1f}s   VRAM: {gpu_mem_mb():.0f} MB")


def smoke_vggt() -> None:
    from vggt.models.vggt import VGGT

    t0 = time.time()
    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model = model.cuda() if torch.cuda.is_available() else model
    print(f"  load: {time.time() - t0:.1f}s   VRAM: {gpu_mem_mb():.0f} MB")


SMOKE = {
    "dust3r": smoke_dust3r,
    "mast3r": smoke_mast3r,
    "vggt": smoke_vggt,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=list(SMOKE.keys()), choices=list(SMOKE.keys()))
    args = p.parse_args()

    report_env()
    fails: list[str] = []
    for name in args.models:
        print(f"[{name}]")
        try:
            SMOKE[name]()
            print("  OK\n")
        except Exception:
            traceback.print_exc()
            print(f"  FAIL\n")
            fails.append(name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if fails:
        print(f"FAILED: {fails}")
        return 1
    print("All smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
