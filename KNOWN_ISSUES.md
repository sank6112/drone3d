# Known issues

## CroCo-family repos are NOT pip-installable (`pip install -e` fails)

`mast3r/`, `Point3R/`, `StreamVGGT/`, `monst3r/` ship no top-level `setup.py`/`pyproject.toml`, so `pip install -e third_party/mast3r` fails with *"does not appear to be a Python project"* (seen again during the 2026-09-12 restore). This is expected — they are used as plain folders via `src/drone3d/paths.py` (`P.inject("mast3r")` prepends the right subpaths to `sys.path`). Only **VGGT** is genuinely pip-installable. Do **not** rely on a pip install for the others; `SETUP.md` reflects this.

## Module-name collisions across CroCo-family backbones

All five CroCo-derived repos (`mast3r/`, `cut3r/`, `Point3R/`, `StreamVGGT/`, `monst3r/`) ship modules named `dust3r`, `croco`, and `models`. Once one is imported in a Python process, the others' versions become uninportable because `sys.modules` already holds different code under the same key.

**Workaround:** run each backbone in its own subprocess. The Phase 1 benchmark script will do this — `scripts/01_baseline_inference.py` will `subprocess.run` per backbone with a clean Python interpreter.

**What works in-process:** DUSt3R + MASt3R (they share the `mast3r/dust3r` namespace by design), and VGGT (separately pip-installed).

## RoPE2D CUDA kernel not built

You will see:
```
Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch version instead
```
Cosmetic — uses a PyTorch fallback. We can build the CUDA kernel later if inference time becomes a bottleneck:
```bash
cd third_party/mast3r/dust3r/croco/models/curope && python setup.py build_ext --inplace
```

## VGGT-1B at fp32

VGGT-1B occupies ~4 GB. On the laptop's 6 GB GPU it leaves ~2 GB for inputs/activations — fine for ≤3 images at 224 px. For full-quality inference we'll either move to fp16 or run on cloud GPU. Smoke test loads it to CPU for safety.

## Point3R / StreamVGGT / CUT3R checkpoints

**Note (2026-09-12 restore):** `checkpoints/` is gitignored and was lost with the local folder. Only the deployable MASt3R stack (DUSt3R + MASt3R metric) is being re-downloaded now; the heavier backbones below are re-downloaded on demand for cloud/research runs. Original set:
- `checkpoints/point3r.pth` (3.1 GB, Google Drive)
- `checkpoints/cut3r_512_dpt_4_64.pth` (3.0 GB, Google Drive)
- `checkpoints/streamvggt_hf/` (5 GB safetensors + 5 GB .pth, HuggingFace)
- `checkpoints/monst3r_hf/` (2.2 GB, HuggingFace)
- `checkpoints/aerial_mast3r_hf/` (2.7 GB, HuggingFace)

## CUT3R demo at non-512 image size

`third_party/cut3r/demo.py --size 224` and `--size 384` trigger:
```
File "src/croco/models/pos_embed.py", line 172, in forward
    D, int(positions.max()) + 1, tokens.device, tokens.dtype
RuntimeError: CUDA error: device-side assert triggered
```

The 512-DPT checkpoint expects 512-resolution positional embeddings. Two paths:
1. Download the 224-linear checkpoint (`cut3r_224_linear_4.pth`) and use it at 224.
2. Run at 512 — won't fit on 6 GB GPU; needs cloud.

Deferred until cloud runs. Documented but not blocking.

## Point3R / StreamVGGT inference

Neither ships a clean `demo.py` for arbitrary input. Point3R has `eval/relpose/launch.py` for the relative-pose benchmark; StreamVGGT has `demo_gradio.py` (UI). Both need a custom wrapper script for our pipeline. Deferred to Phase 4 cloud runs.
