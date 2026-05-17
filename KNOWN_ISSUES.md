# Known issues

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

Not downloaded yet. Phase 1 will pull them from the respective HuggingFace / project pages. Smoke test currently only confirms imports.
