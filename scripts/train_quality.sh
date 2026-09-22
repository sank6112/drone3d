#!/bin/bash
# Quality-max gsplat trainer (16 GB+ GPUs, e.g. Kaggle P100). Wraps MCMC 3DGS
# with the quality knobs and exports a viewer-ready .ply/.splat + prints metrics.
#
# Usage:
#   scripts/train_quality.sh <data_dir> <factor> <result_dir> <cap> <steps> [extra gsplat flags...]
# Example (full quality):
#   scripts/train_quality.sh data/dronesplat/Sculpture 1 outputs/sculpt_q 2000000 30000 \
#       --antialiased --app_opt --use_bilateral_grid --pose_opt
#
# The ablation in the Kaggle notebook calls this repeatedly with different
# trailing flags so we can measure which knobs actually help on our data.
set -uo pipefail
here() { cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd; }
ROOT=$(here); cd "$ROOT"
EXABS=$(readlink -f third_party/gsplat/examples)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA=$1 FACTOR=$2 RD=$3 CAP=$4 STEPS=$5; shift 5
EXTRA=("$@")

# gsplat's COLMAP loader needs images_<factor> to exist for factor>1
if [ "$FACTOR" -gt 1 ] && [ ! -e "$DATA/images_$FACTOR" ]; then ln -sfn images "$DATA/images_$FACTOR"; fi
mkdir -p "$RD"
ADATA=$(readlink -f "$DATA"); ARD=$(readlink -f "$RD")

echo ">>> train_quality: data=$DATA factor=$FACTOR cap=$CAP steps=$STEPS knobs=[${EXTRA[*]:-none}]"
python "$EXABS/simple_trainer.py" mcmc \
  --strategy.cap-max "$CAP" --data_dir "$ADATA" --data_factor "$FACTOR" \
  --max-steps "$STEPS" --disable_viewer --disable-video \
  --result_dir "$ARD" "${EXTRA[@]}"
rc=$?
[ $rc -ne 0 ] && { echo "TRAIN FAILED (rc=$rc) for $RD"; exit $rc; }

# export a viewer-ready splat/ply from the final checkpoint
CK=$(ls -t "$RD"/ckpts/ckpt_*_rank0.pt 2>/dev/null | head -1)
if [ -n "$CK" ]; then
  PYTHONPATH="$EXABS" python - "$CK" "$RD" <<'PY'
import sys, torch
from gsplat.exporter import export_splats
ck, rd = sys.argv[1], sys.argv[2]
s = torch.load(ck, map_location="cpu", weights_only=False)["splats"]
for fmt, ext in [("ply","ply"),("splat","splat")]:
    export_splats(means=s["means"], scales=s["scales"], quats=s["quats"],
                  opacities=s["opacities"], sh0=s["sh0"], shN=s["shN"],
                  format=fmt, save_to=f"{rd}/model.{ext}")
print(f"exported {rd}/model.ply + .splat")
PY
fi

# echo final held-out metrics for the ablation table
V=$(ls -t "$RD"/stats/val_step*.json 2>/dev/null | head -1)
[ -n "$V" ] && echo "METRICS $RD :: $(cat "$V")"
