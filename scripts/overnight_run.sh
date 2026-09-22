#!/bin/bash
# Autonomous overnight runner for drone3d. Each stage is independent: a failure
# in one is logged and the next still runs. Detached from any Claude session.
# Results + metrics land in outputs/ and outputs/SUMMARY.md.
set -uo pipefail
cd /home/rudra/projects/drone3d
mkdir -p outputs
LOG=outputs/overnight.log
exec >>"$LOG" 2>&1

CONDA="conda run -n drone3d"
EXABS=$(readlink -f third_party/gsplat/examples)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

banner(){ echo; echo "=================================================="; echo ">>> $* :: $(date)"; echo "=================================================="; }
banner "OVERNIGHT RUN STARTED"

# train an mcmc splat. args: <data_dir> <factor> <result_dir> <cap> [extra flags...]
train_mcmc(){
  local dd=$1 factor=$2 rd=$3 cap=$4; shift 4
  if [ "$factor" -gt 1 ] && [ ! -e "$dd/images_$factor" ]; then ln -sfn images "$dd/images_$factor"; fi
  local add ard; add=$(readlink -f "$dd"); mkdir -p "$rd"; ard=$(readlink -f "$rd")
  $CONDA env PYTHONPATH="$EXABS" python "$EXABS/simple_trainer.py" mcmc \
      --strategy.cap-max "$cap" --data_dir "$add" --data_factor "$factor" \
      --disable_viewer --result_dir "$ard" "$@"
}
export_ply(){  # <result_dir>
  local rd=$1 ck
  ck=$(ls -t "$rd"/ckpts/ckpt_*_rank0.pt 2>/dev/null | head -1)
  [ -z "$ck" ] && { echo "no ckpt in $rd"; return 1; }
  $CONDA python -c "
import torch; from gsplat.exporter import export_splats
s=torch.load('$ck',map_location='cpu',weights_only=False)['splats']
for fmt,ext in [('ply','ply'),('splat','splat')]:
    export_splats(means=s['means'],scales=s['scales'],quats=s['quats'],opacities=s['opacities'],sh0=s['sh0'],shN=s['shN'],format=fmt,save_to='$rd/model.'+ext)
print('exported $rd/model.ply + .splat')"
}
train_psnr(){ $CONDA python scripts/train_psnr.py --ckpt "$(ls -t "$1"/ckpts/*_rank0.pt|head -1)" --data_dir "$2" --data_factor "${3:-1}"; }

# ---- STAGE A: MASt3R->gsplat glue (product path), Sculpture-10 [VALIDATED] ---
banner "STAGE A: MASt3R-posed splat (Sculpture-10)"
if [ -d outputs/sculpture_mast3r_colmap/sparse/0 ]; then
  train_mcmc outputs/sculpture_mast3r_colmap 1 outputs/sculpture_mast3r_3dgs 300000 --disable-video \
    && export_ply outputs/sculpture_mast3r_3dgs \
    && train_psnr outputs/sculpture_mast3r_3dgs outputs/sculpture_mast3r_colmap 1 \
    && echo "STAGE A OK" || echo "STAGE A FAILED"
else
  echo "STAGE A SKIPPED: exporter output missing"
fi

# ---- STAGE B: high-quality dense demo, Sculpture-42 @ factor 2 [GUARANTEED] --
banner "STAGE B: high-res Sculpture-42 splat (factor 2, with fly-through video)"
train_mcmc data/dronesplat/Sculpture 2 outputs/sculpture_hi_3dgs 800000 \
  && export_ply outputs/sculpture_hi_3dgs \
  && echo "STAGE B OK" || echo "STAGE B FAILED"

# ---- STAGE C: Rubble big-data demo (self-validating) ------------------------
banner "STAGE C: Rubble big-data demo"
TGZ=data/rubble-pixsfm.tgz
if [ ! -f "$TGZ" ]; then
  echo "STAGE C SKIPPED: $TGZ missing"
else
  [ -d data/rubble/rubble-pixsfm ] || { mkdir -p data/rubble && echo "extracting rubble..." && tar xzf "$TGZ" -C data/rubble; }
  TRAIN=$(find data/rubble -type d -name train -path '*rubble*' | head -1)
  if [ -z "$TRAIN" ] || [ ! -d "$TRAIN/rgbs" ]; then
    echo "STAGE C SKIPPED: no train/rgbs found"; find data/rubble -maxdepth 3 -type d | head
  else
    # probe: 60 imgs, 7k steps, validate the Mega-NeRF->COLMAP pose conversion
    $CONDA python scripts/meganerf_to_colmap.py --src "$TRAIN" --out data/rubble_probe --n 60 --long 640
    train_mcmc data/rubble_probe 1 outputs/rubble_probe 400000 --disable-video \
        --max-steps 7000 --eval-steps 7000 --save-steps 7000
    PR=$(train_psnr outputs/rubble_probe data/rubble_probe 1 | grep TRAIN_PSNR | cut -d= -f2)
    echo "STAGE C probe TRAIN_PSNR=$PR"
    ok=$(python3 -c "print(1 if float('${PR:-0}')>=17 else 0)" 2>/dev/null || echo 0)
    if [ "$ok" = "1" ]; then
      echo "probe good -> full Rubble run (300 imgs)"
      $CONDA python scripts/meganerf_to_colmap.py --src "$TRAIN" --out data/rubble_sub --n 300 --long 768
      train_mcmc data/rubble_sub 1 outputs/rubble_3dgs 1000000 --disable-video \
        && export_ply outputs/rubble_3dgs && echo "STAGE C OK" || echo "STAGE C FAILED (full train)"
    else
      echo "STAGE C STOPPED: probe train-PSNR too low ($PR) — Mega-NeRF axis convention likely differs; not wasting hours. Needs a convention fix."
    fi
  fi
fi

banner "WRITING SUMMARY"
$CONDA python scripts/write_summary.py || echo "summary failed"
banner "OVERNIGHT RUN FINISHED"
