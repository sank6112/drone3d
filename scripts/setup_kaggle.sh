#!/bin/bash
# One-shot environment + data setup for drone3d on Kaggle (or any fresh cloud box).
# Run from the repo root:  bash scripts/setup_kaggle.sh
# Fetches everything from public sources — nothing is uploaded from a laptop.
#
# Env toggles:
#   RUBBLE=1   also download Mill-19 Rubble (9.2 GB)   [default: 0]
#   MAST3R=1   also clone MASt3R + download its checkpoints [default: 1]
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
RUBBLE=${RUBBLE:-0}; MAST3R=${MAST3R:-1}
mkdir -p third_party checkpoints data outputs

echo "==== [1/5] gsplat + trainer deps ===="
pip -q install gsplat==1.5.3 gdown || pip -q install gsplat gdown
[ -d third_party/gsplat ] || git clone --depth 1 --branch v1.5.3 https://github.com/nerfstudio-project/gsplat third_party/gsplat
# fused-ssim / fused-bilagrid are CUDA extensions -> must skip build isolation (torch hidden otherwise)
pip -q install --no-build-isolation -r third_party/gsplat/examples/requirements.txt

echo "==== [2/5] MASt3R (product front-end) ===="
if [ "$MAST3R" = "1" ]; then
  [ -d third_party/mast3r ] || git clone --recursive https://github.com/naver/mast3r third_party/mast3r
  pip -q install -r third_party/mast3r/requirements.txt 2>/dev/null || true
  pip -q install -r third_party/mast3r/dust3r/requirements.txt 2>/dev/null || true
  NAVER=https://download.europe.naverlabs.com/ComputerVision
  wget -c -q --show-progress -P checkpoints \
    "$NAVER/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
  wget -c -q --show-progress -P checkpoints \
    "$NAVER/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
fi

echo "==== [3/5] DroneSplat scenes (Sculpture + Simingshan, ~103 MB) ===="
[ -d data/dronesplat/Sculpture ] || gdown --folder -q \
  https://drive.google.com/drive/folders/1DWm-foUQC2QBsrr3QC6Tx8bDmWTfgAzu -O data/dronesplat

echo "==== [4/5] Mill-19 Rubble (optional, 9.2 GB) ===="
if [ "$RUBBLE" = "1" ]; then
  [ -f data/rubble-pixsfm.tgz ] || wget -c -q --show-progress \
    https://storage.cmusatyalab.org/mega-nerf-data/rubble-pixsfm.tgz -O data/rubble-pixsfm.tgz
  [ -d data/rubble/rubble-pixsfm ] || { mkdir -p data/rubble && tar xzf data/rubble-pixsfm.tgz -C data/rubble; }
fi

echo "==== [5/5] sanity check ===="
python - <<'PY'
import torch, gsplat
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("gsplat", gsplat.__version__)
PY
echo "setup done."
