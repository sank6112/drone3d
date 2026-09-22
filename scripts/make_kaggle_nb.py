"""Generates notebooks/drone3d_kaggle.ipynb — the end-to-end Kaggle notebook.
Run:  python scripts/make_kaggle_nb.py
"""
import json, pathlib

def md(*s):  return {"cell_type": "markdown", "metadata": {}, "source": list(s)}
def code(*s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": list(s)}

cells = [
 md("# drone3d — quality-max reconstruction on Kaggle\n",
    "**Before running:** Settings → Accelerator = **GPU P100**, and **Internet = ON**.\n",
    "Everything is fetched from public sources; nothing is uploaded. Use *Save & Run All (Commit)* for 9 h background runs."),

 md("## 1. Clone the repo"),
 code("import os, subprocess, glob, json\n",
      "os.chdir('/kaggle/working')\n",
      "if not os.path.isdir('drone3d'):\n",
      "    subprocess.run(['git','clone','--depth','1',\n",
      "                    'https://github.com/sank6112/drone3d'], check=True)\n",
      "os.chdir('/kaggle/working/drone3d'); print('cwd:', os.getcwd())"),

 md("## 2. Environment + data setup\n",
    "Set `RUBBLE=1` to also pull Mill-19 Rubble (9.2 GB, the big aerial scene)."),
 code("os.environ['RUBBLE']='0'   # '1' to also fetch Rubble\n",
      "os.environ['MAST3R']='1'\n",
      "!bash scripts/setup_kaggle.sh"),

 md("## 2b. Smoke test (~2 min) — confirm GPU + deps before the long runs\n",
    "Runs 500 steps on Sculpture at low res. If a `.splat` appears, the environment is good."),
 code("subprocess.run(['bash','scripts/train_quality.sh',\n",
      "                'data/dronesplat/Sculpture','4','outputs/_smoke','300000','500',\n",
      "                '--save-steps','500','--eval-steps','500'])\n",
      "ok = os.path.exists('outputs/_smoke/model.splat')\n",
      "print('SMOKE TEST', 'PASSED ✅' if ok else 'FAILED ❌ — check the log above')"),

 md("## 3. Ablation on Sculpture — which quality knobs actually help?\n",
    "Short 7k runs at factor 2 so we can compare knobs fast, then lock the winner."),
 code("ABL = {\n",
      "  'A0_baseline':            [],\n",
      "  'A1_aa':                  ['--antialiased'],\n",
      "  'A2_aa_app':              ['--antialiased','--app_opt'],\n",
      "  'A3_aa_app_bilat':        ['--antialiased','--app_opt','--use_bilateral_grid'],\n",
      "  'A4_aa_app_pose':         ['--antialiased','--app_opt','--pose_opt'],\n",
      "}\n",
      "for name, flags in ABL.items():\n",
      "    rd = f'outputs/abl_{name}'\n",
      "    subprocess.run(['bash','scripts/train_quality.sh',\n",
      "                    'data/dronesplat/Sculpture','2',rd,'1500000','7000',*flags])"),

 md("### Ablation results"),
 code("rows=[]\n",
      "for d in sorted(glob.glob('outputs/abl_*')):\n",
      "    js=sorted(glob.glob(d+'/stats/val_step*.json'))\n",
      "    if not js: continue\n",
      "    v=json.load(open(js[-1]))\n",
      "    rows.append((d.split('abl_')[1], v['psnr'], v['ssim'], v['lpips'], v['num_GS']))\n",
      "print(f\"{'config':24s}{'PSNR':>8}{'SSIM':>8}{'LPIPS':>8}{'#GS':>11}\")\n",
      "for r in rows: print(f'{r[0]:24s}{r[1]:8.2f}{r[2]:8.3f}{r[3]:8.3f}{r[4]:11,}')\n",
      "if rows:\n",
      "    best=min(rows,key=lambda r:r[3]); print('\\nBEST by LPIPS:', best[0])"),

 md("## 4. Final full-res Sculpture (factor 1, 30k) with the best knobs\n",
    "Edit `BEST_FLAGS` from the ablation above if a different combo won."),
 code("BEST_FLAGS=['--antialiased','--app_opt','--pose_opt']\n",
      "subprocess.run(['bash','scripts/train_quality.sh',\n",
      "                'data/dronesplat/Sculpture','1','outputs/sculpture_final','2500000','30000',*BEST_FLAGS])"),

 md("## 5. Product path — MASt3R poses from raw images (no COLMAP)\n",
    "Kaggle's 32 GB RAM allows more frames than the 6 GB laptop; `--pose_opt` refines the MASt3R poses."),
 code("subprocess.run(['python','scripts/mast3r_to_colmap.py',\n",
      "                '--frames','data/dronesplat/Sculpture/images',\n",
      "                '--out','outputs/sculpture_mast3r_colmap','--max-frames','20'])\n",
      "subprocess.run(['bash','scripts/train_quality.sh',\n",
      "                'outputs/sculpture_mast3r_colmap','1','outputs/sculpture_mast3r_final',\n",
      "                '2000000','30000','--antialiased','--app_opt','--pose_opt'])"),

 md("## 6. Rubble — big aerial scene (needs RUBBLE=1 in step 2)"),
 code("if os.path.isdir('data/rubble/rubble-pixsfm/train/rgbs'):\n",
      "    subprocess.run(['python','scripts/meganerf_to_colmap.py',\n",
      "                    '--src','data/rubble/rubble-pixsfm/train','--out','data/rubble_sub','--n','500','--long','1024'])\n",
      "    subprocess.run(['bash','scripts/train_quality.sh',\n",
      "                    'data/rubble_sub','1','outputs/rubble_final','3000000','40000',\n",
      "                    '--antialiased','--app_opt','--use_bilateral_grid'])\n",
      "else:\n",
      "    print('Rubble not present — set RUBBLE=1 in step 2 and re-run setup.')"),

 md("## 7. Collect deliverables\n",
    "`.splat` files (drag into supersplat.com) + a metrics summary. Download from the Output panel."),
 code("os.makedirs('/kaggle/working/deliverables', exist_ok=True)\n",
      "for f in glob.glob('outputs/*/model.splat'):\n",
      "    dst='/kaggle/working/deliverables/'+f.split('/')[-2]+'.splat'\n",
      "    subprocess.run(['cp',f,dst])\n",
      "subprocess.run(['python','scripts/write_summary.py'])\n",
      "print(open('outputs/SUMMARY.md').read())\n",
      "print('\\nDeliverables:'); print('\\n'.join(sorted(glob.glob('/kaggle/working/deliverables/*'))))"),
]

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}

out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "drone3d_kaggle.ipynb"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
