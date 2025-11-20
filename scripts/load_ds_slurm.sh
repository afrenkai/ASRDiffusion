#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64g
#SBATCH -J "s"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH -e "asr_diff_%j_err.txt"
#SBATCH --gres=gpu:1
#SBATCH -C H100|L40S|H100
#SBATCH --output=asr_diff_%j_log.txt
set -e


module load cuda

if ! command -v uv >/dev/null 2>&1; then
  echo "uv could not be found, downloading"
  wget -qO- https://astral.sh/uv/install.sh | sh
fi

#assumes this is run from scripts/. if base, just remove the ../
if [ ! -d "../Data/" ]; then
  uv run -m dataset.ds_utils
  echo "fetched dataset"
else
  echo "already have the dataset, skipping ..."
fi

#TODO: add like training code, preprocessing, etc... 
