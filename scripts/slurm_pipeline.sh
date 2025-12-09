#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64g
#SBATCH -J "asr_test_%j.txt"
#SBATCH -p short
#SBATCH -t 12:00:00
#SBATCH -e "asr_diff_%j_err.txt"
#SBATCH --gres=gpu:1
#SBATCH -C H100|L40S|H100
#SBATCH --output=asr_diff_%j_log.txt

set -e
#defaults to 2 as with the gres above, but can be overridden with first arg
# NUM_GPUS=${1:-2}

module load cuda

if ! command -v uv >/dev/null 2>&1; then
  echo "uv could not be found, downloading"
  wget -qO- https://astral.sh/uv/install.sh | sh
fi

#assumes this is run from scripts/. if base, just remove the ../
if [ ! -d "../Data/" ]; then
  uv run -m dataset.ds_utils \
    --splits train.clean.100 validation.clean \
    --subset clean
  echo "fetched dataset"
else
  echo "already have the dataset, skipping ..."
fi

cd scripts/

if [ ! -d "../tokenizer/" ]; then
  echo "Preparing Tokenizer"
  cd ..
  uv run -m utils.tokenize
  echo "fetched tokenizer"
else
  echo "already have the tokenizer, skipping ..."
fi

uv run -m tests.debug_single_batch

# echo "Starting Distributed Training on $NUM_GPUS GPUs"
# uv run torchrun \
#     --nproc_per_node=$NUM_GPUS \
#     --standalone \
#     -m train.train \
#     --batch_size 16 \
#     --epochs 10 \
#     --lr 1e-4 \
#     --d_model 256 \
#     --n_layers 4 \
#     --num_steps 1000
