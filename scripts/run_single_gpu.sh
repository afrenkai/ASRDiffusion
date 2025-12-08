#!/bin/bash
set -e


echo "Preparing Dataset (LibriSpeech train.clean.100 & validation.clean)"
#just like with the slurm version, assumes this is run from scripts/. if base, just remove the ../
if [ ! -d "../Data/" ]; then
  uv run -m dataset.ds_utils \
    --splits train.100 validation test\
    --subset clean
  echo "fetched dataset"
else
  echo "already have the dataset, skipping ..."
fi

if [ ! -d "../tokenizer/" ]; then
  echo "Preparing Tokenizer"
  cd ..
  uv run -m utils.tokenize
  echo "fetched tokenizer"
else
  echo "already have the tokenizer, skipping ..."
fi


echo "Starting Single GPU Training (~ RTX 3090 w 24gb vram)"
cd ..
uv run -m train.train \
    --batch_size 16 \
    --epochs 10 \
    --lr 1e-4 \
    --d_model 256 \
    --n_layers 4 \
    --num_steps 1000
