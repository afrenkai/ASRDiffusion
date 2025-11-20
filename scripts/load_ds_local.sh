#!/bin/bash

#assumes this is run from scripts/. if base, just remove the ../
if [ ! -d "../Data/" ]; then
  uv run -m dataset.ds_utils
  echo "fetched dataset"
else
  echo "already have the dataset, skipping ..."
fi

