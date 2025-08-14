#!/usr/bin/env bash

set -euo pipefail

# Simple wrapper to launch Gemma-Le training via the vendored LeRobot entrypoint.
# Adjust paths as needed for your environment.

python lerobot/lerobot/scripts/train.py \
  --policy.type gemma_le \
  --dataset.repo_id local/robot_sim.PickNPlace \
  --dataset.root ${DATASET_ROOT:-/scratch/cbjp404/Isaac-GR00T/demo_data/robot_sim.PickNPlace} \
  --dataset.episodes "[0,1,2,3,4]" \
  --batch_size ${BATCH_SIZE:-3} \
  --steps ${STEPS:-200000} \
  --log_freq ${LOG_FREQ:-100} \
  --save_freq ${SAVE_FREQ:-5000} \
  --policy.vision_model_id ${VISION_MODEL_ID:-google/siglip-so400m-patch14-384} \
  --policy.text_model_id ${TEXT_MODEL_ID:-google/gemma-3-4b-it} \
  --policy.use_amp ${USE_AMP:-true} \
  --progress_bar ${PROGRESS_BAR:-true}


