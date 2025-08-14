#!/bin/bash
# One-GPU interactive runner for the VQE notebook

set -euo pipefail

NOTEBOOK=${NOTEBOOK:-/scratch/cbjp404/Isaac-GR00T/test_notebook5.ipynb}
CONDA_ENV=${CONDA_ENV:-qrisp-jax}
KERNEL_NAME=${KERNEL_NAME:-qrisp-jax}
GPU_ID=${GPU_ID:-0}

module load miniforge
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

source /scratch/cbjp404/Isaac-GR00T/scripts/bootstrap_qrisp.sh
source /scratch/cbjp404/Isaac-GR00T/scripts/bootstrap_ipykernel.sh "$KERNEL_NAME"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export MPLBACKEND=Agg

python /scratch/cbjp404/Isaac-GR00T/scripts/run_notebook.py --kernel "$KERNEL_NAME" --in "$NOTEBOOK" --save-html


