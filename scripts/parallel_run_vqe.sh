#!/bin/bash
# Launch three parallel notebook runs on 3 GPUs (L40s) using env-driven filters
# Usage:
#   bash scripts/parallel_run_vqe.sh

set -euo pipefail

NOTEBOOK=${NOTEBOOK:-/scratch/cbjp404/Isaac-GR00T/test_notebook5.ipynb}
CONDA_ENV=${CONDA_ENV:-qrisp-jax}

module load miniforge
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu

# Split workloads across three GPUs by ansatz type

CUDA_VISIBLE_DEVICES=0 VQE_ANSATZ_FILTER=trotter \
  python /scratch/cbjp404/Isaac-GR00T/scripts/run_notebook.py --in "$NOTEBOOK" --save-html &

CUDA_VISIBLE_DEVICES=1 VQE_ANSATZ_FILTER=per_hamiltonian \
  python /scratch/cbjp404/Isaac-GR00T/scripts/run_notebook.py --in "$NOTEBOOK" --save-html &

CUDA_VISIBLE_DEVICES=2 VQE_ANSATZ_FILTER=hardware_efficient \
  python /scratch/cbjp404/Isaac-GR00T/scripts/run_notebook.py --in "$NOTEBOOK" --save-html &

wait
echo "All VQE notebook runs completed."


