#!/bin/bash
# Three-GPU interactive runner: splits work by ansatz type across GPUs 0/1/2

set -euo pipefail

NOTEBOOK=${NOTEBOOK:-/scratch/cbjp404/Isaac-GR00T/test_notebook5.ipynb}
CONDA_ENV=${CONDA_ENV:-qrisp-jax}
KERNEL_NAME=${KERNEL_NAME:-qrisp-jax}

module load miniforge
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

source /scratch/cbjp404/Isaac-GR00T/scripts/bootstrap_qrisp.sh
source /scratch/cbjp404/Isaac-GR00T/scripts/bootstrap_ipykernel.sh "$KERNEL_NAME"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
export MPLBACKEND=Agg

CUDA_VISIBLE_DEVICES=0 VQE_ANSATZ_FILTER=trotter \
  python /scratch/cbjp404/Isaac-GR00T/scripts/run_notebook.py --kernel "$KERNEL_NAME" --in "$NOTEBOOK" --save-html &

CUDA_VISIBLE_DEVICES=1 VQE_ANSATZ_FILTER=per_hamiltonian \
  python /scratch/cbjp404/Isaac-GR00T/scripts/run_notebook.py --kernel "$KERNEL_NAME" --in "$NOTEBOOK" --save-html &

CUDA_VISIBLE_DEVICES=2 VQE_ANSATZ_FILTER=hardware_efficient \
  python /scratch/cbjp404/Isaac-GR00T/scripts/run_notebook.py --kernel "$KERNEL_NAME" --in "$NOTEBOOK" --save-html &

wait
echo "Interactive 3-GPU runs finished."


