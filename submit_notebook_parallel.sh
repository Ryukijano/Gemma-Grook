#!/bin/bash
#SBATCH --job-name=run_nb_vqe_parallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:3
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

export XDG_CACHE_HOME=/scratch/cbjp404/.cache
export HF_HOME=/scratch/cbjp404/.cache/hf
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export TMPDIR=/scratch/cbjp404/tmp
export TOKENIZERS_PARALLELISM=false

module load miniforge
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV:-qrisp-jax}

NOTEBOOK=${NOTEBOOK:-/scratch/cbjp404/Isaac-GR00T/test_notebook5.ipynb}

# Run 3 parallel tasks pinned to GPUs 0,1,2

srun --exclusive -N1 -n1 bash -lc "CUDA_VISIBLE_DEVICES=0 VQE_ANSATZ_FILTER=trotter python /scratch/cbjp404/Isaac-GR00T/scripts/run_notebook.py --in $NOTEBOOK --save-html" &
srun --exclusive -N1 -n1 bash -lc "CUDA_VISIBLE_DEVICES=1 VQE_ANSATZ_FILTER=per_hamiltonian python /scratch/cbjp404/Isaac-GR00T/scripts/run_notebook.py --in $NOTEBOOK --save-html" &
srun --exclusive -N1 -n1 bash -lc "CUDA_VISIBLE_DEVICES=2 VQE_ANSATZ_FILTER=hardware_efficient python /scratch/cbjp404/Isaac-GR00T/scripts/run_notebook.py --in $NOTEBOOK --save-html" &

wait
echo "Parallel VQE notebook runs finished."


