#!/bin/bash
#SBATCH --job-name=run_nb_vqe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=02:00:00
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
OUT_PATH=${OUT_PATH:-}

if [[ -z "${OUT_PATH}" ]]; then
  python /scratch/cbjp404/Isaac-GR00T/scripts/run_notebook.py --in "$NOTEBOOK" --save-html
else
  python /scratch/cbjp404/Isaac-GR00T/scripts/run_notebook.py --in "$NOTEBOOK" --out "$OUT_PATH" --save-html
fi


