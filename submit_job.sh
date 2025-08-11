#!/bin/bash
#SBATCH --job-name=gemma_le_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:3
#SBATCH --partition=gpu
#SBATCH --time=40:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Set up environment (caches on scratch)
export PYTHONPATH=/scratch/cbjp404/Isaac-GR00T/lerobot/lerobot:$PYTHONPATH
export XDG_CACHE_HOME=/scratch/cbjp404/.cache
export HF_HOME=/scratch/cbjp404/.cache/hf
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export WANDB_DIR=/scratch/cbjp404/.cache/wandb
export PIP_CACHE_DIR=/scratch/cbjp404/.cache/pip
export TMPDIR=/scratch/cbjp404/tmp
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32,expandable_segments:True"
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_IB_DISABLE=1

# (Optional) distributed env vars if needed later
export MASTER_PORT=${MASTER_PORT:-12345}
export MASTER_ADDR=${MASTER_ADDR:-$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)}
export WORLD_SIZE=${WORLD_SIZE:-$SLURM_NTASKS}

# Activate conda environment (HPC module + conda)
module load miniforge
conda activate gr00t

# Login to Hugging Face (expects $HF_TOKEN to be set in your env; otherwise it will no-op)
if [[ -n "$HF_TOKEN" ]]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential False || true
fi

# Launch training (single process using DataParallel across 3 GPUs)
srun --export=ALL \
    python lerobot/lerobot/scripts/train.py \
      --policy.type gemma_le \
      --config_path outputs/train/2025-08-11/01-15-25_gemma_le/checkpoints/last/pretrained_model/train_config.json \
      --dataset.repo_id local/robot_sim.PickNPlace \
      --dataset.root /scratch/cbjp404/Isaac-GR00T/demo_data/robot_sim.PickNPlace \
      --dataset.episodes "[0,1,2,3,4]" \
      --dataset.use_imagenet_stats false \
      --batch_size 3 \
      --steps 60000 \
      --log_freq 100 \
      --save_freq 20000 \
      --num_workers 8 \
      --progress_bar true \
      --policy.use_amp true \
      --policy.vision_model_id /scratch/cbjp404/.cache/hf/models/siglip-so400m-patch14-384 \
      --policy.text_model_id /scratch/cbjp404/.cache/hf/models/gemma-3-4b-it \
      --push_to_hub false \
      --resume true \
      --push_repo_id Ryukijano/gemma-groot \
      --push_branch main \
      --push_exist_ok true