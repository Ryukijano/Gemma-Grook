#!/bin/bash
#SBATCH --job-name=gr00t_train
#SBATCH --nodes=2              # 2 nodes
#SBATCH --ntasks-per-node=3    # 3 tasks (1 per GPU)
#SBATCH --cpus-per-task=8      # 8 CPU cores per task (24 total per node)
#SBATCH --gres=gpu:3           # 3 GPUs per node
#SBATCH --partition=gpu        # Specify the GPU partition
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Set up environment
export PYTHONPATH=/scratch/cbjp404/Isaac-GR00T:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32,expandable_segments:True"
export PYTORCH_NO_CUDA_MEMORY_MANAGEMENT=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

# Set up distributed training
export MASTER_PORT=12345
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export WORLD_SIZE=$((SLURM_NTASKS_PER_NODE * SLURM_JOB_NUM_NODES))

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate gr00t

# Launch training
srun --export=ALL \
    --nodes=2 \
    --ntasks-per-node=3 \
    python scripts/gr00t_finetune.py \
    --dataset-path /scratch/cbjp404/Isaac-GR00T/demo_data/robot_sim.PickNPlace \
    --output-dir /scratch/cbjp404/Isaac-GR00T/output \
    --data-config robot_sim_picknplace \
    --batch-size 1 \
    --gradient-checkpointing \
    --tune-diffusion-model \
    --tune-projector \
    --num-gpus 3 \
    --gradient-accumulation-steps 4 \
    --max-steps 10000 \
    --save-steps 1000