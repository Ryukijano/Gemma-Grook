#!/bin/bash
#SBATCH --job-name=test_gemma_groot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=test_%j.log
#SBATCH --error=test_%j.err

# Load required modules
module purge
module load cuda/12.1
module load python/3.10
module load miniconda3

# Initialize conda for bash
source $(conda info --base)/etc/profile.d/conda.sh

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)/..

# Create conda environment if it doesn't exist
ENV_NAME="groot-test"
if ! conda env list | grep -q "$ENV_NAME" ; then
    echo "Creating conda environment $ENV_NAME..."
    conda create -y -n $ENV_NAME python=3.10
    conda activate $ENV_NAME
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install -r ../requirements.txt
else
    conda activate $ENV_NAME
fi

# Run the test script
echo "Starting model testing..."
echo "Python path: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

cd $(dirname "$0")
python test_model.py

echo "Testing completed. Check the log files for details."
