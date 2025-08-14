#!/usr/bin/env python3
"""
Script to export trained model weights for deployment.
"""
import os
import shutil
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def export_weights(checkpoint_dir: str, output_dir: str):
    """
    Export model weights from a checkpoint to a specified directory.
    
    Args:
        checkpoint_dir: Path to the checkpoint directory
        output_dir: Directory to save the exported weights
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy relevant files
    print(f"Exporting weights from {checkpoint_dir} to {output_dir}")
    
    # Copy safetensors files and index
    for ext in ['.safetensors', '.safetensors.index.json']:
        for f in Path(checkpoint_dir).glob(f'*{ext}'):
            shutil.copy2(f, os.path.join(output_dir, f.name))
    
    # Copy config files
    config_files = ['config.json', 'training_args.bin', 'trainer_state.json']
    for f in config_files:
        src = os.path.join(os.path.dirname(checkpoint_dir), f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, f))
    
    print("Weights exported successfully!")
    print(f"Exported files to: {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export model weights for deployment')
    parser.add_argument('--checkpoint', type=str, default='output/checkpoint-30000',
                      help='Path to the checkpoint directory')
    parser.add_argument('--output_dir', type=str, default='exported_weights',
                      help='Directory to save the exported weights')
    
    args = parser.parse_args()
    export_weights(args.checkpoint, args.output_dir)
