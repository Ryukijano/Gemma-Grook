#!/usr/bin/env python3
"""
Test script for evaluating the Gemma-GR00T model on the robot_sim.PickNPlace dataset.

This script loads the trained model and evaluates it on the test split of the dataset.
"""
import os
import sys
import json
import torch
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import model and dataset classes
from gr00t.policy.gr00t_policy import Gr00tPolicy
from gr00t.common.embodiment import EmbodimentTag
from gr00t.common.modality_config import ModalityConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load model configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_environment():
    """Set up PyTorch environment and device."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    logger.info(f"Using device: {device}")
    return device

def load_model(model_dir, device):
    """Load the trained model."""
    logger.info(f"Loading model from {model_dir}")
    
    # Load model configuration
    config_path = os.path.join(model_dir, 'config.json')
    config = load_config(config_path)
    
    # Load metadata
    metadata_path = os.path.join(model_dir, 'experiment_cfg', 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Get modality configuration
    modality_config = ModalityConfig(**metadata['gr1']['modality_config'])
    
    # Initialize model
    model = Gr00tPolicy(
        model_dir=model_dir,
        modality_config=modality_config,
        device=device,
        **config['model_args']
    )
    
    # Load weights
    weights_path = os.path.join(model_dir, 'pytorch_model.bin')
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded weights from {weights_path}")
    else:
        logger.warning(f"No weights found at {weights_path}, using random initialization")
    
    model.eval()
    return model

def evaluate_model(model, test_loader, device, num_batches=10):
    """Evaluate the model on the test set."""
    model.eval()
    total_loss = 0.0
    metrics = {}
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            if i >= num_batches:
                break
                
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            loss, batch_metrics = model.compute_loss(batch)
            
            # Update metrics
            total_loss += loss.item()
            for k, v in batch_metrics.items():
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(v.item() if torch.is_tensor(v) else v)
    
    # Calculate average metrics
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    avg_metrics['loss'] = total_loss / num_batches
    
    return avg_metrics

def main():
    # Configuration
    model_dir = "/scratch/cbjp404/Isaac-GR00T/exported_weights"
    dataset_path = "/scratch/cbjp404/Isaac-GR00T/demo_data/robot_sim.PickNPlace"
    batch_size = 32
    num_workers = 4
    
    # Set up environment
    device = setup_environment()
    
    # Load model
    model = load_model(model_dir, device)
    
    # TODO: Set up test dataset and dataloader
    # This is a placeholder - you'll need to implement the actual dataset loading
    # based on your dataset format and requirements
    test_loader = None  # Replace with actual test dataloader
    
    if test_loader is None:
        logger.error("Test dataloader not implemented. Please implement dataset loading.")
        return
    
    # Evaluate model
    logger.info("Starting evaluation...")
    metrics = evaluate_model(model, test_loader, device)
    
    # Print results
    logger.info("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        logger.info(f"{k}: {v:.4f}")
    
    # Save results
    results_path = os.path.join(model_dir, "test_results.json")
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Results saved to {results_path}")

if __name__ == "__main__":
    main()
