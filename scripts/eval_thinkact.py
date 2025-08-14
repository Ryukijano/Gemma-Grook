#!/usr/bin/env python3
"""
Evaluation script for ThinkAct model.
"""
import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from gr00t.model.thinkact import ThinkActPolicy, ThinkActConfig
from gr00t.data.dataset import RobotDataset
from gr00t.utils.misc import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ThinkAct model")
    parser.add_argument("--model_dir", type=str, required=True, 
                       help="Directory containing model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, 
                       help="Path to evaluation dataset")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=1000, 
                       help="Number of samples to evaluate")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run evaluation on")
    parser.add_argument("--output_dir", type=str, default="eval_results",
                       help="Directory to save evaluation results")
    return parser.parse_args()

def load_model(model_dir: str, device: str) -> ThinkActPolicy:
    """Load trained ThinkAct model."""
    # Load config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    config = ThinkActConfig(**config_dict)
    
    # Create model
    model = ThinkActPolicy(config, local_model_path=model_dir)
    
    # Load weights
    checkpoint_path = os.path.join(model_dir, "best_model.pt")
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataloader, device, num_samples: int):
    """Evaluate model on the given dataloader."""
    model.eval()
    
    metrics = {
        'action_mse': [],
        'success_rate': [],
        'reasoning_steps': [],
    }
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if i * dataloader.batch_size >= num_samples:
                break
                
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            
            # Compute metrics
            pred_actions = outputs["action_pred"]
            target_actions = batch["action"]
            
            # Action MSE
            mse = torch.nn.functional.mse_loss(pred_actions, target_actions, reduction='none')
            metrics['action_mse'].append(mse.mean().item())
            
            # Success rate (if goal info is available)
            if 'goal_reached' in batch:
                success = batch['goal_reached'].float().mean().item()
                metrics['success_rate'].append(success)
            
            # Track reasoning steps if available
            if hasattr(model, 'reasoning') and hasattr(model.reasoning, 'last_reasoning_steps'):
                metrics['reasoning_steps'].append(model.reasoning.last_reasoning_steps)
    
    # Aggregate metrics
    avg_metrics = {}
    for k, v in metrics.items():
        if v:  # Only compute if we have values
            avg_metrics[f'avg_{k}'] = np.mean(v)
            avg_metrics[f'std_{k}'] = np.std(v)
    
    return avg_metrics

def main():
    args = parse_args()
    set_seed(42)  # For reproducibility
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_dir}")
    model = load_model(args.model_dir, args.device)
    
    # Create dataloader
    modality_config = model.get_modality_config()
    dataset = RobotDataset(
        data_dir=args.data_dir,
        modality_config=modality_config,
        split="test"
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Run evaluation
    print(f"Evaluating on {min(len(dataset), args.num_samples)} samples...")
    metrics = evaluate_model(model, dataloader, args.device, args.num_samples)
    
    # Print and save results
    print("\n=== Evaluation Results ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    main()
