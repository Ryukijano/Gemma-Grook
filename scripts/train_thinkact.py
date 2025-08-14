#!/usr/bin/env python3
"""
Training script for ThinkAct model with Gemma 3, combining:
1. Supervised learning for the action model
2. Reinforcement learning for the reasoning module
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from gr00t.model.thinkact import ThinkActPolicy, ThinkActConfig
from gr00t.data.dataset import RobotDataset
from gr00t.data.transform import create_modality_transform
from gr00t.utils.misc import set_seed

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train ThinkAct model with Gemma 3")
    parser.add_argument("--config", type=str, required=True, 
                      help="Path to config file")
    parser.add_argument("--output_dir", type=str, required=True, 
                      help="Output directory for checkpoints and logs")
    parser.add_argument("--data_dir", type=str, required=True, 
                      help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=8, 
                      help="Batch size (smaller for Gemma 3's memory requirements)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                      help="Number of gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=100, 
                      help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, 
                      help="Learning rate (smaller for fine-tuning Gemma 3)")
    parser.add_argument("--warmup_steps", type=int, default=100,
                      help="Number of warmup steps for learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                      help="Weight decay for optimizer")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                      help="Maximum gradient norm for clipping")
    parser.add_argument("--seed", type=int, default=42, 
                      help="Random seed")
    parser.add_argument("--resume", type=str, default=None, 
                      help="Path to checkpoint to resume training from")
    parser.add_argument("--device", type=str, 
                      default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to train on (cuda/cpu)")
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                      choices=["no", "fp16", "bf16"],
                      help="Mixed precision training (recommended: bf16 for Ampere+ GPUs)")
    parser.add_argument("--log_interval", type=int, default=10,
                      help="Log training metrics every N steps")
    parser.add_argument("--save_interval", type=int, default=1000,
                      help="Save checkpoint every N steps")
    parser.add_argument("--eval_interval", type=int, default=500,
                      help="Evaluate on validation set every N steps")
    return parser.parse_args()

def load_config(config_path: str) -> ThinkActConfig:
    """Load model configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    return ThinkActConfig(**config_dict)

def create_dataloader(data_dir: str, batch_size: int, modality_config: dict, split: str = "train"):
    """Create a dataloader for the dataset."""
    dataset = RobotDataset(
        data_dir=data_dir,
        modality_config=modality_config,
        split=split
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    return dataloader

def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epoch: int,
    args,
    global_step: int = 0
) -> Dict[str, float]:
    """Train the model for one epoch with mixed precision and gradient accumulation."""
    model.train()
    total_loss = 0.0
    total_rl_reward = 0.0
    
    # Initialize gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision == "fp16")
    
    progress = tqdm(dataloader, desc=f"Epoch {epoch}")
    for step, batch in enumerate(progress):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(
            enabled=args.mixed_precision != "no",
            dtype=torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
        ):
            # Forward pass
            outputs = model(batch)
            loss = outputs["loss"]
            
            # Add RL reward if using RL
            rl_reward = 0.0
            if hasattr(model, "compute_reward") and model.config.use_rr:
                rl_reward = model.compute_reward(batch)
                loss = loss + rl_reward
            
            # Normalize loss for gradient accumulation
            loss = loss / args.gradient_accumulation_steps
        
        # Backward pass with gradient scaling for fp16
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (step + 1) % args.gradient_accumulation_steps == 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                args.max_grad_norm
            )
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            
            if lr_scheduler is not None:
                lr_scheduler.step()
            
            optimizer.zero_grad()
            global_step += 1
        
        # Logging
        total_loss += loss.item() * args.gradient_accumulation_steps
        total_rl_reward += rl_reward.item() if hasattr(model, "compute_reward") else 0.0
        
        # Log metrics
        if global_step % args.log_interval == 0:
            lr = optimizer.param_groups[0]['lr']
            progress.set_postfix({
                "loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}",
                "lr": f"{lr:.2e}",
                "rl_reward": f"{rl_reward:.4f}" if hasattr(model, "compute_reward") else "N/A"
            })
            
            # Log to file or wandb/tensorboard if configured
            if args.wandb:
                import wandb
                wandb.log({
                    "train/loss": loss.item() * args.gradient_accumulation_steps,
                    "train/lr": lr,
                    "train/rl_reward": rl_reward,
                    "epoch": epoch,
                    "step": global_step
                })
        
        # Save checkpoint
        if global_step % args.save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                step=global_step,
                output_dir=args.output_dir
            )
        
        # Evaluate on validation set
        if global_step % args.eval_interval == 0:
            val_metrics = validate(model, val_loader, device)
            if args.wandb:
                wandb.log({
                    f"val/{k}": v for k, v in val_metrics.items()
                }, step=global_step)
    
    metrics = {
        "train/loss": total_loss / len(dataloader),
        "train/rl_reward": total_rl_reward / len(dataloader)
    }
    
    return metrics, global_step

def validate(model, dataloader, device):
    """Validate the model on the validation set."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            loss = outputs["loss"]
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    step: int,
    output_dir: str,
    is_best: bool = False
) -> None:
    """Save model checkpoint with additional metadata."""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model.config.to_dict(),
    }
    
    if lr_scheduler is not None:
        checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}_step_{step}.pt")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save as latest checkpoint
    latest_path = os.path.join(output_dir, "checkpoint_latest.pt")
    torch.save(checkpoint, latest_path)
    
    # Save as best checkpoint if applicable
    if is_best:
        best_path = os.path.join(output_dir, "checkpoint_best.pt")
        torch.save(checkpoint, best_path)
        logger.info(f"New best checkpoint saved to {best_path}")
    
    # Save config separately for easier loading
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(model.config.to_dict(), f, indent=2)
    
    # Clean up old checkpoints (keep only the last N)
    if hasattr(args, 'max_checkpoints') and args.max_checkpoints > 0:
        checkpoints = sorted(
            [f for f in os.listdir(output_dir) if f.startswith("checkpoint_epoch_")],
            key=lambda x: (int(x.split('_')[2]), int(x.split('_')[4].split('.')[0]))
        )
        for old_checkpoint in checkpoints[:-args.max_checkpoints]:
            os.remove(os.path.join(output_dir, old_checkpoint))

def setup_wandb(args, config):
    """Initialize Weights & Biases logging if configured."""
    if not hasattr(args, 'wandb') or not args.wandb:
        return None
    
    try:
        import wandb
        wandb.init(
            project=args.wandb_project or "thinkact-gemma3",
            name=args.wandb_run_name or f"thinkact-{os.path.basename(args.output_dir)}",
            config={
                **vars(args),
                **config.to_dict()
            },
            dir=args.output_dir,
            resume="auto"
        )
        return wandb
    except ImportError:
        logger.warning("wandb not installed. Run 'pip install wandb' to enable logging.")
        return None

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and validate config
    config = load_config(args.config)
    config.compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() and args.mixed_precision == "bf16" else torch.float32
    
    # Initialize Weights & Biases if configured
    wandb_run = setup_wandb(args, config)
    
    # Log command-line arguments
    logger.info(f"Starting training with arguments: {args}")
    logger.info(f"Using device: {args.device}")
    logger.info(f"Mixed precision: {args.mixed_precision}")
    
    # Create model and move to device
    logger.info("Initializing model...")
    model = ThinkActPolicy(config, local_model_path=args.output_dir)
    model = model.to(args.device)
    
    # Log model architecture
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad_):,}")
    
    # Load checkpoint if resuming
    start_epoch = 0
    global_step = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        global_step = checkpoint.get('step', 0)
        logger.info(f"Resumed training from epoch {start_epoch}, global step {global_step}")
    
    # Create dataloaders
    logger.info("Setting up data loaders...")
    modality_config = model.get_modality_config()
    train_loader = create_dataloader(
        args.data_dir, 
        args.batch_size, 
        modality_config, 
        split="train"
    )
    val_loader = create_dataloader(
        args.data_dir, 
        args.batch_size, 
        modality_config, 
        split="val"
    )
    
    # Setup optimizer and learning rate scheduler
    logger.info("Initializing optimizer and learning rate scheduler...")
    
    # Separate parameters for different learning rates
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation_steps
    warmup_steps = min(args.warmup_steps, total_steps // 10)  # Cap warmup at 10% of total steps
    
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        
        # Train for one epoch
        train_metrics, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=args.device,
            epoch=epoch,
            args=args,
            global_step=global_step
        )
        
        # Validate
        val_metrics = validate(model, val_loader, args.device)
        
        # Log metrics
        logger.info(f"Epoch {epoch + 1} - "
                   f"Train Loss: {train_metrics['train/loss']:.4f}, "
                   f"Val Loss: {val_metrics['val/loss']:.4f}")
        
        # Save checkpoint
        is_best = val_metrics['val/loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['val/loss']
        
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            epoch=epoch,
            step=global_step,
            output_dir=args.output_dir,
            is_best=is_best
        )
    
    logger.info("Training completed!")
    
    if wandb_run:
        wandb_run.finish()

if __name__ == "__main__":
    main()
