#!/usr/bin/env python3
"""
# Create default config
python -m cs336_basics.train --create-config config.json

# Train with config file
python -m cs336_basics.train --config config.json

# Train with command-line overrides
python -m cs336_basics.train --train-data train.npy --val-data val.npy --batch-size 8
--learning-rate 1e-4 --wandb

# Resume from checkpoint
python -m cs336_basics.train --config config.json --resume checkpoints/checkpoint_5000.pt
"""
import argparse
import json
import logging
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import wandb
from tqdm import tqdm

from .data import data_loading, save_checkpoint, load_checkpoint
from .optimizer import cross_entropy, get_lr_cosine_schedule, gradient_clipping, AdamW
from .transformer import TransformerLM


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )


def get_device() -> str:
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon M chips
    else:
        return "cpu"


def load_memmap_dataset(data_path: str) -> np.ndarray:
    """Load dataset using memory mapping for efficient memory usage."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")
    
    # Try to load as memmap first, fall back to regular numpy if needed
    try:
        dataset = np.memmap(data_path, dtype=np.int32, mode='r')
        logging.info(f"Loaded dataset with {len(dataset)} tokens using memory mapping")
    except Exception as e:
        logging.warning(f"Failed to load as memmap, falling back to regular numpy: {e}")
        dataset = np.load(data_path)
        logging.info(f"Loaded dataset with {len(dataset)} tokens")
    
    return dataset


def create_model(config: dict) -> TransformerLM:
    """Create the transformer model based on configuration."""
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"], 
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"]
    )
    
    # Initialize weights
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    
    return model


def evaluate_model(model: torch.nn.Module, val_dataset: np.ndarray, config: dict) -> float:
    """Evaluate the model on validation dataset."""
    model.eval()
    total_loss = 0.0
    num_batches = config["val_batches"]
    
    with torch.no_grad():
        for _ in range(num_batches):
            x_val, y_val = data_loading(
                val_dataset, 
                config["batch_size"], 
                config["context_length"], 
                config["device"]
            )
            logits = model(x_val)
            loss = cross_entropy(
                logits.view(-1, logits.size(-1)), 
                y_val.view(-1)
            )
            total_loss += loss.item()
    
    model.train()
    return total_loss / num_batches


def train_model(config: dict):
    """Main training loop."""
    # Setup
    device = config["device"]
    logging.info(f"Using device: {device}")
    
    # Load datasets
    logging.info("Loading datasets...")
    train_dataset = load_memmap_dataset(config["train_data_path"])
    val_dataset = load_memmap_dataset(config["val_data_path"]) if config.get("val_data_path") else None
    
    # Create model
    logging.info("Creating model...")
    model = create_model(config).to(device)
    logging.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup optimizer - AdamW parameters
    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],          # AdamW: learning rate
        weight_decay=config["weight_decay"],  # AdamW: weight decay for regularization
        betas=(config["beta1"], config["beta2"]),  # AdamW: momentum coefficients
        eps=config["eps"]                     # AdamW: numerical stability epsilon
    )
    
    # Load checkpoint if resuming
    start_iteration = 0
    if config.get("resume_from_checkpoint"):
        logging.info(f"Loading checkpoint from {config['resume_from_checkpoint']}")
        start_iteration = load_checkpoint(config["resume_from_checkpoint"], model, optimizer)
        logging.info(f"Resumed from iteration {start_iteration}")
    
    # Initialize wandb if configured
    if config.get("use_wandb"):
        wandb.init(
            project=config.get("wandb_project", "cs336-assignment-1"),
            config=config,
            name=datetime.now().strftime("%Y-%m-%d %H:%M"),
            resume="allow" if config.get("resume_from_checkpoint") else None
        )
        wandb.watch(model)

    # Training loop
    model.train()
    logging.info("Starting training...")
    
    for iteration in tqdm(range(start_iteration, config["num_iterations"]), desc="Training"):
        # Learning rate schedule - cosine annealing with warmup
        lr = get_lr_cosine_schedule(
            iteration,
            config["learning_rate"],      # Maximum learning rate
            config["min_learning_rate"],  # Minimum learning rate
            config["warmup_iterations"],  # Warmup period
            config["num_iterations"]      # Total iterations
        )
        
        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        
        # Forward pass
        x, y = data_loading(
            train_dataset,
            config["batch_size"],
            config["context_length"], 
            device
        )
        
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping - prevent exploding gradients
        if config["gradient_clip_norm"] > 0:
            gradient_clipping(model.parameters(), config["gradient_clip_norm"])
        
        optimizer.step()
        
        # Logging
        if iteration % config["log_interval"] == 0:
            logging.info(f"Iteration {iteration}: loss={loss.item():.4f}, lr={lr:.6f}")
            
            metrics = {
                "train/loss": loss.item(),
                "train/learning_rate": lr,
                "iteration": iteration
            }
            
            # Validation
            if val_dataset is not None and iteration % config["val_interval"] == 0:
                val_loss = evaluate_model(model, val_dataset, config)
                logging.info(f"Validation loss: {val_loss:.4f}")
                metrics["val/loss"] = val_loss
            
            # Log to wandb
            if config.get("use_wandb"):
                wandb.log(metrics)
        
        # Save checkpoint
        if iteration % config["save_interval"] == 0 and iteration > 0:
            checkpoint_path = Path(config["checkpoint_dir"]) / f"checkpoint_{iteration}.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, iteration, checkpoint_path)
            logging.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final checkpoint
    final_checkpoint = Path(config["checkpoint_dir"]) / "final_checkpoint.pt"
    final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(model, optimizer, config["num_iterations"], final_checkpoint)
    logging.info(f"Saved final checkpoint to {final_checkpoint}")


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set device if not specified
    if "device" not in config:
        config["device"] = get_device()
    
    return config


def create_default_config() -> dict:
    """Create default configuration."""
    return {
        # Model hyperparameters - Transformer architecture
        "vocab_size": 10000,      # Vocabulary size
        "context_length": 256,   # Maximum sequence length
        "d_model": 512,          # Model dimensionality
        "num_layers": 12,        # Number of transformer blocks
        "num_heads": 12,         # Number of attention heads
        "d_ff": 3072,            # Feed-forward network hidden size
        "rope_theta": 10000.0,   # RoPE (Rotary Position Embedding) theta parameter
        
        # Optimizer hyperparameters - AdamW
        "learning_rate": 3e-4,      # AdamW: initial learning rate
        "min_learning_rate": 3e-5,  # Learning rate schedule: minimum LR
        "weight_decay": 0.1,        # AdamW: weight decay coefficient
        "beta1": 0.9,              # AdamW: first momentum coefficient
        "beta2": 0.95,             # AdamW: second momentum coefficient
        "eps": 1e-8,               # AdamW: numerical stability epsilon
        
        # Training hyperparameters
        "batch_size": 4,                # Training batch size
        "num_iterations": 10000,        # Total training iterations
        "warmup_iterations": 2000,      # Learning rate warmup period
        "gradient_clip_norm": 1.0,      # Gradient clipping: max L2 norm

        # Data paths
        "train_data_path": "train_data.npy",
        "val_data_path": "val_data.npy",
        
        # Logging and checkpointing
        "log_interval": 100,         # Log every N iterations
        "val_interval": 500,         # Validate every N iterations
        "save_interval": 1000,       # Save checkpoint every N iterations
        "val_batches": 10,           # Number of batches for validation
        "checkpoint_dir": "checkpoints",
        
        # Wandb configuration
        "use_wandb": False,
        "wandb_project": "transformer-training",
        
        # Other
        "device": get_device(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer language model")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--train-data", type=str, help="Path to training data file")
    parser.add_argument("--val-data", type=str, help="Path to validation data file")
    parser.add_argument("--checkpoint-dir", type=str, help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--learning-rate", type=float, help="Learning rate")
    parser.add_argument("--num-iterations", type=int, help="Number of training iterations")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--create-config", type=str, help="Create default config file and exit")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Create default config if requested
    if args.create_config:
        config = create_default_config()
        with open(args.create_config, 'w') as f:
            json.dump(config, f, indent=2)
        logging.info(f"Created default configuration at {args.create_config}")
        return
    
    # Load or create configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = create_default_config()
        logging.info("Using default configuration")
    
    # Override config with command line arguments
    if args.train_data:
        config["train_data_path"] = args.train_data
    if args.val_data:
        config["val_data_path"] = args.val_data
    if args.checkpoint_dir:
        config["checkpoint_dir"] = args.checkpoint_dir
    if args.resume:
        config["resume_from_checkpoint"] = args.resume
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.learning_rate:
        config["learning_rate"] = args.learning_rate
    if args.num_iterations:
        config["num_iterations"] = args.num_iterations
    if args.wandb:
        config["use_wandb"] = True
    
    # Validate required paths
    if not os.path.exists(config["train_data_path"]):
        raise FileNotFoundError(f"Training data not found: {config['train_data_path']}")
    
    # Train model
    train_model(config)


if __name__ == "__main__":
    main()