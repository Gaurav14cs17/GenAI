#!/usr/bin/env python3
"""
Training script for SoFlow models.

Supports both ImageNet (full training) and CIFAR-10 (demo/testing).

Usage:
    # CIFAR-10 demo (lightweight, CPU-friendly)
    python scripts/train.py --dataset cifar10 --epochs 30
    
    # ImageNet training
    python scripts/train.py --dataset imagenet --data_path /path/to/imagenet
    
    # With accelerate for multi-GPU
    accelerate launch scripts/train.py --dataset imagenet --data_path /path/to/data
"""

import os
import sys
import math
import json
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from soflow.models import create_soflow_model, DIT_MODELS
from soflow.models.soflow import SoFlowModel
from soflow.models.dit import DiT
from soflow.losses import SoFlowLoss
from soflow.utils import EMA

# Optional imports
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    HAS_ACCELERATE = True
except ImportError:
    HAS_ACCELERATE = False

    def set_seed(seed):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train SoFlow models")
    
    # Dataset
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "imagenet"],
                        help="Dataset to use")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to dataset")
    parser.add_argument("--use_latents", action="store_true",
                        help="Use pre-extracted latents (ImageNet only)")
    
    # Model
    parser.add_argument("--model", type=str, default="auto",
                        help="Model type (auto, DiT-B/2, DiT-L/2, etc.) or size (tiny, small, medium)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping")
    parser.add_argument("--warmup_steps", type=int, default=200,
                        help="LR warmup steps")
    
    # Loss
    parser.add_argument("--lambda_fm", type=float, default=1.0,
                        help="Flow matching loss weight")
    parser.add_argument("--lambda_cons", type=float, default=1.0,
                        help="Consistency loss weight")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Save samples every N epochs")
    
    # Logging
    parser.add_argument("--wandb", action="store_true",
                        help="Use wandb logging")
    parser.add_argument("--wandb_project", type=str, default="soflow",
                        help="Wandb project name")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Data loader workers")
    parser.add_argument("--subset", type=int, default=0,
                        help="Use subset of data (0 = full dataset)")
    
    return parser.parse_args()


def create_model(args):
    """Create model based on dataset and model arg."""
    if args.dataset == "cifar10":
        # Small models for CIFAR-10
        configs = {
            "tiny": {"hidden_size": 128, "depth": 4, "num_heads": 4, "patch_size": 4, "mlp_ratio": 2.0},
            "small": {"hidden_size": 256, "depth": 6, "num_heads": 8, "patch_size": 2, "mlp_ratio": 4.0},
            "medium": {"hidden_size": 384, "depth": 8, "num_heads": 8, "patch_size": 2, "mlp_ratio": 4.0},
        }
        
        size = args.model if args.model in configs else "small"
        cfg = configs.get(size, configs["small"])
        
        backbone = DiT(
            input_size=32,
            patch_size=cfg["patch_size"],
            in_channels=3,
            hidden_size=cfg["hidden_size"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            mlp_ratio=cfg["mlp_ratio"],
            num_classes=10,
            class_dropout_prob=0.1,
        )
        return SoFlowModel(backbone)
    else:
        # ImageNet models
        model_type = args.model if args.model in DIT_MODELS else "DiT-B/2"
        return create_soflow_model(
            model_type=model_type,
            input_size=32,
            in_channels=4,
            num_classes=1000,
            class_dropout_prob=0.1,
        )


def create_dataloader(args):
    """Create dataloader based on dataset."""
    if args.dataset == "cifar10":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        
        dataset = datasets.CIFAR10(
            root=args.data_path, train=True,
            transform=transform, download=True,
        )
        
        if args.subset > 0:
            indices = torch.randperm(len(dataset))[:args.subset]
            dataset = Subset(dataset, indices)
        
        return DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=args.num_workers,
            drop_last=True,
        )
    else:
        # ImageNet
        from soflow.data import create_imagenet_dataloader
        return create_imagenet_dataloader(
            root=args.data_path,
            split="train",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_latents=args.use_latents,
        )


def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    """Cosine LR scheduler with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def generate_samples(model, num_samples, num_classes, in_channels, img_size, cfg_scale, device):
    """Generate samples for visualization."""
    model.eval()
    # FIX: Scale noise to match data distribution (consistent with training)
    noise = torch.randn(num_samples, in_channels, img_size, img_size, device=device) * 0.5
    labels = torch.arange(num_classes, device=device).repeat(num_samples // num_classes + 1)[:num_samples]
    samples = model.sample(noise, labels, cfg_scale=cfg_scale)
    
    # #region agent log - H1,H4: Sample output ranges (post-fix)
    import json, time
    log_entry = {"location": "train.py:generate_samples", "message": "sample_output_ranges_postfix", "hypothesisId": "H1,H4", "timestamp": int(time.time()*1000), "sessionId": "debug-session", "runId": "post-fix", "data": {"cfg_scale": cfg_scale, "noise_min": float(noise.min()), "noise_max": float(noise.max()), "sample_min": float(samples.min()), "sample_max": float(samples.max()), "sample_mean": float(samples.mean()), "sample_std": float(samples.std())}}
    with open("/home/ggoswami/Project/SoFlow/.cursor/debug.log", "a") as f: f.write(json.dumps(log_entry) + "\n")
    # #endregion
    
    model.train()
    return samples


def save_samples(samples, path):
    """Save samples as image grid."""
    try:
        from torchvision.utils import save_image
        samples = (samples + 1) / 2
        samples = samples.clamp(0, 1)
        save_image(samples, path, nrow=4)
    except Exception as e:
        logger.warning(f"Could not save samples: {e}")


def train_step(model, loss_fn, batch, optimizer, scheduler, step, total_steps, accelerator=None):
    """Single training step."""
    x_0, y = batch
    # FIX: Scale noise to match data distribution (data is normalized to [-1,1] with std~0.5)
    # Standard Gaussian has std=1, so scale it down to match data
    x_1 = torch.randn_like(x_0) * 0.5
    
    # #region agent log - H1: Data-Noise scale mismatch (post-fix)
    if step % 500 == 0:
        import json, time
        log_entry = {"location": "train.py:train_step", "message": "data_noise_ranges_postfix", "hypothesisId": "H1", "timestamp": int(time.time()*1000), "sessionId": "debug-session", "runId": "post-fix", "data": {"step": step, "x0_min": float(x_0.min()), "x0_max": float(x_0.max()), "x1_min": float(x_1.min()), "x1_max": float(x_1.max()), "x0_std": float(x_0.std()), "x1_std": float(x_1.std())}}
        with open("/home/ggoswami/Project/SoFlow/.cursor/debug.log", "a") as f: f.write(json.dumps(log_entry) + "\n")
    # #endregion
    
    loss_dict = loss_fn(model, x_0, x_1, y, step=step, total_steps=total_steps, return_dict=True)
    loss = loss_dict["loss"]
    
    if accelerator:
        accelerator.backward(loss)
    else:
        loss.backward()
    
    return {k: v.item() for k, v in loss_dict.items()}


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup output
    output_dir = f"{args.output_dir}/soflow_{args.dataset}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/samples", exist_ok=True)
    
    # Setup device/accelerator
    accelerator = None
    if HAS_ACCELERATE and args.dataset == "imagenet":
        accelerator = Accelerator(mixed_precision="bf16")
        device = accelerator.device
        is_main = accelerator.is_main_process
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_main = True
    
    # Wandb
    if is_main and args.wandb:
        if HAS_WANDB:
            wandb.init(project=args.wandb_project, config=vars(args))
        else:
            logger.warning("wandb not installed")
            args.wandb = False
    
    # Model
    logger.info(f"Creating model for {args.dataset}...")
    model = create_model(args)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model parameters: {num_params:.2f}M")
    model = model.to(device)
    
    # EMA (for ImageNet)
    ema = EMA(model, decay=0.9999) if args.dataset == "imagenet" else None
    
    # Loss
    loss_fn = SoFlowLoss(
        lambda_fm=args.lambda_fm,
        lambda_cons=args.lambda_cons,
        warmup_steps=args.warmup_steps,
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Data
    logger.info("Loading data...")
    train_loader = create_dataloader(args)
    logger.info(f"Dataset: {len(train_loader.dataset)} samples, {len(train_loader)} batches/epoch")
    
    # Scheduler
    total_steps = args.epochs * len(train_loader)
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, total_steps)
    
    # Accelerator prepare
    if accelerator:
        model, optimizer, train_loader, scheduler = accelerator.prepare(
            model, optimizer, train_loader, scheduler
        )
    
    # Get dataset-specific params
    if args.dataset == "cifar10":
        num_classes, in_channels, img_size = 10, 3, 32
    else:
        num_classes, in_channels, img_size = 1000, 4, 32
    
    # Training
    logger.info(f"Training for {args.epochs} epochs ({total_steps} steps)")
    loss_history = {"total": [], "fm": [], "cons": []}
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not is_main)
        for batch in pbar:
            if not accelerator:
                batch = (batch[0].to(device), batch[1].to(device))
            
            optimizer.zero_grad()
            metrics = train_step(model, loss_fn, batch, optimizer, scheduler, global_step, total_steps, accelerator)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            
            if ema:
                ema.update()
            
            global_step += 1
            epoch_loss += metrics["loss"]
            
            # Track losses
            loss_history["total"].append(metrics["loss"])
            loss_history["fm"].append(metrics.get("loss_fm", 0))
            loss_history["cons"].append(metrics.get("loss_cons", 0))
            
            pbar.set_postfix(loss=f"{metrics['loss']:.4f}")
            
            if args.wandb and global_step % 100 == 0:
                wandb.log({"train/loss": metrics["loss"], "train/step": global_step})
        
        # Epoch summary
        if is_main:
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}")
            
            # Save samples
            if (epoch + 1) % args.save_every == 0:
                sample_model = ema.get_model() if ema else model
                samples = generate_samples(sample_model, 16, num_classes, in_channels, img_size, 2.0, device)
                save_samples(samples, f"{output_dir}/samples/epoch_{epoch+1:02d}.png")
    
    # Save final
    if is_main:
        torch.save({"model": model.state_dict(), "args": vars(args)}, f"{output_dir}/model.pt")
        with open(f"{output_dir}/loss_history.json", "w") as f:
            json.dump(loss_history, f)
        logger.info(f"Done! Saved to {output_dir}")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
