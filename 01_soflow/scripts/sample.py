#!/usr/bin/env python3
"""
Sampling script for SoFlow models.

Generate samples using trained SoFlow models with one-step generation.

Usage:
    python scripts/sample.py --checkpoint path/to/checkpoint.pt --output_dir ./samples
"""

import os
import sys
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from soflow.models import create_soflow_model, DIT_MODELS

# Try to import PIL for saving images
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples with SoFlow")
    
    # Model arguments
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--model", type=str, default="DiT-B/2",
                        choices=list(DIT_MODELS.keys()),
                        help="Model architecture")
    parser.add_argument("--input_size", type=int, default=32,
                        help="Latent spatial size")
    parser.add_argument("--in_channels", type=int, default=4,
                        help="Number of latent channels")
    parser.add_argument("--num_classes", type=int, default=1000,
                        help="Number of classes")
    
    # Sampling arguments
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for sampling")
    parser.add_argument("--cfg_scale", type=float, default=1.5,
                        help="Classifier-free guidance scale")
    parser.add_argument("--num_steps", type=int, default=1,
                        help="Number of sampling steps (1 for one-step)")
    
    # Class arguments
    parser.add_argument("--class_labels", type=str, default=None,
                        help="Comma-separated class labels (random if not specified)")
    parser.add_argument("--samples_per_class", type=int, default=50,
                        help="Samples per class (for FID evaluation)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./samples",
                        help="Output directory for samples")
    parser.add_argument("--save_latents", action="store_true",
                        help="Save raw latents instead of decoding")
    parser.add_argument("--vae_checkpoint", type=str, default=None,
                        help="Path to VAE checkpoint for decoding")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--use_ema", action="store_true", default=True,
                        help="Use EMA weights for sampling")
    
    return parser.parse_args()


def load_model(args):
    """Load the SoFlow model from checkpoint."""
    print(f"Loading model from: {args.checkpoint}")
    
    # Create model
    model = create_soflow_model(
        model_type=args.model,
        input_size=args.input_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        class_dropout_prob=0.0,  # No dropout during inference
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    
    # Load weights
    if args.use_ema and "ema" in checkpoint:
        print("Using EMA weights")
        model.load_state_dict(checkpoint["ema"]["ema_model"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(args.device)
    model.eval()
    
    return model


@torch.no_grad()
def generate_samples(
    model,
    num_samples: int,
    batch_size: int,
    num_classes: int,
    latent_size: int,
    in_channels: int,
    cfg_scale: float,
    num_steps: int,
    device: str,
    class_labels: list = None,
) -> tuple:
    """Generate samples in batches."""
    all_samples = []
    all_labels = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Generating samples"):
        current_batch_size = min(batch_size, num_samples - batch_idx * batch_size)
        
        # Generate noise
        noise = torch.randn(
            current_batch_size, in_channels, latent_size, latent_size,
            device=device
        )
        
        # Generate or use specified class labels
        if class_labels is not None:
            # Cycle through specified labels
            labels = torch.tensor(
                [class_labels[i % len(class_labels)] for i in range(current_batch_size)],
                device=device
            )
        else:
            # Random labels
            labels = torch.randint(0, num_classes, (current_batch_size,), device=device)
        
        # Generate samples
        if num_steps == 1:
            samples = model.sample(noise, labels, cfg_scale=cfg_scale)
        else:
            samples = model.multi_step_sample(
                noise, labels, num_steps=num_steps, cfg_scale=cfg_scale
            )
        
        all_samples.append(samples.cpu())
        all_labels.append(labels.cpu())
    
    all_samples = torch.cat(all_samples, dim=0)[:num_samples]
    all_labels = torch.cat(all_labels, dim=0)[:num_samples]
    
    return all_samples, all_labels


def save_latents(samples: torch.Tensor, labels: torch.Tensor, output_dir: str):
    """Save latent samples as numpy arrays."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as single files for FID evaluation
    np.save(os.path.join(output_dir, "samples.npy"), samples.numpy())
    np.save(os.path.join(output_dir, "labels.npy"), labels.numpy())
    
    print(f"Saved {len(samples)} latent samples to {output_dir}")


def decode_and_save_images(
    samples: torch.Tensor,
    labels: torch.Tensor,
    output_dir: str,
    vae_checkpoint: str = None,
):
    """Decode latents and save as images."""
    os.makedirs(output_dir, exist_ok=True)
    
    if vae_checkpoint is None:
        print("Warning: No VAE checkpoint provided. Saving latents as pseudo-images.")
        # Save latents as grayscale images for visualization
        for i, (sample, label) in enumerate(zip(samples, labels)):
            # Take first channel and normalize
            img = sample[0].numpy()
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img * 255).astype(np.uint8)
            
            if HAS_PIL:
                Image.fromarray(img).save(
                    os.path.join(output_dir, f"{i:06d}_class{label.item():04d}.png")
                )
        return
    
    # Load VAE and decode
    # This would require a VAE implementation
    print("VAE decoding not implemented. Please provide a VAE decoder.")


def generate_for_fid(
    model: nn.Module,
    num_samples_per_class: int,
    num_classes: int,
    batch_size: int,
    latent_size: int,
    in_channels: int,
    cfg_scale: float,
    device: str,
    output_dir: str,
):
    """Generate samples for FID evaluation (balanced per class)."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_samples = []
    all_labels = []
    
    for class_idx in tqdm(range(num_classes), desc="Generating per class"):
        class_samples = []
        
        num_batches = (num_samples_per_class + batch_size - 1) // batch_size
        
        for _ in range(num_batches):
            current_batch = min(batch_size, num_samples_per_class - len(class_samples))
            
            noise = torch.randn(
                current_batch, in_channels, latent_size, latent_size,
                device=device
            )
            labels = torch.full((current_batch,), class_idx, device=device)
            
            samples = model.sample(noise, labels, cfg_scale=cfg_scale)
            class_samples.append(samples.cpu())
        
        class_samples = torch.cat(class_samples, dim=0)[:num_samples_per_class]
        all_samples.append(class_samples)
        all_labels.extend([class_idx] * num_samples_per_class)
    
    all_samples = torch.cat(all_samples, dim=0)
    all_labels = torch.tensor(all_labels)
    
    # Save
    np.save(os.path.join(output_dir, "samples.npy"), all_samples.numpy())
    np.save(os.path.join(output_dir, "labels.npy"), all_labels.numpy())
    
    print(f"Saved {len(all_samples)} samples ({num_samples_per_class} per class) to {output_dir}")


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args)
    
    # Parse class labels if provided
    class_labels = None
    if args.class_labels:
        class_labels = [int(x.strip()) for x in args.class_labels.split(",")]
        print(f"Using class labels: {class_labels}")
    
    # Generate samples
    print(f"Generating {args.num_samples} samples with CFG scale {args.cfg_scale}")
    
    if args.samples_per_class > 0 and args.num_samples == 50000:
        # FID evaluation mode
        generate_for_fid(
            model=model,
            num_samples_per_class=args.samples_per_class,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            latent_size=args.input_size,
            in_channels=args.in_channels,
            cfg_scale=args.cfg_scale,
            device=args.device,
            output_dir=args.output_dir,
        )
    else:
        samples, labels = generate_samples(
            model=model,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            num_classes=args.num_classes,
            latent_size=args.input_size,
            in_channels=args.in_channels,
            cfg_scale=args.cfg_scale,
            num_steps=args.num_steps,
            device=args.device,
            class_labels=class_labels,
        )
        
        if args.save_latents:
            save_latents(samples, labels, args.output_dir)
        else:
            decode_and_save_images(
                samples, labels, args.output_dir, args.vae_checkpoint
            )
    
    print("Done!")


if __name__ == "__main__":
    main()

