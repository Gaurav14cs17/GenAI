#!/usr/bin/env python3
"""
Extract VAE latents from ImageNet images.

Pre-extracting latents speeds up training significantly by avoiding
repeated VAE encoding during training.

Usage:
    python scripts/extract_latents.py \
        --data_dir /path/to/imagenet/train \
        --output_dir /path/to/latents \
        --vae_checkpoint /path/to/vae.pt
"""

import os
import sys
import argparse
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="Extract VAE latents")
    
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to ImageNet images")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for latents")
    parser.add_argument("--vae_checkpoint", type=str, default=None,
                        help="Path to VAE checkpoint (uses SD VAE if not provided)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for encoding")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of data loading workers")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use")
    parser.add_argument("--save_individual", action="store_true",
                        help="Save individual latent files instead of single array")
    
    return parser.parse_args()


def load_vae(checkpoint_path: str, device: str):
    """
    Load VAE encoder.
    
    If no checkpoint is provided, attempts to load the Stable Diffusion VAE
    from diffusers library.
    """
    if checkpoint_path is None:
        try:
            from diffusers import AutoencoderKL
            print("Loading Stable Diffusion VAE from diffusers...")
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sd-vae-ft-ema",
                torch_dtype=torch.float32,
            )
            vae = vae.to(device)
            vae.eval()
            
            # Wrapper to match expected interface
            class VAEWrapper:
                def __init__(self, vae):
                    self.vae = vae
                    self.scaling_factor = 0.18215
                
                def encode(self, x):
                    with torch.no_grad():
                        latent = self.vae.encode(x).latent_dist.sample()
                        latent = latent * self.scaling_factor
                    return latent
            
            return VAEWrapper(vae)
            
        except ImportError:
            print("diffusers not installed. Please provide a VAE checkpoint.")
            print("Install with: pip install diffusers")
            sys.exit(1)
    else:
        # Load custom VAE checkpoint
        print(f"Loading VAE from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # This assumes a specific VAE architecture
        # Modify as needed for your VAE
        raise NotImplementedError(
            "Custom VAE loading not implemented. "
            "Please modify this function for your VAE architecture."
        )


@torch.no_grad()
def extract_latents(
    vae,
    dataloader: DataLoader,
    output_dir: str,
    device: str,
    save_individual: bool = False,
):
    """Extract and save latents."""
    os.makedirs(output_dir, exist_ok=True)
    
    all_latents = []
    all_labels = []
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Extracting latents")):
        images = images.to(device)
        
        # Encode images
        latents = vae.encode(images)
        
        if save_individual:
            # Save individual files
            for i, (latent, label) in enumerate(zip(latents, labels)):
                idx = batch_idx * dataloader.batch_size + i
                data = {
                    "latent": latent.cpu().numpy(),
                    "label": label.item(),
                }
                np.save(os.path.join(output_dir, f"{idx:08d}.npy"), data)
        else:
            all_latents.append(latents.cpu())
            all_labels.extend(labels.tolist())
    
    if not save_individual:
        # Save as single arrays
        all_latents = torch.cat(all_latents, dim=0).numpy()
        all_labels = np.array(all_labels)
        
        np.save(os.path.join(output_dir, "latents.npy"), all_latents)
        np.save(os.path.join(output_dir, "labels.npy"), all_labels)
        
        # Save metadata
        import json
        metadata = {
            "num_samples": len(all_labels),
            "latent_shape": list(all_latents.shape[1:]),
            "num_classes": int(all_labels.max()) + 1,
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Saved latents to: {output_dir}")


def main():
    args = parse_args()
    
    # Setup transforms
    transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    
    # Load dataset
    from torchvision.datasets import ImageFolder
    dataset = ImageFolder(args.data_dir, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    print(f"Loaded {len(dataset)} images from {args.data_dir}")
    
    # Load VAE
    vae = load_vae(args.vae_checkpoint, args.device)
    
    # Extract latents
    extract_latents(
        vae=vae,
        dataloader=dataloader,
        output_dir=args.output_dir,
        device=args.device,
        save_individual=args.save_individual,
    )
    
    print("Done!")


if __name__ == "__main__":
    main()

