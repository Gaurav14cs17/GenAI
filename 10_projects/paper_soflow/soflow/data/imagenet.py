"""
ImageNet dataset utilities for SoFlow training.

Supports both raw ImageNet and pre-extracted latents for
efficient training with a frozen VAE encoder.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Optional, Tuple, Callable
import json


class ImageNetDataset(Dataset):
    """
    ImageNet dataset for SoFlow training.
    
    Supports two modes:
    1. Raw images: Load and transform images on-the-fly
    2. Pre-extracted latents: Load pre-computed VAE latents
    
    Args:
        root: Path to dataset root.
        split: Dataset split ("train" or "val").
        transform: Optional transform for images.
        use_latents: Whether to use pre-extracted latents.
        latent_path: Path to pre-extracted latents (if use_latents=True).
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        use_latents: bool = False,
        latent_path: Optional[str] = None,
    ):
        self.root = root
        self.split = split
        self.use_latents = use_latents
        self.latent_path = latent_path
        
        if use_latents:
            self._setup_latents()
        else:
            self._setup_images(transform)

    def _setup_images(self, transform: Optional[Callable]) -> None:
        """Setup for raw image loading."""
        from torchvision.datasets import ImageFolder
        
        split_dir = os.path.join(self.root, self.split)
        
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        
        self.dataset = ImageFolder(split_dir, transform=transform)
        self.num_classes = 1000

    def _setup_latents(self) -> None:
        """Setup for pre-extracted latent loading."""
        if self.latent_path is None:
            self.latent_path = os.path.join(self.root, f"{self.split}_latents")
        
        # Load metadata
        meta_path = os.path.join(self.latent_path, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.metadata = json.load(f)
            self.num_samples = self.metadata["num_samples"]
            self.num_classes = self.metadata.get("num_classes", 1000)
        else:
            # Infer from files
            latent_files = [f for f in os.listdir(self.latent_path) if f.endswith(".npy")]
            self.num_samples = len(latent_files)
            self.num_classes = 1000
        
        # Check for memory-mapped file
        mmap_path = os.path.join(self.latent_path, "latents.npy")
        labels_path = os.path.join(self.latent_path, "labels.npy")
        
        if os.path.exists(mmap_path):
            self.latents = np.load(mmap_path, mmap_mode="r")
            self.labels = np.load(labels_path)
        else:
            self.latents = None
            self.labels = None

    def __len__(self) -> int:
        if self.use_latents:
            return self.num_samples
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.use_latents:
            return self._get_latent(idx)
        return self._get_image(idx)

    def _get_image(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get raw image and label."""
        image, label = self.dataset[idx]
        return image, label

    def _get_latent(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get pre-extracted latent and label."""
        if self.latents is not None:
            latent = torch.from_numpy(self.latents[idx].copy()).float()
            label = int(self.labels[idx])
        else:
            # Load individual file
            latent_file = os.path.join(self.latent_path, f"{idx:08d}.npy")
            data = np.load(latent_file, allow_pickle=True).item()
            latent = torch.from_numpy(data["latent"]).float()
            label = data["label"]
        
        return latent, label


def create_imagenet_dataloader(
    root: str,
    split: str = "train",
    batch_size: int = 256,
    num_workers: int = 8,
    use_latents: bool = False,
    latent_path: Optional[str] = None,
    **kwargs,
) -> DataLoader:
    """
    Create an ImageNet dataloader.
    
    Args:
        root: Path to ImageNet root.
        split: Dataset split.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        use_latents: Whether to use pre-extracted latents.
        latent_path: Path to latents.
        **kwargs: Additional DataLoader arguments.
        
    Returns:
        DataLoader instance.
    """
    dataset = ImageNetDataset(
        root=root,
        split=split,
        use_latents=use_latents,
        latent_path=latent_path,
    )
    
    shuffle = split == "train"
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=split == "train",
        **kwargs,
    )
    
    return dataloader


class LatentDataset(Dataset):
    """
    Simple dataset for pre-extracted latents stored as a single file.
    
    Expects:
    - latents.npy: Shape (N, C, H, W)
    - labels.npy: Shape (N,)
    """
    
    def __init__(self, latent_dir: str):
        self.latent_dir = latent_dir
        
        # Load data
        self.latents = np.load(os.path.join(latent_dir, "latents.npy"), mmap_mode="r")
        self.labels = np.load(os.path.join(latent_dir, "labels.npy"))
        
        self.num_samples = len(self.labels)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        latent = torch.from_numpy(self.latents[idx].copy()).float()
        label = int(self.labels[idx])
        return latent, label

