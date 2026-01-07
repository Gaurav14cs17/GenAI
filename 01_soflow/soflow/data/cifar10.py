"""
CIFAR-10 dataset utilities for SoFlow training.

CIFAR-10 is a good open-source dataset for testing and development.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from typing import Optional, Tuple


class CIFAR10Dataset(Dataset):
    """
    CIFAR-10 dataset wrapper for SoFlow.
    
    Args:
        root: Path to download/store the dataset.
        train: Whether to use training set.
        transform: Optional transform to apply.
        download: Whether to download if not present.
    """
    
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        transform: Optional[transforms.Compose] = None,
        download: bool = True,
    ):
        if transform is None:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )
        self.num_classes = 10

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]


def create_cifar10_dataloader(
    root: str = "./data",
    train: bool = True,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create a CIFAR-10 dataloader.
    
    Args:
        root: Path to dataset.
        train: Whether to use training set.
        batch_size: Batch size.
        num_workers: Number of workers.
        download: Whether to download.
        
    Returns:
        DataLoader instance.
    """
    dataset = CIFAR10Dataset(
        root=root,
        train=train,
        download=download,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        **kwargs,
    )
    
    return dataloader

