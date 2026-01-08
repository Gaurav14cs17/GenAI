"""
Visualization utilities for SoFlow.

Tools for visualizing samples, training progress, and model analysis.
"""

import torch
import numpy as np
from typing import Optional, List, Tuple
import math


def make_grid(
    images: torch.Tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Make a grid of images.
    
    Args:
        images: Tensor of shape (N, C, H, W).
        nrow: Number of images per row.
        padding: Padding between images.
        normalize: Whether to normalize to [0, 1].
        value_range: Min and max values for normalization.
        pad_value: Value for padding pixels.
        
    Returns:
        Grid tensor of shape (C, grid_H, grid_W).
    """
    if images.dim() == 3:
        images = images.unsqueeze(0)
    
    N, C, H, W = images.shape
    
    if normalize:
        if value_range is None:
            value_range = (images.min(), images.max())
        images = (images - value_range[0]) / (value_range[1] - value_range[0] + 1e-8)
        images = images.clamp(0, 1)
    
    # Calculate grid size
    ncol = nrow
    nrow = int(math.ceil(N / ncol))
    
    # Create grid
    grid_h = H * nrow + padding * (nrow + 1)
    grid_w = W * ncol + padding * (ncol + 1)
    grid = torch.full((C, grid_h, grid_w), pad_value, dtype=images.dtype, device=images.device)
    
    # Fill grid
    for idx in range(N):
        i = idx // ncol
        j = idx % ncol
        y = padding + i * (H + padding)
        x = padding + j * (W + padding)
        grid[:, y:y+H, x:x+W] = images[idx]
    
    return grid


def save_image_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = 8,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None,
):
    """
    Save a grid of images to file.
    
    Args:
        images: Tensor of shape (N, C, H, W).
        path: Output file path.
        nrow: Number of images per row.
        normalize: Whether to normalize.
        value_range: Value range for normalization.
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL not installed. Cannot save images.")
        return
    
    grid = make_grid(images, nrow=nrow, normalize=normalize, value_range=value_range)
    
    # Convert to numpy
    grid = grid.cpu().numpy()
    if grid.shape[0] == 1:
        grid = grid[0]  # Grayscale
    else:
        grid = np.transpose(grid, (1, 2, 0))  # CHW -> HWC
    
    # Scale to [0, 255]
    grid = (grid * 255).astype(np.uint8)
    
    # Save
    Image.fromarray(grid).save(path)


def plot_training_curves(
    metrics: dict,
    output_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4),
):
    """
    Plot training curves.
    
    Args:
        metrics: Dict with keys like 'loss', 'loss_fm', 'loss_cons'.
        output_path: Optional path to save figure.
        figsize: Figure size.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Cannot plot.")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Total loss
    if "loss" in metrics:
        axes[0].plot(metrics["loss"])
        axes[0].set_title("Total Loss")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
    
    # FM loss
    if "loss_fm" in metrics:
        axes[1].plot(metrics["loss_fm"])
        axes[1].set_title("Flow Matching Loss")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Loss")
    
    # Consistency loss
    if "loss_cons" in metrics:
        axes[2].plot(metrics["loss_cons"])
        axes[2].set_title("Consistency Loss")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("Loss")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    
    plt.close()


def visualize_trajectory(
    model: torch.nn.Module,
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    y: torch.Tensor,
    num_steps: int = 10,
    output_path: Optional[str] = None,
):
    """
    Visualize the generation trajectory from noise to data.
    
    Args:
        model: SoFlow model.
        x_0: Data sample (for reference).
        x_1: Noise sample.
        y: Class label.
        num_steps: Number of intermediate steps to visualize.
        output_path: Optional path to save figure.
    """
    device = x_0.device
    
    # Generate trajectory
    timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
    
    trajectory = [x_1]
    
    with torch.no_grad():
        x = x_1.unsqueeze(0)
        y_batch = y.unsqueeze(0)
        
        for i in range(num_steps):
            t = timesteps[i].expand(1)
            s = timesteps[i + 1].expand(1)
            x = model.forward(x, t, s, y_batch)
            trajectory.append(x.squeeze(0))
    
    # Add ground truth
    trajectory.append(x_0)
    
    # Stack and visualize
    trajectory = torch.stack(trajectory, dim=0)
    
    save_image_grid(
        trajectory,
        output_path or "trajectory.png",
        nrow=num_steps + 2,
        normalize=True,
    )


class MetricLogger:
    """Simple metric logger for training."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = {}
        self.history = {}
    
    def update(self, **kwargs):
        """Update metrics."""
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
                self.history[key] = []
            
            self.metrics[key].append(value)
            self.history[key].append(value)
            
            # Keep only recent values for running average
            if len(self.metrics[key]) > self.window_size:
                self.metrics[key].pop(0)
    
    def get_average(self, key: str) -> float:
        """Get running average of a metric."""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return sum(self.metrics[key]) / len(self.metrics[key])
    
    def get_history(self, key: str) -> List[float]:
        """Get full history of a metric."""
        return self.history.get(key, [])
    
    def get_all_averages(self) -> dict:
        """Get running averages of all metrics."""
        return {key: self.get_average(key) for key in self.metrics}

