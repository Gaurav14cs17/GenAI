"""
Solution Consistency Loss for SoFlow.

The consistency loss ensures that the model learns a valid solution function
without requiring expensive Jacobian-vector product (JVP) calculations.

Key insight: Given three time points s < l < t, the solution function satisfies:
    f(x_t, t, s) = f(f(x_t, t, l), l, s)

This is the semi-group property of ODE solutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class SolutionConsistencyLoss(nn.Module):
    """
    Solution Consistency Loss for training SoFlow.
    
    Given a data-noise pair (x_0, x_1) defining a trajectory, and three
    time points s < l < t, we enforce:
    
        f_θ(x_t, t, s) ≈ f_θ(x_l, l, s)  [with stop gradient on target]
    
    where x_l is computed by stepping from x_t using the ground truth path:
        x_l = α(l) * x_0 + β(l) * x_1
    
    The loss is:
        L_cons = E_{s,l,t,x_0,x_1} [ ||f_θ(x_t, t, s) - sg(f_θ(x_l, l, s))||² ]
    
    where sg denotes stop-gradient.
    
    Args:
        schedule_fn: Function r(k, K) that determines intermediate point l.
                     Default: linear decay from 1 to 0.
    """
    
    def __init__(
        self,
        schedule_fn: Optional[Callable[[int, int], float]] = None,
    ):
        super().__init__()
        if schedule_fn is None:
            # Default: linear schedule r(k, K) = 1 - k/K
            self.schedule_fn = lambda k, K: max(0.0, 1.0 - k / K)
        else:
            self.schedule_fn = schedule_fn

    def compute_intermediate_time(
        self,
        t: torch.Tensor,
        s: torch.Tensor,
        step: int,
        total_steps: int,
    ) -> torch.Tensor:
        """
        Compute the intermediate time l given t and s.
        
        l is computed as:
            l = s + r(k, K) * (t - s)
        
        where r(k, K) is the schedule function that decreases during training.
        
        Args:
            t: Current time, shape (B,).
            s: Target time, shape (B,).
            step: Current training step.
            total_steps: Total training steps.
            
        Returns:
            Intermediate time l, shape (B,).
        """
        r = self.schedule_fn(step, total_steps)
        l = s + r * (t - s)
        return l

    def forward(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        step: int = 0,
        total_steps: int = 1,
    ) -> torch.Tensor:
        """
        Compute the Solution Consistency loss.
        
        Args:
            model: The SoFlow model.
            x_0: Data samples, shape (B, C, H, W).
            x_1: Noise samples, shape (B, C, H, W).
            y: Class labels, shape (B,).
            t: Current time, shape (B,).
            s: Target time, shape (B,).
            step: Current training step.
            total_steps: Total training steps.
            
        Returns:
            Scalar loss value.
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # Compute intermediate time l
        l = self.compute_intermediate_time(t, s, step, total_steps)
        
        # Compute x_t and x_l on the ground truth trajectory
        t_expanded = t.view(-1, 1, 1, 1)
        l_expanded = l.view(-1, 1, 1, 1)
        
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        x_l = (1 - l_expanded) * x_0 + l_expanded * x_1
        
        # Model predictions
        # Prediction from x_t to s
        pred_from_t = model.forward(x_t, t, s, y)
        
        # Target: prediction from x_l to s (with stop gradient)
        with torch.no_grad():
            target_from_l = model.forward(x_l, l, s, y)
        
        # MSE loss
        loss = F.mse_loss(pred_from_t, target_from_l)
        
        return loss


class AdaptiveConsistencyLoss(nn.Module):
    """
    Adaptive Solution Consistency Loss with curriculum learning.
    
    This variant uses a more sophisticated schedule that:
    1. Starts with l close to t (easy targets)
    2. Gradually moves l closer to s (harder targets)
    
    The schedule is designed to stabilize training and improve convergence.
    """
    
    def __init__(
        self,
        warmup_steps: int = 1000,
        min_ratio: float = 0.1,
        schedule_type: str = "cosine",
    ):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.min_ratio = min_ratio
        self.schedule_type = schedule_type

    def get_ratio(self, step: int, total_steps: int) -> float:
        """Get the current ratio r(k, K) based on schedule type."""
        if step < self.warmup_steps:
            # During warmup, use high ratio (l close to t)
            progress = step / self.warmup_steps
            return 1.0 - progress * (1.0 - 0.5)  # Warmup to 0.5
        
        # After warmup, decay to min_ratio
        progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
        progress = min(1.0, progress)
        
        if self.schedule_type == "linear":
            ratio = 0.5 - progress * (0.5 - self.min_ratio)
        elif self.schedule_type == "cosine":
            import math
            ratio = self.min_ratio + (0.5 - self.min_ratio) * (1 + math.cos(math.pi * progress)) / 2
        else:
            ratio = 0.5 * (1 - progress) + self.min_ratio * progress
        
        return max(self.min_ratio, ratio)

    def forward(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        y: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        s: Optional[torch.Tensor] = None,
        step: int = 0,
        total_steps: int = 1,
    ) -> torch.Tensor:
        """
        Compute the adaptive consistency loss.
        
        Args:
            model: The SoFlow model.
            x_0: Data samples, shape (B, C, H, W).
            x_1: Noise samples, shape (B, C, H, W).
            y: Class labels, shape (B,).
            t: Current time (sampled if None).
            s: Target time (sampled if None).
            step: Current training step.
            total_steps: Total training steps.
            
        Returns:
            Scalar loss value.
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # Sample t uniformly from [0.5, 1] (focus on denoising direction)
        if t is None:
            t = torch.rand(B, device=device) * 0.5 + 0.5
        
        # s is always 0 for one-step generation objective
        if s is None:
            s = torch.zeros(B, device=device)
        
        # Get current ratio
        r = self.get_ratio(step, total_steps)
        
        # Compute l
        l = s + r * (t - s)
        
        # Compute points on trajectory
        t_exp = t.view(-1, 1, 1, 1)
        l_exp = l.view(-1, 1, 1, 1)
        
        x_t = (1 - t_exp) * x_0 + t_exp * x_1
        x_l = (1 - l_exp) * x_0 + l_exp * x_1
        
        # Predictions
        pred = model.forward(x_t, t, s, y)
        
        with torch.no_grad():
            target = model.forward(x_l, l, s, y)
        
        loss = F.mse_loss(pred, target)
        
        return loss

