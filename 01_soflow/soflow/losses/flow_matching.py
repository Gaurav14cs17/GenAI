"""
Flow Matching Loss for SoFlow.

The Flow Matching loss allows the model to provide estimated velocity fields
for Classifier-Free Guidance (CFG) during training.

Reference: "Flow Matching for Generative Modeling" (Lipman et al., 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class FlowMatchingLoss(nn.Module):
    """
    Flow Matching loss for training the velocity component of SoFlow.
    
    Given a data-noise pair (x_0, x_1), the intermediate point is:
        x_t = α(t) * x_0 + β(t) * x_1
    
    The ground truth velocity is:
        v(x_t, t) = α'(t) * x_0 + β'(t) * x_1
    
    For linear interpolation (α(t) = 1-t, β(t) = t):
        v(x_t, t) = x_1 - x_0
    
    The loss is:
        L_FM = E_{t, x_0, x_1} [ ||v_θ(x_t, t) - v(x_t, t)||² ]
    
    Args:
        sigma_min: Minimum noise level (for numerical stability).
    """
    
    def __init__(self, sigma_min: float = 0.0):
        super().__init__()
        self.sigma_min = sigma_min

    def get_interpolation(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the interpolated point and ground truth velocity.
        
        Uses linear interpolation (optimal transport path):
            x_t = (1 - t) * x_0 + t * x_1
            v_t = x_1 - x_0
        
        Args:
            x_0: Data samples, shape (B, C, H, W).
            x_1: Noise samples, shape (B, C, H, W).
            t: Timesteps, shape (B,).
            
        Returns:
            Tuple of (x_t, v_t) where:
                x_t: Interpolated point, shape (B, C, H, W).
                v_t: Ground truth velocity, shape (B, C, H, W).
        """
        # Reshape t for broadcasting
        t = t.view(-1, 1, 1, 1)
        
        # Linear interpolation: x_t = (1 - t) * x_0 + t * x_1
        # With optional minimum sigma for stability
        if self.sigma_min > 0:
            # x_t = (1 - (1 - σ_min) * t) * x_0 + t * x_1
            alpha_t = 1 - (1 - self.sigma_min) * t
            x_t = alpha_t * x_0 + t * x_1
            # v_t = -(1 - σ_min) * x_0 + x_1
            v_t = x_1 - (1 - self.sigma_min) * x_0
        else:
            # Standard linear interpolation
            x_t = (1 - t) * x_0 + t * x_1
            v_t = x_1 - x_0
        
        return x_t, v_t

    def forward(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        y: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the Flow Matching loss.
        
        Args:
            model: The SoFlow model.
            x_0: Data samples, shape (B, C, H, W).
            x_1: Noise samples, shape (B, C, H, W).
            y: Class labels, shape (B,).
            t: Optional timesteps (sampled uniformly if not provided).
            
        Returns:
            Scalar loss value.
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # Sample timesteps uniformly from [0, 1]
        if t is None:
            t = torch.rand(B, device=device)
        
        # Get interpolated point and ground truth velocity
        x_t, v_t = self.get_interpolation(x_0, x_1, t)
        
        # For SoFlow, we predict f(x_t, t, s) where s is slightly before t
        # The displacement scaled by dt gives velocity
        # Use a small fixed dt for stability
        dt = 0.01
        s = torch.clamp(t - dt, min=0.0)
        
        # Model predicts x_s from x_t
        x_s_pred = model.forward(x_t, t, s, y)
        
        # Ground truth x_s on the trajectory
        s_exp = s.view(-1, 1, 1, 1)
        x_s_gt = (1 - s_exp) * x_0 + s_exp * x_1
        
        # MSE loss on predicted vs ground truth position
        loss = F.mse_loss(x_s_pred, x_s_gt)
        
        return loss


class DirectFlowMatchingLoss(nn.Module):
    """
    Alternative Flow Matching loss that directly uses the backbone.
    
    Instead of extracting velocity from the solution function via finite
    differences, this loss uses the backbone's output when s ≈ t as the
    velocity prediction.
    
    This is more efficient but requires the model to be designed such that
    the output approximates velocity when s → t.
    """
    
    def __init__(self, sigma_min: float = 0.0, eps: float = 1e-4):
        super().__init__()
        self.sigma_min = sigma_min
        self.eps = eps

    def forward(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        y: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the Flow Matching loss.
        
        Args:
            model: The SoFlow model.
            x_0: Data samples, shape (B, C, H, W).
            x_1: Noise samples, shape (B, C, H, W).
            y: Class labels, shape (B,).
            t: Optional timesteps.
            
        Returns:
            Scalar loss value.
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # Sample timesteps
        if t is None:
            t = torch.rand(B, device=device)
        
        # Reshape for broadcasting
        t_expanded = t.view(-1, 1, 1, 1)
        
        # Interpolate
        if self.sigma_min > 0:
            alpha_t = 1 - (1 - self.sigma_min) * t_expanded
            x_t = alpha_t * x_0 + t_expanded * x_1
            v_t = x_1 - (1 - self.sigma_min) * x_0
        else:
            x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
            v_t = x_1 - x_0
        
        # For FM loss, we want the model to predict velocity
        # Use s = t - eps and scale the displacement to get velocity
        s = torch.clamp(t - self.eps, min=0.0)
        
        # Model predicts f_θ(x_t, t, s) = x_t + displacement
        # The displacement / (s - t) ≈ velocity
        f_out = model.forward(x_t, t, s, y)
        
        # v_pred = (f_out - x_t) / (s - t)
        dt = (s - t).view(-1, 1, 1, 1)
        v_pred = (f_out - x_t) / dt
        
        # MSE loss
        loss = F.mse_loss(v_pred, v_t)
        
        return loss

