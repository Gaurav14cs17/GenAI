"""
SoFlow Model: Solution Flow Models for One-Step Generative Modeling.

This module implements the SoFlow wrapper that handles:
- The solution function f_θ(x_t, t, s)
- Velocity extraction for Flow Matching loss
- One-step generation with CFG
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .dit import DiT, DIT_MODELS


class SoFlowModel(nn.Module):
    """
    SoFlow model wrapper for one-step generation.
    
    The model learns a solution function f_θ(x_t, t, s) that maps a state x_t
    at time t directly to the state at time s. For one-step generation,
    we use f_θ(x_1, 1, 0) to map noise directly to data.
    
    Key properties:
    - f_θ(x_t, t, t) = x_t (identity at same timestep)
    - ∂f_θ/∂s = v(f_θ(x_t, t, s), s) (satisfies velocity ODE)
    
    Args:
        backbone: The backbone model (DiT).
        alpha_fn: Function α(t) for the interpolation schedule.
        beta_fn: Function β(t) for the interpolation schedule.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        alpha_fn: callable = None,
        beta_fn: callable = None,
    ):
        super().__init__()
        self.backbone = backbone
        
        # Default to linear interpolation: x_t = (1-t) * x_0 + t * x_1
        # α(t) = 1 - t, β(t) = t
        if alpha_fn is None:
            self.alpha_fn = lambda t: 1 - t
        else:
            self.alpha_fn = alpha_fn
            
        if beta_fn is None:
            self.beta_fn = lambda t: t
        else:
            self.beta_fn = beta_fn

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the solution function f_θ(x_t, t, s).
        
        The model predicts the displacement from x_t to x_s, and we add it
        to x_t to get the final prediction.
        
        Args:
            x_t: Noisy input at time t, shape (B, C, H, W).
            t: Current timestep, shape (B,).
            s: Target timestep, shape (B,).
            y: Class labels, shape (B,).
            
        Returns:
            Predicted state at time s, shape (B, C, H, W).
        """
        # The backbone predicts the displacement
        displacement = self.backbone(x_t, t, s, y)
        
        # f_θ(x_t, t, s) = x_t + displacement
        # This ensures f_θ(x_t, t, t) ≈ x_t when displacement → 0
        return x_t + displacement

    def get_velocity(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        eps: float = 1e-4,
    ) -> torch.Tensor:
        """
        Extract velocity v_θ(x_t, t) from the solution function.
        
        The velocity is computed as the derivative of f_θ with respect to s
        evaluated at s = t:
        
        v_θ(x_t, t) = ∂f_θ(x_t, t, s)/∂s |_{s=t}
        
        We approximate this using finite differences:
        v_θ(x_t, t) ≈ (f_θ(x_t, t, t-ε) - x_t) / (-ε)
        
        Args:
            x_t: Noisy input at time t, shape (B, C, H, W).
            t: Current timestep, shape (B,).
            y: Class labels, shape (B,).
            eps: Small epsilon for finite difference.
            
        Returns:
            Estimated velocity, shape (B, C, H, W).
        """
        # Compute f_θ(x_t, t, t - ε)
        s = t - eps
        s = torch.clamp(s, min=0.0)
        
        f_out = self.forward(x_t, t, s, y)
        
        # v ≈ (f_θ(x_t, t, t-ε) - x_t) / (t-ε - t) = (f_out - x_t) / (-ε)
        velocity = (f_out - x_t) / (s - t).view(-1, 1, 1, 1)
        
        return velocity

    def sample(
        self,
        noise: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        One-step sampling from noise to data.
        
        Uses f_θ(x_1, 1, 0) to directly map noise to data.
        
        Args:
            noise: Random noise x_1 ~ N(0, I), shape (B, C, H, W).
            y: Class labels, shape (B,).
            cfg_scale: CFG scale for guided generation.
            
        Returns:
            Generated samples, shape (B, C, H, W).
        """
        B = noise.shape[0]
        device = noise.device
        
        # t = 1 (noise), s = 0 (data)
        t = torch.ones(B, device=device)
        s = torch.zeros(B, device=device)
        
        if cfg_scale != 1.0:
            # Classifier-free guidance
            return self._sample_with_cfg(noise, t, s, y, cfg_scale)
        else:
            return self.forward(noise, t, s, y)

    def _sample_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        Sample with classifier-free guidance.
        
        CFG formula: f_cfg = f_uncond + cfg_scale * (f_cond - f_uncond)
        """
        # Conditional prediction
        f_cond = self.forward(x, t, s, y)
        
        # Unconditional prediction (null class)
        y_null = torch.full_like(y, self.backbone.y_embedder.num_classes)
        f_uncond = self.forward(x, t, s, y_null)
        
        # Apply CFG
        f_cfg = f_uncond + cfg_scale * (f_cond - f_uncond)
        
        return f_cfg

    def multi_step_sample(
        self,
        noise: torch.Tensor,
        y: torch.Tensor,
        num_steps: int = 1,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Multi-step sampling (optional, for comparison).
        
        Divides the trajectory into multiple steps for potentially
        higher quality at the cost of more NFEs.
        
        Args:
            noise: Random noise, shape (B, C, H, W).
            y: Class labels, shape (B,).
            num_steps: Number of sampling steps.
            cfg_scale: CFG scale.
            
        Returns:
            Generated samples, shape (B, C, H, W).
        """
        B = noise.shape[0]
        device = noise.device
        
        x = noise
        timesteps = torch.linspace(1, 0, num_steps + 1, device=device)
        
        for i in range(num_steps):
            t = timesteps[i].expand(B)
            s = timesteps[i + 1].expand(B)
            
            if cfg_scale != 1.0:
                x = self._sample_with_cfg(x, t, s, y, cfg_scale)
            else:
                x = self.forward(x, t, s, y)
        
        return x

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, model_type: str = "DiT-XL/2", **kwargs):
        """
        Load a pretrained SoFlow model.
        
        Args:
            checkpoint_path: Path to the checkpoint file.
            model_type: Type of DiT backbone.
            **kwargs: Additional arguments for the model.
            
        Returns:
            Loaded SoFlow model.
        """
        # Create backbone
        backbone = DIT_MODELS[model_type](**kwargs)
        
        # Create SoFlow wrapper
        model = cls(backbone)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        return model


def create_soflow_model(
    model_type: str = "DiT-XL/2",
    input_size: int = 32,
    in_channels: int = 4,
    num_classes: int = 1000,
    class_dropout_prob: float = 0.1,
    **kwargs,
) -> SoFlowModel:
    """
    Factory function to create a SoFlow model.
    
    Args:
        model_type: Type of DiT backbone ("DiT-XL/2", "DiT-L/2", etc.).
        input_size: Spatial size of latent (32 for 256px with 8x VAE).
        in_channels: Number of latent channels.
        num_classes: Number of classes.
        class_dropout_prob: CFG dropout probability.
        
    Returns:
        SoFlow model instance.
    """
    # Create backbone
    backbone = DIT_MODELS[model_type](
        input_size=input_size,
        in_channels=in_channels,
        num_classes=num_classes,
        class_dropout_prob=class_dropout_prob,
        **kwargs,
    )
    
    # Wrap in SoFlow
    model = SoFlowModel(backbone)
    
    return model

