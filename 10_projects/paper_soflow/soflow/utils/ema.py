"""
Exponential Moving Average (EMA) utilities for SoFlow.

EMA is crucial for stable training and better sample quality
in diffusion and flow-based models.
"""

import torch
import torch.nn as nn
from typing import Optional, Iterable
from copy import deepcopy


def update_ema(
    ema_model: nn.Module,
    model: nn.Module,
    decay: float = 0.9999,
) -> None:
    """
    Update EMA model parameters.
    
    EMA update rule:
        θ_ema = decay * θ_ema + (1 - decay) * θ
    
    Args:
        ema_model: The EMA model to update.
        model: The source model.
        decay: EMA decay rate (higher = slower update).
    """
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)


class EMA:
    """
    Exponential Moving Average wrapper for models.
    
    This class manages an EMA copy of a model and provides
    utilities for updating and using it.
    
    Args:
        model: The model to track.
        decay: EMA decay rate.
        warmup_steps: Number of steps before starting EMA.
        update_every: Update EMA every N steps.
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_steps: int = 0,
        update_every: int = 1,
    ):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.update_every = update_every
        
        # Create EMA model
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)
        
        # Step counter
        self.step = 0

    def update(self) -> None:
        """Update EMA model if conditions are met."""
        self.step += 1
        
        if self.step < self.warmup_steps:
            # During warmup, just copy parameters
            self._copy_params()
            return
        
        if self.step % self.update_every != 0:
            return
        
        # Compute adaptive decay based on step
        decay = self._get_decay()
        
        # Update EMA parameters
        update_ema(self.ema_model, self.model, decay)

    def _get_decay(self) -> float:
        """Get current decay rate (can be adaptive)."""
        # Optionally implement warmup decay
        if self.step < self.warmup_steps:
            return 0.0
        
        # Could implement adaptive decay here
        return self.decay

    def _copy_params(self) -> None:
        """Copy parameters from model to EMA model."""
        with torch.no_grad():
            for ema_param, param in zip(
                self.ema_model.parameters(), self.model.parameters()
            ):
                ema_param.data.copy_(param.data)

    def get_model(self) -> nn.Module:
        """Get the EMA model for inference."""
        return self.ema_model

    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        return {
            "ema_model": self.ema_model.state_dict(),
            "decay": self.decay,
            "step": self.step,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load state dict from checkpoint."""
        self.ema_model.load_state_dict(state_dict["ema_model"])
        self.decay = state_dict.get("decay", self.decay)
        self.step = state_dict.get("step", 0)


class EMAWithBuffers(EMA):
    """
    EMA that also tracks buffers (e.g., batch norm statistics).
    """
    
    def update(self) -> None:
        """Update EMA model including buffers."""
        self.step += 1
        
        if self.step < self.warmup_steps:
            self._copy_all()
            return
        
        if self.step % self.update_every != 0:
            return
        
        decay = self._get_decay()
        
        # Update parameters
        update_ema(self.ema_model, self.model, decay)
        
        # Update buffers
        with torch.no_grad():
            for ema_buf, buf in zip(
                self.ema_model.buffers(), self.model.buffers()
            ):
                ema_buf.data.mul_(decay).add_(buf.data, alpha=1 - decay)

    def _copy_all(self) -> None:
        """Copy all parameters and buffers."""
        self._copy_params()
        with torch.no_grad():
            for ema_buf, buf in zip(
                self.ema_model.buffers(), self.model.buffers()
            ):
                ema_buf.data.copy_(buf.data)

