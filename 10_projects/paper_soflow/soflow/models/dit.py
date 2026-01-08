"""
Diffusion Transformer (DiT) implementation for SoFlow.

Based on "Scalable Diffusion Models with Transformers" (Peebles & Xie, 2023).
Modified to support the solution function f_θ(x_t, t, s) with two time inputs.

Reference: https://arxiv.org/abs/2212.09748
"""

import math
import torch
import torch.nn as nn
from einops import rearrange

from .layers import (
    PatchEmbed,
    TimestepEmbedder,
    LabelEmbedder,
    DiTBlock,
    FinalLayer,
)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """
    Generate 2D sinusoidal positional embeddings.
    
    Args:
        embed_dim: Embedding dimension.
        grid_size: Size of the grid (height = width).
        
    Returns:
        Positional embeddings of shape (grid_size^2, embed_dim).
    """
    grid_h = torch.arange(grid_size, dtype=torch.float32)
    grid_w = torch.arange(grid_size, dtype=torch.float32)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
    grid = torch.stack(grid, dim=0).reshape(2, 1, grid_size, grid_size)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: torch.Tensor) -> torch.Tensor:
    """Generate positional embeddings from a grid."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = torch.cat([emb_h, emb_w], dim=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: torch.Tensor) -> torch.Tensor:
    """Generate 1D sinusoidal positional embeddings."""
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb


class DiT(nn.Module):
    """
    Diffusion Transformer for SoFlow.
    
    This model implements the solution function f_θ(x_t, t, s) that maps
    a noisy state x_t at time t to the predicted state at time s.
    
    For standard flow matching (velocity prediction), set s=t and the model
    outputs the velocity v_θ(x_t, t).
    
    Args:
        input_size: Spatial size of the input (assumed square).
        patch_size: Size of each patch.
        in_channels: Number of input channels (e.g., 4 for latent diffusion).
        hidden_size: Transformer hidden dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension ratio.
        num_classes: Number of classes for conditioning.
        class_dropout_prob: Dropout probability for classifier-free guidance.
        learn_sigma: Whether to predict variance (unused in flow matching).
    """
    
    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_classes: int = 1000,
        class_dropout_prob: float = 0.1,
        learn_sigma: bool = False,
    ):
        super().__init__()
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # Patch embedding
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        
        # Time embeddings for both t and s
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.s_embedder = TimestepEmbedder(hidden_size)
        
        # Class embedding
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        
        # Number of patches
        num_patches = self.x_embedder.num_patches
        
        # Positional embedding (fixed, not learned)
        self.register_buffer(
            "pos_embed",
            torch.zeros(1, num_patches, hidden_size),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        
        # Final layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        
        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        """Initialize model weights."""
        # Initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize positional embeddings with sin-cos
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size, int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(pos_embed.unsqueeze(0))

        # Initialize patch embedding like nn.Linear
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.s_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.s_embedder.mlp[2].weight, std=0.02)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patch tokens back to image format.
        
        Args:
            x: Tensor of shape (B, N, patch_size^2 * C).
            
        Returns:
            Image tensor of shape (B, C, H, W).
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = rearrange(x, "b h w p1 p2 c -> b c (h p1) (w p2)")
        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Forward pass of the DiT model.
        
        Implements the solution function f_θ(x_t, t, s) that predicts
        the state at time s given the state x_t at time t.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            t: Current timestep of shape (B,), values in [0, 1].
            s: Target timestep of shape (B,), values in [0, 1].
            y: Class labels of shape (B,).
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance).
            
        Returns:
            Predicted output of shape (B, C, H, W).
        """
        # Embed patches
        x = self.x_embedder(x) + self.pos_embed  # (B, N, D)
        
        # Embed timesteps
        t_emb = self.t_embedder(t)  # (B, D)
        s_emb = self.s_embedder(s)  # (B, D)
        
        # Embed class labels
        y_emb = self.y_embedder(y, self.training)  # (B, D)
        
        # Combine conditioning: c = t_emb + s_emb + y_emb
        c = t_emb + s_emb + y_emb  # (B, D)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, c)
        
        # Final layer
        x = self.final_layer(x, c)  # (B, N, patch_size^2 * C)
        
        # Unpatchify to image format
        x = self.unpatchify(x)  # (B, C, H, W)
        
        return x

    def forward_with_cfg(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        s: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float = 1.5,
    ) -> torch.Tensor:
        """
        Forward pass with classifier-free guidance.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            t: Current timestep of shape (B,).
            s: Target timestep of shape (B,).
            y: Class labels of shape (B,).
            cfg_scale: CFG scale (>1.0 for stronger guidance).
            
        Returns:
            CFG-adjusted output of shape (B, C, H, W).
        """
        # Prepare inputs for conditional and unconditional forward passes
        half = x.shape[0] // 2
        
        # Create combined batch for efficient computation
        combined_x = torch.cat([x, x], dim=0)
        combined_t = torch.cat([t, t], dim=0)
        combined_s = torch.cat([s, s], dim=0)
        
        # Conditional labels and unconditional (null) labels
        y_null = torch.full_like(y, self.y_embedder.num_classes)
        combined_y = torch.cat([y, y_null], dim=0)
        
        # Forward pass
        model_out = self.forward(combined_x, combined_t, combined_s, combined_y)
        
        # Split conditional and unconditional outputs
        cond_out, uncond_out = model_out.chunk(2, dim=0)
        
        # Apply CFG
        cfg_out = uncond_out + cfg_scale * (cond_out - uncond_out)
        
        return cfg_out


# Model configurations following DiT paper
def DiT_XL_2(**kwargs) -> DiT:
    """DiT-XL/2 model (largest)."""
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_L_2(**kwargs) -> DiT:
    """DiT-L/2 model."""
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_M_2(**kwargs) -> DiT:
    """DiT-M/2 model (medium, not in original DiT but used in SoFlow)."""
    return DiT(depth=20, hidden_size=896, patch_size=2, num_heads=14, **kwargs)


def DiT_B_2(**kwargs) -> DiT:
    """DiT-B/2 model (base)."""
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_S_2(**kwargs) -> DiT:
    """DiT-S/2 model (small)."""
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


# Model registry
DIT_MODELS = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-L/2": DiT_L_2,
    "DiT-M/2": DiT_M_2,
    "DiT-B/2": DiT_B_2,
    "DiT-S/2": DiT_S_2,
}

