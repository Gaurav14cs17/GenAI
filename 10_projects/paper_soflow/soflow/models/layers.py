"""
Custom layers for the Diffusion Transformer (DiT) architecture.

Includes:
- Patch embedding
- Timestep and label embeddings
- Adaptive Layer Normalization (AdaLN)
- Final prediction layer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive modulation: x * (1 + scale) + shift."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding.
    
    Args:
        img_size: Input image size (assumed square).
        patch_size: Patch size (assumed square).
        in_channels: Number of input channels.
        embed_dim: Embedding dimension.
        bias: Whether to use bias in projection.
    """
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 2,
        in_channels: int = 4,
        embed_dim: int = 768,
        bias: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim).
        """
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    
    Uses sinusoidal positional embeddings followed by an MLP.
    
    Args:
        hidden_size: Output embedding dimension.
        frequency_embedding_size: Dimension of sinusoidal embedding.
    """
    
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        
        Args:
            t: 1D tensor of N timesteps (can be fractional).
            dim: Dimension of the output.
            max_period: Controls the minimum frequency of the embeddings.
            
        Returns:
            Tensor of shape (N, dim) with positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Tensor of timesteps of shape (B,).
            
        Returns:
            Timestep embeddings of shape (B, hidden_size).
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations.
    
    Supports dropout for classifier-free guidance training.
    
    Args:
        num_classes: Number of classes.
        hidden_size: Embedding dimension.
        dropout_prob: Probability of dropping labels (for CFG).
    """
    
    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float = 0.0):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels: torch.Tensor, force_drop_ids: torch.Tensor = None) -> torch.Tensor:
        """
        Drop labels to enable classifier-free guidance.
        
        Args:
            labels: Class labels of shape (B,).
            force_drop_ids: Optional mask to force dropping specific labels.
            
        Returns:
            Labels with some replaced by the null class.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(
        self, labels: torch.Tensor, train: bool = True, force_drop_ids: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            labels: Class labels of shape (B,).
            train: Whether in training mode.
            force_drop_ids: Optional mask for CFG.
            
        Returns:
            Label embeddings of shape (B, hidden_size).
        """
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class AdaLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN) with zero initialization.
    
    Modulates normalized features using conditioning signal.
    
    Args:
        hidden_size: Feature dimension.
    """
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        # Zero-initialize the modulation
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> tuple:
        """
        Args:
            x: Input features of shape (B, N, D).
            c: Conditioning signal of shape (B, D).
            
        Returns:
            Tuple of (shift1, scale1, gate1, shift2, scale2, gate2).
        """
        return self.adaLN_modulation(c).chunk(6, dim=1)


class FinalLayer(nn.Module):
    """
    Final layer of DiT that predicts the output.
    
    Uses adaptive layer norm for final modulation.
    
    Args:
        hidden_size: Feature dimension.
        patch_size: Patch size for unpatchifying.
        out_channels: Number of output channels.
    """
    
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )
        # Zero-initialize
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape (B, N, D).
            c: Conditioning signal of shape (B, D).
            
        Returns:
            Output of shape (B, N, patch_size^2 * out_channels).
        """
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention with QKV projection.
    
    Args:
        dim: Input dimension.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in QKV projection.
        attn_drop: Attention dropout rate.
        proj_drop: Output projection dropout rate.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (B, N, D).
            
        Returns:
            Output of shape (B, N, D).
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Use scaled dot-product attention (Flash Attention when available)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP block with GELU activation.
    
    Args:
        in_features: Input dimension.
        hidden_features: Hidden dimension (default: 4x input).
        out_features: Output dimension (default: same as input).
        drop: Dropout rate.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm conditioning.
    
    Args:
        hidden_size: Feature dimension.
        num_heads: Number of attention heads.
        mlp_ratio: MLP hidden dimension ratio.
    """
    
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )
        # Zero-initialize
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (B, N, D).
            c: Conditioning of shape (B, D).
            
        Returns:
            Output of shape (B, N, D).
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

