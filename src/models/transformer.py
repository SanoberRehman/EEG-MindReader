"""
EEG Transformer Architecture for Motor Imagery Classification.

Applies the Transformer architecture to EEG signals by:
1. Patching the temporal dimension into fixed-size windows
2. Linear embedding of patches
3. Adding positional encodings
4. Multi-head self-attention layers
5. Classification via [CLS] token

Inspired by Vision Transformer (ViT) adapted for 1D EEG signals.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import transformer_config, TransformerConfig


class PatchEmbedding(nn.Module):
    """
    Embed EEG signal patches into a latent space.

    Splits the temporal dimension into patches and projects each patch
    along with channel information into an embedding vector.
    """

    def __init__(
        self,
        n_channels: int,
        n_timepoints: int,
        patch_size: int,
        embed_dim: int
    ):
        """
        Initialize patch embedding.

        Args:
            n_channels: Number of EEG channels.
            n_timepoints: Number of time samples.
            patch_size: Size of each temporal patch.
            embed_dim: Embedding dimension.
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Number of patches
        self.n_patches = n_timepoints // patch_size

        # Patch embedding: project (channels * patch_size) -> embed_dim
        self.projection = nn.Linear(n_channels * patch_size, embed_dim)

        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create patch embeddings.

        Args:
            x: Input of shape (batch, channels, timepoints).

        Returns:
            Embeddings of shape (batch, n_patches, embed_dim).
        """
        batch_size = x.shape[0]

        # Reshape into patches: (batch, channels, n_patches, patch_size)
        x = x.unfold(dimension=2, size=self.patch_size, step=self.patch_size)

        # Flatten channels and patch: (batch, n_patches, channels * patch_size)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, self.n_patches, -1)

        # Project to embedding dimension
        x = self.projection(x)
        x = self.norm(x)

        return x


class PositionalEncoding(nn.Module):
    """Learnable positional encodings for sequence positions."""

    def __init__(self, n_positions: int, embed_dim: int, dropout: float = 0.1):
        """
        Initialize positional encoding.

        Args:
            n_positions: Maximum sequence length.
            embed_dim: Embedding dimension.
            dropout: Dropout probability.
        """
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        # Learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, n_positions, embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings.

        Args:
            x: Input of shape (batch, seq_len, embed_dim).

        Returns:
            Output with positional encodings added.
        """
        x = x + self.pos_embedding[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            dropout: Attention dropout probability.
        """
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Store attention weights for visualization
        self._attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-head self-attention.

        Args:
            x: Input of shape (batch, seq_len, embed_dim).

        Returns:
            Output of shape (batch, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        self._attention_weights = attn.detach()
        attn = self.attn_dropout(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        x = self.proj(x)
        x = self.proj_dropout(x)

        return x


class TransformerBlock(nn.Module):
    """Single Transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1
    ):
        """
        Initialize Transformer block.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: MLP hidden dim = embed_dim * mlp_ratio.
            dropout: Dropout probability.
            attention_dropout: Attention dropout probability.
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attention_dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connections.

        Args:
            x: Input of shape (batch, seq_len, embed_dim).

        Returns:
            Output of shape (batch, seq_len, embed_dim).
        """
        # Self-attention with residual
        x = x + self.attn(self.norm1(x))

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class EEGTransformer(nn.Module):
    """
    Transformer model for EEG classification.

    Treats EEG signals as sequences of temporal patches and applies
    self-attention to learn relationships between different time windows.
    """

    def __init__(
        self,
        n_channels: int = None,
        n_timepoints: int = None,
        n_classes: int = None,
        patch_size: int = None,
        embed_dim: int = None,
        num_heads: int = None,
        num_layers: int = None,
        mlp_ratio: float = None,
        dropout_rate: float = None,
        attention_dropout: float = None,
        config: Optional[TransformerConfig] = None
    ):
        """
        Initialize EEG Transformer.

        Args:
            n_channels: Number of EEG channels.
            n_timepoints: Number of time samples.
            n_classes: Number of output classes.
            patch_size: Temporal patch size.
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of Transformer layers.
            mlp_ratio: MLP expansion ratio.
            dropout_rate: Dropout probability.
            attention_dropout: Attention dropout probability.
            config: TransformerConfig object.
        """
        super().__init__()

        # Use config or defaults
        if config is None:
            config = transformer_config

        self.n_channels = n_channels or config.n_channels
        self.n_timepoints = n_timepoints or config.n_timepoints
        self.n_classes = n_classes or config.n_classes
        self.patch_size = patch_size or config.patch_size
        self.embed_dim = embed_dim or config.embed_dim
        self.num_heads = num_heads or config.num_heads
        self.num_layers = num_layers or config.num_layers
        self.mlp_ratio = mlp_ratio or config.mlp_ratio
        self.dropout_rate = dropout_rate or config.dropout_rate
        self.attention_dropout = attention_dropout or config.attention_dropout

        # Number of patches
        self.n_patches = self.n_timepoints // self.patch_size

        self._build_network()

    def _build_network(self) -> None:
        """Construct network layers."""

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            n_channels=self.n_channels,
            n_timepoints=self.n_timepoints,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim
        )

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        # Positional encoding (patches + 1 for CLS token)
        self.pos_encoding = PositionalEncoding(
            n_positions=self.n_patches + 1,
            embed_dim=self.embed_dim,
            dropout=self.dropout_rate
        )

        # Transformer encoder blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout_rate,
                attention_dropout=self.attention_dropout
            )
            for _ in range(self.num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(self.embed_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.embed_dim // 2, self.n_classes)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        # Initialize CLS token
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, channels, timepoints).

        Returns:
            Logits of shape (batch, n_classes).
        """
        batch_size = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)  # (batch, n_patches, embed_dim)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, n_patches + 1, embed_dim)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        # Use CLS token for classification
        cls_output = x[:, 0]  # (batch, embed_dim)

        # Classify
        logits = self.classifier(cls_output)

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features (CLS token embedding).

        Args:
            x: Input of shape (batch, channels, timepoints).

        Returns:
            Features of shape (batch, embed_dim).
        """
        batch_size = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoding(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)

        return x[:, 0]

    def get_attention_maps(self, x: torch.Tensor) -> list:
        """
        Get attention maps from all layers.

        Args:
            x: Input tensor.

        Returns:
            List of attention maps, one per layer.
        """
        batch_size = x.shape[0]

        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_encoding(x)

        attention_maps = []
        for block in self.transformer_blocks:
            x = block(x)
            attention_maps.append(block.attn._attention_weights)

        return attention_maps

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"EEGTransformer(\n"
            f"  channels={self.n_channels}, timepoints={self.n_timepoints}, classes={self.n_classes}\n"
            f"  patch_size={self.patch_size}, n_patches={self.n_patches}\n"
            f"  embed_dim={self.embed_dim}, heads={self.num_heads}, layers={self.num_layers}\n"
            f"  dropout={self.dropout_rate}, params={self.count_parameters():,}\n"
            f")"
        )


class EEGConformer(nn.Module):
    """
    EEG Conformer: Combines CNN and Transformer.

    Uses convolutional layers for local feature extraction followed by
    Transformer layers for global context modeling.
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_timepoints: int = 500,
        n_classes: int = 4,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize EEG Conformer.

        Args:
            n_channels: Number of EEG channels.
            n_timepoints: Number of time samples.
            n_classes: Number of classes.
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of Transformer layers.
            dropout: Dropout probability.
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints

        # Temporal convolution
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 40, kernel_size=(1, 25), padding=(0, 12)),
            nn.BatchNorm2d(40),
            nn.ELU()
        )

        # Spatial convolution
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(40, 40, kernel_size=(n_channels, 1), groups=1),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout)
        )

        # Calculate sequence length after convolutions
        seq_len = n_timepoints // 8

        # Project to embed_dim
        self.projection = nn.Linear(40, embed_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(seq_len, embed_dim, dropout)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio=2.0, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, channels, timepoints).

        Returns:
            Logits of shape (batch, n_classes).
        """
        # Add channel dim: (batch, 1, channels, timepoints)
        x = x.unsqueeze(1)

        # Convolutions
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)  # (batch, 40, 1, seq_len)

        # Reshape for transformer: (batch, seq_len, 40)
        x = x.squeeze(2).permute(0, 2, 1)

        # Project to embed_dim
        x = self.projection(x)

        # Positional encoding
        x = self.pos_encoding(x)

        # Transformer
        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classify
        return self.classifier(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = x.squeeze(2).permute(0, 2, 1)
        x = self.projection(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x)
        x = self.norm(x)
        return x.mean(dim=1)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_transformer(
    n_channels: int = 22,
    n_timepoints: int = 500,
    n_classes: int = 4,
    variant: str = "standard"
) -> nn.Module:
    """
    Factory function for Transformer variants.

    Args:
        n_channels: Number of EEG channels.
        n_timepoints: Number of time samples.
        n_classes: Number of classes.
        variant: "standard" or "conformer".

    Returns:
        Transformer model.
    """
    if variant == "standard":
        return EEGTransformer(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        )
    elif variant == "conformer":
        return EEGConformer(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    # Test Transformer models
    print("Testing EEG Transformer...")
    print("=" * 50)

    # Create model
    model = EEGTransformer(n_channels=22, n_timepoints=500, n_classes=4)
    print(model)

    # Test forward pass
    batch_size = 8
    x = torch.randn(batch_size, 22, 500)
    print(f"\nInput shape: {x.shape}")

    output = model(x)
    print(f"Output shape: {output.shape}")

    # Test feature extraction
    features = model.get_features(x)
    print(f"Features shape: {features.shape}")

    # Test attention maps
    attn_maps = model.get_attention_maps(x)
    print(f"Attention maps: {len(attn_maps)} layers, shape {attn_maps[0].shape}")

    # Test Conformer
    print("\n" + "=" * 50)
    print("Testing EEG Conformer...")
    conformer = EEGConformer(n_channels=22, n_timepoints=500, n_classes=4)
    output_conf = conformer(x)
    print(f"Conformer output shape: {output_conf.shape}")
    print(f"Conformer params: {conformer.count_parameters():,}")

    print("\nAll tests passed!")
