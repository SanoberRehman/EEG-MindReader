"""
EEGNet: A Compact Convolutional Neural Network for EEG-based BCIs.

Implementation based on:
Lawhern, V. J., et al. (2018). EEGNet: A compact convolutional neural network
for EEG-based brain-computer interfaces. Journal of Neural Engineering, 15(5).

Architecture:
    1. Temporal Convolution: Learn frequency filters
    2. Depthwise Convolution: Learn spatial filters (per-channel)
    3. Separable Convolution: Learn temporal patterns
    4. Classification Head: Dense layer with softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..config import eegnet_config, EEGNetConfig


class EEGNet(nn.Module):
    """
    EEGNet: Compact CNN for EEG classification.

    The architecture uses depthwise and separable convolutions to efficiently
    learn spatial and temporal features from EEG signals with minimal parameters.
    """

    def __init__(
        self,
        n_channels: int = None,
        n_timepoints: int = None,
        n_classes: int = None,
        F1: int = None,
        D: int = None,
        F2: int = None,
        kernel_length: int = None,
        pool1_size: Tuple[int, int] = None,
        pool2_size: Tuple[int, int] = None,
        dropout_rate: float = None,
        config: Optional[EEGNetConfig] = None
    ):
        """
        Initialize EEGNet.

        Args:
            n_channels: Number of EEG channels.
            n_timepoints: Number of time samples per trial.
            n_classes: Number of output classes.
            F1: Number of temporal filters.
            D: Depth multiplier for depthwise convolution.
            F2: Number of pointwise filters (F1 * D).
            kernel_length: Length of temporal convolution kernel.
            pool1_size: First pooling kernel size.
            pool2_size: Second pooling kernel size.
            dropout_rate: Dropout probability.
            config: EEGNetConfig object (overrides individual params).
        """
        super().__init__()

        # Use config or defaults
        if config is None:
            config = eegnet_config

        self.n_channels = n_channels or config.n_channels
        self.n_timepoints = n_timepoints or config.n_timepoints
        self.n_classes = n_classes or config.n_classes
        self.F1 = F1 or config.F1
        self.D = D or config.D
        self.F2 = F2 or config.F2
        self.kernel_length = kernel_length or config.kernel_length
        self.pool1_size = pool1_size or config.pool1_size
        self.pool2_size = pool2_size or config.pool2_size
        self.dropout_rate = dropout_rate or config.dropout_rate

        # Ensure F2 = F1 * D
        self.F2 = self.F1 * self.D

        # Build the network
        self._build_network()

    def _build_network(self) -> None:
        """Construct all network layers."""

        # Block 1: Temporal Convolution
        # Input: (batch, 1, channels, timepoints)
        # Learns F1 temporal filters of length kernel_length
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.F1,
            kernel_size=(1, self.kernel_length),
            padding=(0, self.kernel_length // 2),
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.F1)

        # Block 1: Depthwise Convolution
        # Learns spatial filters (one per channel, D copies per temporal filter)
        self.conv2 = nn.Conv2d(
            in_channels=self.F1,
            out_channels=self.F1 * self.D,
            kernel_size=(self.n_channels, 1),
            groups=self.F1,  # Depthwise: each input channel gets its own filter
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(self.F1 * self.D)
        self.pool1 = nn.AvgPool2d(kernel_size=self.pool1_size)
        self.dropout1 = nn.Dropout(self.dropout_rate)

        # Block 2: Separable Convolution
        # Depthwise temporal convolution
        self.conv3 = nn.Conv2d(
            in_channels=self.F2,
            out_channels=self.F2,
            kernel_size=(1, 16),  # Temporal kernel
            padding=(0, 8),
            groups=self.F2,  # Depthwise
            bias=False
        )
        # Pointwise convolution (1x1)
        self.conv4 = nn.Conv2d(
            in_channels=self.F2,
            out_channels=self.F2,
            kernel_size=(1, 1),
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.F2)
        self.pool2 = nn.AvgPool2d(kernel_size=self.pool2_size)
        self.dropout2 = nn.Dropout(self.dropout_rate)

        # Calculate flattened size
        self._flat_size = self._get_flat_size()

        # Classification head
        self.classifier = nn.Linear(self._flat_size, self.n_classes)

    def _get_flat_size(self) -> int:
        """Calculate the flattened feature size after convolutions."""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_timepoints)
            x = self._forward_features(x)
            return x.view(1, -1).size(1)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolutional layers."""
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, timepoints)
               or (batch, 1, channels, timepoints).

        Returns:
            Logits of shape (batch, n_classes).
        """
        # Add channel dimension if needed: (B, C, T) -> (B, 1, C, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # Feature extraction
        x = self._forward_features(x)

        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before the classification layer.

        Useful for t-SNE visualization and interpretability.

        Args:
            x: Input tensor of shape (batch, channels, timepoints).

        Returns:
            Features of shape (batch, flat_size).
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self._forward_features(x)
        x = x.view(x.size(0), -1)

        return x

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"EEGNet(\n"
            f"  channels={self.n_channels}, timepoints={self.n_timepoints}, classes={self.n_classes}\n"
            f"  F1={self.F1}, D={self.D}, F2={self.F2}, kernel_length={self.kernel_length}\n"
            f"  dropout={self.dropout_rate}, params={self.count_parameters():,}\n"
            f")"
        )


class EEGNetWithAttention(nn.Module):
    """
    EEGNet variant with channel attention mechanism.

    Adds a squeeze-and-excitation style attention block to weight
    the importance of different EEG channels.
    """

    def __init__(
        self,
        n_channels: int = None,
        n_timepoints: int = None,
        n_classes: int = None,
        config: Optional[EEGNetConfig] = None,
        reduction_ratio: int = 4
    ):
        """
        Initialize EEGNet with attention.

        Args:
            n_channels: Number of EEG channels.
            n_timepoints: Number of time samples.
            n_classes: Number of output classes.
            config: EEGNetConfig object.
            reduction_ratio: Reduction ratio for attention bottleneck.
        """
        super().__init__()

        if config is None:
            config = eegnet_config

        self.n_channels = n_channels or config.n_channels
        self.n_timepoints = n_timepoints or config.n_timepoints
        self.n_classes = n_classes or config.n_classes

        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.n_channels, self.n_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(self.n_channels // reduction_ratio, self.n_channels),
            nn.Sigmoid()
        )

        # Base EEGNet
        self.eegnet = EEGNet(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes,
            config=config
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with channel attention.

        Args:
            x: Input tensor of shape (batch, channels, timepoints).

        Returns:
            Logits of shape (batch, n_classes).
        """
        # Compute channel attention weights
        attn = self.channel_attention(x)  # (batch, channels)
        attn = attn.unsqueeze(-1)  # (batch, channels, 1)

        # Apply attention
        x = x * attn

        # Pass through EEGNet
        return self.eegnet(x)

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get channel attention weights for interpretability."""
        with torch.no_grad():
            attn = self.channel_attention(x)
        return attn


def create_eegnet(
    n_channels: int = 22,
    n_timepoints: int = 500,
    n_classes: int = 4,
    variant: str = "standard"
) -> nn.Module:
    """
    Factory function to create EEGNet variants.

    Args:
        n_channels: Number of EEG channels.
        n_timepoints: Number of time samples.
        n_classes: Number of output classes.
        variant: Model variant ("standard" or "attention").

    Returns:
        EEGNet model.
    """
    if variant == "standard":
        return EEGNet(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        )
    elif variant == "attention":
        return EEGNetWithAttention(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    # Test EEGNet
    print("Testing EEGNet...")
    print("=" * 50)

    # Create model
    model = EEGNet(n_channels=22, n_timepoints=500, n_classes=4)
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

    # Test attention variant
    print("\n" + "=" * 50)
    print("Testing EEGNet with Attention...")
    model_attn = EEGNetWithAttention(n_channels=22, n_timepoints=500, n_classes=4)
    output_attn = model_attn(x)
    print(f"Output shape: {output_attn.shape}")

    attn_weights = model_attn.get_attention_weights(x)
    print(f"Attention weights shape: {attn_weights.shape}")

    print("\nAll tests passed!")
