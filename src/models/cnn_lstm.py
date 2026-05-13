"""
CNN-LSTM Hybrid Architecture for EEG Classification.

Combines spatial CNN feature extraction with temporal LSTM processing
to capture both spatial patterns across electrodes and temporal dynamics.

Architecture:
    1. Spatial CNN: Extract features from channel dimension
    2. Temporal CNN: Extract local temporal patterns
    3. Bidirectional LSTM: Capture long-range temporal dependencies
    4. Attention: Weight important time steps
    5. Classification Head: Dense layers with softmax
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from ..config import cnn_lstm_config, CNNLSTMConfig


class SpatialBlock(nn.Module):
    """CNN block for spatial feature extraction across EEG channels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_electrodes: int,
        kernel_size: int = 3,
        dropout: float = 0.5
    ):
        """
        Initialize spatial block.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.
            n_electrodes: Number of EEG electrodes.
            kernel_size: Convolution kernel size.
            dropout: Dropout probability.
        """
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels * n_electrodes,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        x = self.bn(x)
        x = F.elu(x)
        x = self.dropout(x)
        return x


class TemporalAttention(nn.Module):
    """Attention mechanism to weight important time steps."""

    def __init__(self, hidden_size: int):
        """
        Initialize attention.

        Args:
            hidden_size: Size of hidden representations.
        """
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attention to LSTM outputs.

        Args:
            lstm_output: LSTM output of shape (batch, seq_len, hidden_size).

        Returns:
            Tuple of (context_vector, attention_weights).
        """
        # Compute attention scores
        scores = self.attention(lstm_output)  # (batch, seq_len, 1)
        weights = F.softmax(scores, dim=1)  # (batch, seq_len, 1)

        # Weighted sum
        context = torch.sum(weights * lstm_output, dim=1)  # (batch, hidden_size)

        return context, weights.squeeze(-1)


class CNNLSTM(nn.Module):
    """
    CNN-LSTM hybrid for EEG classification.

    Uses CNN layers to extract spatial and local temporal features,
    then LSTM layers to capture long-range temporal dependencies.
    """

    def __init__(
        self,
        n_channels: int = None,
        n_timepoints: int = None,
        n_classes: int = None,
        cnn_filters: List[int] = None,
        cnn_kernel_size: int = None,
        cnn_pool_size: int = None,
        lstm_hidden_size: int = None,
        lstm_num_layers: int = None,
        lstm_bidirectional: bool = None,
        fc_hidden_size: int = None,
        dropout_rate: float = None,
        config: Optional[CNNLSTMConfig] = None
    ):
        """
        Initialize CNN-LSTM model.

        Args:
            n_channels: Number of EEG channels.
            n_timepoints: Number of time samples.
            n_classes: Number of output classes.
            cnn_filters: List of CNN filter counts.
            cnn_kernel_size: CNN kernel size.
            cnn_pool_size: CNN pooling size.
            lstm_hidden_size: LSTM hidden state size.
            lstm_num_layers: Number of LSTM layers.
            lstm_bidirectional: Whether to use bidirectional LSTM.
            fc_hidden_size: Fully connected hidden size.
            dropout_rate: Dropout probability.
            config: CNNLSTMConfig object.
        """
        super().__init__()

        # Use config or defaults
        if config is None:
            config = cnn_lstm_config

        self.n_channels = n_channels or config.n_channels
        self.n_timepoints = n_timepoints or config.n_timepoints
        self.n_classes = n_classes or config.n_classes
        self.cnn_filters = cnn_filters or config.cnn_filters
        self.cnn_kernel_size = cnn_kernel_size or config.cnn_kernel_size
        self.cnn_pool_size = cnn_pool_size or config.cnn_pool_size
        self.lstm_hidden_size = lstm_hidden_size or config.lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers or config.lstm_num_layers
        self.lstm_bidirectional = lstm_bidirectional if lstm_bidirectional is not None else config.lstm_bidirectional
        self.fc_hidden_size = fc_hidden_size or config.fc_hidden_size
        self.dropout_rate = dropout_rate or config.dropout_rate

        self._build_network()

    def _build_network(self) -> None:
        """Construct network layers."""

        # CNN feature extractor
        # Input: (batch, channels, timepoints)
        cnn_layers = []
        in_channels = self.n_channels

        for i, out_channels in enumerate(self.cnn_filters):
            cnn_layers.extend([
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=self.cnn_kernel_size,
                    padding=self.cnn_kernel_size // 2
                ),
                nn.BatchNorm1d(out_channels),
                nn.ELU(),
                nn.MaxPool1d(self.cnn_pool_size),
                nn.Dropout(self.dropout_rate)
            ])
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers)

        # Calculate sequence length after CNN
        self._seq_len = self.n_timepoints
        for _ in self.cnn_filters:
            self._seq_len = self._seq_len // self.cnn_pool_size

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_filters[-1],
            hidden_size=self.lstm_hidden_size,
            num_layers=self.lstm_num_layers,
            batch_first=True,
            bidirectional=self.lstm_bidirectional,
            dropout=self.dropout_rate if self.lstm_num_layers > 1 else 0
        )

        # Attention mechanism
        lstm_output_size = self.lstm_hidden_size * (2 if self.lstm_bidirectional else 1)
        self.attention = TemporalAttention(lstm_output_size)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_size, self.fc_hidden_size),
            nn.ELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.fc_hidden_size, self.n_classes)
        )

        # Store feature size for external use
        self._feature_size = lstm_output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, timepoints).

        Returns:
            Logits of shape (batch, n_classes).
        """
        # CNN feature extraction
        # (batch, channels, timepoints) -> (batch, cnn_filters[-1], reduced_timepoints)
        x = self.cnn(x)

        # Prepare for LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_hidden * num_directions)

        # Attention pooling
        context, self._attention_weights = self.attention(lstm_out)

        # Classification
        logits = self.classifier(context)

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification layer.

        Args:
            x: Input tensor of shape (batch, channels, timepoints).

        Returns:
            Features of shape (batch, feature_size).
        """
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        return context

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get temporal attention weights.

        Args:
            x: Input tensor.

        Returns:
            Attention weights of shape (batch, seq_len).
        """
        with torch.no_grad():
            _ = self.forward(x)
        return self._attention_weights

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"CNNLSTM(\n"
            f"  channels={self.n_channels}, timepoints={self.n_timepoints}, classes={self.n_classes}\n"
            f"  cnn_filters={self.cnn_filters}, lstm_hidden={self.lstm_hidden_size}\n"
            f"  lstm_layers={self.lstm_num_layers}, bidirectional={self.lstm_bidirectional}\n"
            f"  dropout={self.dropout_rate}, params={self.count_parameters():,}\n"
            f")"
        )


class CNNLSTM2D(nn.Module):
    """
    Alternative CNN-LSTM with 2D convolutions.

    Treats EEG as a 2D image (channels x time) and uses 2D convolutions
    to jointly learn spatial and temporal patterns.
    """

    def __init__(
        self,
        n_channels: int = 22,
        n_timepoints: int = 500,
        n_classes: int = 4,
        dropout_rate: float = 0.5
    ):
        """
        Initialize 2D CNN-LSTM.

        Args:
            n_channels: Number of EEG channels.
            n_timepoints: Number of time samples.
            n_classes: Number of output classes.
            dropout_rate: Dropout probability.
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.n_classes = n_classes

        # 2D CNN: treats input as (1, channels, timepoints)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 5), padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((1, 2))

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((1, 2))

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 2))

        self.dropout = nn.Dropout(dropout_rate)

        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            dummy = self._forward_cnn(dummy)
            self._cnn_output_size = dummy.shape[1]
            self._seq_len = dummy.shape[2]

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self._cnn_output_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, n_classes)
        )

    def _forward_cnn(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through CNN layers."""
        x = F.elu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout(x)

        x = F.elu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout(x)

        x = F.elu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = self.dropout(x)

        # Reshape: (batch, channels, h, w) -> (batch, channels*h, w)
        batch, c, h, w = x.shape
        x = x.view(batch, c * h, w)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input of shape (batch, n_channels, n_timepoints).

        Returns:
            Logits of shape (batch, n_classes).
        """
        # Add channel dim: (batch, 1, channels, timepoints)
        x = x.unsqueeze(1)

        # CNN
        x = self._forward_cnn(x)

        # Prepare for LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, (h_n, _) = self.lstm(x)

        # Use last hidden state from both directions
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_combined = torch.cat([h_forward, h_backward], dim=1)

        # Classify
        logits = self.classifier(h_combined)

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier."""
        x = x.unsqueeze(1)
        x = self._forward_cnn(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        return torch.cat([h_forward, h_backward], dim=1)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_cnn_lstm(
    n_channels: int = 22,
    n_timepoints: int = 500,
    n_classes: int = 4,
    variant: str = "standard"
) -> nn.Module:
    """
    Factory function for CNN-LSTM variants.

    Args:
        n_channels: Number of EEG channels.
        n_timepoints: Number of time samples.
        n_classes: Number of classes.
        variant: "standard" or "2d".

    Returns:
        CNN-LSTM model.
    """
    if variant == "standard":
        return CNNLSTM(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        )
    elif variant == "2d":
        return CNNLSTM2D(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


if __name__ == "__main__":
    # Test CNN-LSTM
    print("Testing CNN-LSTM...")
    print("=" * 50)

    # Create model
    model = CNNLSTM(n_channels=22, n_timepoints=500, n_classes=4)
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

    # Test attention weights
    attn = model.get_attention_weights(x)
    print(f"Attention weights shape: {attn.shape}")

    # Test 2D variant
    print("\n" + "=" * 50)
    print("Testing CNN-LSTM 2D...")
    model_2d = CNNLSTM2D(n_channels=22, n_timepoints=500, n_classes=4)
    output_2d = model_2d(x)
    print(f"2D variant output shape: {output_2d.shape}")
    print(f"2D variant params: {model_2d.count_parameters():,}")

    print("\nAll tests passed!")
