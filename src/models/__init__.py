"""
Deep Learning Models for EEG Classification

Available architectures:
- EEGNet: Compact CNN baseline (Lawhern et al., 2018)
- CNN-LSTM: Hybrid spatial-temporal architecture
- EEGTransformer: Attention-based model
"""

from .eegnet import EEGNet
from .cnn_lstm import CNNLSTM
from .transformer import EEGTransformer

__all__ = ["EEGNet", "CNNLSTM", "EEGTransformer"]
