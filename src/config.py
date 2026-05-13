"""
Configuration module for EEG Motor Imagery Classification.

All hyperparameters, paths, and settings are centralized here for reproducibility.
"""

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model checkpoints
MODELS_DIR = PROJECT_ROOT / "models"

# Reports and figures
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """
    Set random seeds for full reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for the BCI Competition IV Dataset 2a."""

    # Dataset info
    dataset_name: str = "BNCI2014001"
    n_subjects: int = 9
    n_channels: int = 22
    n_classes: int = 4
    sampling_rate: int = 250  # Hz

    # Class labels
    class_names: List[str] = field(default_factory=lambda: [
        "Left Hand",
        "Right Hand",
        "Feet",
        "Tongue"
    ])

    # Subjects to use (1-indexed as in the dataset)
    subjects: List[int] = field(default_factory=lambda: list(range(1, 10)))

    # Default subject for quick training
    default_subject: int = 1

    # Channel names (10-20 system)
    channel_names: List[str] = field(default_factory=lambda: [
        "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
        "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
        "CP3", "CP1", "CPz", "CP2", "CP4",
        "P1", "Pz", "P2", "POz"
    ])


# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================

@dataclass
class PreprocessConfig:
    """Configuration for EEG preprocessing pipeline."""

    # Bandpass filter (mu and beta rhythms for motor imagery)
    low_freq: float = 8.0   # Hz
    high_freq: float = 30.0  # Hz

    # Epoching (relative to cue onset)
    tmin: float = 0.5   # seconds after cue
    tmax: float = 2.5   # seconds after cue
    baseline: Optional[Tuple[float, float]] = None  # No baseline correction

    # Computed properties
    @property
    def epoch_duration(self) -> float:
        """Duration of each epoch in seconds."""
        return self.tmax - self.tmin

    @property
    def n_timepoints(self) -> int:
        """Number of time samples per epoch at 250 Hz."""
        return int(self.epoch_duration * 250)

    # Normalization
    normalize: bool = True
    normalization_mode: str = "zscore"  # "zscore" or "minmax"

    # Train/val/test split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Artifact rejection threshold (in microvolts)
    reject_threshold: Optional[float] = 100e-6  # 100 µV


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

@dataclass
class EEGNetConfig:
    """
    Configuration for EEGNet (Lawhern et al., 2018).

    Architecture: Temporal Conv -> Depthwise Conv -> Separable Conv -> Classifier
    """
    # Input shape
    n_channels: int = 22
    n_timepoints: int = 500  # 2 seconds at 250 Hz
    n_classes: int = 4

    # Temporal convolution
    F1: int = 8  # Number of temporal filters
    kernel_length: int = 64  # Temporal kernel size (250ms at 250Hz)

    # Depthwise convolution
    D: int = 2  # Depth multiplier

    # Separable convolution
    F2: int = 16  # F1 * D

    # Pooling
    pool1_size: Tuple[int, int] = (1, 4)
    pool2_size: Tuple[int, int] = (1, 8)

    # Regularization
    dropout_rate: float = 0.5

    def __post_init__(self):
        self.F2 = self.F1 * self.D


@dataclass
class CNNLSTMConfig:
    """Configuration for CNN-LSTM hybrid architecture."""

    # Input shape
    n_channels: int = 22
    n_timepoints: int = 500
    n_classes: int = 4

    # CNN spatial feature extractor
    cnn_filters: List[int] = field(default_factory=lambda: [32, 64])
    cnn_kernel_size: int = 3
    cnn_pool_size: int = 2

    # LSTM temporal processing
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_bidirectional: bool = True

    # Classification head
    fc_hidden_size: int = 64

    # Regularization
    dropout_rate: float = 0.5


@dataclass
class TransformerConfig:
    """Configuration for EEG Transformer architecture."""

    # Input shape
    n_channels: int = 22
    n_timepoints: int = 500
    n_classes: int = 4

    # Patch embedding
    patch_size: int = 25  # 100ms patches at 250Hz
    embed_dim: int = 128

    # Transformer encoder
    num_heads: int = 8
    num_layers: int = 4
    mlp_ratio: float = 4.0

    # Regularization
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1

    @property
    def num_patches(self) -> int:
        """Number of time patches."""
        return self.n_timepoints // self.patch_size


# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100

    # Early stopping
    early_stopping_patience: int = 15
    early_stopping_min_delta: float = 1e-4

    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_min_lr: float = 1e-6

    # Gradient clipping
    gradient_clip_val: Optional[float] = 1.0

    # Logging
    log_interval: int = 10  # Log every N batches

    # Checkpointing
    save_best_only: bool = True
    checkpoint_metric: str = "val_accuracy"


# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

@dataclass
class EvalConfig:
    """Configuration for model evaluation."""

    # Cross-subject evaluation
    run_loso: bool = False  # Leave-One-Subject-Out (set True for full eval)

    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy",
        "balanced_accuracy",
        "f1_macro",
        "cohen_kappa"
    ])

    # Confusion matrix
    normalize_confusion_matrix: bool = True


# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

@dataclass
class VizConfig:
    """Configuration for visualizations."""

    # Figure settings
    figure_dpi: int = 300
    figure_format: str = "png"

    # Color scheme
    class_colors: List[str] = field(default_factory=lambda: [
        "#FF6B6B",  # Left Hand - Red
        "#4ECDC4",  # Right Hand - Teal
        "#45B7D1",  # Feet - Blue
        "#96CEB4"   # Tongue - Green
    ])

    # Style
    style: str = "seaborn-v0_8-whitegrid"

    # t-SNE
    tsne_perplexity: int = 30
    tsne_n_iter: int = 1000


# =============================================================================
# DEFAULT INSTANCES
# =============================================================================

# Create default config instances
data_config = DataConfig()
preprocess_config = PreprocessConfig()
eegnet_config = EEGNetConfig()
cnn_lstm_config = CNNLSTMConfig()
transformer_config = TransformerConfig()
training_config = TrainingConfig()
eval_config = EvalConfig()
viz_config = VizConfig()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_path(model_name: str, subject_id: Optional[int] = None) -> Path:
    """
    Get the path for saving/loading a model checkpoint.

    Args:
        model_name: Name of the model (e.g., "eegnet", "cnn_lstm", "transformer").
        subject_id: Subject ID for subject-specific models.

    Returns:
        Path to the model checkpoint file.
    """
    if subject_id is not None:
        filename = f"{model_name}_subject{subject_id}.pt"
    else:
        filename = f"{model_name}_loso.pt"
    return MODELS_DIR / filename


def get_figure_path(name: str) -> Path:
    """
    Get the path for saving a figure.

    Args:
        name: Name of the figure (without extension).

    Returns:
        Path to the figure file.
    """
    return FIGURES_DIR / f"{name}.{viz_config.figure_format}"


def print_config() -> None:
    """Print all configuration settings."""
    print("=" * 60)
    print("EEG Motor Imagery Classification - Configuration")
    print("=" * 60)
    print(f"\nDevice: {DEVICE}")
    print(f"Random Seed: {SEED}")
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"\nData Config: {data_config}")
    print(f"\nPreprocess Config: {preprocess_config}")
    print(f"\nTraining Config: {training_config}")
    print("=" * 60)


if __name__ == "__main__":
    set_seed()
    print_config()
