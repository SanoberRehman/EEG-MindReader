"""
Training module for EEG Motor Imagery Classification.

Provides training loop with early stopping, learning rate scheduling,
gradient clipping, checkpointing, and logging.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import (
    training_config,
    TrainingConfig,
    DEVICE,
    MODELS_DIR,
    set_seed,
    get_model_path,
)


@dataclass
class TrainingHistory:
    """Container for training metrics history."""

    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    epoch_times: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_acc: float = 0.0
    total_training_time: float = 0.0

    def to_dict(self) -> Dict:
        """Convert history to dictionary."""
        return {
            "train_loss": self.train_loss,
            "train_acc": self.train_acc,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "learning_rates": self.learning_rates,
            "epoch_times": self.epoch_times,
            "best_epoch": self.best_epoch,
            "best_val_acc": self.best_val_acc,
            "total_training_time": self.total_training_time,
        }


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        mode: str = "max"
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement.
            min_delta: Minimum change to qualify as improvement.
            mode: "max" for accuracy, "min" for loss.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip: Optional[float] = None
) -> Tuple[float, float]:
    """
    Train for one epoch.

    Args:
        model: Neural network model.
        train_loader: Training data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        gradient_clip: Max gradient norm for clipping.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

        optimizer.step()

        # Track metrics
        total_loss += loss.item() * batch_x.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(batch_y).sum().item()
        total += batch_y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate the model.

    Args:
        model: Neural network model.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Device to validate on.

    Returns:
        Tuple of (average_loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[TrainingConfig] = None,
    device: Optional[torch.device] = None,
    model_name: str = "model",
    subject_id: Optional[int] = None,
    verbose: bool = True
) -> Tuple[nn.Module, TrainingHistory]:
    """
    Full training loop with early stopping and checkpointing.

    Args:
        model: Neural network model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Training configuration.
        device: Device to train on.
        model_name: Name for saving checkpoints.
        subject_id: Subject ID for checkpoint naming.
        verbose: Whether to print progress.

    Returns:
        Tuple of (trained_model, training_history).
    """
    if config is None:
        config = training_config
    if device is None:
        device = DEVICE

    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Learning rate scheduler
    scheduler = None
    if config.use_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=config.scheduler_patience,
            factor=config.scheduler_factor,
            min_lr=config.scheduler_min_lr
        )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        mode="max"
    )

    # Training history
    history = TrainingHistory()
    best_model_state = None

    # Checkpoint path
    checkpoint_path = get_model_path(model_name, subject_id)

    if verbose:
        print(f"\nTraining {model_name} on {device}")
        print(f"  Epochs: {config.num_epochs}, Batch size: {config.batch_size}")
        print(f"  Learning rate: {config.learning_rate}, Weight decay: {config.weight_decay}")
        print("-" * 60)

    start_time = time.time()

    # Training loop
    for epoch in range(config.num_epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            gradient_clip=config.gradient_clip_val
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step(val_acc)

        # Record history
        epoch_time = time.time() - epoch_start
        history.train_loss.append(train_loss)
        history.train_acc.append(train_acc)
        history.val_loss.append(val_loss)
        history.val_acc.append(val_acc)
        history.learning_rates.append(current_lr)
        history.epoch_times.append(epoch_time)

        # Check for best model
        if val_acc > history.best_val_acc:
            history.best_val_acc = val_acc
            history.best_epoch = epoch
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if config.save_best_only:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": best_model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                }, checkpoint_path)

        # Print progress
        if verbose:
            print(
                f"Epoch {epoch+1:3d}/{config.num_epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

        # Early stopping check
        if early_stopping(val_acc):
            if verbose:
                print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Record total training time
    history.total_training_time = time.time() - start_time

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    if verbose:
        print("-" * 60)
        print(f"Training complete in {history.total_training_time:.1f}s")
        print(f"Best validation accuracy: {history.best_val_acc:.4f} (epoch {history.best_epoch+1})")

    return model, history


def load_checkpoint(
    model: nn.Module,
    model_name: str,
    subject_id: Optional[int] = None,
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict]:
    """
    Load model from checkpoint.

    Args:
        model: Model architecture (uninitialized weights).
        model_name: Name of the model.
        subject_id: Subject ID.
        device: Device to load to.

    Returns:
        Tuple of (model, checkpoint_info).
    """
    if device is None:
        device = DEVICE

    checkpoint_path = get_model_path(model_name, subject_id)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    info = {
        "epoch": checkpoint.get("epoch", -1),
        "val_acc": checkpoint.get("val_acc", 0),
        "val_loss": checkpoint.get("val_loss", 0),
    }

    return model, info


def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get model predictions on a dataset.

    Args:
        model: Trained model.
        data_loader: Data loader.
        device: Device to run on.

    Returns:
        Tuple of (predictions, probabilities, true_labels).
    """
    if device is None:
        device = DEVICE

    model = model.to(device)
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)

            outputs = model(batch_x)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(batch_y.numpy())

    predictions = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probs)
    true_labels = np.concatenate(all_labels)

    return predictions, probabilities, true_labels


def extract_features(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from the penultimate layer.

    Args:
        model: Trained model (must have get_features method).
        data_loader: Data loader.
        device: Device to run on.

    Returns:
        Tuple of (features, labels).
    """
    if device is None:
        device = DEVICE

    model = model.to(device)
    model.eval()

    all_features = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)

            if hasattr(model, "get_features"):
                features = model.get_features(batch_x)
            else:
                # Fallback: use forward hook on last layer before classifier
                features = batch_x  # Placeholder

            all_features.append(features.cpu().numpy())
            all_labels.append(batch_y.numpy())

    features = np.concatenate(all_features)
    labels = np.concatenate(all_labels)

    return features, labels


if __name__ == "__main__":
    # Test training utilities
    from torch.utils.data import TensorDataset, DataLoader

    print("Testing training module...")
    print("=" * 50)

    set_seed()

    # Create dummy data
    n_samples = 200
    n_channels = 22
    n_timepoints = 500
    n_classes = 4

    X = torch.randn(n_samples, n_channels, n_timepoints)
    y = torch.randint(0, n_classes, (n_samples,))

    # Split
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)

    train_dataset = TensorDataset(X[:train_size], y[:train_size])
    val_dataset = TensorDataset(X[train_size:train_size+val_size], y[train_size:train_size+val_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Create model
    from .models.eegnet import EEGNet
    model = EEGNet(n_channels=n_channels, n_timepoints=n_timepoints, n_classes=n_classes)

    # Train with reduced epochs for testing
    test_config = TrainingConfig(num_epochs=5, early_stopping_patience=3)

    model, history = train_model(
        model, train_loader, val_loader,
        config=test_config,
        model_name="eegnet_test",
        verbose=True
    )

    print("\nTraining history:")
    print(f"  Best val acc: {history.best_val_acc:.4f}")
    print(f"  Total time: {history.total_training_time:.1f}s")

    # Test predictions
    preds, probs, labels = get_predictions(model, val_loader)
    print(f"\nPredictions shape: {preds.shape}")
    print(f"Probabilities shape: {probs.shape}")

    print("\nAll tests passed!")
