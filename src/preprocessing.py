"""
Preprocessing module for EEG Motor Imagery Classification.

Handles bandpass filtering, epoching, normalization, and train/val/test splitting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import mne
from mne import Epochs
from mne.io import BaseRaw
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

from .config import (
    data_config,
    preprocess_config,
    training_config,
    PROCESSED_DATA_DIR,
    set_seed,
    SEED,
)
from .data_loader import load_subject_data, get_epochs_from_raw


# Suppress MNE verbose output
mne.set_log_level("WARNING")


class EEGDataset(Dataset):
    """PyTorch Dataset for EEG data."""

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        Initialize EEG Dataset.

        Args:
            data: EEG data of shape (n_trials, n_channels, n_timepoints).
            labels: Labels of shape (n_trials,).
            transform: Optional transform to apply to data.
        """
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        y = self.labels[idx]

        if self.transform:
            x = self.transform(x)

        return x, y


def bandpass_filter(
    raw: BaseRaw,
    low_freq: float = None,
    high_freq: float = None,
    verbose: bool = False
) -> BaseRaw:
    """
    Apply bandpass filter to raw EEG data.

    Args:
        raw: MNE Raw object.
        low_freq: Low cutoff frequency (default from config).
        high_freq: High cutoff frequency (default from config).
        verbose: Whether to print MNE output.

    Returns:
        Filtered Raw object.
    """
    if low_freq is None:
        low_freq = preprocess_config.low_freq
    if high_freq is None:
        high_freq = preprocess_config.high_freq

    raw_filtered = raw.copy().filter(
        l_freq=low_freq,
        h_freq=high_freq,
        method="fir",
        fir_design="firwin",
        verbose=verbose
    )

    return raw_filtered


def create_epochs(
    raw: BaseRaw,
    events: np.ndarray,
    event_id: Dict[str, int],
    reject_threshold: Optional[float] = None,
    verbose: bool = False
) -> Epochs:
    """
    Create and clean epochs from raw data.

    Args:
        raw: MNE Raw object (should be filtered).
        events: Events array.
        event_id: Event ID dictionary.
        reject_threshold: Artifact rejection threshold in volts.
        verbose: Whether to print progress.

    Returns:
        Cleaned MNE Epochs object.
    """
    if reject_threshold is None:
        reject_threshold = preprocess_config.reject_threshold

    # Create epochs
    epochs = Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=preprocess_config.tmin,
        tmax=preprocess_config.tmax,
        baseline=preprocess_config.baseline,
        preload=True,
        reject={"eeg": reject_threshold} if reject_threshold else None,
        verbose=verbose
    )

    if verbose:
        print(f"Created {len(epochs)} epochs after artifact rejection.")

    return epochs


def normalize_epochs(
    data: np.ndarray,
    mode: str = None
) -> np.ndarray:
    """
    Normalize epoch data.

    Args:
        data: Epoch data of shape (n_trials, n_channels, n_timepoints).
        mode: Normalization mode ("zscore" or "minmax").

    Returns:
        Normalized data.
    """
    if mode is None:
        mode = preprocess_config.normalization_mode

    if mode == "zscore":
        # Z-score normalize per channel across time
        mean = data.mean(axis=2, keepdims=True)
        std = data.std(axis=2, keepdims=True)
        std[std == 0] = 1  # Avoid division by zero
        data_norm = (data - mean) / std

    elif mode == "minmax":
        # Min-max normalize per channel
        min_val = data.min(axis=2, keepdims=True)
        max_val = data.max(axis=2, keepdims=True)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        data_norm = (data - min_val) / range_val

    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    return data_norm


def epochs_to_arrays(
    epochs: Epochs,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert MNE Epochs to numpy arrays.

    Args:
        epochs: MNE Epochs object.
        normalize: Whether to normalize the data.

    Returns:
        Tuple of (data, labels) where:
            - data: shape (n_trials, n_channels, n_timepoints)
            - labels: shape (n_trials,) with values 0-3
    """
    # Get data: (n_trials, n_channels, n_timepoints)
    data = epochs.get_data()

    # Get labels and convert to 0-indexed
    labels = epochs.events[:, 2] - 1  # Convert from 1-4 to 0-3

    if normalize:
        data = normalize_epochs(data)

    return data.astype(np.float32), labels.astype(np.int64)


def split_data(
    data: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = None,
    val_ratio: float = None,
    stratify: bool = True,
    random_state: int = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Split data into train/val/test sets.

    Args:
        data: EEG data of shape (n_trials, n_channels, n_timepoints).
        labels: Labels of shape (n_trials,).
        train_ratio: Training set ratio.
        val_ratio: Validation set ratio.
        stratify: Whether to stratify by class.
        random_state: Random seed.

    Returns:
        Dictionary with "train", "val", "test" keys, each containing (data, labels).
    """
    if train_ratio is None:
        train_ratio = preprocess_config.train_ratio
    if val_ratio is None:
        val_ratio = preprocess_config.val_ratio
    if random_state is None:
        random_state = SEED

    test_ratio = 1 - train_ratio - val_ratio

    # First split: train+val vs test
    stratify_labels = labels if stratify else None
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        data, labels,
        test_size=test_ratio,
        stratify=stratify_labels,
        random_state=random_state
    )

    # Second split: train vs val
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    stratify_labels = y_trainval if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio_adjusted,
        stratify=stratify_labels,
        random_state=random_state
    )

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test)
    }


def preprocess_subject(
    subject_id: int,
    use_moabb: bool = True,
    save: bool = True,
    verbose: bool = True
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Full preprocessing pipeline for a single subject.

    Args:
        subject_id: Subject ID (1-9).
        use_moabb: Whether to use MOABB (falls back to synthetic).
        save: Whether to save processed data.
        verbose: Whether to print progress.

    Returns:
        Dictionary with train/val/test splits.
    """
    if verbose:
        print(f"\n{'='*50}")
        print(f"Preprocessing Subject {subject_id}")
        print(f"{'='*50}")

    # Load raw data
    if verbose:
        print("Loading raw data...")
    raw, events, event_id = load_subject_data(
        subject_id=subject_id,
        use_moabb=use_moabb,
        verbose=verbose
    )

    # Bandpass filter
    if verbose:
        print(f"Applying bandpass filter ({preprocess_config.low_freq}-{preprocess_config.high_freq} Hz)...")
    raw_filtered = bandpass_filter(raw, verbose=False)

    # Create epochs
    if verbose:
        print(f"Creating epochs ({preprocess_config.tmin}s to {preprocess_config.tmax}s)...")
    epochs = create_epochs(raw_filtered, events, event_id, verbose=verbose)

    # Convert to arrays
    if verbose:
        print("Converting to arrays and normalizing...")
    data, labels = epochs_to_arrays(epochs, normalize=preprocess_config.normalize)

    if verbose:
        print(f"Data shape: {data.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Class distribution: {np.bincount(labels)}")

    # Split data
    if verbose:
        print("Splitting into train/val/test...")
    splits = split_data(data, labels)

    if verbose:
        for split_name, (X, y) in splits.items():
            print(f"  {split_name}: {X.shape[0]} trials")

    # Save processed data
    if save:
        save_path = PROCESSED_DATA_DIR / f"subject_{subject_id}.npz"
        if verbose:
            print(f"Saving to {save_path}...")
        np.savez_compressed(
            save_path,
            X_train=splits["train"][0],
            y_train=splits["train"][1],
            X_val=splits["val"][0],
            y_val=splits["val"][1],
            X_test=splits["test"][0],
            y_test=splits["test"][1]
        )

    return splits


def load_processed_subject(subject_id: int) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Load preprocessed data for a subject.

    Args:
        subject_id: Subject ID.

    Returns:
        Dictionary with train/val/test splits, or None if not found.
    """
    save_path = PROCESSED_DATA_DIR / f"subject_{subject_id}.npz"

    if not save_path.exists():
        return None

    data = np.load(save_path)

    return {
        "train": (data["X_train"], data["y_train"]),
        "val": (data["X_val"], data["y_val"]),
        "test": (data["X_test"], data["y_test"])
    }


def get_subject_data(
    subject_id: int,
    use_moabb: bool = True,
    force_reprocess: bool = False,
    verbose: bool = True
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Get preprocessed data for a subject, loading from cache if available.

    Args:
        subject_id: Subject ID (1-9).
        use_moabb: Whether to use MOABB for loading.
        force_reprocess: Force reprocessing even if cached data exists.
        verbose: Whether to print progress.

    Returns:
        Dictionary with train/val/test splits.
    """
    if not force_reprocess:
        cached = load_processed_subject(subject_id)
        if cached is not None:
            if verbose:
                print(f"Loaded cached data for Subject {subject_id}")
            return cached

    return preprocess_subject(
        subject_id=subject_id,
        use_moabb=use_moabb,
        save=True,
        verbose=verbose
    )


def create_dataloaders(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]],
    batch_size: int = None,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders from data splits.

    Args:
        splits: Dictionary with train/val/test splits.
        batch_size: Batch size (default from config).
        num_workers: Number of worker processes.

    Returns:
        Dictionary with train/val/test DataLoaders.
    """
    if batch_size is None:
        batch_size = training_config.batch_size

    loaders = {}

    for split_name, (data, labels) in splits.items():
        dataset = EEGDataset(data, labels)
        shuffle = (split_name == "train")

        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

    return loaders


def preprocess_all_subjects(
    subjects: Optional[List[int]] = None,
    use_moabb: bool = True,
    verbose: bool = True
) -> Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Preprocess all subjects.

    Args:
        subjects: List of subject IDs. If None, process all 9 subjects.
        use_moabb: Whether to use MOABB.
        verbose: Whether to print progress.

    Returns:
        Dictionary mapping subject_id -> splits.
    """
    if subjects is None:
        subjects = data_config.subjects

    all_data = {}

    for subject_id in subjects:
        all_data[subject_id] = preprocess_subject(
            subject_id=subject_id,
            use_moabb=use_moabb,
            save=True,
            verbose=verbose
        )

    return all_data


def prepare_loso_data(
    test_subject: int,
    subjects: Optional[List[int]] = None,
    use_moabb: bool = True,
    verbose: bool = True
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """
    Prepare data for Leave-One-Subject-Out cross-validation.

    Args:
        test_subject: Subject ID to hold out for testing.
        subjects: List of all subject IDs.
        use_moabb: Whether to use MOABB.
        verbose: Whether to print progress.

    Returns:
        Tuple of (train_splits, test_data) where:
            - train_splits: Combined data from other subjects
            - test_data: (X_test, y_test) from the held-out subject
    """
    if subjects is None:
        subjects = data_config.subjects

    train_subjects = [s for s in subjects if s != test_subject]

    if verbose:
        print(f"\nLOSO: Test subject = {test_subject}")
        print(f"      Train subjects = {train_subjects}")

    # Collect training data from all other subjects
    X_train_all = []
    y_train_all = []

    for subject_id in train_subjects:
        splits = get_subject_data(subject_id, use_moabb=use_moabb, verbose=False)
        # Use all data from training subjects (train + val + test)
        for split_name in ["train", "val", "test"]:
            X_train_all.append(splits[split_name][0])
            y_train_all.append(splits[split_name][1])

    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)

    # Split combined training data into train/val
    splits = split_data(X_train, y_train, train_ratio=0.85, val_ratio=0.15)

    # Get test data from held-out subject
    test_splits = get_subject_data(test_subject, use_moabb=use_moabb, verbose=False)
    # Use all data from test subject
    X_test = np.concatenate([
        test_splits["train"][0],
        test_splits["val"][0],
        test_splits["test"][0]
    ], axis=0)
    y_test = np.concatenate([
        test_splits["train"][1],
        test_splits["val"][1],
        test_splits["test"][1]
    ], axis=0)

    if verbose:
        print(f"      Train size: {splits['train'][0].shape[0]}")
        print(f"      Val size: {splits['val'][0].shape[0]}")
        print(f"      Test size: {X_test.shape[0]}")

    return splits, (X_test, y_test)


if __name__ == "__main__":
    # Test preprocessing pipeline
    set_seed()

    print("Testing preprocessing pipeline...")
    print("\n" + "=" * 60)

    # Test single subject preprocessing
    splits = preprocess_subject(subject_id=1, use_moabb=True, save=True, verbose=True)

    print("\n" + "=" * 60)
    print("Creating DataLoaders...")
    print("=" * 60)

    loaders = create_dataloaders(splits)
    for name, loader in loaders.items():
        batch = next(iter(loader))
        print(f"{name}: batch shape = {batch[0].shape}, labels shape = {batch[1].shape}")

    print("\n" + "=" * 60)
    print("Testing LOSO preparation...")
    print("=" * 60)

    train_splits, (X_test, y_test) = prepare_loso_data(test_subject=1, verbose=True)
