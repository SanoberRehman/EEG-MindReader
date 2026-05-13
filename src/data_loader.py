"""
Data loading module for EEG Motor Imagery Classification.

Handles loading the BCI Competition IV Dataset 2a via MOABB,
with fallback to synthetic data generation for testing.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import mne
from mne import Epochs
from mne.io import BaseRaw

from .config import (
    data_config,
    preprocess_config,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    set_seed,
)


# Suppress MNE verbose output
mne.set_log_level("WARNING")


def load_moabb_dataset(
    subjects: Optional[List[int]] = None,
    verbose: bool = True
) -> Dict[int, Dict[str, Union[BaseRaw, np.ndarray]]]:
    """
    Load the BCI Competition IV Dataset 2a via MOABB.

    Args:
        subjects: List of subject IDs to load (1-9). If None, loads all subjects.
        verbose: Whether to print loading progress.

    Returns:
        Dictionary mapping subject_id -> {"raw": MNE Raw, "events": events array, "event_id": dict}
    """
    try:
        from moabb.datasets import BNCI2014001

        if verbose:
            print("Loading BCI Competition IV Dataset 2a via MOABB...")

        dataset = BNCI2014001()

        if subjects is None:
            subjects = data_config.subjects

        data = {}
        for subject_id in subjects:
            if verbose:
                print(f"  Loading Subject {subject_id}...")

            # MOABB returns sessions as a dict
            subject_data = dataset.get_data(subjects=[subject_id])

            # Get the raw data from the first session
            # Structure: {subject_id: {"0train": {"0": raw, "1": raw, ...}, "1test": {...}}}
            sessions = subject_data[subject_id]

            # Combine training sessions
            raws = []
            for session_name, runs in sessions.items():
                for run_name, raw in runs.items():
                    raws.append(raw)

            # Concatenate all runs
            if len(raws) > 1:
                raw_combined = mne.concatenate_raws(raws)
            else:
                raw_combined = raws[0]

            # Get events
            events, event_id = mne.events_from_annotations(raw_combined)

            data[subject_id] = {
                "raw": raw_combined,
                "events": events,
                "event_id": event_id
            }

        if verbose:
            print(f"Successfully loaded {len(data)} subjects.")

        return data

    except Exception as e:
        warnings.warn(f"MOABB loading failed: {e}. Falling back to synthetic data.")
        return generate_synthetic_data(subjects, verbose)


def generate_synthetic_data(
    subjects: Optional[List[int]] = None,
    verbose: bool = True
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Generate synthetic EEG-like data for testing the pipeline.

    Creates realistic-looking motor imagery data with:
    - Mu rhythm (8-12 Hz) modulation for motor imagery
    - Beta rhythm (18-26 Hz) components
    - Pink noise background

    Args:
        subjects: List of subject IDs to generate.
        verbose: Whether to print progress.

    Returns:
        Dictionary with synthetic data in the same format as MOABB loader.
    """
    set_seed()

    if subjects is None:
        subjects = data_config.subjects

    if verbose:
        print("Generating synthetic EEG data for pipeline testing...")

    n_channels = data_config.n_channels
    sfreq = data_config.sampling_rate
    n_classes = data_config.n_classes

    # Number of trials per class per subject
    n_trials_per_class = 72  # Similar to real dataset

    data = {}

    for subject_id in subjects:
        if verbose:
            print(f"  Generating Subject {subject_id}...")

        # Add subject-specific variation
        np.random.seed(data_config.subjects.index(subject_id) + 42)

        trials = []
        labels = []

        for class_idx in range(n_classes):
            for _ in range(n_trials_per_class):
                trial = _generate_single_trial(
                    n_channels=n_channels,
                    sfreq=sfreq,
                    duration=6.0,  # Full trial duration
                    class_idx=class_idx,
                    subject_variation=subject_id * 0.1
                )
                trials.append(trial)
                labels.append(class_idx)

        # Shuffle trials
        indices = np.random.permutation(len(trials))
        trials = [trials[i] for i in indices]
        labels = [labels[i] for i in indices]

        # Create MNE Raw object
        trial_data = np.concatenate(trials, axis=1)

        # Create info object
        ch_names = data_config.channel_names
        ch_types = ["eeg"] * n_channels
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        # Set montage for topographic plots
        montage = mne.channels.make_standard_montage("standard_1020")
        info.set_montage(montage, on_missing="ignore")

        raw = mne.io.RawArray(trial_data, info, verbose=False)

        # Create events array
        trial_length_samples = int(6.0 * sfreq)
        events = []
        for i, label in enumerate(labels):
            onset = i * trial_length_samples + int(2.0 * sfreq)  # Cue at 2s into trial
            events.append([onset, 0, label + 1])  # +1 because MNE events are 1-indexed

        events = np.array(events)

        event_id = {
            "left_hand": 1,
            "right_hand": 2,
            "feet": 3,
            "tongue": 4
        }

        data[subject_id] = {
            "raw": raw,
            "events": events,
            "event_id": event_id
        }

    if verbose:
        print(f"Generated synthetic data for {len(data)} subjects.")

    return data


def _generate_single_trial(
    n_channels: int,
    sfreq: int,
    duration: float,
    class_idx: int,
    subject_variation: float = 0.0
) -> np.ndarray:
    """
    Generate a single synthetic EEG trial with class-specific patterns.

    Args:
        n_channels: Number of EEG channels.
        sfreq: Sampling frequency in Hz.
        duration: Trial duration in seconds.
        class_idx: Class index (0-3) for motor imagery type.
        subject_variation: Subject-specific variation factor.

    Returns:
        Array of shape (n_channels, n_samples).
    """
    n_samples = int(duration * sfreq)
    t = np.linspace(0, duration, n_samples)

    # Base signal: pink noise
    trial = _generate_pink_noise(n_channels, n_samples) * 20e-6  # ~20 µV

    # Add alpha rhythm (8-12 Hz) - posterior channels
    alpha_freq = 10 + subject_variation
    alpha_amp = 15e-6 * (1 + 0.2 * np.random.randn())
    posterior_channels = [18, 19, 20, 21]  # P1, Pz, P2, POz
    for ch in posterior_channels:
        if ch < n_channels:
            trial[ch] += alpha_amp * np.sin(2 * np.pi * alpha_freq * t)

    # Motor imagery specific patterns (after cue at t=2s)
    cue_onset = int(2.0 * sfreq)
    mi_period = slice(cue_onset, cue_onset + int(2.5 * sfreq))

    # Mu rhythm (10 Hz) - motor cortex
    mu_freq = 10 + 0.5 * subject_variation
    mu_amp = 10e-6

    # Beta rhythm (20 Hz) - motor cortex
    beta_freq = 20 + subject_variation
    beta_amp = 5e-6

    # Channel indices for motor areas (C3, C4, Cz region)
    left_motor = [7, 8]    # C3, C1
    right_motor = [10, 11]  # C2, C4
    central = [6, 9, 12]   # C5, Cz, C6
    foot_area = [9, 15]    # Cz, CPz

    if class_idx == 0:  # Left hand - ERD in right motor cortex (C4)
        for ch in right_motor:
            # Event-related desynchronization (power decrease)
            trial[ch, mi_period] *= 0.5
            trial[ch, mi_period] += mu_amp * 0.3 * np.sin(2 * np.pi * mu_freq * t[mi_period])

    elif class_idx == 1:  # Right hand - ERD in left motor cortex (C3)
        for ch in left_motor:
            trial[ch, mi_period] *= 0.5
            trial[ch, mi_period] += mu_amp * 0.3 * np.sin(2 * np.pi * mu_freq * t[mi_period])

    elif class_idx == 2:  # Feet - ERD in central motor cortex (Cz)
        for ch in foot_area:
            trial[ch, mi_period] *= 0.4
            trial[ch, mi_period] += beta_amp * 0.5 * np.sin(2 * np.pi * beta_freq * t[mi_period])

    elif class_idx == 3:  # Tongue - distributed pattern
        for ch in central:
            trial[ch, mi_period] *= 0.6
            trial[ch, mi_period] += mu_amp * 0.2 * np.sin(2 * np.pi * (mu_freq + 2) * t[mi_period])

    # Add some common noise
    trial += np.random.randn(n_channels, n_samples) * 2e-6

    return trial


def _generate_pink_noise(n_channels: int, n_samples: int) -> np.ndarray:
    """Generate pink (1/f) noise."""
    white = np.random.randn(n_channels, n_samples)
    fft = np.fft.rfft(white, axis=1)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1  # Avoid division by zero
    pink_filter = 1 / np.sqrt(freqs)
    pink_fft = fft * pink_filter
    pink = np.fft.irfft(pink_fft, n=n_samples, axis=1)
    return pink


def load_subject_data(
    subject_id: int,
    use_moabb: bool = True,
    verbose: bool = True
) -> Tuple[BaseRaw, np.ndarray, Dict[str, int]]:
    """
    Load data for a single subject.

    Args:
        subject_id: Subject ID (1-9).
        use_moabb: Whether to try MOABB first (falls back to synthetic if fails).
        verbose: Whether to print progress.

    Returns:
        Tuple of (raw, events, event_id).
    """
    if use_moabb:
        data = load_moabb_dataset(subjects=[subject_id], verbose=verbose)
    else:
        data = generate_synthetic_data(subjects=[subject_id], verbose=verbose)

    subject_data = data[subject_id]
    return subject_data["raw"], subject_data["events"], subject_data["event_id"]


def get_epochs_from_raw(
    raw: BaseRaw,
    events: np.ndarray,
    event_id: Dict[str, int],
    tmin: float = None,
    tmax: float = None,
    baseline: Optional[Tuple[float, float]] = None,
    verbose: bool = False
) -> Epochs:
    """
    Create MNE Epochs from raw data.

    Args:
        raw: MNE Raw object.
        events: Events array.
        event_id: Event ID dictionary.
        tmin: Start time relative to event (default from config).
        tmax: End time relative to event (default from config).
        baseline: Baseline correction window.
        verbose: Whether to print MNE output.

    Returns:
        MNE Epochs object.
    """
    if tmin is None:
        tmin = preprocess_config.tmin
    if tmax is None:
        tmax = preprocess_config.tmax
    if baseline is None:
        baseline = preprocess_config.baseline

    epochs = Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        preload=True,
        verbose=verbose
    )

    return epochs


def get_data_info() -> Dict:
    """
    Get information about the dataset.

    Returns:
        Dictionary with dataset information.
    """
    return {
        "name": data_config.dataset_name,
        "n_subjects": data_config.n_subjects,
        "n_channels": data_config.n_channels,
        "n_classes": data_config.n_classes,
        "sampling_rate": data_config.sampling_rate,
        "class_names": data_config.class_names,
        "channel_names": data_config.channel_names,
    }


if __name__ == "__main__":
    # Test data loading
    print("Testing data loader...")
    print("\n" + "=" * 50)
    print("Dataset Info:")
    print("=" * 50)
    for key, value in get_data_info().items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 50)
    print("Loading Subject 1...")
    print("=" * 50)
    raw, events, event_id = load_subject_data(subject_id=1, verbose=True)
    print(f"\nRaw data shape: {raw.get_data().shape}")
    print(f"Number of events: {len(events)}")
    print(f"Event IDs: {event_id}")

    print("\n" + "=" * 50)
    print("Creating epochs...")
    print("=" * 50)
    epochs = get_epochs_from_raw(raw, events, event_id)
    print(f"Epochs shape: {epochs.get_data().shape}")
    print(f"Time points: {epochs.times.shape[0]}")
