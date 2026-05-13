"""
Visualization module for EEG Motor Imagery Classification.

Provides publication-quality plots for:
- EEG signals and topographic maps
- Training curves
- Confusion matrices
- t-SNE feature embeddings
- Model comparison charts
- Attention/saliency maps
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from .config import (
    data_config,
    viz_config,
    VizConfig,
    FIGURES_DIR,
    get_figure_path,
)

# Try to import MNE for topographic plots
try:
    import mne
    from mne.viz import plot_topomap
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

# Try to import sklearn for t-SNE
try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

# Try to import plotly for interactive plots
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def setup_style() -> None:
    """Set up matplotlib style for publication-quality plots."""
    try:
        plt.style.use(viz_config.style)
    except (OSError, ValueError):
        try:
            plt.style.use("seaborn-v0_8-whitegrid")
        except (OSError, ValueError):
            plt.style.use("ggplot")  # Fallback that always exists

    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "figure.dpi": 100,
        "savefig.dpi": viz_config.figure_dpi,
        "savefig.bbox": "tight",
    })


def save_figure(fig: plt.Figure, name: str, close: bool = True) -> Path:
    """
    Save figure to the figures directory.

    Args:
        fig: Matplotlib figure.
        name: Figure name (without extension).
        close: Whether to close the figure after saving.

    Returns:
        Path to saved figure.
    """
    path = get_figure_path(name)
    fig.savefig(path, dpi=viz_config.figure_dpi, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)
    return path


# =============================================================================
# EEG SIGNAL PLOTS
# =============================================================================

def plot_raw_eeg(
    data: np.ndarray,
    sfreq: float = 250,
    channels: Optional[List[int]] = None,
    channel_names: Optional[List[str]] = None,
    title: str = "Raw EEG Signal",
    figsize: Tuple[int, int] = (14, 8),
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Plot raw EEG signals from multiple channels.

    Args:
        data: EEG data of shape (n_channels, n_timepoints).
        sfreq: Sampling frequency in Hz.
        channels: List of channel indices to plot. If None, plots first 8.
        channel_names: Names for each channel.
        title: Plot title.
        figsize: Figure size.
        save_name: If provided, save figure with this name.

    Returns:
        Matplotlib figure.
    """
    setup_style()

    if channels is None:
        channels = list(range(min(8, data.shape[0])))

    if channel_names is None:
        channel_names = data_config.channel_names

    n_channels = len(channels)
    n_timepoints = data.shape[1]
    time = np.arange(n_timepoints) / sfreq

    fig, axes = plt.subplots(n_channels, 1, figsize=figsize, sharex=True)
    if n_channels == 1:
        axes = [axes]

    # Normalize for display
    scale = np.percentile(np.abs(data[channels]), 95) * 2

    for i, ch_idx in enumerate(channels):
        ax = axes[i]
        signal = data[ch_idx]
        ax.plot(time, signal, "b-", linewidth=0.5, alpha=0.8)
        ax.set_ylabel(channel_names[ch_idx] if ch_idx < len(channel_names) else f"Ch{ch_idx}")
        ax.set_ylim(-scale, scale)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, close=False)

    return fig


def plot_single_trial(
    data: np.ndarray,
    label: int,
    sfreq: float = 250,
    channel_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Plot a single EEG trial with all channels as a heatmap.

    Args:
        data: Trial data of shape (n_channels, n_timepoints).
        label: Class label (0-3).
        sfreq: Sampling frequency.
        channel_names: Channel names.
        figsize: Figure size.
        save_name: If provided, save figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()

    if channel_names is None:
        channel_names = data_config.channel_names

    n_channels, n_timepoints = data.shape
    time = np.arange(n_timepoints) / sfreq

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    im = ax.imshow(
        data,
        aspect="auto",
        cmap="RdBu_r",
        extent=[time[0], time[-1], n_channels - 0.5, -0.5],
        vmin=-np.percentile(np.abs(data), 95),
        vmax=np.percentile(np.abs(data), 95)
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_yticks(range(n_channels))
    ax.set_yticklabels(channel_names[:n_channels], fontsize=8)

    class_name = data_config.class_names[label]
    ax.set_title(f"EEG Trial - Class: {class_name}", fontsize=14)

    cbar = plt.colorbar(im, ax=ax, label="Amplitude (normalized)")

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, close=False)

    return fig


def plot_class_average_erp(
    epochs_data: np.ndarray,
    labels: np.ndarray,
    sfreq: float = 250,
    channel_idx: int = 9,  # Cz
    figsize: Tuple[int, int] = (10, 6),
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Plot average ERP for each class.

    Args:
        epochs_data: Epoch data of shape (n_trials, n_channels, n_timepoints).
        labels: Labels of shape (n_trials,).
        sfreq: Sampling frequency.
        channel_idx: Channel index to plot.
        figsize: Figure size.
        save_name: If provided, save figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()

    n_classes = len(data_config.class_names)
    n_timepoints = epochs_data.shape[2]
    time = np.arange(n_timepoints) / sfreq

    fig, ax = plt.subplots(figsize=figsize)

    for class_idx in range(n_classes):
        mask = labels == class_idx
        class_data = epochs_data[mask, channel_idx, :]
        mean = class_data.mean(axis=0)
        std = class_data.std(axis=0)

        color = viz_config.class_colors[class_idx]
        ax.plot(time, mean, color=color, label=data_config.class_names[class_idx], linewidth=2)
        ax.fill_between(time, mean - std, mean + std, color=color, alpha=0.2)

    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5, label="Cue onset")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (normalized)")
    ax.set_title(f"Class-Average ERP at {data_config.channel_names[channel_idx]}")
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, close=False)

    return fig


# =============================================================================
# TOPOGRAPHIC MAPS
# =============================================================================

def plot_topomap_custom(
    values: np.ndarray,
    title: str = "Topographic Map",
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = (6, 5),
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Plot a topographic map of scalp values.

    Args:
        values: Array of shape (n_channels,) with values for each electrode.
        title: Plot title.
        cmap: Colormap name.
        figsize: Figure size.
        save_name: If provided, save figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()

    if not HAS_MNE:
        return _plot_topomap_fallback(values, title, figsize, save_name)

    # Create MNE info object
    ch_names = data_config.channel_names[:len(values)]
    info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types="eeg")

    # Set montage
    montage = mne.channels.make_standard_montage("standard_1020")
    info.set_montage(montage, on_missing="ignore")

    fig, ax = plt.subplots(figsize=figsize)

    # Get positions
    pos = np.array([info.get_montage().get_positions()["ch_pos"][ch][:2]
                    for ch in ch_names if ch in info.get_montage().get_positions()["ch_pos"]])

    if len(pos) == len(values):
        im, _ = plot_topomap(
            values, pos, axes=ax, cmap=cmap, show=False,
            contours=6, sensors=True
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        return _plot_topomap_fallback(values, title, figsize, save_name)

    ax.set_title(title, fontsize=14)

    if save_name:
        save_figure(fig, save_name, close=False)

    return fig


def _plot_topomap_fallback(
    values: np.ndarray,
    title: str,
    figsize: Tuple[int, int],
    save_name: Optional[str] = None
) -> plt.Figure:
    """Fallback topomap using simple scatter plot."""
    setup_style()

    # Approximate 10-20 positions (normalized)
    positions_2d = {
        "Fz": (0.5, 0.85), "FC3": (0.3, 0.7), "FC1": (0.4, 0.7),
        "FCz": (0.5, 0.7), "FC2": (0.6, 0.7), "FC4": (0.7, 0.7),
        "C5": (0.15, 0.5), "C3": (0.3, 0.5), "C1": (0.4, 0.5),
        "Cz": (0.5, 0.5), "C2": (0.6, 0.5), "C4": (0.7, 0.5), "C6": (0.85, 0.5),
        "CP3": (0.3, 0.3), "CP1": (0.4, 0.3), "CPz": (0.5, 0.3),
        "CP2": (0.6, 0.3), "CP4": (0.7, 0.3),
        "P1": (0.4, 0.15), "Pz": (0.5, 0.15), "P2": (0.6, 0.15), "POz": (0.5, 0.05)
    }

    fig, ax = plt.subplots(figsize=figsize)

    # Draw head outline
    circle = plt.Circle((0.5, 0.45), 0.45, fill=False, linewidth=2, color="black")
    ax.add_patch(circle)

    # Draw nose
    ax.plot([0.5, 0.5], [0.9, 0.95], "k-", linewidth=2)

    # Plot electrodes
    ch_names = data_config.channel_names[:len(values)]
    xs, ys, vals = [], [], []

    for i, ch in enumerate(ch_names):
        if ch in positions_2d:
            x, y = positions_2d[ch]
            xs.append(x)
            ys.append(y)
            vals.append(values[i])

    scatter = ax.scatter(xs, ys, c=vals, cmap="RdBu_r", s=300, edgecolors="black", linewidth=1)
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)

    # Add channel labels
    for i, ch in enumerate(ch_names):
        if ch in positions_2d:
            x, y = positions_2d[ch]
            ax.annotate(ch, (x, y), ha="center", va="center", fontsize=7)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=14)

    if save_name:
        save_figure(fig, save_name, close=False)

    return fig


def plot_channel_importance(
    importance: np.ndarray,
    title: str = "Channel Importance",
    figsize: Tuple[int, int] = (10, 5),
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Plot channel importance as both bar chart and topomap.

    Args:
        importance: Importance values of shape (n_channels,).
        title: Plot title.
        figsize: Figure size.
        save_name: If provided, save figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ch_names = data_config.channel_names[:len(importance)]

    # Bar chart
    ax = axes[0]
    colors = plt.cm.RdYlBu_r(importance / importance.max())
    bars = ax.barh(range(len(importance)), importance, color=colors)
    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels(ch_names, fontsize=8)
    ax.set_xlabel("Importance")
    ax.set_title("Channel Importance")
    ax.invert_yaxis()

    # Topomap
    ax = axes[1]
    # Use the fallback method embedded here
    positions_2d = {
        "Fz": (0.5, 0.85), "FC3": (0.3, 0.7), "FC1": (0.4, 0.7),
        "FCz": (0.5, 0.7), "FC2": (0.6, 0.7), "FC4": (0.7, 0.7),
        "C5": (0.15, 0.5), "C3": (0.3, 0.5), "C1": (0.4, 0.5),
        "Cz": (0.5, 0.5), "C2": (0.6, 0.5), "C4": (0.7, 0.5), "C6": (0.85, 0.5),
        "CP3": (0.3, 0.3), "CP1": (0.4, 0.3), "CPz": (0.5, 0.3),
        "CP2": (0.6, 0.3), "CP4": (0.7, 0.3),
        "P1": (0.4, 0.15), "Pz": (0.5, 0.15), "P2": (0.6, 0.15), "POz": (0.5, 0.05)
    }

    circle = plt.Circle((0.5, 0.45), 0.45, fill=False, linewidth=2, color="black")
    ax.add_patch(circle)
    ax.plot([0.5, 0.5], [0.9, 0.95], "k-", linewidth=2)

    xs, ys, vals = [], [], []
    for i, ch in enumerate(ch_names):
        if ch in positions_2d:
            xs.append(positions_2d[ch][0])
            ys.append(positions_2d[ch][1])
            vals.append(importance[i])

    scatter = ax.scatter(xs, ys, c=vals, cmap="hot", s=400, edgecolors="black")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Scalp Map")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, close=False)

    return fig


# =============================================================================
# TRAINING CURVES
# =============================================================================

def plot_training_history(
    history: Dict,
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 4),
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Plot training and validation curves.

    Args:
        history: Dictionary with train_loss, val_loss, train_acc, val_acc.
        title: Plot title.
        figsize: Figure size.
        save_name: If provided, save figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    ax.plot(epochs, history["val_loss"], "r-", label="Validation", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy plot
    ax = axes[1]
    ax.plot(epochs, history["train_acc"], "b-", label="Train", linewidth=2)
    ax.plot(epochs, history["val_acc"], "r-", label="Validation", linewidth=2)

    # Mark best epoch
    if "best_epoch" in history:
        best_epoch = history["best_epoch"] + 1
        best_acc = history["val_acc"][history["best_epoch"]]
        ax.axvline(x=best_epoch, color="green", linestyle="--", alpha=0.7)
        ax.scatter([best_epoch], [best_acc], color="green", s=100, zorder=5, label=f"Best ({best_acc:.3f})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, close=False)

    return fig


def plot_learning_rate(
    learning_rates: List[float],
    figsize: Tuple[int, int] = (8, 4),
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Plot learning rate schedule.

    Args:
        learning_rates: List of learning rates per epoch.
        figsize: Figure size.
        save_name: If provided, save figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(learning_rates) + 1)
    ax.plot(epochs, learning_rates, "b-", linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, close=False)

    return fig


# =============================================================================
# CONFUSION MATRICES
# =============================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    title: str = "Confusion Matrix",
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (8, 6),
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Plot a confusion matrix.

    Args:
        cm: Confusion matrix of shape (n_classes, n_classes).
        class_names: Names of classes.
        normalize: Whether to normalize by row (true class).
        title: Plot title.
        cmap: Colormap.
        figsize: Figure size.
        save_name: If provided, save figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()

    if class_names is None:
        class_names = data_config.class_names

    if normalize:
        cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=title
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm_display.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            value = cm_display[i, j]
            color = "white" if value > thresh else "black"
            text = f"{value:{fmt}}"
            if normalize:
                text += f"\n({cm[i, j]})"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10)

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, close=False)

    return fig


def plot_multi_confusion_matrices(
    cms: Dict[str, np.ndarray],
    class_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 5),
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Plot multiple confusion matrices side by side.

    Args:
        cms: Dictionary mapping model_name -> confusion_matrix.
        class_names: Names of classes.
        figsize: Figure size.
        save_name: If provided, save figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()

    if class_names is None:
        class_names = data_config.class_names

    n_models = len(cms)
    fig, axes = plt.subplots(1, n_models, figsize=figsize)

    if n_models == 1:
        axes = [axes]

    for ax, (model_name, cm) in zip(axes, cms.items()):
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(model_name)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        thresh = cm_norm.max() / 2.0
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm_norm[i, j] > thresh else "black", fontsize=9)

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, close=False)

    return fig


# =============================================================================
# t-SNE VISUALIZATION
# =============================================================================

def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    perplexity: int = None,
    title: str = "t-SNE Feature Visualization",
    figsize: Tuple[int, int] = (10, 8),
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Plot t-SNE visualization of features.

    Args:
        features: Feature array of shape (n_samples, n_features).
        labels: Labels of shape (n_samples,).
        class_names: Names of classes.
        perplexity: t-SNE perplexity parameter.
        title: Plot title.
        figsize: Figure size.
        save_name: If provided, save figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()

    if not HAS_TSNE:
        print("Warning: sklearn not available for t-SNE")
        return None

    if class_names is None:
        class_names = data_config.class_names
    if perplexity is None:
        perplexity = viz_config.tsne_perplexity

    # Compute t-SNE
    print(f"Computing t-SNE with perplexity={perplexity}...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=viz_config.tsne_n_iter,
        random_state=42,
        init="pca"
    )
    features_2d = tsne.fit_transform(features)

    fig, ax = plt.subplots(figsize=figsize)

    for class_idx in range(len(class_names)):
        mask = labels == class_idx
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=viz_config.class_colors[class_idx],
            label=class_names[class_idx],
            alpha=0.7,
            s=50,
            edgecolors="white",
            linewidth=0.5
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.legend(loc="best")

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, close=False)

    return fig


# =============================================================================
# MODEL COMPARISON
# =============================================================================

def plot_model_comparison(
    results: Dict[str, Dict],
    metric: str = "accuracy",
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (10, 6),
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Plot bar chart comparing models.

    Args:
        results: Dictionary mapping model_name -> {metric_name: (mean, std)}.
        metric: Metric to plot.
        title: Plot title.
        figsize: Figure size.
        save_name: If provided, save figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()

    fig, ax = plt.subplots(figsize=figsize)

    model_names = list(results.keys())
    means = [results[m][metric][0] for m in model_names]
    stds = [results[m][metric][1] for m in model_names]

    colors = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    bars = ax.bar(model_names, means, yerr=stds, capsize=5, color=colors, edgecolor="black")

    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_ylim(0, 1)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.02,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, close=False)

    return fig


def plot_subject_accuracy_heatmap(
    results: Dict[str, Dict[int, float]],
    title: str = "Per-Subject Accuracy",
    figsize: Tuple[int, int] = (10, 6),
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Plot heatmap of accuracy per subject per model.

    Args:
        results: Dictionary mapping model_name -> {subject_id: accuracy}.
        title: Plot title.
        figsize: Figure size.
        save_name: If provided, save figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()

    model_names = list(results.keys())
    subjects = sorted(list(results[model_names[0]].keys()))

    # Build matrix
    matrix = np.array([
        [results[model][subj] for subj in subjects]
        for model in model_names
    ])

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(subjects)))
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_xticklabels([f"S{s}" for s in subjects])
    ax.set_yticklabels(model_names)
    ax.set_xlabel("Subject")
    ax.set_ylabel("Model")
    ax.set_title(title)

    # Add text annotations
    for i in range(len(model_names)):
        for j in range(len(subjects)):
            value = matrix[i, j]
            color = "white" if value < 0.5 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=color, fontsize=10)

    plt.colorbar(im, ax=ax, label="Accuracy")
    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, close=False)

    return fig


def plot_within_vs_loso(
    within_results: Dict[str, float],
    loso_results: Dict[str, float],
    figsize: Tuple[int, int] = (10, 6),
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Plot comparison of within-subject vs LOSO accuracy.

    Args:
        within_results: Dictionary mapping model_name -> within-subject accuracy.
        loso_results: Dictionary mapping model_name -> LOSO accuracy.
        figsize: Figure size.
        save_name: If provided, save figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()

    model_names = list(within_results.keys())
    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    within_vals = [within_results[m] for m in model_names]
    loso_vals = [loso_results[m] for m in model_names]

    bars1 = ax.bar(x - width/2, within_vals, width, label="Within-Subject", color="#4ECDC4", edgecolor="black")
    bars2 = ax.bar(x + width/2, loso_vals, width, label="LOSO (Cross-Subject)", color="#FF6B6B", edgecolor="black")

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title("Within-Subject vs Cross-Subject Generalization")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f"{height:.2f}", ha="center", va="bottom", fontsize=10)

    # Add drop annotation
    for i, model in enumerate(model_names):
        drop = within_vals[i] - loso_vals[i]
        if drop > 0:
            ax.annotate(
                f"-{drop:.0%}",
                xy=(x[i], min(within_vals[i], loso_vals[i]) - 0.05),
                ha="center",
                fontsize=9,
                color="red"
            )

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, close=False)

    return fig


# =============================================================================
# CLASS DISTRIBUTION
# =============================================================================

def plot_class_distribution(
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Class Distribution",
    figsize: Tuple[int, int] = (8, 5),
    save_name: Optional[str] = None
) -> plt.Figure:
    """
    Plot class distribution.

    Args:
        labels: Labels array.
        class_names: Names of classes.
        title: Plot title.
        figsize: Figure size.
        save_name: If provided, save figure.

    Returns:
        Matplotlib figure.
    """
    setup_style()

    if class_names is None:
        class_names = data_config.class_names

    unique, counts = np.unique(labels, return_counts=True)

    fig, ax = plt.subplots(figsize=figsize)

    colors = [viz_config.class_colors[i] for i in unique]
    bars = ax.bar([class_names[i] for i in unique], counts, color=colors, edgecolor="black")

    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Trials")
    ax.set_title(title)

    # Add count labels
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha="center", va="bottom")

    plt.tight_layout()

    if save_name:
        save_figure(fig, save_name, close=False)

    return fig


if __name__ == "__main__":
    # Test visualization module
    print("Testing visualization module...")
    print("=" * 50)

    setup_style()

    # Test with dummy data
    np.random.seed(42)

    # Test class distribution
    labels = np.random.randint(0, 4, 200)
    fig = plot_class_distribution(labels, title="Test Class Distribution")
    print("Class distribution plot: OK")
    plt.close(fig)

    # Test confusion matrix
    cm = np.random.randint(10, 50, (4, 4))
    np.fill_diagonal(cm, np.random.randint(50, 100, 4))
    fig = plot_confusion_matrix(cm, title="Test Confusion Matrix")
    print("Confusion matrix plot: OK")
    plt.close(fig)

    # Test training history
    history = {
        "train_loss": [1.0 - i*0.05 for i in range(20)],
        "val_loss": [1.1 - i*0.04 for i in range(20)],
        "train_acc": [0.3 + i*0.03 for i in range(20)],
        "val_acc": [0.25 + i*0.025 for i in range(20)],
        "best_epoch": 15
    }
    fig = plot_training_history(history, title="Test Training History")
    print("Training history plot: OK")
    plt.close(fig)

    # Test t-SNE
    if HAS_TSNE:
        features = np.random.randn(100, 64)
        labels = np.random.randint(0, 4, 100)
        fig = plot_tsne(features, labels, title="Test t-SNE", perplexity=10)
        print("t-SNE plot: OK")
        plt.close(fig)

    # Test topomap
    values = np.random.randn(22)
    fig = plot_topomap_custom(values, title="Test Topomap")
    print("Topomap plot: OK")
    plt.close(fig)

    print("\nAll visualization tests passed!")
