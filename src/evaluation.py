"""
Evaluation module for EEG Motor Imagery Classification.

Provides metrics computation, confusion matrices, and cross-subject evaluation.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
)

from .config import (
    data_config,
    eval_config,
    training_config,
    EvalConfig,
    DEVICE,
)
from .preprocessing import (
    get_subject_data,
    create_dataloaders,
    prepare_loso_data,
    EEGDataset,
)
from .training import train_model, get_predictions, TrainingHistory


@dataclass
class EvaluationResults:
    """Container for evaluation metrics."""

    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    cohen_kappa: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    per_class_accuracy: Optional[np.ndarray] = None
    classification_report: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "cohen_kappa": self.cohen_kappa,
            "per_class_accuracy": self.per_class_accuracy.tolist() if self.per_class_accuracy is not None else None,
        }

    def __repr__(self) -> str:
        return (
            f"EvaluationResults(\n"
            f"  accuracy={self.accuracy:.4f},\n"
            f"  balanced_accuracy={self.balanced_accuracy:.4f},\n"
            f"  f1_macro={self.f1_macro:.4f},\n"
            f"  cohen_kappa={self.cohen_kappa:.4f}\n"
            f")"
        )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> EvaluationResults:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: Names of classes for report.

    Returns:
        EvaluationResults object with all metrics.
    """
    if class_names is None:
        class_names = data_config.class_names

    results = EvaluationResults()

    # Basic metrics
    results.accuracy = accuracy_score(y_true, y_pred)
    results.balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    results.f1_macro = f1_score(y_true, y_pred, average="macro")
    results.f1_weighted = f1_score(y_true, y_pred, average="weighted")
    results.cohen_kappa = cohen_kappa_score(y_true, y_pred)

    # Confusion matrix
    results.confusion_matrix = confusion_matrix(y_true, y_pred)

    # Per-class accuracy
    cm = results.confusion_matrix
    results.per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

    # Classification report
    results.classification_report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4
    )

    return results


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: Optional[torch.device] = None,
    class_names: Optional[List[str]] = None
) -> EvaluationResults:
    """
    Evaluate a model on test data.

    Args:
        model: Trained model.
        test_loader: Test data loader.
        device: Device to run on.
        class_names: Names of classes.

    Returns:
        EvaluationResults object.
    """
    if device is None:
        device = DEVICE

    # Get predictions
    y_pred, y_probs, y_true = get_predictions(model, test_loader, device)

    # Compute metrics
    results = compute_metrics(y_true, y_pred, class_names)

    return results


def evaluate_within_subject(
    model_class: type,
    model_kwargs: Dict,
    subject_id: int,
    use_moabb: bool = True,
    verbose: bool = True
) -> Tuple[nn.Module, EvaluationResults, TrainingHistory]:
    """
    Train and evaluate a model on a single subject (within-subject classification).

    Args:
        model_class: Model class to instantiate.
        model_kwargs: Keyword arguments for model.
        subject_id: Subject ID to train/test on.
        use_moabb: Whether to use MOABB for data loading.
        verbose: Whether to print progress.

    Returns:
        Tuple of (trained_model, evaluation_results, training_history).
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Within-Subject Evaluation: Subject {subject_id}")
        print(f"{'='*60}")

    # Get data
    splits = get_subject_data(subject_id, use_moabb=use_moabb, verbose=verbose)
    loaders = create_dataloaders(splits)

    # Create model
    model = model_class(**model_kwargs)

    # Train
    model, history = train_model(
        model,
        loaders["train"],
        loaders["val"],
        model_name=model_class.__name__.lower(),
        subject_id=subject_id,
        verbose=verbose
    )

    # Evaluate on test set
    results = evaluate_model(model, loaders["test"])

    if verbose:
        print(f"\nTest Results:")
        print(f"  Accuracy: {results.accuracy:.4f}")
        print(f"  Balanced Accuracy: {results.balanced_accuracy:.4f}")
        print(f"  F1 (macro): {results.f1_macro:.4f}")
        print(f"  Cohen's Kappa: {results.cohen_kappa:.4f}")

    return model, results, history


def evaluate_all_subjects_within(
    model_class: type,
    model_kwargs: Dict,
    subjects: Optional[List[int]] = None,
    use_moabb: bool = True,
    verbose: bool = True
) -> Dict[int, Tuple[EvaluationResults, TrainingHistory]]:
    """
    Evaluate within-subject classification for all subjects.

    Args:
        model_class: Model class to instantiate.
        model_kwargs: Keyword arguments for model.
        subjects: List of subject IDs.
        use_moabb: Whether to use MOABB.
        verbose: Whether to print progress.

    Returns:
        Dictionary mapping subject_id -> (results, history).
    """
    if subjects is None:
        subjects = data_config.subjects

    all_results = {}

    for subject_id in subjects:
        _, results, history = evaluate_within_subject(
            model_class,
            model_kwargs,
            subject_id,
            use_moabb=use_moabb,
            verbose=verbose
        )
        all_results[subject_id] = (results, history)

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("Within-Subject Summary")
        print(f"{'='*60}")
        print(f"{'Subject':<10} {'Accuracy':<12} {'Balanced':<12} {'F1':<12} {'Kappa':<12}")
        print("-" * 60)

        accs = []
        for subject_id, (results, _) in all_results.items():
            print(
                f"{subject_id:<10} "
                f"{results.accuracy:<12.4f} "
                f"{results.balanced_accuracy:<12.4f} "
                f"{results.f1_macro:<12.4f} "
                f"{results.cohen_kappa:<12.4f}"
            )
            accs.append(results.accuracy)

        print("-" * 60)
        print(f"{'Mean':<10} {np.mean(accs):<12.4f}")
        print(f"{'Std':<10} {np.std(accs):<12.4f}")

    return all_results


def evaluate_loso(
    model_class: type,
    model_kwargs: Dict,
    subjects: Optional[List[int]] = None,
    use_moabb: bool = True,
    verbose: bool = True
) -> Dict[int, Tuple[EvaluationResults, TrainingHistory]]:
    """
    Leave-One-Subject-Out cross-validation.

    Trains on N-1 subjects, tests on the held-out subject.
    Repeats for all subjects.

    Args:
        model_class: Model class to instantiate.
        model_kwargs: Keyword arguments for model.
        subjects: List of subject IDs.
        use_moabb: Whether to use MOABB.
        verbose: Whether to print progress.

    Returns:
        Dictionary mapping test_subject_id -> (results, history).
    """
    if subjects is None:
        subjects = data_config.subjects

    all_results = {}
    total_start = time.time()

    for test_subject in subjects:
        if verbose:
            print(f"\n{'='*60}")
            print(f"LOSO: Test Subject = {test_subject}")
            print(f"{'='*60}")

        # Prepare LOSO data
        train_splits, (X_test, y_test) = prepare_loso_data(
            test_subject,
            subjects=subjects,
            use_moabb=use_moabb,
            verbose=verbose
        )

        # Create data loaders
        train_loaders = create_dataloaders(train_splits)

        # Create test loader
        test_dataset = EEGDataset(X_test, y_test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=training_config.batch_size,
            shuffle=False
        )

        # Create and train model
        model = model_class(**model_kwargs)
        model, history = train_model(
            model,
            train_loaders["train"],
            train_loaders["val"],
            model_name=f"{model_class.__name__.lower()}_loso",
            subject_id=test_subject,
            verbose=verbose
        )

        # Evaluate
        results = evaluate_model(model, test_loader)
        all_results[test_subject] = (results, history)

        if verbose:
            print(f"\nLOSO Test Results (Subject {test_subject}):")
            print(f"  Accuracy: {results.accuracy:.4f}")
            print(f"  Balanced Accuracy: {results.balanced_accuracy:.4f}")

    total_time = time.time() - total_start

    # Print summary
    if verbose:
        print(f"\n{'='*60}")
        print("LOSO Cross-Validation Summary")
        print(f"{'='*60}")
        print(f"{'Test Subj':<12} {'Accuracy':<12} {'Balanced':<12} {'F1':<12} {'Kappa':<12}")
        print("-" * 60)

        accs = []
        balanced_accs = []
        for test_subject, (results, _) in all_results.items():
            print(
                f"{test_subject:<12} "
                f"{results.accuracy:<12.4f} "
                f"{results.balanced_accuracy:<12.4f} "
                f"{results.f1_macro:<12.4f} "
                f"{results.cohen_kappa:<12.4f}"
            )
            accs.append(results.accuracy)
            balanced_accs.append(results.balanced_accuracy)

        print("-" * 60)
        print(f"{'Mean':<12} {np.mean(accs):<12.4f} {np.mean(balanced_accs):<12.4f}")
        print(f"{'Std':<12} {np.std(accs):<12.4f} {np.std(balanced_accs):<12.4f}")
        print(f"\nTotal LOSO time: {total_time/60:.1f} minutes")

    return all_results


def create_results_summary(
    within_subject_results: Optional[Dict[int, Tuple[EvaluationResults, TrainingHistory]]] = None,
    loso_results: Optional[Dict[int, Tuple[EvaluationResults, TrainingHistory]]] = None,
    model_name: str = "Model"
) -> Dict:
    """
    Create a summary of evaluation results.

    Args:
        within_subject_results: Results from within-subject evaluation.
        loso_results: Results from LOSO evaluation.
        model_name: Name of the model.

    Returns:
        Summary dictionary.
    """
    summary = {"model": model_name}

    if within_subject_results is not None:
        accs = [r.accuracy for r, _ in within_subject_results.values()]
        bal_accs = [r.balanced_accuracy for r, _ in within_subject_results.values()]
        f1s = [r.f1_macro for r, _ in within_subject_results.values()]
        kappas = [r.cohen_kappa for r, _ in within_subject_results.values()]
        times = [h.total_training_time for _, h in within_subject_results.values()]

        summary["within_subject"] = {
            "accuracy_mean": np.mean(accs),
            "accuracy_std": np.std(accs),
            "balanced_accuracy_mean": np.mean(bal_accs),
            "balanced_accuracy_std": np.std(bal_accs),
            "f1_macro_mean": np.mean(f1s),
            "f1_macro_std": np.std(f1s),
            "cohen_kappa_mean": np.mean(kappas),
            "cohen_kappa_std": np.std(kappas),
            "training_time_mean": np.mean(times),
            "per_subject_accuracy": {s: r.accuracy for s, (r, _) in within_subject_results.items()},
        }

    if loso_results is not None:
        accs = [r.accuracy for r, _ in loso_results.values()]
        bal_accs = [r.balanced_accuracy for r, _ in loso_results.values()]
        f1s = [r.f1_macro for r, _ in loso_results.values()]
        kappas = [r.cohen_kappa for r, _ in loso_results.values()]

        summary["loso"] = {
            "accuracy_mean": np.mean(accs),
            "accuracy_std": np.std(accs),
            "balanced_accuracy_mean": np.mean(bal_accs),
            "balanced_accuracy_std": np.std(bal_accs),
            "f1_macro_mean": np.mean(f1s),
            "f1_macro_std": np.std(f1s),
            "cohen_kappa_mean": np.mean(kappas),
            "cohen_kappa_std": np.std(kappas),
            "per_subject_accuracy": {s: r.accuracy for s, (r, _) in loso_results.items()},
        }

    return summary


def print_comparison_table(
    summaries: List[Dict],
    title: str = "Model Comparison"
) -> None:
    """
    Print a comparison table of multiple models.

    Args:
        summaries: List of summary dictionaries from create_results_summary.
        title: Table title.
    """
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")

    # Within-subject comparison
    if all("within_subject" in s for s in summaries):
        print("\nWithin-Subject Classification:")
        print(f"{'Model':<20} {'Accuracy':<20} {'Balanced Acc':<20} {'F1 (macro)':<20}")
        print("-" * 80)
        for s in summaries:
            ws = s["within_subject"]
            print(
                f"{s['model']:<20} "
                f"{ws['accuracy_mean']:.4f} ± {ws['accuracy_std']:.4f}   "
                f"{ws['balanced_accuracy_mean']:.4f} ± {ws['balanced_accuracy_std']:.4f}   "
                f"{ws['f1_macro_mean']:.4f} ± {ws['f1_macro_std']:.4f}"
            )

    # LOSO comparison
    if all("loso" in s for s in summaries):
        print("\nLeave-One-Subject-Out (Cross-Subject):")
        print(f"{'Model':<20} {'Accuracy':<20} {'Balanced Acc':<20} {'F1 (macro)':<20}")
        print("-" * 80)
        for s in summaries:
            loso = s["loso"]
            print(
                f"{s['model']:<20} "
                f"{loso['accuracy_mean']:.4f} ± {loso['accuracy_std']:.4f}   "
                f"{loso['balanced_accuracy_mean']:.4f} ± {loso['balanced_accuracy_std']:.4f}   "
                f"{loso['f1_macro_mean']:.4f} ± {loso['f1_macro_std']:.4f}"
            )

    print(f"{'='*80}")


if __name__ == "__main__":
    # Test evaluation module
    print("Testing evaluation module...")
    print("=" * 50)

    # Test metrics computation
    np.random.seed(42)
    y_true = np.random.randint(0, 4, 100)
    y_pred = np.random.randint(0, 4, 100)

    results = compute_metrics(y_true, y_pred)
    print(results)
    print(f"\nConfusion Matrix:\n{results.confusion_matrix}")
    print(f"\nPer-class Accuracy: {results.per_class_accuracy}")

    print("\nAll tests passed!")
