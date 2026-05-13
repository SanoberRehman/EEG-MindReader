"""
Smoke Tests for EEG Motor Imagery Classification Pipeline.

Run with: pytest tests/test_pipeline.py -v

These tests verify the entire pipeline works end-to-end using synthetic data,
ensuring nothing is broken before running the full training on real data.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import (
    set_seed, SEED, DEVICE,
    data_config, preprocess_config, training_config,
    eegnet_config, cnn_lstm_config, transformer_config,
    DataConfig, PreprocessConfig, TrainingConfig,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def dummy_data():
    """Generate dummy EEG data for testing."""
    set_seed(SEED)

    n_samples = 64
    n_channels = data_config.n_channels  # 22
    n_timepoints = 500  # 2 seconds at 250 Hz
    n_classes = data_config.n_classes  # 4

    X = np.random.randn(n_samples, n_channels, n_timepoints).astype(np.float32)
    y = np.random.randint(0, n_classes, n_samples).astype(np.int64)

    return X, y, n_channels, n_timepoints, n_classes


@pytest.fixture(scope="module")
def data_loaders(dummy_data):
    """Create train/val/test data loaders."""
    X, y, _, _, _ = dummy_data

    # Split
    train_size = int(0.7 * len(X))
    val_size = int(0.15 * len(X))

    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=8, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val)),
        batch_size=8, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
        batch_size=8, shuffle=False
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# CONFIG TESTS
# =============================================================================

class TestConfig:
    """Test configuration module."""

    def test_seed_reproducibility(self):
        """Test that setting seed produces reproducible results."""
        set_seed(42)
        a1 = np.random.rand(10)
        t1 = torch.rand(10)

        set_seed(42)
        a2 = np.random.rand(10)
        t2 = torch.rand(10)

        np.testing.assert_array_equal(a1, a2)
        torch.testing.assert_close(t1, t2)

    def test_device_available(self):
        """Test that a device is available."""
        assert DEVICE is not None
        assert DEVICE.type in ['cpu', 'cuda', 'mps']

    def test_data_config(self):
        """Test data configuration values."""
        assert data_config.n_subjects == 9
        assert data_config.n_channels == 22
        assert data_config.n_classes == 4
        assert data_config.sampling_rate == 250
        assert len(data_config.class_names) == 4
        assert len(data_config.channel_names) == 22

    def test_preprocess_config(self):
        """Test preprocessing configuration values."""
        assert preprocess_config.low_freq == 8.0
        assert preprocess_config.high_freq == 30.0
        assert preprocess_config.tmin == 0.5
        assert preprocess_config.tmax == 2.5
        assert preprocess_config.train_ratio + preprocess_config.val_ratio + preprocess_config.test_ratio == 1.0


# =============================================================================
# DATA LOADER TESTS
# =============================================================================

class TestDataLoader:
    """Test data loading and preprocessing modules."""

    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        from src.data_loader import generate_synthetic_data

        data = generate_synthetic_data(subjects=[1], verbose=False)

        assert 1 in data
        assert 'raw' in data[1]
        assert 'events' in data[1]
        assert 'event_id' in data[1]

    def test_data_info(self):
        """Test data info retrieval."""
        from src.data_loader import get_data_info

        info = get_data_info()

        assert 'n_subjects' in info
        assert 'n_channels' in info
        assert 'n_classes' in info
        assert info['n_classes'] == 4


# =============================================================================
# PREPROCESSING TESTS
# =============================================================================

class TestPreprocessing:
    """Test preprocessing module."""

    def test_eeg_dataset(self):
        """Test EEGDataset class."""
        from src.preprocessing import EEGDataset

        X = np.random.randn(32, 22, 500).astype(np.float32)
        y = np.random.randint(0, 4, 32).astype(np.int64)

        dataset = EEGDataset(X, y)

        assert len(dataset) == 32

        x_sample, y_sample = dataset[0]
        assert x_sample.shape == (22, 500)
        assert isinstance(y_sample, torch.Tensor)

    def test_normalize_epochs(self):
        """Test epoch normalization."""
        from src.preprocessing import normalize_epochs

        data = np.random.randn(10, 22, 500)

        # Z-score normalization
        normalized = normalize_epochs(data, mode='zscore')

        # Check shape preserved
        assert normalized.shape == data.shape

        # Check roughly normalized (mean ~ 0 per channel)
        channel_means = normalized.mean(axis=2)
        np.testing.assert_array_almost_equal(channel_means, np.zeros_like(channel_means), decimal=5)

    def test_split_data(self):
        """Test data splitting."""
        from src.preprocessing import split_data

        X = np.random.randn(100, 22, 500)
        y = np.random.randint(0, 4, 100)

        splits = split_data(X, y, train_ratio=0.7, val_ratio=0.15)

        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits

        total = sum(len(splits[k][1]) for k in splits)
        assert total == 100


# =============================================================================
# MODEL TESTS
# =============================================================================

class TestEEGNet:
    """Test EEGNet model."""

    def test_model_creation(self, dummy_data):
        """Test EEGNet instantiation."""
        from src.models import EEGNet

        _, _, n_channels, n_timepoints, n_classes = dummy_data

        model = EEGNet(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        )

        assert model is not None
        assert model.count_parameters() > 0

    def test_forward_pass(self, dummy_data):
        """Test EEGNet forward pass."""
        from src.models import EEGNet

        X, y, n_channels, n_timepoints, n_classes = dummy_data

        model = EEGNet(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        )

        # Test with batch
        x_batch = torch.FloatTensor(X[:8])
        output = model(x_batch)

        assert output.shape == (8, n_classes)

    def test_feature_extraction(self, dummy_data):
        """Test EEGNet feature extraction."""
        from src.models import EEGNet

        X, _, n_channels, n_timepoints, n_classes = dummy_data

        model = EEGNet(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        )

        x_batch = torch.FloatTensor(X[:8])
        features = model.get_features(x_batch)

        assert features.ndim == 2
        assert features.shape[0] == 8


class TestCNNLSTM:
    """Test CNN-LSTM model."""

    def test_model_creation(self, dummy_data):
        """Test CNN-LSTM instantiation."""
        from src.models import CNNLSTM

        _, _, n_channels, n_timepoints, n_classes = dummy_data

        model = CNNLSTM(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        )

        assert model is not None
        assert model.count_parameters() > 0

    def test_forward_pass(self, dummy_data):
        """Test CNN-LSTM forward pass."""
        from src.models import CNNLSTM

        X, _, n_channels, n_timepoints, n_classes = dummy_data

        model = CNNLSTM(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        )

        x_batch = torch.FloatTensor(X[:8])
        output = model(x_batch)

        assert output.shape == (8, n_classes)


class TestTransformer:
    """Test EEG Transformer model."""

    def test_model_creation(self, dummy_data):
        """Test Transformer instantiation."""
        from src.models import EEGTransformer

        _, _, n_channels, n_timepoints, n_classes = dummy_data

        model = EEGTransformer(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        )

        assert model is not None
        assert model.count_parameters() > 0

    def test_forward_pass(self, dummy_data):
        """Test Transformer forward pass."""
        from src.models import EEGTransformer

        X, _, n_channels, n_timepoints, n_classes = dummy_data

        model = EEGTransformer(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        )

        x_batch = torch.FloatTensor(X[:8])
        output = model(x_batch)

        assert output.shape == (8, n_classes)

    def test_attention_maps(self, dummy_data):
        """Test Transformer attention map extraction."""
        from src.models import EEGTransformer

        X, _, n_channels, n_timepoints, n_classes = dummy_data

        model = EEGTransformer(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        )

        x_batch = torch.FloatTensor(X[:4])
        attn_maps = model.get_attention_maps(x_batch)

        assert len(attn_maps) == model.num_layers
        assert attn_maps[0].shape[0] == 4  # batch size


# =============================================================================
# TRAINING TESTS
# =============================================================================

class TestTraining:
    """Test training module."""

    def test_train_epoch(self, data_loaders, dummy_data):
        """Test single training epoch."""
        from src.models import EEGNet
        from src.training import train_epoch

        train_loader, _, _ = data_loaders
        _, _, n_channels, n_timepoints, n_classes = dummy_data

        model = EEGNet(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        ).to(DEVICE)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        loss, acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1

    def test_validate(self, data_loaders, dummy_data):
        """Test validation."""
        from src.models import EEGNet
        from src.training import validate

        _, val_loader, _ = data_loaders
        _, _, n_channels, n_timepoints, n_classes = dummy_data

        model = EEGNet(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        ).to(DEVICE)

        criterion = torch.nn.CrossEntropyLoss()

        loss, acc = validate(model, val_loader, criterion, DEVICE)

        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert 0 <= acc <= 1

    def test_train_model_short(self, data_loaders, dummy_data):
        """Test full training loop (short)."""
        from src.models import EEGNet
        from src.training import train_model
        from src.config import TrainingConfig

        train_loader, val_loader, _ = data_loaders
        _, _, n_channels, n_timepoints, n_classes = dummy_data

        model = EEGNet(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        )

        # Short training config
        config = TrainingConfig(
            num_epochs=2,
            early_stopping_patience=5,
            batch_size=8
        )

        trained_model, history = train_model(
            model, train_loader, val_loader,
            config=config,
            model_name='test_eegnet',
            verbose=False
        )

        assert trained_model is not None
        assert len(history.train_loss) == 2
        assert len(history.val_acc) == 2

    def test_get_predictions(self, data_loaders, dummy_data):
        """Test prediction extraction."""
        from src.models import EEGNet
        from src.training import get_predictions

        _, _, test_loader = data_loaders
        _, _, n_channels, n_timepoints, n_classes = dummy_data

        model = EEGNet(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        )

        preds, probs, labels = get_predictions(model, test_loader)

        assert len(preds) == len(labels)
        assert probs.shape[1] == n_classes
        assert all(0 <= p < n_classes for p in preds)


# =============================================================================
# EVALUATION TESTS
# =============================================================================

class TestEvaluation:
    """Test evaluation module."""

    def test_compute_metrics(self):
        """Test metrics computation."""
        from src.evaluation import compute_metrics

        y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
        y_pred = np.array([0, 1, 2, 3, 1, 1, 2, 3])  # One error

        results = compute_metrics(y_true, y_pred)

        assert 0 <= results.accuracy <= 1
        assert 0 <= results.balanced_accuracy <= 1
        assert 0 <= results.f1_macro <= 1
        assert results.confusion_matrix.shape == (4, 4)

    def test_evaluate_model(self, data_loaders, dummy_data):
        """Test model evaluation."""
        from src.models import EEGNet
        from src.evaluation import evaluate_model

        _, _, test_loader = data_loaders
        _, _, n_channels, n_timepoints, n_classes = dummy_data

        model = EEGNet(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        )

        results = evaluate_model(model, test_loader)

        assert results is not None
        assert hasattr(results, 'accuracy')
        assert hasattr(results, 'confusion_matrix')


# =============================================================================
# VISUALIZATION TESTS
# =============================================================================

class TestVisualization:
    """Test visualization module."""

    def test_setup_style(self):
        """Test style setup."""
        from src.visualization import setup_style

        # Should not raise
        setup_style()

    def test_plot_confusion_matrix(self):
        """Test confusion matrix plotting."""
        from src.visualization import plot_confusion_matrix
        import matplotlib.pyplot as plt

        cm = np.array([
            [10, 2, 1, 0],
            [1, 12, 2, 1],
            [0, 1, 15, 0],
            [1, 0, 1, 11]
        ])

        fig = plot_confusion_matrix(cm, title="Test CM")

        assert fig is not None
        plt.close(fig)

    def test_plot_training_history(self):
        """Test training history plotting."""
        from src.visualization import plot_training_history
        import matplotlib.pyplot as plt

        history = {
            'train_loss': [1.0, 0.8, 0.6],
            'val_loss': [1.1, 0.9, 0.7],
            'train_acc': [0.3, 0.5, 0.7],
            'val_acc': [0.25, 0.45, 0.65],
            'best_epoch': 2
        }

        fig = plot_training_history(history, title="Test History")

        assert fig is not None
        plt.close(fig)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_eegnet(self, dummy_data):
        """Test full pipeline with EEGNet."""
        from src.models import EEGNet
        from src.preprocessing import EEGDataset, split_data
        from src.training import train_model, get_predictions
        from src.evaluation import compute_metrics
        from src.config import TrainingConfig
        from torch.utils.data import DataLoader

        X, y, n_channels, n_timepoints, n_classes = dummy_data

        # Preprocess
        splits = split_data(X, y)

        train_loader = DataLoader(
            EEGDataset(*splits['train']), batch_size=8, shuffle=True
        )
        val_loader = DataLoader(
            EEGDataset(*splits['val']), batch_size=8
        )
        test_loader = DataLoader(
            EEGDataset(*splits['test']), batch_size=8
        )

        # Create model
        model = EEGNet(
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            n_classes=n_classes
        )

        # Train
        config = TrainingConfig(num_epochs=2, early_stopping_patience=5)
        model, history = train_model(
            model, train_loader, val_loader,
            config=config, model_name='integration_test',
            verbose=False
        )

        # Evaluate
        preds, _, labels = get_predictions(model, test_loader)
        results = compute_metrics(labels, preds)

        # Assertions
        assert history.total_training_time > 0
        assert results.accuracy >= 0  # Random chance is 0.25
        assert results.confusion_matrix.sum() == len(splits['test'][1])

    def test_all_models_forward(self, dummy_data):
        """Test all models can do forward pass."""
        from src.models import EEGNet, CNNLSTM, EEGTransformer

        X, _, n_channels, n_timepoints, n_classes = dummy_data
        x_batch = torch.FloatTensor(X[:4])

        models = [
            EEGNet(n_channels, n_timepoints, n_classes),
            CNNLSTM(n_channels, n_timepoints, n_classes),
            EEGTransformer(n_channels, n_timepoints, n_classes),
        ]

        for model in models:
            output = model(x_batch)
            assert output.shape == (4, n_classes), f"{model.__class__.__name__} failed"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
