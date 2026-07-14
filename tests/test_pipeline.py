"""
tests/test_pipeline.py — Integration Tests for the ML Pipeline
===============================================================
Tests the end-to-end pipeline using synthetic data (no file I/O dependencies).

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.train import split_data, compute_metrics, _count_combos
from src.utils import safe_divide, clamp, memory_usage


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def sample_X_y():
    """Generate a simple binary classification dataset."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.random((200, 10)),
        columns=[f"feature_{i}" for i in range(10)],
    )
    y = pd.Series((rng.random(200) > 0.75).astype(int), name="Churn")
    return X, y


@pytest.fixture()
def trained_rf(sample_X_y):
    """Fit a small RF for testing."""
    X, y = sample_X_y
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model, X, y


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------
class TestSplitData:
    def test_proportions(self, sample_X_y):
        X, y = sample_X_y
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        assert len(X_train) == pytest.approx(160, abs=5)
        assert len(X_test) == pytest.approx(40, abs=5)

    def test_stratified(self, sample_X_y):
        X, y = sample_X_y
        _, _, y_train, y_test = split_data(X, y, test_size=0.2)
        # Churn rate in both splits should be close
        assert abs(y_train.mean() - y_test.mean()) < 0.10

    def test_no_overlap(self, sample_X_y):
        X, y = sample_X_y
        X_train, X_test, _, _ = split_data(X, y)
        common_idx = set(X_train.index) & set(X_test.index)
        assert len(common_idx) == 0


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------
class TestComputeMetrics:
    def test_all_keys_present(self, trained_rf):
        model, X, y = trained_rf
        metrics = compute_metrics(model, X, y)
        required = {"accuracy", "precision", "recall", "f1_score", "roc_auc",
                    "confusion_matrix", "n_samples"}
        assert required.issubset(metrics.keys())

    def test_auc_in_range(self, trained_rf):
        model, X, y = trained_rf
        metrics = compute_metrics(model, X, y)
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_accuracy_in_range(self, trained_rf):
        model, X, y = trained_rf
        metrics = compute_metrics(model, X, y)
        assert 0.0 <= metrics["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# _count_combos
# ---------------------------------------------------------------------------
class TestCountCombos:
    def test_single_param(self):
        assert _count_combos({"n_estimators": [100, 200, 300]}) == 3

    def test_multiple_params(self):
        grid = {"a": [1, 2], "b": [10, 20, 30]}
        assert _count_combos(grid) == 6


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
class TestUtils:
    def test_safe_divide_normal(self):
        assert safe_divide(10, 2) == 5.0

    def test_safe_divide_by_zero(self):
        assert safe_divide(10, 0) == 0.0

    def test_safe_divide_custom_default(self):
        assert safe_divide(10, 0, default=-1.0) == -1.0

    def test_clamp_within_range(self):
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_clamp_below_min(self):
        assert clamp(-0.1, 0.0, 1.0) == 0.0

    def test_clamp_above_max(self):
        assert clamp(1.5, 0.0, 1.0) == 1.0

    def test_memory_usage_returns_string(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = memory_usage(df)
        assert isinstance(result, str)
        assert any(unit in result for unit in ["B", "KB", "MB", "GB"])
