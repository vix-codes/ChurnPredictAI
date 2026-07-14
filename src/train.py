"""
src/train.py — Model Training Pipeline for ChurnPredict AI
===========================================================
Trains a Random Forest classifier with GridSearchCV hyperparameter tuning.
Optionally trains XGBoost and LightGBM for model comparison.
Saves the best model, metrics, and a comparison report.

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import config as cfg
from config import paths
from src.utils import get_logger, timer, save_json
from src.preprocess import run_preprocessing

warnings.filterwarnings("ignore")
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data Splitting
# ---------------------------------------------------------------------------
def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = cfg.TEST_SIZE,
    random_state: int = cfg.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified 80/20 train-test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(
        "Split — Train: %d | Test: %d | Churn rate train: %.2f%% | test: %.2f%%",
        len(X_train),
        len(X_test),
        y_train.mean() * 100,
        y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Random Forest Training
# ---------------------------------------------------------------------------
@timer
def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict | None = None,
    cv: int = cfg.RF_CV_FOLDS,
    scoring: str = cfg.RF_SCORING,
    n_jobs: int = cfg.RF_N_JOBS,
) -> RandomForestClassifier:
    """
    Train Random Forest with GridSearchCV.

    Parameters
    ----------
    X_train    : pd.DataFrame
    y_train    : pd.Series
    param_grid : dict | None  — overrides cfg.RF_PARAM_GRID
    cv         : int          — cross-validation folds
    scoring    : str          — optimisation metric
    n_jobs     : int          — parallel jobs

    Returns
    -------
    RandomForestClassifier  — best estimator from GridSearchCV
    """
    grid = param_grid or cfg.RF_PARAM_GRID

    logger.info("Starting GridSearchCV for Random Forest — %d param combos, %d-fold CV",
                _count_combos(grid), cv)

    base_rf = RandomForestClassifier(random_state=cfg.RANDOM_STATE)
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=cfg.RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=base_rf,
        param_grid=grid,
        cv=cv_splitter,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
        refit=True,
        return_train_score=True,
    )

    grid_search.fit(X_train, y_train)

    best = grid_search.best_estimator_
    logger.info("Best params  : %s", grid_search.best_params_)
    logger.info("Best CV AUC  : %.4f", grid_search.best_score_)

    return best


# ---------------------------------------------------------------------------
# Optional: XGBoost Training
# ---------------------------------------------------------------------------
def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any | None:
    """
    Train XGBoost with GridSearchCV if xgboost is available.

    Returns best estimator or None if xgboost is not installed.
    """
    try:
        from xgboost import XGBClassifier  # type: ignore

        logger.info("Training XGBoost …")
        base_xgb = XGBClassifier(
            random_state=cfg.RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        )
        cv_splitter = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=cfg.RANDOM_STATE
        )
        grid_search = GridSearchCV(
            estimator=base_xgb,
            param_grid=cfg.XGB_PARAM_GRID,
            cv=cv_splitter,
            scoring="roc_auc",
            n_jobs=cfg.RF_N_JOBS,
            verbose=0,
            refit=True,
        )
        grid_search.fit(X_train, y_train)
        best = grid_search.best_estimator_
        logger.info("XGBoost best CV AUC: %.4f", grid_search.best_score_)
        return best
    except ImportError:
        logger.warning("XGBoost not installed — skipping.")
        return None


# ---------------------------------------------------------------------------
# Optional: LightGBM Training
# ---------------------------------------------------------------------------
def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any | None:
    """
    Train LightGBM with GridSearchCV if lightgbm is available.

    Returns best estimator or None if lightgbm is not installed.
    """
    try:
        from lightgbm import LGBMClassifier  # type: ignore

        logger.info("Training LightGBM …")
        base_lgb = LGBMClassifier(
            random_state=cfg.RANDOM_STATE,
            verbosity=-1,
        )
        cv_splitter = StratifiedKFold(
            n_splits=3, shuffle=True, random_state=cfg.RANDOM_STATE
        )
        grid_search = GridSearchCV(
            estimator=base_lgb,
            param_grid=cfg.LGB_PARAM_GRID,
            cv=cv_splitter,
            scoring="roc_auc",
            n_jobs=cfg.RF_N_JOBS,
            verbose=0,
            refit=True,
        )
        grid_search.fit(X_train, y_train)
        best = grid_search.best_estimator_
        logger.info("LightGBM best CV AUC: %.4f", grid_search.best_score_)
        return best
    except ImportError:
        logger.warning("LightGBM not installed — skipping.")
        return None


# ---------------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------------
def compute_metrics(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = cfg.DEFAULT_THRESHOLD,
    split_name: str = "test",
) -> dict:
    """
    Compute a full suite of classification metrics.

    Parameters
    ----------
    model       : fitted sklearn-compatible estimator
    X           : pd.DataFrame  — feature matrix
    y           : pd.Series     — true labels
    threshold   : float         — decision threshold
    split_name  : str           — label for logging

    Returns
    -------
    dict — metric name → value
    """
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics = {
        "split": split_name,
        "accuracy": round(accuracy_score(y, preds), 4),
        "precision": round(precision_score(y, preds, zero_division=0), 4),
        "recall": round(recall_score(y, preds, zero_division=0), 4),
        "f1_score": round(f1_score(y, preds, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
        "classification_report": classification_report(y, preds),
        "threshold": threshold,
        "n_samples": len(y),
        "churn_rate": round(float(y.mean()), 4),
    }

    logger.info(
        "[%s] Accuracy=%.4f | Precision=%.4f | Recall=%.4f | F1=%.4f | AUC=%.4f",
        split_name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["roc_auc"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Cross-Validation
# ---------------------------------------------------------------------------
def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
) -> dict:
    """
    Run stratified K-fold cross-validation and return mean ± std metrics.

    Returns
    -------
    dict  — {metric: {"mean": float, "std": float}}
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=cfg.RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    results = cross_validate(
        model, X, y, cv=cv_splitter, scoring=scoring, n_jobs=cfg.RF_N_JOBS
    )

    summary: dict = {}
    for metric in scoring:
        key = f"test_{metric}"
        vals = results[key]
        summary[metric] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
        }
        logger.info("CV %s: %.4f ± %.4f", metric, vals.mean(), vals.std())

    return summary


# ---------------------------------------------------------------------------
# Save Model
# ---------------------------------------------------------------------------
def save_model(model: Any, path: Path) -> None:
    """Persist a fitted model to disk using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Model saved → %s", path)


# ---------------------------------------------------------------------------
# Master Training Pipeline
# ---------------------------------------------------------------------------
@timer
def run_training(
    data_filepath: Path | None = None,
    compare_models: bool = True,
) -> dict:
    """
    End-to-end training pipeline.

    Steps
    -----
    1. Preprocess data
    2. Train/test split
    3. Train Random Forest (GridSearchCV)
    4. Optionally train XGBoost + LightGBM
    5. Evaluate on train + test sets
    6. Cross-validate best model
    7. Save model artifacts + metrics

    Returns
    -------
    dict — model metrics summary
    """
    # ---- Preprocess ----
    X, y, feature_cols, encoder, scaler = run_preprocessing(
        filepath=data_filepath,
        save_processed=True,
        fit_transformers=True,
    )

    # ---- Split ----
    X_train, X_test, y_train, y_test = split_data(X, y)

    # ---- Train Random Forest ----
    rf_model = train_random_forest(X_train, y_train)

    # ---- Evaluate RF ----
    train_metrics = compute_metrics(rf_model, X_train, y_train, split_name="train")
    test_metrics = compute_metrics(rf_model, X_test, y_test, split_name="test")

    # ---- Cross-Validate ----
    cv_metrics = cross_validate_model(rf_model, X, y)

    # ---- Save RF model ----
    save_model(rf_model, paths.MODEL_FILE)

    # ---- Save classification report ----
    paths.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(paths.CLASSIFICATION_REPORT_FILE, "w") as f:
        f.write("=== ChurnPredict AI — Classification Report ===\n\n")
        f.write(test_metrics["classification_report"])
    logger.info("Classification report saved → %s", paths.CLASSIFICATION_REPORT_FILE)

    # ---- Compile metrics ----
    metrics_out: dict = {
        "model": "RandomForestClassifier",
        "train": {k: v for k, v in train_metrics.items() if k != "classification_report"},
        "test": {k: v for k, v in test_metrics.items() if k != "classification_report"},
        "cross_validation": cv_metrics,
        "feature_count": len(feature_cols),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
    }

    # ---- Optional model comparison ----
    comparison: dict = {"RandomForest": test_metrics["roc_auc"]}
    if compare_models:
        xgb = train_xgboost(X_train, y_train)
        if xgb is not None:
            save_model(xgb, paths.XGB_MODEL_FILE)
            xgb_metrics = compute_metrics(xgb, X_test, y_test, split_name="xgboost-test")
            comparison["XGBoost"] = xgb_metrics["roc_auc"]
            metrics_out["xgboost_test"] = {
                k: v for k, v in xgb_metrics.items() if k != "classification_report"
            }

        lgb = train_lightgbm(X_train, y_train)
        if lgb is not None:
            save_model(lgb, paths.LGB_MODEL_FILE)
            lgb_metrics = compute_metrics(lgb, X_test, y_test, split_name="lightgbm-test")
            comparison["LightGBM"] = lgb_metrics["roc_auc"]
            metrics_out["lightgbm_test"] = {
                k: v for k, v in lgb_metrics.items() if k != "classification_report"
            }

    metrics_out["model_comparison_auc"] = comparison

    # ---- Save metrics JSON ----
    paths.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(metrics_out, paths.METRICS_FILE)
    logger.info("Metrics saved → %s", paths.METRICS_FILE)

    save_json(comparison, paths.COMPARISON_METRICS_FILE)

    return metrics_out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _count_combos(param_grid: dict) -> int:
    """Count total hyperparameter combinations in a grid."""
    total = 1
    for v in param_grid.values():
        total *= len(v)
    return total


if __name__ == "__main__":
    metrics = run_training(compare_models=True)
    print("\n=== Training Complete ===")
    print(f"Test AUC  : {metrics['test']['roc_auc']}")
    print(f"Test F1   : {metrics['test']['f1_score']}")
    print(f"Test Acc  : {metrics['test']['accuracy']}")
