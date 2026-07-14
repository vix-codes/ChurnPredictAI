"""
src/pipeline.py — End-to-End Orchestration for ChurnPredict AI
===============================================================
Single entry point that chains preprocessing → training → evaluation
→ explainability. Designed to be run from the command line.

Usage
-----
    python src/pipeline.py                   # full pipeline
    python src/pipeline.py --skip-explain    # skip SHAP (faster)
    python src/pipeline.py --data path/to/file.csv

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from config import config as cfg
from config import paths
from src.utils import get_logger, timer, save_json
from src.preprocess import run_preprocessing
from src.train import run_training, split_data, compute_metrics
from src.evaluate import run_evaluation
from src.explain import run_explain

# Ensure all output directories exist
paths.ensure_directories()

logger = get_logger(__name__, log_file=paths.PIPELINE_LOG_FILE)


# ---------------------------------------------------------------------------
# Pipeline Steps
# ---------------------------------------------------------------------------
@timer
def step_preprocess(data_path: Path | None) -> tuple:
    """Run full preprocessing pipeline and return artifacts."""
    logger.info("=" * 60)
    logger.info("STEP 1 — DATA PREPROCESSING")
    logger.info("=" * 60)
    X, y, feature_cols, encoder, scaler = run_preprocessing(
        filepath=data_path,
        save_processed=True,
        fit_transformers=True,
    )
    return X, y, feature_cols, encoder, scaler


@timer
def step_train(data_path: Path | None, compare_models: bool) -> dict:
    """Run training pipeline and return metrics."""
    logger.info("=" * 60)
    logger.info("STEP 2 — MODEL TRAINING")
    logger.info("=" * 60)
    metrics = run_training(data_filepath=data_path, compare_models=compare_models)
    return metrics


@timer
def step_evaluate(model, X_test, y_test, feature_cols: list[str]) -> None:
    """Run evaluation and generate all plots."""
    logger.info("=" * 60)
    logger.info("STEP 3 — MODEL EVALUATION")
    logger.info("=" * 60)
    run_evaluation(model, X_test, y_test, feature_cols)


@timer
def step_explain(model, X_train, X_test, y_test) -> None:
    """Run SHAP explainability."""
    logger.info("=" * 60)
    logger.info("STEP 4 — MODEL EXPLAINABILITY")
    logger.info("=" * 60)
    run_explain(model, X_train, X_test, y_test)


# ---------------------------------------------------------------------------
# Master Pipeline
# ---------------------------------------------------------------------------
@timer
def run_pipeline(
    data_path: Path | None = None,
    compare_models: bool = True,
    run_explain_step: bool = True,
) -> None:
    """
    Execute the full ChurnPredict AI pipeline.

    Parameters
    ----------
    data_path       : Path | None  — raw CSV path
    compare_models  : bool         — include XGBoost/LightGBM comparison
    run_explain_step: bool         — run SHAP explainability step
    """
    start_time = time.perf_counter()

    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║         ChurnPredict AI — Pipeline Starting          ║")
    logger.info("╚══════════════════════════════════════════════════════╝")

    # ---- Step 1: Preprocessing ----
    X, y, feature_cols, encoder, scaler = step_preprocess(data_path)

    # ---- Step 2: Training (also re-preprocesses internally for clean split) ----
    metrics = step_train(data_path, compare_models)

    # ---- Load trained model for evaluation ----
    model = joblib.load(paths.MODEL_FILE)

    # ---- Re-split data for evaluation plots ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.TEST_SIZE,
        random_state=cfg.RANDOM_STATE,
        stratify=y,
    )

    # ---- Step 3: Evaluation ----
    step_evaluate(model, X_test, y_test, feature_cols)

    # ---- Step 4: Explainability ----
    if run_explain_step:
        step_explain(model, X_train, X_test, y_test)

    total_time = time.perf_counter() - start_time

    logger.info("╔══════════════════════════════════════════════════════╗")
    logger.info("║         ChurnPredict AI — Pipeline Complete          ║")
    logger.info("║  Total time : %.1f seconds                            ║", total_time)
    logger.info("╚══════════════════════════════════════════════════════╝")

    # ---- Print summary ----
    print("\n" + "═" * 55)
    print("  ChurnPredict AI — Pipeline Complete")
    print("═" * 55)
    print(f"  Test Accuracy  : {metrics['test']['accuracy']:.4f}")
    print(f"  Test Precision : {metrics['test']['precision']:.4f}")
    print(f"  Test Recall    : {metrics['test']['recall']:.4f}")
    print(f"  Test F1        : {metrics['test']['f1_score']:.4f}")
    print(f"  Test AUC       : {metrics['test']['roc_auc']:.4f}")
    print(f"  Total Time     : {total_time:.1f}s")
    print("═" * 55)
    print(f"\n  ✔  Model     → {paths.MODEL_FILE}")
    print(f"  ✔  Metrics   → {paths.METRICS_FILE}")
    print(f"  ✔  Figures   → {paths.FIGURES_DIR}")
    print(f"  ✔  Reports   → {paths.REPORTS_DIR}")
    print("\n  Run the dashboard:")
    print("  streamlit run app/main.py")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ChurnPredict AI — End-to-End ML Pipeline"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to raw CSV file (default: data/raw/telco_churn.csv)",
    )
    parser.add_argument(
        "--skip-compare",
        action="store_true",
        help="Skip XGBoost/LightGBM model comparison",
    )
    parser.add_argument(
        "--skip-explain",
        action="store_true",
        help="Skip SHAP explainability step (faster)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        data_path=args.data,
        compare_models=not args.skip_compare,
        run_explain_step=not args.skip_explain,
    )
