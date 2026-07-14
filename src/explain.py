"""
src/explain.py — Model Explainability for ChurnPredict AI
==========================================================
Generates SHAP-based explanations: global summary plots, waterfall charts,
dependence plots, and permutation importance.

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import paths
from config.config import SHAP_MAX_DISPLAY, TOP_N_FEATURES
from src.utils import get_logger, timer

logger = get_logger(__name__)

# Dark theme for all SHAP plots
PLOT_STYLE = {
    "figure.facecolor": "#0F172A",
    "axes.facecolor": "#1E293B",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#CBD5E1",
    "xtick.color": "#CBD5E1",
    "ytick.color": "#CBD5E1",
    "text.color": "#F1F5F9",
}


def _apply_style() -> None:
    plt.rcParams.update(PLOT_STYLE)


# ---------------------------------------------------------------------------
# SHAP Explainer Factory
# ---------------------------------------------------------------------------
def build_explainer(model: Any, X_background: pd.DataFrame) -> Any:
    """
    Create a SHAP TreeExplainer for tree-based models.

    Parameters
    ----------
    model        : fitted tree-based estimator
    X_background : pd.DataFrame  — background dataset for SHAP

    Returns
    -------
    shap.TreeExplainer
    """
    try:
        import shap  # type: ignore
    except ImportError:
        raise ImportError("SHAP is not installed. Run: pip install shap")

    logger.info("Building SHAP TreeExplainer …")
    explainer = shap.TreeExplainer(model, feature_perturbation="interventional")
    logger.info("SHAP explainer ready.")
    return explainer


def compute_shap_values(
    explainer: Any,
    X: pd.DataFrame,
    max_rows: int = 500,
) -> Any:
    """
    Compute SHAP values for up to *max_rows* samples.

    Large datasets are sampled for speed.

    Parameters
    ----------
    explainer : shap.TreeExplainer
    X         : pd.DataFrame
    max_rows  : int  — cap for computation speed

    Returns
    -------
    shap.Explanation or np.ndarray
    """
    import shap  # type: ignore

    if len(X) > max_rows:
        logger.info("Sampling %d rows for SHAP computation", max_rows)
        X = X.sample(max_rows, random_state=42)

    logger.info("Computing SHAP values for %d samples …", len(X))
    shap_values = explainer(X)
    logger.info("SHAP computation complete.")
    return shap_values


# ---------------------------------------------------------------------------
# SHAP Summary Plot
# ---------------------------------------------------------------------------
@timer
def plot_shap_summary(
    shap_values: Any,
    X: pd.DataFrame,
    max_display: int = SHAP_MAX_DISPLAY,
    save_path: Path | None = None,
) -> None:
    """
    Generate SHAP beeswarm (dot) summary plot and save to figures/.

    Parameters
    ----------
    shap_values  : shap.Explanation
    X            : pd.DataFrame  — (possibly sampled) feature matrix
    max_display  : int
    save_path    : Path | None
    """
    import shap  # type: ignore

    _apply_style()
    fig, ax = plt.subplots(figsize=(12, 8))

    # Handle both old (ndarray) and new (Explanation) SHAP API
    vals = shap_values.values if hasattr(shap_values, "values") else shap_values

    # For binary classification, use class-1 SHAP values
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    shap.summary_plot(
        vals,
        X,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )

    plt.title("SHAP Feature Importance — ChurnPredict AI", fontsize=14, fontweight="bold",
              color="#F1F5F9")
    fig = plt.gcf()
    fig.patch.set_facecolor("#0F172A")

    out = save_path or paths.SHAP_SUMMARY_PLOT
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0F172A")
    logger.info("SHAP summary plot saved → %s", out)
    plt.close("all")


# ---------------------------------------------------------------------------
# SHAP Waterfall Plot (single prediction)
# ---------------------------------------------------------------------------
@timer
def plot_shap_waterfall(
    shap_values: Any,
    row_index: int = 0,
    save_path: Path | None = None,
) -> None:
    """
    Generate SHAP waterfall plot for a single prediction.

    Parameters
    ----------
    shap_values : shap.Explanation
    row_index   : int  — which row to explain
    save_path   : Path | None
    """
    import shap  # type: ignore

    _apply_style()

    try:
        shap.waterfall_plot(shap_values[row_index], max_display=15, show=False)
    except Exception:
        # Fallback for older SHAP versions
        vals = shap_values.values if hasattr(shap_values, "values") else shap_values
        if vals.ndim == 3:
            vals = vals[:, :, 1]
        logger.warning("Waterfall plot fallback — using bar plot")
        shap.summary_plot(vals[[row_index]], show=False, plot_type="bar")

    plt.title("SHAP Waterfall — Single Customer Explanation", fontsize=13,
              fontweight="bold", color="#F1F5F9")
    fig = plt.gcf()
    fig.patch.set_facecolor("#0F172A")

    out = save_path or paths.SHAP_WATERFALL_PLOT
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0F172A")
    logger.info("SHAP waterfall saved → %s", out)
    plt.close("all")


# ---------------------------------------------------------------------------
# SHAP Dependence Plot
# ---------------------------------------------------------------------------
def plot_shap_dependence(
    shap_values: Any,
    X: pd.DataFrame,
    feature: str,
    interaction_feature: str | None = "auto",
    save_path: Path | None = None,
) -> None:
    """
    Plot a SHAP dependence plot for a single feature.

    Parameters
    ----------
    shap_values         : shap.Explanation or np.ndarray
    X                   : pd.DataFrame
    feature             : str   — primary feature to plot
    interaction_feature : str | None — coloring feature ("auto" = SHAP picks)
    save_path           : Path | None
    """
    import shap  # type: ignore

    _apply_style()

    vals = shap_values.values if hasattr(shap_values, "values") else shap_values
    if vals.ndim == 3:
        vals = vals[:, :, 1]

    fig, ax = plt.subplots(figsize=(9, 6))
    shap.dependence_plot(
        feature,
        vals,
        X,
        interaction_index=interaction_feature,
        ax=ax,
        show=False,
    )
    ax.set_title(f"SHAP Dependence: {feature}", fontsize=13, fontweight="bold")
    fig.patch.set_facecolor("#0F172A")

    out = save_path or (paths.FIGURES_DIR / f"shap_dependence_{feature}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0F172A")
    logger.info("SHAP dependence plot saved → %s", out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Permutation Importance
# ---------------------------------------------------------------------------
@timer
def compute_permutation_importance(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_repeats: int = 10,
    top_n: int = TOP_N_FEATURES,
) -> pd.DataFrame:
    """
    Compute permutation feature importance using sklearn.

    Parameters
    ----------
    model     : fitted estimator
    X_test    : pd.DataFrame
    y_test    : pd.Series
    n_repeats : int
    top_n     : int

    Returns
    -------
    pd.DataFrame — sorted by mean importance descending
    """
    from sklearn.inspection import permutation_importance  # type: ignore

    logger.info("Computing permutation importance (%d repeats) …", n_repeats)
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        scoring="roc_auc",
        n_jobs=-1,
    )

    pi_df = (
        pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    logger.info("Top feature by permutation: %s", pi_df.iloc[0]["feature"])
    return pi_df


# ---------------------------------------------------------------------------
# Master Explainability Runner
# ---------------------------------------------------------------------------
@timer
def run_explain(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    max_rows: int = 300,
) -> None:
    """
    Generate all explainability artefacts: SHAP summary, waterfall,
    dependence plots, permutation importance.
    """
    try:
        import shap  # type: ignore  # noqa: F401
    except ImportError:
        logger.warning("SHAP not installed — skipping explanation plots. Run: pip install shap")
        return

    explainer = build_explainer(model, X_train)
    shap_vals = compute_shap_values(explainer, X_test, max_rows=max_rows)

    # Align X_test to the same sample used for SHAP
    X_shap = X_test
    if len(X_test) > max_rows:
        X_shap = X_test.sample(max_rows, random_state=42)

    plot_shap_summary(shap_vals, X_shap)
    plot_shap_waterfall(shap_vals, row_index=0)

    # Dependence plots for top 3 features
    model_fi = pd.Series(model.feature_importances_, index=X_test.columns)
    top3 = model_fi.nlargest(3).index.tolist()
    for feat in top3:
        try:
            plot_shap_dependence(shap_vals, X_shap, feature=feat)
        except Exception as e:
            logger.warning("Skipping dependence plot for %s: %s", feat, e)

    # Permutation importance
    perm_df = compute_permutation_importance(model, X_test, y_test)
    perm_path = paths.REPORTS_DIR / "permutation_importance.csv"
    perm_df.to_csv(perm_path, index=False)
    logger.info("Permutation importance saved → %s", perm_path)

    logger.info("All explainability artefacts generated.")
