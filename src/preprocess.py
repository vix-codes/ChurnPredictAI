"""
src/preprocess.py — Data Preprocessing Pipeline for ChurnPredict AI
====================================================================
Handles loading, cleaning, encoding, scaling, and persisting of all
data artifacts required for model training and inference.

Author  : ChurnPredict AI Engineering Team
Version : 1.0.0
"""

from __future__ import annotations

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from config import config as cfg
from config import paths
from src.utils import get_logger, timer, describe_dataframe, validate_columns
from src.features import engineer_features

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------
@timer
def load_raw_data(filepath: Path | None = None) -> pd.DataFrame:
    """
    Load raw CSV into a DataFrame.

    Parameters
    ----------
    filepath : Path | None
        Override the default raw data path from config/paths.py.

    Returns
    -------
    pd.DataFrame
        Raw unmodified customer data.
    """
    path = filepath or paths.RAW_DATA_FILE
    if not path.exists():
        # Fall back to sample data for development/demo
        sample = paths.SAMPLE_DATA_FILE
        if sample.exists():
            logger.warning("Raw file not found — using sample data: %s", sample)
            path = sample
        else:
            raise FileNotFoundError(
                f"Raw data file not found at {path}.\n"
                "Please place 'telco_churn.csv' in data/raw/ "
                "or run: python data/sample/generate_sample.py"
            )

    logger.info("Loading data from: %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns", *df.shape)
    describe_dataframe(df, logger)
    return df


# ---------------------------------------------------------------------------
# 2. Basic Cleaning
# ---------------------------------------------------------------------------
@timer
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply raw-level data cleaning:

    * Strip whitespace from string columns
    * Convert TotalCharges blanks → NaN → numeric
    * Fill TotalCharges NaN with tenure × MonthlyCharges
    * Drop duplicate rows
    * Remove rows missing the target column

    Parameters
    ----------
    df : pd.DataFrame  — raw input

    Returns
    -------
    pd.DataFrame  — cleaned dataframe
    """
    df = df.copy()

    # Strip whitespace from object columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

    # TotalCharges: blank strings → NaN
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        nulls = df["TotalCharges"].isnull().sum()
        if nulls:
            logger.info("Imputing %d TotalCharges nulls with tenure × MonthlyCharges", nulls)
            df["TotalCharges"] = df["TotalCharges"].fillna(
                df["tenure"] * df["MonthlyCharges"]
            )

    # Drop duplicates
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    if before != after:
        logger.info("Dropped %d duplicate rows", before - after)

    # Drop rows with missing target
    if cfg.TARGET_COL in df.columns:
        missing_target = df[cfg.TARGET_COL].isnull().sum()
        if missing_target:
            logger.warning("Dropping %d rows with missing target", missing_target)
            df.dropna(subset=[cfg.TARGET_COL], inplace=True)

    logger.info("Clean data shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# 3. Type Conversions
# ---------------------------------------------------------------------------
@timer
def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw string columns to numeric/binary representations:

    * Yes/No → 1/0  for binary columns
    * Male/Female → 1/0
    * Churn Yes/No → 1/0
    * SeniorCitizen already 0/1

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    # gender: Male → 1, Female → 0
    if "gender" in df.columns:
        df["gender"] = (df["gender"].str.lower() == "male").astype(int)

    # Binary Yes/No columns
    for col in cfg.BINARY_YES_NO_COLS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0}).fillna(df[col])

    logger.info("Type conversions applied.")
    return df


# ---------------------------------------------------------------------------
# 4. Encoding + Scaling
# ---------------------------------------------------------------------------
@timer
def encode_and_scale(
    df: pd.DataFrame,
    fit: bool = True,
    encoder: OneHotEncoder | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, list[str], OneHotEncoder, StandardScaler]:
    """
    Apply One-Hot Encoding on categoricals and Standard Scaling on numerics.

    Parameters
    ----------
    df      : pd.DataFrame  — feature-engineered DataFrame (no target col)
    fit     : bool          — True during training, False during inference
    encoder : OneHotEncoder | None  — pre-fitted encoder for inference
    scaler  : StandardScaler | None — pre-fitted scaler for inference

    Returns
    -------
    tuple of:
        - pd.DataFrame       — fully encoded + scaled feature matrix
        - list[str]          — ordered feature column names
        - OneHotEncoder      — (fitted) encoder
        - StandardScaler     — (fitted) scaler
    """
    df = df.copy()

    # Identify which categorical columns are present
    cat_cols = [c for c in cfg.CATEGORICAL_COLS if c in df.columns]
    num_cols = [c for c in cfg.NUMERIC_COLS if c in df.columns]

    logger.info("Categorical columns for OHE : %s", cat_cols)
    logger.info("Numeric columns for scaling  : %s", num_cols)

    # Fill any remaining NaN in categoricals before encoding
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    if fit:
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            drop=None,
        )
        encoder.fit(df[cat_cols])

        scaler = StandardScaler()
        scaler.fit(df[num_cols])

    # Transform
    ohe_array = encoder.transform(df[cat_cols])
    ohe_cols = encoder.get_feature_names_out(cat_cols).tolist()
    ohe_df = pd.DataFrame(ohe_array, index=df.index, columns=ohe_cols)

    scaled_array = scaler.transform(df[num_cols])
    scaled_df = pd.DataFrame(scaled_array, index=df.index, columns=num_cols)

    # All remaining columns (binary ints, etc.)
    remaining_cols = [
        c for c in df.columns
        if c not in cat_cols and c not in num_cols
    ]
    remaining_df = df[remaining_cols].reset_index(drop=True)
    ohe_df = ohe_df.reset_index(drop=True)
    scaled_df = scaled_df.reset_index(drop=True)

    result = pd.concat([remaining_df, scaled_df, ohe_df], axis=1)
    feature_cols = result.columns.tolist()

    logger.info("Encoded + scaled shape: %s", result.shape)
    return result, feature_cols, encoder, scaler


# ---------------------------------------------------------------------------
# 5. Save Artifacts
# ---------------------------------------------------------------------------
def save_artifacts(
    encoder: OneHotEncoder,
    scaler: StandardScaler,
    feature_columns: list[str],
) -> None:
    """Persist encoder, scaler, and feature column list to models/."""
    paths.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(encoder, paths.ENCODER_FILE)
    logger.info("Encoder saved → %s", paths.ENCODER_FILE)

    joblib.dump(scaler, paths.SCALER_FILE)
    logger.info("Scaler saved  → %s", paths.SCALER_FILE)

    joblib.dump(feature_columns, paths.FEATURE_COLUMNS_FILE)
    logger.info("Feature cols  → %s", paths.FEATURE_COLUMNS_FILE)


# ---------------------------------------------------------------------------
# 6. Master Preprocessing Function
# ---------------------------------------------------------------------------
@timer
def run_preprocessing(
    filepath: Path | None = None,
    save_processed: bool = True,
    fit_transformers: bool = True,
    encoder: OneHotEncoder | None = None,
    scaler: StandardScaler | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], OneHotEncoder, StandardScaler]:
    """
    Full preprocessing pipeline: load → clean → convert → engineer → encode/scale.

    Parameters
    ----------
    filepath           : Path | None   — raw CSV path
    save_processed     : bool          — persist processed parquet/csv
    fit_transformers   : bool          — fit new encoder+scaler vs use provided
    encoder            : OneHotEncoder — optional pre-fitted encoder
    scaler             : StandardScaler — optional pre-fitted scaler

    Returns
    -------
    X              : pd.DataFrame  — feature matrix (ready for model)
    y              : pd.Series     — target (0/1)
    feature_cols   : list[str]     — ordered feature names
    encoder        : OneHotEncoder
    scaler         : StandardScaler
    """
    # --- Load ---
    df = load_raw_data(filepath)

    # --- Clean ---
    df = clean_data(df)

    # --- Type Conversion ---
    df = convert_types(df)

    # --- Feature Engineering ---
    df = engineer_features(df)

    # --- Separate target ---
    y = df[cfg.TARGET_COL].copy().astype(int)

    # --- Drop columns not used for modelling ---
    drop = [cfg.TARGET_COL] + [
        c for c in cfg.DROP_COLS if c in df.columns
    ]
    # Also drop any intermediate engineered categoricals not in CATEGORICAL_COLS
    X = df.drop(columns=drop, errors="ignore")

    # --- Encode + Scale ---
    X_encoded, feature_cols, encoder, scaler = encode_and_scale(
        X,
        fit=fit_transformers,
        encoder=encoder,
        scaler=scaler,
    )

    # Drop target if it crept into X_encoded
    X_encoded = X_encoded.drop(columns=[cfg.TARGET_COL], errors="ignore")
    feature_cols = [c for c in feature_cols if c != cfg.TARGET_COL]

    if save_processed:
        paths.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        X_encoded.to_parquet(paths.PROCESSED_DATA_FILE, index=False)
        X_encoded.to_csv(paths.PROCESSED_CSV_FILE, index=False)
        logger.info("Processed data saved → %s", paths.PROCESSED_DATA_FILE)

    if fit_transformers:
        save_artifacts(encoder, scaler, feature_cols)

    logger.info(
        "Preprocessing complete — X: %s | y distribution: %s",
        X_encoded.shape,
        y.value_counts().to_dict(),
    )
    return X_encoded, y, feature_cols, encoder, scaler


# ---------------------------------------------------------------------------
# 7. Inference Preprocessing (single row / batch)
# ---------------------------------------------------------------------------
def preprocess_input(
    input_df: pd.DataFrame,
    encoder: OneHotEncoder,
    scaler: StandardScaler,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Preprocess a raw customer DataFrame for inference.

    Applies the SAME transformations as training but using pre-fitted
    encoder and scaler (no refitting).

    Parameters
    ----------
    input_df        : pd.DataFrame     — raw customer data (1 or more rows)
    encoder         : OneHotEncoder    — fitted during training
    scaler          : StandardScaler   — fitted during training
    feature_columns : list[str]        — expected feature names in order

    Returns
    -------
    pd.DataFrame  — encoded, scaled, column-aligned feature matrix
    """
    df = input_df.copy()

    # Clean & convert
    df = clean_data(df)
    df = convert_types(df)
    df = engineer_features(df)

    # Drop target if present
    df = df.drop(columns=[cfg.TARGET_COL] + cfg.DROP_COLS, errors="ignore")

    # Encode + scale with pre-fitted transformers
    X_encoded, _, _, _ = encode_and_scale(
        df, fit=False, encoder=encoder, scaler=scaler
    )

    # Align columns to training schema
    X_aligned = X_encoded.reindex(columns=feature_columns, fill_value=0)
    return X_aligned
