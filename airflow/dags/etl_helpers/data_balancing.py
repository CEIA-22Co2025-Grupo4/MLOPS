"""
Data Balancing Functions

Functions for balancing the training dataset using SMOTE and undersampling.
"""

import logging

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

from etl_helpers.config import (
    BALANCE_RANDOM_STATE,
    SMOTE_SAMPLING_STRATEGY,
    UNDERSAMPLE_STRATEGY,
)

logger = logging.getLogger(__name__)


def _validate_numeric_features(X):
    """
    Validate that all features are numeric (required for SMOTE).
    
    Raises:
        ValueError: If non-numeric columns are found
    """
    non_numeric_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if non_numeric_cols:
        logger.error(f"Non-numeric columns found: {non_numeric_cols}")
        raise ValueError(
            f"Cannot balance data with non-numeric columns: {non_numeric_cols}. "
            f"All features must be numeric for SMOTE."
        )
    
    nan_counts = X.isna().sum()
    cols_with_nan = nan_counts[nan_counts > 0]
    if len(cols_with_nan) > 0:
        logger.error(f"Columns with NaN values: {cols_with_nan.to_dict()}")
        raise ValueError(
            f"Cannot balance data with NaN values. "
            f"Found NaN in columns: {cols_with_nan.index.tolist()}"
        )


def _log_class_distribution(y, prefix=""):
    """Log class distribution statistics."""
    counts = y.value_counts()
    ratio = counts.min() / counts.max()
    
    class_0 = counts.loc[0] if 0 in counts.index else 0
    class_1 = counts.loc[1] if 1 in counts.index else 0
    total = len(y)
    
    logger.info(f"{prefix}Class distribution:")
    logger.info(f"  Class 0: {class_0} ({100 * class_0 / total:.1f}%)")
    logger.info(f"  Class 1: {class_1} ({100 * class_1 / total:.1f}%)")
    logger.info(f"  Ratio (min/max): {ratio:.3f}")
    
    return counts, ratio


def balance_data(train_df, target_column="arrest"):
    """
    Balance training data using combined SMOTE + RandomUnderSampler strategy.

    Strategy:
    1. SMOTE: Generate synthetic samples for minority class
    2. RandomUnderSampler: Reduce majority class

    Args:
        train_df: Training dataframe
        target_column: Target column name

    Returns:
        Balanced training dataframe
    """
    logger.info("Starting data balancing...")

    if target_column not in train_df.columns:
        raise ValueError(f"Target column '{target_column}' not found")

    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]

    _validate_numeric_features(X)
    _log_class_distribution(y, prefix="Original ")

    over = SMOTE(
        sampling_strategy=SMOTE_SAMPLING_STRATEGY,
        random_state=BALANCE_RANDOM_STATE
    )
    under = RandomUnderSampler(
        sampling_strategy=UNDERSAMPLE_STRATEGY,
        random_state=BALANCE_RANDOM_STATE
    )

    pipeline = Pipeline(steps=[("oversample", over), ("undersample", under)])

    logger.info("Applying SMOTE + RandomUnderSampler...")
    X_resampled, y_resampled = pipeline.fit_resample(X, y)

    _log_class_distribution(y_resampled, prefix="Balanced ")

    balanced_df = pd.concat([X_resampled, y_resampled], axis=1)
    logger.info(f"Balancing completed: {len(train_df)} -> {len(balanced_df)} samples")

    return balanced_df
