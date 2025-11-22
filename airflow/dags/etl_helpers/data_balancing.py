"""
Data Balancing Functions

Functions for balancing the training dataset using SMOTE and undersampling.
"""

import pandas as pd
import logging
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def balance_data(train_df, target_column="arrest"):
    """
    Balance training data using combined SMOTE + RandomUnderSampler strategy.

    Strategy:
    1. SMOTE (oversample to 0.5): Generate synthetic samples for minority class
       - Minority class will be 50% of majority class size
    2. RandomUnderSampler (undersample to 0.8): Reduce majority class
       - Final ratio: minority = 80% of majority

    Args:
        train_df (pd.DataFrame): Training dataframe
        target_column (str): Target column name (default: 'arrest')

    Returns:
        pd.DataFrame: Balanced training dataframe
    """
    logger.info("Starting data balancing...")

    # Separate features and target
    if target_column not in train_df.columns:
        logger.error(f"Target column '{target_column}' not found in DataFrame")
        raise ValueError(f"Target column '{target_column}' not found")

    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]

    # Log original class distribution
    original_counts = y.value_counts()
    original_ratio = original_counts.min() / original_counts.max()
    logger.info("Original class distribution:")
    logger.info(
        f"  Class 0: {original_counts.get(0, 0)} ({100 * original_counts.get(0, 0) / len(y):.1f}%)"
    )
    logger.info(
        f"  Class 1: {original_counts.get(1, 0)} ({100 * original_counts.get(1, 0) / len(y):.1f}%)"
    )
    logger.info(f"  Original ratio (min/max): {original_ratio:.3f}")

    # Create balancing pipeline
    # Step 1: SMOTE - oversample minority to 50% of majority
    over = SMOTE(sampling_strategy=0.5, random_state=17)

    # Step 2: RandomUnderSampler - reduce majority so minority = 80% of majority
    under = RandomUnderSampler(sampling_strategy=0.8, random_state=17)

    # Combine both steps
    steps = [("oversample", over), ("undersample", under)]
    pipeline = Pipeline(steps=steps)

    # Apply balancing
    logger.info("Applying SMOTE + RandomUnderSampler...")
    X_resampled, y_resampled = pipeline.fit_resample(X, y)

    # Log final class distribution
    final_counts = y_resampled.value_counts()
    final_ratio = final_counts.min() / final_counts.max()
    logger.info("Balanced class distribution:")
    logger.info(
        f"  Class 0: {final_counts.get(0, 0)} ({100 * final_counts.get(0, 0) / len(y_resampled):.1f}%)"
    )
    logger.info(
        f"  Class 1: {final_counts.get(1, 0)} ({100 * final_counts.get(1, 0) / len(y_resampled):.1f}%)"
    )
    logger.info(f"  Final ratio (min/max): {final_ratio:.3f}")

    # Combine back into DataFrame
    balanced_df = pd.concat([X_resampled, y_resampled], axis=1)

    logger.info(f"Balancing completed: {len(train_df)} → {len(balanced_df)} samples")

    return balanced_df
