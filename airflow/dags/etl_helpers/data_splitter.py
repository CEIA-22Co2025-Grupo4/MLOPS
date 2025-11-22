"""
Data Splitting Functions

Functions for preprocessing and splitting Chicago crime data into train/test sets.
"""

import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def preprocess_for_split(df):
    """
    Preprocess data before splitting into train/test.
    Removes duplicates and fills null values in location_description.

    Args:
        df (pd.DataFrame): Raw enriched dataframe

    Returns:
        pd.DataFrame: Preprocessed dataframe ready for splitting
    """
    logger.info("Preprocessing data for split...")

    try:
        # Get initial record count
        initial_count = len(df)
        logger.info(f"Initial dataset: {initial_count} records")

        # Remove duplicates
        df_clean = df.drop_duplicates()
        duplicates_removed = initial_count - len(df_clean)

        if duplicates_removed > 0:
            logger.info(
                f"Removed {duplicates_removed} duplicate records ({100 * duplicates_removed / initial_count:.2f}%)"
            )
        else:
            logger.info("No duplicates found")

        # Fill null values in location_description with 'UNKNOWN'
        if "location_description" in df_clean.columns:
            null_count = df_clean["location_description"].isna().sum()
            if null_count > 0:
                df_clean = df_clean.copy()
                df_clean["location_description"] = df_clean[
                    "location_description"
                ].fillna("UNKNOWN")
                logger.info(
                    f"Filled {null_count} null values in 'location_description' with 'UNKNOWN'"
                )
            else:
                logger.info("No null values found in 'location_description'")

        logger.info(f"Preprocessing completed: {len(df_clean)} records ready for split")
        return df_clean

    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise


def split_train_test(df, test_size=0.2, random_state=42, stratify_column="arrest"):
    """
    Split dataframe into stratified train and test sets.

    Args:
        df (pd.DataFrame): Preprocessed dataframe
        test_size (float): Proportion of data for test set (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        stratify_column (str): Column name to use for stratification (default: 'arrest')

    Returns:
        tuple: (train_df, test_df) DataFrames
    """
    logger.info(
        f"Splitting data: {100 * (1 - test_size):.0f}% train, {100 * test_size:.0f}% test"
    )

    try:
        # Check if stratify column exists
        if stratify_column not in df.columns:
            logger.warning(
                f"Stratify column '{stratify_column}' not found. Performing random split."
            )
            stratify_data = None
        else:
            stratify_data = df[stratify_column]

            # Log class distribution
            class_dist = stratify_data.value_counts()
            logger.info(f"Class distribution in '{stratify_column}':")
            for value, count in class_dist.items():
                logger.info(f"  {value}: {count} ({100 * count / len(df):.2f}%)")

        # Perform split
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=stratify_data
        )

        logger.info(
            f"Split completed: {len(train_df)} train records, {len(test_df)} test records"
        )

        # Verify stratification
        if stratify_data is not None:
            train_dist = train_df[stratify_column].value_counts(normalize=True)
            test_dist = test_df[stratify_column].value_counts(normalize=True)
            logger.info("Stratification verification:")
            for value in train_dist.index:
                logger.info(
                    f"  {value} - Train: {100 * train_dist[value]:.2f}%, Test: {100 * test_dist[value]:.2f}%"
                )

        return train_df, test_df

    except Exception as e:
        logger.error(f"Error in train/test split: {e}")
        raise
