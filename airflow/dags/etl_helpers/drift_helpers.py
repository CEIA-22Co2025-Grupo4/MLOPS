"""
Drift Monitoring Helpers

Functions for calculating data drift, prediction drift, and concept drift.
Uses PSI (Population Stability Index) and KS (Kolmogorov-Smirnov) tests.
"""

import logging

import numpy as np
import pandas as pd
import requests
from scipy.stats import ks_2samp

from drift_config import drift_config

logger = logging.getLogger(__name__)


def calculate_psi(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    """
    Calculate Population Stability Index between reference and current distributions.

    PSI < 0.1: No significant change
    PSI 0.1-0.2: Moderate change
    PSI > 0.2: Significant change

    Args:
        ref: Reference distribution (training data)
        cur: Current distribution (production data)
        bins: Number of bins for discretization

    Returns:
        PSI value (float)
    """
    try:
        ref_counts, bin_edges = pd.cut(ref, bins=bins, retbins=True)
        cur_counts = pd.cut(cur, bins=bin_edges)

        ref_dist = ref_counts.value_counts(normalize=True).sort_index()
        cur_dist = cur_counts.value_counts(normalize=True).sort_index()

        # Align indices
        all_bins = ref_dist.index.union(cur_dist.index)
        ref_dist = ref_dist.reindex(all_bins, fill_value=0)
        cur_dist = cur_dist.reindex(all_bins, fill_value=0)

        # Add small epsilon to avoid log(0)
        ref_dist = ref_dist + 1e-8
        cur_dist = cur_dist + 1e-8

        psi = ((cur_dist - ref_dist) * np.log(cur_dist / ref_dist)).sum()
        return float(psi)
    except Exception as e:
        logger.warning(f"PSI calculation failed: {e}")
        return 0.0


def ks_test(ref: pd.Series, cur: pd.Series) -> float:
    """
    Calculate Kolmogorov-Smirnov statistic between two distributions.

    The KS statistic measures the maximum distance between the cumulative
    distribution functions of two samples.

    Args:
        ref: Reference distribution
        cur: Current distribution

    Returns:
        KS statistic (float between 0 and 1)
    """
    try:
        return float(ks_2samp(ref.dropna(), cur.dropna()).statistic)
    except Exception as e:
        logger.warning(f"KS test failed: {e}")
        return 0.0


def compute_feature_drift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    target_column: str = None,
    prediction_column: str = None,
    psi_threshold: float = None,
    ks_threshold: float = None,
) -> pd.DataFrame:
    """
    Compute drift metrics for each feature in the dataset.

    Args:
        ref_df: Reference dataset (from training)
        cur_df: Current dataset (from production)
        target_column: Name of target column to exclude (default from config)
        prediction_column: Name of prediction column to exclude (default from config)
        psi_threshold: Threshold for PSI flag (default from config)
        ks_threshold: Threshold for KS flag (default from config)

    Returns:
        DataFrame with columns: feature, psi, ks, psi_flag, ks_flag
    """
    target_column = target_column or drift_config.TARGET_COLUMN
    prediction_column = prediction_column or drift_config.PREDICTION_COLUMN
    psi_threshold = (
        psi_threshold if psi_threshold is not None else drift_config.PSI_THRESHOLD
    )
    ks_threshold = (
        ks_threshold if ks_threshold is not None else drift_config.KS_THRESHOLD
    )

    drift_rows = []

    for col in ref_df.columns:
        if col in [target_column, prediction_column]:
            continue

        if col not in cur_df.columns:
            continue

        try:
            ref = ref_df[col].dropna().astype(float)
            cur = cur_df[col].dropna().astype(float)

            if len(ref) == 0 or len(cur) == 0:
                continue

            psi = calculate_psi(ref, cur)
            ks = ks_test(ref, cur)

            drift_rows.append(
                {
                    "feature": col,
                    "psi": psi,
                    "ks": ks,
                    "psi_flag": psi > psi_threshold,
                    "ks_flag": ks > ks_threshold,
                }
            )
        except Exception as e:
            logger.error(f"Error processing feature {col}: {e}")

    return pd.DataFrame(drift_rows)


def compute_prediction_drift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    prediction_column: str = None,
    psi_threshold: float = None,
    ks_threshold: float = None,
) -> pd.DataFrame:
    """
    Compute drift in model predictions between reference and current data.

    Args:
        ref_df: Reference dataset with predictions
        cur_df: Current dataset with predictions
        prediction_column: Name of prediction column (default from config)
        psi_threshold: Threshold for PSI flag (default from config)
        ks_threshold: Threshold for KS flag (default from config)

    Returns:
        DataFrame with PSI and KS metrics for predictions
    """
    prediction_column = prediction_column or drift_config.PREDICTION_COLUMN
    psi_threshold = (
        psi_threshold if psi_threshold is not None else drift_config.PSI_THRESHOLD
    )
    ks_threshold = (
        ks_threshold if ks_threshold is not None else drift_config.KS_THRESHOLD
    )

    if (
        prediction_column not in ref_df.columns
        or prediction_column not in cur_df.columns
    ):
        return pd.DataFrame(
            [
                {
                    "psi": None,
                    "ks": None,
                    "psi_flag": False,
                    "ks_flag": False,
                    "note": "Prediction column missing",
                }
            ]
        )

    ref = ref_df[prediction_column].dropna().astype(float)
    cur = cur_df[prediction_column].dropna().astype(float)

    psi = calculate_psi(ref, cur)
    ks = ks_test(ref, cur)

    return pd.DataFrame(
        [
            {
                "psi": psi,
                "ks": ks,
                "psi_flag": psi > psi_threshold,
                "ks_flag": ks > ks_threshold,
            }
        ]
    )


def compute_concept_drift(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    target_column: str = None,
    prediction_column: str = None,
    concept_drift_delta: float = None,
) -> pd.DataFrame:
    """
    Compute concept drift by comparing model accuracy across periods.

    Concept drift occurs when the relationship between features and target changes,
    even if the feature distributions remain the same.

    Args:
        ref_df: Reference dataset with labels and predictions
        cur_df: Current dataset with labels and predictions
        target_column: Name of target column (default from config)
        prediction_column: Name of prediction column (default from config)
        concept_drift_delta: Threshold for accuracy delta (default from config)

    Returns:
        DataFrame with accuracy comparison and drift flag
    """
    target_column = target_column or drift_config.TARGET_COLUMN
    prediction_column = prediction_column or drift_config.PREDICTION_COLUMN
    concept_drift_delta = (
        concept_drift_delta
        if concept_drift_delta is not None
        else drift_config.CONCEPT_DRIFT_DELTA
    )

    if target_column not in cur_df.columns or prediction_column not in cur_df.columns:
        return pd.DataFrame(
            [
                {
                    "concept_drift": None,
                    "flag": False,
                    "note": "Labels or predictions not available",
                }
            ]
        )

    if target_column not in ref_df.columns or prediction_column not in ref_df.columns:
        return pd.DataFrame(
            [
                {
                    "concept_drift": None,
                    "flag": False,
                    "note": "Reference labels not available",
                }
            ]
        )

    ref_acc = (ref_df[prediction_column] == ref_df[target_column]).mean()
    cur_acc = (cur_df[prediction_column] == cur_df[target_column]).mean()

    delta = ref_acc - cur_acc

    return pd.DataFrame(
        [
            {
                "ref_acc": ref_acc,
                "cur_acc": cur_acc,
                "delta": delta,
                "flag": abs(delta) > concept_drift_delta,
            }
        ]
    )


def call_batch_prediction_api(
    features_df: pd.DataFrame,
    api_url: str = None,
    timeout: int = 120,
) -> list:
    """
    Call the prediction API with a batch of features.

    Args:
        features_df: DataFrame with preprocessed features
        api_url: Base URL of the prediction API (default from env)
        timeout: Request timeout in seconds

    Returns:
        List of predictions (0 or 1)
    """
    import os

    api_url = api_url or os.getenv("PREDICTION_API_URL", "http://api:8800")

    try:
        instances = features_df.to_dict(orient="records")

        response = requests.post(
            f"{api_url}/predict/batch",
            json={"instances": instances},
            timeout=timeout,
        )
        response.raise_for_status()

        result = response.json()
        predictions = [int(p["prediction"]) for p in result["predictions"]]
        logger.info(f"Got {len(predictions)} predictions from API")
        return predictions

    except requests.exceptions.RequestException as e:
        logger.warning(f"API call failed: {e}. Using simulated predictions.")
        return simulate_predictions(features_df)


def simulate_predictions(features_df: pd.DataFrame) -> list:
    """
    Simulate predictions when API is unavailable.

    Uses a simple heuristic based on feature values. This is a fallback
    for testing when the prediction API is not running.

    Args:
        features_df: DataFrame with features

    Returns:
        List of simulated predictions (0 or 1)
    """
    if "primary_type_freq" in features_df.columns:
        probs = features_df["primary_type_freq"] * 2
        probs = probs.clip(0, 1)
        predictions = (probs > 0.5).astype(int).tolist()
    else:
        predictions = [0] * len(features_df)

    logger.info(f"Simulated {len(predictions)} predictions")
    return predictions
