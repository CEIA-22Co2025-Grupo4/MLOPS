"""
Drift Monitoring Configuration

Centralized configuration for the drift monitoring pipeline.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DriftConfig:
    """Drift monitoring configuration parameters."""

    # Storage prefixes
    PREFIX_REFERENCE: str = "drift/reference/"
    PREFIX_CURRENT: str = "drift/current/"
    PREFIX_RESULTS: str = "drift/results/"

    # Column names
    TARGET_COLUMN: str = "arrest"
    PREDICTION_COLUMN: str = "prediction"

    # Drift thresholds
    # PSI < 0.1: No significant change
    # PSI 0.1-0.2: Moderate change
    # PSI > 0.2: Significant change
    PSI_THRESHOLD: float = 0.2
    KS_THRESHOLD: float = 0.1
    CONCEPT_DRIFT_DELTA: float = 0.05

    # Schedule parameters (defaults for production)
    # Days of data to fetch for drift analysis (matches weekly schedule)
    DRIFT_WINDOW_DAYS: int = 7
    # Chicago Data Portal has a delay in publishing data (typically 3-7 days)
    DATA_DELAY_DAYS: int = 7

    # Minimum delay for testing (Chicago usually has at least 1-2 days delay)
    MIN_DATA_DELAY_DAYS: int = 2

    # API configuration (can be overridden by environment variables)
    @property
    def api_url(self) -> str:
        return os.getenv("PREDICTION_API_URL", "http://api:8800")

    @property
    def bucket_name(self) -> str:
        return os.getenv("DATA_REPO_BUCKET_NAME", "data")


# Singleton instance
drift_config = DriftConfig()
