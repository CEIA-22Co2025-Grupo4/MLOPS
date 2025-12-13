"""
Pytest Configuration and Fixtures

Shared fixtures for ETL pipeline tests.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add the dags directory to path so we can import etl_helpers
DAGS_DIR = Path(__file__).parent.parent / "airflow" / "dags"
sys.path.insert(0, str(DAGS_DIR))


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_crimes_df(fixtures_dir: Path) -> pd.DataFrame:
    """Load sample crime data from CSV fixture."""
    csv_path = fixtures_dir / "sample_crimes.csv"
    return pd.read_csv(csv_path)


@pytest.fixture
def sample_stations_df(fixtures_dir: Path) -> pd.DataFrame:
    """Load sample police stations data from CSV fixture."""
    csv_path = fixtures_dir / "sample_stations.csv"
    return pd.read_csv(csv_path)


@pytest.fixture
def enriched_df(
    sample_crimes_df: pd.DataFrame, sample_stations_df: pd.DataFrame
) -> pd.DataFrame:
    """Return enriched crime data with geospatial and temporal features."""
    from etl_helpers.data_enrichment import enrich_crime_data

    return enrich_crime_data(sample_crimes_df, sample_stations_df)


@pytest.fixture
def preprocessed_df(enriched_df: pd.DataFrame) -> pd.DataFrame:
    """Return preprocessed data ready for splitting."""
    from etl_helpers.data_splitter import preprocess_for_split

    return preprocess_for_split(enriched_df)


@pytest.fixture
def train_test_split_dfs(preprocessed_df: pd.DataFrame) -> tuple:
    """Return train and test DataFrames after splitting."""
    from etl_helpers.data_splitter import split_train_test

    train_df, test_df = split_train_test(preprocessed_df)
    return train_df, test_df


@pytest.fixture
def train_df(train_test_split_dfs: tuple) -> pd.DataFrame:
    """Return training DataFrame."""
    return train_test_split_dfs[0]


@pytest.fixture
def test_df(train_test_split_dfs: tuple) -> pd.DataFrame:
    """Return test DataFrame."""
    return train_test_split_dfs[1]
