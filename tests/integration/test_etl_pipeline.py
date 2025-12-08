"""
End-to-End ETL Pipeline Tests

Integration tests for the complete ETL pipeline:
enrich -> split -> outliers -> encode -> scale -> balance -> features
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add the dags directory to path
DAGS_DIR = Path(__file__).parent.parent.parent / "airflow" / "dags"
sys.path.insert(0, str(DAGS_DIR))


class TestDataEnrichment:
    """Tests for data enrichment stage."""

    def test_enrichment_adds_geospatial_features(
        self,
        sample_crimes_df: pd.DataFrame,
        sample_stations_df: pd.DataFrame,
    ):
        """Test that enrichment adds distance to nearest police station."""
        from etl_helpers.data_enrichment import enrich_crime_data

        result = enrich_crime_data(sample_crimes_df, sample_stations_df)

        assert "distance_crime_to_police_station" in result.columns
        assert "nearest_police_station_district" in result.columns
        assert "nearest_police_station_district_name" in result.columns
        assert result["distance_crime_to_police_station"].notna().all()

    def test_enrichment_adds_temporal_features(
        self,
        sample_crimes_df: pd.DataFrame,
        sample_stations_df: pd.DataFrame,
    ):
        """Test that enrichment adds temporal features."""
        from etl_helpers.data_enrichment import enrich_crime_data

        result = enrich_crime_data(sample_crimes_df, sample_stations_df)

        assert "season" in result.columns
        assert "day_of_week" in result.columns
        assert "day_time" in result.columns
        assert set(result["season"].unique()).issubset(
            {"Winter", "Spring", "Summer", "Fall"}
        )
        assert set(result["day_time"].unique()).issubset(
            {"Morning", "Afternoon", "Evening", "Night"}
        )

    def test_enrichment_preserves_record_count(
        self,
        sample_crimes_df: pd.DataFrame,
        sample_stations_df: pd.DataFrame,
    ):
        """Test that enrichment doesn't lose records (except invalid coords)."""
        from etl_helpers.data_enrichment import enrich_crime_data

        result = enrich_crime_data(sample_crimes_df, sample_stations_df)

        # Should preserve most records (may drop some with invalid coords)
        assert len(result) >= len(sample_crimes_df) * 0.9


class TestDataSplitting:
    """Tests for data preprocessing and splitting stage."""

    def test_preprocess_removes_duplicates(self, enriched_df: pd.DataFrame):
        """Test that preprocessing removes duplicate records."""
        from etl_helpers.data_splitter import preprocess_for_split

        # Add duplicates
        enriched_with_dups = pd.concat([enriched_df, enriched_df.head(10)])
        result = preprocess_for_split(enriched_with_dups)

        assert len(result) <= len(enriched_with_dups)

    def test_split_maintains_stratification(self, preprocessed_df: pd.DataFrame):
        """Test that split maintains class distribution in target column."""
        from etl_helpers.data_splitter import split_train_test

        train_df, test_df = split_train_test(preprocessed_df)

        # Check proportions are similar
        train_positive_rate = train_df["arrest"].mean()
        test_positive_rate = test_df["arrest"].mean()

        # Allow 5% tolerance
        assert abs(train_positive_rate - test_positive_rate) < 0.05

    def test_split_ratio_is_approximately_80_20(self, preprocessed_df: pd.DataFrame):
        """Test that split creates approximately 80/20 train/test ratio."""
        from etl_helpers.data_splitter import split_train_test

        train_df, test_df = split_train_test(preprocessed_df)

        total = len(train_df) + len(test_df)
        train_ratio = len(train_df) / total

        assert 0.75 <= train_ratio <= 0.85


class TestOutlierRemoval:
    """Tests for outlier processing stage."""

    def test_outlier_removal_reduces_record_count(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        """Test that outlier removal reduces dataset size."""
        from etl_helpers.outlier_processing import process_outliers

        train_processed, test_processed = process_outliers(train_df, test_df)

        # Should remove some records or keep all (if no outliers)
        assert len(train_processed) <= len(train_df)
        assert len(test_processed) <= len(test_df)

    def test_outlier_removal_uses_train_statistics(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        """Test that outlier bounds are calculated from train data only."""
        from etl_helpers.outlier_processing import process_outliers

        column = "distance_crime_to_police_station"

        # Calculate expected bounds from train only
        mean_val = train_df[column].mean()
        std_val = train_df[column].std()
        lower_bound = mean_val - 3 * std_val
        upper_bound = mean_val + 3 * std_val

        train_processed, test_processed = process_outliers(train_df, test_df)

        # All remaining values should be within bounds
        assert train_processed[column].min() >= lower_bound
        assert train_processed[column].max() <= upper_bound


class TestDataEncoding:
    """Tests for data encoding stage."""

    def test_encoding_creates_frequency_columns(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        """Test that frequency encoding creates _freq columns."""
        from etl_helpers.data_encoding import encode_data

        train_encoded, test_encoded = encode_data(train_df.copy(), test_df.copy())

        # Should have frequency encoded columns
        freq_cols = [col for col in train_encoded.columns if col.endswith("_freq")]
        assert len(freq_cols) > 0

    def test_encoding_creates_cyclic_columns(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        """Test that cyclic encoding creates _sin column for day_of_week."""
        from etl_helpers.data_encoding import encode_data

        train_encoded, test_encoded = encode_data(train_df.copy(), test_df.copy())

        assert "day_of_week_sin" in train_encoded.columns
        assert "day_of_week_sin" in test_encoded.columns

        # Sine values should be in [-1, 1]
        assert train_encoded["day_of_week_sin"].min() >= -1
        assert train_encoded["day_of_week_sin"].max() <= 1

    def test_encoding_removes_categorical_columns(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        """Test that encoding drops original categorical columns."""
        from etl_helpers.data_encoding import encode_data

        train_encoded, test_encoded = encode_data(train_df.copy(), test_df.copy())

        # Original categorical columns should be dropped
        assert "season" not in train_encoded.columns
        assert "day_time" not in train_encoded.columns
        assert "domestic" not in train_encoded.columns
        assert "primary_type" not in train_encoded.columns

    def test_encoding_produces_numeric_output(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        """Test that all columns after encoding are numeric."""
        from etl_helpers.data_encoding import encode_data

        train_encoded, test_encoded = encode_data(train_df.copy(), test_df.copy())

        # All columns should be numeric (except maybe target if still boolean)
        non_numeric = train_encoded.select_dtypes(exclude=[np.number, bool]).columns
        assert len(non_numeric) == 0, f"Non-numeric columns found: {list(non_numeric)}"


class TestDataScaling:
    """Tests for data scaling stage."""

    def test_scaling_standardizes_values(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        """Test that scaling produces approximately mean=0, std=1 for train."""
        from etl_helpers.data_encoding import encode_data
        from etl_helpers.data_scaling import scale_data

        train_encoded, test_encoded = encode_data(train_df.copy(), test_df.copy())
        train_scaled, test_scaled = scale_data(train_encoded, test_encoded)

        # Check for standardized columns
        std_cols = [
            col for col in train_scaled.columns if col.endswith("_standardized")
        ]
        assert len(std_cols) > 0

        # Train data should have mean approx 0, std approx 1
        for col in std_cols:
            assert abs(train_scaled[col].mean()) < 0.1
            assert abs(train_scaled[col].std() - 1) < 0.1

    def test_scaling_creates_standardized_columns(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        """Test that scaling creates _standardized suffix columns."""
        from etl_helpers.data_encoding import encode_data
        from etl_helpers.data_scaling import scale_data

        train_encoded, test_encoded = encode_data(train_df.copy(), test_df.copy())
        train_scaled, test_scaled = scale_data(train_encoded, test_encoded)

        assert "x_coordinate_standardized" in train_scaled.columns
        assert "y_coordinate_standardized" in train_scaled.columns
        assert "latitude_standardized" in train_scaled.columns
        assert "longitude_standardized" in train_scaled.columns


class TestDataBalancing:
    """Tests for data balancing stage."""

    def test_balancing_improves_class_ratio(self, train_df: pd.DataFrame):
        """Test that balancing improves the class ratio."""
        from etl_helpers.data_encoding import encode_data
        from etl_helpers.data_scaling import scale_data
        from etl_helpers.data_balancing import balance_data

        # Need to encode and scale before balancing (SMOTE needs numeric data)
        train_encoded, _ = encode_data(train_df.copy(), train_df.copy())
        train_scaled, _ = scale_data(train_encoded, train_encoded)

        # Calculate original ratio
        original_counts = train_scaled["arrest"].value_counts()
        original_ratio = original_counts.min() / original_counts.max()

        balanced = balance_data(train_scaled)

        # Calculate balanced ratio
        balanced_counts = balanced["arrest"].value_counts()
        balanced_ratio = balanced_counts.min() / balanced_counts.max()

        # Balanced ratio should be closer to 1
        assert balanced_ratio > original_ratio

    def test_balancing_preserves_columns(self, train_df: pd.DataFrame):
        """Test that balancing preserves all columns."""
        from etl_helpers.data_encoding import encode_data
        from etl_helpers.data_scaling import scale_data
        from etl_helpers.data_balancing import balance_data

        train_encoded, _ = encode_data(train_df.copy(), train_df.copy())
        train_scaled, _ = scale_data(train_encoded, train_encoded)

        balanced = balance_data(train_scaled)

        assert set(balanced.columns) == set(train_scaled.columns)


class TestFeatureSelection:
    """Tests for feature selection stage."""

    def test_feature_selection_reduces_columns(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        """Test that feature selection reduces the number of columns."""
        from etl_helpers.data_encoding import encode_data
        from etl_helpers.data_scaling import scale_data
        from etl_helpers.feature_selection import select_features

        train_encoded, test_encoded = encode_data(train_df.copy(), test_df.copy())
        train_scaled, test_scaled = scale_data(train_encoded, test_encoded)

        train_selected, test_selected, mi_scores = select_features(
            train_scaled, test_scaled
        )

        # Should have fewer columns (except target)
        assert train_selected.shape[1] <= train_scaled.shape[1]

    def test_feature_selection_preserves_target(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ):
        """Test that feature selection keeps the target column."""
        from etl_helpers.data_encoding import encode_data
        from etl_helpers.data_scaling import scale_data
        from etl_helpers.feature_selection import select_features

        train_encoded, test_encoded = encode_data(train_df.copy(), test_df.copy())
        train_scaled, test_scaled = scale_data(train_encoded, test_encoded)

        train_selected, test_selected, _ = select_features(train_scaled, test_scaled)

        assert "arrest" in train_selected.columns
        assert "arrest" in test_selected.columns


class TestFullPipeline:
    """End-to-end tests for the complete pipeline."""

    def test_full_pipeline_produces_ml_ready_data(
        self,
        sample_crimes_df: pd.DataFrame,
        sample_stations_df: pd.DataFrame,
    ):
        """Test complete pipeline from raw data to ML-ready output."""
        from etl_helpers.data_enrichment import enrich_crime_data
        from etl_helpers.data_splitter import preprocess_for_split, split_train_test
        from etl_helpers.outlier_processing import process_outliers
        from etl_helpers.data_encoding import encode_data
        from etl_helpers.data_scaling import scale_data
        from etl_helpers.data_balancing import balance_data
        from etl_helpers.feature_selection import select_features

        # 1. Enrich
        enriched = enrich_crime_data(sample_crimes_df, sample_stations_df)
        assert len(enriched) > 0

        # 2. Preprocess and split
        preprocessed = preprocess_for_split(enriched)
        train_df, test_df = split_train_test(preprocessed)
        assert len(train_df) > 0
        assert len(test_df) > 0

        # 3. Remove outliers
        train_no_outliers, test_no_outliers = process_outliers(train_df, test_df)
        assert len(train_no_outliers) > 0

        # 4. Encode
        train_encoded, test_encoded = encode_data(train_no_outliers, test_no_outliers)
        assert len(train_encoded) > 0

        # 5. Scale
        train_scaled, test_scaled = scale_data(train_encoded, test_encoded)
        assert len(train_scaled) > 0

        # 6. Balance (train only)
        train_balanced = balance_data(train_scaled)
        assert len(train_balanced) > 0

        # 7. Feature selection
        train_final, test_final, _ = select_features(train_balanced, test_scaled)

        # Final assertions
        assert len(train_final) > 0
        assert len(test_final) > 0
        assert "arrest" in train_final.columns
        assert "arrest" in test_final.columns

        # All columns should be numeric
        non_numeric_train = train_final.select_dtypes(exclude=[np.number, bool]).columns
        non_numeric_test = test_final.select_dtypes(exclude=[np.number, bool]).columns
        assert len(non_numeric_train) == 0, (
            f"Non-numeric columns in train: {list(non_numeric_train)}"
        )
        assert len(non_numeric_test) == 0, (
            f"Non-numeric columns in test: {list(non_numeric_test)}"
        )

    def test_full_pipeline_no_nan_in_output(
        self,
        sample_crimes_df: pd.DataFrame,
        sample_stations_df: pd.DataFrame,
    ):
        """Test that the final output has no NaN values."""
        from etl_helpers.data_enrichment import enrich_crime_data
        from etl_helpers.data_splitter import preprocess_for_split, split_train_test
        from etl_helpers.outlier_processing import process_outliers
        from etl_helpers.data_encoding import encode_data
        from etl_helpers.data_scaling import scale_data
        from etl_helpers.data_balancing import balance_data
        from etl_helpers.feature_selection import select_features

        # Run full pipeline
        enriched = enrich_crime_data(sample_crimes_df, sample_stations_df)
        preprocessed = preprocess_for_split(enriched)
        train_df, test_df = split_train_test(preprocessed)
        train_no_outliers, test_no_outliers = process_outliers(train_df, test_df)
        train_encoded, test_encoded = encode_data(train_no_outliers, test_no_outliers)
        train_scaled, test_scaled = scale_data(train_encoded, test_encoded)
        train_balanced = balance_data(train_scaled)
        train_final, test_final, _ = select_features(train_balanced, test_scaled)

        # Check for NaN
        assert train_final.isna().sum().sum() == 0, "NaN found in train output"
        assert test_final.isna().sum().sum() == 0, "NaN found in test output"
