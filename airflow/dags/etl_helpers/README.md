# ETL Helpers

Modular ETL components for the Chicago Crime data pipeline.

## Architecture

```
etl_process_taskflow.py (DAG)
         │
         ├── config.py              # Centralized configuration
         ├── minio_utils.py         # S3/MinIO operations
         ├── data_loader.py         # Socrata API client
         ├── data_enrichment.py     # Geospatial features
         ├── data_splitter.py       # Train/test split
         ├── outlier_processing.py  # Outlier removal
         ├── data_encoding.py       # Feature encoding
         ├── data_scaling.py        # Normalization
         ├── data_balancing.py      # SMOTE balancing
         ├── feature_selection.py   # MI-based selection
         └── monitoring.py          # MLflow logging
```

## Modules

### config.py

Centralized configuration with environment variable overrides.

```python
from etl_helpers.config import (
    BUCKET_NAME,           # MinIO bucket (default: "data")
    PREFIXES,              # Storage prefixes for each stage
    TARGET_COLUMN,         # Target variable (default: "arrest")
    SPLIT_TEST_SIZE,       # Test split ratio (default: 0.2)
    MI_THRESHOLD,          # Feature selection threshold (default: 0.05)
)
```

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_REPO_BUCKET_NAME` | `data` | MinIO bucket name |
| `LIFECYCLE_TTL_DAYS` | `60` | Data retention days |
| `ROLLING_WINDOW_DAYS` | `365` | Rolling window for data |
| `TARGET_COLUMN` | `arrest` | Target variable |
| `SPLIT_TEST_SIZE` | `0.2` | Test set proportion |
| `SPLIT_RANDOM_STATE` | `42` | Random seed for split |
| `OUTLIER_STD_THRESHOLD` | `3` | Std devs for outlier removal |
| `MI_THRESHOLD` | `0.05` | Mutual Info threshold |
| `SMOTE_SAMPLING_STRATEGY` | `0.5` | SMOTE ratio |
| `UNDERSAMPLE_STRATEGY` | `0.8` | Undersampling ratio |

### minio_utils.py

S3-compatible storage operations.

```python
from etl_helpers.minio_utils import (
    create_bucket_if_not_exists,
    upload_from_dataframe,
    download_to_dataframe,
    check_file_exists,
    list_objects,
)

# Upload DataFrame
upload_from_dataframe(df, "data", "prefix/file.csv")

# Download DataFrame
df = download_to_dataframe("data", "prefix/file.csv")
```

### data_loader.py

Chicago Data Portal (Socrata) API client.

```python
from etl_helpers.data_loader import (
    download_crimes_incremental,
    download_police_stations,
)

# Download crimes for date range
df = download_crimes_incremental(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 12, 31),
    output_file="/tmp/crimes.csv"
)

# Download police stations
download_police_stations(output_file="/tmp/stations.csv")
```

### data_enrichment.py

Adds geospatial and temporal features.

```python
from etl_helpers.data_enrichment import enrich_crime_data

enriched_df = enrich_crime_data(crimes_df, stations_df)
```

**Added Features:**
- `distance_crime_to_police_station`: Distance to nearest station
- `season`: Winter/Spring/Summer/Fall
- `day_of_week`: 0-6 (Monday-Sunday)
- `day_time`: Morning/Afternoon/Evening/Night

### data_splitter.py

Train/test split with preprocessing.

```python
from etl_helpers.data_splitter import preprocess_for_split, split_train_test

# Clean and prepare data
df_clean = preprocess_for_split(df)

# Stratified split
train_df, test_df = split_train_test(
    df_clean,
    test_size=0.2,
    random_state=42,
    stratify_column="arrest"
)
```

### outlier_processing.py

Outlier removal using standard deviation method.

```python
from etl_helpers.outlier_processing import process_outliers

train_clean, test_clean = process_outliers(
    train_df, 
    test_df, 
    n_std=3  # Remove values beyond ±3σ
)
```

### data_encoding.py

Feature encoding transformations.

```python
from etl_helpers.data_encoding import encode_data

train_encoded, test_encoded = encode_data(train_df, test_df)
```

**Encoding Methods:**
- **Log transform**: `distance_crime_to_police_station`
- **Cyclic**: `day_of_week` → sin/cos
- **One-hot**: `season`, `day_time`
- **Label**: `domestic`
- **Frequency**: `primary_type`, `location_description`, `beat`, `ward`, `community_area`

### data_scaling.py

Standardization (zero mean, unit variance).

```python
from etl_helpers.data_scaling import scale_data

train_scaled, test_scaled = scale_data(train_df, test_df)
```

**Scaled Features:**
- `x_coordinate`
- `y_coordinate`
- `latitude`
- `longitude`
- `distance_crime_to_police_station`

### data_balancing.py

Class balancing using SMOTE + RandomUnderSampler.

```python
from etl_helpers.data_balancing import balance_data

train_balanced = balance_data(
    train_df,
    target_column="arrest"
)
```

**Strategy:**
1. SMOTE: Oversample minority to 50% of majority
2. RandomUnderSampler: Final ratio 80% (minority = 80% of majority)

### feature_selection.py

Mutual Information based feature selection.

```python
from etl_helpers.feature_selection import select_features

train_selected, test_selected, mi_scores = select_features(
    train_df,
    test_df,
    target_column="arrest",
    mi_threshold=0.05
)
```

### monitoring.py

MLflow logging functions.

```python
from etl_helpers.monitoring import (
    log_raw_data_metrics,
    log_split_metrics,
    log_balance_metrics,
    log_feature_selection_metrics,
    log_pipeline_summary,
)

# Log data quality metrics
log_raw_data_metrics(df, run_name="raw_data_2024-01-01")

# Log split statistics
log_split_metrics(train_df, test_df, target_column="arrest")

# Log pipeline overview
log_pipeline_summary(
    raw_count=500000,
    enriched_count=498000,
    train_count=398400,
    test_count=99600,
    ...
)
```

## Data Flow

```
Raw Data (Socrata API)
    │
    ▼ download_data
0-raw-data/
    │
    ▼ merge_data  
1-merged-data/
    │
    ▼ enrich_data (+ distance, temporal features)
2-enriched-data/
    │
    ▼ split_data (80/20 stratified)
3-split-data/
    │
    ▼ process_outliers (±3σ removal)
4-outliers/
    │
    ▼ encode_data (frequency, cyclic, one-hot)
5-encoded/
    │
    ▼ scale_data (StandardScaler)
6-scaled/
    │
    ▼ balance_data (SMOTE + undersampling)
7-balanced/
    │
    ▼ extract_features (MI selection)
ml-ready-data/  ← FINAL OUTPUT
```

## Testing

Run individual modules:

```python
# In Airflow CLI container
python -c "
from etl_helpers.config import BUCKET_NAME, PREFIXES
print(f'Bucket: {BUCKET_NAME}')
print(f'Prefixes: {PREFIXES}')
"
```

## Adding New Modules

1. Create new file in `etl_helpers/`
2. Add import to `__init__.py`
3. Import in `etl_process_taskflow.py`
4. Create new `@task.python` function in DAG

