"""
ETL Helper Modules for Chicago Crime Data Pipeline

Utility functions for:
- MinIO/S3 operations (minio)
- Data loading from Socrata API (data_loader)
- Data preprocessing and transformations
- Monitoring and MLflow logging (monitoring)
"""

import sys
import os

# Setup path once for all modules - enables importing etl_config from parent directory
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Re-export config for convenience
from etl_config import config  # noqa: E402
from drift_config import drift_config  # noqa: E402

# Re-export MinIO functions for backward compatibility
from .minio import (  # noqa: E402
    get_minio_client,
    set_bucket_lifecycle_policy,
    create_bucket_if_not_exists,
    upload_to_minio,
    download_from_minio,
    check_file_exists,
    list_objects,
    delete_object,
    download_to_dataframe,
    upload_from_dataframe,
    upload_json,
    download_json,
)

# Re-export monitoring functions for backward compatibility
from .monitoring import (  # noqa: E402
    log_raw_data_metrics,
    log_split_metrics,
    log_balance_metrics,
    log_feature_selection_metrics,
)

# Re-export exceptions
from .exceptions import (  # noqa: E402
    ETLError,
    DataValidationError,
    MinIOError,
    DataLoadError,
    EnrichmentError,
    FeatureEngineeringError,
)

# Re-export inference preprocessing functions
from .inference_preprocessing import (  # noqa: E402
    preprocess_for_inference,
    preprocess_raw_crimes,
    extract_model_features,
    MODEL_FEATURES,
)

# Re-export drift monitoring functions
from .drift_helpers import (  # noqa: E402
    calculate_psi,
    ks_test,
    compute_feature_drift,
    compute_prediction_drift,
    compute_concept_drift,
    call_batch_prediction_api,
    simulate_predictions,
)

__version__ = "2.0.0"

__all__ = [
    # Config
    "config",
    "drift_config",
    # MinIO
    "get_minio_client",
    "set_bucket_lifecycle_policy",
    "create_bucket_if_not_exists",
    "upload_to_minio",
    "download_from_minio",
    "check_file_exists",
    "list_objects",
    "delete_object",
    "download_to_dataframe",
    "upload_from_dataframe",
    "upload_json",
    "download_json",
    # Monitoring
    "log_raw_data_metrics",
    "log_split_metrics",
    "log_balance_metrics",
    "log_feature_selection_metrics",
    # Exceptions
    "ETLError",
    "DataValidationError",
    "MinIOError",
    "DataLoadError",
    "EnrichmentError",
    "FeatureEngineeringError",
    # Inference
    "preprocess_for_inference",
    "preprocess_raw_crimes",
    "extract_model_features",
    "MODEL_FEATURES",
    # Drift monitoring
    "calculate_psi",
    "ks_test",
    "compute_feature_drift",
    "compute_prediction_drift",
    "compute_concept_drift",
    "call_batch_prediction_api",
    "simulate_predictions",
]
