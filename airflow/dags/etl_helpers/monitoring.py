"""
Monitoring and MLflow Logging Functions

Functions for monitoring ETL pipeline stages and logging metrics/artifacts to MLflow.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import tempfile
import os

logger = logging.getLogger(__name__)

# Configure MLflow tracking URI
try:
    import mlflow

    # Set to MLflow server running in Docker
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
except ImportError:
    logger.warning("MLflow not installed")
except Exception as e:
    logger.warning(f"Failed to configure MLflow: {e}")


# Generic helper functions
def _save_and_log_artifact(filepath, artifact_path):
    """Save file artifact to MLflow."""
    try:
        import mlflow

        mlflow.log_artifact(filepath, artifact_path)
    except ImportError:
        logger.warning("MLflow not installed, skipping artifact logging")


def _save_figure_and_log(fig, artifact_path, dpi=100):
    """Save matplotlib figure to temp file and log to MLflow.

    Args:
        fig: Matplotlib figure
        artifact_path: Full path including filename (e.g., 'charts/raw_data_overview.png')
        dpi: Resolution for saved image
    """
    import mlflow

    # Extract directory and filename from artifact_path
    if "/" in artifact_path:
        artifact_dir = "/".join(artifact_path.split("/")[:-1])
        filename = artifact_path.split("/")[-1]
    else:
        artifact_dir = None
        filename = artifact_path

    # Create temp file with the correct final name
    temp_dir = tempfile.mkdtemp()
    temp_filepath = os.path.join(temp_dir, filename)

    try:
        fig.savefig(temp_filepath, dpi=dpi, bbox_inches="tight")
        mlflow.log_artifact(temp_filepath, artifact_dir)
    finally:
        os.unlink(temp_filepath)
        os.rmdir(temp_dir)
        plt.close(fig)


def _get_value_distribution(series, normalize=True, as_percentage=True):
    """Get value distribution from a pandas Series."""
    dist = series.value_counts(normalize=normalize)
    if as_percentage:
        dist = dist * 100
    return dist.to_dict()


def _log_metrics(metrics_dict):
    """Log multiple metrics to MLflow."""
    try:
        import mlflow

        for key, value in metrics_dict.items():
            mlflow.log_metric(key, value)
    except ImportError:
        pass


def _log_params(params_dict):
    """Log multiple parameters to MLflow."""
    try:
        import mlflow

        for key, value in params_dict.items():
            mlflow.log_param(key, value)
    except ImportError:
        pass


def _create_bar_chart(
    values, labels, title, xlabel, ylabel, horizontal=False, figsize=(10, 6)
):
    """Create a bar chart."""
    fig, ax = plt.subplots(figsize=figsize)

    if horizontal:
        ax.barh(range(len(values)), values)
        ax.set_yticks(range(len(values)))
        ax.set_yticklabels(labels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.invert_yaxis()
    else:
        ax.bar(range(len(values)), values)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    ax.set_title(title)
    ax.grid(axis="y" if not horizontal else "x", alpha=0.3)
    plt.tight_layout()
    return fig


def _create_comparison_bar_chart(
    values1, values2, labels, label1, label2, title, xlabel, ylabel, figsize=(10, 6)
):
    """Create a side-by-side comparison bar chart."""
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, values1, width, label=label1, alpha=0.8)
    ax.bar(x + width / 2, values2, width, label=label2, alpha=0.8)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def log_raw_data_metrics(df, run_name="raw_data"):
    """
    Log raw data quality metrics and visualizations to MLflow.

    Args:
        df (pd.DataFrame): Raw crime data
        run_name (str): MLflow run name

    Returns:
        dict: Summary metrics
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow not installed, skipping monitoring")
        return {}

    logger.info(f"Logging raw data metrics to MLflow: {run_name}")

    with mlflow.start_run(run_name=run_name):
        # Basic metrics
        total_records = len(df)
        metrics = {
            "total_records": total_records,
            "total_columns": len(df.columns),
        }

        # Date range
        params = {}
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            start_date = df["date"].min()
            end_date = df["date"].max()
            params["start_date"] = start_date.strftime("%Y-%m-%d")
            params["end_date"] = end_date.strftime("%Y-%m-%d")
            metrics["days_coverage"] = (end_date - start_date).days

        # Missing values
        missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
        for col, pct in missing_pct.items():
            if pct > 0:
                metrics[f"missing_pct_{col}"] = pct

        # Arrest distribution
        if "arrest" in df.columns:
            arrest_dist = _get_value_distribution(df["arrest"])
            metrics["arrest_rate_pct"] = arrest_dist.get(True, 0)
            metrics["no_arrest_rate_pct"] = arrest_dist.get(False, 0)

        # Unique values for key columns
        key_columns = ["district", "primary_type", "location_description"]
        for col in key_columns:
            if col in df.columns:
                metrics[f"unique_{col}"] = df[col].nunique()

        # Log all metrics and params
        _log_metrics(metrics)
        _log_params(params)

        # Generate and log visualizations
        _log_raw_data_charts(df)

        logger.info("Raw data metrics logged successfully")
        return {
            "total_records": total_records,
            "arrest_rate": arrest_dist.get(True, 0) if "arrest" in df.columns else None,
        }


def _log_raw_data_charts(df):
    """Generate and log charts for raw data."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Chart 1: Top 10 districts
    if "district" in df.columns:
        district_counts = df["district"].value_counts().head(10)
        axes[0, 0].barh(range(len(district_counts)), district_counts.values)
        axes[0, 0].set_yticks(range(len(district_counts)))
        axes[0, 0].set_yticklabels(district_counts.index)
        axes[0, 0].set_xlabel("Crime Count")
        axes[0, 0].set_title("Top 10 Districts by Crime Count")
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(axis="x", alpha=0.3)

    # Chart 2: Daily crime trend
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        daily_counts = df.groupby(df["date"].dt.date).size()
        axes[0, 1].plot(daily_counts.index, daily_counts.values)
        axes[0, 1].set_xlabel("Date")
        axes[0, 1].set_ylabel("Crime Count")
        axes[0, 1].set_title("Daily Crime Trend")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].grid(axis="y", alpha=0.3)

    # Chart 3: Arrest distribution
    if "arrest" in df.columns:
        arrest_counts = df["arrest"].value_counts()
        axes[1, 0].pie(
            arrest_counts.values,
            labels=["No Arrest", "Arrest"],
            autopct="%1.1f%%",
            startangle=90,
        )
        axes[1, 0].set_title("Arrest vs No Arrest Distribution")

    # Chart 4: Top 10 crime types
    if "primary_type" in df.columns:
        type_counts = df["primary_type"].value_counts().head(10)
        axes[1, 1].barh(range(len(type_counts)), type_counts.values)
        axes[1, 1].set_yticks(range(len(type_counts)))
        axes[1, 1].set_yticklabels(type_counts.index)
        axes[1, 1].set_xlabel("Crime Count")
        axes[1, 1].set_title("Top 10 Crime Types")
        axes[1, 1].invert_yaxis()
        axes[1, 1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    _save_figure_and_log(fig, "charts/raw_data_overview.png")


def log_split_metrics(train_df, test_df, target_column="arrest", run_name="split"):
    """
    Log train/test split metrics to MLflow.

    Args:
        train_df (pd.DataFrame): Training data
        test_df (pd.DataFrame): Test data
        target_column (str): Target column name
        run_name (str): MLflow run name

    Returns:
        dict: Summary metrics
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow not installed, skipping monitoring")
        return {}

    logger.info(f"Logging split metrics to MLflow: {run_name}")

    with mlflow.start_run(run_name=run_name):
        # Basic metrics
        metrics = {
            "train_size": len(train_df),
            "test_size": len(test_df),
            "train_test_ratio": len(train_df) / (len(train_df) + len(test_df)),
        }

        # Class distribution
        if target_column in train_df.columns:
            train_dist = _get_value_distribution(train_df[target_column])
            test_dist = _get_value_distribution(test_df[target_column])

            # Add class metrics with prefixes
            for class_val, pct in train_dist.items():
                metrics[f"train_class_{class_val}_pct"] = pct
            for class_val, pct in test_dist.items():
                metrics[f"test_class_{class_val}_pct"] = pct

            # Generate comparison chart
            _log_split_chart(train_dist, test_dist, target_column)

        _log_metrics(metrics)

        logger.info("Split metrics logged successfully")
        return {
            "train_size": len(train_df),
            "test_size": len(test_df),
        }


def _log_split_chart(train_dist, test_dist, target_column):
    """Generate and log train/test distribution comparison chart."""
    train_values = [train_dist.get(0, 0), train_dist.get(1, 0)]
    test_values = [test_dist.get(0, 0), test_dist.get(1, 0)]

    fig = _create_comparison_bar_chart(
        train_values,
        test_values,
        labels=["Class 0", "Class 1"],
        label1="Train",
        label2="Test",
        title=f"Train/Test Class Distribution - {target_column}",
        xlabel="Class",
        ylabel="Percentage (%)",
    )

    _save_figure_and_log(fig, "charts/split_distribution.png")


def log_balance_metrics(
    original_df, balanced_df, target_column="arrest", run_name="balance"
):
    """
    Log data balancing metrics to MLflow.

    Args:
        original_df (pd.DataFrame): Original training data
        balanced_df (pd.DataFrame): Balanced training data
        target_column (str): Target column name
        run_name (str): MLflow run name

    Returns:
        dict: Summary metrics
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow not installed, skipping monitoring")
        return {}

    logger.info(f"Logging balance metrics to MLflow: {run_name}")

    with mlflow.start_run(run_name=run_name):
        # Size metrics
        metrics = {
            "original_size": len(original_df),
            "balanced_size": len(balanced_df),
            "size_change_pct": (len(balanced_df) - len(original_df))
            / len(original_df)
            * 100,
        }

        # Class distribution
        if target_column in original_df.columns:
            orig_dist = _get_value_distribution(original_df[target_column])
            bal_dist = _get_value_distribution(balanced_df[target_column])

            # Add class metrics with prefixes
            for class_val, pct in orig_dist.items():
                metrics[f"original_class_{class_val}_pct"] = pct
            for class_val, pct in bal_dist.items():
                metrics[f"balanced_class_{class_val}_pct"] = pct

            # Calculate ratio improvement
            orig_ratio = orig_dist.get(1, 0) / orig_dist.get(0, 1)
            bal_ratio = bal_dist.get(1, 0) / bal_dist.get(0, 1)
            metrics["original_class_ratio"] = orig_ratio
            metrics["balanced_class_ratio"] = bal_ratio

            # Generate comparison chart
            _log_balance_chart(orig_dist, bal_dist, target_column)

        _log_metrics(metrics)

        logger.info("Balance metrics logged successfully")
        return {
            "original_size": len(original_df),
            "balanced_size": len(balanced_df),
        }


def _log_balance_chart(orig_dist, bal_dist, target_column):
    """Generate and log before/after balancing comparison chart."""
    orig_values = [orig_dist.get(0, 0), orig_dist.get(1, 0)]
    bal_values = [bal_dist.get(0, 0), bal_dist.get(1, 0)]

    fig = _create_comparison_bar_chart(
        orig_values,
        bal_values,
        labels=["Class 0", "Class 1"],
        label1="Before Balance",
        label2="After Balance",
        title=f"Class Distribution Before/After Balancing - {target_column}",
        xlabel="Class",
        ylabel="Percentage (%)",
    )

    _save_figure_and_log(fig, "charts/balance_comparison.png")


def log_feature_selection_metrics(
    original_df,
    selected_df,
    mi_scores_df=None,
    target_column="arrest",
    run_name="features",
):
    """
    Log feature selection metrics to MLflow.

    Args:
        original_df (pd.DataFrame): Original dataframe before selection
        selected_df (pd.DataFrame): Dataframe after feature selection
        mi_scores_df (pd.DataFrame): Mutual information scores (optional)
        target_column (str): Target column name
        run_name (str): MLflow run name

    Returns:
        dict: Summary metrics
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow not installed, skipping monitoring")
        return {}

    logger.info(f"Logging feature selection metrics to MLflow: {run_name}")

    with mlflow.start_run(run_name=run_name):
        # Feature counts
        orig_features = [c for c in original_df.columns if c != target_column]
        selected_features = [c for c in selected_df.columns if c != target_column]
        dropped_features = list(set(orig_features) - set(selected_features))

        metrics = {
            "original_features": len(orig_features),
            "selected_features": len(selected_features),
            "dropped_features": len(dropped_features),
            "feature_reduction_pct": (len(dropped_features) / len(orig_features)) * 100
            if len(orig_features) > 0
            else 0,
        }

        params = {
            "selected_features_list": ",".join(selected_features),
            "dropped_features_list": ",".join(dropped_features),
        }

        _log_metrics(metrics)
        _log_params(params)

        # Generate visualizations
        if mi_scores_df is not None:
            _log_feature_importance_chart(mi_scores_df)

        _log_correlation_heatmap(selected_df, target_column)

        logger.info("Feature selection metrics logged successfully")
        return {
            "selected_features": len(selected_features),
            "dropped_features": len(dropped_features),
        }


def _log_feature_importance_chart(mi_scores_df):
    """Generate and log feature importance chart based on MI scores."""
    top_features = mi_scores_df.head(10)

    fig = _create_bar_chart(
        values=top_features["mi_score"].values,
        labels=top_features["feature"].values,
        title="Top 10 Features by Mutual Information",
        xlabel="Mutual Information Score",
        ylabel="Feature",
        horizontal=True,
        figsize=(10, 8),
    )

    _save_figure_and_log(fig, "charts/feature_importance.png")


def _log_correlation_heatmap(df, target_column):
    """Generate and log correlation heatmap of final features."""
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    if len(numeric_df.columns) < 2:
        logger.warning("Not enough numeric features for correlation heatmap")
        return

    fig, ax = plt.subplots(figsize=(12, 10))

    corr_matrix = numeric_df.corr()
    sns.heatmap(
        corr_matrix,
        annot=False,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Correlation"},
    )
    ax.set_title("Feature Correlation Heatmap (Final Dataset)")

    plt.tight_layout()
    _save_figure_and_log(fig, "charts/correlation_heatmap.png")


def log_pipeline_summary(
    raw_count,
    enriched_count,
    train_count,
    test_count,
    balanced_count,
    final_train_count,
    final_test_count,
    feature_count,
    run_name="pipeline_summary",
):
    """
    Log overall pipeline summary showing data flow through all stages.

    Args:
        raw_count (int): Number of raw records
        enriched_count (int): Number of enriched records (after cleaning)
        train_count (int): Number of training records after split
        test_count (int): Number of test records after split
        balanced_count (int): Number of records after balancing
        final_train_count (int): Number of final training records after feature selection
        final_test_count (int): Number of final test records after feature selection
        feature_count (int): Number of final features
        run_name (str): MLflow run name

    Returns:
        dict: Summary metrics
    """
    try:
        import mlflow
    except ImportError:
        logger.warning("MLflow not installed, skipping monitoring")
        return {}

    logger.info(f"Logging pipeline summary to MLflow: {run_name}")

    with mlflow.start_run(run_name=run_name):
        # Calculate retention rates
        metrics = {
            "raw_records": raw_count,
            "enriched_records": enriched_count,
            "train_records": train_count,
            "test_records": test_count,
            "balanced_records": balanced_count,
            "final_train_records": final_train_count,
            "final_test_records": final_test_count,
            "final_features": feature_count,
            "enrichment_retention_pct": (enriched_count / raw_count * 100)
            if raw_count > 0
            else 0,
            "balancing_change_pct": ((balanced_count - train_count) / train_count * 100)
            if train_count > 0
            else 0,
            "total_final_records": final_train_count + final_test_count,
            "overall_retention_pct": (
                (final_train_count + final_test_count) / raw_count * 100
            )
            if raw_count > 0
            else 0,
        }

        _log_metrics(metrics)

        # Create pipeline flow visualization
        _log_pipeline_flow_chart(
            raw_count,
            enriched_count,
            train_count,
            test_count,
            balanced_count,
            final_train_count,
            final_test_count,
        )

        logger.info("Pipeline summary logged successfully")
        return metrics


def _log_pipeline_flow_chart(
    raw_count,
    enriched_count,
    train_count,
    test_count,
    balanced_count,
    final_train_count,
    final_test_count,
):
    """Generate and log pipeline data flow chart."""
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define stages and counts
    stages = [
        "Raw\nData",
        "Enriched\nData",
        "Train\nSplit",
        "Balanced\nTrain",
        "Final\nTrain",
        "Final\nTest",
    ]
    counts = [
        raw_count,
        enriched_count,
        train_count,
        balanced_count,
        final_train_count,
        final_test_count,
    ]

    # Create positions for bars
    x_positions = [0, 1, 2.5, 3.5, 5, 5]
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c", "#9b59b6", "#1abc9c"]

    # Plot bars
    bars = ax.bar(x_positions, counts, color=colors, alpha=0.7, width=0.6)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Add stage labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stages, fontsize=11)

    # Add retention percentages between stages
    retention_texts = [
        f"{(enriched_count / raw_count * 100):.1f}%" if raw_count > 0 else "N/A",
        f"{(train_count / enriched_count * 100):.1f}%" if enriched_count > 0 else "N/A",
        f"{(balanced_count / train_count * 100):.1f}%" if train_count > 0 else "N/A",
        f"{(final_train_count / balanced_count * 100):.1f}%"
        if balanced_count > 0
        else "N/A",
    ]

    arrow_positions = [(0, 1), (1, 2.5), (2.5, 3.5), (3.5, 5)]

    for (start, end), text in zip(arrow_positions, retention_texts):
        mid_x = (start + end) / 2
        mid_y = max(counts) * 0.5
        ax.annotate(
            text,
            xy=(mid_x, mid_y),
            fontsize=9,
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Add test split annotation
    ax.annotate(
        f"Test: {test_count:,}\n({(test_count / enriched_count * 100):.1f}%)",
        xy=(2.5, train_count),
        xytext=(1.5, max(counts) * 0.3),
        fontsize=9,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.3),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", lw=1.5),
    )

    ax.set_ylabel("Record Count", fontsize=12, fontweight="bold")
    ax.set_title(
        "ETL Pipeline Data Flow - Record Count by Stage", fontsize=14, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, max(counts) * 1.15)

    plt.tight_layout()
    _save_figure_and_log(fig, "charts/pipeline_flow.png")
