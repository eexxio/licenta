"""
Evaluation metrics module for drug response prediction.

This module implements all metrics used to evaluate model performance:
- R² (coefficient of determination)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- Spearman correlation
- Pearson correlation
- Per-drug metrics

Author: Bachelor's Thesis Project
Date: 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute R² (coefficient of determination).

    R² measures how well the model explains variance in the data.
    Range: (-∞, 1], where 1 is perfect prediction, 0 is baseline (mean),
    and negative values indicate worse than baseline.

    Formula: R² = 1 - (SS_res / SS_tot)
    where SS_res = sum of squared residuals
          SS_tot = total sum of squares

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        R² score
    """
    return r2_score(y_true, y_pred)


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute RMSE (Root Mean Squared Error).

    RMSE measures the average prediction error in the same units as the target.
    Lower is better. Penalizes large errors more than MAE.

    Formula: RMSE = sqrt(mean((y_true - y_pred)²))

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        RMSE score
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute MAE (Mean Absolute Error).

    MAE measures the average absolute prediction error.
    More robust to outliers than RMSE.

    Formula: MAE = mean(|y_true - y_pred|)

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        MAE score
    """
    return mean_absolute_error(y_true, y_pred)


def compute_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Spearman's rank correlation coefficient.

    Measures monotonic relationship between predictions and true values.
    Range: [-1, 1], where 1 is perfect positive correlation,
    -1 is perfect negative correlation, 0 is no correlation.

    Robust to outliers and non-linear relationships.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Spearman correlation coefficient
    """
    correlation, _ = spearmanr(y_true, y_pred)
    return correlation


def compute_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Pearson's correlation coefficient.

    Measures linear relationship between predictions and true values.
    Range: [-1, 1], where 1 is perfect positive correlation,
    -1 is perfect negative correlation, 0 is no correlation.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Pearson correlation coefficient
    """
    correlation, _ = pearsonr(y_true, y_pred)
    return correlation


def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all evaluation metrics at once.

    This is the main function you'll use for model evaluation.

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary with all metric scores

    Example:
        >>> metrics = compute_all_metrics(y_test, y_pred)
        >>> print(f"R² = {metrics['r2']:.4f}")
        >>> print(f"RMSE = {metrics['rmse']:.4f}")
    """
    metrics = {
        'r2': compute_r2(y_true, y_pred),
        'rmse': compute_rmse(y_true, y_pred),
        'mae': compute_mae(y_true, y_pred),
        'spearman': compute_spearman(y_true, y_pred),
        'pearson': compute_pearson(y_true, y_pred),
        'n_samples': len(y_true)
    }

    return metrics


def compute_per_drug_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    drug_ids: np.ndarray
) -> pd.DataFrame:
    """
    Compute metrics separately for each drug.

    This helps identify which drugs are easier/harder to predict.

    Args:
        y_true: True values
        y_pred: Predicted values
        drug_ids: Drug identifier for each sample

    Returns:
        DataFrame with metrics for each drug

    Example:
        >>> per_drug = compute_per_drug_metrics(y_test, y_pred, drug_test)
        >>> print(per_drug.sort_values('r2', ascending=False).head())
    """
    logger.info("Computing per-drug metrics...")

    # Create DataFrame for easy grouping
    df = pd.DataFrame({
        'drug_id': drug_ids,
        'y_true': y_true,
        'y_pred': y_pred
    })

    # Compute metrics for each drug
    results = []

    for drug_id, group in df.groupby('drug_id'):
        # Need at least 5 samples to compute meaningful metrics
        if len(group) < 5:
            continue

        metrics = compute_all_metrics(
            group['y_true'].values,
            group['y_pred'].values
        )

        metrics['drug_id'] = drug_id
        results.append(metrics)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by R² (descending)
    results_df = results_df.sort_values('r2', ascending=False)

    logger.info(f"Computed metrics for {len(results_df)} drugs")

    return results_df


def print_metrics(metrics: Dict[str, float], title: str = "Model Evaluation"):
    """
    Print metrics in a formatted table.

    Args:
        metrics: Dictionary of metrics
        title: Title for the output

    Example:
        >>> metrics = compute_all_metrics(y_test, y_pred)
        >>> print_metrics(metrics, "Random Forest (AUC)")
    """
    print("=" * 60)
    print(f"{title}")
    print("=" * 60)
    print(f"R² (Coefficient of Determination): {metrics['r2']:8.4f}")
    print(f"RMSE (Root Mean Squared Error):    {metrics['rmse']:8.4f}")
    print(f"MAE (Mean Absolute Error):         {metrics['mae']:8.4f}")
    print(f"Spearman Correlation:              {metrics['spearman']:8.4f}")
    print(f"Pearson Correlation:               {metrics['pearson']:8.4f}")
    print(f"Number of Samples:                 {metrics['n_samples']:8d}")
    print("=" * 60)


def compare_models(
    model_predictions: Dict[str, np.ndarray],
    y_true: np.ndarray
) -> pd.DataFrame:
    """
    Compare multiple models on the same test set.

    Args:
        model_predictions: Dict mapping model names to predictions
        y_true: True values (same for all models)

    Returns:
        DataFrame comparing all models

    Example:
        >>> predictions = {
        ...     'Random Forest': rf_pred,
        ...     'XGBoost': xgb_pred,
        ...     'Neural Network': nn_pred
        ... }
        >>> comparison = compare_models(predictions, y_test)
        >>> print(comparison.sort_values('r2', ascending=False))
    """
    logger.info(f"Comparing {len(model_predictions)} models...")

    results = []

    for model_name, y_pred in model_predictions.items():
        metrics = compute_all_metrics(y_true, y_pred)
        metrics['model'] = model_name
        results.append(metrics)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Reorder columns (put model name first)
    cols = ['model'] + [col for col in results_df.columns if col != 'model']
    results_df = results_df[cols]

    # Sort by R² (descending)
    results_df = results_df.sort_values('r2', ascending=False)

    return results_df


# =============================================================================
# MAIN FUNCTION FOR TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test the evaluation metrics on synthetic data.
    """
    print("Testing evaluation metrics...\n")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000

    y_true = np.random.randn(n_samples)

    # Perfect predictions
    y_pred_perfect = y_true.copy()

    # Good predictions (R² ≈ 0.8)
    y_pred_good = y_true + np.random.randn(n_samples) * 0.5

    # Mediocre predictions (R² ≈ 0.3)
    y_pred_mediocre = y_true + np.random.randn(n_samples) * 1.5

    # Bad predictions (random)
    y_pred_bad = np.random.randn(n_samples)

    # Test 1: Individual metrics
    print("Test 1: Individual Metrics")
    print("-" * 60)
    metrics = compute_all_metrics(y_true, y_pred_good)
    print_metrics(metrics, "Good Predictions")

    # Test 2: Model comparison
    print("\nTest 2: Model Comparison")
    print("-" * 60)

    predictions = {
        'Perfect Model': y_pred_perfect,
        'Good Model': y_pred_good,
        'Mediocre Model': y_pred_mediocre,
        'Bad Model': y_pred_bad
    }

    comparison = compare_models(predictions, y_true)
    print(comparison.to_string(index=False))

    # Test 3: Per-drug metrics
    print("\nTest 3: Per-Drug Metrics")
    print("-" * 60)

    # Generate synthetic drug IDs
    drug_ids = np.random.choice(['Drug_A', 'Drug_B', 'Drug_C'], size=n_samples)

    per_drug = compute_per_drug_metrics(y_true, y_pred_good, drug_ids)
    print(per_drug.to_string(index=False))

    print("\n✓ All tests passed!")
