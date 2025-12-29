"""
Plotting functions for drug response prediction visualization.

This module provides reusable plotting functions for:
- Model comparison
- Predictions vs actual values
- Feature importance
- Learning curves
- Residual analysis

All plots use Romanian labels from config.py.

Author: Bachelor's Thesis Project
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

from src.config import ROMANIAN_LABELS


def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    target_name: str,
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot predicted vs actual values with R² score.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model
        target_name: Target variable name (IC50 or AUC)
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        Matplotlib figure object

    Example:
        >>> fig = plot_predictions_vs_actual(
        >>>     y_test, y_pred, "XGBoost", "AUC",
        >>>     save_path="results/figures/xgb_auc_predictions.png"
        >>> )
    """
    from sklearn.metrics import r2_score

    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'r--', linewidth=2, label='Predicție perfectă')

    ax.set_xlabel(f'{ROMANIAN_LABELS["actual_value"]} ({target_name})', fontsize=12)
    ax.set_ylabel(f'{ROMANIAN_LABELS["predicted_value"]} ({target_name})', fontsize=12)
    ax.set_title(f'{model_name} - {target_name}\nR² = {r2:.4f}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_model_comparison(
    model_results: Dict[str, Dict[str, float]],
    target_name: str,
    metrics: List[str] = ['r2', 'rmse', 'spearman'],
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot comparison of multiple models across different metrics.

    Args:
        model_results: Dictionary mapping model names to their metrics
                      Example: {'RandomForest': {'r2': 0.45, 'rmse': 0.23, ...}, ...}
        target_name: Target variable name (IC50 or AUC)
        metrics: List of metrics to plot
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        Matplotlib figure object

    Example:
        >>> results = {
        >>>     'RandomForest': {'r2': 0.45, 'rmse': 0.23, 'spearman': 0.65},
        >>>     'XGBoost': {'r2': 0.52, 'rmse': 0.20, 'spearman': 0.71},
        >>>     'NeuralNetwork': {'r2': 0.49, 'rmse': 0.21, 'spearman': 0.68}
        >>> }
        >>> fig = plot_model_comparison(results, "AUC")
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    model_names = list(model_results.keys())

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Get values for this metric
        values = [model_results[model][metric] for model in model_names]

        bars = ax.bar(model_names, values, edgecolor='black', alpha=0.7)

        # Highlight best model
        if metric in ['r2', 'spearman', 'pearson']:
            best_idx = np.argmax(values)
        else:  # Lower is better for RMSE, MAE
            best_idx = np.argmin(values)

        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)

        # Labels
        metric_label = ROMANIAN_LABELS.get(metric, metric.upper())
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add value labels on bars
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'{ROMANIAN_LABELS["model_comparison"]} - {target_name}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_feature_importance(
    importance: np.ndarray,
    feature_names: List[str],
    model_name: str,
    target_name: str,
    top_n: int = 30,
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot feature importance scores.

    Args:
        importance: Feature importance scores
        feature_names: Names of features
        model_name: Name of the model
        target_name: Target variable name
        top_n: Number of top features to display
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        Matplotlib figure object

    Example:
        >>> importance = model.get_feature_importance()
        >>> fig = plot_feature_importance(
        >>>     importance, gene_names, "XGBoost", "AUC",
        >>>     top_n=30
        >>> )
    """
    # Get top N features
    top_indices = np.argsort(importance)[-top_n:][::-1]
    top_importance = importance[top_indices]
    top_names = [feature_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
    y_pos = np.arange(len(top_names))

    ax.barh(y_pos, top_importance, edgecolor='black', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(ROMANIAN_LABELS['importance'], fontsize=12)
    ax.set_title(f'{ROMANIAN_LABELS["feature_importance"]} - {model_name} ({target_name})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_learning_curves(
    train_losses: List[float],
    val_losses: List[float],
    model_name: str,
    target_name: str,
    loss_name: str = "Loss",
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot training and validation loss curves.

    Args:
        train_losses: Training loss values per epoch/iteration
        val_losses: Validation loss values per epoch/iteration
        model_name: Name of the model
        target_name: Target variable name
        loss_name: Name of the loss metric
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        Matplotlib figure object

    Example:
        >>> fig = plot_learning_curves(
        >>>     model.train_losses,
        >>>     model.val_losses,
        >>>     "NeuralNetwork", "AUC"
        >>> )
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label='Antrenare', linewidth=2)
    ax.plot(epochs, val_losses, label='Validare', linewidth=2)

    # Mark best epoch (lowest validation loss)
    best_epoch = np.argmin(val_losses) + 1
    best_val_loss = min(val_losses)
    ax.axvline(best_epoch, color='red', linestyle='--', alpha=0.5,
               label=f'Cea mai bună epocă: {best_epoch}')
    ax.scatter([best_epoch], [best_val_loss], color='red', s=100, zorder=5)

    ax.set_xlabel('Epocă', fontsize=12)
    ax.set_ylabel(loss_name, fontsize=12)
    ax.set_title(f'{ROMANIAN_LABELS["learning_curve"]} - {model_name} ({target_name})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    target_name: str,
    save_path: Optional[str] = None,
    show: bool = False
) -> plt.Figure:
    """
    Plot residuals (prediction errors).

    Args:
        y_true: Actual values
        y_pred: Predicted values
        model_name: Name of the model
        target_name: Target variable name
        save_path: Path to save figure (optional)
        show: Whether to display the plot

    Returns:
        Matplotlib figure object

    Note:
        Residuals = y_true - y_pred
        A good model should have residuals centered around 0 with no pattern.

    Example:
        >>> fig = plot_residuals(y_test, y_pred, "XGBoost", "AUC")
    """
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Residual plot
    ax1.scatter(y_pred, residuals, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
    ax1.axhline(0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel(f'{ROMANIAN_LABELS["predicted_value"]} ({target_name})', fontsize=12)
    ax1.set_ylabel('Reziduu (Actual - Predicted)', fontsize=12)
    ax1.set_title(f'Grafic Reziduuri - {model_name}', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Histogram of residuals
    ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Reziduu', fontsize=12)
    ax2.set_ylabel('Frecvență', fontsize=12)
    ax2.set_title('Distribuție Reziduuri', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add statistics
    mean_residual = residuals.mean()
    std_residual = residuals.std()
    ax2.text(0.05, 0.95, f'Media: {mean_residual:.4f}\nStd: {std_residual:.4f}',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'{model_name} - {target_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# MAIN FUNCTION FOR TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test the plotting functions on synthetic data.
    """
    print("Testing visualization functions...\n")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 500
    y_true = np.random.randn(n_samples) * 2 + 5
    y_pred = y_true + np.random.randn(n_samples) * 0.5  # Add some noise

    # Test predictions vs actual
    print("1. Testing plot_predictions_vs_actual...")
    plot_predictions_vs_actual(
        y_true, y_pred, "TestModel", "AUC",
        save_path="test_predictions.png"
    )
    print("   ✓ Saved to: test_predictions.png")

    # Test model comparison
    print("\n2. Testing plot_model_comparison...")
    results = {
        'RandomForest': {'r2': 0.45, 'rmse': 0.23, 'spearman': 0.65},
        'XGBoost': {'r2': 0.52, 'rmse': 0.20, 'spearman': 0.71},
        'NeuralNetwork': {'r2': 0.49, 'rmse': 0.21, 'spearman': 0.68}
    }
    plot_model_comparison(results, "AUC", save_path="test_comparison.png")
    print("   ✓ Saved to: test_comparison.png")

    # Test feature importance
    print("\n3. Testing plot_feature_importance...")
    importance = np.random.rand(100)
    feature_names = [f"Gene_{i}" for i in range(100)]
    plot_feature_importance(
        importance, feature_names, "TestModel", "AUC",
        top_n=20, save_path="test_importance.png"
    )
    print("   ✓ Saved to: test_importance.png")

    # Test learning curves
    print("\n4. Testing plot_learning_curves...")
    train_losses = [1.0 - i*0.01 + np.random.rand()*0.05 for i in range(50)]
    val_losses = [1.1 - i*0.008 + np.random.rand()*0.05 for i in range(50)]
    plot_learning_curves(
        train_losses, val_losses, "TestModel", "AUC",
        save_path="test_learning.png"
    )
    print("   ✓ Saved to: test_learning.png")

    # Test residuals
    print("\n5. Testing plot_residuals...")
    plot_residuals(y_true, y_pred, "TestModel", "AUC", save_path="test_residuals.png")
    print("   ✓ Saved to: test_residuals.png")

    print("\n✓ All visualization functions tested successfully!")
    print("\nClean up test files:")
    print("  rm test_*.png")
