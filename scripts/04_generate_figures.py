"""
Script to generate all thesis figures.

This script generates 15+ publication-quality figures:
- Data exploration figures (Chapter 2)
- Model comparison figures (Chapter 5)
- Predictions vs actual (Chapter 5)
- Feature importance (Chapter 5)
- Learning curves (Chapter 5)

All figures use Romanian labels and are saved at 300 DPI.

Prerequisites:
    - Run scripts/01_preprocess_data.py
    - Run scripts/02_train_models.py
    - Run scripts/03_evaluate_models.py

Usage:
    python scripts/04_generate_figures.py

Author: Bachelor's Thesis Project
Date: 2026
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    ROMANIAN_LABELS,
    get_experiment_dir
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def setup_figure_dir():
    """Create figures directory."""
    figures_dir = Path(RESULTS_DIR) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def plot_target_distributions(y_train_ic50, y_train_auc, figures_dir):
    """
    Generate histograms of IC50 and AUC distributions.

    Chapter 2 figures.
    """
    logger.info("\nGenerating target distribution figures...")

    # IC50 distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_train_ic50, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel(ROMANIAN_LABELS['ic50'], fontsize=12)
    ax.set_ylabel(ROMANIAN_LABELS['frequency'], fontsize=12)
    ax.set_title(ROMANIAN_LABELS['ic50_distribution'], fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_ic50 = y_train_ic50.mean()
    std_ic50 = y_train_ic50.std()
    ax.axvline(mean_ic50, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_ic50:.2f}')
    ax.legend()

    plt.tight_layout()
    ic50_path = figures_dir / "ch2_fig1_ic50_distribution.png"
    plt.savefig(ic50_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved: {ic50_path}")

    # AUC distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(y_train_auc, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.set_xlabel(ROMANIAN_LABELS['auc'], fontsize=12)
    ax.set_ylabel(ROMANIAN_LABELS['frequency'], fontsize=12)
    ax.set_title(ROMANIAN_LABELS['auc_distribution'], fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add statistics
    mean_auc = y_train_auc.mean()
    std_auc = y_train_auc.std()
    ax.axvline(mean_auc, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_auc:.2f}')
    ax.legend()

    plt.tight_layout()
    auc_path = figures_dir / "ch2_fig2_auc_distribution.png"
    plt.savefig(auc_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved: {auc_path}")


def plot_samples_per_drug(drug_ids, figures_dir):
    """
    Generate bar chart of sample counts per drug.

    Chapter 2 figure.
    """
    logger.info("\nGenerating samples per drug figure...")

    drug_counts = pd.Series(drug_ids).value_counts().head(20)

    fig, ax = plt.subplots(figsize=(12, 6))
    drug_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_xlabel(ROMANIAN_LABELS['drug_id'], fontsize=12)
    ax.set_ylabel(ROMANIAN_LABELS['n_samples'], fontsize=12)
    ax.set_title(ROMANIAN_LABELS['samples_per_drug'] + ' (Top 20)', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    samples_path = figures_dir / "ch2_fig3_samples_per_drug.png"
    plt.savefig(samples_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved: {samples_path}")


def plot_model_comparison(metrics_df, target_name, figures_dir):
    """
    Generate bar chart comparing models.

    Chapter 5 figure.
    """
    logger.info(f"\nGenerating model comparison figure ({target_name})...")

    # Filter for specific target
    df = metrics_df[metrics_df['target'] == target_name].copy()

    # Metrics to plot
    metrics_to_plot = ['r2', 'rmse', 'spearman']
    metric_labels = [ROMANIAN_LABELS['r2'], ROMANIAN_LABELS['rmse'], ROMANIAN_LABELS['spearman']]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
        ax = axes[idx]
        models = df['model'].values
        values = df[metric].values

        bars = ax.bar(models, values, edgecolor='black', alpha=0.7)

        # Color best model
        best_idx = np.argmax(values) if metric in ['r2', 'spearman'] else np.argmin(values)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)

        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add value labels
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'{ROMANIAN_LABELS["model_comparison"]} - {target_name}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    filename = f"ch5_fig{'1' if target_name == 'AUC' else '2'}_model_comparison_{target_name.lower()}.png"
    comparison_path = figures_dir / filename
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved: {comparison_path}")


def plot_predictions_vs_actual(predictions_df, model_name, target_name, figures_dir):
    """
    Generate scatter plot of predictions vs actual values.

    Chapter 5 figure.
    """
    logger.info(f"\nGenerating predictions vs actual ({model_name}, {target_name})...")

    y_true = predictions_df['y_true'].values
    y_pred = predictions_df[f'{model_name}_pred'].values

    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.5, s=20, edgecolors='black', linewidth=0.5)

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicție perfectă')

    ax.set_xlabel(f'{ROMANIAN_LABELS["actual_value"]} ({target_name})', fontsize=12)
    ax.set_ylabel(f'{ROMANIAN_LABELS["predicted_value"]} ({target_name})', fontsize=12)
    ax.set_title(f'{model_name} - {target_name}\nR² = {r2:.4f}',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    model_short = model_name.lower().replace(' ', '_')
    filename = f"ch5_fig3_predictions_{model_short}_{target_name.lower()}.png"
    pred_path = figures_dir / filename
    plt.savefig(pred_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved: {pred_path}")


def plot_feature_importance(model, model_name, target_name, gene_names, figures_dir):
    """
    Generate bar chart of top important features.

    Chapter 5 figure.
    """
    logger.info(f"\nGenerating feature importance ({model_name}, {target_name})...")

    # Get feature importance
    importance = model.get_feature_importance()

    if importance is None:
        logger.warning(f"  ⚠ {model_name} does not provide feature importance")
        return

    # Get top 30 features
    top_indices = np.argsort(importance)[-30:][::-1]
    top_importance = importance[top_indices]
    top_genes = [gene_names[i] if i < len(gene_names) else f'Feature {i}'
                 for i in top_indices]

    fig, ax = plt.subplots(figsize=(10, 12))
    y_pos = np.arange(len(top_genes))

    ax.barh(y_pos, top_importance, edgecolor='black', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_genes, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(ROMANIAN_LABELS['importance'], fontsize=12)
    ax.set_title(f'{ROMANIAN_LABELS["feature_importance"]} - {model_name} ({target_name})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()

    model_short = model_name.lower().replace(' ', '_')
    filename = f"ch5_fig5_importance_{model_short}_{target_name.lower()}.png"
    importance_path = figures_dir / filename
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved: {importance_path}")


def plot_learning_curves(exp_name, model_name, target_name, figures_dir):
    """
    Generate learning curves for models with training history.

    Chapter 5 figure.
    """
    logger.info(f"\nGenerating learning curves ({model_name}, {target_name})...")

    exp_dir = get_experiment_dir(exp_name)
    history_path = exp_dir / "training_history.json"

    if not history_path.exists():
        logger.warning(f"  ⚠ No training history found for {model_name}")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))

    if 'train_losses' in history:
        # Neural Network format
        epochs = range(1, len(history['train_losses']) + 1)
        ax.plot(epochs, history['train_losses'], label='Antrenare', linewidth=2)
        ax.plot(epochs, history['val_losses'], label='Validare', linewidth=2)
        ax.set_xlabel('Epocă', fontsize=12)
        ax.set_ylabel('Loss (MSE)', fontsize=12)

    else:
        # XGBoost format
        if 'validation_0' in history and 'rmse' in history['validation_0']:
            iterations = range(1, len(history['validation_0']['rmse']) + 1)
            ax.plot(iterations, history['validation_0']['rmse'], label='Antrenare', linewidth=2)
            if 'validation_1' in history:
                ax.plot(iterations, history['validation_1']['rmse'], label='Validare', linewidth=2)
            ax.set_xlabel('Iterație', fontsize=12)
            ax.set_ylabel('RMSE', fontsize=12)

    ax.set_title(f'{ROMANIAN_LABELS["learning_curve"]} - {model_name} ({target_name})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    model_short = model_name.lower().replace(' ', '_')
    filename = f"ch5_fig6_learning_{model_short}_{target_name.lower()}.png"
    learning_path = figures_dir / filename
    plt.savefig(learning_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved: {learning_path}")


def plot_per_drug_performance(per_drug_df, model_name, target_name, figures_dir):
    """
    Generate heatmap of per-drug R² scores.

    Chapter 5 figure.
    """
    logger.info(f"\nGenerating per-drug performance ({model_name}, {target_name})...")

    # Sort by R² and get top/bottom 20
    df_sorted = per_drug_df.sort_values('r2', ascending=False)
    top_20 = df_sorted.head(20)
    bottom_20 = df_sorted.tail(20)
    selected_drugs = pd.concat([top_20, bottom_20])

    fig, ax = plt.subplots(figsize=(10, 12))

    drug_ids = selected_drugs['drug_id'].values
    r2_values = selected_drugs['r2'].values

    colors = ['green' if r2 > 0.5 else 'orange' if r2 > 0.3 else 'red' for r2 in r2_values]

    y_pos = np.arange(len(drug_ids))
    bars = ax.barh(y_pos, r2_values, color=colors, edgecolor='black', alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(drug_ids, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('R²', fontsize=12)
    ax.set_title(f'{ROMANIAN_LABELS["per_drug_performance"]} - {model_name} ({target_name})\n(Top/Bottom 20)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    ax.axvline(0.3, color='orange', linestyle='--', alpha=0.5)
    ax.axvline(0.5, color='green', linestyle='--', alpha=0.5)

    plt.tight_layout()

    model_short = model_name.lower().replace(' ', '_')
    filename = f"ch5_fig4_per_drug_{model_short}_{target_name.lower()}.png"
    per_drug_path = figures_dir / filename
    plt.savefig(per_drug_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ Saved: {per_drug_path}")


def main():
    """Main figure generation pipeline."""

    logger.info("=" * 80)
    logger.info("GENERATING ALL THESIS FIGURES")
    logger.info("=" * 80)

    # Setup figure directory
    figures_dir = setup_figure_dir()
    logger.info(f"\nFigures will be saved to: {figures_dir}")

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: LOADING DATA")
    logger.info("=" * 80)

    # Load training data (for distributions)
    train_path = Path(PROCESSED_DATA_DIR) / "train_data.npz"
    train_data = np.load(train_path)
    y_train_ic50 = train_data['y_train_ic50']
    y_train_auc = train_data['y_train_auc']
    drug_ids_train = train_data['drug_ids_train']
    gene_names = train_data['selected_genes']

    # Load metrics
    metrics_path = Path(RESULTS_DIR) / "metrics" / "model_comparison.csv"
    metrics_df = pd.read_csv(metrics_path)

    # Load predictions
    ic50_pred_path = Path(RESULTS_DIR) / "predictions" / "ic50_predictions.csv"
    auc_pred_path = Path(RESULTS_DIR) / "predictions" / "auc_predictions.csv"
    ic50_predictions = pd.read_csv(ic50_pred_path)
    auc_predictions = pd.read_csv(auc_pred_path)

    logger.info("✓ Data loaded successfully")

    # =========================================================================
    # STEP 2: CHAPTER 2 FIGURES (Data Exploration)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: GENERATING CHAPTER 2 FIGURES (Data Exploration)")
    logger.info("=" * 80)

    plot_target_distributions(y_train_ic50, y_train_auc, figures_dir)
    plot_samples_per_drug(drug_ids_train, figures_dir)

    # =========================================================================
    # STEP 3: CHAPTER 5 FIGURES (Results)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: GENERATING CHAPTER 5 FIGURES (Results)")
    logger.info("=" * 80)

    # Model comparison
    plot_model_comparison(metrics_df, 'AUC', figures_dir)
    plot_model_comparison(metrics_df, 'IC50', figures_dir)

    # Predictions vs actual (all models)
    for model in ['RandomForest', 'XGBoost', 'NeuralNetwork']:
        plot_predictions_vs_actual(ic50_predictions, model, 'IC50', figures_dir)
        plot_predictions_vs_actual(auc_predictions, model, 'AUC', figures_dir)

    # Feature importance (RF and XGBoost only)
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING FEATURE IMPORTANCE FIGURES")
    logger.info("=" * 80)

    from src.models.random_forest import RandomForestModel
    from src.models.xgboost_model import XGBoostModel

    # Load models for feature importance
    rf_auc = RandomForestModel.load(str(get_experiment_dir("rf_auc") / "model.pkl"))
    xgb_auc = XGBoostModel.load(str(get_experiment_dir("xgb_auc") / "model.pkl"))

    plot_feature_importance(rf_auc, "RandomForest", "AUC", gene_names, figures_dir)
    plot_feature_importance(xgb_auc, "XGBoost", "AUC", gene_names, figures_dir)

    # Learning curves
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING LEARNING CURVE FIGURES")
    logger.info("=" * 80)

    plot_learning_curves("xgb_ic50", "XGBoost", "IC50", figures_dir)
    plot_learning_curves("xgb_auc", "XGBoost", "AUC", figures_dir)
    plot_learning_curves("nn_ic50", "NeuralNetwork", "IC50", figures_dir)
    plot_learning_curves("nn_auc", "NeuralNetwork", "AUC", figures_dir)

    # Per-drug performance
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING PER-DRUG PERFORMANCE FIGURES")
    logger.info("=" * 80)

    # Load per-drug metrics
    metrics_dir = Path(RESULTS_DIR) / "metrics"
    for model in ['randomforest', 'xgboost', 'neuralnetwork']:
        for target in ['ic50', 'auc']:
            per_drug_path = metrics_dir / f"per_drug_{target}_{model}.csv"
            if per_drug_path.exists():
                per_drug_df = pd.read_csv(per_drug_path)
                model_name = model.title().replace('forest', 'Forest').replace('boost', 'Boost').replace('network', 'Network')
                plot_per_drug_performance(per_drug_df, model_name, target.upper(), figures_dir)

    # =========================================================================
    # STEP 4: SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("FIGURE GENERATION SUMMARY")
    logger.info("=" * 80)

    figure_files = list(figures_dir.glob("*.png"))
    logger.info(f"\n✓ Generated {len(figure_files)} figures")
    logger.info(f"\nAll figures saved to: {figures_dir}")

    logger.info("\nChapter 2 Figures (Data Exploration):")
    for fig_file in sorted(figures_dir.glob("ch2_*.png")):
        logger.info(f"  - {fig_file.name}")

    logger.info("\nChapter 5 Figures (Results):")
    for fig_file in sorted(figures_dir.glob("ch5_*.png")):
        logger.info(f"  - {fig_file.name}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL FIGURES GENERATED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info("\nYou can now use these figures in your LaTeX thesis!")
    logger.info("")


if __name__ == "__main__":
    main()
