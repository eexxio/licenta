"""
Script to evaluate all trained models on the test set.

This script:
1. Loads all 6 trained models
2. Evaluates each model on the test set
3. Computes all metrics (R², RMSE, MAE, Spearman, Pearson)
4. Computes per-drug performance
5. Saves results to CSV files

Prerequisites:
    - Run scripts/01_preprocess_data.py
    - Run scripts/02_train_models.py

Usage:
    python scripts/03_evaluate_models.py

Author: Bachelor's Thesis Project
Date: 2026
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    get_experiment_dir
)
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.neural_network import NeuralNetworkModel
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_per_drug_metrics,
    compare_models,
    print_metrics
)

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_preprocessed_data():
    """Load preprocessed test data."""
    logger.info("Loading preprocessed test data...")

    test_path = PROCESSED_DATA_DIR / 'test_data.npz'

    if not test_path.exists():
        raise FileNotFoundError(
            f"Preprocessed test data not found: {test_path}\n"
            "Please run scripts/01_preprocess_data.py first!"
        )

    test_data = np.load(test_path)

    data = {
        'X_test': test_data['X_test'],
        'y_test_ic50': test_data['y_test_ic50'],
        'y_test_auc': test_data['y_test_auc'],
        'drug_ids_test': test_data['drug_ids_test']
    }

    logger.info(f"✓ Test data loaded:")
    logger.info(f"  - Samples: {len(data['X_test']):,}")
    logger.info(f"  - Features: {data['X_test'].shape[1]:,}")

    return data


def load_model(exp_name, model_type, n_genes=None, n_drugs=None):
    """Load a trained model from experiments directory."""
    exp_dir = get_experiment_dir(exp_name)

    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    # Load based on model type
    if model_type == "RandomForest":
        model_path = exp_dir / "model.pkl"
        model = RandomForestModel.load(str(model_path))

    elif model_type == "XGBoost":
        model_path = exp_dir / "model.pkl"
        model = XGBoostModel.load(str(model_path))

    elif model_type == "NeuralNetwork":
        if n_genes is None or n_drugs is None:
            raise ValueError("n_genes and n_drugs required for Neural Network")
        model_path = exp_dir / "model.pt"
        model = NeuralNetworkModel.load(str(model_path), n_genes=n_genes, n_drugs=n_drugs)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"✓ Loaded {model_type} from {exp_name}")
    return model


def evaluate_model(model, X_test, y_test, drug_ids_test, model_name, target_name):
    """Evaluate a single model."""
    logger.info(f"\nEvaluating {model_name} ({target_name})...")

    # Make predictions
    y_pred = model.predict(X_test)

    # Compute all metrics
    metrics = compute_all_metrics(y_test, y_pred)

    # Print metrics
    print_metrics(metrics, model_name)

    # Add model and target info
    results = {
        'model': model_name,
        'target': target_name,
        **metrics
    }

    # Compute per-drug metrics
    per_drug_metrics = compute_per_drug_metrics(y_test, y_pred, drug_ids_test)

    return results, y_pred, per_drug_metrics


def main():
    """Main evaluation pipeline."""

    logger.info("=" * 80)
    logger.info("EVALUATING ALL MODELS")
    logger.info("=" * 80)

    # =========================================================================
    # STEP 1: LOAD TEST DATA
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: LOADING TEST DATA")
    logger.info("=" * 80)

    try:
        data = load_preprocessed_data()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    X_test = data['X_test']
    y_test_ic50 = data['y_test_ic50']
    y_test_auc = data['y_test_auc']
    drug_ids_test = data['drug_ids_test']

    n_genes = X_test.shape[1] - 1  # Subtract drug feature
    n_drugs = len(np.unique(drug_ids_test))

    logger.info(f"\nTest set info:")
    logger.info(f"  - Samples: {len(X_test):,}")
    logger.info(f"  - Genes: {n_genes:,}")
    logger.info(f"  - Drugs: {n_drugs:,}")

    # =========================================================================
    # STEP 2: LOAD ALL MODELS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: LOADING TRAINED MODELS")
    logger.info("=" * 80)

    try:
        # Random Forest models
        rf_ic50 = load_model("rf_ic50", "RandomForest")
        rf_auc = load_model("rf_auc", "RandomForest")

        # XGBoost models
        xgb_ic50 = load_model("xgb_ic50", "XGBoost")
        xgb_auc = load_model("xgb_auc", "XGBoost")

        # Neural Network models
        nn_ic50 = load_model("nn_ic50", "NeuralNetwork", n_genes=n_genes, n_drugs=n_drugs)
        nn_auc = load_model("nn_auc", "NeuralNetwork", n_genes=n_genes, n_drugs=n_drugs)

        logger.info("\n✓ All models loaded successfully!")

    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.error("Make sure you've run scripts/02_train_models.py first!")
        sys.exit(1)

    # =========================================================================
    # STEP 3: EVALUATE IC50 MODELS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: EVALUATING IC50 MODELS")
    logger.info("=" * 80)

    ic50_results = []
    ic50_predictions = {}
    ic50_per_drug = {}

    # Random Forest (IC50)
    res, pred, per_drug = evaluate_model(
        rf_ic50, X_test, y_test_ic50, drug_ids_test,
        "RandomForest", "IC50"
    )
    ic50_results.append(res)
    ic50_predictions['RandomForest'] = pred
    ic50_per_drug['RandomForest'] = per_drug

    # XGBoost (IC50)
    res, pred, per_drug = evaluate_model(
        xgb_ic50, X_test, y_test_ic50, drug_ids_test,
        "XGBoost", "IC50"
    )
    ic50_results.append(res)
    ic50_predictions['XGBoost'] = pred
    ic50_per_drug['XGBoost'] = per_drug

    # Neural Network (IC50)
    # Prepare data with drug IDs for NN
    X_test_with_drugs = np.column_stack([X_test[:, :-1], drug_ids_test])
    res, pred, per_drug = evaluate_model(
        nn_ic50, X_test_with_drugs, y_test_ic50, drug_ids_test,
        "NeuralNetwork", "IC50"
    )
    ic50_results.append(res)
    ic50_predictions['NeuralNetwork'] = pred
    ic50_per_drug['NeuralNetwork'] = per_drug

    # =========================================================================
    # STEP 4: EVALUATE AUC MODELS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: EVALUATING AUC MODELS")
    logger.info("=" * 80)

    auc_results = []
    auc_predictions = {}
    auc_per_drug = {}

    # Random Forest (AUC)
    res, pred, per_drug = evaluate_model(
        rf_auc, X_test, y_test_auc, drug_ids_test,
        "RandomForest", "AUC"
    )
    auc_results.append(res)
    auc_predictions['RandomForest'] = pred
    auc_per_drug['RandomForest'] = per_drug

    # XGBoost (AUC)
    res, pred, per_drug = evaluate_model(
        xgb_auc, X_test, y_test_auc, drug_ids_test,
        "XGBoost", "AUC"
    )
    auc_results.append(res)
    auc_predictions['XGBoost'] = pred
    auc_per_drug['XGBoost'] = per_drug

    # Neural Network (AUC)
    res, pred, per_drug = evaluate_model(
        nn_auc, X_test_with_drugs, y_test_auc, drug_ids_test,
        "NeuralNetwork", "AUC"
    )
    auc_results.append(res)
    auc_predictions['NeuralNetwork'] = pred
    auc_per_drug['NeuralNetwork'] = per_drug

    # =========================================================================
    # STEP 5: SAVE RESULTS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: SAVING RESULTS")
    logger.info("=" * 80)

    # Create results directory
    metrics_dir = Path(RESULTS_DIR) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    predictions_dir = Path(RESULTS_DIR) / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Save overall metrics
    all_results = ic50_results + auc_results
    results_df = pd.DataFrame(all_results)
    results_path = metrics_dir / "model_comparison.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"✓ Overall metrics saved to: {results_path}")

    # Save predictions (IC50)
    ic50_pred_df = pd.DataFrame({
        'y_true': y_test_ic50,
        'drug_id': drug_ids_test,
        **{f'{model}_pred': pred for model, pred in ic50_predictions.items()}
    })
    ic50_pred_path = predictions_dir / "ic50_predictions.csv"
    ic50_pred_df.to_csv(ic50_pred_path, index=False)
    logger.info(f"✓ IC50 predictions saved to: {ic50_pred_path}")

    # Save predictions (AUC)
    auc_pred_df = pd.DataFrame({
        'y_true': y_test_auc,
        'drug_id': drug_ids_test,
        **{f'{model}_pred': pred for model, pred in auc_predictions.items()}
    })
    auc_pred_path = predictions_dir / "auc_predictions.csv"
    auc_pred_df.to_csv(auc_pred_path, index=False)
    logger.info(f"✓ AUC predictions saved to: {auc_pred_path}")

    # Save per-drug metrics (IC50)
    for model_name, per_drug_df in ic50_per_drug.items():
        per_drug_path = metrics_dir / f"per_drug_ic50_{model_name.lower()}.csv"
        per_drug_df.to_csv(per_drug_path, index=False)
        logger.info(f"✓ Per-drug IC50 metrics ({model_name}) saved to: {per_drug_path}")

    # Save per-drug metrics (AUC)
    for model_name, per_drug_df in auc_per_drug.items():
        per_drug_path = metrics_dir / f"per_drug_auc_{model_name.lower()}.csv"
        per_drug_df.to_csv(per_drug_path, index=False)
        logger.info(f"✓ Per-drug AUC metrics ({model_name}) saved to: {per_drug_path}")

    # =========================================================================
    # STEP 6: FINAL SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)

    logger.info("\nIC50 Prediction Results:")
    ic50_df = pd.DataFrame(ic50_results)
    logger.info("\n" + ic50_df.to_string(index=False))

    best_ic50_idx = ic50_df['r2'].idxmax()
    best_ic50_model = ic50_df.loc[best_ic50_idx, 'model']
    best_ic50_r2 = ic50_df.loc[best_ic50_idx, 'r2']
    logger.info(f"\n🏆 Best IC50 model: {best_ic50_model} (R² = {best_ic50_r2:.4f})")

    logger.info("\nAUC Prediction Results:")
    auc_df = pd.DataFrame(auc_results)
    logger.info("\n" + auc_df.to_string(index=False))

    best_auc_idx = auc_df['r2'].idxmax()
    best_auc_model = auc_df.loc[best_auc_idx, 'model']
    best_auc_r2 = auc_df.loc[best_auc_idx, 'r2']
    logger.info(f"\n🏆 Best AUC model: {best_auc_model} (R² = {best_auc_r2:.4f})")

    logger.info("\n" + "=" * 80)
    logger.info("✓ EVALUATION COMPLETE!")
    logger.info("=" * 80)
    logger.info("\nNext step:")
    logger.info("  - Generate figures: python scripts/04_generate_figures.py")
    logger.info("")


if __name__ == "__main__":
    main()
