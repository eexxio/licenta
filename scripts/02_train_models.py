"""
Script to train all drug response prediction models.

This script trains 6 models total:
- Random Forest (IC50 & AUC)
- XGBoost (IC50 & AUC)
- Neural Network (IC50 & AUC)

Models are saved to experiments/ directory for later evaluation.

Prerequisites:
    Run scripts/01_preprocess_data.py first!

Usage:
    python scripts/02_train_models.py

Author: Bachelor's Thesis Project
Date: 2026
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    PROCESSED_DATA_DIR,
    EXPERIMENTS_DIR,
    RANDOM_SEED,
    set_seeds,
    get_experiment_dir
)
from src.models.random_forest import RandomForestModel
from src.models.xgboost_model import XGBoostModel
from src.models.neural_network import NeuralNetworkModel

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_preprocessed_data():
    """Load preprocessed data from disk."""
    logger.info("Loading preprocessed data...")

    train_path = PROCESSED_DATA_DIR / 'train_data.npz'
    test_path = PROCESSED_DATA_DIR / 'test_data.npz'

    if not train_path.exists():
        raise FileNotFoundError(
            f"Preprocessed training data not found: {train_path}\n"
            "Please run scripts/01_preprocess_data.py first!"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Preprocessed test data not found: {test_path}\n"
            "Please run scripts/01_preprocess_data.py first!"
        )

    # Load data
    train_data = np.load(train_path)
    test_data = np.load(test_path)

    # Extract arrays
    data = {
        'X_train': train_data['X_train'],
        'y_train_ic50': train_data['y_train_ic50'],
        'y_train_auc': train_data['y_train_auc'],
        'drug_ids_train': train_data['drug_ids_train'],
        'selected_genes': train_data['selected_genes'],
        'X_test': test_data['X_test'],
        'y_test_ic50': test_data['y_test_ic50'],
        'y_test_auc': test_data['y_test_auc'],
        'drug_ids_test': test_data['drug_ids_test']
    }

    logger.info(f"✓ Data loaded:")
    logger.info(f"  - Training samples: {len(data['X_train']):,}")
    logger.info(f"  - Test samples: {len(data['X_test']):,}")
    logger.info(f"  - Features: {data['X_train'].shape[1]:,}")
    logger.info(f"  - Unique drugs: {len(np.unique(data['drug_ids_train'])):,}")

    return data


def create_validation_split(X_train, y_train, drug_ids_train, val_size=0.2):
    """
    Create validation set from training data.

    For XGBoost and Neural Network early stopping.
    """
    logger.info(f"Creating validation split ({val_size*100:.0f}% of training data)...")

    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val, drug_tr, drug_val = train_test_split(
        X_train,
        y_train,
        drug_ids_train,
        test_size=val_size,
        random_state=RANDOM_SEED
    )

    logger.info(f"  - Training: {len(X_tr):,} samples")
    logger.info(f"  - Validation: {len(X_val):,} samples")

    return X_tr, X_val, y_tr, y_val, drug_tr, drug_val


def train_random_forest(X_train, y_train, target_name, exp_dir):
    """Train Random Forest model."""
    logger.info(f"\nTraining Random Forest ({target_name})...")

    # Random Forest doesn't use validation set
    model = RandomForestModel()
    model.fit(X_train, y_train)

    # Save model
    model_path = exp_dir / "model.pkl"
    model.save(str(model_path))
    logger.info(f"✓ Model saved to: {model_path}")

    # Save config
    config = {
        'model_type': 'RandomForest',
        'target': target_name,
        'n_samples': len(X_train),
        'n_features': X_train.shape[1],
        'random_seed': RANDOM_SEED,
        'trained_at': datetime.now().isoformat()
    }

    config_path = exp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return model


def train_xgboost(X_train, y_train, X_val, y_val, target_name, exp_dir):
    """Train XGBoost model with early stopping."""
    logger.info(f"\nTraining XGBoost ({target_name})...")

    model = XGBoostModel()
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    # Save model
    model_path = exp_dir / "model.pkl"
    model.save(str(model_path))
    logger.info(f"✓ Model saved to: {model_path}")

    # Save training history
    history = model.get_training_history()
    history_path = exp_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save config
    config = {
        'model_type': 'XGBoost',
        'target': target_name,
        'n_samples_train': len(X_train),
        'n_samples_val': len(X_val),
        'n_features': X_train.shape[1],
        'random_seed': RANDOM_SEED,
        'trained_at': datetime.now().isoformat()
    }

    if hasattr(model.model, 'best_iteration'):
        config['best_iteration'] = int(model.model.best_iteration)

    config_path = exp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return model


def train_neural_network(X_train, y_train, drug_ids_train, X_val, y_val, drug_ids_val,
                        n_drugs, target_name, exp_dir):
    """Train Neural Network model with early stopping."""
    logger.info(f"\nTraining Neural Network ({target_name})...")

    n_genes = X_train.shape[1] - 1  # Subtract drug feature

    model = NeuralNetworkModel(n_genes=n_genes, n_drugs=n_drugs)

    # Prepare data with drug IDs
    X_train_with_drugs = np.column_stack([X_train[:, :-1], drug_ids_train])
    X_val_with_drugs = np.column_stack([X_val[:, :-1], drug_ids_val])

    model.fit(
        X_train_with_drugs,
        y_train,
        X_val=X_val_with_drugs,
        y_val=y_val
    )

    # Save model
    model_path = exp_dir / "model.pt"
    model.save(str(model_path))
    logger.info(f"✓ Model saved to: {model_path}")

    # Save training history
    history = {
        'train_losses': [float(x) for x in model.train_losses],
        'val_losses': [float(x) for x in model.val_losses]
    }
    history_path = exp_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save config
    config = {
        'model_type': 'NeuralNetwork',
        'target': target_name,
        'n_samples_train': len(X_train),
        'n_samples_val': len(X_val),
        'n_genes': n_genes,
        'n_drugs': n_drugs,
        'random_seed': RANDOM_SEED,
        'trained_at': datetime.now().isoformat(),
        'epochs_trained': len(model.train_losses),
        'best_epoch': int(np.argmin(model.val_losses)) if model.val_losses else 0
    }

    config_path = exp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return model


def main():
    """Main training pipeline."""

    logger.info("=" * 80)
    logger.info("TRAINING ALL MODELS")
    logger.info("=" * 80)

    # Set random seed
    set_seeds()
    logger.info(f"Random seed set to: {RANDOM_SEED}")

    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: LOADING PREPROCESSED DATA")
    logger.info("=" * 80)

    try:
        data = load_preprocessed_data()
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)

    # Extract data
    X_train_full = data['X_train']
    y_train_ic50_full = data['y_train_ic50']
    y_train_auc_full = data['y_train_auc']
    drug_ids_train_full = data['drug_ids_train']

    n_drugs = len(np.unique(drug_ids_train_full))
    logger.info(f"\nNumber of unique drugs: {n_drugs}")

    # =========================================================================
    # STEP 2: CREATE VALIDATION SPLIT
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: CREATING VALIDATION SPLIT")
    logger.info("=" * 80)

    # For IC50
    X_train_ic50, X_val_ic50, y_train_ic50, y_val_ic50, drug_train_ic50, drug_val_ic50 = \
        create_validation_split(X_train_full, y_train_ic50_full, drug_ids_train_full)

    # For AUC
    X_train_auc, X_val_auc, y_train_auc, y_val_auc, drug_train_auc, drug_val_auc = \
        create_validation_split(X_train_full, y_train_auc_full, drug_ids_train_full)

    # =========================================================================
    # STEP 3: TRAIN RANDOM FOREST MODELS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: TRAINING RANDOM FOREST MODELS")
    logger.info("=" * 80)

    # Random Forest (IC50)
    rf_ic50_dir = get_experiment_dir("random_forest", "IC50")
    rf_ic50_dir.mkdir(parents=True, exist_ok=True)
    rf_ic50_model = train_random_forest(X_train_full, y_train_ic50_full, "IC50", rf_ic50_dir)

    # Random Forest (AUC)
    rf_auc_dir = get_experiment_dir("random_forest", "AUC")
    rf_auc_dir.mkdir(parents=True, exist_ok=True)
    rf_auc_model = train_random_forest(X_train_full, y_train_auc_full, "AUC", rf_auc_dir)

    logger.info("\n✓ Random Forest models trained successfully!")

    # =========================================================================
    # STEP 4: TRAIN XGBOOST MODELS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: TRAINING XGBOOST MODELS")
    logger.info("=" * 80)

    # XGBoost (IC50)
    xgb_ic50_dir = get_experiment_dir("xgboost", "IC50")
    xgb_ic50_dir.mkdir(parents=True, exist_ok=True)
    xgb_ic50_model = train_xgboost(
        X_train_ic50, y_train_ic50,
        X_val_ic50, y_val_ic50,
        "IC50", xgb_ic50_dir
    )

    # XGBoost (AUC)
    xgb_auc_dir = get_experiment_dir("xgboost", "AUC")
    xgb_auc_dir.mkdir(parents=True, exist_ok=True)
    xgb_auc_model = train_xgboost(
        X_train_auc, y_train_auc,
        X_val_auc, y_val_auc,
        "AUC", xgb_auc_dir
    )

    logger.info("\n✓ XGBoost models trained successfully!")

    # =========================================================================
    # STEP 5: TRAIN NEURAL NETWORK MODELS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: TRAINING NEURAL NETWORK MODELS")
    logger.info("=" * 80)

    # Neural Network (IC50)
    nn_ic50_dir = get_experiment_dir("neural_network", "IC50")
    nn_ic50_dir.mkdir(parents=True, exist_ok=True)
    nn_ic50_model = train_neural_network(
        X_train_ic50, y_train_ic50, drug_train_ic50,
        X_val_ic50, y_val_ic50, drug_val_ic50,
        n_drugs, "IC50", nn_ic50_dir
    )

    # Neural Network (AUC)
    nn_auc_dir = get_experiment_dir("neural_network", "AUC")
    nn_auc_dir.mkdir(parents=True, exist_ok=True)
    nn_auc_model = train_neural_network(
        X_train_auc, y_train_auc, drug_train_auc,
        X_val_auc, y_val_auc, drug_val_auc,
        n_drugs, "AUC", nn_auc_dir
    )

    logger.info("\n✓ Neural Network models trained successfully!")

    # =========================================================================
    # STEP 6: SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)

    logger.info("\nModels trained:")
    logger.info("  1. Random Forest (IC50)")
    logger.info("  2. Random Forest (AUC)")
    logger.info("  3. XGBoost (IC50)")
    logger.info("  4. XGBoost (AUC)")
    logger.info("  5. Neural Network (IC50)")
    logger.info("  6. Neural Network (AUC)")

    logger.info("\nModel directories:")
    logger.info(f"  - {rf_ic50_dir}")
    logger.info(f"  - {rf_auc_dir}")
    logger.info(f"  - {xgb_ic50_dir}")
    logger.info(f"  - {xgb_auc_dir}")
    logger.info(f"  - {nn_ic50_dir}")
    logger.info(f"  - {nn_auc_dir}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ ALL MODELS TRAINED SUCCESSFULLY!")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("  1. Evaluate models: python scripts/03_evaluate_models.py")
    logger.info("  2. Generate figures: python scripts/04_generate_figures.py")
    logger.info("")


if __name__ == "__main__":
    main()
