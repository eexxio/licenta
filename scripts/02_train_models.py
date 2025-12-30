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
    USE_GPU_RF,
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
        'cosmic_ids_train': train_data['cosmic_ids_train'],  # Cell line IDs for validation split
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


def create_validation_split(X_train, y_train, drug_ids_train, cosmic_ids_train, val_size=0.2):
    """
    Create validation set from training data BY CELL LINE to avoid data leakage.

    CRITICAL: We split by cell line, not randomly, to ensure no cell line
    appears in both training and validation sets. This prevents data leakage
    where the model could memorize cell line-specific patterns.

    For XGBoost and Neural Network early stopping.
    """
    logger.info(f"Creating validation split BY CELL LINE ({val_size*100:.0f}% of training data)...")
    logger.info("CRITICAL: Splitting by cell line to prevent data leakage!")

    # Get unique cell lines
    unique_cell_lines = np.unique(cosmic_ids_train)
    n_cell_lines = len(unique_cell_lines)
    n_val_cell_lines = int(n_cell_lines * val_size)

    logger.info(f"  - Total cell lines: {n_cell_lines}")
    logger.info(f"  - Validation cell lines: {n_val_cell_lines}")

    # Shuffle and split cell lines
    np.random.seed(RANDOM_SEED)
    shuffled_cell_lines = np.random.permutation(unique_cell_lines)
    val_cell_lines = set(shuffled_cell_lines[:n_val_cell_lines])
    train_cell_lines = set(shuffled_cell_lines[n_val_cell_lines:])

    # Create masks for training and validation
    val_mask = np.array([cid in val_cell_lines for cid in cosmic_ids_train])
    train_mask = ~val_mask

    # Split data
    X_tr = X_train[train_mask]
    X_val = X_train[val_mask]
    y_tr = y_train[train_mask]
    y_val = y_train[val_mask]
    drug_tr = drug_ids_train[train_mask]
    drug_val = drug_ids_train[val_mask]

    logger.info(f"  - Training: {len(X_tr):,} samples from {len(train_cell_lines)} cell lines")
    logger.info(f"  - Validation: {len(X_val):,} samples from {len(val_cell_lines)} cell lines")
    logger.info("  - ✓ No cell line overlap between training and validation!")

    return X_tr, X_val, y_tr, y_val, drug_tr, drug_val


def create_drug_one_hot(drug_ids, unique_drugs):
    """
    Create one-hot encoding for drug IDs.
    
    This allows tree-based models (RF, XGBoost) to use drug information,
    making the comparison with Neural Networks fair.
    
    Args:
        drug_ids: Array of drug IDs
        unique_drugs: Array of all unique drug IDs (defines the encoding)
    
    Returns:
        One-hot encoded drug features (n_samples, n_drugs)
    """
    drug_to_idx = {drug: idx for idx, drug in enumerate(unique_drugs)}
    n_samples = len(drug_ids)
    n_drugs = len(unique_drugs)
    
    one_hot = np.zeros((n_samples, n_drugs), dtype=np.float32)
    for i, drug_id in enumerate(drug_ids):
        if drug_id in drug_to_idx:
            one_hot[i, drug_to_idx[drug_id]] = 1.0
    
    return one_hot


def train_random_forest(X_train, y_train, drug_ids_train, unique_drugs, target_name, exp_dir):
    """Train Random Forest model with drug one-hot encoding."""
    logger.info(f"\nTraining Random Forest ({target_name})...")

    # Get gene expression features (exclude last column which is drug ID)
    X_genes = X_train[:, :-1]
    
    # Create one-hot encoding for drug IDs
    # This gives RF the same drug information that NN gets via embeddings
    drug_one_hot = create_drug_one_hot(drug_ids_train, unique_drugs)
    logger.info(f"  Drug one-hot encoding shape: {drug_one_hot.shape}")
    
    # Combine gene features with drug one-hot encoding
    X_train_with_drugs = np.hstack([X_genes, drug_one_hot])
    logger.info(f"  Total features: {X_train_with_drugs.shape[1]} (genes: {X_genes.shape[1]}, drugs: {drug_one_hot.shape[1]})")
    
    # Random Forest doesn't use validation set
    # Use GPU if configured and available
    model = RandomForestModel(use_gpu=USE_GPU_RF)
    model.fit(X_train_with_drugs, y_train)

    # Save model
    model_path = exp_dir / "model.pkl"
    model.save(str(model_path))
    logger.info(f"✓ Model saved to: {model_path}")

    # Save unique drugs for inference
    unique_drugs_path = exp_dir / "unique_drugs.json"
    with open(unique_drugs_path, 'w') as f:
        json.dump([int(d) for d in unique_drugs], f)
    logger.info(f"✓ Unique drugs saved to: {unique_drugs_path}")

    # Save config
    config = {
        'model_type': 'RandomForest',
        'target': target_name,
        'n_samples': len(X_train),
        'n_gene_features': X_genes.shape[1],
        'n_drug_features': len(unique_drugs),
        'n_total_features': X_train_with_drugs.shape[1],
        'random_seed': RANDOM_SEED,
        'trained_at': datetime.now().isoformat()
    }

    config_path = exp_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return model


def train_xgboost(X_train, y_train, X_val, y_val, drug_ids_train, drug_ids_val, unique_drugs, target_name, exp_dir):
    """Train XGBoost model with early stopping and drug one-hot encoding."""
    logger.info(f"\nTraining XGBoost ({target_name})...")

    # Get gene expression features (exclude last column which is drug ID)
    X_train_genes = X_train[:, :-1]
    X_val_genes = X_val[:, :-1]
    
    # Create one-hot encoding for drug IDs
    # This gives XGBoost the same drug information that NN gets via embeddings
    train_drug_one_hot = create_drug_one_hot(drug_ids_train, unique_drugs)
    val_drug_one_hot = create_drug_one_hot(drug_ids_val, unique_drugs)
    logger.info(f"  Drug one-hot encoding shape: {train_drug_one_hot.shape}")
    
    # Combine gene features with drug one-hot encoding
    X_train_with_drugs = np.hstack([X_train_genes, train_drug_one_hot])
    X_val_with_drugs = np.hstack([X_val_genes, val_drug_one_hot])
    logger.info(f"  Total features: {X_train_with_drugs.shape[1]} (genes: {X_train_genes.shape[1]}, drugs: {train_drug_one_hot.shape[1]})")

    model = XGBoostModel()
    model.fit(X_train_with_drugs, y_train, X_val=X_val_with_drugs, y_val=y_val)

    # Save model
    model_path = exp_dir / "model.pkl"
    model.save(str(model_path))
    logger.info(f"✓ Model saved to: {model_path}")

    # Save training history
    history = model.get_training_history()
    history_path = exp_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save unique drugs for inference
    unique_drugs_path = exp_dir / "unique_drugs.json"
    with open(unique_drugs_path, 'w') as f:
        json.dump([int(d) for d in unique_drugs], f)
    logger.info(f"✓ Unique drugs saved to: {unique_drugs_path}")

    # Save config
    config = {
        'model_type': 'XGBoost',
        'target': target_name,
        'n_samples_train': len(X_train),
        'n_samples_val': len(X_val),
        'n_gene_features': X_train_genes.shape[1],
        'n_drug_features': len(unique_drugs),
        'n_total_features': X_train_with_drugs.shape[1],
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
                        n_drugs, target_name, exp_dir, drug_id_mapping=None):
    """Train Neural Network model with early stopping."""
    logger.info(f"\nTraining Neural Network ({target_name})...")

    n_genes = X_train.shape[1] - 1  # Subtract drug feature

    model = NeuralNetworkModel(n_genes=n_genes, n_drugs=n_drugs)

    # Map drug IDs to 0-indexed values for embedding layer
    # The embedding layer expects indices in range [0, n_drugs-1]
    if drug_id_mapping is not None:
        drug_ids_train_mapped = np.array([drug_id_mapping[d] for d in drug_ids_train])
        drug_ids_val_mapped = np.array([drug_id_mapping[d] for d in drug_ids_val])
    else:
        drug_ids_train_mapped = drug_ids_train
        drug_ids_val_mapped = drug_ids_val

    # Prepare data with drug IDs
    X_train_with_drugs = np.column_stack([X_train[:, :-1], drug_ids_train_mapped])
    X_val_with_drugs = np.column_stack([X_val[:, :-1], drug_ids_val_mapped])

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
        'train_losses': [float(x) for x in model.history['train_loss']],
        'val_losses': [float(x) for x in model.history['val_loss']]
    }
    history_path = exp_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Save drug ID mapping for inference
    if drug_id_mapping is not None:
        mapping_path = exp_dir / "drug_id_mapping.json"
        # Convert keys to strings for JSON serialization
        mapping_for_json = {str(k): v for k, v in drug_id_mapping.items()}
        with open(mapping_path, 'w') as f:
            json.dump(mapping_for_json, f, indent=2)
        logger.info(f"✓ Drug ID mapping saved to: {mapping_path}")

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
        'epochs_trained': len(model.history['train_loss']),
        'best_epoch': int(np.argmin(model.history['val_loss'])) if model.history['val_loss'] else 0
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
    cosmic_ids_train_full = data['cosmic_ids_train']  # Cell line IDs for validation split

    # Create drug ID to 0-indexed mapping for neural network embedding layer
    # The embedding layer expects indices in range [0, n_drugs-1]
    unique_drugs = np.unique(drug_ids_train_full)
    n_drugs = len(unique_drugs)
    drug_id_mapping = {drug_id: idx for idx, drug_id in enumerate(unique_drugs)}
    logger.info(f"\nNumber of unique drugs: {n_drugs}")
    logger.info(f"Drug ID range: {unique_drugs.min()} - {unique_drugs.max()} -> mapped to 0-{n_drugs-1}")

    # =========================================================================
    # STEP 2: CREATE VALIDATION SPLIT (BY CELL LINE TO PREVENT DATA LEAKAGE)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: CREATING VALIDATION SPLIT (BY CELL LINE)")
    logger.info("=" * 80)

    # For IC50
    X_train_ic50, X_val_ic50, y_train_ic50, y_val_ic50, drug_train_ic50, drug_val_ic50 = \
        create_validation_split(X_train_full, y_train_ic50_full, drug_ids_train_full, cosmic_ids_train_full)

    # For AUC
    X_train_auc, X_val_auc, y_train_auc, y_val_auc, drug_train_auc, drug_val_auc = \
        create_validation_split(X_train_full, y_train_auc_full, drug_ids_train_full, cosmic_ids_train_full)

    # =========================================================================
    # STEP 3: TRAIN RANDOM FOREST MODELS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: TRAINING RANDOM FOREST MODELS")
    logger.info("=" * 80)

    # Random Forest (IC50)
    rf_ic50_dir = get_experiment_dir("random_forest", "IC50")
    rf_ic50_dir.mkdir(parents=True, exist_ok=True)
    rf_ic50_model_path = rf_ic50_dir / "model.pkl"
    if rf_ic50_model_path.exists():
        logger.info(f"\n⏭ Skipping Random Forest (IC50) - model already exists: {rf_ic50_model_path}")
    else:
        rf_ic50_model = train_random_forest(X_train_full, y_train_ic50_full, drug_ids_train_full, unique_drugs, "IC50", rf_ic50_dir)

    # Random Forest (AUC)
    rf_auc_dir = get_experiment_dir("random_forest", "AUC")
    rf_auc_dir.mkdir(parents=True, exist_ok=True)
    rf_auc_model_path = rf_auc_dir / "model.pkl"
    if rf_auc_model_path.exists():
        logger.info(f"\n⏭ Skipping Random Forest (AUC) - model already exists: {rf_auc_model_path}")
    else:
        rf_auc_model = train_random_forest(X_train_full, y_train_auc_full, drug_ids_train_full, unique_drugs, "AUC", rf_auc_dir)

    logger.info("\n✓ Random Forest models done!")

    # =========================================================================
    # STEP 4: TRAIN XGBOOST MODELS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: TRAINING XGBOOST MODELS")
    logger.info("=" * 80)

    # XGBoost (IC50)
    xgb_ic50_dir = get_experiment_dir("xgboost", "IC50")
    xgb_ic50_dir.mkdir(parents=True, exist_ok=True)
    xgb_ic50_model_path = xgb_ic50_dir / "model.pkl"
    if xgb_ic50_model_path.exists():
        logger.info(f"\n⏭ Skipping XGBoost (IC50) - model already exists: {xgb_ic50_model_path}")
    else:
        xgb_ic50_model = train_xgboost(
            X_train_ic50, y_train_ic50,
            X_val_ic50, y_val_ic50,
            drug_train_ic50, drug_val_ic50, unique_drugs,
            "IC50", xgb_ic50_dir
        )

    # XGBoost (AUC)
    xgb_auc_dir = get_experiment_dir("xgboost", "AUC")
    xgb_auc_dir.mkdir(parents=True, exist_ok=True)
    xgb_auc_model_path = xgb_auc_dir / "model.pkl"
    if xgb_auc_model_path.exists():
        logger.info(f"\n⏭ Skipping XGBoost (AUC) - model already exists: {xgb_auc_model_path}")
    else:
        xgb_auc_model = train_xgboost(
            X_train_auc, y_train_auc,
            X_val_auc, y_val_auc,
            drug_train_auc, drug_val_auc, unique_drugs,
            "AUC", xgb_auc_dir
        )

    logger.info("\n✓ XGBoost models done!")

    # =========================================================================
    # STEP 5: TRAIN NEURAL NETWORK MODELS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: TRAINING NEURAL NETWORK MODELS")
    logger.info("=" * 80)

    # Neural Network (IC50)
    nn_ic50_dir = get_experiment_dir("neural_network", "IC50")
    nn_ic50_dir.mkdir(parents=True, exist_ok=True)
    nn_ic50_model_path = nn_ic50_dir / "model.pt"
    if nn_ic50_model_path.exists():
        logger.info(f"\n⏭ Skipping Neural Network (IC50) - model already exists: {nn_ic50_model_path}")
    else:
        nn_ic50_model = train_neural_network(
            X_train_ic50, y_train_ic50, drug_train_ic50,
            X_val_ic50, y_val_ic50, drug_val_ic50,
            n_drugs, "IC50", nn_ic50_dir, drug_id_mapping
        )

    # Neural Network (AUC)
    nn_auc_dir = get_experiment_dir("neural_network", "AUC")
    nn_auc_dir.mkdir(parents=True, exist_ok=True)
    nn_auc_model_path = nn_auc_dir / "model.pt"
    if nn_auc_model_path.exists():
        logger.info(f"\n⏭ Skipping Neural Network (AUC) - model already exists: {nn_auc_model_path}")
    else:
        nn_auc_model = train_neural_network(
            X_train_auc, y_train_auc, drug_train_auc,
            X_val_auc, y_val_auc, drug_val_auc,
            n_drugs, "AUC", nn_auc_dir, drug_id_mapping
        )

    logger.info("\n✓ Neural Network models done!")

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
