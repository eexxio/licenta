"""
Central configuration file for the thesis project.

This file contains all configuration settings including:
- File paths
- Model hyperparameters
- Training settings
- Random seeds for reproducibility

Author: Bachelor's Thesis Project
Date: 2026
"""

import os
from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Project root directory
# Using Path(__file__).parent.parent gets us from src/config.py to project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
ARCHIVE_DIR = PROJECT_ROOT / "archive"
RAW_DATA_DIR = ARCHIVE_DIR  # Alias for raw/downloaded data
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

# Source code directories
SRC_DIR = PROJECT_ROOT / "src"

# Output directories
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
RESULTS_DIR = PROJECT_ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

# Create directories if they don't exist
for directory in [PROCESSED_DATA_DIR, SPLITS_DIR, EXPERIMENTS_DIR,
                  METRICS_DIR, FIGURES_DIR, PREDICTIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA FILE PATHS
# =============================================================================

# Raw data files (in archive/)
GDSC_DATASET_PATH = ARCHIVE_DIR / "GDSC_DATASET.csv"
COMPOUNDS_ANNOTATION_PATH = ARCHIVE_DIR / "Compounds-annotation.csv"
CELL_LINES_DETAILS_PATH = ARCHIVE_DIR / "Cell_Lines_Details.xlsx"

# Processed data files (will be created)
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "gdsc_processed.csv"
GENE_EXPRESSION_PATH = PROCESSED_DATA_DIR / "gene_expression.csv"
DRUG_RESPONSE_PATH = PROCESSED_DATA_DIR / "drug_response.csv"
METADATA_PATH = PROCESSED_DATA_DIR / "metadata.csv"

# Data split files (will be created)
TRAIN_INDICES_PATH = SPLITS_DIR / "train_indices.pkl"
TEST_INDICES_PATH = SPLITS_DIR / "test_indices.pkl"
CV_FOLDS_PATH = SPLITS_DIR / "cv_folds.pkl"

# =============================================================================
# REPRODUCIBILITY
# =============================================================================

# Random seed for reproducibility
# Using 42 as it's a common convention (and a Hitchhiker's Guide reference!)
RANDOM_SEED = 42

# Set seeds for all random number generators
def set_seeds(seed=RANDOM_SEED):
    """
    Set random seeds for all libraries to ensure reproducibility.

    This function should be called at the start of every script to ensure
    that results are reproducible across different runs.

    Args:
        seed: Random seed value (default: 42)
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If using CUDA (GPU), also set CUDA seeds
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# =============================================================================
# DATA PREPROCESSING PARAMETERS
# =============================================================================

# Feature selection
N_TOP_GENES = 5000  # Number of top-variance genes to keep
# Rationale: 17,419 genes is too many (curse of dimensionality)
# Top 5,000 genes by variance capture most of the signal while reducing
# computational cost and overfitting risk

# Data filtering
MIN_SAMPLES_PER_DRUG = 30  # Minimum samples required to keep a drug
# Rationale: Drugs with too few samples can't be reliably modeled
# 30 is a reasonable threshold for meaningful statistical analysis

# Train/test split
TEST_SIZE = 0.2  # 80% train, 20% test
# Split by cell line to avoid data leakage (same cell line shouldn't be in both)

# Cross-validation
N_CV_FOLDS = 5  # Number of cross-validation folds for hyperparameter tuning

# Missing value handling
IMPUTATION_STRATEGY = "median"  # Strategy for imputing missing gene expression
# Median is robust to outliers, common choice for gene expression data

# =============================================================================
# TARGET VARIABLES
# =============================================================================

# Which response metrics to predict
TARGETS = ["AUC", "IC50"]
# AUC: Area Under the dose-response Curve (more stable, recommended)
# IC50: Half-maximal Inhibitory Concentration (noisier but traditional)

PRIMARY_TARGET = "AUC"  # Primary target for main experiments

# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

# Random Forest
RF_CONFIG = {
    "n_estimators": 500,          # Number of trees in the forest
    "max_depth": 20,              # Maximum depth of each tree
    "min_samples_split": 10,      # Minimum samples required to split a node
    "min_samples_leaf": 5,        # Minimum samples required at leaf node
    "max_features": "sqrt",       # Number of features for best split
    "n_jobs": -1,                 # Use all CPU cores
    "random_state": RANDOM_SEED,
    "verbose": 1                  # Print progress
}

# XGBoost
XGB_CONFIG = {
    "learning_rate": 0.05,        # Step size shrinkage (smaller = more robust)
    "max_depth": 6,               # Maximum tree depth
    "n_estimators": 1000,         # Number of boosting rounds
    "min_child_weight": 5,        # Minimum sum of weights in a child node
    "subsample": 0.8,             # Fraction of samples for each tree
    "colsample_bytree": 0.8,      # Fraction of features for each tree
    "gamma": 0,                   # Minimum loss reduction for split
    "reg_alpha": 0.1,             # L1 regularization
    "reg_lambda": 1.0,            # L2 regularization
    "early_stopping_rounds": 50,  # Stop if no improvement for N rounds
    "random_state": RANDOM_SEED,
    "n_jobs": -1,                 # Use all CPU cores
    "tree_method": "gpu_hist",    # GPU-accelerated histogram algorithm
    "device": "cuda"              # Use NVIDIA GPU
}

# Neural Network (PyTorch)
NN_CONFIG = {
    # Architecture
    "hidden_dims": [1024, 512, 256, 128],  # Hidden layer dimensions
    "drug_embedding_dim": 64,               # Dimension of drug embeddings
    # Alternative to one-hot encoding: learn 64-dim vector per drug

    # Regularization
    "dropout_rates": [0.3, 0.3, 0.2, 0.2],  # Dropout per layer (prevent overfitting)
    "use_batch_norm": True,                  # Use batch normalization (stabilize training)

    # Training
    "batch_size": 128,                       # Samples per batch
    "learning_rate": 0.001,                  # Initial learning rate for Adam
    "max_epochs": 200,                       # Maximum training epochs
    "early_stopping_patience": 20,           # Stop if no improvement for N epochs

    # Optimization
    "optimizer": "adam",                     # Adam optimizer (adaptive learning rate)
    "weight_decay": 1e-5,                    # L2 regularization on weights

    # Learning rate scheduler
    "use_lr_scheduler": True,
    "lr_scheduler_factor": 0.5,              # Reduce LR by this factor
    "lr_scheduler_patience": 10,             # Reduce LR if no improvement for N epochs
    "lr_scheduler_min_lr": 1e-6,             # Minimum learning rate

    # Device
    "device": "cuda",  # Use "cuda" for GPU, "cpu" for CPU
    # Will automatically fall back to CPU if CUDA not available

    # Data loading (for GPU training)
    "num_workers": 4,  # Number of CPU workers for data loading (0-8 recommended for GPU)
}

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Validation split (from training data)
VALIDATION_SIZE = 0.1  # 10% of training data for validation during training

# TensorBoard logging
USE_TENSORBOARD = True
TENSORBOARD_LOG_DIR = EXPERIMENTS_DIR / "tensorboard_logs"

# Model checkpointing
SAVE_CHECKPOINTS = True
CHECKPOINT_METRIC = "val_r2"  # Metric to monitor for saving best model
CHECKPOINT_MODE = "max"        # "max" for R², "min" for loss

# =============================================================================
# EVALUATION PARAMETERS
# =============================================================================

# Metrics to compute
METRICS = [
    "r2",           # R² (coefficient of determination)
    "rmse",         # Root Mean Squared Error
    "mae",          # Mean Absolute Error
    "spearman",     # Spearman correlation (rank-based)
    "pearson"       # Pearson correlation (linear)
]

# Per-drug analysis
COMPUTE_PER_DRUG_METRICS = True  # Compute metrics for each drug separately

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

# Figure settings
FIGURE_DPI = 300                # Resolution for publication-quality figures
FIGURE_FORMAT = "png"           # Output format
FIGURE_SIZE = (10, 8)           # Default figure size in inches

# Color palette
COLOR_PALETTE = "Set2"          # Seaborn color palette

# Font settings (Romanian labels)
FONT_FAMILY = "DejaVu Sans"     # Font that supports Romanian characters
FONT_SIZE = 12

# Labels in Romanian for thesis
ROMANIAN_LABELS = {
    "r2": "R² (Coeficient de determinare)",
    "rmse": "RMSE (Eroarea medie pătratică)",
    "mae": "MAE (Eroarea medie absolută)",
    "spearman": "Corelația Spearman",
    "pearson": "Corelația Pearson",
    "auc": "AUC",
    "ic50": "IC50",
    "gene_expression": "Expresia genică",
    "drug_response": "Răspunsul la medicament",
    "train": "Antrenare",
    "test": "Testare",
    "validation": "Validare",
    "predicted": "Prezis",
    "actual": "Real",
    "model": "Model",
    "performance": "Performanță",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "neural_network": "Rețea Neuronală",
}

# =============================================================================
# ADVANCED FEATURES
# =============================================================================

# SHAP analysis for interpretability
USE_SHAP = True
SHAP_N_SAMPLES = 1000  # Number of samples for SHAP analysis (less = faster)
SHAP_MAX_DISPLAY = 30  # Maximum features to display in SHAP plots

# Ensemble methods
USE_ENSEMBLE = True
ENSEMBLE_WEIGHTS = {
    "random_forest": 0.2,
    "xgboost": 0.4,
    "neural_network": 0.4
}  # Weights for ensemble prediction (should sum to 1)

# Hyperparameter optimization with Optuna
USE_OPTUNA = False  # Set to True to enable automated hyperparameter search
OPTUNA_N_TRIALS = 50  # Number of trials for hyperparameter optimization
OPTUNA_TIMEOUT = 3600  # Timeout in seconds (1 hour)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_experiment_dir(model_name: str, target: str) -> Path:
    """
    Get the experiment directory for a specific model and target.

    Args:
        model_name: Name of the model (e.g., "random_forest", "xgboost", "neural_network")
        target: Target variable (e.g., "AUC", "IC50")

    Returns:
        Path to the experiment directory
    """
    exp_dir = EXPERIMENTS_DIR / f"{model_name}_{target.lower()}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir

def get_model_path(model_name: str, target: str) -> Path:
    """
    Get the path to save/load a model checkpoint.

    Args:
        model_name: Name of the model
        target: Target variable

    Returns:
        Path to the model file
    """
    exp_dir = get_experiment_dir(model_name, target)

    # Different file extensions for different model types
    if model_name == "neural_network":
        return exp_dir / "model.pth"  # PyTorch models
    else:
        return exp_dir / "model.pkl"  # Scikit-learn/XGBoost models

def get_tensorboard_dir(model_name: str, target: str) -> Path:
    """
    Get the TensorBoard log directory for a specific model and target.

    Args:
        model_name: Name of the model
        target: Target variable

    Returns:
        Path to the TensorBoard log directory
    """
    tb_dir = TENSORBOARD_LOG_DIR / f"{model_name}_{target.lower()}"
    tb_dir.mkdir(parents=True, exist_ok=True)
    return tb_dir

# =============================================================================
# PRINT CONFIGURATION SUMMARY
# =============================================================================

def print_config_summary():
    """
    Print a summary of the current configuration.
    Useful for debugging and logging.
    """
    print("=" * 80)
    print("CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Random Seed: {RANDOM_SEED}")
    print(f"Top Genes: {N_TOP_GENES}")
    print(f"Min Samples per Drug: {MIN_SAMPLES_PER_DRUG}")
    print(f"Test Size: {TEST_SIZE}")
    print(f"CV Folds: {N_CV_FOLDS}")
    print(f"Targets: {TARGETS}")
    print(f"Metrics: {METRICS}")
    print(f"Use TensorBoard: {USE_TENSORBOARD}")
    print(f"Use SHAP: {USE_SHAP}")
    print(f"Use Ensemble: {USE_ENSEMBLE}")
    print("=" * 80)

if __name__ == "__main__":
    # If this file is run directly, print configuration summary
    print_config_summary()
