"""
Script to preprocess GDSC data for drug response prediction.

This script:
1. Loads raw GDSC data (gene expression + drug response)
2. Runs the complete preprocessing pipeline
3. Saves processed data to data/processed/
4. Prints summary statistics

Run this script BEFORE training models!

Usage:
    python scripts/01_preprocess_data.py

Author: Bachelor's Thesis Project
Date: 2026
"""

import os
import sys
import gc
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    N_TOP_GENES,
    MIN_SAMPLES_PER_DRUG,
    RANDOM_SEED,
    set_seeds
)
from src.data.loader import (
    load_gene_expression,
    load_gdsc_drug_response,
    merge_expression_and_response
)
from src.data.splitter import GDSCDataSplitter

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main preprocessing pipeline."""

    logger.info("=" * 80)
    logger.info("GDSC DATA PREPROCESSING PIPELINE")
    logger.info("=" * 80)

    # Set random seed for reproducibility
    set_seeds()
    logger.info(f"Random seed set to: {RANDOM_SEED}")

    # =========================================================================
    # STEP 1: LOAD RAW DATA
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: LOADING RAW DATA")
    logger.info("=" * 80)

    # Load gene expression data
    gene_expr_path = RAW_DATA_DIR / "Cell_line_RMA_proc_basalExp.txt"
    logger.info(f"Loading gene expression from: {gene_expr_path}")

    try:
        gene_expr_df = load_gene_expression(gene_expr_path)
        logger.info(f"✓ Gene expression loaded: {gene_expr_df.shape}")
        logger.info(f"  - Genes: {gene_expr_df.shape[1]}")
        logger.info(f"  - Cell lines: {gene_expr_df.shape[0]}")
    except Exception as e:
        logger.error(f"Failed to load gene expression: {e}")
        logger.info("\nNote: You may need to adapt the loader for the actual GDSC format.")
        logger.info("Check the file structure and update src/data/loader.py accordingly.")
        sys.exit(1)

    # Load drug response data
    drug_response_path = RAW_DATA_DIR / "GDSC1_fitted_dose_response.xlsx"
    logger.info(f"\nLoading drug response from: {drug_response_path}")

    try:
        drug_response_df = load_gdsc_drug_response(drug_response_path)
        logger.info(f"✓ Drug response loaded: {drug_response_df.shape}")
        logger.info(f"  - Rows: {len(drug_response_df)}")
        logger.info(f"  - Columns: {drug_response_df.columns.tolist()}")
    except Exception as e:
        logger.error(f"Failed to load drug response: {e}")
        sys.exit(1)

    # =========================================================================
    # STEP 2: SPLIT CELL LINES FIRST (BEFORE MERGING - MEMORY EFFICIENT)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: SPLITTING CELL LINES (BEFORE MERGING)")
    logger.info("=" * 80)
    logger.info("CRITICAL: We split cell lines FIRST to prevent data leakage!")
    logger.info("Gene selection will be done ONLY on training cell lines.")

    # Get cell lines that exist in both gene expression and drug response
    expr_cell_lines = set(gene_expr_df.index)
    drug_cell_lines = set(drug_response_df['COSMIC_ID'].unique())
    common_cell_lines = list(expr_cell_lines & drug_cell_lines)
    
    logger.info(f"Cell lines in expression data: {len(expr_cell_lines)}")
    logger.info(f"Cell lines in drug response: {len(drug_cell_lines)}")
    logger.info(f"Common cell lines: {len(common_cell_lines)}")

    # Split cell lines into train/test (80/20)
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(common_cell_lines)
    n_test = int(len(common_cell_lines) * 0.2)
    test_cell_lines = set(common_cell_lines[:n_test])
    train_cell_lines = set(common_cell_lines[n_test:])

    logger.info(f"✓ Cell line split:")
    logger.info(f"  - Train cell lines: {len(train_cell_lines)}")
    logger.info(f"  - Test cell lines: {len(test_cell_lines)}")
    logger.info("  - No overlap between train and test!")

    # =========================================================================
    # STEP 3: SELECT TOP GENES (ONLY FROM TRAINING CELL LINES - NO LEAKAGE!)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: GENE SELECTION (TRAINING CELL LINES ONLY)")
    logger.info("=" * 80)
    logger.info(f"Selecting top {N_TOP_GENES} genes by variance from TRAINING cell lines only")

    try:
        # Get gene expression for training cell lines only
        train_cell_lines_list = [cl for cl in gene_expr_df.index if cl in train_cell_lines]
        gene_expr_train = gene_expr_df.loc[train_cell_lines_list]

        # Calculate variance for each gene (ONLY from training cell lines!)
        logger.info(f"Calculating variance for {gene_expr_train.shape[1]} genes...")
        gene_variances = gene_expr_train.var(axis=0).values  # Convert to numpy array

        # Select top N genes with highest variance
        top_gene_indices = np.argsort(gene_variances)[-N_TOP_GENES:]
        selected_gene_columns = gene_expr_df.columns[top_gene_indices].tolist()

        logger.info(f"✓ Selected {len(selected_gene_columns)} genes based on training variance")

        # Filter gene expression to keep only selected genes (for ALL cell lines)
        gene_expr_filtered = gene_expr_df.iloc[:, top_gene_indices].copy()
        logger.info(f"  - Filtered gene expression shape: {gene_expr_filtered.shape}")

        # Rename columns with GENE_ prefix
        gene_expr_filtered.columns = [f"GENE_{col}" for col in gene_expr_filtered.columns]
        selected_genes = gene_expr_filtered.columns.tolist()

        # Free memory
        del gene_expr_train, gene_expr_df
        import gc
        gc.collect()

    except Exception as e:
        logger.error(f"Failed to select genes: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # =========================================================================
    # STEP 4: CREATE TRAIN AND TEST DATASETS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: CREATING TRAIN AND TEST DATASETS")
    logger.info("=" * 80)

    try:
        # Filter drug response for train/test cell lines
        train_drug_response = drug_response_df[drug_response_df['COSMIC_ID'].isin(train_cell_lines)].copy()
        test_drug_response = drug_response_df[drug_response_df['COSMIC_ID'].isin(test_cell_lines)].copy()

        logger.info(f"Train drug response samples: {len(train_drug_response)}")
        logger.info(f"Test drug response samples: {len(test_drug_response)}")

        # Free memory
        del drug_response_df
        gc.collect()

    except Exception as e:
        logger.error(f"Failed to filter drug response: {e}")
        sys.exit(1)

    # =========================================================================
    # STEP 5: NORMALIZE TRAINING DATA
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: NORMALIZING AND PREPARING TRAINING DATA")
    logger.info("=" * 80)

    try:
        # Get gene expression for training cell lines
        train_expr = gene_expr_filtered.loc[gene_expr_filtered.index.isin(train_cell_lines)]
        
        # Fit scaler on training data
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        train_expr_normalized = scaler.fit_transform(train_expr.values.astype(np.float32))
        
        # Create training data arrays
        X_train_list = []
        y_train_ic50_list = []
        y_train_auc_list = []
        drug_ids_train_list = []
        cosmic_ids_train_list = []

        for cosmic_id in train_cell_lines:
            if cosmic_id not in train_expr.index:
                continue
            
            # Get gene expression for this cell line
            cell_idx = train_expr.index.get_loc(cosmic_id)
            cell_expr = train_expr_normalized[cell_idx]
            
            # Get all drug responses for this cell line
            cell_drugs = train_drug_response[train_drug_response['COSMIC_ID'] == cosmic_id]
            
            for _, row in cell_drugs.iterrows():
                X_train_list.append(np.append(cell_expr, row['DRUG_ID']))
                y_train_ic50_list.append(row['LN_IC50'])
                y_train_auc_list.append(row['AUC'])
                drug_ids_train_list.append(row['DRUG_ID'])
                cosmic_ids_train_list.append(cosmic_id)

        X_train = np.array(X_train_list, dtype=np.float32)
        y_train_ic50 = np.array(y_train_ic50_list, dtype=np.float32)
        y_train_auc = np.array(y_train_auc_list, dtype=np.float32)
        drug_ids_train = np.array(drug_ids_train_list)
        cosmic_ids_train = np.array(cosmic_ids_train_list)

        logger.info(f"✓ Training data prepared:")
        logger.info(f"  - Features shape: {X_train.shape}")
        logger.info(f"  - IC50 targets: {y_train_ic50.shape}")
        logger.info(f"  - AUC targets: {y_train_auc.shape}")
        logger.info(f"  - Unique drugs: {len(np.unique(drug_ids_train))}")
        logger.info(f"  - Unique cell lines: {len(np.unique(cosmic_ids_train))}")

        # Free memory
        del X_train_list, y_train_ic50_list, y_train_auc_list, drug_ids_train_list, cosmic_ids_train_list
        del train_drug_response, train_expr_normalized
        gc.collect()

    except Exception as e:
        logger.error(f"Failed to prepare training data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # =========================================================================
    # STEP 6: NORMALIZE TEST DATA
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: NORMALIZING AND PREPARING TEST DATA")
    logger.info("=" * 80)
    logger.info("Using scaler fitted on training data (no data leakage!)")

    try:
        # Get gene expression for test cell lines
        test_expr = gene_expr_filtered.loc[gene_expr_filtered.index.isin(test_cell_lines)]
        
        # Transform test data using training scaler
        test_expr_normalized = scaler.transform(test_expr.values.astype(np.float32))
        
        # Create test data arrays
        X_test_list = []
        y_test_ic50_list = []
        y_test_auc_list = []
        drug_ids_test_list = []
        cosmic_ids_test_list = []

        for cosmic_id in test_cell_lines:
            if cosmic_id not in test_expr.index:
                continue
            
            # Get gene expression for this cell line
            cell_idx = test_expr.index.get_loc(cosmic_id)
            cell_expr = test_expr_normalized[cell_idx]
            
            # Get all drug responses for this cell line
            cell_drugs = test_drug_response[test_drug_response['COSMIC_ID'] == cosmic_id]
            
            for _, row in cell_drugs.iterrows():
                X_test_list.append(np.append(cell_expr, row['DRUG_ID']))
                y_test_ic50_list.append(row['LN_IC50'])
                y_test_auc_list.append(row['AUC'])
                drug_ids_test_list.append(row['DRUG_ID'])
                cosmic_ids_test_list.append(cosmic_id)

        X_test = np.array(X_test_list, dtype=np.float32)
        y_test_ic50 = np.array(y_test_ic50_list, dtype=np.float32)
        y_test_auc = np.array(y_test_auc_list, dtype=np.float32)
        drug_ids_test = np.array(drug_ids_test_list)
        cosmic_ids_test = np.array(cosmic_ids_test_list)

        logger.info(f"✓ Test data prepared:")
        logger.info(f"  - Features shape: {X_test.shape}")
        logger.info(f"  - IC50 targets: {y_test_ic50.shape}")
        logger.info(f"  - AUC targets: {y_test_auc.shape}")
        logger.info(f"  - Unique drugs: {len(np.unique(drug_ids_test))}")
        logger.info(f"  - Unique cell lines: {len(np.unique(cosmic_ids_test))}")

        # Free memory
        del X_test_list, y_test_ic50_list, y_test_auc_list, drug_ids_test_list, cosmic_ids_test_list
        del test_drug_response, test_expr_normalized, gene_expr_filtered
        gc.collect()

    except Exception as e:
        logger.error(f"Failed to prepare test data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # =========================================================================
    # STEP 7: SAVE PREPROCESSED DATA
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: SAVING PREPROCESSED DATA")
    logger.info("=" * 80)

    # Create output directory
    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)

    # Save training data
    train_data = {
        'X_train': X_train,
        'y_train_ic50': y_train_ic50,
        'y_train_auc': y_train_auc,
        'drug_ids_train': drug_ids_train,
        'cosmic_ids_train': cosmic_ids_train,  # For cell line-based validation split
        'selected_genes': selected_genes
    }

    train_path = PROCESSED_DATA_DIR / 'train_data.npz'
    np.savez(train_path, **train_data)
    logger.info(f"✓ Training data saved to: {train_path}")

    # Save test data
    test_data = {
        'X_test': X_test,
        'y_test_ic50': y_test_ic50,
        'y_test_auc': y_test_auc,
        'drug_ids_test': drug_ids_test,
        'cosmic_ids_test': cosmic_ids_test  # Added for completeness
    }

    test_path = PROCESSED_DATA_DIR / 'test_data.npz'
    np.savez(test_path, **test_data)
    logger.info(f"✓ Test data saved to: {test_path}")

    # Save scaler
    import pickle
    scaler_path = PROCESSED_DATA_DIR / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"✓ Scaler saved to: {scaler_path}")

    # Save gene names
    gene_names_path = PROCESSED_DATA_DIR / 'selected_genes.txt'
    with open(gene_names_path, 'w') as f:
        for gene in selected_genes:
            f.write(f"{gene}\n")
    logger.info(f"✓ Gene names saved to: {gene_names_path}")

    # =========================================================================
    # STEP 8: SUMMARY STATISTICS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    logger.info("\nDataset Statistics:")
    logger.info(f"  Training samples: {len(X_train):,}")
    logger.info(f"  Test samples: {len(X_test):,}")
    logger.info(f"  Train/test split: {len(X_train)/(len(X_train)+len(X_test))*100:.1f}% / {len(X_test)/(len(X_train)+len(X_test))*100:.1f}%")

    logger.info("\nFeature Statistics:")
    logger.info(f"  Selected genes: {len(selected_genes):,}")
    logger.info(f"  Unique drugs (train): {len(np.unique(drug_ids_train)):,}")
    logger.info(f"  Unique drugs (test): {len(np.unique(drug_ids_test)):,}")
    logger.info(f"  Total features: {X_train.shape[1]:,}")

    logger.info("\nCell Line Statistics:")
    logger.info(f"  Train cell lines: {len(train_cell_lines):,}")
    logger.info(f"  Test cell lines: {len(test_cell_lines):,}")

    logger.info("\nTarget Statistics (Training Set):")
    logger.info(f"  IC50 (log-scale):")
    logger.info(f"    - Mean: {y_train_ic50.mean():.3f}")
    logger.info(f"    - Std: {y_train_ic50.std():.3f}")
    logger.info(f"    - Min: {y_train_ic50.min():.3f}")
    logger.info(f"    - Max: {y_train_ic50.max():.3f}")

    logger.info(f"  AUC:")
    logger.info(f"    - Mean: {y_train_auc.mean():.3f}")
    logger.info(f"    - Std: {y_train_auc.std():.3f}")
    logger.info(f"    - Min: {y_train_auc.min():.3f}")
    logger.info(f"    - Max: {y_train_auc.max():.3f}")

    logger.info("\n" + "=" * 80)
    logger.info("✓ PREPROCESSING COMPLETE - NO DATA LEAKAGE!")
    logger.info("=" * 80)
    logger.info("  - Gene selection: Based on TRAINING cell lines only")
    logger.info("  - Normalization: Fitted on TRAINING data only")
    logger.info("  - Cell lines: No overlap between train and test")
    logger.info("\nNext steps:")
    logger.info("  1. Train models: python scripts/02_train_models.py")
    logger.info("  2. Evaluate models: python scripts/03_evaluate_models.py")
    logger.info("  3. Generate figures: python scripts/04_generate_figures.py")
    logger.info("")


if __name__ == "__main__":
    main()
