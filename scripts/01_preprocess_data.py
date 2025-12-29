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
    # STEP 2: SELECT TOP GENES (BEFORE MERGING - MEMORY OPTIMIZATION)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: GENE SELECTION (BEFORE MERGING)")
    logger.info("=" * 80)
    logger.info(f"Selecting top {N_TOP_GENES} genes by variance to reduce memory usage")
    logger.info(f"Original genes: {gene_expr_df.shape[1]}")

    try:
        # Calculate variance for each gene (across cell lines)
        logger.info(f"Calculating variance for {gene_expr_df.shape[1]} genes...")
        gene_variances = gene_expr_df.var(axis=0)

        # Select top N genes with highest variance using integer-based indexing
        # argsort returns integer positions, we want the top N
        top_gene_positions = gene_variances.argsort()[-N_TOP_GENES:].values

        logger.info(f"Selected {len(top_gene_positions)} gene positions (expected {N_TOP_GENES})")

        # Filter gene expression to keep only top genes using integer positions
        gene_expr_df_filtered = gene_expr_df.iloc[:, top_gene_positions].copy()

        logger.info(f"✓ Gene selection complete:")
        logger.info(f"  - Original shape: {gene_expr_df.shape}")
        logger.info(f"  - Filtered shape: {gene_expr_df_filtered.shape}")
        logger.info(f"  - Memory reduction: {gene_expr_df.shape[1]} → {gene_expr_df_filtered.shape[1]} genes")

        if gene_expr_df_filtered.shape[1] != N_TOP_GENES:
            logger.error(f"ERROR: Expected {N_TOP_GENES} genes but got {gene_expr_df_filtered.shape[1]}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Failed to select genes: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # =========================================================================
    # STEP 3: MERGE DATA
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: MERGING GENE EXPRESSION + DRUG RESPONSE")
    logger.info("=" * 80)

    try:
        merged_df = merge_expression_and_response(gene_expr_df_filtered, drug_response_df)
        logger.info(f"✓ Data merged: {merged_df.shape}")
        logger.info(f"  - Total samples: {len(merged_df)}")
        logger.info(f"  - Unique cell lines: {merged_df['COSMIC_ID'].nunique()}")
        logger.info(f"  - Unique drugs: {merged_df['DRUG_ID'].nunique()}")
    except Exception as e:
        logger.error(f"Failed to merge data: {e}")
        sys.exit(1)

    # =========================================================================
    # STEP 4: SPLIT DATA (BY CELL LINE)
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: SPLITTING INTO TRAIN/TEST (BY CELL LINE)")
    logger.info("=" * 80)
    logger.info("CRITICAL: We split by cell line to prevent data leakage!")
    logger.info("This ensures no cell line appears in both train and test sets.")

    splitter = GDSCDataSplitter()

    try:
        train_df, test_df = splitter.split_by_cell_line(
            merged_df,
            test_size=0.2,
            cell_line_col='COSMIC_ID'
        )

        logger.info(f"✓ Data split complete:")
        logger.info(f"  - Training samples: {len(train_df)}")
        logger.info(f"  - Test samples: {len(test_df)}")
        logger.info(f"  - Train cell lines: {train_df['COSMIC_ID'].nunique()}")
        logger.info(f"  - Test cell lines: {test_df['COSMIC_ID'].nunique()}")

        # Verify no overlap
        train_cell_lines = set(train_df['COSMIC_ID'].unique())
        test_cell_lines = set(test_df['COSMIC_ID'].unique())
        overlap = train_cell_lines & test_cell_lines

        if len(overlap) > 0:
            logger.error(f"ERROR: Found {len(overlap)} cell lines in both train and test!")
            logger.error("This is data leakage and will invalidate results!")
            sys.exit(1)
        else:
            logger.info("✓ No overlap between train and test cell lines - data leakage prevented!")

    except Exception as e:
        logger.error(f"Failed to split data: {e}")
        sys.exit(1)

    # Save split indices for reproducibility
    splits_dir = Path(PROCESSED_DATA_DIR) / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    train_indices_path = splits_dir / "train_indices.txt"
    test_indices_path = splits_dir / "test_indices.txt"

    train_df.index.to_series().to_csv(train_indices_path, index=False, header=False)
    test_df.index.to_series().to_csv(test_indices_path, index=False, header=False)

    logger.info(f"\n✓ Split indices saved:")
    logger.info(f"  - Train: {train_indices_path}")
    logger.info(f"  - Test: {test_indices_path}")

    # =========================================================================
    # STEP 5: NORMALIZE TRAINING DATA AND PREPARE ARRAYS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: NORMALIZING TRAINING DATA")
    logger.info("=" * 80)

    try:
        # Extract gene columns from training data
        gene_columns = [col for col in train_df.columns if col.startswith('GENE_')]
        # Use float32 to reduce memory usage (3.41 GiB -> ~1.7 GiB)
        X_train_genes = train_df[gene_columns].values.astype(np.float32)

        # Normalize gene expression (StandardScaler - zero mean, unit variance)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_genes_normalized = scaler.fit_transform(X_train_genes).astype(np.float32)

        # Extract drug IDs
        drug_ids_train = train_df['DRUG_ID'].values

        # Prepare features: [gene_expression, drug_id]
        X_train = np.column_stack([X_train_genes_normalized, drug_ids_train])

        # Extract targets
        y_train_ic50 = train_df['LN_IC50'].values
        y_train_auc = train_df['AUC'].values

        selected_genes = gene_columns

        logger.info(f"✓ Training data normalized:")
        logger.info(f"  - Features shape: {X_train.shape}")
        logger.info(f"  - IC50 targets: {y_train_ic50.shape}")
        logger.info(f"  - AUC targets: {y_train_auc.shape}")
        logger.info(f"  - Number of genes: {len(selected_genes)}")
        logger.info(f"  - Unique drugs: {len(np.unique(drug_ids_train))}")

    except Exception as e:
        logger.error(f"Failed to normalize training data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # =========================================================================
    # STEP 6: NORMALIZE TEST DATA
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: NORMALIZING TEST DATA")
    logger.info("=" * 80)
    logger.info("Using fitted scaler from training data (no data leakage!)")

    try:
        # Extract gene columns from test data (same columns as training)
        # Use float32 to reduce memory usage (3.41 GiB -> ~1.7 GiB)
        X_test_genes = test_df[gene_columns].values.astype(np.float32)

        # Normalize using training scaler (no data leakage!)
        X_test_genes_normalized = scaler.transform(X_test_genes).astype(np.float32)

        # Extract drug IDs
        drug_ids_test = test_df['DRUG_ID'].values

        # Prepare features: [gene_expression, drug_id]
        X_test = np.column_stack([X_test_genes_normalized, drug_ids_test])

        # Extract targets
        y_test_ic50 = test_df['LN_IC50'].values
        y_test_auc = test_df['AUC'].values

        logger.info(f"✓ Test data normalized:")
        logger.info(f"  - Features shape: {X_test.shape}")
        logger.info(f"  - IC50 targets: {y_test_ic50.shape}")
        logger.info(f"  - AUC targets: {y_test_auc.shape}")
        logger.info(f"  - Unique drugs: {len(np.unique(drug_ids_test))}")

    except Exception as e:
        logger.error(f"Failed to normalize test data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # =========================================================================
    # STEP 6: SAVE PREPROCESSED DATA
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: SAVING PREPROCESSED DATA")
    logger.info("=" * 80)

    # Create output directory
    Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)

    # Save training data
    train_data = {
        'X_train': X_train,
        'y_train_ic50': y_train_ic50,
        'y_train_auc': y_train_auc,
        'drug_ids_train': drug_ids_train,
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
        'drug_ids_test': drug_ids_test
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
    # STEP 7: SUMMARY STATISTICS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    logger.info("\nDataset Statistics:")
    logger.info(f"  Total samples: {len(merged_df):,}")
    logger.info(f"  Training samples: {len(X_train):,}")
    logger.info(f"  Test samples: {len(X_test):,}")
    logger.info(f"  Train/test split: {len(X_train)/(len(X_train)+len(X_test))*100:.1f}% / {len(X_test)/(len(X_train)+len(X_test))*100:.1f}%")

    logger.info("\nFeature Statistics:")
    logger.info(f"  Original genes: {gene_expr_df.shape[1]:,}")
    logger.info(f"  Selected genes: {len(selected_genes):,}")
    logger.info(f"  Unique drugs (after filtering): {len(np.unique(drug_ids_train)):,}")
    logger.info(f"  Total features: {X_train.shape[1]:,}")

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
    logger.info("✓ PREPROCESSING COMPLETE!")
    logger.info("=" * 80)
    logger.info("\nNext steps:")
    logger.info("  1. Train models: python scripts/02_train_models.py")
    logger.info("  2. Evaluate models: python scripts/03_evaluate_models.py")
    logger.info("  3. Generate figures: python scripts/04_generate_figures.py")
    logger.info("")


if __name__ == "__main__":
    main()
