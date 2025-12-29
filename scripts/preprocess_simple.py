"""Simple preprocessing script - minimal version to get data ready quickly."""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ARCHIVE_DIR, PROCESSED_DATA_DIR, N_TOP_GENES, RANDOM_SEED
from src.data.loader import load_gene_expression, load_gdsc_drug_response, merge_expression_and_response
from src.data.splitter import GDSCDataSplitter

print("=" * 80)
print("SIMPLE GDSC PREPROCESSING")
print("=" * 80)

# Set seed
np.random.seed(RANDOM_SEED)

# Step 1: Load data
print("\n1. Loading gene expression...")
gene_expr_path = ARCHIVE_DIR / "Cell_line_RMA_proc_basalExp.txt"
gene_expr_df = load_gene_expression(gene_expr_path)
print(f"   Loaded: {gene_expr_df.shape}")

print("\n2. Loading drug response...")
drug_response_path = ARCHIVE_DIR / "GDSC1_fitted_dose_response.xlsx"
drug_response_df = load_gdsc_drug_response(drug_response_path)
print(f"   Loaded: {drug_response_df.shape}")

# Step 2: Select top genes
print(f"\n3. Selecting top {N_TOP_GENES} genes...")
gene_variances = gene_expr_df.var(axis=0)
top_gene_positions = gene_variances.argsort()[-N_TOP_GENES:].values
gene_expr_df_filtered = gene_expr_df.iloc[:, top_gene_positions]
print(f"   Selected: {gene_expr_df_filtered.shape}")

# Step 3: Merge
print("\n4. Merging...")
merged_df = merge_expression_and_response(gene_expr_df_filtered, drug_response_df)
print(f"   Merged: {merged_df.shape}")

# Step 4: Split by cell line
print("\n5. Splitting by cell line...")
splitter = GDSCDataSplitter()
train_df, test_df = splitter.split_by_cell_line(merged_df, test_size=0.2, cell_line_col='COSMIC_ID')
print(f"   Train: {train_df.shape}, Test: {test_df.shape}")

# Step 5: Normalize
print("\n6. Normalizing...")
gene_columns = [col for col in train_df.columns if col.startswith('GENE_')]
print(f"   Found {len(gene_columns)} gene columns")

# Train
X_train_genes = train_df[gene_columns].values
scaler = StandardScaler()
X_train_genes_norm = scaler.fit_transform(X_train_genes)
drug_ids_train = train_df['DRUG_ID'].values
X_train = np.column_stack([X_train_genes_norm, drug_ids_train])
y_train_ic50 = train_df['LN_IC50'].values
y_train_auc = train_df['AUC'].values
print(f"   Train normalized: {X_train.shape}")

# Test
X_test_genes = test_df[gene_columns].values
X_test_genes_norm = scaler.transform(X_test_genes)
drug_ids_test = test_df['DRUG_ID'].values
X_test = np.column_stack([X_test_genes_norm, drug_ids_test])
y_test_ic50 = test_df['LN_IC50'].values
y_test_auc = test_df['AUC'].values
print(f"   Test normalized: {X_test.shape}")

# Step 6: Save
print("\n7. Saving...")
Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)

# Save train data
train_path = PROCESSED_DATA_DIR / 'train_data.npz'
np.savez(train_path,
         X_train=X_train,
         y_train_ic50=y_train_ic50,
         y_train_auc=y_train_auc,
         drug_ids_train=drug_ids_train,
         selected_genes=gene_columns)
print(f"   Saved: {train_path}")

# Save test data
test_path = PROCESSED_DATA_DIR / 'test_data.npz'
np.savez(test_path,
         X_test=X_test,
         y_test_ic50=y_test_ic50,
         y_test_auc=y_test_auc,
         drug_ids_test=drug_ids_test)
print(f"   Saved: {test_path}")

# Save scaler
scaler_path = PROCESSED_DATA_DIR / 'scaler.pkl'
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"   Saved: {scaler_path}")

print("\n" + "=" * 80)
print("✓ PREPROCESSING COMPLETE!")
print("=" * 80)
print(f"\nDataset summary:")
print(f"  Train samples: {len(X_train):,}")
print(f"  Test samples: {len(X_test):,}")
print(f"  Features: {X_train.shape[1]:,} ({len(gene_columns)} genes + 1 drug ID)")
print(f"  Unique drugs (train): {len(np.unique(drug_ids_train))}")
print(f"  Unique drugs (test): {len(np.unique(drug_ids_test))}")
