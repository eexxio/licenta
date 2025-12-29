"""
Data preprocessing module for GDSC dataset.

This module handles all data preprocessing steps:
1. Loading and merging data sources
2. Handling missing values
3. Filtering drugs and genes
4. Feature selection (top variance genes)
5. Normalization (Z-score)
6. Encoding categorical variables

The preprocessor uses scikit-learn's pipeline approach for reproducibility.

Author: Bachelor's Thesis Project
Date: 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import pickle
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Import configuration
from src.config import (
    N_TOP_GENES,
    MIN_SAMPLES_PER_DRUG,
    IMPUTATION_STRATEGY,
    RANDOM_SEED,
    PROCESSED_DATA_PATH,
    set_seeds
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GDSCPreprocessor:
    """
    Preprocess GDSC data for machine learning.

    Pipeline:
    1. Load gene expression + drug response data
    2. Handle missing values (impute or drop)
    3. Filter drugs (keep only drugs with enough samples)
    4. Select top genes by variance (dimensionality reduction)
    5. Normalize gene expression (Z-score)
    6. Encode drug IDs (one-hot or keep as categorical)
    7. Split features (X) and targets (y_IC50, y_AUC)
    """

    def __init__(
        self,
        n_top_genes: int = N_TOP_GENES,
        min_samples_per_drug: int = MIN_SAMPLES_PER_DRUG,
        random_seed: int = RANDOM_SEED
    ):
        """
        Initialize the preprocessor.

        Args:
            n_top_genes: Number of top-variance genes to keep
            min_samples_per_drug: Minimum samples required for a drug
            random_seed: Random seed for reproducibility
        """
        self.n_top_genes = n_top_genes
        self.min_samples_per_drug = min_samples_per_drug
        self.random_seed = random_seed

        # These will be fitted on training data
        self.gene_scaler = StandardScaler()  # For Z-score normalization
        self.imputer = SimpleImputer(strategy=IMPUTATION_STRATEGY)
        self.selected_genes = None  # Top variance genes
        self.selected_drugs = None  # Drugs with enough samples
        self.drug_to_idx = None  # Mapping from drug ID to index

        set_seeds(random_seed)
        logger.info(f"Preprocessor initialized with {n_top_genes} top genes")

    def load_gene_expression(self, path: Path) -> pd.DataFrame:
        """
        Load gene expression data from RMA normalized file.

        The file structure is typically:
        - Rows: Cell lines (COSMIC_ID or cell line names)
        - Columns: Genes (gene symbols or Ensembl IDs)
        - Values: RMA normalized expression levels

        Args:
            path: Path to gene expression file

        Returns:
            DataFrame with gene expression (cell_lines × genes)
        """
        logger.info(f"Loading gene expression from: {path}")

        # The GDSC gene expression file is tab-separated
        # First column is cell line identifier
        df = pd.read_csv(path, sep='\t', index_col=0)

        logger.info(f"Loaded expression for {len(df)} cell lines, {len(df.columns)} genes")

        return df

    def load_drug_response(self, path: Path) -> pd.DataFrame:
        """
        Load drug response data (IC50, AUC).

        Args:
            path: Path to drug response file (Excel or CSV)

        Returns:
            DataFrame with drug response data
        """
        logger.info(f"Loading drug response from: {path}")

        # GDSC fitted dose response is Excel format
        if path.suffix == '.xlsx':
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)

        logger.info(f"Loaded {len(df)} drug response measurements")

        return df

    def merge_expression_and_response(
        self,
        gene_expr_df: pd.DataFrame,
        drug_response_df: pd.DataFrame,
        cell_line_col: str = 'COSMIC_ID'
    ) -> pd.DataFrame:
        """
        Merge gene expression and drug response data.

        Each row in the result will represent one (cell_line, drug) pair
        with gene expression features and response values (IC50, AUC).

        Args:
            gene_expr_df: Gene expression matrix (cell_lines × genes)
            drug_response_df: Drug response data
            cell_line_col: Column name for cell line identifier

        Returns:
            Merged DataFrame
        """
        logger.info("Merging gene expression and drug response data...")

        # Gene expression has cell lines as index
        # Drug response has cell lines in a column (COSMIC_ID)

        # Reset index to make cell line ID a column
        gene_expr_df = gene_expr_df.reset_index()
        gene_expr_df = gene_expr_df.rename(columns={'index': cell_line_col})

        # Merge on cell line ID
        merged_df = drug_response_df.merge(
            gene_expr_df,
            on=cell_line_col,
            how='inner'  # Keep only samples with both gene expression and drug response
        )

        logger.info(f"Merged dataset: {len(merged_df)} samples")
        logger.info(f"Unique cell lines: {merged_df[cell_line_col].nunique()}")
        logger.info(f"Unique drugs: {merged_df['DRUG_ID'].nunique()}")

        return merged_df

    def filter_drugs(
        self,
        df: pd.DataFrame,
        min_samples: int = None
    ) -> pd.DataFrame:
        """
        Filter drugs to keep only those with sufficient samples.

        Rationale: Drugs with too few samples cannot be reliably modeled.
        We need at least ~30 samples per drug for meaningful predictions.

        Args:
            df: DataFrame with drug response data
            min_samples: Minimum number of samples (default: from config)

        Returns:
            Filtered DataFrame
        """
        if min_samples is None:
            min_samples = self.min_samples_per_drug

        logger.info(f"Filtering drugs with at least {min_samples} samples...")

        # Count samples per drug
        drug_counts = df['DRUG_ID'].value_counts()

        # Keep drugs with enough samples
        valid_drugs = drug_counts[drug_counts >= min_samples].index.tolist()

        logger.info(f"Before filtering: {df['DRUG_ID'].nunique()} drugs")
        logger.info(f"After filtering: {len(valid_drugs)} drugs")

        # Filter the dataframe
        df_filtered = df[df['DRUG_ID'].isin(valid_drugs)].copy()

        # Store selected drugs
        self.selected_drugs = valid_drugs

        logger.info(f"Kept {len(df_filtered):,} samples from {len(valid_drugs)} drugs")

        return df_filtered

    def select_top_genes(
        self,
        df: pd.DataFrame,
        gene_columns: List[str],
        n_top: int = None
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top N genes by variance.

        Rationale: Not all genes are equally informative.
        Genes with low variance (similar expression across samples) don't
        contribute much to predictions. Selecting top variance genes reduces
        dimensionality and improves model performance.

        Args:
            df: DataFrame with gene expression columns
            gene_columns: List of gene column names
            n_top: Number of top genes to keep (default: from config)

        Returns:
            Tuple of (filtered DataFrame, selected gene names)
        """
        if n_top is None:
            n_top = self.n_top_genes

        logger.info(f"Selecting top {n_top} genes by variance...")

        # Extract gene expression columns
        gene_expr = df[gene_columns]

        # Compute variance for each gene
        # axis=0 means compute variance across rows (samples) for each column (gene)
        gene_variances = gene_expr.var(axis=0)

        # Select top N genes with highest variance
        # .nlargest() returns the N largest values
        top_genes = gene_variances.nlargest(n_top).index.tolist()

        logger.info(f"Selected {len(top_genes)} genes out of {len(gene_columns)}")
        logger.info(f"Variance range: {gene_variances[top_genes].min():.4f} - {gene_variances[top_genes].max():.4f}")

        # Store selected genes
        self.selected_genes = top_genes

        # Keep only metadata + selected gene columns
        non_gene_cols = [col for col in df.columns if col not in gene_columns]
        df_filtered = df[non_gene_cols + top_genes].copy()

        return df_filtered, top_genes

    def handle_missing_values(
        self,
        df: pd.DataFrame,
        gene_columns: List[str]
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Strategy:
        - For gene expression: Impute with median (robust to outliers)
        - For IC50/AUC: Drop rows with missing values (can't train without target)

        Args:
            df: DataFrame with potential missing values
            gene_columns: List of gene column names

        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values...")

        initial_rows = len(df)

        # Check for missing values
        missing_summary = df.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]

        if len(missing_summary) > 0:
            logger.info(f"Found missing values in {len(missing_summary)} columns")

        # 1. Drop rows with missing IC50 or AUC
        # We cannot train without target values
        df = df.dropna(subset=['LN_IC50', 'AUC'])

        logger.info(f"Dropped {initial_rows - len(df)} rows with missing IC50/AUC")

        # 2. Impute missing gene expression values
        gene_expr = df[gene_columns].values

        if np.isnan(gene_expr).any():
            logger.info("Imputing missing gene expression values...")
            gene_expr_imputed = self.imputer.fit_transform(gene_expr)
            df[gene_columns] = gene_expr_imputed
            logger.info(f"Imputed {np.isnan(gene_expr).sum()} missing values")

        return df

    def normalize_gene_expression(
        self,
        df: pd.DataFrame,
        gene_columns: List[str],
        fit: bool = True
    ) -> pd.DataFrame:
        """
        Normalize gene expression using Z-score normalization.

        Z-score: (x - mean) / std
        This puts all genes on the same scale, preventing genes with
        naturally high expression from dominating the model.

        Args:
            df: DataFrame with gene expression
            gene_columns: List of gene column names
            fit: If True, fit the scaler (use for training data)
                 If False, use previously fitted scaler (use for test data)

        Returns:
            DataFrame with normalized gene expression
        """
        logger.info("Normalizing gene expression (Z-score)...")

        # Extract gene expression
        gene_expr = df[gene_columns].values

        # Fit and transform (training) or just transform (test)
        if fit:
            gene_expr_normalized = self.gene_scaler.fit_transform(gene_expr)
            logger.info("Fitted scaler on gene expression data")
        else:
            gene_expr_normalized = self.gene_scaler.transform(gene_expr)
            logger.info("Transformed gene expression using fitted scaler")

        # Update dataframe
        df[gene_columns] = gene_expr_normalized

        return df

    def encode_drugs(
        self,
        df: pd.DataFrame,
        method: str = 'index'
    ) -> pd.DataFrame:
        """
        Encode drug IDs.

        Methods:
        - 'index': Convert drug IDs to integer indices (0, 1, 2, ...)
                   Used for neural network drug embeddings
        - 'onehot': One-hot encoding (separate binary column for each drug)
                    Used for Random Forest and XGBoost

        Args:
            df: DataFrame with DRUG_ID column
            method: Encoding method ('index' or 'onehot')

        Returns:
            DataFrame with encoded drug IDs
        """
        logger.info(f"Encoding drugs using method: {method}")

        # Create drug to index mapping
        unique_drugs = df['DRUG_ID'].unique()
        self.drug_to_idx = {drug: idx for idx, drug in enumerate(unique_drugs)}

        if method == 'index':
            # Map drug IDs to integer indices
            df['DRUG_IDX'] = df['DRUG_ID'].map(self.drug_to_idx)
            logger.info(f"Encoded {len(unique_drugs)} drugs as indices (0-{len(unique_drugs)-1})")

        elif method == 'onehot':
            # One-hot encoding
            drug_onehot = pd.get_dummies(df['DRUG_ID'], prefix='DRUG')
            df = pd.concat([df, drug_onehot], axis=1)
            logger.info(f"Created {len(drug_onehot.columns)} one-hot encoded drug columns")

        return df

    def prepare_features_and_targets(
        self,
        df: pd.DataFrame,
        gene_columns: List[str],
        drug_encoding: str = 'index'
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into features (X) and targets (y_IC50, y_AUC).

        Args:
            df: Preprocessed DataFrame
            gene_columns: List of gene column names
            drug_encoding: How drugs are encoded ('index' or 'onehot')

        Returns:
            Tuple of (X, y_IC50, y_AUC)
        """
        logger.info("Preparing features and targets...")

        # Features: gene expression + drug encoding
        if drug_encoding == 'index':
            feature_cols = gene_columns + ['DRUG_IDX']
        else:  # onehot
            drug_cols = [col for col in df.columns if col.startswith('DRUG_')]
            feature_cols = gene_columns + drug_cols

        X = df[feature_cols].copy()

        # Targets: IC50 and AUC
        y_IC50 = df['LN_IC50'].copy()  # Natural log of IC50
        y_AUC = df['AUC'].copy()

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"IC50 target samples: {len(y_IC50)}")
        logger.info(f"AUC target samples: {len(y_AUC)}")

        return X, y_IC50, y_AUC

    def fit_transform(
        self,
        gene_expr_df: pd.DataFrame,
        drug_response_df: pd.DataFrame,
        drug_encoding: str = 'index'
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
        """
        Fit preprocessor and transform data (use for training data).

        This is the main method that runs the entire preprocessing pipeline.

        Args:
            gene_expr_df: Gene expression matrix
            drug_response_df: Drug response data
            drug_encoding: Drug encoding method

        Returns:
            Tuple of (X, y_IC50, y_AUC, gene_columns)
        """
        logger.info("=" * 80)
        logger.info("PREPROCESSING PIPELINE (FIT + TRANSFORM)")
        logger.info("=" * 80)

        # Step 1: Merge gene expression and drug response
        df = self.merge_expression_and_response(gene_expr_df, drug_response_df)

        # Identify gene columns (all columns except metadata)
        metadata_cols = ['COSMIC_ID', 'DRUG_ID', 'DRUG_NAME', 'LN_IC50', 'AUC',
                        'CELL_LINE_NAME', 'TCGA_DESC', 'TISSUE']
        gene_columns = [col for col in df.columns if col not in metadata_cols]

        logger.info(f"Initial dataset: {len(df)} samples, {len(gene_columns)} genes")

        # Step 2: Filter drugs with insufficient samples
        df = self.filter_drugs(df)

        # Step 3: Handle missing values
        df = self.handle_missing_values(df, gene_columns)

        # Step 4: Select top genes by variance
        df, selected_genes = self.select_top_genes(df, gene_columns)

        # Step 5: Normalize gene expression (fit scaler)
        df = self.normalize_gene_expression(df, selected_genes, fit=True)

        # Step 6: Encode drugs
        df = self.encode_drugs(df, method=drug_encoding)

        # Step 7: Prepare features and targets
        X, y_IC50, y_AUC = self.prepare_features_and_targets(
            df, selected_genes, drug_encoding
        )

        logger.info("=" * 80)
        logger.info("PREPROCESSING COMPLETE")
        logger.info(f"Final dataset: {len(X)} samples, {X.shape[1]} features")
        logger.info("=" * 80)

        return X, y_IC50, y_AUC, selected_genes

    def transform(
        self,
        gene_expr_df: pd.DataFrame,
        drug_response_df: pd.DataFrame,
        drug_encoding: str = 'index'
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Transform new data using fitted preprocessor (use for test data).

        This uses the scaler and gene selection fitted on training data.

        Args:
            gene_expr_df: Gene expression matrix
            drug_response_df: Drug response data
            drug_encoding: Drug encoding method

        Returns:
            Tuple of (X, y_IC50, y_AUC)
        """
        logger.info("=" * 80)
        logger.info("PREPROCESSING PIPELINE (TRANSFORM ONLY)")
        logger.info("=" * 80)

        if self.selected_genes is None:
            raise ValueError("Preprocessor not fitted! Call fit_transform() first.")

        # Merge data
        df = self.merge_expression_and_response(gene_expr_df, drug_response_df)

        # Filter to selected drugs only
        df = df[df['DRUG_ID'].isin(self.selected_drugs)].copy()

        # Keep only selected genes
        metadata_cols = ['COSMIC_ID', 'DRUG_ID', 'DRUG_NAME', 'LN_IC50', 'AUC',
                        'CELL_LINE_NAME', 'TCGA_DESC', 'TISSUE']
        df = df[metadata_cols + self.selected_genes].copy()

        # Handle missing values (using fitted imputer)
        gene_expr = df[self.selected_genes].values
        if np.isnan(gene_expr).any():
            gene_expr_imputed = self.imputer.transform(gene_expr)
            df[self.selected_genes] = gene_expr_imputed

        # Normalize (using fitted scaler)
        df = self.normalize_gene_expression(df, self.selected_genes, fit=False)

        # Encode drugs
        df = self.encode_drugs(df, method=drug_encoding)

        # Prepare features and targets
        X, y_IC50, y_AUC = self.prepare_features_and_targets(
            df, self.selected_genes, drug_encoding
        )

        logger.info("=" * 80)
        logger.info(f"Transformed {len(X)} samples with {X.shape[1]} features")
        logger.info("=" * 80)

        return X, y_IC50, y_AUC

    def save(self, path: Path):
        """Save fitted preprocessor to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Saved preprocessor to: {path}")

    @staticmethod
    def load(path: Path):
        """Load fitted preprocessor from disk."""
        with open(path, 'rb') as f:
            preprocessor = pickle.load(f)
        logger.info(f"Loaded preprocessor from: {path}")
        return preprocessor


# =============================================================================
# MAIN FUNCTION FOR TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test the preprocessor on actual GDSC data.
    """
    from src.config import ARCHIVE_DIR

    print("Testing preprocessor on GDSC data...\n")

    # File paths
    gene_expr_path = ARCHIVE_DIR / "Cell_line_RMA_proc_basalExp.txt"
    drug_response_path = ARCHIVE_DIR / "GDSC1_fitted_dose_response.xlsx"

    # Initialize preprocessor
    preprocessor = GDSCPreprocessor(n_top_genes=1000)  # Use fewer genes for testing

    try:
        # Load data
        gene_expr = preprocessor.load_gene_expression(gene_expr_path)
        drug_response = preprocessor.load_drug_response(drug_response_path)

        # Fit and transform
        X, y_IC50, y_AUC, genes = preprocessor.fit_transform(
            gene_expr,
            drug_response,
            drug_encoding='index'
        )

        print("\n✓ Preprocessing test successful!")
        print(f"Features: {X.shape}")
        print(f"IC50: {y_IC50.shape}")
        print(f"AUC: {y_AUC.shape}")

    except Exception as e:
        print(f"\n✗ Preprocessing test failed: {e}")
