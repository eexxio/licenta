"""
Data loading module for GDSC dataset.

This module provides functions to load and combine different data sources:
- GDSC drug response data (IC50, AUC)
- Gene expression data
- Compound (drug) annotations
- Cell line details

Author: Bachelor's Thesis Project
Date: 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import logging

# Import configuration
from src.config import (
    GDSC_DATASET_PATH,
    COMPOUNDS_ANNOTATION_PATH,
    CELL_LINES_DETAILS_PATH
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_gdsc_drug_response(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load GDSC drug response data.

    The GDSC drug response file contains drug response measurements (IC50, AUC)
    for different cell lines tested against various drugs.

    Supported formats:
    - CSV files (.csv): GDSC_DATASET.csv, GDSC1_public_raw_data_27Oct23.csv
    - Excel files (.xlsx): GDSC1_fitted_dose_response.xlsx

    Structure:
    - Each row represents one (cell_line, drug) pair
    - Columns may include: COSMIC_ID, CELL_LINE_NAME, DRUG_ID, DRUG_NAME, IC50, AUC, LN_IC50
    - Also includes tissue type, cancer type, and other metadata

    Args:
        path: Path to GDSC dataset file (CSV or Excel, default: from config)

    Returns:
        DataFrame with drug response data

    Example:
        >>> df = load_gdsc_drug_response()
        >>> print(df.columns)
        >>> print(f"Shape: {df.shape}")
    """
    if path is None:
        path = GDSC_DATASET_PATH

    logger.info(f"Loading GDSC drug response data from {path}")

    # Check if file exists
    if not Path(path).exists():
        raise FileNotFoundError(
            f"GDSC dataset not found at {path}. "
            f"Please ensure the data file is in the correct location."
        )

    # Determine file type and load accordingly
    path_str = str(path).lower()
    if path_str.endswith('.xlsx') or path_str.endswith('.xls'):
        logger.info("Detected Excel format - using pd.read_excel()")
        df = pd.read_excel(path)
    elif path_str.endswith('.csv'):
        logger.info("Detected CSV format - using pd.read_csv()")
        df = pd.read_csv(path, low_memory=False)
    else:
        # Try CSV as default
        logger.warning(f"Unknown file extension, attempting to read as CSV")
        df = pd.read_csv(path, low_memory=False)

    logger.info(f"Loaded {len(df)} drug response measurements")
    logger.info(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns

    # Try to report cell line and drug counts if columns exist
    if 'COSMIC_ID' in df.columns:
        logger.info(f"Unique cell lines (COSMIC_ID): {df['COSMIC_ID'].nunique()}")
    elif 'COSMIC identifier' in df.columns:
        logger.info(f"Unique cell lines (COSMIC identifier): {df['COSMIC identifier'].nunique()}")

    if 'DRUG_ID' in df.columns:
        logger.info(f"Unique drugs (DRUG_ID): {df['DRUG_ID'].nunique()}")
    elif 'DRUG_NAME' in df.columns:
        logger.info(f"Unique drugs (DRUG_NAME): {df['DRUG_NAME'].nunique()}")

    return df


def load_compound_annotations(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load compound (drug) annotation data.

    The Compounds-annotation.csv file contains metadata about drugs:
    - Drug names, IDs, targets, pathways
    - Chemical properties
    - PubChem IDs

    Args:
        path: Path to compound annotations CSV file (default: from config)

    Returns:
        DataFrame with compound metadata

    Note:
        This can be used to add drug features to the main dataset
        or for filtering/analysis by drug properties.
    """
    if path is None:
        path = COMPOUNDS_ANNOTATION_PATH

    logger.info(f"Loading compound annotations from {path}")

    if not Path(path).exists():
        logger.warning(f"Compound annotations file not found at {path}")
        return pd.DataFrame()  # Return empty DataFrame

    df = pd.read_csv(path)

    logger.info(f"Loaded annotations for {len(df)} compounds")

    return df


def load_cell_line_details(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load cell line details.

    The Cell_Lines_Details.xlsx file contains detailed information about
    cancer cell lines used in the GDSC project.

    Args:
        path: Path to cell line details Excel file (default: from config)

    Returns:
        DataFrame with cell line metadata

    Note:
        Requires openpyxl library to read Excel files.
    """
    if path is None:
        path = CELL_LINES_DETAILS_PATH

    logger.info(f"Loading cell line details from {path}")

    if not Path(path).exists():
        logger.warning(f"Cell line details file not found at {path}")
        return pd.DataFrame()

    # Read Excel file
    # openpyxl is required (installed via requirements.txt)
    df = pd.read_excel(path)

    logger.info(f"Loaded details for {len(df)} cell lines")

    return df


def load_gene_expression(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load gene expression data from GDSC Cell_line_RMA_proc_basalExp.txt.

    The GDSC gene expression file format:
    - Tab-separated file
    - Column 1: GENE_SYMBOLS (gene names)
    - Column 2: GENE_title (gene descriptions)
    - Columns 3+: DATA.XXXXXX where XXXXXX is the COSMIC_ID

    We transpose this to get:
    - Rows: cell lines (indexed by COSMIC_ID)
    - Columns: genes (prefixed with "GENE_")

    Args:
        path: Path to gene expression data file

    Returns:
        DataFrame with gene expression data
        Rows: cell lines (COSMIC_ID as index)
        Columns: genes (GENE_SYMBOL format)
        Values: RMA normalized expression levels

    Example:
        >>> gene_expr = load_gene_expression("archive/Cell_line_RMA_proc_basalExp.txt")
        >>> gene_expr.shape
        (1001, 17419)  # 1001 cell lines × 17,419 genes
    """
    if path is None:
        raise ValueError("Path to gene expression file is required")

    logger.info(f"Loading gene expression data from {path}")

    # Read the tab-separated file
    # This file has genes as rows and cell lines as columns
    df = pd.read_csv(path, sep='\t')

    logger.info(f"Raw data shape: {df.shape}")

    # Extract gene symbols (column 1)
    gene_symbols = df['GENE_SYMBOLS'].values

    # Get cell line columns (all columns starting with "DATA.")
    cell_line_cols = [col for col in df.columns if col.startswith('DATA.')]

    logger.info(f"Found {len(gene_symbols)} genes and {len(cell_line_cols)} cell lines")

    # Extract expression matrix (genes × cell lines)
    expression_matrix = df[cell_line_cols].values

    # Transpose to get (cell lines × genes)
    expression_matrix_T = expression_matrix.T

    # Extract COSMIC IDs from column names (remove "DATA." prefix)
    # Some IDs may have decimal points (e.g., "DATA.1503362.1"), extract just the integer part
    cosmic_ids = []
    for col in cell_line_cols:
        id_str = col.replace('DATA.', '')
        # Take only the part before the first decimal point
        if '.' in id_str:
            id_str = id_str.split('.')[0]
        cosmic_ids.append(int(id_str))

    # Create gene column names with "GENE_" prefix
    gene_col_names = [f"GENE_{symbol}" for symbol in gene_symbols]

    # Create DataFrame with cell lines as rows and genes as columns
    gene_expr_df = pd.DataFrame(
        expression_matrix_T,
        index=cosmic_ids,
        columns=gene_col_names
    )

    # Name the index
    gene_expr_df.index.name = 'COSMIC_ID'

    # Handle duplicate COSMIC_IDs (keep first occurrence)
    if gene_expr_df.index.duplicated().any():
        n_duplicates = gene_expr_df.index.duplicated().sum()
        logger.warning(f"Found {n_duplicates} duplicate COSMIC_IDs - keeping first occurrence")
        gene_expr_df = gene_expr_df[~gene_expr_df.index.duplicated(keep='first')]

    logger.info(f"Gene expression data loaded: {gene_expr_df.shape}")
    logger.info(f"  - Cell lines: {len(gene_expr_df)}")
    logger.info(f"  - Genes: {len(gene_expr_df.columns)}")
    logger.info(f"  - Sample cell line IDs: {gene_expr_df.index[:5].tolist()}")

    return gene_expr_df


def merge_expression_and_response(
    gene_expression_df: pd.DataFrame,
    drug_response_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge gene expression and drug response data.

    This is a simplified version of merge_all_data that specifically handles
    the two main data sources needed for model training.

    Args:
        gene_expression_df: Gene expression matrix (cell lines × genes)
                           with COSMIC_ID as index
        drug_response_df: Drug response data with COSMIC_ID, DRUG_ID, IC50, AUC

    Returns:
        Merged DataFrame where each row is a (cell_line, drug) pair with:
        - All gene expression features
        - Drug response values (IC50, AUC)
        - Metadata (DRUG_ID, COSMIC_ID, tissue type, etc.)

    Note:
        Uses inner join on COSMIC_ID to keep only samples with both
        gene expression and drug response data.
    """
    logger.info("Merging gene expression and drug response data...")
    logger.info(f"Gene expression shape: {gene_expression_df.shape}")
    logger.info(f"Drug response shape: {drug_response_df.shape}")

    # Reset index on gene expression to make COSMIC_ID a column
    gene_expr_for_merge = gene_expression_df.reset_index()

    # Ensure COSMIC_ID column exists in both DataFrames
    if 'COSMIC_ID' not in gene_expr_for_merge.columns:
        raise ValueError("Gene expression DataFrame must have COSMIC_ID as index or column")
    if 'COSMIC_ID' not in drug_response_df.columns:
        raise ValueError("Drug response DataFrame must have COSMIC_ID column")

    # Merge on COSMIC_ID (inner join: keep only cell lines with both gene expression and response)
    merged_df = drug_response_df.merge(
        gene_expr_for_merge,
        on='COSMIC_ID',
        how='inner'
    )

    logger.info(f"Merged dataset shape: {merged_df.shape}")
    logger.info(f"Cell lines in merged data: {merged_df['COSMIC_ID'].nunique()}")
    logger.info(f"Drugs in merged data: {merged_df['DRUG_ID'].nunique()}")
    logger.info(f"Total (cell_line, drug) pairs: {len(merged_df)}")

    return merged_df


def merge_all_data(
    drug_response_df: pd.DataFrame,
    gene_expression_df: Optional[pd.DataFrame] = None,
    compound_annotations_df: Optional[pd.DataFrame] = None,
    cell_line_details_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Merge all data sources into a single DataFrame.

    This function combines:
    - Drug response data (IC50, AUC) - main data
    - Gene expression data (if available)
    - Compound annotations (drug metadata)
    - Cell line details (cell line metadata)

    The goal is to create a complete dataset where each row represents:
    (cell_line, drug) with all available features and response values.

    Args:
        drug_response_df: GDSC drug response data (required)
        gene_expression_df: Gene expression matrix (optional)
        compound_annotations_df: Drug annotations (optional)
        cell_line_details_df: Cell line details (optional)

    Returns:
        Merged DataFrame with all available data

    Note:
        Gene expression is merged by cell line ID (COSMIC_ID)
        Compound annotations are merged by drug ID (DRUG_ID)
        Cell line details are merged by cell line ID (COSMIC_ID)
    """
    logger.info("Merging all data sources...")

    # Start with drug response data
    merged_df = drug_response_df.copy()

    # Merge compound annotations if available
    if compound_annotations_df is not None and not compound_annotations_df.empty:
        logger.info("Merging compound annotations...")
        # Merge on DRUG_ID (or appropriate drug identifier)
        if 'DRUG_ID' in merged_df.columns and 'DRUG_ID' in compound_annotations_df.columns:
            merged_df = merged_df.merge(
                compound_annotations_df,
                on='DRUG_ID',
                how='left',
                suffixes=('', '_compound')
            )
            logger.info(f"After compound merge: {merged_df.shape}")

    # Merge cell line details if available
    if cell_line_details_df is not None and not cell_line_details_df.empty:
        logger.info("Merging cell line details...")
        # Merge on COSMIC_ID or CELL_LINE_NAME
        merge_key = 'COSMIC_ID' if 'COSMIC_ID' in cell_line_details_df.columns else 'CELL_LINE_NAME'
        if merge_key in merged_df.columns:
            merged_df = merged_df.merge(
                cell_line_details_df,
                on=merge_key,
                how='left',
                suffixes=('', '_cell_line')
            )
            logger.info(f"After cell line merge: {merged_df.shape}")

    # Merge gene expression if available
    if gene_expression_df is not None and not gene_expression_df.empty:
        logger.info("Merging gene expression data...")

        # Gene expression matrix should have cell lines as rows (index)
        # We need to merge based on COSMIC_ID

        # If gene expression index is COSMIC_ID, merge directly
        gene_expr_for_merge = gene_expression_df.reset_index()

        # Rename index column to COSMIC_ID if needed
        if 'index' in gene_expr_for_merge.columns:
            gene_expr_for_merge = gene_expr_for_merge.rename(columns={'index': 'COSMIC_ID'})

        # Merge on COSMIC_ID
        if 'COSMIC_ID' in merged_df.columns and 'COSMIC_ID' in gene_expr_for_merge.columns:
            merged_df = merged_df.merge(
                gene_expr_for_merge,
                on='COSMIC_ID',
                how='inner',  # Inner join: keep only samples with gene expression
                suffixes=('', '_gene_expr')
            )
            logger.info(f"After gene expression merge: {merged_df.shape}")
        else:
            logger.warning("Could not merge gene expression: COSMIC_ID column not found")

    logger.info(f"Final merged dataset shape: {merged_df.shape}")

    return merged_df


def get_data_summary(df: pd.DataFrame) -> Dict:
    """
    Generate a summary of the dataset.

    Useful for understanding the data structure and for reporting in the thesis.

    Args:
        df: DataFrame to summarize

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "n_cell_lines": df['COSMIC_ID'].nunique() if 'COSMIC_ID' in df.columns else 0,
        "n_drugs": df['DRUG_ID'].nunique() if 'DRUG_ID' in df.columns else 0,
        "missing_values": df.isnull().sum().sum(),
        "missing_percentage": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        "columns": list(df.columns),
    }

    # Add target variable statistics if present
    if 'LN_IC50' in df.columns:
        summary['ln_ic50_mean'] = df['LN_IC50'].mean()
        summary['ln_ic50_std'] = df['LN_IC50'].std()
        summary['ln_ic50_missing'] = df['LN_IC50'].isnull().sum()

    if 'AUC' in df.columns:
        summary['auc_mean'] = df['AUC'].mean()
        summary['auc_std'] = df['AUC'].std()
        summary['auc_missing'] = df['AUC'].isnull().sum()

    return summary


def print_data_summary(df: pd.DataFrame):
    """
    Print a formatted summary of the dataset.

    Args:
        df: DataFrame to summarize
    """
    summary = get_data_summary(df)

    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)
    print(f"Total rows: {summary['n_rows']:,}")
    print(f"Total columns: {summary['n_columns']:,}")
    print(f"Unique cell lines: {summary['n_cell_lines']:,}")
    print(f"Unique drugs: {summary['n_drugs']:,}")
    print(f"Missing values: {summary['missing_values']:,} ({summary['missing_percentage']:.2f}%)")

    if 'ln_ic50_mean' in summary:
        print(f"\nLN_IC50: mean={summary['ln_ic50_mean']:.3f}, std={summary['ln_ic50_std']:.3f}, missing={summary['ln_ic50_missing']}")

    if 'auc_mean' in summary:
        print(f"AUC: mean={summary['auc_mean']:.3f}, std={summary['auc_std']:.3f}, missing={summary['auc_missing']}")

    print("=" * 80)


# =============================================================================
# MAIN FUNCTION FOR TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test the data loading functions.
    Run this script directly to verify data loading works correctly.
    """
    print("Testing data loading functions...\n")

    # Test 1: Load drug response data
    try:
        drug_response = load_gdsc_drug_response()
        print("\n✓ Successfully loaded drug response data")
        print_data_summary(drug_response)
    except Exception as e:
        print(f"\n✗ Error loading drug response data: {e}")

    # Test 2: Load compound annotations
    try:
        compounds = load_compound_annotations()
        if not compounds.empty:
            print(f"\n✓ Successfully loaded {len(compounds)} compound annotations")
    except Exception as e:
        print(f"\n✗ Error loading compound annotations: {e}")

    # Test 3: Load cell line details
    try:
        cell_lines = load_cell_line_details()
        if not cell_lines.empty:
            print(f"\n✓ Successfully loaded {len(cell_lines)} cell line details")
    except Exception as e:
        print(f"\n✗ Error loading cell line details: {e}")

    print("\nData loading tests complete!")
