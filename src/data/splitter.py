"""
Data splitting module for train/test/validation splits.

CRITICAL: We split by cell line, NOT randomly!
This prevents data leakage where the same cell line appears in both train and test.

Author: Bachelor's Thesis Project
Date: 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import pickle
import logging

from sklearn.model_selection import KFold, StratifiedKFold

# Import configuration
from src.config import (
    RANDOM_SEED,
    TEST_SIZE,
    N_CV_FOLDS,
    TRAIN_INDICES_PATH,
    TEST_INDICES_PATH,
    CV_FOLDS_PATH,
    set_seeds
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GDSCDataSplitter:
    """
    Create train/test/validation splits for GDSC data.

    Key principle: Split by cell line, not by individual samples!

    Why? If the same cell line appears in both train and test, the model
    will have seen that cell line's molecular profile during training,
    making test performance unrealistically high (data leakage).

    Example:
        Train: cell lines 1-800
        Test: cell lines 801-1000
        NOT: random 80% of all (cell_line, drug) pairs
    """

    def __init__(self, random_seed: int = RANDOM_SEED):
        """
        Initialize the data splitter.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        set_seeds(random_seed)
        logger.info(f"Data splitter initialized with seed: {random_seed}")

    def split_by_cell_line(
        self,
        df: pd.DataFrame,
        test_size: float = TEST_SIZE,
        cell_line_col: str = 'COSMIC_ID',
        stratify_col: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data by cell line into train and test sets.

        Args:
            df: DataFrame with drug response data
            test_size: Fraction of cell lines for test set (default: 0.2)
            cell_line_col: Column name for cell line identifier
            stratify_col: Optional column for stratified splitting
                         (e.g., tissue type to ensure balanced tissue distribution)

        Returns:
            Tuple of (train_df, test_df)

        Note:
            This ensures NO cell line appears in both train and test sets.
        """
        logger.info("=" * 80)
        logger.info("SPLITTING DATA BY CELL LINE")
        logger.info("=" * 80)

        # Get unique cell lines
        cell_lines = df[cell_line_col].unique()
        n_cell_lines = len(cell_lines)

        logger.info(f"Total unique cell lines: {n_cell_lines}")
        logger.info(f"Test size: {test_size * 100:.1f}%")

        # Calculate number of test cell lines
        n_test = int(n_cell_lines * test_size)
        n_train = n_cell_lines - n_test

        logger.info(f"Train cell lines: {n_train}")
        logger.info(f"Test cell lines: {n_test}")

        if stratify_col is not None:
            # Stratified split: ensure tissue type distribution is similar
            # in train and test sets
            logger.info(f"Using stratified split by: {stratify_col}")

            # Get tissue type for each cell line
            # (assuming each cell line has one tissue type)
            cell_line_tissue = df[[cell_line_col, stratify_col]].drop_duplicates()
            cell_line_tissue = cell_line_tissue.set_index(cell_line_col)[stratify_col]

            # Use stratified split
            from sklearn.model_selection import train_test_split
            train_cell_lines, test_cell_lines = train_test_split(
                cell_lines,
                test_size=test_size,
                random_state=self.random_seed,
                stratify=cell_line_tissue.loc[cell_lines].values
            )
        else:
            # Random split (but still by cell line, not by samples)
            np.random.shuffle(cell_lines)
            train_cell_lines = cell_lines[:n_train]
            test_cell_lines = cell_lines[n_train:]

        # Split the dataframe
        train_df = df[df[cell_line_col].isin(train_cell_lines)].copy()
        test_df = df[df[cell_line_col].isin(test_cell_lines)].copy()

        # Verify no overlap
        train_cell_set = set(train_df[cell_line_col].unique())
        test_cell_set = set(test_df[cell_line_col].unique())
        overlap = train_cell_set & test_cell_set

        if len(overlap) > 0:
            raise ValueError(
                f"Data leakage detected! {len(overlap)} cell lines appear in both "
                f"train and test sets. This should never happen!"
            )

        logger.info(f"\nTrain set: {len(train_df):,} samples from {len(train_cell_set)} cell lines")
        logger.info(f"Test set: {len(test_df):,} samples from {len(test_cell_set)} cell lines")
        logger.info(f"✓ Verified: No cell line overlap between train and test")
        logger.info("=" * 80)

        return train_df, test_df

    def create_cv_folds(
        self,
        df: pd.DataFrame,
        n_folds: int = N_CV_FOLDS,
        cell_line_col: str = 'COSMIC_ID',
        stratify_col: Optional[str] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create cross-validation folds (by cell line).

        Used for hyperparameter tuning and model selection.

        Args:
            df: DataFrame with drug response data
            n_folds: Number of CV folds (default: 5)
            cell_line_col: Column name for cell line identifier
            stratify_col: Optional column for stratified CV

        Returns:
            List of (train_indices, val_indices) tuples for each fold

        Note:
            Each fold ensures no cell line appears in both train and validation.
        """
        logger.info(f"Creating {n_folds}-fold cross-validation splits by cell line...")

        # Get unique cell lines
        cell_lines = df[cell_line_col].unique()

        # Create mapping from cell line to indices in the dataframe
        cell_line_to_indices = df.groupby(cell_line_col).groups

        if stratify_col is not None:
            # Stratified K-Fold
            logger.info(f"Using stratified CV by: {stratify_col}")

            # Get tissue type for each cell line
            cell_line_tissue = df[[cell_line_col, stratify_col]].drop_duplicates()
            cell_line_tissue = cell_line_tissue.set_index(cell_line_col)[stratify_col]

            skf = StratifiedKFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=self.random_seed
            )

            # Split cell lines
            folds = []
            for train_cl_idx, val_cl_idx in skf.split(
                cell_lines,
                cell_line_tissue.loc[cell_lines].values
            ):
                train_cell_lines = cell_lines[train_cl_idx]
                val_cell_lines = cell_lines[val_cl_idx]

                # Get sample indices for these cell lines
                train_indices = np.concatenate([
                    cell_line_to_indices[cl] for cl in train_cell_lines
                ])
                val_indices = np.concatenate([
                    cell_line_to_indices[cl] for cl in val_cell_lines
                ])

                folds.append((train_indices, val_indices))

        else:
            # Regular K-Fold
            kf = KFold(
                n_splits=n_folds,
                shuffle=True,
                random_state=self.random_seed
            )

            # Split cell lines
            folds = []
            for train_cl_idx, val_cl_idx in kf.split(cell_lines):
                train_cell_lines = cell_lines[train_cl_idx]
                val_cell_lines = cell_lines[val_cl_idx]

                # Get sample indices for these cell lines
                train_indices = np.concatenate([
                    cell_line_to_indices[cl] for cl in train_cell_lines
                ])
                val_indices = np.concatenate([
                    cell_line_to_indices[cl] for cl in val_cell_lines
                ])

                folds.append((train_indices, val_indices))

        logger.info(f"✓ Created {len(folds)} CV folds")
        for i, (train_idx, val_idx) in enumerate(folds):
            logger.info(f"  Fold {i+1}: {len(train_idx)} train, {len(val_idx)} val samples")

        return folds

    def save_split(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_path: Path = TRAIN_INDICES_PATH,
        test_path: Path = TEST_INDICES_PATH
    ):
        """
        Save train/test split indices to disk for reproducibility.

        Args:
            train_df: Training dataframe
            test_df: Test dataframe
            train_path: Path to save train indices
            test_path: Path to save test indices

        Note:
            We save indices (not the full dataframe) to save space.
            The indices can be used to reconstruct the split from the original data.
        """
        logger.info("Saving train/test split indices...")

        # Create directory if it doesn't exist
        train_path.parent.mkdir(parents=True, exist_ok=True)

        # Save indices
        with open(train_path, 'wb') as f:
            pickle.dump(train_df.index.tolist(), f)

        with open(test_path, 'wb') as f:
            pickle.dump(test_df.index.tolist(), f)

        logger.info(f"✓ Saved train indices to: {train_path}")
        logger.info(f"✓ Saved test indices to: {test_path}")

    def load_split(
        self,
        df: pd.DataFrame,
        train_path: Path = TRAIN_INDICES_PATH,
        test_path: Path = TEST_INDICES_PATH
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load previously saved train/test split.

        Args:
            df: Original dataframe
            train_path: Path to train indices
            test_path: Path to test indices

        Returns:
            Tuple of (train_df, test_df)

        Note:
            This ensures you use the exact same split across different runs.
        """
        logger.info("Loading saved train/test split...")

        with open(train_path, 'rb') as f:
            train_indices = pickle.load(f)

        with open(test_path, 'rb') as f:
            test_indices = pickle.load(f)

        train_df = df.loc[train_indices]
        test_df = df.loc[test_indices]

        logger.info(f"✓ Loaded train set: {len(train_df):,} samples")
        logger.info(f"✓ Loaded test set: {len(test_df):,} samples")

        return train_df, test_df

    def save_cv_folds(
        self,
        folds: List[Tuple[np.ndarray, np.ndarray]],
        path: Path = CV_FOLDS_PATH
    ):
        """
        Save cross-validation folds to disk.

        Args:
            folds: List of (train_indices, val_indices) tuples
            path: Path to save folds
        """
        logger.info(f"Saving {len(folds)} CV folds...")

        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'wb') as f:
            pickle.dump(folds, f)

        logger.info(f"✓ Saved CV folds to: {path}")

    def load_cv_folds(self, path: Path = CV_FOLDS_PATH) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Load previously saved CV folds.

        Args:
            path: Path to CV folds

        Returns:
            List of (train_indices, val_indices) tuples
        """
        logger.info("Loading saved CV folds...")

        with open(path, 'rb') as f:
            folds = pickle.load(f)

        logger.info(f"✓ Loaded {len(folds)} CV folds")

        return folds


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_and_save_splits(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    n_cv_folds: int = N_CV_FOLDS,
    cell_line_col: str = 'COSMIC_ID',
    stratify_col: Optional[str] = None,
    save: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, List]:
    """
    Convenience function to create and save all splits.

    Args:
        df: DataFrame with drug response data
        test_size: Fraction for test set
        n_cv_folds: Number of cross-validation folds
        cell_line_col: Column name for cell line identifier
        stratify_col: Optional column for stratified splitting
        save: Whether to save splits to disk

    Returns:
        Tuple of (train_df, test_df, cv_folds)
    """
    splitter = GDSCDataSplitter()

    # Create train/test split
    train_df, test_df = splitter.split_by_cell_line(
        df=df,
        test_size=test_size,
        cell_line_col=cell_line_col,
        stratify_col=stratify_col
    )

    # Create CV folds on training data
    cv_folds = splitter.create_cv_folds(
        df=train_df,  # CV folds only on training data!
        n_folds=n_cv_folds,
        cell_line_col=cell_line_col,
        stratify_col=stratify_col
    )

    # Save if requested
    if save:
        splitter.save_split(train_df, test_df)
        splitter.save_cv_folds(cv_folds)

    return train_df, test_df, cv_folds


# =============================================================================
# MAIN FUNCTION FOR TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test the data splitting functions.
    """
    print("Testing data splitting functions...\n")

    # Create a mock dataset for testing
    np.random.seed(42)

    n_cell_lines = 100
    n_drugs = 10
    cell_lines = [f"CL_{i}" for i in range(n_cell_lines)]
    drugs = [f"DRUG_{i}" for i in range(n_drugs)]
    tissues = ['lung', 'breast', 'colon', 'brain', 'liver']

    # Create all combinations
    data = []
    for cl in cell_lines:
        tissue = np.random.choice(tissues)
        for drug in drugs:
            data.append({
                'COSMIC_ID': cl,
                'DRUG_ID': drug,
                'TISSUE': tissue,
                'AUC': np.random.rand(),
                'IC50': np.random.rand()
            })

    df = pd.DataFrame(data)
    print(f"Created mock dataset: {len(df)} samples from {n_cell_lines} cell lines")
    print(f"Columns: {list(df.columns)}\n")

    # Test splitting
    splitter = GDSCDataSplitter()

    # Test 1: Basic split
    print("Test 1: Basic train/test split")
    train_df, test_df = splitter.split_by_cell_line(df)

    # Test 2: Stratified split
    print("\nTest 2: Stratified split by tissue type")
    train_df, test_df = splitter.split_by_cell_line(
        df,
        stratify_col='TISSUE'
    )

    # Check tissue distribution
    print("\nTissue distribution:")
    print("Train:", train_df.groupby('TISSUE')['COSMIC_ID'].nunique().to_dict())
    print("Test:", test_df.groupby('TISSUE')['COSMIC_ID'].nunique().to_dict())

    # Test 3: CV folds
    print("\nTest 3: Creating CV folds")
    folds = splitter.create_cv_folds(train_df, n_folds=5)

    print("\n✓ All tests passed!")
