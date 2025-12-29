"""
Random Forest model for drug response prediction.

Random Forest is an ensemble of decision trees that:
- Reduces overfitting compared to single decision trees
- Provides feature importance (which genes matter most)
- Is robust and easy to tune
- Serves as a strong baseline

Author: Bachelor's Thesis Project
Date: 2026
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.models.base_model import BaseModel
from src.config import RF_CONFIG, set_seeds

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest model for drug response prediction.

    This wraps scikit-learn's RandomForestRegressor with our common interface.

    Hyperparameters (from config.py):
    - n_estimators: 500 (number of trees)
    - max_depth: 20 (maximum tree depth)
    - min_samples_split: 10 (minimum samples to split a node)
    - n_jobs: -1 (use all CPU cores)

    Example:
        >>> from src.models.random_forest import RandomForestModel
        >>>
        >>> # Create and train model
        >>> model = RandomForestModel()
        >>> model.fit(X_train, y_train)
        >>>
        >>> # Make predictions
        >>> y_pred = model.predict(X_test)
        >>>
        >>> # Get feature importance
        >>> importance = model.get_feature_importance()
        >>> top_genes = np.argsort(importance)[-30:][::-1]  # Top 30 genes
    """

    def __init__(self, **kwargs):
        """
        Initialize Random Forest model.

        Args:
            **kwargs: Override default hyperparameters from config.py
                     Example: RandomForestModel(n_estimators=1000)
        """
        super().__init__(model_name="RandomForest")

        # Use config hyperparameters, but allow overrides
        config = RF_CONFIG.copy()
        config.update(kwargs)

        # Create the scikit-learn model
        self.model = RandomForestRegressor(**config)

        logger.info(f"Created Random Forest with {config['n_estimators']} trees")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        **kwargs
    ) -> 'RandomForestModel':
        """
        Train the Random Forest model.

        Args:
            X_train: Training features (n_samples × n_features)
            y_train: Training targets (n_samples,)
            **kwargs: Additional arguments (ignored for Random Forest)

        Returns:
            self (for method chaining)

        Note:
            Random Forest doesn't use validation data during training.
            It has built-in out-of-bag (OOB) error estimation instead.

        Example:
            >>> model = RandomForestModel()
            >>> model.fit(X_train, y_train)
            >>> print(f"OOB Score: {model.model.oob_score_:.4f}")
        """
        logger.info(f"Training Random Forest on {len(X_train)} samples...")

        # Set random seed for reproducibility
        set_seeds()

        # Train the model
        # Random Forest parallelizes across trees (n_jobs=-1 uses all cores)
        self.model.fit(X_train, y_train)

        self.is_fitted = True

        logger.info("✓ Random Forest training complete")

        # Log OOB score if available
        if hasattr(self.model, 'oob_score_'):
            logger.info(f"Out-of-Bag R² Score: {self.model.oob_score_:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained Random Forest.

        Args:
            X: Features (n_samples × n_features)

        Returns:
            Predictions (n_samples,)

        Raises:
            ValueError: If model is not fitted

        Note:
            Random Forest prediction is the average of all tree predictions.

        Example:
            >>> y_pred = model.predict(X_test)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions!")

        logger.info(f"Making predictions on {len(X)} samples...")

        # Predict (parallelized across trees)
        predictions = self.model.predict(X)

        return predictions

    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores from the Random Forest.

        Feature importance measures how much each feature (gene) contributes
        to reducing impurity across all trees in the forest.

        Higher importance = more useful for prediction.

        Returns:
            Array of feature importances (length = n_features)
            Values sum to 1.0

        Raises:
            ValueError: If model is not fitted

        Note:
            Use this to identify which genes are most important for predicting
            drug response. This is a key result for your thesis!

        Example:
            >>> importance = model.get_feature_importance()
            >>>
            >>> # Get top 30 most important genes
            >>> top_indices = np.argsort(importance)[-30:][::-1]
            >>> top_importance = importance[top_indices]
            >>>
            >>> # If you have gene names:
            >>> top_genes = [gene_names[i] for i in top_indices]
            >>> for gene, imp in zip(top_genes, top_importance):
            >>>     print(f"{gene}: {imp:.4f}")
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance!")

        return self.model.feature_importances_

    def get_tree_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from individual trees.

        This can be useful for uncertainty estimation:
        - High variance across trees → high uncertainty
        - Low variance across trees → high confidence

        Args:
            X: Features (n_samples × n_features)

        Returns:
            Array of shape (n_samples, n_estimators)
            Each column is predictions from one tree

        Example:
            >>> tree_preds = model.get_tree_predictions(X_test)
            >>>
            >>> # Compute prediction uncertainty
            >>> prediction_mean = tree_preds.mean(axis=1)
            >>> prediction_std = tree_preds.std(axis=1)
            >>>
            >>> # High std = high uncertainty
            >>> uncertain_samples = np.where(prediction_std > threshold)[0]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted!")

        # Get predictions from each tree
        # Shape: (n_samples, n_estimators)
        tree_predictions = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ]).T

        return tree_predictions


# =============================================================================
# MAIN FUNCTION FOR TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test the Random Forest model on synthetic data.
    """
    print("Testing Random Forest model...\n")

    # Generate synthetic data
    np.random.seed(42)

    n_samples = 1000
    n_features = 100

    X = np.random.randn(n_samples, n_features)
    # True relationship: y depends on first 10 features
    y = X[:, :10].sum(axis=1) + np.random.randn(n_samples) * 0.5

    # Split into train/test
    n_train = 800
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Create and train model
    print("Training Random Forest...")
    model = RandomForestModel(n_estimators=100)  # Fewer trees for testing
    model.fit(X_train, y_train)

    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)

    # Evaluate
    from sklearn.metrics import r2_score, mean_squared_error

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"\nResults:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    # Feature importance
    print("\nTop 10 most important features:")
    importance = model.get_feature_importance()
    top_indices = np.argsort(importance)[-10:][::-1]

    for i, idx in enumerate(top_indices):
        print(f"{i+1}. Feature {idx}: {importance[idx]:.4f}")

    # Check if model correctly identified important features (0-9)
    top_20_features = np.argsort(importance)[-20:]
    important_features_found = sum(1 for i in range(10) if i in top_20_features)

    print(f"\nImportant features identified: {important_features_found}/10")

    print("\n✓ Random Forest test complete!")
