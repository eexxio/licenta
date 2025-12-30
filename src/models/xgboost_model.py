"""
XGBoost model for drug response prediction.

XGBoost (eXtreme Gradient Boosting) is a state-of-the-art gradient boosting algorithm:
- Often achieves the best performance on tabular data
- Supports early stopping (stops training when validation performance plateaus)
- Provides feature importance
- Can use GPU acceleration (enabled by default if CUDA available)

This is likely to be your best-performing model!

Author: Bachelor's Thesis Project
Date: 2026
"""

import numpy as np
import xgboost as xgb

from src.models.base_model import BaseModel
from src.config import XGB_CONFIG, set_seeds

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_gpu_available():
    """Check if GPU is available for XGBoost."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            return True
    except ImportError:
        pass
    
    # Alternative check without PyTorch
    try:
        # Try creating a small GPU model to check
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("NVIDIA GPU detected via nvidia-smi")
            return True
    except Exception:
        pass
    
    logger.info("No GPU detected, using CPU")
    return False


class XGBoostModel(BaseModel):
    """
    XGBoost model for drug response prediction.

    Hyperparameters (from config.py):
    - learning_rate: 0.05 (step size for each boosting round)
    - max_depth: 6 (maximum tree depth)
    - n_estimators: 1000 (number of boosting rounds)
    - early_stopping_rounds: 50 (stop if no improvement for 50 rounds)
    - device: "cuda" (uses GPU if available)

    Example:
        >>> from src.models.xgboost_model import XGBoostModel
        >>>
        >>> # Create and train model (with early stopping)
        >>> model = XGBoostModel()
        >>> model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        >>>
        >>> # Make predictions
        >>> y_pred = model.predict(X_test)
        >>>
        >>> # Get feature importance
        >>> importance = model.get_feature_importance()
    """

    def __init__(self, **kwargs):
        """
        Initialize XGBoost model.

        Args:
            **kwargs: Override default hyperparameters from config.py
                     Example: XGBoostModel(learning_rate=0.1)
        """
        super().__init__(model_name="XGBoost")

        # Use config hyperparameters, but allow overrides
        config = XGB_CONFIG.copy()
        config.update(kwargs)

        # Configure GPU usage
        gpu_available = check_gpu_available()
        if gpu_available:
            config['device'] = 'cuda'
            config['tree_method'] = 'hist'  # XGBoost 2.0+ uses 'hist' with device='cuda'
            logger.info("XGBoost configured for GPU training (device='cuda')")
        else:
            config['device'] = 'cpu'
            config['tree_method'] = 'hist'
            logger.info("XGBoost configured for CPU training")

        # Create the XGBoost model
        self.model = xgb.XGBRegressor(**config)

        # Store for reference
        self.learning_rate = config['learning_rate']
        self.n_estimators = config['n_estimators']
        self.device = config['device']

        logger.info(f"Created XGBoost with lr={self.learning_rate}, "
                   f"n_estimators={self.n_estimators}, device={self.device}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **kwargs
    ) -> 'XGBoostModel':
        """
        Train the XGBoost model.

        Args:
            X_train: Training features (n_samples × n_features)
            y_train: Training targets (n_samples,)
            X_val: Validation features (optional, for early stopping)
            y_val: Validation targets (optional, for early stopping)
            **kwargs: Additional arguments for XGBoost fit()

        Returns:
            self (for method chaining)

        Note:
            If validation data is provided, early stopping will be used.
            This prevents overfitting by stopping training when validation
            performance stops improving.

        Example:
            >>> # With early stopping (recommended)
            >>> model = XGBoostModel()
            >>> model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            >>>
            >>> # Without early stopping
            >>> model.fit(X_train, y_train)
        """
        logger.info(f"Training XGBoost on {len(X_train)} samples...")

        # Set random seed for reproducibility
        set_seeds()

        # Prepare evaluation set for early stopping
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            eval_names = ['train', 'val']
            logger.info(f"Using validation set ({len(X_val)} samples) for early stopping")
        else:
            eval_set = [(X_train, y_train)]
            eval_names = ['train']
            logger.warning("No validation set provided - early stopping disabled")

        # Train the model
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=False  # Set to True to see training progress
        )

        self.is_fitted = True

        # Log best iteration if early stopping was used
        if hasattr(self.model, 'best_iteration'):
            logger.info(f"✓ XGBoost training complete (stopped at iteration {self.model.best_iteration})")
        else:
            logger.info("✓ XGBoost training complete")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained XGBoost model.

        Args:
            X: Features (n_samples × n_features)

        Returns:
            Predictions (n_samples,)

        Raises:
            ValueError: If model is not fitted

        Example:
            >>> y_pred = model.predict(X_test)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions!")

        logger.info(f"Making predictions on {len(X)} samples...")

        # Predict using the best iteration (if early stopping was used)
        predictions = self.model.predict(X)

        return predictions

    def get_feature_importance(
        self,
        importance_type: str = 'gain'
    ) -> np.ndarray:
        """
        Get feature importance scores from XGBoost.

        XGBoost provides multiple types of feature importance:
        - 'gain': Average gain when feature is used (default, most informative)
        - 'weight': Number of times feature is used
        - 'cover': Average coverage of splits using this feature

        Args:
            importance_type: Type of importance ('gain', 'weight', or 'cover')

        Returns:
            Array of feature importances (length = n_features)

        Raises:
            ValueError: If model is not fitted

        Note:
            'gain' is the recommended importance type as it measures
            the actual improvement in model performance from each feature.

        Example:
            >>> importance = model.get_feature_importance()
            >>>
            >>> # Get top 30 most important genes
            >>> top_indices = np.argsort(importance)[-30:][::-1]
            >>>
            >>> # Compare with Random Forest importance
            >>> rf_importance = rf_model.get_feature_importance()
            >>> # Genes important in both models are highly reliable!
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance!")

        # Get importance from XGBoost
        # Returns a dictionary {feature_name: importance}
        importance_dict = self.model.get_booster().get_score(
            importance_type=importance_type
        )

        # Convert to array (maintaining feature order)
        n_features = self.model.n_features_in_
        importance = np.zeros(n_features)

        for feature_name, score in importance_dict.items():
            # Feature names are like "f0", "f1", ...
            feature_idx = int(feature_name[1:])
            importance[feature_idx] = score

        # Normalize to sum to 1 (like Random Forest)
        if importance.sum() > 0:
            importance = importance / importance.sum()

        return importance

    def get_training_history(self) -> dict:
        """
        Get training history (loss curves).

        Returns:
            Dictionary with training and validation losses per iteration

        Useful for plotting learning curves to diagnose overfitting.

        Example:
            >>> history = model.get_training_history()
            >>>
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(history['train'], label='Train')
            >>> plt.plot(history['val'], label='Validation')
            >>> plt.xlabel('Iteration')
            >>> plt.ylabel('Loss (RMSE)')
            >>> plt.legend()
            >>> plt.title('XGBoost Learning Curves')
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted!")

        # Get evaluation results
        evals_result = self.model.evals_result()

        return evals_result


# =============================================================================
# MAIN FUNCTION FOR TESTING
# =============================================================================

if __name__ == "__main__":
    """
    Test the XGBoost model on synthetic data.
    """
    print("Testing XGBoost model...\n")

    # Generate synthetic data
    np.random.seed(42)

    n_samples = 1000
    n_features = 100

    X = np.random.randn(n_samples, n_features)
    # True relationship: y depends on first 10 features (non-linear)
    y = (X[:, :10] ** 2).sum(axis=1) + np.random.randn(n_samples) * 0.5

    # Split into train/val/test
    n_train = 600
    n_val = 200
    X_train = X[:n_train]
    y_train = y[:n_train]
    X_val = X[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    X_test = X[n_train+n_val:]
    y_test = y[n_train+n_val:]

    # Create and train model with early stopping
    print("Training XGBoost...")
    model = XGBoostModel(n_estimators=500)  # Fewer rounds for testing
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

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
    print("\nTop 10 most important features (gain):")
    importance = model.get_feature_importance()
    top_indices = np.argsort(importance)[-10:][::-1]

    for i, idx in enumerate(top_indices):
        print(f"{i+1}. Feature {idx}: {importance[idx]:.4f}")

    # Check if model correctly identified important features (0-9)
    top_20_features = np.argsort(importance)[-20:]
    important_features_found = sum(1 for i in range(10) if i in top_20_features)

    print(f"\nImportant features identified: {important_features_found}/10")

    # Training history
    print("\nTraining history:")
    history = model.get_training_history()
    if 'validation_0' in history:
        final_train_rmse = history['validation_0']['rmse'][-1]
        print(f"Final training RMSE: {final_train_rmse:.4f}")
    if 'validation_1' in history:
        final_val_rmse = history['validation_1']['rmse'][-1]
        print(f"Final validation RMSE: {final_val_rmse:.4f}")

    print("\n✓ XGBoost test complete!")
