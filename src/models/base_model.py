"""
Base model class for all drug response prediction models.

This defines a common interface that all models (Random Forest, XGBoost, Neural Network)
must implement. This ensures consistency and makes it easy to swap models.

Author: Bachelor's Thesis Project
Date: 2026
"""

from abc import ABC, abstractmethod
from pathlib import Path
import pickle
from typing import Any, Optional
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all drug response prediction models.

    All models must implement:
    - fit(): Train the model
    - predict(): Make predictions
    - save(): Save model to disk
    - load(): Load model from disk

    Optional methods:
    - get_feature_importance(): Get feature importance (for tree-based models)
    """

    def __init__(self, model_name: str = "BaseModel"):
        """
        Initialize the model.

        Args:
            model_name: Name of the model (for logging and saving)
        """
        self.model_name = model_name
        self.is_fitted = False
        logger.info(f"Initialized {model_name}")

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> 'BaseModel':
        """
        Train the model on training data.

        Args:
            X_train: Training features (n_samples × n_features)
            y_train: Training targets (n_samples,)
            **kwargs: Additional arguments (e.g., validation data)

        Returns:
            self (for method chaining)

        Example:
            >>> model.fit(X_train, y_train)
            >>> # Or with validation data:
            >>> model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Features (n_samples × n_features)

        Returns:
            Predictions (n_samples,)

        Raises:
            ValueError: If model is not fitted

        Example:
            >>> predictions = model.predict(X_test)
        """
        pass

    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Path where to save the model

        Note:
            For scikit-learn/XGBoost models, uses pickle.
            For PyTorch models, uses torch.save() (overridden in subclass).

        Example:
            >>> model.save(Path("experiments/rf_auc/model.pkl"))
        """
        if not self.is_fitted:
            logger.warning("Saving unfitted model!")

        # Convert to Path if string
        if isinstance(path, str):
            path = Path(path)

        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save using pickle (default for sklearn/xgboost models)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

        logger.info(f"Saved {self.model_name} to: {path}")

    @classmethod
    def load(cls, path: Path) -> 'BaseModel':
        """
        Load model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded model instance

        Example:
            >>> model = RandomForestModel.load(Path("experiments/rf_auc/model.pkl"))
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"Loaded model from: {path}")

        return model

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores (if available).

        Returns:
            Array of feature importances, or None if not applicable

        Note:
            Only tree-based models (Random Forest, XGBoost) have feature importance.
            Neural networks return None (or can use SHAP values).
        """
        logger.warning(f"{self.model_name} does not support feature importance")
        return None

    def __repr__(self) -> str:
        """String representation of the model."""
        fitted_status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.model_name}({fitted_status})"
