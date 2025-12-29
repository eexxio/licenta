"""
Neural Network model for drug response prediction using PyTorch.

This implements a deep neural network with:
- Drug embedding layer (64-dim learned representations)
- Multi-layer architecture: [1024, 512, 256, 128]
- Batch normalization (stabilizes training)
- Dropout (prevents overfitting)
- Early stopping
- GPU support

This is your most advanced model!

Author: Bachelor's Thesis Project
Date: 2026
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Optional, Tuple
import logging

from src.models.base_model import BaseModel
from src.config import NN_CONFIG, set_seeds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DrugResponseNN(nn.Module):
    """
    Neural Network architecture for drug response prediction.

    Architecture:
    - Input: Gene expression (n_genes) + Drug embedding (64-dim)
    - Hidden layers: [1024, 512, 256, 128] with BatchNorm + ReLU + Dropout
    - Output: Single value (predicted IC50 or AUC)

    Key Features:
    - Drug embeddings: Learns 64-dim representation for each drug
      (instead of sparse one-hot encoding)
    - Batch normalization: Stabilizes training, allows higher learning rates
    - Dropout: Prevents overfitting by randomly dropping neurons
    """

    def __init__(
        self,
        n_genes: int,
        n_drugs: int,
        embedding_dim: int = 64,
        hidden_dims: list = [1024, 512, 256, 128],
        dropout_rates: list = [0.3, 0.3, 0.2, 0.2],
        use_batch_norm: bool = True
    ):
        """
        Initialize the neural network.

        Args:
            n_genes: Number of input gene features
            n_drugs: Number of unique drugs (for embedding)
            embedding_dim: Dimension of drug embeddings
            hidden_dims: List of hidden layer sizes
            dropout_rates: Dropout probability for each hidden layer
            use_batch_norm: Whether to use batch normalization
        """
        super(DrugResponseNN, self).__init__()

        self.n_genes = n_genes
        self.n_drugs = n_drugs
        self.embedding_dim = embedding_dim
        self.use_batch_norm = use_batch_norm

        # Drug embedding layer
        # Learns a 64-dimensional representation for each drug
        # This is more efficient than one-hot encoding (which would be sparse)
        self.drug_embedding = nn.Embedding(
            num_embeddings=n_drugs,
            embedding_dim=embedding_dim
        )

        # Input dimension: genes + drug embedding
        input_dim = n_genes + embedding_dim

        # Build hidden layers dynamically
        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()

        prev_dim = input_dim
        for hidden_dim, dropout_rate in zip(hidden_dims, dropout_rates):
            # Linear layer
            self.hidden_layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization (optional)
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

            # Dropout
            self.dropouts.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        # Output layer (single value for regression)
        self.output_layer = nn.Linear(prev_dim, 1)

        # Initialize weights using He initialization
        # (good for ReLU activations)
        self._initialize_weights()

        logger.info(f"Created DrugResponseNN with {n_genes} genes, "
                   f"{n_drugs} drugs, architecture: {hidden_dims}")

    def _initialize_weights(self):
        """
        Initialize network weights using He initialization.

        He initialization is optimal for ReLU activations.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.1)

    def forward(
        self,
        gene_expr: torch.Tensor,
        drug_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            gene_expr: Gene expression tensor (batch_size, n_genes)
            drug_idx: Drug indices (batch_size,) - integers from 0 to n_drugs-1

        Returns:
            Predictions (batch_size, 1)

        Example:
            >>> gene_expr = torch.randn(32, 5000)  # batch_size=32, 5000 genes
            >>> drug_idx = torch.randint(0, 265, (32,))  # 265 drugs
            >>> predictions = model(gene_expr, drug_idx)
            >>> print(predictions.shape)  # torch.Size([32, 1])
        """
        # Get drug embeddings
        # Input: (batch_size,) integers
        # Output: (batch_size, embedding_dim) continuous vectors
        drug_emb = self.drug_embedding(drug_idx)

        # Concatenate gene expression and drug embedding
        # Shape: (batch_size, n_genes + embedding_dim)
        x = torch.cat([gene_expr, drug_emb], dim=1)

        # Pass through hidden layers
        for i, (linear, dropout) in enumerate(zip(self.hidden_layers, self.dropouts)):
            # Linear transformation
            x = linear(x)

            # Batch normalization (if enabled)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)

            # ReLU activation
            x = F.relu(x)

            # Dropout (only active during training)
            x = dropout(x)

        # Output layer (no activation for regression)
        x = self.output_layer(x)

        return x


class NeuralNetworkModel(BaseModel):
    """
    PyTorch Neural Network wrapper for drug response prediction.

    This wraps DrugResponseNN with training logic, early stopping,
    and model persistence.

    Example:
        >>> from src.models.neural_network import NeuralNetworkModel
        >>>
        >>> # Create model
        >>> model = NeuralNetworkModel(n_genes=5000, n_drugs=265)
        >>>
        >>> # Train with early stopping
        >>> model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        >>>
        >>> # Predict
        >>> y_pred = model.predict(X_test)
        >>>
        >>> # Save model
        >>> model.save(Path("experiments/nn_auc/model.pth"))
    """

    def __init__(self, n_genes: int, n_drugs: int, **kwargs):
        """
        Initialize Neural Network model.

        Args:
            n_genes: Number of gene features
            n_drugs: Number of unique drugs
            **kwargs: Override NN_CONFIG parameters
        """
        super().__init__(model_name="NeuralNetwork")

        # Merge config with overrides
        config = NN_CONFIG.copy()
        config.update(kwargs)

        # Create the PyTorch model
        self.model = DrugResponseNN(
            n_genes=n_genes,
            n_drugs=n_drugs,
            embedding_dim=config['drug_embedding_dim'],
            hidden_dims=config['hidden_dims'],
            dropout_rates=config['dropout_rates'],
            use_batch_norm=config['use_batch_norm']
        )

        # Training parameters
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.max_epochs = config['max_epochs']
        self.early_stopping_patience = config['early_stopping_patience']

        # Device (GPU if available)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and config['device'] == 'cuda'
            else 'cpu'
        )
        self.model.to(self.device)

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=config.get('weight_decay', 1e-5)
        )

        if config.get('use_lr_scheduler', True):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config['lr_scheduler_factor'],
                patience=config['lr_scheduler_patience'],
                min_lr=config['lr_scheduler_min_lr']
            )
        else:
            self.scheduler = None

        # Loss function
        self.criterion = nn.MSELoss()

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': []
        }

        logger.info(f"Created Neural Network on device: {self.device}")

    def _prepare_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Prepare DataLoader from numpy arrays.

        Args:
            X: Features (n_samples, n_features)
               Last column should be drug indices
            y: Targets (n_samples,)
            shuffle: Whether to shuffle data

        Returns:
            DataLoader for batching
        """
        # Split features into gene expression and drug indices
        # Assuming last column is drug index
        gene_expr = X[:, :-1].astype(np.float32)
        drug_idx = X[:, -1].astype(np.int64)
        y = y.astype(np.float32).reshape(-1, 1)

        # Convert to PyTorch tensors
        gene_expr_tensor = torch.from_numpy(gene_expr)
        drug_idx_tensor = torch.from_numpy(drug_idx)
        y_tensor = torch.from_numpy(y)

        # Create dataset
        dataset = TensorDataset(gene_expr_tensor, drug_idx_tensor, y_tensor)

        # Create DataLoader
        # num_workers > 0 speeds up data loading on GPU (workers load data on CPU while GPU trains)
        num_workers = self.config.get('num_workers', 0)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )

        return dataloader

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        verbose: bool = True
    ) -> 'NeuralNetworkModel':
        """
        Train the neural network.

        Args:
            X_train: Training features (n_samples, n_features)
                    Last column = drug indices
            y_train: Training targets (n_samples,)
            X_val: Validation features (optional, for early stopping)
            y_val: Validation targets (optional)
            verbose: Print training progress

        Returns:
            self

        Example:
            >>> model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        """
        logger.info(f"Training Neural Network on {len(X_train)} samples...")

        set_seeds()

        # Prepare data loaders
        train_loader = self._prepare_data(X_train, y_train, shuffle=True)

        if X_val is not None and y_val is not None:
            val_loader = self._prepare_data(X_val, y_val, shuffle=False)
            use_early_stopping = True
            logger.info(f"Using validation set ({len(X_val)} samples) for early stopping")
        else:
            val_loader = None
            use_early_stopping = False
            logger.warning("No validation set - early stopping disabled")

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        # Training loop
        for epoch in range(self.max_epochs):
            # Train
            train_loss = self._train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)

            # Validate
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                self.history['val_loss'].append(val_loss)

                # Learning rate scheduling
                if self.scheduler is not None:
                    self.scheduler.step(val_loss)

                # Early stopping check
                if use_early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1

                    if patience_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        # Restore best model
                        self.model.load_state_dict(best_model_state)
                        break

                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.max_epochs}: "
                              f"Train Loss = {train_loss:.4f}, "
                              f"Val Loss = {val_loss:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(f"Epoch {epoch+1}/{self.max_epochs}: "
                              f"Train Loss = {train_loss:.4f}")

        self.is_fitted = True
        logger.info("✓ Neural Network training complete")

        return self

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        for gene_expr, drug_idx, targets in train_loader:
            # Move to device
            gene_expr = gene_expr.to(self.device)
            drug_idx = drug_idx.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            predictions = self.model(gene_expr, drug_idx)

            # Compute loss
            loss = self.criterion(predictions, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(gene_expr)

        return total_loss / len(train_loader.dataset)

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for gene_expr, drug_idx, targets in val_loader:
                gene_expr = gene_expr.to(self.device)
                drug_idx = drug_idx.to(self.device)
                targets = targets.to(self.device)

                predictions = self.model(gene_expr, drug_idx)
                loss = self.criterion(predictions, targets)

                total_loss += loss.item() * len(gene_expr)

        return total_loss / len(val_loader.dataset)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features (n_samples, n_features)
               Last column = drug indices

        Returns:
            Predictions (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions!")

        logger.info(f"Making predictions on {len(X)} samples...")

        self.model.eval()

        # Prepare data
        gene_expr = X[:, :-1].astype(np.float32)
        drug_idx = X[:, -1].astype(np.int64)

        gene_expr_tensor = torch.from_numpy(gene_expr).to(self.device)
        drug_idx_tensor = torch.from_numpy(drug_idx).to(self.device)

        # Predict in batches
        predictions = []
        with torch.no_grad():
            for i in range(0, len(gene_expr), self.batch_size):
                batch_genes = gene_expr_tensor[i:i+self.batch_size]
                batch_drugs = drug_idx_tensor[i:i+self.batch_size]

                batch_pred = self.model(batch_genes, batch_drugs)
                predictions.append(batch_pred.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0).flatten()

        return predictions

    def save(self, path: Path) -> None:
        """Save model to disk."""
        if not self.is_fitted:
            logger.warning("Saving unfitted model!")

        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model state and metadata
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'n_genes': self.model.n_genes,
            'n_drugs': self.model.n_drugs,
            'is_fitted': self.is_fitted
        }, path)

        logger.info(f"Saved Neural Network to: {path}")

    @classmethod
    def load(cls, path: Path, n_genes: int = None, n_drugs: int = None) -> 'NeuralNetworkModel':
        """Load model from disk."""
        checkpoint = torch.load(path)

        # Create model instance
        model = cls(
            n_genes=checkpoint['n_genes'],
            n_drugs=checkpoint['n_drugs']
        )

        # Load state
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.history = checkpoint['history']
        model.is_fitted = checkpoint['is_fitted']

        logger.info(f"Loaded Neural Network from: {path}")

        return model


# =============================================================================
# MAIN FUNCTION FOR TESTING
# =============================================================================

if __name__ == "__main__":
    """Test the Neural Network on synthetic data."""
    print("Testing Neural Network model...\n")

    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 1000
    n_genes = 100
    n_drugs = 10

    # Generate synthetic data
    X = np.random.randn(n_samples, n_genes).astype(np.float32)
    drug_idx = np.random.randint(0, n_drugs, size=n_samples)

    # True relationship: depends on first 10 genes + drug effect
    y = X[:, :10].sum(axis=1) + drug_idx * 0.5 + np.random.randn(n_samples) * 0.5
    y = y.astype(np.float32)

    # Add drug indices as last column
    X_with_drugs = np.column_stack([X, drug_idx])

    # Split
    n_train = 600
    n_val = 200
    X_train = X_with_drugs[:n_train]
    y_train = y[:n_train]
    X_val = X_with_drugs[n_train:n_train+n_val]
    y_val = y[n_train:n_train+n_val]
    X_test = X_with_drugs[n_train+n_val:]
    y_test = y[n_train+n_val:]

    # Create and train model
    print("Training Neural Network...")
    model = NeuralNetworkModel(
        n_genes=n_genes,
        n_drugs=n_drugs,
        hidden_dims=[128, 64],  # Smaller for testing
        max_epochs=50
    )

    model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)

    # Predict
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)

    # Evaluate
    from sklearn.metrics import r2_score, mean_squared_error

    r2 = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"\nResults:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")

    print(f"\nTraining history:")
    print(f"Final train loss: {model.history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {model.history['val_loss'][-1]:.4f}")

    print("\n✓ Neural Network test complete!")
