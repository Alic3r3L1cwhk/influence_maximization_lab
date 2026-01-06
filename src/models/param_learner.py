"""
Parameter learning module using PyTorch.
Learns IC edge propagation probabilities from cascade data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.calibration import calibration_curve
import warnings


class EdgeDataset(Dataset):
    """Dataset for edge features and labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Initialize the dataset.

        Args:
            features: Edge features (N x D)
            labels: Edge labels (N,)
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class EdgeProbabilityMLP(nn.Module):
    """MLP model for predicting edge propagation probabilities."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.3):
        """
        Initialize the MLP model.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super(EdgeProbabilityMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Output probability

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.model(x).squeeze()


class ParameterLearner:
    """Learn diffusion parameters from cascade data."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.3, learning_rate: float = 0.001,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 seed: Optional[int] = None):
        """
        Initialize the parameter learner.

        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            device: Device to use ('cuda' or 'cpu')
            seed: Random seed
        """
        self.device = device
        self.model = EdgeProbabilityMLP(input_dim, hidden_dims, dropout).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_auc': [],
            'val_auc': []
        }

    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training data loader

        Returns:
            Tuple of (average loss, AUC score)
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for features, labels in dataloader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        # Metrics expect binary labels; training labels can be fractional (activation rate per edge)
        binary_labels = (np.array(all_labels) > 0).astype(int)
        if binary_labels.max() == binary_labels.min():
            warnings.warn("AUC is undefined when only one class is present; returning NaN.")
            auc = float('nan')
        else:
            auc = roc_auc_score(binary_labels, all_preds)

        return avg_loss, auc

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float, float]:
        """
        Evaluate on validation/test set.

        Args:
            dataloader: Validation/test data loader

        Returns:
            Tuple of (average loss, AUC score, accuracy)
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        binary_labels = (np.array(all_labels) > 0).astype(int)

        if binary_labels.max() == binary_labels.min():
            warnings.warn("AUC is undefined when only one class is present; returning NaN.")
            auc = float('nan')
        else:
            auc = roc_auc_score(binary_labels, all_preds)

        # Binary accuracy with 0.5 threshold
        binary_preds = (np.array(all_preds) > 0.5).astype(int)
        accuracy = accuracy_score(binary_labels, binary_preds)

        return avg_loss, auc, accuracy

    def fit(self, train_features: np.ndarray, train_labels: np.ndarray,
            val_features: Optional[np.ndarray] = None,
            val_labels: Optional[np.ndarray] = None,
            epochs: int = 100, batch_size: int = 256,
            early_stopping_patience: int = 10,
            verbose: bool = True) -> Dict:
        """
        Train the model.

        Args:
            train_features: Training edge features
            train_labels: Training edge labels
            val_features: Validation edge features
            val_labels: Validation edge labels
            epochs: Number of training epochs
            batch_size: Batch size
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress

        Returns:
            Training history dictionary
        """
        # Create datasets and dataloaders
        train_dataset = EdgeDataset(train_features, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=0)

        if val_features is not None and val_labels is not None:
            val_dataset = EdgeDataset(val_features, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                   shuffle=False, num_workers=0)
            use_validation = True
        else:
            use_validation = False

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss, train_auc = self.train_epoch(train_loader)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_auc'].append(train_auc)

            if use_validation:
                val_loss, val_auc, val_acc = self.evaluate(val_loader)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_auc'].append(val_auc)

                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f} - "
                          f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, "
                          f"Val Acc: {val_acc:.4f}")

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    self.best_model_state = self.model.state_dict()
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    # Restore best model
                    self.model.load_state_dict(self.best_model_state)
                    break
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")

        return self.training_history

    def predict(self, features: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """
        Predict probabilities for edges.

        Args:
            features: Edge features
            batch_size: Batch size for prediction

        Returns:
            Predicted probabilities
        """
        self.model.eval()
        dataset = EdgeDataset(features, np.zeros(len(features)))  # Dummy labels
        dataloader = DataLoader(dataset, batch_size=batch_size,
                              shuffle=False, num_workers=0)

        predictions = []
        with torch.no_grad():
            for batch_features, _ in dataloader:
                batch_features = batch_features.to(self.device)
                outputs = self.model(batch_features)
                predictions.extend(outputs.cpu().numpy())

        return np.array(predictions)

    def calibrate_probabilities(self, val_features: np.ndarray,
                               val_labels: np.ndarray,
                               method: str = 'isotonic') -> None:
        """
        Calibrate predicted probabilities using validation set.

        Args:
            val_features: Validation features
            val_labels: Validation labels
            method: Calibration method ('isotonic' or 'sigmoid')
        """
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        # Get predictions
        predictions = self.predict(val_features)

        if method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        elif method == 'sigmoid':
            self.calibrator = LogisticRegression()
        else:
            raise ValueError(f"Unknown calibration method: {method}")

        if method == 'sigmoid':
            self.calibrator.fit(predictions.reshape(-1, 1), val_labels)
        else:
            self.calibrator.fit(predictions, val_labels)

        print(f"Probabilities calibrated using {method} regression")

    def predict_calibrated(self, features: np.ndarray,
                          batch_size: int = 256) -> np.ndarray:
        """
        Predict calibrated probabilities.

        Args:
            features: Edge features
            batch_size: Batch size

        Returns:
            Calibrated probabilities
        """
        predictions = self.predict(features, batch_size)

        if hasattr(self, 'calibrator'):
            if hasattr(self.calibrator, 'predict_proba'):
                # Logistic regression
                predictions = self.calibrator.predict_proba(
                    predictions.reshape(-1, 1))[:, 1]
            else:
                # Isotonic regression
                predictions = self.calibrator.predict(predictions)

        return predictions

    def save_model(self, filepath: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history
        }, filepath)

    def load_model(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_history = checkpoint['training_history']
