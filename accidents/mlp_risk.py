"""
accidents/mlp_risk.py
---------------------
Proposed Model: Multi-Layer Perceptron (MLP) in PyTorch for Accident Severity/Risk
Prediction with balanced class loss handling and early stopping.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class SeverityMLPNetwork(nn.Module):
    """
    Feedforward Neural Network architecture for accident severity classification.
    """

    def __init__(self, in_features: int, hidden_dim1: int = 64, hidden_dim2: int = 32, num_classes: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)

        self.out = nn.Linear(hidden_dim2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        if out.size(0) > 1:
            out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        if out.size(0) > 1:
            out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        return self.out(out)


class AccidentRiskMLP:
    """
    Multi-Layer Perceptron wrapper providing fit/predict/predict_proba scikit-style API.
    Handles class imbalance using weighted Cross-Entropy Loss.
    """

    def __init__(
        self,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        lr: float = 0.002,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        epochs: int = 80,
        patience: int = 15,
        random_state: int = 42,
        device: Optional[str] = None,
    ) -> None:
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        torch.manual_seed(random_state)
        self.network: Optional[SeverityMLPNetwork] = None
        self.in_features: Optional[int] = None
        self.history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "val_acc": []}
        self.is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        class_weights: Optional[Dict[int, float]] = None,
    ) -> "AccidentRiskMLP":
        """
        Train the MLP with cross-entropy loss and optional class weights.
        """
        self.in_features = X_train.shape[1]
        self.network = SeverityMLPNetwork(
            in_features=self.in_features,
            hidden_dim1=self.hidden_dim1,
            hidden_dim2=self.hidden_dim2,
            num_classes=3,
        ).to(self.device)

        # Build class weights tensor
        if class_weights is not None:
            weights_list = [class_weights[i] for i in range(3)]
            weight_tensor = torch.tensor(weights_list, dtype=torch.float32).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        if X_val is not None and y_val is not None:
            val_ds = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long),
            )
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self.history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(self.epochs):
            self.network.train()
            running_loss = 0.0
            total_samples = 0

            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = self.network(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * bx.size(0)
                total_samples += bx.size(0)

            epoch_train_loss = running_loss / max(total_samples, 1)
            self.history["train_loss"].append(epoch_train_loss)

            # Validation phase
            if val_loader is not None:
                self.network.eval()
                val_loss = 0.0
                val_correct = 0
                val_samples = 0

                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(self.device), vy.to(self.device)
                        v_logits = self.network(vx)
                        v_loss = criterion(v_logits, vy)
                        val_loss += v_loss.item() * vx.size(0)
                        preds = torch.argmax(v_logits, dim=1)
                        val_correct += (preds == vy).sum().item()
                        val_samples += vx.size(0)

                epoch_val_loss = val_loss / max(val_samples, 1)
                epoch_val_acc = val_correct / max(val_samples, 1)
                self.history["val_loss"].append(epoch_val_loss)
                self.history["val_acc"].append(epoch_val_acc)

                # Early stopping check
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    best_state = copy.deepcopy(self.network.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        if best_state is not None:
            self.network.load_state_dict(best_state)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns softmax probability distribution over classes."""
        if not self.is_fitted or self.network is None:
            raise RuntimeError("Model is not fitted. Call fit() before predicting.")

        self.network.eval()
        with torch.no_grad():
            t_x = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.network(t_x)
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class labels (0: slight, 1: serious, 2: fatal)."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def save(self, filepath: str) -> None:
        """Saves model state and metadata."""
        if not self.is_fitted or self.network is None:
            raise RuntimeError("Cannot save unfitted model.")
        payload = {
            "state_dict": self.network.state_dict(),
            "in_features": self.in_features,
            "hidden_dim1": self.hidden_dim1,
            "hidden_dim2": self.hidden_dim2,
            "history": self.history,
        }
        torch.save(payload, filepath)

    def load(self, filepath: str) -> None:
        """Loads model weights."""
        payload = torch.load(filepath, map_location=self.device)
        self.in_features = payload["in_features"]
        self.hidden_dim1 = payload["hidden_dim1"]
        self.hidden_dim2 = payload["hidden_dim2"]
        self.history = payload.get("history", {})

        self.network = SeverityMLPNetwork(
            in_features=self.in_features,
            hidden_dim1=self.hidden_dim1,
            hidden_dim2=self.hidden_dim2,
            num_classes=3,
        ).to(self.device)
        self.network.load_state_dict(payload["state_dict"])
        self.network.eval()
        self.is_fitted = True

