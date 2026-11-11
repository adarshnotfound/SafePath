"""
accidents/baseline_risk.py
--------------------------
Baseline Model for Accident Severity/Risk Prediction:
Multinomial Logistic Regression with optional balanced class weighting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression


class AccidentRiskBaseline:
    """
    Multinomial Logistic Regression baseline for accident severity prediction.

    Attributes:
        handle_imbalance: Whether to apply inverse-frequency balanced class weights.
        max_iter: Maximum optimization iterations.
        random_state: Reproducibility seed.
    """

    def __init__(
        self,
        handle_imbalance: bool = True,
        max_iter: int = 1000,
        random_state: int = 42,
    ) -> None:
        self.handle_imbalance = handle_imbalance
        self.max_iter = max_iter
        self.random_state = random_state

        class_weight = "balanced" if handle_imbalance else None
        self.model = LogisticRegression(
            solver="lbfgs",
            class_weight=class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "AccidentRiskBaseline":
        """Fit multinomial logistic regression on training features and labels."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels: 0 (slight), 1 (serious), 2 (fatal)."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() before predicting.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities: [P(slight), P(serious), P(fatal)]."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() before predicting.")
        return self.model.predict_proba(X)

    def get_feature_importances(self, feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Returns model coefficients for explainability.
        In multinomial logistic regression, coefficients indicate log-odds change
        for each class per unit change in feature.
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        classes = ["slight", "serious", "fatal"]
        coefs = self.model.coef_  # Shape (n_classes, n_features)
        intercept = self.model.intercept_

        result: Dict[str, Any] = {}
        for c_idx, class_name in enumerate(classes):
            class_coefs = {}
            for f_idx, val in enumerate(coefs[c_idx]):
                fname = feature_names[f_idx] if feature_names and f_idx < len(feature_names) else f"feature_{f_idx}"
                class_coefs[fname] = float(val)
            result[class_name] = {
                "intercept": float(intercept[c_idx]),
                "weights": class_coefs,
            }
        return result

