"""
accidents/data_loader.py
------------------------
Data ingestion, synthesis, preprocessing, and leakage-free stratified splitting
for accident severity risk prediction (Module 2).

Features:
    - road_type: motorway, primary, secondary, tertiary, residential, unclassified
    - speed_limit: numeric (mph / km/h)
    - time_of_day: day, dusk, night
    - lighting: well_lit, dim, dark
    - weather: fine, rain, fog, snow
    - junction: yes, no

Target:
    - severity: 0 (slight), 1 (serious), 2 (fatal)
Dataset provenance: UK STATS19 (Substitute Dataset per SafePath specification).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

SEVERITY_CLASSES = ["slight", "serious", "fatal"]
CATEGORICAL_FEATURES = ["road_type", "time_of_day", "lighting", "weather", "junction"]
NUMERICAL_FEATURES = ["speed_limit"]


def generate_synthetic_stats19_sample(
    n_samples: int = 3000,
    random_state: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Generates a realistic synthetic UK STATS19 tabular accident dataset.
    Follows real-world STATS19 marginal distributions and accident dynamics:
    - Heavy class imbalance: ~83% slight, ~15% serious, ~2% fatal.
    - Higher speed limits, darkness, rain/fog, and junctions statistically
      elevate severity risk.

    Args:
        n_samples: Total number of records to synthesize.
        random_state: Seed for reproducibility.
        output_path: Optional path to save CSV.

    Returns:
        pd.DataFrame containing tabular accident records.
    """
    rng = np.random.default_rng(random_state)

    road_types = ["motorway", "primary", "secondary", "tertiary", "residential", "unclassified"]
    road_probs = [0.08, 0.28, 0.25, 0.15, 0.20, 0.04]

    times_of_day = ["day", "dusk", "night"]
    time_probs = [0.65, 0.15, 0.20]

    lighting_conditions = ["well_lit", "dim", "dark"]
    weather_conditions = ["fine", "rain", "fog", "snow"]
    weather_probs = [0.72, 0.20, 0.05, 0.03]

    junction_choices = ["no", "yes"]
    junction_probs = [0.45, 0.55]

    data: List[Dict[str, Any]] = []

    for idx in range(n_samples):
        record_id = f"STATS19_2022_{idx + 1:06d}"
        road = rng.choice(road_types, p=road_probs)

        # Speed limits aligned with UK road classifications
        if road == "motorway":
            speed_limit = float(rng.choice([60, 70], p=[0.2, 0.8]))
        elif road == "primary":
            speed_limit = float(rng.choice([40, 50, 60], p=[0.3, 0.4, 0.3]))
        elif road == "secondary":
            speed_limit = float(rng.choice([30, 40, 50], p=[0.5, 0.3, 0.2]))
        elif road == "tertiary":
            speed_limit = float(rng.choice([30, 40], p=[0.7, 0.3]))
        elif road == "residential":
            speed_limit = float(rng.choice([20, 30], p=[0.4, 0.6]))
        else:
            speed_limit = float(rng.choice([30, 40, 50], p=[0.6, 0.3, 0.1]))

        tod = rng.choice(times_of_day, p=time_probs)

        # Lighting dependent on time of day
        if tod == "day":
            light = rng.choice(lighting_conditions, p=[0.92, 0.07, 0.01])
        elif tod == "dusk":
            light = rng.choice(lighting_conditions, p=[0.55, 0.35, 0.10])
        else:  # night
            light = rng.choice(lighting_conditions, p=[0.40, 0.30, 0.30])

        weather = rng.choice(weather_conditions, p=weather_probs)
        junction = rng.choice(junction_choices, p=junction_probs)

        # Severity risk log-odds score (physical intuition from road safety research)
        # Baseline severity score: Slight is dominant
        risk_score = 0.0
        # Speed impact (higher kinetic energy increases fatal/serious probability)
        risk_score += (speed_limit - 30.0) * 0.04
        # Lighting impact
        if light == "dark":
            risk_score += 0.85
        elif light == "dim":
            risk_score += 0.35
        # Weather impact
        if weather in ["fog", "snow"]:
            risk_score += 0.70
        elif weather == "rain":
            risk_score += 0.30
        # Junction impact
        if junction == "yes":
            risk_score += 0.35
        # Road type impact
        if road in ["motorway", "primary"]:
            risk_score += 0.25

        # Sample latent severity using Dirichlet / Softmax logits
        # [logit_slight, logit_serious, logit_fatal]
        # Base logits giving approx ~83% slight, 15% serious, 2% fatal
        logits = np.array([2.2, 0.4 + 0.6 * risk_score, -1.5 + 0.8 * risk_score])
        # Add Gumbel noise for stochastic classification
        gumbel_noise = -np.log(-np.log(rng.uniform(1e-6, 1.0 - 1e-6, size=3)))
        severity = int(np.argmax(logits + gumbel_noise))

        data.append(
            {
                "accident_id": record_id,
                "road_type": road,
                "speed_limit": speed_limit,
                "time_of_day": tod,
                "lighting": light,
                "weather": weather,
                "junction": junction,
                "severity": severity,
                "severity_label": SEVERITY_CLASSES[severity],
                "provenance": "substitute_stats19",
            }
        )

    df = pd.DataFrame(data)

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


class AccidentDataPreprocessor:
    """
    Preprocesses tabular accident records and guarantees ZERO data leakage:
    - Fits encoders and scalers strictly on training fold.
    - Transforms validation and test folds without refitting.
    """

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.feature_names: List[str] = []
        self._is_fitted = False

    def fit(self, X_train: pd.DataFrame) -> "AccidentDataPreprocessor":
        """Fit scaler on numerical features and encoder on categorical features."""
        # Fit scaler on numerical
        self.scaler.fit(X_train[NUMERICAL_FEATURES])

        # Fit encoder on categorical
        self.encoder.fit(X_train[CATEGORICAL_FEATURES])

        # Save feature names for inspectability
        cat_names = self.encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        self.feature_names = NUMERICAL_FEATURES + cat_names
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform numerical and categorical features."""
        if not self._is_fitted:
            raise RuntimeError("AccidentDataPreprocessor must be fitted on training data first!")

        num_scaled = self.scaler.transform(X[NUMERICAL_FEATURES])
        cat_encoded = self.encoder.transform(X[CATEGORICAL_FEATURES])

        return np.hstack([num_scaled, cat_encoded])

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        """Fit on X_train and transform it."""
        return self.fit(X_train).transform(X_train)


def prepare_accident_splits(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    AccidentDataPreprocessor,
    Dict[int, float],
]:
    """
    Performs stratified 3-way split (Train/Val/Test) with strict leakage prevention.

    Args:
        df: Raw DataFrame containing features and 'severity' target.
        test_size: Proportion of dataset for test split (default 0.15).
        val_size: Proportion of dataset for validation split (default 0.15).
        random_state: Seed for reproducible splits.

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test, preprocessor, class_weights)
    """
    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = df["severity"].values.astype(int)

    # First split off the test set (stratified)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Calculate adjusted val size from remaining train_val
    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=adjusted_val_size, stratify=y_train_val, random_state=random_state
    )

    # Fit preprocessor STRICTLY on X_train to prevent leakage
    preprocessor = AccidentDataPreprocessor()
    preprocessor.fit(X_train)

    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # Calculate class weights for handling severe imbalance
    unique_classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=y_train)
    class_weight_dict = {int(cls): float(w) for cls, w in zip(unique_classes, weights)}

    return (
        X_train_trans,
        y_train,
        X_val_trans,
        y_val,
        X_test_trans,
        y_test,
        preprocessor,
        class_weight_dict,
    )

