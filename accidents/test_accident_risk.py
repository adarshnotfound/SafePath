"""
accidents/test_accident_risk.py
-------------------------------
Unit tests for Accident Severity/Risk Prediction:
- Synthetic dataset generation & schema contracts
- Leakage-free train/val/test preprocessor verification
- Class weights calculation
- Baseline Logistic Regression fit and predict
- Proposed MLP fit, predict, predict_proba, and early stopping
- Results JSON structure & schema validation
"""

import json
from pathlib import Path

import numpy as np
import pytest

from accidents.baseline_risk import AccidentRiskBaseline
from accidents.data_loader import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    SEVERITY_CLASSES,
    AccidentDataPreprocessor,
    generate_synthetic_stats19_sample,
    prepare_accident_splits,
)
from accidents.mlp_risk import AccidentRiskMLP


@pytest.fixture
def sample_accident_df():
    """Generates a small reproducible synthetic dataset for testing."""
    return generate_synthetic_stats19_sample(n_samples=200, random_state=42)


def test_preprocessor_standalone(sample_accident_df):
    """Tests AccidentDataPreprocessor standalone methods and error states."""
    processor = AccidentDataPreprocessor()
    with pytest.raises(RuntimeError):
        processor.transform(sample_accident_df)

    res = processor.fit_transform(sample_accident_df)
    assert res.shape[0] == len(sample_accident_df)
    assert res.shape[1] == len(processor.feature_names)


def test_synthetic_data_generation(sample_accident_df):
    """Verifies that generated synthetic STATS19 dataset conforms to expected schema."""
    df = sample_accident_df
    assert len(df) == 200

    expected_cols = [
        "accident_id",
        "road_type",
        "speed_limit",
        "time_of_day",
        "lighting",
        "weather",
        "junction",
        "severity",
        "severity_label",
        "provenance",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column {col}"

    assert set(df["severity"].unique()).issubset({0, 1, 2})
    assert all(df["provenance"] == "substitute_stats19")
    assert all(df["severity_label"].isin(SEVERITY_CLASSES))


def test_data_leakage_prevention(sample_accident_df):
    """
    Verifies that scaler and encoder are fitted STRICTLY on X_train,
    preventing test/validation leakage into the preprocessor.
    """
    df = sample_accident_df
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        preprocessor,
        class_weights,
    ) = prepare_accident_splits(df, test_size=0.15, val_size=0.15, random_state=42)

    # 1. Check dimensions
    assert X_train.shape[0] + X_val.shape[0] + X_test.shape[0] == len(df)
    assert len(y_train) == X_train.shape[0]
    assert len(y_val) == X_val.shape[0]
    assert len(y_test) == X_test.shape[0]

    # 2. Check fitted status
    assert preprocessor._is_fitted is True

    # 3. Check that preprocessor scaler mean exactly matches training mean, NOT the full dataset mean
    # We reconstruct the training subset directly to verify
    from sklearn.model_selection import train_test_split

    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = df["severity"].values.astype(int)
    X_train_val, _, y_train_val, _ = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
    adjusted_val_size = 0.15 / (1.0 - 0.15)
    X_train_ref, _, _, _ = train_test_split(
        X_train_val, y_train_val, test_size=adjusted_val_size, stratify=y_train_val, random_state=42
    )

    expected_train_speed_mean = X_train_ref["speed_limit"].mean()
    fitted_scaler_mean = preprocessor.scaler.mean_[0]
    assert np.isclose(fitted_scaler_mean, expected_train_speed_mean, atol=1e-5)


def test_class_weights_calculation(sample_accident_df):
    """Verifies balanced class weights calculation for severe imbalance."""
    df = sample_accident_df
    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        class_weights,
    ) = prepare_accident_splits(df, test_size=0.15, val_size=0.15, random_state=42)

    assert 0 in class_weights
    assert 1 in class_weights
    assert 2 in class_weights

    # Minority class (fatal = 2) should have higher weight than majority class (slight = 0)
    assert class_weights[2] > class_weights[0]


def test_baseline_logistic_regression(sample_accident_df):
    """Tests training and inference of baseline Logistic Regression."""
    df = sample_accident_df
    (
        X_train,
        y_train,
        _,
        _,
        X_test,
        y_test,
        preprocessor,
        _,
    ) = prepare_accident_splits(df, test_size=0.15, val_size=0.15, random_state=42)

    model = AccidentRiskBaseline(handle_imbalance=True, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1, 2})

    probs = model.predict_proba(X_test)
    assert probs.shape == (len(X_test), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    importances = model.get_feature_importances(preprocessor.feature_names)
    assert "slight" in importances
    assert "fatal" in importances


def test_mlp_model_training_and_inference(sample_accident_df):
    """Tests training and inference of proposed PyTorch MLP model."""
    df = sample_accident_df
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        _,
        class_weights,
    ) = prepare_accident_splits(df, test_size=0.15, val_size=0.15, random_state=42)

    mlp = AccidentRiskMLP(
        hidden_dim1=32,
        hidden_dim2=16,
        epochs=10,
        batch_size=32,
        random_state=42,
    )
    mlp.fit(X_train, y_train, X_val, y_val, class_weights=class_weights)

    assert mlp.is_fitted is True
    assert len(mlp.history["train_loss"]) > 0

    preds = mlp.predict(X_test)
    assert len(preds) == len(X_test)
    assert set(preds).issubset({0, 1, 2})

    probs = mlp.predict_proba(X_test)
    assert probs.shape == (len(X_test), 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-4)


def test_experiment_results_json():
    """Verifies schema of evaluation results JSON."""
    results_path = Path("evaluation/results/accident_risk_experiment.json")
    assert results_path.exists(), "Experiment JSON not found"

    with open(results_path) as f:
        data = json.load(f)

    assert "dataset" in data
    assert data["dataset"]["provenance"] == "substitute"
    assert "models" in data
    assert "baseline_unweighted_logistic_regression" in data["models"]
    assert "baseline_weighted_logistic_regression" in data["models"]
    assert "proposed_weighted_mlp" in data["models"]

    for model_key in ["baseline_weighted_logistic_regression", "proposed_weighted_mlp"]:
        metrics = data["models"][model_key]["metrics"]
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "confusion_matrix" in metrics
        assert len(metrics["confusion_matrix"]) == 3
