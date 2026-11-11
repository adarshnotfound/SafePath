"""Accident Risk Prediction Neural Network (MLP vs. Logistic Regression) - SafePath Module B3/B4.

Trains and evaluates accident risk prediction models at the road segment level:
1. Feature extraction fusing road segment physical attributes with accident hotspot proximity:
   - Distance to nearest accident hotspot (meters)
   - Local hotspot density within 250m buffer
   - Nearby hotspot severity weight
   - Road type, speed limit, lighting proxy, segment length, isolation, and time-of-day context
2. Models:
   - Non-linear Neural Network: Multilayer Perceptron (MLP) with ReLU activations and Adam optimizer.
   - Linear Baseline: Logistic Regression (classification) and Ridge Regression (continuous risk).
3. Evaluation:
   - Classification: Accuracy, Precision, Recall, F1-score, and ROC-AUC.
   - Continuous score regression: Mean Squared Error (MSE), RMSE, MAE, and R^2.
   - ROC curves and loss convergence plots exported to evaluation/plots/accident_risk_roc_curve.png.
   - Metrics manifest saved to evaluation/results/accident_risk_metrics.json.

Owner: Member B (Hazard map and accident-data lead)
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from accidents.clustering import AccidentHotspot, haversine_distance_meters

# Reference road segment geometries matching Bengaluru corridor
DEFAULT_SEGMENTS: Dict[str, Dict[str, Any]] = {
    "seg_101": {"length_m": 240.5, "road_type": "secondary",   "speed_limit": 40, "lit": 1, "isolation": 0.20, "mid_lat": 12.97185, "mid_lon": 77.59485},
    "seg_102": {"length_m": 185.0, "road_type": "tertiary",    "speed_limit": 30, "lit": 0, "isolation": 0.45, "mid_lat": 12.97258, "mid_lon": 77.59555},
    "seg_103": {"length_m": 310.2, "road_type": "secondary",   "speed_limit": 40, "lit": 1, "isolation": 0.15, "mid_lat": 12.97378, "mid_lon": 77.59675},
    "seg_104": {"length_m": 150.0, "road_type": "residential", "speed_limit": 25, "lit": 0, "isolation": 0.60, "mid_lat": 12.97295, "mid_lon": 77.59598},
    "seg_105": {"length_m": 420.0, "road_type": "primary",     "speed_limit": 50, "lit": 1, "isolation": 0.10, "mid_lat": 12.97105, "mid_lon": 77.59390},
    "seg_106": {"length_m": 275.5, "road_type": "secondary",   "speed_limit": 40, "lit": 0, "isolation": 0.75, "mid_lat": 12.97525, "mid_lon": 77.59830},
    "seg_107": {"length_m": 190.0, "road_type": "tertiary",    "speed_limit": 30, "lit": 1, "isolation": 0.30, "mid_lat": 12.97205, "mid_lon": 77.59520},
    "seg_108": {"length_m": 350.0, "road_type": "primary",     "speed_limit": 50, "lit": 1, "isolation": 0.10, "mid_lat": 12.97315, "mid_lon": 77.59632},
}

ROAD_TYPE_MAP = {"primary": 3, "secondary": 2, "tertiary": 1, "residential": 0}
TIME_OF_DAY_MAP = {"day": 0, "dusk": 1, "night": 2}


@dataclass
class SegmentRiskFeatures:
    """Feature vector for accident risk prediction on a road segment."""

    segment_id: str
    length_m: float
    road_type_enc: int
    speed_limit: int
    lit: int
    isolation_score: float
    time_of_day_enc: int
    dist_to_nearest_hotspot_m: float
    hotspot_density_250m: int
    nearby_hotspot_max_severity: float
    continuous_risk_target: float  # [0.0, 1.0]
    binary_risk_target: int        # 1 = high risk (>= 0.50), 0 = normal/low risk

    def to_feature_array(self) -> np.ndarray:
        """Convert to numerical feature array."""
        return np.array([
            self.length_m,
            float(self.road_type_enc),
            float(self.speed_limit),
            float(self.lit),
            self.isolation_score,
            float(self.time_of_day_enc),
            self.dist_to_nearest_hotspot_m,
            float(self.hotspot_density_250m),
            self.nearby_hotspot_max_severity,
        ], dtype=float)


def compute_segment_hotspot_features(
    mid_lat: float,
    mid_lon: float,
    hotspots: List[AccidentHotspot],
    buffer_radius_m: float = 250.0,
) -> Tuple[float, int, float]:
    """Calculate proximity, density, and severity features from nearest accident hotspots.

    Args:
        mid_lat: Segment midpoint latitude.
        mid_lon: Segment midpoint longitude.
        hotspots: List of detected accident hotspots.
        buffer_radius_m: Buffer radius in meters to calculate density.

    Returns:
        (min_distance_m, hotspot_count_within_buffer, max_severity_within_buffer)
    """
    if not hotspots:
        return 999.0, 0, 0.0

    distances = [
        haversine_distance_meters(mid_lat, mid_lon, h.lat, h.lon)
        for h in hotspots
    ]
    min_dist = min(distances)

    nearby = [
        h for h, d in zip(hotspots, distances)
        if d <= buffer_radius_m
    ]
    density = len(nearby)
    max_sev = max((h.severity_weight for h in nearby), default=0.0)

    return min_dist, density, max_sev


def generate_segment_risk_dataset(
    hotspots: List[AccidentHotspot],
    samples_per_segment: int = 25,
    seed: int = 42,
) -> List[SegmentRiskFeatures]:
    """Synthesize representative training dataset across road segments, weather, and lighting contexts.

    Args:
        hotspots: Clustered accident hotspots.
        samples_per_segment: Number of contextual samples per segment.
        seed: Random seed for repeatability.

    Returns:
        List of SegmentRiskFeatures.
    """
    rng = np.random.RandomState(seed)
    dataset: List[SegmentRiskFeatures] = []

    times_of_day = ["day", "dusk", "night"]

    for seg_id, seg_data in DEFAULT_SEGMENTS.items():
        base_len = float(seg_data["length_m"])
        road_type = seg_data["road_type"]
        base_speed = int(seg_data["speed_limit"])
        base_lit = int(seg_data["lit"])
        base_iso = float(seg_data["isolation"])
        c_lat = float(seg_data["mid_lat"])
        c_lon = float(seg_data["mid_lon"])

        min_d, density, max_sev = compute_segment_hotspot_features(c_lat, c_lon, hotspots)

        for _ in range(samples_per_segment):
            # Contextual variations: time of day, weather lighting dropouts
            tod = rng.choice(times_of_day)
            tod_enc = TIME_OF_DAY_MAP[tod]

            # Lighting can occasionally be unlit at night/dusk due to outages
            lit_val = base_lit
            if tod in ("night", "dusk") and rng.rand() < 0.25:
                lit_val = 0

            # Slight perturbations in speed/isolation context
            speed_val = base_speed + rng.choice([-5, 0, 5])
            iso_val = max(0.0, min(1.0, base_iso + rng.normal(0.0, 0.05)))
            len_val = max(50.0, base_len + rng.normal(0.0, 10.0))

            # Ground-truth non-linear physical risk formulation:
            # High risk: near high-severity blackspot, unlit conditions, night context, high speed
            proximity_factor = math.exp(-min_d / 80.0)  # Decays smoothly beyond 80m
            severity_factor = (max_sev / 5.0) if max_sev > 0 else 0.1
            darkness_factor = 0.35 if (lit_val == 0 and tod == "night") else (0.15 if tod == "night" else 0.0)
            speed_factor = (speed_val - 25.0) / 35.0 * 0.20
            isolation_factor = iso_val * 0.15

            # Non-linear interaction between proximity and darkness
            interaction = 0.25 * proximity_factor * (1.0 if lit_val == 0 else 0.3)

            base_risk = (
                0.40 * proximity_factor * severity_factor
                + darkness_factor
                + speed_factor
                + isolation_factor
                + interaction
                + rng.normal(0.0, 0.04)  # Small observational noise
            )
            continuous_risk = float(np.clip(base_risk, 0.05, 0.95))
            binary_risk = 1 if continuous_risk >= 0.50 else 0

            dataset.append(SegmentRiskFeatures(
                segment_id=seg_id,
                length_m=round(len_val, 1),
                road_type_enc=ROAD_TYPE_MAP.get(road_type, 1),
                speed_limit=speed_val,
                lit=lit_val,
                isolation_score=round(iso_val, 2),
                time_of_day_enc=tod_enc,
                dist_to_nearest_hotspot_m=round(min_d, 1),
                hotspot_density_250m=density,
                nearby_hotspot_max_severity=round(max_sev, 1),
                continuous_risk_target=round(continuous_risk, 3),
                binary_risk_target=binary_risk,
            ))

    return dataset


class AccidentRiskModelPipeline:
    """Trains, compares, and evaluates MLP Neural Network vs. Logistic Regression Baseline."""

    def __init__(self, random_seed: int = 42):
        self.seed = random_seed
        self.scaler = StandardScaler()

        # Non-linear Multi-Layer Perceptron (Neural Network)
        self.mlp_classifier = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.005,
            alpha=0.001,
            max_iter=800,
            random_state=self.seed,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
        )
        self.mlp_regressor = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            learning_rate_init=0.005,
            alpha=0.001,
            max_iter=800,
            random_state=self.seed,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
        )

        # Linear Baselines
        self.logreg_baseline = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=500,
            random_state=self.seed,
        )
        self.ridge_baseline = Ridge(alpha=1.0, random_state=self.seed)

        self.is_trained = False
        self.feature_names = [
            "length_m",
            "road_type",
            "speed_limit",
            "lit",
            "isolation_score",
            "time_of_day",
            "dist_to_nearest_hotspot_m",
            "hotspot_density_250m",
            "nearby_hotspot_max_severity",
        ]

    def train_and_evaluate(
        self,
        dataset: List[SegmentRiskFeatures],
        test_size: float = 0.20,
    ) -> Dict[str, Any]:
        """Fit both models and compile comparative evaluation metrics and ROC data."""
        X = np.array([f.to_feature_array() for f in dataset])
        y_bin = np.array([f.binary_risk_target for f in dataset])
        y_cont = np.array([f.continuous_risk_target for f in dataset])

        # Train / Test Split
        X_train, X_test, y_bin_train, y_bin_test, y_cont_train, y_cont_test = train_test_split(
            X, y_bin, y_cont, test_size=test_size, random_state=self.seed, stratify=y_bin
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 1. Fit MLP Neural Networks
        self.mlp_classifier.fit(X_train_scaled, y_bin_train)
        self.mlp_regressor.fit(X_train_scaled, y_cont_train)

        # 2. Fit Linear Baselines
        self.logreg_baseline.fit(X_train_scaled, y_bin_train)
        self.ridge_baseline.fit(X_train_scaled, y_cont_train)
        self.is_trained = True

        # Predictions
        mlp_pred_bin = self.mlp_classifier.predict(X_test_scaled)
        mlp_prob = self.mlp_classifier.predict_proba(X_test_scaled)[:, 1]
        mlp_pred_cont = self.mlp_regressor.predict(X_test_scaled)

        logreg_pred_bin = self.logreg_baseline.predict(X_test_scaled)
        logreg_prob = self.logreg_baseline.predict_proba(X_test_scaled)[:, 1]
        ridge_pred_cont = self.ridge_baseline.predict(X_test_scaled)

        # Compute Classification Metrics
        def classification_metrics(y_true, y_pred, y_prob):
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            return {
                "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
                "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
                "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
                "roc_auc": round(float(auc(fpr, tpr)), 4),
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
            }

        # Compute Continuous Regression Metrics
        def regression_metrics(y_true, y_pred):
            mse = float(mean_squared_error(y_true, y_pred))
            return {
                "mse": round(mse, 4),
                "rmse": round(float(math.sqrt(mse)), 4),
                "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
                "r2": round(float(r2_score(y_true, y_pred)), 4),
            }

        mlp_clf_eval = classification_metrics(y_bin_test, mlp_pred_bin, mlp_prob)
        logreg_clf_eval = classification_metrics(y_bin_test, logreg_pred_bin, logreg_prob)

        mlp_reg_eval = regression_metrics(y_cont_test, mlp_pred_cont)
        ridge_reg_eval = regression_metrics(y_cont_test, ridge_pred_cont)

        # Loss curve from MLP training
        loss_curve = [float(loss) for loss in self.mlp_classifier.loss_curve_]

        report = {
            "dataset_summary": {
                "total_samples": len(dataset),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "high_risk_ratio": round(float(np.mean(y_bin)), 4),
                "features": self.feature_names,
            },
            "models": {
                "mlp_neural_network": {
                    "architecture": "MLP (64, 32) -> ReLU -> Adam",
                    "classification": {
                        "accuracy": mlp_clf_eval["accuracy"],
                        "precision": mlp_clf_eval["precision"],
                        "recall": mlp_clf_eval["recall"],
                        "f1_score": mlp_clf_eval["f1_score"],
                        "roc_auc": mlp_clf_eval["roc_auc"],
                    },
                    "continuous_risk_regression": mlp_reg_eval,
                    "epochs_converged": len(loss_curve),
                },
                "logistic_regression_baseline": {
                    "architecture": "Logistic Regression / Ridge Linear Baseline",
                    "classification": {
                        "accuracy": logreg_clf_eval["accuracy"],
                        "precision": logreg_clf_eval["precision"],
                        "recall": logreg_clf_eval["recall"],
                        "f1_score": logreg_clf_eval["f1_score"],
                        "roc_auc": logreg_clf_eval["roc_auc"],
                    },
                    "continuous_risk_regression": ridge_reg_eval,
                },
            },
            "roc_curves": {
                "mlp": {"fpr": mlp_clf_eval["fpr"], "tpr": mlp_clf_eval["tpr"], "auc": mlp_clf_eval["roc_auc"]},
                "logistic_regression": {"fpr": logreg_clf_eval["fpr"], "tpr": logreg_clf_eval["tpr"], "auc": logreg_clf_eval["roc_auc"]},
            },
            "training_loss_curve": loss_curve,
        }
        return report

    def predict_segment_risk(self, feature: SegmentRiskFeatures) -> Dict[str, float]:
        """Predict continuous accident risk score [0, 1] and binary high-risk flag for a segment."""
        if not self.is_trained:
            raise RuntimeError("Model pipeline must be trained before inference.")
        x_scaled = self.scaler.transform(feature.to_feature_array().reshape(1, -1))
        cont_risk = float(np.clip(self.mlp_regressor.predict(x_scaled)[0], 0.0, 1.0))
        high_risk_prob = float(self.mlp_classifier.predict_proba(x_scaled)[0][1])
        return {
            "accident_score": round(cont_risk, 3),
            "high_risk_probability": round(high_risk_prob, 3),
            "is_high_risk": 1 if high_risk_prob >= 0.50 else 0,
        }


def plot_risk_comparison_roc(
    report: Dict[str, Any],
    output_png_path: Path,
) -> None:
    """Render publication-quality ROC Curves and Training Loss progression."""
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    mlp_roc = report["roc_curves"]["mlp"]
    logreg_roc = report["roc_curves"]["logistic_regression"]
    loss_curve = report["training_loss_curve"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=300)

    # Subplot 1: ROC Curves
    ax1.plot(mlp_roc["fpr"], mlp_roc["tpr"], color="#2980B9", lw=2.5, label=f"MLP Neural Net (AUC = {mlp_roc['auc']:.3f})")
    ax1.plot(logreg_roc["fpr"], logreg_roc["tpr"], color="#E67E22", lw=2.0, linestyle="--", label=f"Logistic Regression (AUC = {logreg_roc['auc']:.3f})")
    ax1.plot([0, 1], [0, 1], color="#95A5A6", linestyle=":", lw=1.5, label="Random Chance (AUC = 0.500)")

    ax1.set_title("ROC Curve: Accident Risk Classification", fontsize=11, fontweight="bold")
    ax1.set_xlabel("False Positive Rate", fontsize=10)
    ax1.set_ylabel("True Positive Rate", fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])

    # Subplot 2: MLP Loss Convergence
    epochs = range(1, len(loss_curve) + 1)
    ax2.plot(epochs, loss_curve, color="#27AE60", lw=2.2, marker="o", markersize=3, label="Training Loss (Cross-Entropy)")
    ax2.set_title("MLP Training Loss Convergence", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Iteration Epoch", fontsize=10)
    ax2.set_ylabel("Loss", fontsize=10)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_png_path)
    plt.close()


def run_accident_risk_pipeline(
    hotspots_csv_path: Optional[Path] = None,
    output_metrics_path: Optional[Path] = None,
    output_plot_path: Optional[Path] = None,
    seed: int = 42,
) -> Tuple[AccidentRiskModelPipeline, Dict[str, Any]]:
    """Train accident risk neural network, evaluate vs baseline, and export report and plots."""
    repo_root = Path(__file__).resolve().parent.parent

    if hotspots_csv_path is None:
        hotspots_csv_path = repo_root / "data" / "processed" / "accident_hotspots.csv"
    if output_metrics_path is None:
        output_metrics_path = repo_root / "evaluation" / "results" / "accident_risk_metrics.json"
    if output_plot_path is None:
        output_plot_path = repo_root / "evaluation" / "plots" / "accident_risk_roc_curve.png"

    # Load hotspots
    hotspots: List[AccidentHotspot] = []
    if hotspots_csv_path.exists():
        with open(hotspots_csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hotspots.append(AccidentHotspot(
                    hotspot_id=row["hotspot_id"],
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    radius_m=float(row["radius_m"]),
                    accident_count=int(row["accident_count"]),
                    severity_weight=float(row["severity_weight"]),
                    segment_id=row["segment_id"],
                    source_country=row.get("source_country", "India"),
                ))

    # Generate contextual segment risk dataset
    dataset = generate_segment_risk_dataset(hotspots, samples_per_segment=35, seed=seed)

    # Train and evaluate pipeline
    pipeline = AccidentRiskModelPipeline(random_seed=seed)
    report = pipeline.train_and_evaluate(dataset)

    # Save metrics JSON per Rule 4
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_path, mode="w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Plot and save ROC Curve
    plot_risk_comparison_roc(report, output_plot_path)

    # Also save copy in evaluation/results for unified reporting
    results_plot_path = repo_root / "evaluation" / "results" / "accident_risk_roc_curve.png"
    results_plot_path.parent.mkdir(parents=True, exist_ok=True)
    plot_risk_comparison_roc(report, results_plot_path)

    return pipeline, report


if __name__ == "__main__":
    _, rep = run_accident_risk_pipeline()
    mlp = rep["models"]["mlp_neural_network"]
    lr = rep["models"]["logistic_regression_baseline"]

    print("=" * 70)
    print("SafePath Accident Risk Model: MLP Neural Net vs. Logistic Regression")
    print("=" * 70)
    print(f"{'Metric':<28} | {'MLP Neural Network':<20} | {'Logistic Regression':<20}")
    print("-" * 70)
    print(f"{'Accuracy':<28} | {mlp['classification']['accuracy']:<20} | {lr['classification']['accuracy']:<20}")
    print(f"{'Precision':<28} | {mlp['classification']['precision']:<20} | {lr['classification']['precision']:<20}")
    print(f"{'Recall':<28} | {mlp['classification']['recall']:<20} | {lr['classification']['recall']:<20}")
    print(f"{'F1-Score':<28} | {mlp['classification']['f1_score']:<20} | {lr['classification']['f1_score']:<20}")
    print(f"{'ROC-AUC':<28} | {mlp['classification']['roc_auc']:<20} | {lr['classification']['roc_auc']:<20}")
    print("-" * 70)
    print(f"{'Continuous Risk MSE':<28} | {mlp['continuous_risk_regression']['mse']:<20} | {lr['continuous_risk_regression']['mse']:<20}")
    print(f"{'Continuous Risk R^2':<28} | {mlp['continuous_risk_regression']['r2']:<20} | {lr['continuous_risk_regression']['r2']:<20}")
    print("=" * 70)
    print("Metrics written to: evaluation/results/accident_risk_metrics.json")
    print("Plot written to:    evaluation/plots/accident_risk_roc_curve.png")
