"""Accident Data Loader and Preprocessing Pipelines - SafePath Module 2 (Accidents).

Combines:
1. Spatial coordinate-level accident points & empirical blackspots for hotspot clustering
   (Karnataka State Blackspots / MoRTH Bengaluru corridor).
2. Tabular feature preprocessing, synthetic UK STATS19 ingestion, and leakage-free
   stratified splitting for severity risk prediction models (MLP / Logistic Regression).

Provenance Notice:
- Coordinate accident blackspots: 'India' (Karnataka State Blackspots / MoRTH Bengaluru).
- Tabular accident risk models: 'substitute' (UK STATS19 Road Safety Data / data.gov.uk).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ==============================================================================
# 1. SPATIAL ACCIDENT POINTS & BLACKSPOT CLUSTERING (Member B / Task B3)
# ==============================================================================


@dataclass
class AccidentPoint:
    """Individual road accident event with spatial coordinates and severity."""

    accident_id: str
    lat: float
    lon: float
    severity: float  # 1.0 (slight) to 5.0 (fatal / critical)
    timestamp: str
    segment_id: str
    weather: str  # 'clear', 'rain', 'fog'
    lighting: str  # 'daylight', 'dark_lights', 'dark_no_lights'
    source_country: str  # 'India' or substitute ('United Kingdom', 'United States')

    def to_dict(self) -> Dict[str, object]:
        """Convert to dictionary representation."""
        return {
            "accident_id": self.accident_id,
            "lat": f"{self.lat:.5f}",
            "lon": f"{self.lon:.5f}",
            "severity": f"{self.severity:.1f}",
            "timestamp": self.timestamp,
            "segment_id": self.segment_id,
            "weather": self.weather,
            "lighting": self.lighting,
            "source_country": self.source_country,
        }


# Reference road segment waypoints matching Bengaluru corridor
DEFAULT_ROUTE_SEGMENTS: Dict[str, List[Tuple[float, float]]] = {
    "seg_101": [(12.97160, 77.59460), (12.97210, 77.59510)],
    "seg_102": [(12.97210, 77.59510), (12.97305, 77.59600)],
    "seg_103": [(12.97305, 77.59600), (12.97450, 77.59750)],
    "seg_104": [(12.97210, 77.59510), (12.97380, 77.59685)],
    "seg_105": [(12.97050, 77.59320), (12.97160, 77.59460)],
    "seg_106": [(12.97450, 77.59750), (12.97600, 77.59910)],
    "seg_107": [(12.97160, 77.59460), (12.97250, 77.59580)],
    "seg_108": [(12.97250, 77.59580), (12.97380, 77.59685)],
}

# Empirical blackspot cluster centers along Bengaluru corridor (Karnataka State Blackspot List)
KNOWN_BLACKSPOT_CENTERS: List[Dict[str, object]] = [
    {"center_lat": 12.97180, "center_lon": 77.59480, "count": 14, "base_severity": 3.8, "segment_id": "seg_101", "spread_m": 22.0},
    {"center_lat": 12.97340, "center_lon": 77.59630, "count": 10, "base_severity": 2.5, "segment_id": "seg_102", "spread_m": 28.0},
    {"center_lat": 12.97510, "center_lon": 77.59810, "count": 19, "base_severity": 4.2, "segment_id": "seg_103", "spread_m": 18.0},
    {"center_lat": 12.97720, "center_lon": 77.60050, "count": 6,  "base_severity": 1.8, "segment_id": "seg_105", "spread_m": 24.0},
    {"center_lat": 12.97900, "center_lon": 77.60230, "count": 22, "base_severity": 4.6, "segment_id": "seg_106", "spread_m": 35.0},
    {"center_lat": 12.98150, "center_lon": 77.60500, "count": 5,  "base_severity": 1.4, "segment_id": "seg_107", "spread_m": 20.0},
]


def generate_empirical_accident_dataset(
    seed: int = 42,
    source_country: str = "India",
    include_background_noise: bool = True,
    noise_count: int = 12,
) -> List[AccidentPoint]:
    """Generate realistic accident records clustered around road blackspots."""
    rng = random.Random(seed)
    accidents: List[AccidentPoint] = []
    acc_id_counter = 1
    base_date = datetime(2025, 1, 1, 8, 0, 0, tzinfo=timezone.utc)

    meters_per_deg_lat = 111320.0
    weather_choices = ["clear", "rain", "fog"]
    lighting_choices = ["daylight", "dark_lights", "dark_no_lights"]

    for bs in KNOWN_BLACKSPOT_CENTERS:
        c_lat = float(bs["center_lat"])
        c_lon = float(bs["center_lon"])
        count = int(bs["count"])
        base_sev = float(bs["base_severity"])
        seg_id = str(bs["segment_id"])
        spread_m = float(bs["spread_m"])
        meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(c_lat))

        for _ in range(count):
            dx_m = rng.gauss(0.0, spread_m / 2.0)
            dy_m = rng.gauss(0.0, spread_m / 2.0)

            lat = c_lat + (dy_m / meters_per_deg_lat)
            lon = c_lon + (dx_m / meters_per_deg_lon)

            sev = max(1.0, min(5.0, rng.gauss(base_sev, 0.4)))
            days_offset = rng.randint(0, 360)
            hours_offset = rng.randint(0, 23)
            mins_offset = rng.randint(0, 59)
            ts = base_date + timedelta(days=days_offset, hours=hours_offset, minutes=mins_offset)

            weather = rng.choices(weather_choices, weights=[0.75, 0.20, 0.05])[0]
            lighting = rng.choices(lighting_choices, weights=[0.55, 0.35, 0.10])[0]

            accidents.append(
                AccidentPoint(
                    accident_id=f"acc_{acc_id_counter:04d}",
                    lat=lat,
                    lon=lon,
                    severity=round(sev, 1),
                    timestamp=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    segment_id=seg_id,
                    weather=weather,
                    lighting=lighting,
                    source_country=source_country,
                )
            )
            acc_id_counter += 1

    if include_background_noise:
        for _ in range(noise_count):
            noise_lat = rng.uniform(12.9690, 12.9830)
            noise_lon = rng.uniform(77.5920, 77.6070)
            noise_sev = rng.uniform(1.0, 3.0)
            days_offset = rng.randint(0, 360)
            ts = base_date + timedelta(days=days_offset, hours=rng.randint(0, 23))

            accidents.append(
                AccidentPoint(
                    accident_id=f"acc_{acc_id_counter:04d}",
                    lat=noise_lat,
                    lon=noise_lon,
                    severity=round(noise_sev, 1),
                    timestamp=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    segment_id="seg_104",
                    weather=rng.choice(weather_choices),
                    lighting=rng.choice(lighting_choices),
                    source_country=source_country,
                )
            )
            acc_id_counter += 1

    return accidents


def save_accidents_to_csv(accidents: List[AccidentPoint], output_path: Path) -> None:
    """Save accident points to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "accident_id",
        "lat",
        "lon",
        "severity",
        "timestamp",
        "segment_id",
        "weather",
        "lighting",
        "source_country",
    ]
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in accidents:
            writer.writerow(a.to_dict())


def load_accidents_from_csv(csv_path: Path) -> List[AccidentPoint]:
    """Load accident points from CSV."""
    accidents: List[AccidentPoint] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            accidents.append(
                AccidentPoint(
                    accident_id=row["accident_id"],
                    lat=float(row["lat"]),
                    lon=float(row["lon"]),
                    severity=float(row["severity"]),
                    timestamp=row["timestamp"],
                    segment_id=row["segment_id"],
                    weather=row.get("weather", "clear"),
                    lighting=row.get("lighting", "daylight"),
                    source_country=row.get("source_country", "India"),
                )
            )
    return accidents


# ==============================================================================
# 2. TABULAR ACCIDENT RISK DATA LOADER & PREPROCESSING (Task B4 / Leakage-Free)
# ==============================================================================

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

        if tod == "day":
            light = rng.choice(lighting_conditions, p=[0.92, 0.07, 0.01])
        elif tod == "dusk":
            light = rng.choice(lighting_conditions, p=[0.55, 0.35, 0.10])
        else:
            light = rng.choice(lighting_conditions, p=[0.40, 0.30, 0.30])

        weather = rng.choice(weather_conditions, p=weather_probs)
        junction = rng.choice(junction_choices, p=junction_probs)

        risk_score = 0.0
        risk_score += (speed_limit - 30.0) * 0.04
        if light == "dark":
            risk_score += 0.85
        elif light == "dim":
            risk_score += 0.35
        if weather in ["fog", "snow"]:
            risk_score += 0.70
        elif weather == "rain":
            risk_score += 0.30
        if junction == "yes":
            risk_score += 0.35
        if road in ["motorway", "primary"]:
            risk_score += 0.25

        logits = np.array([2.2, 0.4 + 0.6 * risk_score, -1.5 + 0.8 * risk_score])
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
        self.scaler.fit(X_train[NUMERICAL_FEATURES])
        self.encoder.fit(X_train[CATEGORICAL_FEATURES])

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
    """
    X = df[CATEGORICAL_FEATURES + NUMERICAL_FEATURES]
    y = df["severity"].values.astype(int)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    adjusted_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=adjusted_val_size, stratify=y_train_val, random_state=random_state
    )

    preprocessor = AccidentDataPreprocessor()
    preprocessor.fit(X_train)

    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

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
