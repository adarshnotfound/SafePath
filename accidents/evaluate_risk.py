"""
accidents/evaluate_risk.py
--------------------------
Comparative Evaluation Runner for Accident Severity / Risk Prediction:
- Baseline (Unweighted Logistic Regression)
- Baseline (Balanced Class-Weighted Logistic Regression)
- Proposed (Balanced Class-Weighted Multi-Layer Perceptron)

Metrics:
- Accuracy, Per-Class Precision, Recall, F1
- Macro F1, Weighted F1
- 3x3 Confusion Matrix
- Saves structured evaluation results to evaluation/results/accident_risk_experiment.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from accidents.baseline_risk import AccidentRiskBaseline
from accidents.data_loader import (
    SEVERITY_CLASSES,
    generate_synthetic_stats19_sample,
    prepare_accident_splits,
)
from accidents.mlp_risk import AccidentRiskMLP


def compute_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes comprehensive classification metrics and confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report = classification_report(
        y_true,
        y_pred,
        target_names=SEVERITY_CLASSES,
        labels=[0, 1, 2],
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for idx, cname in enumerate(SEVERITY_CLASSES):
        per_class[cname] = {
            "class_id": idx,
            "precision": round(float(report[cname]["precision"]), 4),
            "recall": round(float(report[cname]["recall"]), 4),
            "f1_score": round(float(report[cname]["f1-score"]), 4),
            "support": int(report[cname]["support"]),
        }

    return {
        "accuracy": round(float(report["accuracy"]), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def run_accident_risk_benchmark(
    csv_path: str = "data/sample/accident_records_stats19.csv",
    output_json: str = "evaluation/results/accident_risk_experiment.json",
    n_samples: int = 3000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes the full comparative benchmark pipeline.
    """
    # 1. Load or synthesize dataset
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"[*] Dataset not found at {csv_path}. Generating synthetic UK STATS19 sample ({n_samples} records)...")
        df = generate_synthetic_stats19_sample(n_samples=n_samples, random_state=random_state, output_path=csv_file)
    else:
        print(f"[*] Loading accident records from {csv_path}...")
        df = pd.read_csv(csv_path)

    print(f"[*] Total records: {len(df)}")
    counts = df["severity"].value_counts().to_dict()
    print("    Class distribution:")
    for cls_id in [0, 1, 2]:
        cname = SEVERITY_CLASSES[cls_id]
        cnt = counts.get(cls_id, 0)
        pct = (cnt / len(df)) * 100.0
        print(f"      - Class {cls_id} ({cname}): {cnt} ({pct:.1f}%)")

    # 2. Prepare leakage-free splits
    print("[*] Performing stratified 70/15/15 train/val/test split with zero leakage...")
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        preprocessor,
        class_weights,
    ) = prepare_accident_splits(df, test_size=0.15, val_size=0.15, random_state=random_state)

    print(f"    Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}")
    print(f"    Encoded feature dimension: {X_train.shape[1]}")
    print(f"    Computed balanced class weights: {class_weights}")

    # 3. Train Baseline (Unweighted Logistic Regression)
    print("\n[*] [1/3] Training Baseline (Unweighted Logistic Regression)...")
    baseline_unweighted = AccidentRiskBaseline(handle_imbalance=False, random_state=random_state)
    baseline_unweighted.fit(X_train, y_train)
    y_pred_base_unweighted = baseline_unweighted.predict(X_test)
    metrics_base_unweighted = compute_metrics_dict(y_test, y_pred_base_unweighted)

    # 4. Train Baseline (Weighted Logistic Regression)
    print("[*] [2/3] Training Baseline (Balanced Class-Weighted Logistic Regression)...")
    baseline_weighted = AccidentRiskBaseline(handle_imbalance=True, random_state=random_state)
    baseline_weighted.fit(X_train, y_train)
    y_pred_base_weighted = baseline_weighted.predict(X_test)
    metrics_base_weighted = compute_metrics_dict(y_test, y_pred_base_weighted)

    # 5. Train Proposed (Balanced Class-Weighted PyTorch MLP)
    print("[*] [3/3] Training Proposed (Balanced Class-Weighted Multi-Layer Perceptron)...")
    mlp_model = AccidentRiskMLP(
        hidden_dim1=64,
        hidden_dim2=32,
        lr=0.002,
        weight_decay=1e-4,
        batch_size=64,
        epochs=80,
        patience=15,
        random_state=random_state,
    )
    mlp_model.fit(X_train, y_train, X_val, y_val, class_weights=class_weights)
    y_pred_mlp = mlp_model.predict(X_test)
    metrics_mlp = compute_metrics_dict(y_test, y_pred_mlp)

    # Compile benchmark results
    results: Dict[str, Any] = {
        "dataset": {
            "name": "UK STATS19 (Department for Transport)",
            "provenance": "substitute",
            "total_records": len(df),
            "splits": {
                "train_samples": int(X_train.shape[0]),
                "val_samples": int(X_val.shape[0]),
                "test_samples": int(X_test.shape[0]),
            },
            "features": preprocessor.feature_names,
            "classes": SEVERITY_CLASSES,
            "class_weights": class_weights,
        },
        "models": {
            "baseline_unweighted_logistic_regression": {
                "name": "Logistic Regression (Unweighted)",
                "type": "baseline",
                "imbalance_handling": "none",
                "metrics": metrics_base_unweighted,
            },
            "baseline_weighted_logistic_regression": {
                "name": "Logistic Regression (Class-Weighted Balanced)",
                "type": "baseline",
                "imbalance_handling": "class_weight='balanced'",
                "metrics": metrics_base_weighted,
            },
            "proposed_weighted_mlp": {
                "name": "Multi-Layer Perceptron (Class-Weighted)",
                "type": "proposed",
                "architecture": "Linear(64) -> BatchNorm -> ReLU -> Dropout(0.2) -> Linear(32) -> BatchNorm -> ReLU -> Dropout(0.2) -> Linear(3)",
                "imbalance_handling": "weighted_cross_entropy_loss",
                "metrics": metrics_mlp,
            },
        },
        "summary_comparison": {
            "accuracy": {
                "baseline_unweighted": metrics_base_unweighted["accuracy"],
                "baseline_weighted": metrics_base_weighted["accuracy"],
                "proposed_mlp": metrics_mlp["accuracy"],
            },
            "macro_f1": {
                "baseline_unweighted": metrics_base_unweighted["macro_f1"],
                "baseline_weighted": metrics_base_weighted["macro_f1"],
                "proposed_mlp": metrics_mlp["macro_f1"],
            },
            "fatal_recall": {
                "baseline_unweighted": metrics_base_unweighted["per_class"]["fatal"]["recall"],
                "baseline_weighted": metrics_base_weighted["per_class"]["fatal"]["recall"],
                "proposed_mlp": metrics_mlp["per_class"]["fatal"]["recall"],
            },
        },
    }

    # Save to JSON
    out_path = Path(output_json)
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Results successfully written to {output_json}")

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'MODEL':<35} | {'ACCURACY':<10} | {'MACRO F1':<10} | {'FATAL RECALL':<12} | {'FATAL F1':<10}")
    print("-" * 80)
    for model_key, model_data in results["models"].items():
        m = model_data["metrics"]
        print(
            f"{model_data['name']:<35} | "
            f"{m['accuracy']:<10.4f} | "
            f"{m['macro_f1']:<10.4f} | "
            f"{m['per_class']['fatal']['recall']:<12.4f} | "
            f"{m['per_class']['fatal']['f1_score']:<10.4f}"
        )
    print("=" * 80)

    print("\n[*] Confusion Matrix (Proposed MLP):")
    header_str = "True / Pred"
    print(f"{header_str:<15} | {'Slight':<10} | {'Serious':<10} | {'Fatal':<10}")
    print("-" * 50)
    for idx, cname in enumerate(SEVERITY_CLASSES):
        row = metrics_mlp["confusion_matrix"][idx]
        print(f"{cname:<15} | {row[0]:<10} | {row[1]:<10} | {row[2]:<10}")
    print("=" * 50)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Accident Severity Risk Prediction Evaluation")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/sample/accident_records_stats19.csv",
        help="Path to tabular accident CSV",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="evaluation/results/accident_risk_experiment.json",
        help="Path to output JSON",
    )
    parser.add_argument("--samples", type=int, default=3000, help="Number of samples to synthesize if CSV absent")
    args = parser.parse_args()

    run_accident_risk_benchmark(
        csv_path=args.csv_path,
        output_json=args.output_json,
        n_samples=args.samples,
    )

