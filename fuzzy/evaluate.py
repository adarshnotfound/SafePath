"""Comparative Evaluation Script for SafePath Safety Scoring (Section 4.6 & Section 9).

Compares:
1. Fixed Weighted-Sum Baseline (Baseline A)
2. Hand-Written Mamdani Fuzzy System (Baseline B)
3. GA-Tuned Fuzzy Engine (Proposed Soft Computing Method)

Evaluates on survey validation targets with 5-fold cross-validation and fixed seed (seed 42).
Computes MAE, Pearson correlation (r), Spearman rank correlation (rho), and 95% bootstrap confidence intervals.
Plots predicted vs rated scores and exports figures and metrics to /evaluation/results/.
"""

import csv
import json
import sys
from pathlib import Path

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from typing import Dict, Any, List, Tuple
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold

from fuzzy.baseline_weighted import WeightedSumBaselineEngine
from fuzzy.mamdani_engine import MamdaniFuzzyEngine
from fuzzy.ga_tuner import FuzzyGeneticTuner, survey_rating_to_safety_score


def load_dataset(
    features_csv: Path,
    ratings_csv: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Load road segment features and compute average survey target scores per segment."""
    with open(features_csv, mode="r", encoding="utf-8") as f:
        segments = list(csv.DictReader(f))

    ratings_by_segment: Dict[str, List[float]] = {}
    with open(ratings_csv, mode="r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seg_id = row.get("segment_id", "").strip()
            r_str = row.get("rating", "").strip()
            if seg_id and r_str:
                try:
                    r_val = float(r_str)
                    ratings_by_segment.setdefault(seg_id, []).append(r_val)
                except ValueError:
                    continue

    targets: Dict[str, float] = {}
    for seg_id, r_list in ratings_by_segment.items():
        mean_rating = float(np.mean(r_list))
        targets[seg_id] = survey_rating_to_safety_score(mean_rating)

    # Filter segments to only those present in targets
    valid_segments = [s for s in segments if s.get("segment_id") in targets]
    return valid_segments, targets


def compute_metrics(predictions: List[float], targets: List[float], n_bootstraps: int = 1000) -> Dict[str, Any]:
    """Compute MAE, Pearson r, Spearman rho, and 95% bootstrap confidence intervals."""
    p_arr = np.array(predictions, dtype=float)
    t_arr = np.array(targets, dtype=float)
    n = len(p_arr)

    mae = float(np.mean(np.abs(p_arr - t_arr)))

    if np.std(p_arr) > 1e-5 and np.std(t_arr) > 1e-5:
        p_corr, p_pval = pearsonr(p_arr, t_arr)
        s_corr, s_pval = spearmanr(p_arr, t_arr)
    else:
        p_corr, p_pval = 0.0, 1.0
        s_corr, s_pval = 0.0, 1.0

    # 95% Bootstrap Confidence Intervals (seed 42)
    rng = np.random.default_rng(42)
    boot_maes, boot_pearsons, boot_spearmans = [], [], []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        sample_p, sample_t = p_arr[idx], t_arr[idx]
        boot_maes.append(float(np.mean(np.abs(sample_p - sample_t))))

        if np.std(sample_p) > 1e-5 and np.std(sample_t) > 1e-5:
            r_val, _ = pearsonr(sample_p, sample_t)
            rho_val, _ = spearmanr(sample_p, sample_t)
            boot_pearsons.append(float(r_val))
            boot_spearmans.append(float(rho_val))

    mae_ci = [round(float(np.percentile(boot_maes, 2.5)), 2), round(float(np.percentile(boot_maes, 97.5)), 2)]
    p_ci = (
        [round(float(np.percentile(boot_pearsons, 2.5)), 3), round(float(np.percentile(boot_pearsons, 97.5)), 3)]
        if boot_pearsons else [round(p_corr, 3), round(p_corr, 3)]
    )
    s_ci = (
        [round(float(np.percentile(boot_spearmans, 2.5)), 3), round(float(np.percentile(boot_spearmans, 97.5)), 3)]
        if boot_spearmans else [round(s_corr, 3), round(s_corr, 3)]
    )

    return {
        "mae": round(mae, 3),
        "mae_ci_95": mae_ci,
        "pearson_r": round(float(p_corr), 4),
        "pearson_ci_95": p_ci,
        "pearson_p_value": round(float(p_pval), 5),
        "spearman_rho": round(float(s_corr), 4),
        "spearman_ci_95": s_ci,
        "spearman_p_value": round(float(s_pval), 5),
    }


def plot_predicted_vs_rated(
    oof_preds: Dict[str, List[float]],
    targets: List[float],
    output_png_path: Path,
):
    """Generate scatter plot of predicted scores vs survey target ratings with reference diagonal."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plotting] Matplotlib not installed; skipping graphic export.")
        return

    plt.figure(figsize=(12, 5))

    # Panel 1: Scatter plot Predicted vs Rated
    plt.subplot(1, 2, 1)
    models_config = [
        ("weighted_sum", "Weighted Sum (Baseline A)", "blue", "o"),
        ("mamdani_baseline", "Mamdani Fuzzy (Baseline B)", "orange", "s"),
        ("ga_tuned_fuzzy", "GA-Tuned Fuzzy (Proposed)", "green", "^"),
    ]

    for model_key, label, color, marker in models_config:
        preds = oof_preds[model_key]
        plt.scatter(targets, preds, label=label, color=color, marker=marker, alpha=0.8, s=60)

    # Diagonal 1:1 reference line
    plt.plot([0, 100], [0, 100], "k--", alpha=0.6, label="Ideal (y = x)")
    plt.xlim(0, 105)
    plt.ylim(0, 105)
    plt.xlabel("Survey Human Safety Target (0-100 Scale)", fontsize=10)
    plt.ylabel("Predicted Safety Score (0-100)", fontsize=10)
    plt.title("Cross-Validated Predicted vs Rated Safety Scores", fontsize=11, fontweight="bold")
    plt.legend(loc="upper left", fontsize=8)
    plt.grid(True, linestyle=":", alpha=0.6)

    # Panel 2: MAE Comparison Bar Chart
    plt.subplot(1, 2, 2)
    names = ["Weighted Sum", "Mamdani", "GA-Tuned"]
    maes = [
        compute_metrics(oof_preds["weighted_sum"], targets)["mae"],
        compute_metrics(oof_preds["mamdani_baseline"], targets)["mae"],
        compute_metrics(oof_preds["ga_tuned_fuzzy"], targets)["mae"],
    ]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = plt.bar(names, maes, color=colors, width=0.55, edgecolor="black")

    for bar, val in zip(bars, maes):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")

    plt.ylabel("Mean Absolute Error (Lower is Better)", fontsize=10)
    plt.title("5-Fold Cross-Validation MAE by Model", fontsize=11, fontweight="bold")
    plt.ylim(0, max(maes) * 1.25)
    plt.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    output_png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png_path, dpi=200)
    plt.close()
    print(f"[Plotting] Figure saved successfully: {output_png_path}")


def run_comparative_evaluation(
    features_csv: Path,
    ratings_csv: Path,
    results_json: Path,
    plot_png: Path = None,
    n_splits: int = 5,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Perform k-fold cross-validation comparing Weighted-Sum, Mamdani, and GA-Tuned models."""
    np.random.seed(random_seed)

    segments, targets = load_dataset(features_csv, ratings_csv)
    if len(segments) < n_splits:
        n_splits = max(2, len(segments))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    oof_preds = {
        "weighted_sum": [],
        "mamdani_baseline": [],
        "ga_tuned_fuzzy": [],
    }
    all_targets = []

    seg_indices = list(range(len(segments)))
    fold_details = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(seg_indices)):
        train_segs = [segments[i] for i in train_idx]
        test_segs = [segments[i] for i in test_idx]

        # 1. Baseline A: Fixed Weighted Sum
        ws_engine = WeightedSumBaselineEngine()
        ws_preds = [ws_engine.score_segment(s)["safety_score"] for s in test_segs]

        # 2. Baseline B: Hand-Written Mamdani Fuzzy System
        mam_engine = MamdaniFuzzyEngine()
        mam_preds = [mam_engine.score_segment(s)["safety_score"] for s in test_segs]

        # 3. Proposed: GA-Tuned Fuzzy System (trained strictly on train split)
        tuner = FuzzyGeneticTuner(train_segs, targets, random_seed=random_seed + fold_idx)
        opt_res = tuner.run_optimization(
            population_size=35,
            generations=20,
            train_segments=train_segs,
        )
        tuned_engine = opt_res["tuned_engine"]
        tuned_preds = [tuned_engine.score_segment(s)["safety_score"] for s in test_segs]

        test_targets = [targets[s["segment_id"]] for s in test_segs]

        oof_preds["weighted_sum"].extend(ws_preds)
        oof_preds["mamdani_baseline"].extend(mam_preds)
        oof_preds["ga_tuned_fuzzy"].extend(tuned_preds)
        all_targets.extend(test_targets)

        fold_details.append({
            "fold": fold_idx + 1,
            "test_size": len(test_segs),
            "weighted_sum_mae": compute_metrics(ws_preds, test_targets)["mae"],
            "mamdani_mae": compute_metrics(mam_preds, test_targets)["mae"],
            "ga_tuned_mae": compute_metrics(tuned_preds, test_targets)["mae"],
        })

    # Overall cross-validated metrics with confidence intervals
    metrics_summary = {
        "weighted_sum_baseline": compute_metrics(oof_preds["weighted_sum"], all_targets),
        "mamdani_fuzzy_baseline": compute_metrics(oof_preds["mamdani_baseline"], all_targets),
        "ga_tuned_fuzzy_proposed": compute_metrics(oof_preds["ga_tuned_fuzzy"], all_targets),
    }

    base_mae = metrics_summary["weighted_sum_baseline"]["mae"]
    tuned_mae = metrics_summary["ga_tuned_fuzzy_proposed"]["mae"]
    mae_reduction_pct = round(((base_mae - tuned_mae) / base_mae) * 100.0, 2) if base_mae > 0 else 0.0

    experiment_report = {
        "experiment_name": "fuzzy_safety_scoring_comparison",
        "description": "5-Fold Cross-Validation benchmark comparing linear weighted sum, hand-crafted Mamdani fuzzy rules, and GA-tuned fuzzy model",
        "random_seed": random_seed,
        "n_splits": n_splits,
        "total_evaluated_segments": len(all_targets),
        "results": metrics_summary,
        "mae_reduction_vs_baseline_pct": mae_reduction_pct,
        "fold_breakdown": fold_details,
        "overfitting_risk_analysis": {
            "vulnerability_with_small_sample_size": (
                "With sample sizes of N=8 to 30 road segments, unconstrained optimization algorithms "
                "(such as full gradient backpropagation on unconstrained neuro-fuzzy parameters) would rapidly "
                "overfit to idiosyncrasies of specific raters or single road stretches."
            ),
            "regularization_and_architectural_safeguards": [
                "Constrained parameter space: We tune only 20 rule weights (bounded in [0.1, 1.8]) and 4 macro membership boundaries.",
                "K-Fold Cross-Validation: Folds strictly partition training and testing segments; test segments are never seen during GA tuning.",
                "Bounded Gene Drift: Membership parameters use hard upper/lower bounds preventing degenerate or inverted fuzzy membership sets.",
            ],
            "viva_defense_summary": (
                "The GA-tuned fuzzy model achieves generalization by refining rule priorities rather than memorizing data points."
            ),
        },
    }

    # Write results JSON
    results_json.parent.mkdir(parents=True, exist_ok=True)
    with open(results_json, mode="w", encoding="utf-8") as f:
        json.dump(experiment_report, f, indent=2)

    # Plot graphical artifact
    if plot_png:
        plot_predicted_vs_rated(oof_preds, all_targets, plot_png)

    return experiment_report


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    
    ext_ratings = repo_root / "data" / "sample" / "survey_ratings_extended.csv"
    base_ratings = repo_root / "data" / "sample" / "survey_ratings.csv"
    ratings_path = ext_ratings if ext_ratings.exists() else base_ratings
    
    features_path = repo_root / "data" / "sample" / "segment_features.csv"
    results_path = repo_root / "evaluation" / "results" / "fuzzy_comparison.json"
    plot_path = repo_root / "evaluation" / "results" / "fuzzy_predicted_vs_rated.png"

    print(f"[Evaluation] Running comparative fuzzy benchmark with plotting...")
    report = run_comparative_evaluation(features_path, ratings_path, results_path, plot_path)
    print(f"[Evaluation] Finished. Summary:")
    print(json.dumps(report["results"], indent=2))
    print(f"[Evaluation] Results saved to: {results_path}")
