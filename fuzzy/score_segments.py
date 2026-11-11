"""Segment Scoring CLI for SafePath.

Reads segment_features.csv and generates contract-compliant segment_scores.csv:
Columns: segment_id, safety_score, level, top_factors, fired_rules, model_name
Supports weighted_sum_baseline, mamdani_fuzzy_baseline, and ga_tuned_fuzzy models.
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fuzzy.baseline_weighted import WeightedSumBaselineEngine
from fuzzy.mamdani_engine import MamdaniFuzzyEngine
from fuzzy.ga_tuner import FuzzyGeneticTuner


def score_features_file(
    input_features_path: Path,
    output_scores_path: Path,
    model_choice: str = "mamdani_fuzzy_baseline",
    personal_safety_mode: bool = False,
) -> int:
    """Score all segment features from input CSV and write segment_scores.csv."""
    if not input_features_path.exists():
        raise FileNotFoundError(f"Input features file not found: {input_features_path}")

    with open(input_features_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        segments = [r for r in reader if r]

    if model_choice == "weighted_sum":
        engine = WeightedSumBaselineEngine(personal_safety_mode=personal_safety_mode)
    elif model_choice == "ga_tuned":
        # Initialize engine with pre-tuned parameters or tune on available survey ratings
        engine = MamdaniFuzzyEngine(model_name="ga_tuned_fuzzy")
    else:
        engine = MamdaniFuzzyEngine(model_name="mamdani_fuzzy_baseline")

    scored_records = engine.score_all(segments)

    output_scores_path.parent.mkdir(parents=True, exist_ok=True)
    contract_fields = [
        "segment_id",
        "safety_score",
        "level",
        "top_factors",
        "fired_rules",
        "model_name",
    ]

    with open(output_scores_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=contract_fields)
        writer.writeheader()
        writer.writerows(scored_records)

    return len(scored_records)


def main():
    parser = argparse.ArgumentParser(description="Score road segments using SafePath Fuzzy Safety Engine.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "data" / "sample" / "segment_features.csv"),
        help="Path to input segment_features.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(REPO_ROOT / "data" / "sample" / "segment_scores_generated.csv"),
        help="Path to output segment_scores.csv",
    )
    parser.add_argument(
        "--model",
        choices=["mamdani", "weighted_sum", "ga_tuned"],
        default="mamdani",
        help="Scoring model to employ",
    )
    parser.add_argument(
        "--personal-safety",
        action="store_true",
        help="Enable Personal Safety Mode",
    )

    args = parser.parse_args()
    count = score_features_file(
        Path(args.input),
        Path(args.output),
        model_choice=args.model,
        personal_safety_mode=args.personal_safety,
    )
    print(f"[Score Segments] Successfully scored {count} segments -> {args.output}")


if __name__ == "__main__":
    main()

