"""Data cleaning, contract enforcement, and inter-rater agreement analysis for SafePath.

Ingests raw survey submissions (both standardized schema and direct Google Forms exports),
standardizes columns to Section 8 contracts, removes invalid entries, and calculates
statistical inter-rater agreement.
"""

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def compute_inter_rater_agreement(ratings_by_segment: Dict[str, Dict[str, int]]) -> Dict[str, float]:
    """Calculate inter-rater agreement metrics across raters for all road segments.
    
    Computes:
      1. Mean Pairwise Absolute Difference (lower is better, 0 = perfect agreement)
      2. Mean Standard Deviation across ratings per segment
      3. Percent Exact Agreement (percentage of pairs giving the exact same score)
      4. Intraclass Correlation Coefficient proxy (variance between segments vs within segments)
    """
    segment_ids = list(ratings_by_segment.keys())
    if not segment_ids:
        return {"num_segments": 0, "mean_std": 0.0, "mean_abs_diff": 0.0, "percent_exact": 0.0, "icc_approx": 0.0}

    segment_means = []
    segment_stds = []
    pairwise_diffs = []
    exact_matches = 0
    total_pairs = 0

    for seg_id, rater_dict in ratings_by_segment.items():
        vals = list(rater_dict.values())
        if len(vals) < 2:
            continue
        segment_means.append(np.mean(vals))
        segment_stds.append(np.std(vals, ddof=1))

        # Pairwise comparison
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                diff = abs(vals[i] - vals[j])
                pairwise_diffs.append(diff)
                if diff == 0:
                    exact_matches += 1
                total_pairs += 1

    mean_std = float(np.mean(segment_stds)) if segment_stds else 0.0
    mean_abs_diff = float(np.mean(pairwise_diffs)) if pairwise_diffs else 0.0
    pct_exact = float(exact_matches / total_pairs * 100.0) if total_pairs > 0 else 0.0
    
    # Intraclass Correlation Coefficient ICC(1, 1) approximation
    between_var = float(np.var(segment_means, ddof=1)) if len(segment_means) > 1 else 0.0
    within_var = float(np.mean([s**2 for s in segment_stds])) if segment_stds else 0.0
    k_approx = float(np.mean([len(r) for r in ratings_by_segment.values()]))
    if (between_var + within_var) > 0 and k_approx > 1:
        icc = (between_var - within_var / k_approx) / (between_var + (k_approx - 1) * within_var / k_approx)
    else:
        icc = 0.0
    icc = float(np.clip(icc, 0.0, 1.0))

    return {
        "num_segments": len(segment_ids),
        "mean_std_per_segment": round(mean_std, 3),
        "mean_pairwise_abs_diff": round(mean_abs_diff, 3),
        "percent_exact_agreement": round(pct_exact, 1),
        "icc_approx": round(icc, 3),
    }


def parse_google_form_csv(
    input_file: Path,
    segment_contexts: Dict[str, str] = None,
) -> Tuple[List[Dict[str, any]], int]:
    """Parse a wide-format Google Form CSV into standardized SafePath records."""
    cleaned_rows = []
    skipped_count = 0
    contexts = segment_contexts or {}

    with open(input_file, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Identify which columns correspond to segment ratings
        # Matches patterns like "seg_101", "Segment 101", or "seg_101 [Rating]"
        seg_col_map = {}
        for col_idx, col_name in enumerate(header):
            match = re.search(r"(seg_\d+|segment[_\s-]?\d+)", col_name, re.IGNORECASE)
            if match:
                clean_seg_id = match.group(1).lower().replace("segment", "seg_").replace(" ", "").replace("-", "_")
                if "seg__" in clean_seg_id:
                    clean_seg_id = clean_seg_id.replace("seg__", "seg_")
                seg_col_map[col_idx] = clean_seg_id

        if not seg_col_map:
            return [], 0

        # Process each response row
        for row_idx, row in enumerate(reader):
            rater_id = f"rater_{row_idx + 1:02d}"
            for col_idx, seg_id in seg_col_map.items():
                if col_idx >= len(row):
                    skipped_count += 1
                    continue
                val_str = row[col_idx].strip()
                if not val_str:
                    skipped_count += 1
                    continue
                try:
                    rating = int(round(float(val_str)))
                    if rating < 1 or rating > 5:
                        skipped_count += 1
                        continue
                except ValueError:
                    skipped_count += 1
                    continue

                time_ctx = contexts.get(seg_id, "night" if "night" in seg_id else "day")

                cleaned_rows.append({
                    "segment_id": seg_id,
                    "rater_id": rater_id,
                    "rating": rating,
                    "time_of_day_context": time_ctx,
                })

    return cleaned_rows, skipped_count


def clean_survey_responses(
    input_file: Path,
    output_file: Path,
    report_file: Path = None,
    segment_contexts: Dict[str, str] = None,
) -> Dict[str, any]:
    """Clean survey responses, enforce Section 8 contracts, and export cleaned CSV.
    
    Expected contract columns:
      segment_id, rater_id, rating, time_of_day_context
    """
    if not input_file.exists():
        raise FileNotFoundError(f"Input survey file not found: {input_file}")

    cleaned_rows = []
    ratings_by_segment: Dict[str, Dict[str, int]] = {}
    skipped_count = 0

    # First attempt reading as standard contract format
    with open(input_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        is_standard_contract = all(col in fields for col in ["segment_id", "rater_id", "rating"])

    if is_standard_contract:
        with open(input_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                seg_id = row.get("segment_id", "").strip()
                rater_id = row.get("rater_id", "").strip()
                rating_str = row.get("rating", "").strip()
                time_ctx = row.get("time_of_day_context", "").strip().lower()

                if not seg_id or not rater_id or not rating_str:
                    skipped_count += 1
                    continue

                try:
                    rating = int(round(float(rating_str)))
                    if rating < 1 or rating > 5:
                        skipped_count += 1
                        continue
                except ValueError:
                    skipped_count += 1
                    continue

                if time_ctx not in ("day", "dusk", "night"):
                    time_ctx = "day"

                cleaned_rows.append({
                    "segment_id": seg_id,
                    "rater_id": rater_id,
                    "rating": rating,
                    "time_of_day_context": time_ctx,
                })
    else:
        # Attempt parsing wide-format Google Form export
        cleaned_rows, skipped_count = parse_google_form_csv(input_file, segment_contexts)

    # Build ratings_by_segment dictionary for statistical agreement
    for r in cleaned_rows:
        seg_id = r["segment_id"]
        rater_id = r["rater_id"]
        if seg_id not in ratings_by_segment:
            ratings_by_segment[seg_id] = {}
        ratings_by_segment[seg_id][rater_id] = r["rating"]

    # Write cleaned survey_ratings.csv matching Section 8 contract
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["segment_id", "rater_id", "rating", "time_of_day_context"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    agreement_metrics = compute_inter_rater_agreement(ratings_by_segment)
    summary = {
        "status": "success",
        "total_rows_read": len(cleaned_rows) + skipped_count,
        "valid_ratings": len(cleaned_rows),
        "skipped_rows": skipped_count,
        "unique_segments": len(ratings_by_segment),
        "unique_raters": len({r["rater_id"] for r in cleaned_rows}),
        "inter_rater_agreement": agreement_metrics,
        "output_path": str(output_file),
    }

    if report_file:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        with open(report_file, mode="w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    sample_survey = repo_root / "data" / "sample" / "survey_ratings.csv"
    out_cleaned = repo_root / "survey" / "cleaned_survey_ratings.csv"
    report_path = repo_root / "evaluation" / "results" / "survey_agreement.json"
    result = clean_survey_responses(sample_survey, out_cleaned, report_path)
    print(f"[Survey Cleaner] Complete. Processed {result['valid_ratings']} ratings.")
    print(f"[Inter-Rater Agreement] {result['inter_rater_agreement']}")
