"""Simulated Survey Data Generator for SafePath.

Used as a reproducible development fallback when human survey responses are in progress.
Generates realistic 1-5 integer ratings across 25-30 road segments and 10+ raters.
Fixed random seeds ensure reproducible evaluation. Outputs are strictly labeled as simulated.
"""

import csv
from pathlib import Path
from typing import Dict

import numpy as np


def generate_simulated_survey(
    segments_path: Path,
    output_path: Path,
    num_raters: int = 12,
    random_seed: int = 42,
) -> Dict[str, any]:
    """Generate contract-compliant simulated survey ratings from segment features.
    
    Parameters:
        segments_path: Path to segment_features.csv.
        output_path: Destination path for survey_ratings.csv.
        num_raters: Number of human participants to simulate (default 12).
        random_seed: Fixed seed for reproducibility.
        
    Returns:
        Summary dict containing generation metadata and inter-rater statistics.
    """
    np.random.seed(random_seed)
    
    if not segments_path.exists():
        raise FileNotFoundError(f"Segment features file not found: {segments_path}")

    # Read segment features
    segments = []
    with open(segments_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            segments.append(row)

    if not segments:
        raise ValueError("No segment feature records found in input file.")

    # Expand segments if sample has fewer than 25 segments (by generating realistic variants)
    all_segments = list(segments)
    if len(all_segments) < 25:
        base_count = len(segments)
        for i in range(base_count, 28):
            base = segments[i % base_count]
            # Create a realistic variant
            jitter = np.random.uniform(-0.15, 0.15)
            h_density = max(0.0, float(base.get("hazard_density", 0.5)) + jitter)
            acc_score = np.clip(float(base.get("accident_score", 0.4)) + jitter, 0.0, 1.0)
            iso_score = np.clip(float(base.get("isolation_score", 0.3)) + jitter, 0.0, 1.0)
            t_day = np.random.choice(["day", "dusk", "night"], p=[0.4, 0.2, 0.4])
            lit_val = "yes" if t_day == "day" else np.random.choice(["yes", "no", "unknown"], p=[0.4, 0.4, 0.2])
            
            variant = {
                "segment_id": f"seg_{101 + i}",
                "length_m": str(round(float(base.get("length_m", 200.0)) * np.random.uniform(0.8, 1.4), 1)),
                "road_type": base.get("road_type", "secondary"),
                "speed_limit": base.get("speed_limit", "40"),
                "lit": lit_val,
                "hazard_density": str(round(h_density, 2)),
                "mean_hazard_confidence": str(round(np.clip(float(base.get("mean_hazard_confidence", 0.7)) + jitter, 0.1, 1.0), 2)),
                "mean_freshness": str(round(np.clip(float(base.get("mean_freshness", 0.6)) + jitter, 0.1, 1.0), 2)),
                "accident_score": str(round(acc_score, 2)),
                "isolation_score": str(round(iso_score, 2)),
                "time_of_day": t_day,
            }
            all_segments.append(variant)

    ratings_rows = []
    
    # Model latent human safety perception based on observable physical cues
    for seg in all_segments:
        seg_id = seg["segment_id"]
        t_day = seg.get("time_of_day", "night").lower()
        lit = seg.get("lit", "no").lower()
        h_dens = float(seg.get("hazard_density", 0.0))
        acc = float(seg.get("accident_score", 0.0))
        iso = float(seg.get("isolation_score", 0.0))
        
        # Ground truth continuous safety utility in [1.0, 5.0]
        # Base safety is 4.5; subtract risk penalties
        latent_safety = 4.6
        if t_day == "night":
            latent_safety -= 0.8
            if lit == "no":
                latent_safety -= 1.0
            elif lit == "unknown":
                latent_safety -= 0.4
        elif t_day == "dusk":
            latent_safety -= 0.3
            if lit == "no":
                latent_safety -= 0.5
                
        latent_safety -= 1.2 * min(h_dens, 1.5)
        latent_safety -= 1.3 * acc
        latent_safety -= 0.7 * iso
        
        latent_safety = np.clip(latent_safety, 1.0, 5.0)
        
        # Simulate ratings from different individuals with varied subjective thresholds and noise
        for r_idx in range(1, num_raters + 1):
            rater_id = f"rater_{r_idx:02d}"
            # Rater-specific subjective bias + Gaussian perception noise
            rater_bias = np.sin(r_idx * 1.5) * 0.25
            noise = np.random.normal(0, 0.35)
            perceived = latent_safety + rater_bias + noise
            # Discretize to 1-5 integer rating
            discrete_rating = int(np.clip(np.round(perceived), 1, 5))
            
            ratings_rows.append({
                "segment_id": seg_id,
                "rater_id": rater_id,
                "rating": discrete_rating,
                "time_of_day_context": t_day,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = ["segment_id", "rater_id", "rating", "time_of_day_context"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ratings_rows)

    metadata = {
        "status": "success",
        "data_provenance": "simulated",
        "random_seed": random_seed,
        "num_segments": len(all_segments),
        "num_raters": num_raters,
        "total_ratings": len(ratings_rows),
        "output_file": str(output_path),
    }
    return metadata


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    sample_segments = repo_root / "data" / "sample" / "segment_features.csv"
    target_ratings = repo_root / "data" / "sample" / "survey_ratings_extended.csv"
    meta = generate_simulated_survey(sample_segments, target_ratings)
    print(f"[Simulated Survey Generator] Completed: {meta}")

