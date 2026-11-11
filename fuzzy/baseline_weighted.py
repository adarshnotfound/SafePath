"""Weighted-Sum Baseline Engine for SafePath Route Safety (Baseline A).

Calculates a linear weighted sum of normalized road segment features to produce
a 0-100 safety score, classification level, and plain-language explanation.
Strictly adheres to Section 8 contracts.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from fuzzy.config import (
    DEFAULT_FEATURE_WEIGHTS,
    PERSONAL_SAFETY_FEATURE_WEIGHTS,
    SAFETY_LEVEL_THRESHOLDS,
)


def normalize_features(features: Dict[str, Any]) -> Dict[str, float]:
    """Normalize raw segment features into calibrated [0.0, 1.0] scales.
    
    Higher normalized value represents higher safety / lower penalty.
    """
    # 1. Hazard density: raw is hazards per 100m. Range [0, 2.0+].
    # Low hazard is safer (1.0 = 0 hazards, 0.0 = >= 2.0 hazards/100m)
    h_density = float(features.get("hazard_density", 0.0))
    norm_hazard_density = float(np.clip(1.0 - (h_density / 2.0), 0.0, 1.0))

    # 2. Confidence & freshness: high confidence in hazards increases hazard penalty
    # So if hazards exist, high confidence/freshness lowers safety; if zero hazards, neutral 1.0
    conf = float(features.get("mean_hazard_confidence", 0.5))
    fresh = float(features.get("mean_freshness", 0.5))
    if h_density > 0.05:
        norm_conf = float(np.clip(1.0 - conf, 0.0, 1.0))
        norm_fresh = float(np.clip(1.0 - fresh, 0.0, 1.0))
    else:
        norm_conf = 1.0
        norm_fresh = 1.0

    # 3. Accident score: raw is [0.0, 1.0]. 0.0 = safe, 1.0 = severe blackspot
    acc = float(features.get("accident_score", 0.0))
    norm_accident = float(np.clip(1.0 - acc, 0.0, 1.0))

    # 4. Lighting: 'yes' -> 1.0, 'unknown' -> 0.5, 'no' -> 0.0
    lit_val = str(features.get("lit", "unknown")).strip().lower()
    if lit_val in ("yes", "well_lit", "true", "1"):
        norm_lighting = 1.0
    elif lit_val in ("unknown", "dim"):
        norm_lighting = 0.5
    else:
        norm_lighting = 0.0

    # 5. Time of day: 'day' -> 1.0, 'dusk' -> 0.5, 'night' -> 0.0
    tod_val = str(features.get("time_of_day", "day")).strip().lower()
    if tod_val in ("day", "daylight"):
        norm_time = 1.0
    elif tod_val == "dusk":
        norm_time = 0.5
    else:
        norm_time = 0.0

    # 6. Isolation score: raw [0.0, 1.0]. 0 = connected (safe), 1 = isolated
    iso = float(features.get("isolation_score", 0.0))
    norm_isolation = float(np.clip(1.0 - iso, 0.0, 1.0))

    # 7. Speed limit penalty: fast roads amplify crash severity risk
    try:
        speed = float(features.get("speed_limit", 40.0))
    except (ValueError, TypeError):
        speed = 40.0
    # 20-30 km/h is slow/safe; 80+ km/h is fast
    norm_speed = float(np.clip(1.0 - (speed / 100.0), 0.0, 1.0))

    return {
        "hazard_density": norm_hazard_density,
        "mean_hazard_confidence": norm_conf,
        "mean_freshness": norm_fresh,
        "accident_score": norm_accident,
        "lighting": norm_lighting,
        "time_of_day": norm_time,
        "isolation_score": norm_isolation,
        "speed_penalty": norm_speed,
    }


class WeightedSumBaselineEngine:
    """Computes segment safety scores using a fixed linear weighted sum."""

    def __init__(self, weights: Dict[str, float] = None, personal_safety_mode: bool = False):
        if weights is not None:
            self.weights = weights
        elif personal_safety_mode:
            self.weights = PERSONAL_SAFETY_FEATURE_WEIGHTS
        else:
            self.weights = DEFAULT_FEATURE_WEIGHTS

        self.personal_safety_mode = personal_safety_mode

        # Normalize weights so they sum to 1.0
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

    def score_segment(self, segment_features: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single segment dictionary according to Section 8 contracts.
        
        Returns:
            Dict matching segment_scores.csv:
            segment_id, safety_score, level, top_factors, fired_rules, model_name
        """
        seg_id = str(segment_features.get("segment_id", "unknown"))
        normalized = normalize_features(segment_features)

        # Compute weighted sum [0.0, 1.0] -> convert to [0.0, 100.0]
        raw_score = sum(self.weights[k] * normalized[k] for k in self.weights)
        safety_score = float(np.clip(raw_score * 100.0, 0.0, 100.0))

        # Classify level
        if safety_score >= SAFETY_LEVEL_THRESHOLDS["safe_min"]:
            level = "safe"
        elif safety_score >= SAFETY_LEVEL_THRESHOLDS["caution_min"]:
            level = "caution"
        else:
            level = "risky"

        # Determine top negative contributing factors (penalties)
        # Factor penalty = weight * (1.0 - normalized_value)
        factor_penalties = []
        descriptions = {
            "hazard_density": "high hazard density",
            "accident_score": "accident history / hotspot proximity",
            "lighting": "unlit street / poor lighting",
            "isolation_score": "high isolation from amenities",
            "time_of_day": "nighttime travel risk",
            "speed_penalty": "high speed roadway",
            "mean_hazard_confidence": "unverified road surface",
            "mean_freshness": "fresh hazard presence",
        }

        for factor, norm_val in normalized.items():
            penalty = self.weights.get(factor, 0.0) * (1.0 - norm_val)
            if penalty > 0.02 and factor in descriptions:
                factor_penalties.append((penalty, descriptions[factor]))

        factor_penalties.sort(key=lambda x: x[0], reverse=True)
        top_factors_list = [f[1] for f in factor_penalties[:3]]
        
        # If score is very safe, highlight positive attributes
        if not top_factors_list:
            top_factors_str = "clear road; good visibility; low accident history"
        else:
            top_factors_str = "; ".join(top_factors_list)

        model_name = "weighted_sum_personal_safety" if self.personal_safety_mode else "weighted_sum_baseline"

        return {
            "segment_id": seg_id,
            "safety_score": round(safety_score, 1),
            "level": level,
            "top_factors": top_factors_str,
            "fired_rules": "weighted_linear_combination",
            "model_name": model_name,
        }

    def score_all(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score a list of segment feature dictionaries."""
        return [self.score_segment(seg) for seg in segments]
