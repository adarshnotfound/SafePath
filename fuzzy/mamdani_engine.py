"""Mamdani Fuzzy Inference Engine for SafePath (Baseline B & Core Inference).

Implements:
1. Fuzzification of road segment features into linguistic terms.
2. Mamdani IF-THEN rule evaluation with min T-norm and configurable rule weights.
3. Max S-norm aggregation across output fuzzy sets.
4. Centroid defuzzification for 0-100 continuous Safety Score output.
5. Fired rule tracking and plain-language explanation generation.
Strictly adheres to Section 8 contracts.
"""

from typing import Dict, Any, List, Tuple, Optional
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from fuzzy.config import (
    DEFAULT_MEMBERSHIP_CONFIG,
    DEFAULT_MAMDANI_RULES,
    SAFETY_LEVEL_THRESHOLDS,
)
from fuzzy.membership import FuzzySystemVariables, trimf, trapmf


class MamdaniFuzzyEngine:
    """Mamdani Rule-Based Fuzzy Inference System for Road Segment Safety."""

    def __init__(
        self,
        membership_config: Optional[Dict[str, Dict[str, List[float]]]] = None,
        rules: Optional[List[Dict[str, Any]]] = None,
        rule_weights: Optional[List[float]] = None,
        model_name: str = "mamdani_fuzzy_baseline",
    ):
        self.membership_config = membership_config or DEFAULT_MEMBERSHIP_CONFIG
        self.variables = FuzzySystemVariables(self.membership_config)
        self.rules = [dict(r) for r in (rules or DEFAULT_MAMDANI_RULES)]
        self.model_name = model_name

        # If custom rule weights are provided, assign them
        if rule_weights is not None:
            for idx, w in enumerate(rule_weights):
                if idx < len(self.rules):
                    self.rules[idx]["weight"] = float(w)

        # Output universe for centroid defuzzification
        self.y_universe = np.linspace(0.0, 100.0, 201)

    def _prepare_numeric_features(self, raw_features: Dict[str, Any]) -> Dict[str, float]:
        """Convert raw tabular values into continuous variables compatible with fuzzy sets."""
        numeric = {}

        # 1. Hazard density (per 100m)
        numeric["hazard_density"] = float(raw_features.get("hazard_density", 0.0))

        # 2. Mean hazard confidence [0.0, 1.0]
        numeric["mean_hazard_confidence"] = float(raw_features.get("mean_hazard_confidence", 0.5))

        # 3. Mean freshness [0.0, 1.0]
        numeric["mean_freshness"] = float(raw_features.get("mean_freshness", 0.5))

        # 4. Accident score [0.0, 1.0]
        numeric["accident_score"] = float(raw_features.get("accident_score", 0.0))

        # 5. Lighting condition [0.0, 1.0]
        lit_val = str(raw_features.get("lit", "unknown")).strip().lower()
        if lit_val in ("yes", "well_lit", "true", "1"):
            numeric["lighting"] = 0.95
        elif lit_val in ("unknown", "dim"):
            numeric["lighting"] = 0.50
        else:
            numeric["lighting"] = 0.10

        # 6. Time of day [0.0, 1.0]
        tod_val = str(raw_features.get("time_of_day", "day")).strip().lower()
        if tod_val in ("day", "daylight"):
            numeric["time_of_day"] = 0.90
        elif tod_val == "dusk":
            numeric["time_of_day"] = 0.50
        else:
            numeric["time_of_day"] = 0.10

        # 7. Isolation score [0.0, 1.0]
        numeric["isolation_score"] = float(raw_features.get("isolation_score", 0.0))

        # 8. Speed limit (km/h)
        try:
            numeric["speed_limit"] = float(raw_features.get("speed_limit", 40.0))
        except (ValueError, TypeError):
            numeric["speed_limit"] = 40.0

        return numeric

    def evaluate_rules(self, numeric_features: Dict[str, float]) -> List[Tuple[Dict[str, Any], float]]:
        """Evaluate all fuzzy rules and return (rule, activation_strength) pairs."""
        fuzzified = self.variables.fuzzify_all(numeric_features)
        rule_firings = []

        for rule in self.rules:
            antecedents = rule.get("antecedents", {})
            weight = float(rule.get("weight", 1.0))
            if not antecedents:
                continue

            # Min T-norm across all antecedent conditions
            memberships = []
            for var_name, term in antecedents.items():
                if var_name in fuzzified and term in fuzzified[var_name]:
                    memberships.append(fuzzified[var_name][term])
                else:
                    memberships.append(0.0)

            antecedent_strength = min(memberships) if memberships else 0.0
            firing_strength = antecedent_strength * weight

            rule_firings.append((rule, float(firing_strength)))

        return rule_firings

    def defuzzify_centroid(self, rule_firings: List[Tuple[Dict[str, Any], float]]) -> float:
        """Aggregate output fuzzy sets with max S-norm and defuzzify using centroid method."""
        safety_var = self.variables.get_variable("safety_score")
        aggregated_mf = np.zeros_like(self.y_universe)

        for rule, strength in rule_firings:
            if strength <= 0.001:
                continue
            consequent_term = rule.get("consequent", "caution")
            params = safety_var.terms.get(consequent_term)
            if not params:
                continue

            # Evaluate membership of output universe
            if len(params) == 3:
                consequent_mf = trimf(self.y_universe, params)
            else:
                consequent_mf = trapmf(self.y_universe, params)

            # Mamdani implication (min clipping)
            clipped = np.minimum(consequent_mf, strength)
            # Mamdani aggregation (max combination)
            aggregated_mf = np.maximum(aggregated_mf, clipped)

        # Centroid calculation
        total_area = np.sum(aggregated_mf)
        if total_area > 1e-6:
            score = float(np.sum(self.y_universe * aggregated_mf) / total_area)
        else:
            # Neutral default if no rule fired significantly
            score = 50.0

        return float(np.clip(score, 0.0, 100.0))

    def score_segment(self, segment_features: Dict[str, Any]) -> Dict[str, Any]:
        """Score a single road segment and format output strictly to Section 8 contracts.
        
        Returns:
            Dict matching segment_scores.csv:
            segment_id, safety_score, level, top_factors, fired_rules, model_name
        """
        seg_id = str(segment_features.get("segment_id", "unknown"))
        numeric_features = self._prepare_numeric_features(segment_features)

        # Evaluate rules and defuzzify
        rule_firings = self.evaluate_rules(numeric_features)
        raw_score = self.defuzzify_centroid(rule_firings)
        safety_score = round(raw_score, 1)

        # Classify level
        if safety_score >= SAFETY_LEVEL_THRESHOLDS["safe_min"]:
            level = "safe"
        elif safety_score >= SAFETY_LEVEL_THRESHOLDS["caution_min"]:
            level = "caution"
        else:
            level = "risky"

        # Extract fired rules and explainability reasons
        # Filter rules with meaningful activation (> 0.05)
        active_rules = [rf for rf in rule_firings if rf[1] >= 0.05]
        active_rules.sort(key=lambda x: x[1], reverse=True)

        fired_rule_ids = [r[0]["id"] for r in active_rules]
        fired_rules_str = "; ".join(fired_rule_ids) if fired_rule_ids else "R_default_neutral"

        # Top contributing factors from strongest active rules
        top_reasons = []
        for r, strength in active_rules:
            reason = r.get("reason", "")
            if reason and reason not in top_reasons:
                top_reasons.append(reason)
            if len(top_reasons) >= 3:
                break

        if not top_reasons:
            top_factors_str = "moderate road conditions; normal baseline safety"
        else:
            top_factors_str = "; ".join(top_reasons)

        return {
            "segment_id": seg_id,
            "safety_score": safety_score,
            "level": level,
            "top_factors": top_factors_str,
            "fired_rules": fired_rules_str,
            "model_name": self.model_name,
        }

    def score_all(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score a list of road segments."""
        return [self.score_segment(seg) for seg in segments]
