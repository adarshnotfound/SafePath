"""Configuration and parameter definitions for SafePath Fuzzy Safety Engine.

Defines input/output fuzzy variables, membership function parameters (triangular and trapezoidal),
rule base (Mamdani IF-THEN rules), feature normalization bounds, and mode weights.
"""

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class FuzzySetParams:
    """Parameters for triangular or trapezoidal fuzzy membership functions.
    
    If params has 3 points (a, b, c), it defines a triangular function.
    If params has 4 points (a, b, c, d), it defines a trapezoidal function.
    """
    name: str
    shape: str  # 'trimf' or 'trapmf'
    points: List[float]


# Default linguistic input variables and membership definitions (triangular / trapezoidal)
DEFAULT_MEMBERSHIP_CONFIG: Dict[str, Dict[str, List[float]]] = {
    # Hazard density: confidence-weighted hazards per 100m [0.0 to 3.0+]
    "hazard_density": {
        "low": [0.0, 0.0, 0.25, 0.55],        # trapmf
        "medium": [0.25, 0.65, 1.20],          # trimf
        "high": [0.80, 1.30, 10.0, 10.0],      # trapmf
    },
    # Mean hazard confidence: [0.0 to 1.0]
    "mean_hazard_confidence": {
        "low": [0.0, 0.0, 0.30, 0.50],        # trapmf
        "medium": [0.35, 0.65, 0.85],          # trimf
        "high": [0.70, 0.88, 1.0, 1.0],        # trapmf
    },
    # Mean hazard freshness: [0.0 to 1.0] (0 = stale, 1 = fresh)
    "mean_freshness": {
        "stale": [0.0, 0.0, 0.25, 0.45],      # trapmf
        "aging": [0.30, 0.60, 0.80],           # trimf
        "fresh": [0.65, 0.85, 1.0, 1.0],       # trapmf
    },
    # Accident score: [0.0 to 1.0]
    "accident_score": {
        "low": [0.0, 0.0, 0.20, 0.40],        # trapmf
        "medium": [0.25, 0.50, 0.75],          # trimf
        "high": [0.60, 0.80, 1.0, 1.0],        # trapmf
    },
    # Lighting condition: [0.0 to 1.0] (0 = dark, 0.5 = dim, 1.0 = well_lit)
    "lighting": {
        "dark": [0.0, 0.0, 0.25, 0.45],       # trapmf
        "dim": [0.25, 0.50, 0.75],            # trimf
        "well_lit": [0.55, 0.80, 1.0, 1.0],    # trapmf
    },
    # Time of day: [0.0 to 1.0] (0 = night, 0.5 = dusk, 1.0 = day)
    "time_of_day": {
        "night": [0.0, 0.0, 0.25, 0.45],      # trapmf
        "dusk": [0.25, 0.50, 0.75],           # trimf
        "day": [0.55, 0.80, 1.0, 1.0],        # trapmf
    },
    # Isolation score: [0.0 to 1.0] (distance to emergency amenities)
    "isolation_score": {
        "low": [0.0, 0.0, 0.20, 0.40],        # trapmf
        "medium": [0.25, 0.50, 0.75],          # trimf
        "high": [0.55, 0.75, 1.0, 1.0],        # trapmf
    },
    # Speed limit (km/h): [0 to 100+]
    "speed_limit": {
        "slow": [0.0, 0.0, 25.0, 35.0],       # trapmf
        "moderate": [25.0, 45.0, 65.0],        # trimf
        "fast": [50.0, 70.0, 120.0, 120.0],   # trapmf
    },
    # Output: Safety Score [0 to 100]
    "safety_score": {
        "risky": [0.0, 0.0, 25.0, 45.0],      # trapmf
        "caution": [35.0, 52.5, 70.0],         # trimf
        "safe": [60.0, 80.0, 100.0, 100.0],    # trapmf
    },
}


# Safety Level Classification Thresholds
SAFETY_LEVEL_THRESHOLDS = {
    "safe_min": 70.0,       # score >= 70.0 -> safe
    "caution_min": 40.0,    # 40.0 <= score < 70.0 -> caution
    "risky_max": 40.0,      # score < 40.0 -> risky
}


# Hand-written Mamdani IF-THEN Rules (22 readable expert rules, within 15-25 rule target)
# Format: {
#   "id": rule_id,
#   "antecedents": {var_name: linguistic_term, ...},
#   "consequent": linguistic_term_for_safety_score,
#   "weight": rule_weight (default 1.0),
#   "reason": readable rationale snippet
# }
DEFAULT_MAMDANI_RULES: List[Dict[str, Any]] = [
    {
        "id": "R1_night_unlit_isolated",
        "antecedents": {"time_of_day": "night", "lighting": "dark", "isolation_score": "high"},
        "consequent": "risky",
        "weight": 1.0,
        "reason": "unlit street with high isolation at night",
    },
    {
        "id": "R2_high_fresh_hazards",
        "antecedents": {"hazard_density": "high", "mean_freshness": "fresh"},
        "consequent": "risky",
        "weight": 1.0,
        "reason": "high density of newly reported road hazards",
    },
    {
        "id": "R3_accident_hotspot",
        "antecedents": {"accident_score": "high"},
        "consequent": "risky",
        "weight": 0.95,
        "reason": "high accident hotspot proximity or severe crash history",
    },
    {
        "id": "R4_day_clear_low_hazard",
        "antecedents": {"time_of_day": "day", "lighting": "well_lit", "hazard_density": "low"},
        "consequent": "safe",
        "weight": 1.0,
        "reason": "clear daylight visibility with minimal road hazards",
    },
    {
        "id": "R5_moderate_hazard_cautious",
        "antecedents": {"hazard_density": "medium", "accident_score": "medium"},
        "consequent": "caution",
        "weight": 0.85,
        "reason": "moderate road hazards and accident exposure",
    },
    {
        "id": "R6_night_well_lit_connected",
        "antecedents": {"time_of_day": "night", "lighting": "well_lit", "isolation_score": "low"},
        "consequent": "caution",
        "weight": 0.8,
        "reason": "night driving but mitigated by streetlights and nearby amenities",
    },
    {
        "id": "R7_high_hazard_confident",
        "antecedents": {"hazard_density": "high", "mean_hazard_confidence": "high"},
        "consequent": "risky",
        "weight": 0.95,
        "reason": "frequent verified physical hazards on road surface",
    },
    {
        "id": "R8_dusk_dim_isolated",
        "antecedents": {"time_of_day": "dusk", "lighting": "dim", "isolation_score": "high"},
        "consequent": "caution",
        "weight": 0.85,
        "reason": "dim twilight lighting in isolated area",
    },
    {
        "id": "R9_day_safe_no_accidents",
        "antecedents": {"time_of_day": "day", "accident_score": "low", "hazard_density": "low"},
        "consequent": "safe",
        "weight": 0.95,
        "reason": "daytime travel along historically safe, hazard-free segment",
    },
    {
        "id": "R10_isolated_accident_prone",
        "antecedents": {"isolation_score": "high", "accident_score": "high"},
        "consequent": "risky",
        "weight": 0.95,
        "reason": "accident-prone corridor with distant emergency support",
    },
    {
        "id": "R11_fast_road_high_hazard",
        "antecedents": {"speed_limit": "fast", "hazard_density": "high"},
        "consequent": "risky",
        "weight": 0.9,
        "reason": "high speed limit combined with dense road damage",
    },
    {
        "id": "R12_slow_road_low_hazard",
        "antecedents": {"speed_limit": "slow", "hazard_density": "low", "accident_score": "low"},
        "consequent": "safe",
        "weight": 0.85,
        "reason": "calm slow-speed street with zero reported hazards",
    },
    {
        "id": "R13_night_dark_moderate_hazard",
        "antecedents": {"time_of_day": "night", "lighting": "dark", "hazard_density": "medium"},
        "consequent": "risky",
        "weight": 0.9,
        "reason": "unlit night driving encountering active potholes or speed breakers",
    },
    {
        "id": "R14_dusk_residential_cautious",
        "antecedents": {"time_of_day": "dusk", "speed_limit": "slow", "lighting": "dim"},
        "consequent": "caution",
        "weight": 0.8,
        "reason": "twilight visibility on residential street",
    },
    {
        "id": "R15_fast_road_at_night",
        "antecedents": {"time_of_day": "night", "speed_limit": "fast", "lighting": "dark"},
        "consequent": "risky",
        "weight": 0.95,
        "reason": "high-speed unlit driving risk",
    },
    {
        "id": "R16_well_lit_urban_medium_hazards",
        "antecedents": {"lighting": "well_lit", "isolation_score": "low", "hazard_density": "medium"},
        "consequent": "caution",
        "weight": 0.75,
        "reason": "urban lit road with surface potholes requiring caution",
    },
    {
        "id": "R17_stale_hazards_low_accident",
        "antecedents": {"mean_freshness": "stale", "accident_score": "low", "time_of_day": "day"},
        "consequent": "safe",
        "weight": 0.7,
        "reason": "unverified older hazard reports on safe daytime road",
    },
    {
        "id": "R18_high_isolation_moderate_hazards",
        "antecedents": {"isolation_score": "high", "hazard_density": "medium"},
        "consequent": "caution",
        "weight": 0.8,
        "reason": "isolated route with noticeable road damage",
    },
    {
        "id": "R19_night_accident_hotspot",
        "antecedents": {"time_of_day": "night", "accident_score": "high"},
        "consequent": "risky",
        "weight": 1.0,
        "reason": "passing accident blackspot during night hours",
    },
    {
        "id": "R20_clear_dusk_low_hazard",
        "antecedents": {"time_of_day": "dusk", "lighting": "well_lit", "accident_score": "low"},
        "consequent": "safe",
        "weight": 0.85,
        "reason": "well lit dusk route with pristine accident record",
    },
    {
        "id": "R21_night_dark_high_hazards",
        "antecedents": {"time_of_day": "night", "lighting": "dark", "hazard_density": "high"},
        "consequent": "risky",
        "weight": 1.0,
        "reason": "unlit street with heavy potholes or speed breakers at night",
    },
    {
        "id": "R22_dusk_dark_isolated",
        "antecedents": {"time_of_day": "dusk", "lighting": "dark", "isolation_score": "high"},
        "consequent": "caution",
        "weight": 0.85,
        "reason": "unlit isolated road during twilight",
    },
]


# Baseline Weighted-Sum Weights (Section 4.6 & Section 4.8)
DEFAULT_FEATURE_WEIGHTS: Dict[str, float] = {
    "hazard_density": 0.22,
    "mean_hazard_confidence": 0.08,
    "mean_freshness": 0.08,
    "accident_score": 0.22,
    "lighting": 0.16,
    "time_of_day": 0.10,
    "isolation_score": 0.10,
    "speed_penalty": 0.04,
}

# Personal Safety Mode Weights (Section 4.8)
PERSONAL_SAFETY_FEATURE_WEIGHTS: Dict[str, float] = {
    "hazard_density": 0.12,
    "mean_hazard_confidence": 0.04,
    "mean_freshness": 0.04,
    "accident_score": 0.15,
    "lighting": 0.28,
    "time_of_day": 0.15,
    "isolation_score": 0.20,
    "speed_penalty": 0.02,
}
