"""Unit tests and sanity checks for SafePath Fuzzy Engine (Member C).

Includes 5 hand-checked test scenarios (Task C1), membership function verification,
contract compliance verification, and personal safety mode testing.
"""

import unittest
import numpy as np
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fuzzy.membership import trimf, trapmf, FuzzySystemVariables
from fuzzy.baseline_weighted import WeightedSumBaselineEngine
from fuzzy.mamdani_engine import MamdaniFuzzyEngine
from fuzzy.config import DEFAULT_MEMBERSHIP_CONFIG


class TestFuzzyEngine(unittest.TestCase):
    """Test suite covering fuzzy logic functions and 5 hand-checked test cases."""

    def setUp(self):
        self.mamdani_engine = MamdaniFuzzyEngine()
        self.weighted_engine = WeightedSumBaselineEngine()

    def test_trimf_boundary_conditions(self):
        """Test triangular membership function values at critical vertices."""
        params = [0.0, 50.0, 100.0]
        self.assertEqual(trimf(0.0, params), 0.0)
        self.assertEqual(trimf(50.0, params), 1.0)
        self.assertEqual(trimf(100.0, params), 0.0)
        self.assertEqual(trimf(25.0, params), 0.5)
        self.assertEqual(trimf(75.0, params), 0.5)
        self.assertEqual(trimf(-10.0, params), 0.0)
        self.assertEqual(trimf(110.0, params), 0.0)

    def test_trapmf_boundary_conditions(self):
        """Test trapezoidal membership function values at slopes and plateau."""
        params = [10.0, 30.0, 70.0, 90.0]
        self.assertEqual(trapmf(5.0, params), 0.0)
        self.assertEqual(trapmf(10.0, params), 0.0)
        self.assertEqual(trapmf(20.0, params), 0.5)
        self.assertEqual(trapmf(30.0, params), 1.0)
        self.assertEqual(trapmf(50.0, params), 1.0)
        self.assertEqual(trapmf(70.0, params), 1.0)
        self.assertEqual(trapmf(80.0, params), 0.5)
        self.assertEqual(trapmf(90.0, params), 0.0)
        self.assertEqual(trapmf(95.0, params), 0.0)

    # -------------------------------------------------------------
    # 5 Hand-Checked Validation Cases (Mandatory Task C1 Requirement)
    # -------------------------------------------------------------

    def test_case_1_ideal_clear_day_primary_road(self):
        """Case 1: Well-lit day primary road with zero hazards and pristine safety."""
        features = {
            "segment_id": "test_case_1",
            "length_m": "300.0",
            "road_type": "primary",
            "speed_limit": "40",
            "lit": "yes",
            "hazard_density": "0.0",
            "mean_hazard_confidence": "0.0",
            "mean_freshness": "0.0",
            "accident_score": "0.05",
            "isolation_score": "0.05",
            "time_of_day": "day",
        }
        res = self.mamdani_engine.score_segment(features)
        self.assertEqual(res["level"], "safe")
        self.assertGreaterEqual(res["safety_score"], 70.0)
        self.assertIn("R4_day_clear_low_hazard", res["fired_rules"])

    def test_case_2_unlit_night_isolated_hazardous_road(self):
        """Case 2: Pitch dark isolated night road with severe potholes and crash risk."""
        features = {
            "segment_id": "test_case_2",
            "length_m": "250.0",
            "road_type": "secondary",
            "speed_limit": "40",
            "lit": "no",
            "hazard_density": "1.6",
            "mean_hazard_confidence": "0.9",
            "mean_freshness": "0.9",
            "accident_score": "0.85",
            "isolation_score": "0.85",
            "time_of_day": "night",
        }
        res = self.mamdani_engine.score_segment(features)
        self.assertEqual(res["level"], "risky")
        self.assertLess(res["safety_score"], 40.0)
        self.assertIn("R1_night_unlit_isolated", res["fired_rules"])

    def test_case_3_severe_accident_blackspot(self):
        """Case 3: Segment intersecting a major accident hotspot at night."""
        features = {
            "segment_id": "test_case_3",
            "length_m": "200.0",
            "road_type": "primary",
            "speed_limit": "50",
            "lit": "yes",
            "hazard_density": "0.2",
            "mean_hazard_confidence": "0.5",
            "mean_freshness": "0.5",
            "accident_score": "0.95",
            "isolation_score": "0.2",
            "time_of_day": "night",
        }
        res = self.mamdani_engine.score_segment(features)
        self.assertIn(res["level"], ("risky", "caution"))
        self.assertLessEqual(res["safety_score"], 45.0)
        self.assertIn("accident", res["top_factors"].lower())

    def test_case_4_calm_residential_twilight_road(self):
        """Case 4: Low speed residential road at dusk with moderate isolation."""
        features = {
            "segment_id": "test_case_4",
            "length_m": "150.0",
            "road_type": "residential",
            "speed_limit": "25",
            "lit": "dim",
            "hazard_density": "0.2",
            "mean_hazard_confidence": "0.6",
            "mean_freshness": "0.5",
            "accident_score": "0.1",
            "isolation_score": "0.4",
            "time_of_day": "dusk",
        }
        res = self.mamdani_engine.score_segment(features)
        self.assertIn(res["level"], ("caution", "safe"))
        self.assertGreaterEqual(res["safety_score"], 45.0)

    def test_case_5_dense_fresh_hazards(self):
        """Case 5: High density of newly sighted potholes with high detector confidence."""
        features = {
            "segment_id": "test_case_5",
            "length_m": "180.0",
            "road_type": "secondary",
            "speed_limit": "40",
            "lit": "yes",
            "hazard_density": "1.8",
            "mean_hazard_confidence": "0.95",
            "mean_freshness": "0.95",
            "accident_score": "0.2",
            "isolation_score": "0.1",
            "time_of_day": "day",
        }
        res = self.mamdani_engine.score_segment(features)
        self.assertIn(res["level"], ("risky", "caution"))
        self.assertLess(res["safety_score"], 55.0)
        self.assertTrue("R2_high_fresh_hazards" in res["fired_rules"] or "R7_high_hazard_confident" in res["fired_rules"])

    # -------------------------------------------------------------
    # Output Contract Verification (Section 8)
    # -------------------------------------------------------------

    def test_section_8_output_contract(self):
        """Verify output dictionary contains exact contract columns."""
        sample_feature = {
            "segment_id": "seg_contract_test",
            "length_m": "240.0",
            "road_type": "secondary",
            "speed_limit": "40",
            "lit": "yes",
            "hazard_density": "0.5",
            "mean_hazard_confidence": "0.7",
            "mean_freshness": "0.7",
            "accident_score": "0.3",
            "isolation_score": "0.2",
            "time_of_day": "night",
        }
        res = self.mamdani_engine.score_segment(sample_feature)
        expected_keys = {"segment_id", "safety_score", "level", "top_factors", "fired_rules", "model_name"}
        self.assertEqual(set(res.keys()), expected_keys)
        self.assertIn(res["level"], {"safe", "caution", "risky"})
        self.assertIsInstance(res["safety_score"], float)

    def test_personal_safety_mode_elevation(self):
        """Verify that Personal Safety Mode penalizes isolated, unlit night conditions more heavily."""
        dark_isolated_segment = {
            "segment_id": "seg_dark_isolated",
            "length_m": "300.0",
            "road_type": "tertiary",
            "speed_limit": "30",
            "lit": "no",
            "hazard_density": "0.2",
            "mean_hazard_confidence": "0.4",
            "mean_freshness": "0.4",
            "accident_score": "0.2",
            "isolation_score": "0.8",
            "time_of_day": "night",
        }
        std_engine = WeightedSumBaselineEngine(personal_safety_mode=False)
        ps_engine = WeightedSumBaselineEngine(personal_safety_mode=True)

        std_score = std_engine.score_segment(dark_isolated_segment)["safety_score"]
        ps_score = ps_engine.score_segment(dark_isolated_segment)["safety_score"]

        # In personal safety mode, darkness and high isolation must reduce the safety score
        self.assertLess(ps_score, std_score)


if __name__ == "__main__":
    unittest.main()

