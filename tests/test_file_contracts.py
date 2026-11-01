"""Tests verifying exact adherence to file contracts specified in Section 8."""

import unittest
from pathlib import Path
import csv
import json

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


class TestFileContracts(unittest.TestCase):
    def test_detections_csv_contract(self):
        expected_cols = [
            "detection_id",
            "drive_id",
            "frame_id",
            "class",
            "confidence",
            "bbox",
            "lat",
            "lon",
            "timestamp",
            "is_simulated",
        ]
        file_path = SAMPLE_DIR / "detections.csv"
        self.assertTrue(file_path.exists())

        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        self.assertEqual(header, expected_cols)
        self.assertTrue(5 <= len(rows) <= 10)

    def test_hazards_csv_contract(self):
        expected_cols = [
            "hazard_id",
            "type",
            "lat",
            "lon",
            "first_seen",
            "last_seen",
            "observation_count",
            "drive_count",
            "confidence",
            "freshness",
            "status",
            "segment_id",
            "is_simulated",
        ]
        file_path = SAMPLE_DIR / "hazards.csv"
        self.assertTrue(file_path.exists())

        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        self.assertEqual(header, expected_cols)
        self.assertTrue(5 <= len(rows) <= 10)

    def test_accident_hotspots_csv_contract(self):
        expected_cols = [
            "hotspot_id",
            "lat",
            "lon",
            "radius_m",
            "accident_count",
            "severity_weight",
            "segment_id",
            "source_country",
        ]
        file_path = SAMPLE_DIR / "accident_hotspots.csv"
        self.assertTrue(file_path.exists())

        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        self.assertEqual(header, expected_cols)
        self.assertTrue(5 <= len(rows) <= 10)

    def test_segment_features_csv_contract(self):
        expected_cols = [
            "segment_id",
            "length_m",
            "road_type",
            "speed_limit",
            "lit",
            "hazard_density",
            "mean_hazard_confidence",
            "mean_freshness",
            "accident_score",
            "isolation_score",
            "time_of_day",
        ]
        file_path = SAMPLE_DIR / "segment_features.csv"
        self.assertTrue(file_path.exists())

        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        self.assertEqual(header, expected_cols)
        self.assertTrue(5 <= len(rows) <= 10)

    def test_survey_ratings_csv_contract(self):
        expected_cols = [
            "segment_id",
            "rater_id",
            "rating",
            "time_of_day_context",
        ]
        file_path = SAMPLE_DIR / "survey_ratings.csv"
        self.assertTrue(file_path.exists())

        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        self.assertEqual(header, expected_cols)
        self.assertTrue(5 <= len(rows) <= 10)

    def test_segment_scores_csv_contract(self):
        expected_cols = [
            "segment_id",
            "safety_score",
            "level",
            "top_factors",
            "fired_rules",
            "model_name",
        ]
        file_path = SAMPLE_DIR / "segment_scores.csv"
        self.assertTrue(file_path.exists())

        with open(file_path, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = list(reader)

        self.assertEqual(header, expected_cols)
        self.assertTrue(5 <= len(rows) <= 10)

    def test_routes_json_contract(self):
        expected_keys = {
            "route_id",
            "segment_ids",
            "total_time",
            "total_length",
            "aggregate_safety",
            "worst_segment_score",
            "explanation",
        }
        file_path = SAMPLE_DIR / "routes.json"
        self.assertTrue(file_path.exists())

        with open(file_path, mode="r", encoding="utf-8") as f:
            routes = json.load(f)

        self.assertIsInstance(routes, list)
        self.assertGreaterEqual(len(routes), 1)
        for route in routes:
            self.assertEqual(set(route.keys()), expected_keys)
            self.assertIsInstance(route["segment_ids"], list)


if __name__ == "__main__":
    unittest.main()
