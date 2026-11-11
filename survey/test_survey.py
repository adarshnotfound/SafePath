"""Unit tests for SafePath Survey Module (Member C).

Verifies response cleaning, validation bounds, inter-rater reliability calculations,
and adherence to Section 8 survey_ratings.csv contracts.
"""

import unittest
import tempfile
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from survey.clean_survey import clean_survey_responses, compute_inter_rater_agreement
from survey.generate_simulated_ratings import generate_simulated_survey


class TestSurveyModule(unittest.TestCase):
    """Test suite for survey data handling, cleaning, and reliability metrics."""

    def test_inter_rater_agreement_perfect(self):
        """When all raters give identical scores, mean absolute difference is 0."""
        perfect_ratings = {
            "seg_1": {"rater_1": 4, "rater_2": 4, "rater_3": 4},
            "seg_2": {"rater_1": 2, "rater_2": 2, "rater_3": 2},
        }
        metrics = compute_inter_rater_agreement(perfect_ratings)
        self.assertEqual(metrics["mean_pairwise_abs_diff"], 0.0)
        self.assertEqual(metrics["percent_exact_agreement"], 100.0)

    def test_clean_survey_responses_contract(self):
        """Verify invalid values are filtered and output columns strictly follow contract."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            raw_csv = tmp_path / "raw_survey.csv"
            clean_csv = tmp_path / "survey_ratings.csv"

            # Write test data with some invalid rows
            with open(raw_csv, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["segment_id", "rater_id", "rating", "time_of_day_context"])
                writer.writerow(["seg_01", "rater_01", "4", "day"])        # valid
                writer.writerow(["seg_01", "rater_02", "3", "day"])        # valid
                writer.writerow(["seg_02", "rater_01", "6", "night"])      # invalid rating > 5
                writer.writerow(["seg_02", "rater_02", "not_num", "night"])# invalid rating string
                writer.writerow(["", "rater_03", "2", "night"])            # missing segment_id

            summary = clean_survey_responses(raw_csv, clean_csv)
            self.assertEqual(summary["valid_ratings"], 2)
            self.assertEqual(summary["skipped_rows"], 3)

            # Check cleaned file contract
            with open(clean_csv, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)

            expected_header = ["segment_id", "rater_id", "rating", "time_of_day_context"]
            self.assertEqual(header, expected_header)
            self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()

