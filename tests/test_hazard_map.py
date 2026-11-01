"""Unit and sanity tests for SafePath Module B1: Simulated Drive Generator."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from hazard_map.simulated_drive import (
    add_gps_noise,
    generate_ground_truth_hazards,
    is_hazard_present_on_drive,
    run_generator,
    simulate_multi_drive_detections,
)


class TestSimulatedDriveGenerator(unittest.TestCase):
    """Test suite verifying Module B1 functionality, contracts, and reproducibility."""

    def test_reproducibility_with_seed(self):
        """Detections generated with the same seed should be bit-for-bit identical."""
        hazards = generate_ground_truth_hazards(seed=42)
        dets1, meta1 = simulate_multi_drive_detections(hazards, seed=42)
        dets2, meta2 = simulate_multi_drive_detections(hazards, seed=42)

        self.assertEqual(len(dets1), len(dets2))
        self.assertEqual(meta1, meta2)
        for d1, d2 in zip(dets1, dets2):
            self.assertEqual(d1.detection_id, d2.detection_id)
            self.assertEqual(d1.lat, d2.lat)
            self.assertEqual(d1.lon, d2.lon)
            self.assertEqual(d1.confidence, d2.confidence)

    def test_gps_noise_within_bounds(self):
        """GPS noise should remain realistic (approx 5-25m maximum for 99.7% of draws)."""
        base_lat, base_lon = 12.9716, 77.5946
        diffs_meters = []
        for _ in range(100):
            n_lat, n_lon = add_gps_noise(base_lat, base_lon, std_meters=8.0)
            d_lat = (n_lat - base_lat) * 111320.0
            d_lon = (n_lon - base_lon) * (111320.0 * 0.9745)
            dist = (d_lat**2 + d_lon**2) ** 0.5
            diffs_meters.append(dist)

        avg_dist = sum(diffs_meters) / len(diffs_meters)
        self.assertGreater(avg_dist, 3.0)
        self.assertLess(avg_dist, 20.0)

    def test_hazard_lifecycle(self):
        """Transient hazards (animals) or repaired hazards should not appear after repair/expiry."""
        hazards = generate_ground_truth_hazards(seed=42)
        animal_haz = next(h for h in hazards if h.hazard_id == "gt_haz_003")
        self.assertTrue(is_hazard_present_on_drive(animal_haz, "drive_01"))
        self.assertFalse(is_hazard_present_on_drive(animal_haz, "drive_02"))
        self.assertFalse(is_hazard_present_on_drive(animal_haz, "drive_03"))

        repaired_pothole = next(h for h in hazards if h.hazard_id == "gt_haz_004")
        self.assertTrue(is_hazard_present_on_drive(repaired_pothole, "drive_01"))
        self.assertTrue(is_hazard_present_on_drive(repaired_pothole, "drive_02"))
        self.assertFalse(is_hazard_present_on_drive(repaired_pothole, "drive_03"))

    def test_detections_csv_contract_compliance(self):
        """Ensure produced CSV exactly matches the Section 8 contract."""
        with tempfile.TemporaryDirectory() as tmpdir:
            det_path = Path(tmpdir) / "detections.csv"
            gt_path = Path(tmpdir) / "ground_truth_hazards.csv"
            meta_path = Path(tmpdir) / "metrics.json"

            run_generator(
                output_detections_path=det_path,
                output_gt_path=gt_path,
                metrics_path=meta_path,
                seed=42,
            )

            self.assertTrue(det_path.exists())
            self.assertTrue(gt_path.exists())
            self.assertTrue(meta_path.exists())

            # Verify detections contract columns
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
            with open(det_path, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)

            self.assertEqual(header, expected_cols)
            self.assertGreaterEqual(len(rows), 10)

            # Check JSON metrics contains baseline details
            with open(meta_path, mode="r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["random_seed"], 42)
            self.assertIn("total_detections_generated", meta)


if __name__ == "__main__":
    unittest.main()
