"""Unit and integration tests for SafePath video drive processor (Module 1).

Verifies:
- Route waypoint interpolation and progression
- Section 4.2 GPS noise bounds (5-15m)
- End-to-end drive processing output schema matching Section 8
- Simulated data provenance disclosure (is_simulated = True)
"""

import math
import tempfile
import unittest
from pathlib import Path

from detection.baseline_detector import BaselineHazardDetector
from detection.export import load_detections_from_csv
from detection.process_drive import (
    DEFAULT_ROUTE_WAYPOINTS,
    apply_gps_noise,
    create_mock_video_sequence,
    interpolate_route_position,
    process_drive_video,
)


class TestProcessDrive(unittest.TestCase):
    def test_interpolate_route_position(self):
        waypoints = [(10.0, 20.0), (10.2, 20.2)]
        start = interpolate_route_position(0.0, waypoints)
        mid = interpolate_route_position(0.5, waypoints)
        end = interpolate_route_position(1.0, waypoints)

        self.assertEqual(start, (10.0, 20.0))
        self.assertEqual(end, (10.2, 20.2))
        self.assertAlmostEqual(mid[0], 10.1, places=3)
        self.assertAlmostEqual(mid[1], 20.1, places=3)

    def test_apply_gps_noise_bounds(self):
        base_lat, base_lon = 12.9716, 77.5946
        noise_m = 10.0
        angle = math.pi / 4

        noisy_lat, noisy_lon = apply_gps_noise(base_lat, base_lon, noise_m, angle)

        # 10m is ~0.00009 degrees; should be strictly non-zero and within realistic bounds
        self.assertNotEqual(noisy_lat, base_lat)
        self.assertNotEqual(noisy_lon, base_lon)
        self.assertAlmostEqual(noisy_lat, base_lat, delta=0.0005)
        self.assertAlmostEqual(noisy_lon, base_lon, delta=0.0005)

    def test_end_to_end_drive_processing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            video_file = tmp_path / "test_drive.mp4"
            csv_output = tmp_path / "drive_detections.csv"

            # Create minimal test video
            create_mock_video_sequence(video_file, num_frames=6, fps=2)
            self.assertTrue(video_file.exists())

            detector = BaselineHazardDetector(confidence_threshold=0.2, random_seed=42)
            summary = process_drive_video(
                video_path=video_file,
                drive_id="drive_unit_01",
                detector=detector,
                waypoints=DEFAULT_ROUTE_WAYPOINTS,
                sample_interval_sec=1.0,
                start_time_iso="2026-11-01T10:00:00Z",
                output_csv_path=csv_output,
                random_seed=42,
            )

            self.assertEqual(summary["drive_id"], "drive_unit_01")
            self.assertGreaterEqual(summary["frames_sampled"], 2)
            self.assertTrue(csv_output.exists())

            # Verify exported CSV adheres to Section 8 schema
            loaded = load_detections_from_csv(csv_output)
            self.assertEqual(len(loaded), summary["total_detections"])

            for det in loaded:
                self.assertEqual(det.drive_id, "drive_unit_01")
                self.assertIsNotNone(det.lat)
                self.assertIsNotNone(det.lon)
                self.assertTrue(det.timestamp.startswith("2026-11-01T10:00:"))
                self.assertTrue(det.is_simulated)  # Data honesty flag verified


if __name__ == "__main__":
    unittest.main()

