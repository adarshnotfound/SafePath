"""Unit tests for SafePath Module 1 (Hazard Detection).

Verifies:
- Frame extraction and lighting estimation buckets ('well lit', 'dim', 'dark')
- Adherence of baseline and proposed models to Section 8 schema
- Benchmark execution and evaluation output generation
- Error analysis generation
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from detection.baseline_detector import BaselineHazardDetector
from detection.detector_base import VALID_HAZARD_CLASSES, DetectionResult
from detection.error_analysis import generate_error_analysis_report
from detection.evaluate import evaluate_detection_models
from detection.export import SECTION_8_COLUMNS, load_detections_from_csv, save_detections_to_csv
from detection.frame_extractor import estimate_lighting
from detection.yolo_detector import YOLOHazardDetector


class TestDetectionModule(unittest.TestCase):
    def test_lighting_estimation_dark(self):
        # Frame with mean luminance 20 -> 'dark'
        dark_frame = np.full((100, 100, 3), 20, dtype=np.uint8)
        cond, lum = estimate_lighting(dark_frame)
        self.assertEqual(cond, "dark")
        self.assertAlmostEqual(lum, 20.0, places=1)

    def test_lighting_estimation_dim(self):
        # Frame with mean luminance 80 -> 'dim'
        dim_frame = np.full((100, 100, 3), 80, dtype=np.uint8)
        cond, lum = estimate_lighting(dim_frame)
        self.assertEqual(cond, "dim")
        self.assertAlmostEqual(lum, 80.0, places=1)

    def test_lighting_estimation_well_lit(self):
        # Frame with mean luminance 180 -> 'well lit'
        bright_frame = np.full((100, 100, 3), 180, dtype=np.uint8)
        cond, lum = estimate_lighting(bright_frame)
        self.assertEqual(cond, "well lit")
        self.assertAlmostEqual(lum, 180.0, places=1)

    def test_baseline_detector_output_contract(self):
        detector = BaselineHazardDetector(confidence_threshold=0.3, random_seed=123)
        dummy_frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
        detections = detector.predict(dummy_frame, frame_id="frame_0001", drive_id="drive_test")

        for det in detections:
            self.assertIn(det.hazard_class, VALID_HAZARD_CLASSES)
            self.assertTrue(0.0 <= det.confidence <= 1.0)
            self.assertEqual(len(det.bbox), 4)
            # Verify Section 8 dictionary conversion
            contract_dict = det.to_contract_dict()
            for col in SECTION_8_COLUMNS:
                self.assertIn(col, contract_dict)

    def test_yolo_detector_output_contract(self):
        detector = YOLOHazardDetector(confidence_threshold=0.3, random_seed=123)
        dummy_frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
        detections = detector.predict(dummy_frame, frame_id="frame_0002", drive_id="drive_test")

        for det in detections:
            self.assertIn(det.hazard_class, VALID_HAZARD_CLASSES)
            self.assertTrue(0.0 <= det.confidence <= 1.0)
            self.assertEqual(len(det.bbox), 4)
            contract_dict = det.to_contract_dict()
            for col in SECTION_8_COLUMNS:
                self.assertIn(col, contract_dict)

    def test_export_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_detections.csv"
            sample_dets = [
                DetectionResult(
                    detection_id="det_01",
                    drive_id="drv_01",
                    frame_id="frame_0001",
                    hazard_class="pothole",
                    confidence=0.88,
                    bbox=[100, 120, 150, 180],
                    lat=12.9716,
                    lon=77.5946,
                    timestamp="2026-11-01T10:00:12Z",
                    is_simulated=True,
                ),
                DetectionResult(
                    detection_id="det_02",
                    drive_id="drv_01",
                    frame_id="frame_0002",
                    hazard_class="speed_breaker",
                    confidence=0.92,
                    bbox=[200, 220, 250, 350],
                    lat=12.9721,
                    lon=77.5951,
                    timestamp="2026-11-01T10:00:45Z",
                    is_simulated=True,
                ),
            ]
            save_detections_to_csv(sample_dets, csv_path)
            loaded = load_detections_from_csv(csv_path)

            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].detection_id, "det_01")
            self.assertEqual(loaded[0].hazard_class, "pothole")
            self.assertEqual(loaded[0].bbox, [100, 120, 150, 180])
            self.assertAlmostEqual(loaded[0].lat, 12.9716, places=4)

    def test_evaluation_runner_and_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "comparison.json"
            metrics = evaluate_detection_models(num_eval_frames=5, random_seed=42, output_path=json_path)

            self.assertTrue(json_path.exists())
            self.assertIn("baseline", metrics)
            self.assertIn("proposed", metrics)
            self.assertGreater(metrics["proposed"]["mAP_0_5"], metrics["baseline"]["mAP_0_5"])

    def test_error_analysis_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "failure_cases.json"
            report = generate_error_analysis_report(output_path=json_path)

            self.assertTrue(json_path.exists())
            self.assertGreaterEqual(report["total_categories_analyzed"], 4)


if __name__ == "__main__":
    unittest.main()
