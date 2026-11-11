"""Unit tests for SafePath training pipeline, confusion matrix, and failure cases.

Verifies:
- Hardware profiling detection (CPU/GPU)
- Multi-class confusion matrix shape and normalized probabilities
- Generation and disk existence of all 20 failure case images
- Comparative metrics logging to JSON
"""

import tempfile
import unittest
from pathlib import Path

from detection.train import (
    compute_confusion_matrix,
    get_hardware_info,
    train_and_compare_detectors,
)
from detection.visualize_failures import (
    generate_all_failure_cases,
    generate_failure_case_image,
)


class TestTrainingPipeline(unittest.TestCase):
    def test_hardware_profiler(self):
        hw = get_hardware_info()
        self.assertIn("device_type", hw)
        self.assertIn("os", hw)
        self.assertIn(hw["device_type"], ["CPU", "GPU"])

    def test_confusion_matrix_dimensions_and_normalization(self):
        cm = compute_confusion_matrix("yolo", random_seed=42)
        self.assertEqual(len(cm["labels"]), 4)  # 3 classes + background
        self.assertEqual(len(cm["matrix_counts"]), 4)
        self.assertEqual(len(cm["matrix_normalized"]), 4)

        # Check rows sum to approximately 1.0
        for norm_row in cm["matrix_normalized"]:
            self.assertAlmostEqual(sum(norm_row), 1.0, places=2)

    def test_failure_case_image_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            single_record = generate_failure_case_image("shadows", 1, out_dir)
            self.assertTrue(Path(single_record["saved_path"]).exists())

            # Test generation of all 20 failure images
            all_records = generate_all_failure_cases(output_dir=out_dir, samples_per_category=5)
            self.assertEqual(len(all_records), 20)

            # Check all 20 image files exist on disk
            jpg_files = list(out_dir.glob("*.jpg"))
            self.assertEqual(len(jpg_files), 20)

            index_file = out_dir / "index.json"
            self.assertTrue(index_file.exists())

    def test_train_and_compare_detectors_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            res_dir = Path(tmpdir)
            metrics = train_and_compare_detectors(epochs=1, random_seed=42, results_dir=res_dir)

            self.assertTrue((res_dir / "detection_experiment.json").exists())
            self.assertTrue((res_dir / "detection_confusion_matrix.json").exists())
            self.assertIn("metrics", metrics)
            self.assertIn("baseline", metrics["metrics"])
            self.assertIn("proposed", metrics["metrics"])


if __name__ == "__main__":
    unittest.main()

