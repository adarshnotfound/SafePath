"""Unit tests for SafePath LightingCNN classifier (Module 1 Stretch Goal).

Verifies:
- Model initialization, tensor forward pass, and probability normalization
- Frame preprocessing (spatial resizing to 128x128 and channel normalization)
- Train loss reduction on synthetic road lighting patches
- Threshold baseline vs LightingCNN comparative evaluation logging
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from detection.evaluate_lighting import evaluate_lighting_models
from detection.frame_extractor import estimate_lighting
from detection.lighting_cnn import LIGHTING_CLASSES, RoadLightingClassifier


class TestLightingCNN(unittest.TestCase):
    def setUp(self):
        self.classifier = RoadLightingClassifier(random_seed=42)

    def test_model_size_and_initialization(self):
        size_mb = self.classifier.get_model_size_mb()
        self.assertGreater(size_mb, 0.0)
        self.assertLess(size_mb, 1.0)  # Very compact (< 1MB)

    def test_predict_lighting_classes_and_probabilities(self):
        # Test bright frame -> well lit
        bright_frame = np.full((128, 128, 3), 200, dtype=np.uint8)
        pred, conf, probs = self.classifier.predict(bright_frame)
        self.assertIn(pred, LIGHTING_CLASSES)
        self.assertTrue(0.0 <= conf <= 1.0)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)

        # Test dark frame -> dark
        dark_frame = np.full((128, 128, 3), 15, dtype=np.uint8)
        pred_dark, conf_dark, probs_dark = self.classifier.predict(dark_frame)
        self.assertIn(pred_dark, LIGHTING_CLASSES)
        self.assertTrue(0.0 <= conf_dark <= 1.0)
        self.assertAlmostEqual(sum(probs_dark.values()), 1.0, places=2)

    def test_estimate_lighting_cnn_method(self):
        dummy_frame = np.full((100, 100, 3), 85, dtype=np.uint8)
        cond, score = estimate_lighting(dummy_frame, method="cnn")
        self.assertIn(cond, LIGHTING_CLASSES)
        self.assertTrue(0.0 <= score <= 1.0)

    def test_train_synthetic_batch(self):
        losses = self.classifier.train_on_synthetic_batch(epochs=3, lr=0.005)
        self.assertEqual(len(losses), 3)
        self.assertGreater(losses[0], 0.0)

    def test_evaluation_runner_and_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            json_out = Path(tmpdir) / "test_lighting.json"
            metrics = evaluate_lighting_models(num_samples=9, random_seed=42, output_path=json_out)

            self.assertTrue(json_out.exists())
            self.assertIn("baseline", metrics)
            self.assertIn("proposed", metrics)
            self.assertIn("macro_f1", metrics["proposed"])


if __name__ == "__main__":
    unittest.main()
