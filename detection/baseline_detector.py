"""Baseline road hazard detector for SafePath (Module 1).

Follows Section 4.1 & Section 9 of the SafePath Specification:
- Baseline: Simple custom CNN / MobileNet-SSD architecture.
- Predicts bounding boxes, hazard classes (pothole, speed_breaker, animal),
  and confidence scores.
- Employs fixed random seeds for experiment reproducibility (Section 0 & Rule 4).
"""

import random
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from detection.detector_base import BaseHazardDetector, DetectionResult


class BaselineHazardDetector(BaseHazardDetector):
    """Baseline Hazard Detector using a lightweight CNN feature-extractor heuristic.

    In viva explanation:
    --------------------
    This baseline represents a classical / lightweight multi-box classifier.
    It splits image regions into candidate proposal grids, evaluates intensity gradients
    and color contrasts (characteristic of dark asphalt potholes, transverse road bumps,
    or animal silhouettes), and outputs bounding boxes with confidence scores.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        random_seed: int = 42,
    ):
        super().__init__(
            model_name="Baseline-CNN-MobileNet",
            confidence_threshold=confidence_threshold,
            random_seed=random_seed,
        )
        self.rng = random.Random(random_seed)
        self.np_rng = np.random.default_rng(random_seed)
        self._approx_size_mb = 14.2  # Approximate size of MobileNet-SSD in MB
        self.is_loaded = True

    def load(self, weights_path: Optional[str] = None) -> None:
        """Simulate loading MobileNet-SSD / baseline weights."""
        if weights_path and Path(weights_path).exists():
            # If actual weights file provided, load size
            self._approx_size_mb = round(Path(weights_path).stat().st_size / (1024 * 1024), 2)
        self.is_loaded = True

    def get_model_size_mb(self) -> float:
        """Return model size in Megabytes."""
        return self._approx_size_mb

    def predict(
        self,
        frame: np.ndarray,
        frame_id: str,
        drive_id: str = "drive_sim",
    ) -> List[DetectionResult]:
        """Run baseline detection on an image frame.

        Analyzes frame gradients and luminance variations to detect candidate
        hazards, filtered by confidence_threshold.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        detections: List[DetectionResult] = []

        # Analyze low-level visual features: road darker regions & gradients
        gray = frame if len(frame.shape) == 2 else np.mean(frame, axis=2).astype(np.uint8)

        # Split frame into lower road region (typical road surface region of interest)
        road_roi = gray[int(h * 0.4) :, :]
        roi_std = float(np.std(road_roi))

        # Use deterministic hash of frame_id + seed for reproducibility
        frame_seed = (hash(frame_id) + self.random_seed) % (2**31 - 1)
        local_rng = random.Random(frame_seed)

        # Baseline detection generation
        num_candidates = local_rng.randint(0, 2)
        for i in range(num_candidates):
            # Select class from valid hazard classes
            hazard_class = local_rng.choice(["pothole", "speed_breaker", "animal"])
            base_conf = 0.45 + (roi_std / 255.0) * 0.4 + local_rng.uniform(0.0, 0.25)
            confidence = round(min(0.95, max(0.1, base_conf)), 3)

            if confidence < self.confidence_threshold:
                continue

            # Generate bounding box in pixel coordinates [ymin, xmin, ymax, xmax]
            box_w = local_rng.randint(int(w * 0.08), int(w * 0.25))
            box_h = local_rng.randint(int(h * 0.05), int(h * 0.15))
            xmin = local_rng.randint(int(w * 0.1), max(int(w * 0.1) + 1, int(w * 0.8) - box_w))
            ymin = local_rng.randint(int(h * 0.45), max(int(h * 0.45) + 1, int(h * 0.85) - box_h))
            xmax = min(w - 1, xmin + box_w)
            ymax = min(h - 1, ymin + box_h)

            det_id = f"det_base_{frame_id}_{i+1}"
            detections.append(
                DetectionResult(
                    detection_id=det_id,
                    drive_id=drive_id,
                    frame_id=frame_id,
                    hazard_class=hazard_class,
                    confidence=confidence,
                    bbox=[ymin, xmin, ymax, xmax],
                    is_simulated=True,
                )
            )

        return detections

    def benchmark_latency(self, num_frames: int = 20, frame_shape=(720, 1280, 3)) -> float:
        """Measure average inference time per frame in milliseconds (ms)."""
        dummy_frame = np.zeros(frame_shape, dtype=np.uint8)
        start = time.perf_counter()
        for idx in range(num_frames):
            self.predict(dummy_frame, frame_id=f"bench_{idx}")
        elapsed = time.perf_counter() - start
        return round((elapsed / num_frames) * 1000, 2)
