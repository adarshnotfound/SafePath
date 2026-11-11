"""Base classes and data contracts for SafePath Hazard Detectors (Module 1).

Follows Section 4.1 & Section 8 of the SafePath Specification:
- Classes: 'pothole', 'speed_breaker', 'animal'
- Output per detection: class, bbox, confidence, frame_id, lat, lon, timestamp, is_simulated
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# Allowed target classes per Section 4.1
VALID_HAZARD_CLASSES = {"pothole", "speed_breaker", "animal"}

# Common animal classes in COCO/IDD to map into unified 'animal' category
ANIMAL_MAPPINGS = {
    "cow": "animal",
    "dog": "animal",
    "horse": "animal",
    "sheep": "animal",
    "cat": "animal",
    "elephant": "animal",
    "bear": "animal",
    "bird": "animal",
    "buffalo": "animal",
    "goat": "animal",
}


@dataclass
class DetectionResult:
    """Represents a single detected road hazard.

    Conforms to Section 8:
    detection_id, drive_id, frame_id, class, confidence, bbox, lat, lon, timestamp, is_simulated
    """

    detection_id: str
    drive_id: str
    frame_id: str
    hazard_class: str  # 'pothole' | 'speed_breaker' | 'animal'
    confidence: float
    bbox: List[int]  # [ymin, xmin, ymax, xmax] in pixels
    lat: Optional[float] = None
    lon: Optional[float] = None
    timestamp: Optional[str] = None
    is_simulated: bool = True

    def to_contract_dict(self) -> Dict[str, Any]:
        """Convert detection to exact column names specified in Section 8."""
        return {
            "detection_id": self.detection_id,
            "drive_id": self.drive_id,
            "frame_id": self.frame_id,
            "class": self.hazard_class,
            "confidence": round(float(self.confidence), 4),
            "bbox": f"[{self.bbox[0]}, {self.bbox[1]}, {self.bbox[2]}, {self.bbox[3]}]",
            "lat": f"{self.lat:.5f}" if self.lat is not None else "",
            "lon": f"{self.lon:.5f}" if self.lon is not None else "",
            "timestamp": self.timestamp or "",
            "is_simulated": str(self.is_simulated),
        }


class BaseHazardDetector(ABC):
    """Abstract base class for all road hazard perception models."""

    def __init__(
        self,
        model_name: str,
        confidence_threshold: float = 0.5,
        random_seed: int = 42,
    ):
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.random_seed = random_seed
        self.is_loaded = False

    @abstractmethod
    def load(self, weights_path: Optional[str] = None) -> None:
        """Load model weights into memory."""
        pass

    @abstractmethod
    def predict(
        self,
        frame: np.ndarray,
        frame_id: str,
        drive_id: str = "drive_sim",
    ) -> List[DetectionResult]:
        """Run object detection inference on a single image frame.

        Parameters
        ----------
        frame : np.ndarray
            Input image in BGR format.
        frame_id : str
            Identifier for the current frame (e.g. 'frame_0012').
        drive_id : str
            Identifier for the drive session.

        Returns
        -------
        List[DetectionResult]
            List of detected hazards.
        """
        pass

    @abstractmethod
    def get_model_size_mb(self) -> float:
        """Return the model size in Megabytes (MB)."""
        pass
