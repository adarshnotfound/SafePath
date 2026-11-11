"""Export and import utilities for SafePath detection data contract (Section 8).

Ensures all detection results adhere strictly to the schema:
detection_id, drive_id, frame_id, class, confidence, bbox, lat, lon, timestamp, is_simulated
"""

import csv
from pathlib import Path
from typing import List, Union

from detection.detector_base import DetectionResult

SECTION_8_COLUMNS = [
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


def save_detections_to_csv(
    detections: List[DetectionResult],
    output_path: Union[str, Path],
    append: bool = False,
) -> Path:
    """Save a list of DetectionResult objects to a CSV file matching Section 8 contract.

    Parameters
    ----------
    detections : List[DetectionResult]
        Detected hazards to save.
    output_path : Union[str, Path]
        Target file path (e.g. data/sample/detections.csv).
    append : bool
        If True, appends rows to existing file without rewriting header.

    Returns
    -------
    Path
        Path to the saved CSV file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = path.exists()
    mode = "a" if append and file_exists else "w"

    with open(path, mode=mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SECTION_8_COLUMNS)
        if mode == "w" or not file_exists:
            writer.writeheader()
        for det in detections:
            writer.writerow(det.to_contract_dict())

    return path


def load_detections_from_csv(input_path: Union[str, Path]) -> List[DetectionResult]:
    """Load detections from a Section 8 compliant CSV file.

    Parameters
    ----------
    input_path : Union[str, Path]
        Path to the CSV file.

    Returns
    -------
    List[DetectionResult]
        Parsed detection objects.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Detections CSV not found: {path}")

    results = []
    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse bbox string "[ymin, xmin, ymax, xmax]"
            raw_bbox = row.get("bbox", "[0, 0, 0, 0]").strip("[]")
            bbox = [int(v.strip()) for v in raw_bbox.split(",") if v.strip()]
            if len(bbox) != 4:
                bbox = [0, 0, 0, 0]

            lat = float(row["lat"]) if row.get("lat") else None
            lon = float(row["lon"]) if row.get("lon") else None
            is_sim = row.get("is_simulated", "True").lower() in ("true", "1", "yes")

            results.append(
                DetectionResult(
                    detection_id=row["detection_id"],
                    drive_id=row["drive_id"],
                    frame_id=row["frame_id"],
                    hazard_class=row["class"],
                    confidence=float(row["confidence"]),
                    bbox=bbox,
                    lat=lat,
                    lon=lon,
                    timestamp=row.get("timestamp"),
                    is_simulated=is_sim,
                )
            )
    return results
