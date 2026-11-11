"""Batch YOLO object detection on a folder of images for SafePath (Module 1).

Follows User Specification & Section 8 File Contract:
- Runs YOLO model (YOLOHazardDetector) over all image files in a folder.
- Exports detections.csv with EXACT columns:
    detection_id, drive_id, frame_id, class, confidence, bbox, lat, lon, timestamp, is_simulated
- Strictly leaves lat, lon, and timestamp empty ("") prior to GPS map placement.
"""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from detection.yolo_detector import YOLOHazardDetector

# Exact 10 columns specified by Section 8 & user prompt
EXACT_CSV_COLUMNS = [
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


def load_image_file(image_path: Path) -> Optional[np.ndarray]:
    """Load image from disk into BGR numpy array."""
    if cv2 is not None:
        return cv2.imread(str(image_path))

    # Resilient fallback if cv2 is not installed
    try:
        from PIL import Image

        pil_img = Image.open(image_path).convert("RGB")
        rgb = np.array(pil_img)
        # Convert RGB to BGR
        return rgb[:, :, ::-1]
    except Exception:
        return None


def run_batch_detection_on_folder(
    image_dir: Union[str, Path],
    output_csv_path: Union[str, Path],
    drive_id: str = "drive_batch",
    confidence_threshold: float = 0.5,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Process all images in a folder with YOLO and write detections.csv.

    Parameters
    ----------
    image_dir : Union[str, Path]
        Directory containing input images (.jpg, .jpeg, .png).
    output_csv_path : Union[str, Path]
        Output path for detections.csv.
    drive_id : str
        Drive session identifier.
    confidence_threshold : float
        Detection confidence threshold.
    random_seed : int
        Fixed seed for reproducibility.

    Returns
    -------
    List[Dict[str, Any]]
        List of detection dictionaries written to CSV.
    """
    image_dir = Path(image_dir)
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    detector = YOLOHazardDetector(
        confidence_threshold=confidence_threshold,
        random_seed=random_seed,
    )

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in image_extensions])

    csv_rows: List[Dict[str, Any]] = []
    det_counter = 1

    for img_p in image_files:
        frame = load_image_file(img_p)
        if frame is None:
            continue

        frame_id = img_p.stem
        detections = detector.predict(frame, frame_id=frame_id, drive_id=drive_id)

        for det in detections:
            det_id = f"det_{det_counter:04d}"
            det_counter += 1

            # Format bounding box as [ymin, xmin, ymax, xmax]
            bbox_str = f"[{det.bbox[0]}, {det.bbox[1]}, {det.bbox[2]}, {det.bbox[3]}]"

            # Exactly match the 10 requested columns with lat, lon, timestamp left EMPTY
            row = {
                "detection_id": det_id,
                "drive_id": drive_id,
                "frame_id": frame_id,
                "class": det.hazard_class,
                "confidence": round(float(det.confidence), 4),
                "bbox": bbox_str,
                "lat": "",          # Left empty for now as requested
                "lon": "",          # Left empty for now as requested
                "timestamp": "",    # Left empty for now as requested
                "is_simulated": str(det.is_simulated),
            }
            csv_rows.append(row)

    # Write detections.csv with exact header
    with open(output_csv_path, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXACT_CSV_COLUMNS)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)

    return csv_rows


def main():
    parser = argparse.ArgumentParser(description="Run YOLO hazard detection on an image folder.")
    parser.add_argument("--input-dir", type=str, default="data/sample/test_images", help="Folder of input images.")
    parser.add_argument("--output-csv", type=str, default="data/processed/detections.csv", help="Target CSV file path.")
    parser.add_argument("--drive-id", type=str, default="drive_01", help="Identifier for drive.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    args = parser.parse_args()

    rows = run_batch_detection_on_folder(
        image_dir=args.input_dir,
        output_csv_path=args.output_csv,
        drive_id=args.drive_id,
        confidence_threshold=args.conf,
    )
    print(f"Batch detection complete: wrote {len(rows)} detections to {args.output_csv}")


if __name__ == "__main__":
    main()
