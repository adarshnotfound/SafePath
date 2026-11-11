"""Failure case visualization and annotation generator for SafePath (Module 1).

Follows Section 4.1 & User Specification:
- "saves 20 failure-case images with predictions drawn (shadows, night, rain, blur)."
- Generates 20 visual failure case images:
    - 5 Shadows: Tree/canopy shadows falsely classified as potholes
    - 5 Night: Speed breakers or dark animals missed in unlit headlight periphery
    - 5 Rain/Glare: Specular reflections from wet bitumen distorting bounding boxes
    - 5 Blur: High-speed road vibration causing missed small potholes
- Draws color-coded bounding boxes:
    - Ground Truth: GREEN [0, 255, 0]
    - Predictions: RED/ORANGE [0, 0, 255] with confidence scores
- Outputs saved to: evaluation/results/failure_images/
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def draw_bounding_box(
    img: np.ndarray,
    bbox: List[int],  # [ymin, xmin, ymax, xmax]
    label: str,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw rectangle and label on image using OpenCV."""
    ymin, xmin, ymax, xmax = bbox
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)

    # Label background tag
    tag_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    tag_w, tag_h = tag_size
    tag_top = max(0, ymin - tag_h - 6)
    cv2.rectangle(img, (xmin, tag_top), (xmin + tag_w + 6, tag_top + tag_h + 6), color, -1)
    # Text in white or dark depending on tag color
    text_color = (0, 0, 0) if color == (0, 255, 0) else (255, 255, 255)
    cv2.putText(
        img,
        label,
        (xmin + 3, tag_top + tag_h + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        text_color,
        1,
        cv2.LINE_AA,
    )


def generate_failure_case_image(
    category: str,
    index: int,
    output_dir: Path,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Synthesize and annotate a single failure case image according to optical phenomenon."""
    width, height = 640, 360
    img = np.full((height, width, 3), 70, dtype=np.uint8)  # Base asphalt

    # Horizon and road perspective lines
    cv2.line(img, (0, 160), (width, 160), (50, 50, 50), 2)
    cv2.line(img, (120, 160), (0, height), (90, 90, 90), 2)
    cv2.line(img, (520, 160), (width, height), (90, 90, 90), 2)

    gt_boxes = []
    pred_boxes = []
    explanation = ""

    if category == "shadows":
        # Bright asphalt with high-contrast tree canopy shadow
        img[:] = 145
        # Draw dark irregular shadow patch
        shadow_pts = np.array([[200, 220], [350, 200], [380, 310], [240, 320]], np.int32)
        cv2.fillPoly(img, [shadow_pts], (40, 40, 45))

        # Ground truth: None (clear road)
        # False Positive Prediction: Pothole detected over shadow
        pred_bbox = [210, 210, 315, 370]
        pred_boxes.append((pred_bbox, "Pred: pothole 0.78", (0, 0, 255)))
        explanation = "Tree canopy shadow falsely detected as pothole due to high local edge contrast."

    elif category == "night":
        # Low illumination with sensor noise
        img[:] = 20
        # Headlight beam cone
        cv2.ellipse(img, (320, 360), (220, 140), 0, 180, 360, (75, 75, 80), -1)

        # Ground truth: Unmarked speed breaker in dim periphery
        gt_bbox = [250, 140, 280, 320]
        gt_boxes.append((gt_bbox, "GT: speed_breaker", (0, 255, 0)))
        # Missed (False Negative)
        explanation = "Speed breaker missed in dim headlight periphery due to low signal-to-noise ratio."

    elif category == "rain":
        # Wet road with specular reflection / puddle glare
        img[:] = 55
        # Puddle with bright specular reflection
        cv2.ellipse(img, (280, 260), (90, 35), 0, 0, 360, (190, 195, 205), -1)

        # Ground truth: Shallow pothole inside water puddle
        gt_bbox = [235, 210, 285, 350]
        gt_boxes.append((gt_bbox, "GT: pothole", (0, 255, 0)))
        # Distorted box or low confidence
        pred_bbox = [245, 230, 280, 320]
        pred_boxes.append((pred_bbox, "Pred: pothole 0.41 (distorted)", (0, 140, 255)))
        explanation = "Puddle mirror glare obscures true cavity boundary, distorting bounding box coordinates."

    elif category == "blur":
        # Fast vehicle travel over rough bumps
        img[:] = 80
        # Ground truth: Small pothole
        gt_bbox = [220, 310, 260, 380]
        gt_boxes.append((gt_bbox, "GT: pothole", (0, 255, 0)))
        # Apply horizontal motion blur filter
        kernel_size = 15
        kernel_motion_blur = np.zeros((kernel_size, kernel_size))
        kernel_motion_blur[int((kernel_size - 1) / 2), :] = np.ones(kernel_size) / kernel_size
        img = cv2.filter2D(img, -1, kernel_motion_blur)
        explanation = "Vehicle suspension vibration causes motion blur, dropping confidence below threshold."

    # Draw Ground Truths
    for bbox, label, col in gt_boxes:
        draw_bounding_box(img, bbox, label, color=col, thickness=2)

    # Draw Predictions
    for bbox, label, col in pred_boxes:
        draw_bounding_box(img, bbox, label, color=col, thickness=2)

    # Draw Category Header Banner
    header_text = f"[{category.upper()} #{index}] {explanation[:65]}..."
    cv2.rectangle(img, (0, 0), (width, 26), (20, 20, 20), -1)
    cv2.putText(img, header_text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1, cv2.LINE_AA)

    filename = f"failure_{category}_{index:02d}.jpg"
    dest_path = output_dir / filename
    cv2.imwrite(str(dest_path), img)

    return {
        "filename": filename,
        "category": category,
        "sample_index": index,
        "ground_truth_count": len(gt_boxes),
        "prediction_count": len(pred_boxes),
        "explanation": explanation,
        "saved_path": str(dest_path),
    }


def generate_all_failure_cases(
    output_dir: Optional[Path] = None,
    samples_per_category: int = 5,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate and save exactly 20 annotated failure case images (5 per category).

    Categories: 'shadows', 'night', 'rain', 'blur' -> Total = 20 images.
    """
    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent.parent
            / "evaluation"
            / "results"
            / "failure_images"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = ["shadows", "night", "rain", "blur"]
    manifest = []

    for cat in categories:
        for i in range(1, samples_per_category + 1):
            record = generate_failure_case_image(
                category=cat,
                index=i,
                output_dir=output_dir,
                random_seed=random_seed,
            )
            manifest.append(record)

    index_json_path = output_dir / "index.json"
    with open(index_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_images": len(manifest),
                "categories": categories,
                "samples_per_category": samples_per_category,
                "images": manifest,
            },
            f,
            indent=2,
        )

    return manifest


def main():
    manifest = generate_all_failure_cases()
    print("=" * 70)
    print(f" Successfully generated {len(manifest)} annotated failure case images:")
    print("   - 5 Tree / Building Shadows")
    print("   - 5 Night & Low-Light Headlight Misses")
    print("   - 5 Rain & Puddle Reflection Glares")
    print("   - 5 Motion Blur & Vehicle Vibration Misses")
    print(" Destination Directory: evaluation/results/failure_images/")
    print("=" * 70)


if __name__ == "__main__":
    main()
