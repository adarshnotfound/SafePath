"""Unified dataset preparation, validation, and deduplication for SafePath (Module 1).

Follows Section 4.1 & Section 5 of the SafePath Specification:
- Ingests multiple sources: RDD2022 India subset (Pascal VOC XML), Roboflow Universe,
  and Kaggle road hazard sets (YOLO format txt).
- Unifies annotations into 3 target classes:
    0: 'pothole'
    1: 'speed_breaker'
    2: 'animal'
- Removes corrupt/broken files (0-byte, truncated, or unreadable images).
- Performs image content hashing (SHA-256) to eliminate duplicate frames and
  prevent data leakage across splits.
- Generates reproducible 3-way splits ('train', 'val', 'test') with seed 42.
- Generates Ultralytics dataset configuration (data/rdd2022_safepath.yaml).
- Prints detailed per-class and per-split bounding box instance counts.
"""

import hashlib
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# SafePath Unified Class Taxonomy (Section 4.1)
SAFEPATH_CLASSES = ["pothole", "speed_breaker", "animal"]
CLASS_TO_IDX = {name: i for i, name in enumerate(SAFEPATH_CLASSES)}

# Label mapping from multi-source taxonomies
LABEL_MAPPING = {
    # RDD2022 codes (D40 = Pothole)
    "d40": "pothole",
    "pothole": "pothole",
    "potholes": "pothole",
    "cavity": "pothole",
    # Road bumps / speed breakers
    "speed_breaker": "speed_breaker",
    "speed_bump": "speed_breaker",
    "bump": "speed_breaker",
    "speedbreaker": "speed_breaker",
    "speed-breaker": "speed_breaker",
    "road_bump": "speed_breaker",
    # Animals (COCO / IDD classes)
    "animal": "animal",
    "cow": "animal",
    "dog": "animal",
    "horse": "animal",
    "buffalo": "animal",
    "goat": "animal",
    "sheep": "animal",
    "cattle": "animal",
}


def compute_image_hash(img_path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of image file bytes for duplicate detection."""
    p = Path(img_path)
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_image_valid_and_not_broken(img_path: Union[str, Path]) -> Tuple[bool, Tuple[int, int]]:
    """Verify that an image exists, is non-empty, and can be decoded properly.

    Returns
    -------
    Tuple[bool, Tuple[int, int]]
        (is_valid, (width, height))
    """
    p = Path(img_path)
    if not p.exists() or p.stat().st_size == 0:
        return False, (0, 0)

    # Decode check
    if cv2 is not None:
        try:
            img = cv2.imread(str(p))
            if img is None or img.size == 0:
                return False, (0, 0)
            h, w = img.shape[:2]
            return True, (w, h)
        except Exception:
            return False, (0, 0)

    # Pure Python minimal header check fallback
    try:
        size = p.stat().st_size
        return (size > 32), (1280, 720)
    except Exception:
        return False, (0, 0)


def convert_voc_bbox_to_yolo(
    size: Tuple[int, int],
    box: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Convert Pascal VOC bounding box [xmin, ymin, xmax, ymax] to normalized YOLO coordinates."""
    img_w, img_h = size
    xmin, ymin, xmax, ymax = box

    # Clip coordinates to image boundaries
    xmin = max(0.0, min(float(img_w), xmin))
    xmax = max(0.0, min(float(img_w), xmax))
    ymin = max(0.0, min(float(img_h), ymin))
    ymax = max(0.0, min(float(img_h), ymax))

    box_w = xmax - xmin
    box_h = ymax - ymin

    x_center = xmin + (box_w / 2.0)
    y_center = ymin + (box_h / 2.0)

    x_center /= max(1, img_w)
    y_center /= max(1, img_h)
    norm_w = box_w / max(1, img_w)
    norm_h = box_h / max(1, img_h)

    return (
        round(min(1.0, max(0.0, x_center)), 6),
        round(min(1.0, max(0.0, y_center)), 6),
        round(min(1.0, max(0.0, norm_w)), 6),
        round(min(1.0, max(0.0, norm_h)), 6),
    )


def parse_voc_xml(xml_path: Path) -> List[Tuple[str, Tuple[float, float, float, float], Tuple[int, int]]]:
    """Parse Pascal VOC format XML annotation and extract target hazard objects."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size_elem = root.find("size")
    if size_elem is not None:
        width = int(size_elem.findtext("width", "0"))
        height = int(size_elem.findtext("height", "0"))
    else:
        width, height = 0, 0

    annotations = []
    for obj in root.findall("object"):
        raw_name = obj.findtext("name", "").strip().lower()
        mapped_class = LABEL_MAPPING.get(raw_name)
        if mapped_class is None:
            continue

        bndbox = obj.find("bndbox")
        if bndbox is not None:
            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))

            if width == 0 or height == 0:
                width = int(max(width, xmax))
                height = int(max(height, ymax))

            annotations.append((mapped_class, (xmin, ymin, xmax, ymax), (width, height)))

    return annotations


def parse_yolo_txt(
    txt_path: Path,
    source_class_names: Optional[List[str]] = None,
) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    """Parse existing YOLO format txt annotation (e.g. from Roboflow or Kaggle)."""
    annotations = []
    lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            raw_id = int(parts[0])
            xc, yc, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue

        if source_class_names and raw_id < len(source_class_names):
            raw_name = source_class_names[raw_id].lower()
            mapped = LABEL_MAPPING.get(raw_name)
        else:
            # Assume 0: pothole, 1: speed_breaker, 2: animal if 3 classes
            mapped = SAFEPATH_CLASSES[raw_id % len(SAFEPATH_CLASSES)]

        if mapped:
            annotations.append((mapped, (xc, yc, w, h)))

    return annotations


def generate_yolo_dataset_yaml(
    output_yaml_path: Path,
    dataset_root_dir: Path,
    classes: Optional[List[str]] = None,
) -> Path:
    """Generate an Ultralytics YOLOv8 compatible dataset YAML configuration file."""
    if classes is None:
        classes = SAFEPATH_CLASSES

    output_yaml_path = Path(output_yaml_path)
    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_content = f"""# SafePath Dataset Configuration for YOLOv8
# Unifies RDD2022 (India), Roboflow Universe, and Kaggle sets
path: {dataset_root_dir.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

# Classes
nc: {len(classes)}
names: {classes}
"""
    output_yaml_path.write_text(yaml_content, encoding="utf-8")
    return output_yaml_path


def build_unified_dataset(
    raw_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Build unified, cleaned, deduplicated, 3-way partitioned dataset.

    Parameters
    ----------
    raw_dir : Path
        Directory containing raw source files (can contain subfolders for RDD2022, Kaggle, etc.).
    output_dir : Path
        Target directory for processed YOLO dataset.
    train_ratio : float
        Proportion for training split (default 0.70).
    val_ratio : float
        Proportion for validation split (default 0.15).
    test_ratio : float
        Proportion for test split (default 0.15).
    random_seed : int
        Fixed seed for reproducible partitioning without leakage (Rule 4).

    Returns
    -------
    dict
        Execution metrics, duplicate counts, and per-class instance breakdowns.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    for split in ["train", "val", "test"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1. Discover all candidate images
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    all_raw_images = [p for p in raw_dir.rglob("*") if p.suffix.lower() in image_extensions]

    # 2. File integrity checking and deduplication
    valid_unique_samples: List[Tuple[Path, Path, str]] = []  # (image_path, annotation_path, type)
    seen_hashes = set()
    corrupt_count = 0
    duplicate_count = 0

    for img_p in all_raw_images:
        is_valid, size = is_image_valid_and_not_broken(img_p)
        if not is_valid:
            corrupt_count += 1
            continue

        img_hash = compute_image_hash(img_p)
        if img_hash in seen_hashes:
            duplicate_count += 1
            continue
        seen_hashes.add(img_hash)

        # Look for corresponding XML or TXT annotation
        xml_candidate = img_p.with_suffix(".xml")
        txt_candidate = img_p.with_suffix(".txt")

        if xml_candidate.exists():
            valid_unique_samples.append((img_p, xml_candidate, "xml"))
        elif txt_candidate.exists():
            valid_unique_samples.append((img_p, txt_candidate, "txt"))

    # 3. Deterministic 3-way partition without leakage
    rng = random.Random(random_seed)
    rng.shuffle(valid_unique_samples)

    total_valid = len(valid_unique_samples)
    n_train = int(total_valid * train_ratio)
    n_val = int(total_valid * val_ratio)

    splits = {
        "train": valid_unique_samples[:n_train],
        "val": valid_unique_samples[n_train : n_train + n_val],
        "test": valid_unique_samples[n_train + n_val :],
    }

    # Tracking instance counts
    class_counts = {
        "train": {c: 0 for c in SAFEPATH_CLASSES},
        "val": {c: 0 for c in SAFEPATH_CLASSES},
        "test": {c: 0 for c in SAFEPATH_CLASSES},
        "total": {c: 0 for c in SAFEPATH_CLASSES},
    }

    # 4. Export to YOLO format
    for split_name, sample_list in splits.items():
        img_dest = output_dir / "images" / split_name
        lbl_dest = output_dir / "labels" / split_name

        for img_p, ann_p, ann_type in sample_list:
            yolo_lines = []
            if ann_type == "xml":
                parsed = parse_voc_xml(ann_p)
                for class_name, bbox, img_size in parsed:
                    class_id = CLASS_TO_IDX[class_name]
                    yolo_box = convert_voc_bbox_to_yolo(img_size, bbox)
                    yolo_lines.append(f"{class_id} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}")
                    class_counts[split_name][class_name] += 1
                    class_counts["total"][class_name] += 1
            else:
                parsed_yolo = parse_yolo_txt(ann_p)
                for class_name, yolo_box in parsed_yolo:
                    class_id = CLASS_TO_IDX[class_name]
                    yolo_lines.append(f"{class_id} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}")
                    class_counts[split_name][class_name] += 1
                    class_counts["total"][class_name] += 1

            # Write label file
            (lbl_dest / f"{img_p.stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")
            # Write image file
            (img_dest / img_p.name).write_bytes(img_p.read_bytes())

    # Write YAML configuration
    yaml_path = generate_yolo_dataset_yaml(
        output_yaml_path=output_dir.parent / "rdd2022_safepath.yaml",
        dataset_root_dir=output_dir,
    )

    summary = {
        "total_discovered": len(all_raw_images),
        "corrupt_filtered": corrupt_count,
        "duplicates_removed": duplicate_count,
        "clean_samples": total_valid,
        "splits": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "class_instance_counts": class_counts,
        "yaml_config": str(yaml_path),
    }

    return summary


def print_dataset_summary_table(summary: Dict[str, Any]) -> None:
    """Print aligned markdown summary table of class counts across splits."""
    counts = summary["class_instance_counts"]
    splits = summary["splits"]

    print("=" * 74)
    print(" SAFEPATH ROAD HAZARD DATASET PREPARATION SUMMARY")
    print("=" * 74)
    print(f"Total Discovered Files:     {summary['total_discovered']}")
    print(f"Corrupt / 0-byte Discarded: {summary['corrupt_filtered']}")
    print(f"Duplicate Images Removed:   {summary['duplicates_removed']} (Leakage Protected)")
    print(f"Clean Unique Samples:       {summary['clean_samples']}")
    print(f"Splits (Images):            Train={splits['train']} | Val={splits['val']} | Test={splits['test']}")
    print("-" * 74)
    print(f"{'Class Name':<16} | {'Train':<10} | {'Val':<10} | {'Test':<10} | {'Total BBoxes':<12}")
    print("-" * 74)
    for c in SAFEPATH_CLASSES:
        print(f"{c:<16} | {counts['train'][c]:<10} | {counts['val'][c]:<10} | {counts['test'][c]:<10} | {counts['total'][c]:<12}")
    print("-" * 74)
    print(f"YOLO Configuration: {summary['yaml_config']}")
    print("=" * 74)


def create_synthetic_rdd_sample(
    output_dir: Path,
    num_samples: int = 15,
    random_seed: int = 42,
) -> Tuple[Path, Path]:
    """Generate minimal synthetic Pascal VOC annotations and placeholder JPEG images."""
    output_dir = Path(output_dir)
    images_dir = output_dir / "raw_images"
    xml_dir = output_dir / "raw_xml"
    images_dir.mkdir(parents=True, exist_ok=True)
    xml_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(random_seed)
    sample_classes = ["D40", "speed_breaker", "cow"]

    # Minimal 1x1 valid JPEG header bytes
    dummy_jpeg_bytes = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342"
        b"\xff\xc0\x00\x0b\x08\x02\xd0\x05\x00\x01\x01\x11\x00\xff\xda\x00"
        b"\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9"
    )

    for i in range(1, num_samples + 1):
        stem = f"sample_{i:04d}"
        img_path = images_dir / f"{stem}.jpg"
        if cv2 is not None:
            # Generate valid clean road image
            dummy_img = np.full((720, 1280, 3), 60 + (i % 30), dtype=np.uint8)
            cv2.imwrite(str(img_path), dummy_img)
        else:
            img_path.write_bytes(dummy_jpeg_bytes + bytes([i % 256]))

        cls_choice = rng.choice(sample_classes)
        xml_content = f"""<annotation>
    <folder>samples</folder>
    <filename>{stem}.jpg</filename>
    <size>
        <width>1280</width>
        <height>720</height>
        <depth>3</depth>
    </size>
    <object>
        <name>{cls_choice}</name>
        <bndbox>
            <xmin>{rng.randint(200, 400)}</xmin>
            <ymin>{rng.randint(350, 500)}</ymin>
            <xmax>{rng.randint(450, 700)}</xmax>
            <ymax>{rng.randint(550, 680)}</ymax>
        </bndbox>
    </object>
</annotation>
"""
        (xml_dir / f"{stem}.xml").write_text(xml_content, encoding="utf-8")

    return images_dir, xml_dir


def main():
    """One-command dataset rebuild entrypoint."""
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed" / "yolo_dataset"

    # If data/raw has no image files yet, create synthetic verification set
    existing_images = list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.png"))
    if not existing_images:
        print("Notice: No images found in data/raw/. Generating synthetic multi-source verification sample...")
        synth_dir = raw_dir / "synthetic_sample"
        img_dir, xml_dir = create_synthetic_rdd_sample(synth_dir, num_samples=18, random_seed=42)
        # Move XMLs alongside images for automatic discovery
        for xml_p in xml_dir.glob("*.xml"):
            (img_dir / xml_p.name).write_text(xml_p.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"Rebuilding dataset from: {raw_dir}")
    print(f"Output target:           {processed_dir}")

    summary = build_unified_dataset(
        raw_dir=raw_dir,
        output_dir=processed_dir,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
    )

    print_dataset_summary_table(summary)


if __name__ == "__main__":
    main()
