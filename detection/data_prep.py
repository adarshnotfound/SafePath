"""Dataset preparation and annotation converter for SafePath (Module 1).

Follows Section 4.1 & Section 5 of the SafePath Specification:
- Converts Indian road damage datasets (RDD2022 India subset / VOC XML annotations)
  into Ultralytics YOLOv8 format: `<class_id> <x_center> <y_center> <width> <height>`.
- Maps external labels to SafePath classes:
    0: pothole (mapped from RDD2022 D40 / Pothole)
    1: speed_breaker (mapped from speed bumps / road bumps)
    2: animal (mapped from stray cattle / dogs)
- Generates train/validation splits with fixed random seed (Rule 4).
- Generates `data/rdd2022_safepath.yaml` dataset configuration.
"""

import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# SafePath Class Taxonomy (Section 4.1)
SAFEPATH_CLASSES = ["pothole", "speed_breaker", "animal"]
CLASS_TO_IDX = {name: i for i, name in enumerate(SAFEPATH_CLASSES)}

# Label mapping from RDD2022 (Japan/India/Czech CRDDC challenge) and VOC datasets
# In RDD2022: D40 represents Potholes (Rutting, bump, pothole group)
LABEL_MAPPING = {
    # RDD2022 codes
    "d40": "pothole",
    "pothole": "pothole",
    "potholes": "pothole",
    # Road bumps / speed breakers
    "speed_breaker": "speed_breaker",
    "speed_bump": "speed_breaker",
    "bump": "speed_breaker",
    "speedbreaker": "speed_breaker",
    # Animals
    "animal": "animal",
    "cow": "animal",
    "dog": "animal",
    "horse": "animal",
    "buffalo": "animal",
    "goat": "animal",
}


def convert_voc_bbox_to_yolo(
    size: Tuple[int, int],
    box: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Convert Pascal VOC bounding box [xmin, ymin, xmax, ymax] to normalized YOLO coordinates.

    YOLO format:
    x_center, y_center, width, height (all normalized in [0.0, 1.0]).
    """
    img_w, img_h = size
    xmin, ymin, xmax, ymax = box

    # Clip to image boundaries
    xmin = max(0.0, min(float(img_w), xmin))
    xmax = max(0.0, min(float(img_w), xmax))
    ymin = max(0.0, min(float(img_h), ymin))
    ymax = max(0.0, min(float(img_h), ymax))

    box_w = xmax - xmin
    box_h = ymax - ymin

    x_center = xmin + (box_w / 2.0)
    y_center = ymin + (box_h / 2.0)

    # Normalize relative to image dimensions
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


def parse_voc_xml(xml_path: Path) -> List[Tuple[str, Tuple[float, float, float, float]]]:
    """Parse Pascal VOC format XML annotation and extract target hazard objects.

    Returns
    -------
    List[Tuple[str, Tuple[float, float, float, float]]]
        List of (mapped_class_name, (xmin, ymin, xmax, ymax))
    """
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

        # Skip classes not relevant to SafePath core hazards (e.g. general longitudinal cracks)
        if mapped_class is None:
            continue

        bndbox = obj.find("bndbox")
        if bndbox is not None:
            xmin = float(bndbox.findtext("xmin", "0"))
            ymin = float(bndbox.findtext("ymin", "0"))
            xmax = float(bndbox.findtext("xmax", "0"))
            ymax = float(bndbox.findtext("ymax", "0"))

            # If image size was 0 in xml, approximate from coordinates
            if width == 0 or height == 0:
                width = int(max(width, xmax))
                height = int(max(height, ymax))

            annotations.append((mapped_class, (xmin, ymin, xmax, ymax), (width, height)))

    return annotations


def generate_yolo_dataset_yaml(
    output_yaml_path: Path,
    dataset_root_dir: Path,
    classes: Optional[List[str]] = None,
) -> Path:
    """Generate an Ultralytics YOLOv8 compatible dataset YAML configuration file.

    Parameters
    ----------
    output_yaml_path : Path
        Where to save the dataset configuration file (e.g. data/rdd2022_safepath.yaml).
    dataset_root_dir : Path
        Base directory containing images/train and images/val.
    classes : Optional[List[str]]
        Class names list. Defaults to SAFEPATH_CLASSES.

    Returns
    -------
    Path
        Path to the written YAML file.
    """
    if classes is None:
        classes = SAFEPATH_CLASSES

    output_yaml_path = Path(output_yaml_path)
    output_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    yaml_content = f"""# SafePath Dataset Configuration for YOLOv8
# Target classes: potholes, speed breakers, animals
path: {dataset_root_dir.resolve().as_posix()}
train: images/train
val: images/val
test: images/val

# Classes
nc: {len(classes)}
names: {classes}
"""
    output_yaml_path.write_text(yaml_content, encoding="utf-8")
    return output_yaml_path


def split_and_convert_dataset(
    raw_images_dir: Path,
    raw_xml_dir: Path,
    output_yolo_dir: Path,
    val_split_ratio: float = 0.2,
    random_seed: int = 42,
) -> Dict[str, int]:
    """Split raw image/xml pairs into YOLO format directories (images/train, labels/train, etc.).

    Parameters
    ----------
    raw_images_dir : Path
        Directory containing input images (.jpg / .png).
    raw_xml_dir : Path
        Directory containing Pascal VOC XML annotations.
    output_yolo_dir : Path
        Root directory for the output YOLO dataset.
    val_split_ratio : float
        Fraction of samples for validation (default 0.2).
    random_seed : int
        Fixed seed for reproducible dataset partitioning (Rule 4).

    Returns
    -------
    dict
        Summary count of processed items {'train': count, 'val': count}.
    """
    raw_images_dir = Path(raw_images_dir)
    raw_xml_dir = Path(raw_xml_dir)
    output_yolo_dir = Path(output_yolo_dir)

    train_img_dir = output_yolo_dir / "images" / "train"
    val_img_dir = output_yolo_dir / "images" / "val"
    train_lbl_dir = output_yolo_dir / "labels" / "train"
    val_lbl_dir = output_yolo_dir / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect matched image and xml files
    xml_files = sorted(list(raw_xml_dir.glob("*.xml")))
    paired_samples = []

    for xml_p in xml_files:
        stem = xml_p.stem
        img_match = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = raw_images_dir / f"{stem}{ext}"
            if candidate.exists():
                img_match = candidate
                break
        if img_match:
            paired_samples.append((img_match, xml_p))

    # Shuffle deterministically
    rng = random.Random(random_seed)
    rng.shuffle(paired_samples)

    num_val = int(len(paired_samples) * val_split_ratio)
    val_samples = paired_samples[:num_val]
    train_samples = paired_samples[num_val:]

    def _process_split(samples, img_dest, lbl_dest):
        count = 0
        for img_p, xml_p in samples:
            parsed = parse_voc_xml(xml_p)
            lines = []
            for class_name, bbox, img_size in parsed:
                class_id = CLASS_TO_IDX[class_name]
                yolo_box = convert_voc_bbox_to_yolo(img_size, bbox)
                lines.append(f"{class_id} {yolo_box[0]} {yolo_box[1]} {yolo_box[2]} {yolo_box[3]}")

            # Write label file (.txt)
            txt_dest = lbl_dest / f"{img_p.stem}.txt"
            txt_dest.write_text("\n".join(lines), encoding="utf-8")

            # Write or copy image file
            target_img = img_dest / img_p.name
            target_img.write_bytes(img_p.read_bytes())
            count += 1
        return count

    train_count = _process_split(train_samples, train_img_dir, train_lbl_dir)
    val_count = _process_split(val_samples, val_img_dir, val_lbl_dir)

    return {
        "total_paired": len(paired_samples),
        "train_samples": train_count,
        "val_samples": val_count,
    }


def create_synthetic_rdd_sample(
    output_dir: Path,
    num_samples: int = 10,
    random_seed: int = 42,
) -> Tuple[Path, Path]:
    """Generate minimal synthetic Pascal VOC annotations and placeholder JPEG images.

    Allows unit testing, demonstration, and CI execution without needing a 20GB download.
    """
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
        stem = f"India_{i:04d}"
        img_path = images_dir / f"{stem}.jpg"
        img_path.write_bytes(dummy_jpeg_bytes)

        cls_choice = rng.choice(sample_classes)
        xml_content = f"""<annotation>
    <folder>India</folder>
    <filename>{stem}.jpg</filename>
    <size>
        <width>1280</width>
        <height>720</height>
        <depth>3</depth>
    </size>
    <object>
        <name>{cls_choice}</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
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
