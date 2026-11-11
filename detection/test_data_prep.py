"""Unit tests for SafePath RDD2022 dataset preparation pipeline (Module 1).

Verifies:
- Pascal VOC bounding box to YOLO coordinate normalization
- XML label parsing and mapping to SafePath classes (pothole, speed_breaker, animal)
- Dataset train/val splitting reproducibility
- YOLO dataset YAML configuration generation
"""

import tempfile
import unittest
from pathlib import Path

from detection.data_prep import (
    CLASS_TO_IDX,
    convert_voc_bbox_to_yolo,
    create_synthetic_rdd_sample,
    generate_yolo_dataset_yaml,
    parse_voc_xml,
    split_and_convert_dataset,
)


class TestDataPrepPipeline(unittest.TestCase):
    def test_voc_to_yolo_normalization(self):
        # 1280x720 image, box at [100, 100, 300, 200]
        # x_center = 200 / 1280 = 0.15625
        # y_center = 150 / 720 = 0.208333
        # width = 200 / 1280 = 0.15625
        # height = 100 / 720 = 0.138889
        size = (1280, 720)
        box = (100.0, 100.0, 300.0, 200.0)
        xc, yc, w, h = convert_voc_bbox_to_yolo(size, box)

        self.assertAlmostEqual(xc, 0.15625, places=4)
        self.assertAlmostEqual(yc, 0.20833, places=4)
        self.assertAlmostEqual(w, 0.15625, places=4)
        self.assertAlmostEqual(h, 0.13889, places=4)

        # Coordinate clipping check
        clipped_box = (-20.0, -10.0, 1400.0, 800.0)
        xc2, yc2, w2, h2 = convert_voc_bbox_to_yolo(size, clipped_box)
        self.assertTrue(0.0 <= xc2 <= 1.0)
        self.assertTrue(0.0 <= yc2 <= 1.0)
        self.assertTrue(0.0 <= w2 <= 1.0)
        self.assertTrue(0.0 <= h2 <= 1.0)

    def test_xml_parsing_and_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            xml_path = Path(tmpdir) / "test.xml"
            xml_content = """<annotation>
    <size>
        <width>1280</width>
        <height>720</height>
    </size>
    <object>
        <name>D40</name>
        <bndbox>
            <xmin>150</xmin>
            <ymin>200</ymin>
            <xmax>350</xmax>
            <ymax>400</ymax>
        </bndbox>
    </object>
    <object>
        <name>speed_bump</name>
        <bndbox>
            <xmin>100</xmin>
            <ymin>500</ymin>
            <xmax>800</xmax>
            <ymax>600</ymax>
        </bndbox>
    </object>
    <object>
        <name>D00</name> <!-- Linear crack should be filtered out -->
        <bndbox>
            <xmin>10</xmin>
            <ymin>10</ymin>
            <xmax>50</xmax>
            <ymax>50</ymax>
        </bndbox>
    </object>
</annotation>"""
            xml_path.write_text(xml_content, encoding="utf-8")

            parsed = parse_voc_xml(xml_path)
            self.assertEqual(len(parsed), 2)  # D00 filtered out

            classes_found = [p[0] for p in parsed]
            self.assertIn("pothole", classes_found)
            self.assertIn("speed_breaker", classes_found)

    def test_dataset_split_and_conversion_pipeline(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Create synthetic raw dataset
            raw_img_dir, raw_xml_dir = create_synthetic_rdd_sample(
                output_dir=tmp_path / "raw",
                num_samples=10,
                random_seed=42,
            )

            # Convert to YOLO format
            yolo_out = tmp_path / "yolo_dataset"
            summary = split_and_convert_dataset(
                raw_images_dir=raw_img_dir,
                raw_xml_dir=raw_xml_dir,
                output_yolo_dir=yolo_out,
                val_split_ratio=0.2,
                random_seed=42,
            )

            self.assertEqual(summary["total_paired"], 10)
            self.assertEqual(summary["val_samples"], 2)
            self.assertEqual(summary["train_samples"], 8)

            # Check that files exist in appropriate directories
            train_labels = list((yolo_out / "labels" / "train").glob("*.txt"))
            val_labels = list((yolo_out / "labels" / "val").glob("*.txt"))
            self.assertEqual(len(train_labels), 8)
            self.assertEqual(len(val_labels), 2)

            # Inspect label content
            sample_label_text = train_labels[0].read_text(encoding="utf-8").strip()
            parts = sample_label_text.split()
            self.assertIn(int(parts[0]), CLASS_TO_IDX.values())

    def test_yaml_config_generation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test_dataset.yaml"
            generate_yolo_dataset_yaml(
                output_yaml_path=yaml_path,
                dataset_root_dir=Path(tmpdir) / "yolo",
            )
            self.assertTrue(yaml_path.exists())
            content = yaml_path.read_text(encoding="utf-8")
            self.assertIn("train: images/train", content)
            self.assertIn("nc: 3", content)
            self.assertIn("names: ['pothole', 'speed_breaker', 'animal']", content)


if __name__ == "__main__":
    unittest.main()
