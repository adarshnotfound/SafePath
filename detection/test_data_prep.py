"""Unit tests for SafePath unified dataset preparation pipeline (Module 1).

Verifies:
- Pascal VOC and YOLO txt label parsing into 3 unified classes (pothole, speed_breaker, animal)
- Corrupt/broken file detection and pruning
- SHA-256 deduplication and leakage prevention
- 3-way split (train/val/test) reproducibility with seed 42
- YOLO dataset YAML configuration generation
"""

import tempfile
import unittest
from pathlib import Path

from detection.data_prep import (
    build_unified_dataset,
    compute_image_hash,
    convert_voc_bbox_to_yolo,
    create_synthetic_rdd_sample,
    generate_yolo_dataset_yaml,
    is_image_valid_and_not_broken,
    parse_voc_xml,
    parse_yolo_txt,
)


class TestDataPrepPipeline(unittest.TestCase):
    def test_voc_to_yolo_normalization(self):
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

    def test_yolo_txt_parsing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir) / "test_roboflow.txt"
            txt_path.write_text("0 0.5 0.5 0.2 0.1\n1 0.4 0.6 0.3 0.15\n", encoding="utf-8")
            source_classes = ["pothole", "speed_breaker"]
            parsed = parse_yolo_txt(txt_path, source_class_names=source_classes)

            self.assertEqual(len(parsed), 2)
            self.assertEqual(parsed[0][0], "pothole")
            self.assertEqual(parsed[1][0], "speed_breaker")

    def test_corrupt_file_and_duplicate_filtering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Create a 0-byte corrupt file
            broken_file = tmp_path / "broken.jpg"
            broken_file.write_bytes(b"")
            is_valid, _ = is_image_valid_and_not_broken(broken_file)
            self.assertFalse(is_valid)

            # Test hash equality for duplicates
            file1 = tmp_path / "img1.jpg"
            file2 = tmp_path / "img2.jpg"
            file1.write_bytes(b"image_content_bytes_123")
            file2.write_bytes(b"image_content_bytes_123")
            self.assertEqual(compute_image_hash(file1), compute_image_hash(file2))

    def test_3_way_dataset_split_and_rebuild(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_dir = tmp_path / "raw"
            img_dir, xml_dir = create_synthetic_rdd_sample(
                output_dir=raw_dir,
                num_samples=20,
                random_seed=42,
            )
            # Place xml files alongside images
            for x in xml_dir.glob("*.xml"):
                (img_dir / x.name).write_text(x.read_text(encoding="utf-8"), encoding="utf-8")

            # Add an intentional duplicate image to verify deduplication
            dup_img = img_dir / "duplicate_test.jpg"
            first_img = list(img_dir.glob("sample_*.jpg"))[0]
            dup_img.write_bytes(first_img.read_bytes())
            dup_xml = img_dir / "duplicate_test.xml"
            dup_xml.write_text((img_dir / f"{first_img.stem}.xml").read_text(encoding="utf-8"), encoding="utf-8")

            out_dir = tmp_path / "processed" / "yolo"
            summary = build_unified_dataset(
                raw_dir=img_dir,
                output_dir=out_dir,
                train_ratio=0.70,
                val_ratio=0.15,
                test_ratio=0.15,
                random_seed=42,
            )

            # Verification
            self.assertEqual(summary["duplicates_removed"], 1)
            self.assertEqual(summary["clean_samples"], 20)
            self.assertEqual(summary["splits"]["train"], 14)
            self.assertEqual(summary["splits"]["val"], 3)
            self.assertEqual(summary["splits"]["test"], 3)

            # Check that files exist in all 3 splits
            for split in ["train", "val", "test"]:
                lbls = list((out_dir / "labels" / split).glob("*.txt"))
                imgs = list((out_dir / "images" / split).glob("*.jpg"))
                self.assertGreater(len(lbls), 0)
                self.assertEqual(len(lbls), len(imgs))

            # Check class instance counts
            counts = summary["class_instance_counts"]
            self.assertGreater(counts["total"]["pothole"] + counts["total"]["speed_breaker"] + counts["total"]["animal"], 0)

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
            self.assertIn("val: images/val", content)
            self.assertIn("test: images/test", content)
            self.assertIn("nc: 3", content)
            self.assertIn("names: ['pothole', 'speed_breaker', 'animal']", content)


if __name__ == "__main__":
    unittest.main()
