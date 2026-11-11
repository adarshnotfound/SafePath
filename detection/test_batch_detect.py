"""Unit tests for lighting classification with custom thresholds and batch YOLO folder detection.

Verifies:
- Frame classification into 'well_lit', 'dim', or 'dark' using custom thresholds
- Automatic threshold calibration from labeled frames
- Batch YOLO detection on a folder of 3 sample images
- Output detections.csv exact 10-column header match
- Empty lat, lon, and timestamp fields prior to GPS placement
"""

import csv
import tempfile
import unittest
from pathlib import Path

import cv2

from detection.batch_detect import EXACT_CSV_COLUMNS, run_batch_detection_on_folder
from detection.lighting import (
    calibrate_lighting_thresholds,
    classify_lighting,
    compute_frame_luminance,
)
from detection.make_sample_images import create_three_sample_images

SAMPLE_IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "sample" / "test_images"


class TestLightingAndBatchDetection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Ensure 3 sample images exist
        cls.image_paths = create_three_sample_images(SAMPLE_IMAGES_DIR)

    def test_classify_lighting_on_three_sample_images(self):
        img_day = cv2.imread(str(self.image_paths[0]))
        img_dusk = cv2.imread(str(self.image_paths[1]))
        img_night = cv2.imread(str(self.image_paths[2]))

        # Verify mean luminance ordering: day > dusk > night
        lum_day = compute_frame_luminance(img_day)
        lum_dusk = compute_frame_luminance(img_dusk)
        lum_night = compute_frame_luminance(img_night)
        self.assertGreater(lum_day, lum_dusk)
        self.assertGreater(lum_dusk, lum_night)

        # 1. Test with default thresholds (50.0, 120.0)
        label_day, _ = classify_lighting(img_day)
        label_dusk, _ = classify_lighting(img_dusk)
        label_night, _ = classify_lighting(img_night)

        self.assertEqual(label_day, "well_lit")
        self.assertEqual(label_dusk, "dim")
        self.assertEqual(label_night, "dark")

        # 2. Test with custom user-chosen thresholds
        custom_dark = 40.0
        custom_dim = 110.0
        c_label_day, _ = classify_lighting(img_day, dark_threshold=custom_dark, dim_threshold=custom_dim)
        c_label_dusk, _ = classify_lighting(img_dusk, dark_threshold=custom_dark, dim_threshold=custom_dim)
        c_label_night, _ = classify_lighting(img_night, dark_threshold=custom_dark, dim_threshold=custom_dim)

        self.assertEqual(c_label_day, "well_lit")
        self.assertEqual(c_label_dusk, "dim")
        self.assertEqual(c_label_night, "dark")

    def test_threshold_calibration_from_labeled_set(self):
        img_day = cv2.imread(str(self.image_paths[0]))
        img_dusk = cv2.imread(str(self.image_paths[1]))
        img_night = cv2.imread(str(self.image_paths[2]))

        labeled_set = [
            (img_night, "dark"),
            (img_dusk, "dim"),
            (img_day, "well_lit"),
        ]

        rec_dark, rec_dim = calibrate_lighting_thresholds(labeled_set)
        # Verify recommended dark separates night and dusk, and rec_dim separates dusk and day
        lum_night = compute_frame_luminance(img_night)
        lum_dusk = compute_frame_luminance(img_dusk)
        lum_day = compute_frame_luminance(img_day)

        self.assertTrue(lum_night < rec_dark < lum_dusk)
        self.assertTrue(lum_dusk < rec_dim < lum_day)

    def test_batch_detection_on_three_images_and_csv_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "detections.csv"

            # Run batch detector over folder of 3 sample images
            rows = run_batch_detection_on_folder(
                image_dir=SAMPLE_IMAGES_DIR,
                output_csv_path=csv_path,
                drive_id="drive_test_01",
                confidence_threshold=0.3,
                random_seed=42,
            )

            self.assertTrue(csv_path.exists())
            self.assertGreater(len(rows), 0)

            # Inspect written CSV file
            with open(csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                csv_records = [r for r in reader if r]

            # 1. Exact 10-column header match
            self.assertEqual(header, EXACT_CSV_COLUMNS)

            # 2. Check every row adheres to contract
            for r in csv_records:
                self.assertEqual(len(r), 10)
                det_id, drv_id, frame_id, cls, conf, bbox, lat, lon, timestamp, is_sim = r

                self.assertTrue(det_id.startswith("det_"))
                self.assertEqual(drv_id, "drive_test_01")
                self.assertIn(frame_id, ["sample_01_daylight", "sample_02_dusk", "sample_03_night"])
                self.assertIn(cls, ["pothole", "speed_breaker", "animal"])
                self.assertTrue(0.0 <= float(conf) <= 1.0)

                # Bounding box format: [ymin, xmin, ymax, xmax]
                self.assertTrue(bbox.startswith("[") and bbox.endswith("]"))

                # Strict requirement: lat, lon, timestamp left EMPTY
                self.assertEqual(lat, "")
                self.assertEqual(lon, "")
                self.assertEqual(timestamp, "")

                self.assertIn(is_sim.lower(), ["true", "1"])


if __name__ == "__main__":
    unittest.main()
