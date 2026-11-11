"""Unit tests for SafePath Module B3: Accident Hotspot Clustering and Comparison."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from accidents.clustering import (
    AccidentHotspot,
    compare_and_export_hotspots,
    parameter_sweep_dbscan,
    parameter_sweep_fcm,
    project_coordinates_to_xy,
    run_dbscan,
    run_fcm,
)
from accidents.data_loader import (
    AccidentPoint,
    generate_empirical_accident_dataset,
    load_accidents_from_csv,
    save_accidents_to_csv,
)
from accidents.visualization import plot_accident_hotspots_folium


class TestAccidentModule(unittest.TestCase):
    """Test suite for data loading, DBSCAN vs FCM clustering, contracts, and maps."""

    def test_data_generation_and_provenance_labels(self):
        """Verify empirical accident generation and provenance tracking (India vs. substitute)."""
        # Test Indian native dataset
        accidents_in = generate_empirical_accident_dataset(seed=42, source_country="India")
        self.assertGreaterEqual(len(accidents_in), 70)
        for a in accidents_in:
            self.assertEqual(a.source_country, "India")
            self.assertTrue(1.0 <= a.severity <= 5.0)
            self.assertTrue(12.96 <= a.lat <= 12.99)
            self.assertTrue(77.58 <= a.lon <= 77.62)

        # Test substitute non-Indian dataset (UK STATS19 fallback per Section 12)
        accidents_uk = generate_empirical_accident_dataset(seed=42, source_country="United Kingdom")
        for a in accidents_uk:
            self.assertEqual(a.source_country, "United Kingdom")

    def test_dbscan_clustering_and_parameter_sweep(self):
        """Verify DBSCAN clustering, noise exclusion, and silhouette score optimization."""
        accidents = generate_empirical_accident_dataset(seed=42, source_country="India")
        lats = np.array([a.lat for a in accidents])
        lons = np.array([a.lon for a in accidents])
        points_xy = project_coordinates_to_xy(lats, lons, float(np.mean(lats)), float(np.mean(lons)))

        sweep_results = parameter_sweep_dbscan(points_xy)
        self.assertGreaterEqual(len(sweep_results), 5)

        best_cfg = sweep_results[0]
        self.assertIn("eps_meters", best_cfg)
        self.assertIn("min_samples", best_cfg)
        self.assertGreaterEqual(best_cfg["num_clusters"], 4)
        self.assertLessEqual(best_cfg["num_clusters"], 8)
        self.assertIsNotNone(best_cfg["silhouette_score"])
        self.assertGreater(best_cfg["silhouette_score"], 0.70)
        self.assertGreater(best_cfg["noise_count"], 0)

    def test_fuzzy_c_means_clustering_and_parameter_sweep(self):
        """Verify Fuzzy C-Means (scikit-fuzzy) clustering, FPC, and silhouette computation."""
        accidents = generate_empirical_accident_dataset(seed=42, source_country="India")
        lats = np.array([a.lat for a in accidents])
        lons = np.array([a.lon for a in accidents])
        points_xy = project_coordinates_to_xy(lats, lons, float(np.mean(lats)), float(np.mean(lons)))

        sweep_results = parameter_sweep_fcm(points_xy, seed=42)
        self.assertGreaterEqual(len(sweep_results), 5)

        best_cfg = sweep_results[0]
        self.assertIn("c_clusters", best_cfg)
        self.assertIn("m_fuzziness", best_cfg)
        self.assertGreater(best_cfg["fpc"], 0.80)
        self.assertIsNotNone(best_cfg["silhouette_score"])
        self.assertGreater(best_cfg["silhouette_score"], 0.65)

    def test_accident_hotspots_csv_contract_compliance(self):
        """Ensure produced accident_hotspots.csv strictly matches Section 8 contract."""
        with tempfile.TemporaryDirectory() as tmpdir:
            acc_path = Path(tmpdir) / "accidents_raw.csv"
            hotspot_path = Path(tmpdir) / "accident_hotspots.csv"
            metrics_path = Path(tmpdir) / "comparison.json"

            hotspots, report = compare_and_export_hotspots(
                accidents_csv_path=acc_path,
                output_hotspots_csv_path=hotspot_path,
                output_metrics_json_path=metrics_path,
                source_country="India",
                seed=42,
            )

            self.assertTrue(hotspot_path.exists())
            self.assertTrue(metrics_path.exists())

            # Section 8 file contract columns
            expected_cols = [
                "hotspot_id",
                "lat",
                "lon",
                "radius_m",
                "accident_count",
                "severity_weight",
                "segment_id",
                "source_country",
            ]

            with open(hotspot_path, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = [r for r in reader if r]

            self.assertEqual(header, expected_cols)
            # Must adhere to 5 <= rows <= 10 contract
            self.assertTrue(5 <= len(rows) <= 10)

            # Check individual value formats
            for row in rows:
                hid, lat, lon, rad, cnt, sev, seg, country = row
                self.assertTrue(hid.startswith("spot_"))
                self.assertGreater(float(lat), 12.0)
                self.assertGreater(float(lon), 77.0)
                self.assertGreater(float(rad), 10.0)
                self.assertGreaterEqual(int(cnt), 1)
                self.assertTrue(1.0 <= float(sev) <= 5.0)
                self.assertTrue(seg.startswith("seg_"))
                self.assertEqual(country, "India")

    def test_substitute_provenance_flagging(self):
        """Non-Indian data must be clearly flagged as substitute in metrics and reports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            acc_path = Path(tmpdir) / "accidents_uk.csv"
            hotspot_path = Path(tmpdir) / "hotspots_uk.csv"
            metrics_path = Path(tmpdir) / "comparison_uk.json"

            hotspots, report = compare_and_export_hotspots(
                accidents_csv_path=acc_path,
                output_hotspots_csv_path=hotspot_path,
                output_metrics_json_path=metrics_path,
                source_country="United Kingdom",
                seed=42,
            )

            self.assertFalse(report["dataset_metadata"]["is_indian_data"])
            self.assertEqual(report["dataset_metadata"]["provenance"], "substitute")
            self.assertEqual(report["dataset_metadata"]["source_country"], "United Kingdom")

    def test_interactive_folium_map_generation(self):
        """Verify Folium map generation and HTML output validity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            map_path = Path(tmpdir) / "test_map.html"
            hotspots = [
                AccidentHotspot("spot_001", 12.9718, 77.5948, 45.0, 14, 3.8, "seg_101", "India"),
                AccidentHotspot("spot_002", 12.9734, 77.5963, 60.0, 8, 2.5, "seg_102", "India"),
            ]
            accidents = [
                AccidentPoint("acc_001", 12.9718, 77.5948, 4.0, "2025-01-01T10:00:00Z", "seg_101", "clear", "daylight", "India"),
            ]

            folium_map = plot_accident_hotspots_folium(
                hotspots=hotspots,
                accidents=accidents,
                output_html_path=map_path,
                source_country="India",
            )

            self.assertTrue(map_path.exists())
            self.assertGreater(map_path.stat().st_size, 1000)

            content = map_path.read_text(encoding="utf-8")
            self.assertIn("folium", content.lower())
            self.assertIn("spot_001", content)
            self.assertIn("Native Indian Data", content)


if __name__ == "__main__":
    unittest.main()
