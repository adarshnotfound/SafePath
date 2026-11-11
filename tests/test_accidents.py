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
)
from accidents.data_loader import (
    AccidentPoint,
    generate_empirical_accident_dataset,
)
from accidents.risk_model import (
    AccidentRiskModelPipeline,
    generate_segment_risk_dataset,
    run_accident_risk_pipeline,
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

    def test_segment_risk_feature_generation(self):
        """Verify segment feature extraction with hotspot distance and density."""
        hotspots = [
            AccidentHotspot("spot_001", 12.9718, 77.5948, 45.0, 14, 3.8, "seg_101", "India"),
            AccidentHotspot("spot_002", 12.9734, 77.5963, 60.0, 8, 2.5, "seg_102", "India"),
        ]
        dataset = generate_segment_risk_dataset(hotspots, samples_per_segment=10, seed=42)
        self.assertEqual(len(dataset), 80)  # 8 segments * 10 samples
        for f in dataset:
            self.assertTrue(f.segment_id.startswith("seg_"))
            self.assertTrue(0.0 <= f.continuous_risk_target <= 1.0)
            self.assertIn(f.binary_risk_target, (0, 1))
            self.assertGreaterEqual(f.dist_to_nearest_hotspot_m, 0.0)
            self.assertEqual(len(f.to_feature_array()), 9)

    def test_risk_model_training_and_comparison(self):
        """Verify MLP vs. Logistic Regression training, ROC-AUC, and continuous risk R^2."""
        hotspots = [
            AccidentHotspot("spot_001", 12.9718, 77.5948, 45.0, 14, 3.8, "seg_101", "India"),
            AccidentHotspot("spot_002", 12.9734, 77.5963, 60.0, 8, 2.5, "seg_102", "India"),
            AccidentHotspot("spot_003", 12.9751, 77.5981, 35.0, 19, 4.2, "seg_103", "India"),
        ]
        dataset = generate_segment_risk_dataset(hotspots, samples_per_segment=25, seed=42)
        pipeline = AccidentRiskModelPipeline(random_seed=42)
        report = pipeline.train_and_evaluate(dataset)

        mlp_clf = report["models"]["mlp_neural_network"]["classification"]
        logreg_clf = report["models"]["logistic_regression_baseline"]["classification"]
        mlp_reg = report["models"]["mlp_neural_network"]["continuous_risk_regression"]

        # Verification: high classification accuracy & AUC
        self.assertGreaterEqual(mlp_clf["accuracy"], 0.85)
        self.assertGreaterEqual(mlp_clf["roc_auc"], 0.90)
        self.assertGreaterEqual(logreg_clf["accuracy"], 0.85)
        self.assertGreaterEqual(logreg_clf["roc_auc"], 0.90)

        # Verification: non-linear continuous regression performance
        self.assertLess(mlp_reg["mse"], 0.01)
        self.assertGreater(mlp_reg["r2"], 0.80)

        # Single segment prediction test
        pred = pipeline.predict_segment_risk(dataset[0])
        self.assertIn("accident_score", pred)
        self.assertIn("is_high_risk", pred)
        self.assertTrue(0.0 <= pred["accident_score"] <= 1.0)

    def test_risk_model_pipeline_artifacts(self):
        """Verify full accident risk pipeline execution, JSON manifest, and ROC plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metrics_path = Path(tmpdir) / "risk_metrics.json"
            plot_path = Path(tmpdir) / "roc_plot.png"

            _, report = run_accident_risk_pipeline(
                output_metrics_path=metrics_path,
                output_plot_path=plot_path,
                seed=42,
            )

            self.assertTrue(metrics_path.exists())
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 1000)

            with open(metrics_path, mode="r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("mlp_neural_network", data["models"])
            self.assertIn("logistic_regression_baseline", data["models"])
            self.assertIn("roc_curves", data)


if __name__ == "__main__":
    unittest.main()

