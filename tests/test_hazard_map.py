"""Unit and sanity tests for SafePath Module B1 & B2: Hazard Map and Living Map Update."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from hazard_map.evaluate import run_pipeline_and_evaluate
from hazard_map.ga_optimizer import (
    evaluate_scenario,
    individual_to_config,
    plot_ga_convergence_curve,
    run_genetic_algorithm,
)
from hazard_map.living_map import (
    DetectionItem,
    LivingHazardMap,
    LivingMapConfig,
    merge_detections_dbscan,
)
from hazard_map.simulated_drive import (
    add_gps_noise,
    generate_ground_truth_hazards,
    is_hazard_present_on_drive,
    run_generator,
    simulate_multi_drive_detections,
)


class TestSimulatedDriveGenerator(unittest.TestCase):
    """Test suite verifying Module B1 functionality, contracts, and reproducibility."""

    def test_reproducibility_with_seed(self):
        """Detections generated with the same seed should be bit-for-bit identical."""
        hazards = generate_ground_truth_hazards(seed=42)
        dets1, meta1 = simulate_multi_drive_detections(hazards, seed=42)
        dets2, meta2 = simulate_multi_drive_detections(hazards, seed=42)

        self.assertEqual(len(dets1), len(dets2))
        self.assertEqual(meta1, meta2)
        for d1, d2 in zip(dets1, dets2):
            self.assertEqual(d1.detection_id, d2.detection_id)
            self.assertEqual(d1.lat, d2.lat)
            self.assertEqual(d1.lon, d2.lon)
            self.assertEqual(d1.confidence, d2.confidence)

    def test_gps_noise_within_bounds(self):
        """GPS noise should remain realistic (approx 5-25m maximum for 99.7% of draws)."""
        base_lat, base_lon = 12.9716, 77.5946
        diffs_meters = []
        for _ in range(100):
            n_lat, n_lon = add_gps_noise(base_lat, base_lon, std_meters=8.0)
            d_lat = (n_lat - base_lat) * 111320.0
            d_lon = (n_lon - base_lon) * (111320.0 * 0.9745)
            dist = (d_lat**2 + d_lon**2) ** 0.5
            diffs_meters.append(dist)

        avg_dist = sum(diffs_meters) / len(diffs_meters)
        self.assertGreater(avg_dist, 3.0)
        self.assertLess(avg_dist, 20.0)

    def test_hazard_lifecycle(self):
        """Transient hazards (animals) or repaired hazards should not appear after repair/expiry."""
        hazards = generate_ground_truth_hazards(seed=42)
        animal_haz = next(h for h in hazards if h.hazard_id == "gt_haz_003")
        self.assertTrue(is_hazard_present_on_drive(animal_haz, "drive_01"))
        self.assertFalse(is_hazard_present_on_drive(animal_haz, "drive_02"))
        self.assertFalse(is_hazard_present_on_drive(animal_haz, "drive_03"))

        repaired_pothole = next(h for h in hazards if h.hazard_id == "gt_haz_004")
        self.assertTrue(is_hazard_present_on_drive(repaired_pothole, "drive_01"))
        self.assertTrue(is_hazard_present_on_drive(repaired_pothole, "drive_02"))
        self.assertFalse(is_hazard_present_on_drive(repaired_pothole, "drive_03"))

    def test_detections_csv_contract_compliance(self):
        """Ensure produced detections CSV exactly matches the Section 8 contract."""
        with tempfile.TemporaryDirectory() as tmpdir:
            det_path = Path(tmpdir) / "detections.csv"
            gt_path = Path(tmpdir) / "ground_truth_hazards.csv"
            meta_path = Path(tmpdir) / "metrics.json"

            run_generator(
                output_detections_path=det_path,
                output_gt_path=gt_path,
                metrics_path=meta_path,
                seed=42,
            )

            self.assertTrue(det_path.exists())
            self.assertTrue(gt_path.exists())
            self.assertTrue(meta_path.exists())

            expected_cols = [
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
            with open(det_path, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)

            self.assertEqual(header, expected_cols)
            self.assertGreaterEqual(len(rows), 10)

            with open(meta_path, mode="r", encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(meta["random_seed"], 42)
            self.assertIn("total_detections_generated", meta)


class TestLivingHazardMap(unittest.TestCase):
    """Test suite verifying Module B2 functionality, clustering, decay, and contracts."""

    def test_dbscan_clustering_same_class_within_7m(self):
        """Detections of same class within 7m merge; different classes or >7m remain separated."""
        dets = [
            DetectionItem("d1", "drive_01", "pothole", 0.90, 12.97160, 77.59460, "2026-11-01T10:00:00Z"),
            # ~4.4m north of d1: should merge with d1
            DetectionItem("d2", "drive_01", "pothole", 0.85, 12.97164, 77.59460, "2026-11-01T10:00:02Z"),
            # ~22m north of d1: should form a separate cluster (>7m)
            DetectionItem("d3", "drive_01", "pothole", 0.80, 12.97180, 77.59460, "2026-11-01T10:00:05Z"),
            # ~2.2m from d1, but DIFFERENT class: must never merge into pothole cluster
            DetectionItem("d4", "drive_01", "speed_breaker", 0.92, 12.97162, 77.59460, "2026-11-01T10:00:08Z"),
        ]

        clusters = merge_detections_dbscan(dets, eps_meters=7.0, min_samples=1)
        self.assertEqual(len(clusters), 3)

        pothole_clusters = [c for c in clusters if c["class_name"] == "pothole"]
        sb_clusters = [c for c in clusters if c["class_name"] == "speed_breaker"]

        self.assertEqual(len(pothole_clusters), 2)
        self.assertEqual(len(sb_clusters), 1)

        merged_pothole = next(c for c in pothole_clusters if c["observation_count"] == 2)
        self.assertIn("d1", merged_pothole["member_detection_ids"])
        self.assertIn("d2", merged_pothole["member_detection_ids"])

    def test_confidence_rise_on_redetection(self):
        """When a hazard is confirmed in subsequent drives, confidence should rise monotonically."""
        lmap = LivingHazardMap()
        d1 = [DetectionItem("d1", "drive_01", "pothole", 0.80, 12.97160, 77.59460, "2026-11-01T10:00:00Z")]
        lmap.update_with_drive("drive_01", d1)

        h = list(lmap.hazards.values())[0]
        initial_conf = h.confidence
        self.assertEqual(h.drive_count, 1)

        # Drive 2 re-detects the pothole at nearby location
        d2 = [DetectionItem("d2", "drive_02", "pothole", 0.85, 12.97162, 77.59461, "2026-11-02T10:00:00Z")]
        lmap.update_with_drive("drive_02", d2)

        self.assertEqual(h.drive_count, 2)
        self.assertGreater(h.confidence, initial_conf)
        self.assertEqual(h.consecutive_misses, 0)
        self.assertEqual(h.status, "active")

    def test_delayed_decay_k_misses(self):
        """Single misses do not decay when K=2; decay triggers only after K consecutive misses."""
        cfg = LivingMapConfig.load()
        cfg.default_miss_threshold_k = 2
        cfg.type_persistence["pothole"].miss_threshold_k = 2
        lmap = LivingHazardMap(config=cfg)

        # Initial sighting on seg_101
        d1 = [DetectionItem("d1", "drive_01", "pothole", 0.80, 12.97160, 77.59460, "2026-11-01T10:00:00Z")]
        lmap.update_with_drive("drive_01", d1, traversed_segments={"seg_101"})

        h = list(lmap.hazards.values())[0]
        base_conf = h.confidence

        # Drive 2 passes seg_101 with NO detections (miss #1 < K=2)
        lmap.update_with_drive("drive_02", [], traversed_segments={"seg_101"})
        self.assertEqual(h.consecutive_misses, 1)
        self.assertEqual(h.confidence, base_conf)  # No decay yet!

        # Drive 3 passes seg_101 with NO detections (miss #2 >= K=2) -> decay applied!
        lmap.update_with_drive("drive_03", [], traversed_segments={"seg_101"})
        self.assertEqual(h.consecutive_misses, 2)
        self.assertLess(h.confidence, base_conf)  # Confidence decayed!

    def test_type_specific_persistence(self):
        """Speed breakers decay very slowly, potholes moderately, animals rapidly."""
        cfg = LivingMapConfig.load()
        # Set all K=1 to test single decay step directly
        cfg.type_persistence["speed_breaker"].miss_threshold_k = 1
        cfg.type_persistence["pothole"].miss_threshold_k = 1
        cfg.type_persistence["animal"].miss_threshold_k = 1

        lmap = LivingHazardMap(config=cfg)
        d1 = [
            DetectionItem("d_sb", "drive_01", "speed_breaker", 0.80, 12.97210, 77.59510, "2026-11-01T10:00:00Z"),
            DetectionItem("d_ph", "drive_01", "pothole", 0.80, 12.97160, 77.59460, "2026-11-01T10:00:01Z"),
            DetectionItem("d_an", "drive_01", "animal", 0.80, 12.97305, 77.59600, "2026-11-01T10:00:02Z"),
        ]
        lmap.update_with_drive("drive_01", d1, traversed_segments={"seg_101", "seg_102"})

        sb = next(h for h in lmap.hazards.values() if h.type == "speed_breaker")
        ph = next(h for h in lmap.hazards.values() if h.type == "pothole")
        an = next(h for h in lmap.hazards.values() if h.type == "animal")

        # Drive 2 passes all segments but detects nothing
        lmap.update_with_drive("drive_02", [], traversed_segments={"seg_101", "seg_102"})

        # Animal confidence should drop the most, speed breaker the least
        sb_loss = 0.80 - sb.confidence
        ph_loss = 0.80 - ph.confidence
        an_loss = 0.80 - an.confidence

        self.assertLess(sb_loss, ph_loss)
        self.assertLess(ph_loss, an_loss)
        self.assertEqual(an.status, "decayed")

    def test_hazards_csv_contract_compliance(self):
        """Verify living map output hazards.csv adheres strictly to Section 8 schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_hazards = Path(tmpdir) / "hazards.csv"
            out_metrics = Path(tmpdir) / "metrics.json"

            lmap, metrics = run_pipeline_and_evaluate(
                output_hazards_csv_path=out_hazards,
                metrics_json_path=out_metrics,
            )

            self.assertTrue(out_hazards.exists())
            self.assertTrue(out_metrics.exists())

            expected_cols = [
                "hazard_id",
                "type",
                "lat",
                "lon",
                "first_seen",
                "last_seen",
                "observation_count",
                "drive_count",
                "confidence",
                "freshness",
                "status",
                "segment_id",
                "is_simulated",
            ]
            with open(out_hazards, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = [r for r in reader if r]

            self.assertEqual(header, expected_cols)
            self.assertGreaterEqual(len(rows), 5)

            # Check evaluation metrics
            self.assertGreaterEqual(metrics.duplicate_merge_accuracy, 0.80)
            self.assertGreaterEqual(metrics.repair_detection_rate, 0.80)
            self.assertEqual(metrics.false_decay_rate, 0.0)


class TestGADecayOptimizer(unittest.TestCase):
    """Test suite verifying DEAP Genetic Algorithm decay rate optimization."""

    def test_chromosome_mapping_and_bounds(self):
        """Verify chromosome mapping to LivingMapConfig with gene clamping and integer rounding."""
        ind = [0.05, 3.2, 0.45, 1.8, 0.80, 1.1]
        cfg = individual_to_config(ind)
        self.assertAlmostEqual(cfg.type_persistence["speed_breaker"].decay_rate, 0.05, places=2)
        self.assertEqual(cfg.type_persistence["speed_breaker"].miss_threshold_k, 3)
        self.assertAlmostEqual(cfg.type_persistence["pothole"].decay_rate, 0.45, places=2)
        self.assertEqual(cfg.type_persistence["pothole"].miss_threshold_k, 2)
        self.assertAlmostEqual(cfg.type_persistence["animal"].decay_rate, 0.80, places=2)
        self.assertEqual(cfg.type_persistence["animal"].miss_threshold_k, 1)

    def test_scenario_evaluation(self):
        """Verify scenario fitness evaluation returns valid bounded metrics."""
        cfg = LivingMapConfig.load()
        metrics = evaluate_scenario(cfg, seed=42)
        self.assertIn("fitness", metrics)
        self.assertIn("f1_repair", metrics)
        self.assertIn("retention_active", metrics)
        self.assertTrue(0.0 <= metrics["fitness"] <= 1.0)
        self.assertGreater(metrics["retention_active"], 0.80)

    def test_ga_optimization_and_plot(self):
        """Verify GA execution with DEAP, Hall of Fame convergence, and plot export."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = Path(tmpdir) / "ga_plot.png"

            results = run_genetic_algorithm(
                train_seeds=[42],
                test_seeds=[101],
                pop_size=6,
                generations=3,
                random_seed=42,
            )

            self.assertIn("parameters", results)
            self.assertIn("convergence_history", results)
            self.assertEqual(len(results["convergence_history"]), 4)  # Gen 0, 1, 2, 3

            plot_ga_convergence_curve(results["convergence_history"], plot_path)
            self.assertTrue(plot_path.exists())
            self.assertGreater(plot_path.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
