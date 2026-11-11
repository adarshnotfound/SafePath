"""Offline unit tests and sanity checks for SafePath Module 4 (OSM & Snapping).

Uses a small synthetic road network in Prayagraj coordinates so tests run
deterministically without requiring Overpass / network access in CI.
"""

import csv
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import pandas as pd

from routing.build_segment_features import (
    SECTION_8_SEGMENT_COLUMNS,
    build_segment_features,
    compute_tag_coverage,
    normalize_lit_tag,
    normalize_road_type,
    parse_speed_limit,
    save_segment_outputs,
)
from routing.evaluate_features import run_snapping_benchmark
from routing.osm_network import make_segment_id
from routing.snap import SegmentSnapper, snap_to_segment

SAMPLE_FEATURES_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "segment_features.csv"


def _build_synthetic_prayagraj_graph() -> nx.MultiDiGraph:
    """Create a 5-node synthetic drivable network near Prayagraj (25.4358, 81.8463)."""
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    # Nodes spaced roughly ~300 m apart in lat/lon
    nodes = {
        1: (81.8440, 25.4350),
        2: (81.8470, 25.4350),  # ~301 m east of 1
        3: (81.8500, 25.4350),  # ~301 m east of 2
        4: (81.8470, 25.4380),  # ~332 m north of 2
        5: (81.8550, 25.4420),  # far northeast edge (~1.0 km away from POIs)
    }
    for node_id, (lon, lat) in nodes.items():
        graph.add_node(node_id, x=lon, y=lat)

    # Edge 1->2: primary, lit=yes, maxspeed=50
    graph.add_edge(1, 2, key=0, highway="primary", lit="yes", maxspeed="50", length=301.0, osmid=1001, name="MG Marg")
    # Parallel edge 1->2: key=1 to verify distinct segment_id stability
    graph.add_edge(1, 2, key=1, highway="primary", lit="yes", maxspeed="50 km/h", length=301.0, osmid=1002)
    # Edge 2->3: secondary, lit=no, missing maxspeed
    graph.add_edge(2, 3, key=0, highway=["secondary", "tertiary"], lit="no", length=301.0, osmid=1003)
    # Edge 2->4: residential, missing lit -> 'unknown', maxspeed=30 mph
    graph.add_edge(2, 4, key=0, highway="residential", maxspeed="30 mph", length=332.0, osmid=1004)
    # Edge 4->5: tertiary, missing lit and maxspeed, far from center POIs
    graph.add_edge(4, 5, key=0, highway="tertiary", length=920.0, osmid=1005)
    return graph


def _build_synthetic_pois() -> pd.DataFrame:
    """Create one POI of each safety-proxy category near node 1/2."""
    return pd.DataFrame(
        [
            {"poi_id": "p1", "poi_category": "police", "name": "Civil Lines PS", "lat": 25.4351, "lon": 81.8455},
            {"poi_id": "h1", "poi_category": "hospital", "name": "City Hospital", "lat": 25.4352, "lon": 81.8456},
            {"poi_id": "f1", "poi_category": "petrol_pump", "name": "Pump", "lat": 25.4349, "lon": 81.8454},
            {"poi_id": "s1", "poi_category": "shop", "name": "General Store", "lat": 25.4350, "lon": 81.8455},
        ]
    )


class TestRoutingOSMFeatures(unittest.TestCase):
    def test_make_segment_id_is_deterministic_and_unique(self):
        self.assertEqual(make_segment_id(10, 20, 0), "seg_10_20_0")
        self.assertEqual(make_segment_id(10, 20, 0), make_segment_id(10, 20, 0))
        self.assertNotEqual(make_segment_id(10, 20, 0), make_segment_id(10, 20, 1))
        self.assertNotEqual(make_segment_id(10, 20, 0), make_segment_id(20, 10, 0))

    def test_tag_normalizers(self):
        self.assertEqual(normalize_lit_tag("yes"), "yes")
        self.assertEqual(normalize_lit_tag("24/7"), "yes")
        self.assertEqual(normalize_lit_tag("no"), "no")
        self.assertEqual(normalize_lit_tag(None), "unknown")
        self.assertEqual(normalize_lit_tag(float("nan")), "unknown")
        self.assertEqual(normalize_lit_tag(["unknown_tag", "yes"]), "yes")

        self.assertEqual(normalize_road_type("secondary"), "secondary")
        self.assertEqual(normalize_road_type(["residential", "secondary"]), "secondary")
        self.assertEqual(normalize_road_type(None), "unknown")

        self.assertEqual(parse_speed_limit("40"), 40.0)
        self.assertEqual(parse_speed_limit("50 km/h"), 50.0)
        self.assertAlmostEqual(parse_speed_limit("30 mph"), 48.3, places=1)
        self.assertEqual(parse_speed_limit(["40", "60"]), 50.0)
        self.assertIsNone(parse_speed_limit("signals"))
        self.assertIsNone(parse_speed_limit(None))

    def test_segment_features_contract_and_isolation_ordering(self):
        graph = _build_synthetic_prayagraj_graph()
        pois_df = _build_synthetic_pois()

        features_df, metadata_df, summary = build_segment_features(graph, pois_df, impute_missing_speed=False)

        self.assertEqual(list(features_df.columns), SECTION_8_SEGMENT_COLUMNS)
        if SAMPLE_FEATURES_PATH.exists():
            with open(SAMPLE_FEATURES_PATH, mode="r", encoding="utf-8") as f:
                sample_header = next(csv.reader(f))
            self.assertEqual(list(features_df.columns), sample_header)

        self.assertEqual(len(features_df), 5)
        # Downstream columns remain empty
        for empty_col in ["hazard_density", "mean_hazard_confidence", "mean_freshness", "accident_score", "time_of_day"]:
            self.assertTrue((features_df[empty_col] == "").all())

        # Isolation score is normalized in [0, 1] and edge 4->5 (far) is more isolated than 1->2 (near POIs)
        scores = dict(zip(features_df["segment_id"], features_df["isolation_score"]))
        self.assertTrue(all(0.0 <= val <= 1.0 for val in scores.values()))
        self.assertLess(scores["seg_1_2_0"], scores["seg_4_5_0"])

        # Coverage checks: 3/5 edges have lit ('yes' or 'no'), 2/5 'unknown'; 3/5 have maxspeed
        cov = summary["tag_coverage"]
        self.assertEqual(cov["lit_coverage"]["yes_count"], 2)
        self.assertEqual(cov["lit_coverage"]["no_count"], 1)
        self.assertEqual(cov["lit_coverage"]["unknown_count"], 2)
        self.assertEqual(cov["speed_limit_coverage"]["osm_tagged_count"], 3)
        self.assertEqual(cov["speed_limit_coverage"]["missing_count"], 2)

        # Verify impute_missing_speed=True populates fallback speeds while keeping coverage unchanged
        features_imp, _, summary_imp = build_segment_features(graph, pois_df, impute_missing_speed=True)
        self.assertTrue(all(val != "" for val in features_imp["speed_limit"]))
        self.assertEqual(summary_imp["tag_coverage"]["speed_limit_coverage"]["osm_tagged_count"], 3)

        # Also test direct compute_tag_coverage call
        recomputed = compute_tag_coverage(features_df)
        self.assertEqual(recomputed["total_edges"], 5)

        # Save outputs to temp dir and reload snapper from metadata CSV
        with tempfile.TemporaryDirectory() as tmp_dir:
            feat_path, meta_path = save_segment_outputs(features_df, metadata_df, output_dir=tmp_dir)
            self.assertTrue(feat_path.exists())
            self.assertTrue(meta_path.exists())
            snapper_from_csv = SegmentSnapper.from_metadata_csv(meta_path)
            res = snapper_from_csv.snap(25.43508, 81.8485)
            self.assertEqual(res["segment_id"], "seg_2_3_0")
            self.assertLess(res["distance_m"], 20.0)

    def test_snap_to_segment_metric_accuracy_and_benchmark(self):
        graph = _build_synthetic_prayagraj_graph()
        # Query point ~9 m north of edge 2->3 midpoint (25.4350, 81.8485)
        snapped = snap_to_segment(25.43508, 81.8485, graph)
        self.assertEqual(snapped["segment_id"], "seg_2_3_0")
        self.assertTrue(5.0 <= snapped["distance_m"] <= 15.0)

        with self.assertRaises(ValueError):
            snap_to_segment(95.0, 81.8485, graph)

        snapper = SegmentSnapper.from_graph(graph)
        bench = run_snapping_benchmark(graph, snapper, seed=42, num_queries=5, gps_noise_sigma_m=8.0)
        self.assertEqual(bench["random_seed"], 42)
        self.assertGreaterEqual(
            bench["proposed_projected_edge_strtree"]["corridor_match_accuracy"],
            bench["baseline_nearest_node"]["corridor_match_accuracy"],
        )


if __name__ == "__main__":
    unittest.main()

