"""Build Prayagraj OSM segment features, report tag coverage, and benchmark snapping.

Follows Rule 4 of SafePath:
- Fixed random seed (`42`)
- Baseline (`NearestNodeSnapBaseline`) vs Proposed (`SegmentSnapper` STRtree)
- Writes reproducible metrics to `/evaluation/results/routing_osm_features.json`
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import numpy as np
from pyproj import Transformer

from routing.build_segment_features import build_segment_features, save_segment_outputs
from routing.osm_network import (
    DEFAULT_PLACE_NAME,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    PROJECT_ROOT,
    download_city_network,
    download_safety_pois,
    graph_to_edge_geodataframe,
)
from routing.snap import NearestNodeSnapBaseline, SegmentSnapper

DEFAULT_RESULTS_PATH = PROJECT_ROOT / "evaluation" / "results" / "routing_osm_features.json"


def run_snapping_benchmark(
    graph: nx.MultiDiGraph,
    snapper: SegmentSnapper,
    seed: int = 42,
    num_queries: int = 200,
    gps_noise_sigma_m: float = 10.0,
) -> Dict[str, Any]:
    """Compare Baseline nearest-node snapping vs Proposed edge-geometry snapping.

    Samples synthetic GPS query points along edges (`t` in [0.15, 0.85]) and
    adds Gaussian positional noise in projected metres (`sigma=10.0 m`, matching
    the 5-15 m phone GPS error in Section 4.2 of the specification).
    """
    rng = np.random.default_rng(seed)
    edges_gdf = graph_to_edge_geodataframe(graph)
    if edges_gdf.crs is None:
        edges_gdf = edges_gdf.set_crs("EPSG:4326")

    edges_proj = edges_gdf.to_crs(snapper.utm_crs).reset_index(drop=True)
    # Prefer segments >= 20 m so mid-segment snapping vs node snapping is informative
    long_mask = edges_proj.geometry.length >= 20.0
    candidate_indices = np.where(long_mask.to_numpy())[0]
    if len(candidate_indices) == 0:
        candidate_indices = np.arange(len(edges_proj))

    sample_count = min(num_queries, len(candidate_indices))
    chosen_indices = rng.choice(candidate_indices, size=sample_count, replace=False)

    to_wgs84 = Transformer.from_crs(snapper.utm_crs, "EPSG:4326", always_xy=True)
    queries = []
    for idx in chosen_indices:
        row = edges_proj.iloc[int(idx)]
        frac = float(rng.uniform(0.15, 0.85))
        pt_on_edge = row.geometry.interpolate(frac, normalized=True)
        dx, dy = rng.normal(0.0, gps_noise_sigma_m, size=2)
        lon_q, lat_q = to_wgs84.transform(pt_on_edge.x + dx, pt_on_edge.y + dy)
        corridor_pair = frozenset((int(row["u"]), int(row["v"])))
        queries.append(
            {
                "lat": float(lat_q),
                "lon": float(lon_q),
                "true_segment_id": str(row["segment_id"]),
                "true_corridor": corridor_pair,
            }
        )

    baseline = NearestNodeSnapBaseline(graph, snapper.utm_crs)
    baseline_metrics = _evaluate_snapper(baseline, queries, edge_lookup=_build_corridor_lookup(edges_proj))
    proposed_metrics = _evaluate_snapper(snapper, queries, edge_lookup=_build_corridor_lookup(edges_proj))

    return {
        "random_seed": seed,
        "num_queries": int(sample_count),
        "gps_noise_sigma_m": gps_noise_sigma_m,
        "baseline_nearest_node": baseline_metrics,
        "proposed_projected_edge_strtree": proposed_metrics,
    }


def evaluate_city_osm_pipeline(
    place_name: str = DEFAULT_PLACE_NAME,
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    results_path: Path = DEFAULT_RESULTS_PATH,
    seed: int = 42,
    num_queries: int = 200,
    force_refresh: bool = False,
    impute_missing_speed: bool = False,
) -> Tuple[Dict[str, Any], Path, Path]:
    """Run full OSM download, feature extraction, coverage report, and benchmark."""
    random.seed(seed)
    np.random.seed(seed)

    graph, graph_provenance = download_city_network(
        place_name=place_name,
        raw_dir=raw_dir,
        force_refresh=force_refresh,
    )
    pois_df = download_safety_pois(
        place_name=place_name,
        raw_dir=raw_dir,
        force_refresh=force_refresh,
    )

    features_df, metadata_df, feature_summary = build_segment_features(
        graph=graph,
        pois_df=pois_df,
        impute_missing_speed=impute_missing_speed,
    )
    features_path, metadata_path = save_segment_outputs(
        features_df=features_df,
        metadata_df=metadata_df,
        output_dir=processed_dir,
    )

    snapper = SegmentSnapper.from_metadata_csv(metadata_path)
    snapping_benchmark = run_snapping_benchmark(
        graph=graph,
        snapper=snapper,
        seed=seed,
        num_queries=num_queries,
    )

    results = {
        "experiment_name": "OSM Network Segment Features & Projected Edge Snapping",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "city": place_name,
        "random_seed": seed,
        "provenance": {
            "road_network": graph_provenance,
            "pois": "real (OpenStreetMap via OSMnx)",
            "snapping_benchmark_queries": "simulated (5-15 m Gaussian GPS noise along real OSM edges)",
        },
        "graph_stats": {
            "num_nodes": int(graph.number_of_nodes()),
            "num_edges": int(graph.number_of_edges()),
            "utm_crs": feature_summary["utm_crs"],
        },
        "tag_coverage": feature_summary["tag_coverage"],
        "isolation_summary": feature_summary["isolation_summary"],
        "snapping_benchmark": snapping_benchmark,
        "outputs": {
            "segment_features_csv": str(features_path),
            "segment_metadata_csv": str(metadata_path),
            "speed_limit_imputed_in_features_csv": impute_missing_speed,
        },
    }

    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, mode="w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results, features_path, metadata_path


def _build_corridor_lookup(edges_proj) -> Dict[str, frozenset]:
    """Map each `segment_id` to its undirected `{u, v}` node pair."""
    return {
        str(row["segment_id"]): frozenset((int(row["u"]), int(row["v"])))
        for _, row in edges_proj[["segment_id", "u", "v"]].iterrows()
    }


def _evaluate_snapper(
    snapper_obj: Any,
    queries: list,
    edge_lookup: Dict[str, frozenset],
) -> Dict[str, Any]:
    """Evaluate snap accuracy, distance error, and latency for one snapper."""
    exact_hits = 0
    corridor_hits = 0
    distances_m = []

    t_start = time.perf_counter()
    for q in queries:
        res = snapper_obj.snap(q["lat"], q["lon"])
        pred_id = str(res["segment_id"])
        if pred_id == q["true_segment_id"]:
            exact_hits += 1
        if edge_lookup.get(pred_id) == q["true_corridor"]:
            corridor_hits += 1
        distances_m.append(float(res["distance_m"]))
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    n = max(len(queries), 1)
    return {
        "exact_segment_id_accuracy": round(exact_hits / n, 4),
        "corridor_match_accuracy": round(corridor_hits / n, 4),
        "mean_snap_distance_m": round(float(np.mean(distances_m)), 2) if distances_m else 0.0,
        "median_snap_distance_m": round(float(np.median(distances_m)), 2) if distances_m else 0.0,
        "p95_snap_distance_m": round(float(np.percentile(distances_m, 95)), 2) if distances_m else 0.0,
        "mean_query_latency_ms": round(elapsed_ms / n, 4),
    }


def _print_summary(results: Dict[str, Any], results_path: Path) -> None:
    """Print a concise human-readable summary of OSM coverage and snapping metrics."""
    cov = results["tag_coverage"]
    lit = cov["lit_coverage"]
    spd = cov["speed_limit_coverage"]
    iso = results["isolation_summary"]
    bench = results["snapping_benchmark"]

    print(f"=== SafePath OSM Segment Features ({results['city']}) ===")
    print(
        f"Graph: {results['graph_stats']['num_nodes']} nodes, "
        f"{results['graph_stats']['num_edges']} edges, "
        f"{cov['total_length_km']} km ({results['graph_stats']['utm_crs']})"
    )
    print(
        f"lit tag coverage: {lit['explicitly_tagged_count']}/{cov['total_edges']} "
        f"({lit['explicitly_tagged_pct']}%) | yes={lit['yes_count']}, "
        f"no={lit['no_count']}, unknown={lit['unknown_count']}"
    )
    print(
        f"speed_limit (maxspeed) coverage: {spd['osm_tagged_count']}/{cov['total_edges']} "
        f"({spd['osm_tagged_pct']}%) | missing={spd['missing_count']}"
    )
    print(f"POIs found: {iso.get('poi_counts', {})}")
    print(
        f"Snapping benchmark ({bench['num_queries']} noisy queries): "
        f"Baseline corridor acc={bench['baseline_nearest_node']['corridor_match_accuracy']}, "
        f"mean_dist={bench['baseline_nearest_node']['mean_snap_distance_m']} m | "
        f"Proposed corridor acc={bench['proposed_projected_edge_strtree']['corridor_match_accuracy']}, "
        f"mean_dist={bench['proposed_projected_edge_strtree']['mean_snap_distance_m']} m"
    )
    print(f"Metrics written to: {results_path}")


def main(argv: Optional[list] = None) -> None:
    """CLI entrypoint for downloading Prayagraj OSM features and running evaluation."""
    parser = argparse.ArgumentParser(description="Build Prayagraj OSM segment features and evaluate snapping.")
    parser.add_argument("--place", default=DEFAULT_PLACE_NAME, help="City query for OSMnx.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed random seed.")
    parser.add_argument("--num-queries", type=int, default=200, help="Number of synthetic GPS snap queries.")
    parser.add_argument("--refresh", action="store_true", help="Re-download OSM data ignoring local cache.")
    parser.add_argument(
        "--impute-speed-limit",
        action="store_true",
        help="Fill missing maxspeed with road-type fallback in segment_features.csv.",
    )
    args = parser.parse_args(argv)

    results, _, _ = evaluate_city_osm_pipeline(
        place_name=args.place,
        seed=args.seed,
        num_queries=args.num_queries,
        force_refresh=args.refresh,
        impute_missing_speed=args.impute_speed_limit,
    )
    _print_summary(results, DEFAULT_RESULTS_PATH)


if __name__ == "__main__":
    main()
