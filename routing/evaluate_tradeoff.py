"""Evaluate Safest vs. Fastest Route Trade-Off across 20 OD Pairs (Role D).

Spec Requirements (Page 7, 11, 28):
- Baseline: The shortest or fastest route.
- Proposed: Safety-aware route.
- Experiment: Over 20 random start-end pairs (seed 42), compute:
    1. How often the safest route differs from the fastest route (%).
    2. Extra minutes it costs.
    3. Safety score improvement and worst-segment risk reduction.
- Metric persistence: Write to evaluation/results/tradeoff_fastest_vs_safest.json.
- Section 8 contract output: Write sample 2-3 route alternatives to data/processed/routes.json.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
import numpy as np

from routing.router import (
    DATA_DIR,
    PROCESSED_DIR,
    SafePathRouter,
    find_alternative_routes,
    synthesize_route_explanations,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_tradeoff")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "results"
DEFAULT_GRAPH_PATH = DATA_DIR / "raw" / "prayagraj_drive.graphml"


def run_tradeoff_experiment(
    router: SafePathRouter,
    num_pairs: int = 20,
    seed: int = 42,
    min_hops: int = 15,
) -> Dict[str, Any]:
    """Sample 20 random connected OD pairs and compare safest vs fastest routes.

    Parameters:
        router: Initialized SafePathRouter with Prayagraj graph and edge weights.
        num_pairs: Number of test pairs (default: 20 per spec).
        seed: Fixed random seed for exact reproducibility (Rule 4).
        min_hops: Minimum topological distance between start and end nodes to ensure
                  realistic non-trivial urban journeys.

    Returns:
        Dictionary of aggregate metrics and per-pair detailed logs.
    """
    rng = np.random.default_rng(seed)
    G = router.G
    nodes = list(G.nodes())

    if len(nodes) < 2:
        raise ValueError("Graph has fewer than 2 nodes.")

    logger.info("Sampling %d OD pairs with seed=%d...", num_pairs, seed)

    pairs_evaluated: List[Dict[str, Any]] = []
    attempts = 0
    max_attempts = 1000

    while len(pairs_evaluated) < num_pairs and attempts < max_attempts:
        attempts += 1
        orig = rng.choice(nodes)
        dest = rng.choice(nodes)

        if orig == dest:
            continue

        # Check if connected and reasonable journey length
        try:
            shortest_hops = nx.shortest_path_length(G, source=orig, target=dest, weight=None)
            if shortest_hops < min_hops:
                continue
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        # Compute alternative routes
        raw_routes = find_alternative_routes(G, orig, dest, max_routes=3)
        if len(raw_routes) < 2:
            # Need at least fastest and alternative
            continue

        alternatives = synthesize_route_explanations(raw_routes)
        if len(alternatives) < 2:
            continue

        # Identify fastest and safest
        fastest = min(alternatives, key=lambda r: r.total_time)
        safest = max(alternatives, key=lambda r: r.aggregate_safety)

        fastest_segs = set(fastest.segment_ids)
        safest_segs = set(safest.segment_ids)

        differs = (fastest.segment_ids != safest.segment_ids)
        jaccard = (
            len(fastest_segs & safest_segs) / len(fastest_segs | safest_segs)
            if (fastest_segs or safest_segs)
            else 1.0
        )

        extra_time_s = max(0.0, safest.total_time - fastest.total_time)
        extra_time_min = extra_time_s / 60.0
        extra_length_m = safest.total_length - fastest.total_length
        safety_gain = safest.aggregate_safety - fastest.aggregate_safety
        worst_gain = safest.worst_segment_score - fastest.worst_segment_score

        pair_idx = len(pairs_evaluated) + 1
        orig_data = G.nodes[orig]
        dest_data = G.nodes[dest]

        pair_record = {
            "pair_index": pair_idx,
            "origin_node": int(orig),
            "destination_node": int(dest),
            "origin_lat": float(orig_data.get("y", 0.0)),
            "origin_lon": float(orig_data.get("x", 0.0)),
            "destination_lat": float(dest_data.get("y", 0.0)),
            "destination_lon": float(dest_data.get("x", 0.0)),
            "topological_hops": int(shortest_hops),
            "safest_differs_from_fastest": bool(differs),
            "segment_jaccard_overlap": round(float(jaccard), 3),
            "fastest_route": {
                "route_id": fastest.route_id,
                "time_seconds": round(float(fastest.total_time), 1),
                "time_minutes": round(float(fastest.total_time / 60.0), 2),
                "length_m": round(float(fastest.total_length), 1),
                "aggregate_safety": round(float(fastest.aggregate_safety), 1),
                "worst_segment_score": round(float(fastest.worst_segment_score), 1),
            },
            "safest_route": {
                "route_id": safest.route_id,
                "time_seconds": round(float(safest.total_time), 1),
                "time_minutes": round(float(safest.total_time / 60.0), 2),
                "length_m": round(float(safest.total_length), 1),
                "aggregate_safety": round(float(safest.aggregate_safety), 1),
                "worst_segment_score": round(float(safest.worst_segment_score), 1),
            },
            "tradeoff": {
                "extra_time_seconds": round(float(extra_time_s), 1),
                "extra_time_minutes": round(float(extra_time_min), 2),
                "extra_length_m": round(float(extra_length_m), 1),
                "safety_gain": round(float(safety_gain), 1),
                "worst_segment_gain": round(float(worst_gain), 1),
            },
            "available_alternatives_count": len(alternatives),
        }
        pairs_evaluated.append(pair_record)
        logger.info(
            "Pair %02d/%02d: hops=%d, differs=%s, extra_min=%.2f, safety_gain=+%.1f, worst_gain=+%.1f",
            pair_idx, num_pairs, shortest_hops, differs, extra_time_min, safety_gain, worst_gain
        )

    # Calculate aggregate summary statistics
    differ_count = sum(1 for p in pairs_evaluated if p["safest_differs_from_fastest"])
    differ_pct = (differ_count / num_pairs) * 100.0

    extra_mins = [p["tradeoff"]["extra_time_minutes"] for p in pairs_evaluated]
    extra_dists = [p["tradeoff"]["extra_length_m"] for p in pairs_evaluated]
    safety_gains = [p["tradeoff"]["safety_gain"] for p in pairs_evaluated]
    worst_gains = [p["tradeoff"]["worst_segment_gain"] for p in pairs_evaluated]

    # When safest differs specifically
    differing_extra_mins = [p["tradeoff"]["extra_time_minutes"] for p in pairs_evaluated if p["safest_differs_from_fastest"]]
    differing_safety_gains = [p["tradeoff"]["safety_gain"] for p in pairs_evaluated if p["safest_differs_from_fastest"]]

    summary = {
        "num_pairs_evaluated": num_pairs,
        "safest_differs_count": differ_count,
        "safest_differs_percentage": round(float(differ_pct), 1),
        "extra_travel_time_minutes": {
            "mean_overall": round(float(np.mean(extra_mins)), 2),
            "median_overall": round(float(np.median(extra_mins)), 2),
            "max_overall": round(float(np.max(extra_mins)), 2),
            "mean_when_differing": round(float(np.mean(differing_extra_mins)) if differing_extra_mins else 0.0, 2),
        },
        "extra_distance_metres": {
            "mean": round(float(np.mean(extra_dists)), 1),
            "max": round(float(np.max(extra_dists)), 1),
        },
        "safety_improvement": {
            "mean_safety_gain_overall": round(float(np.mean(safety_gains)), 2),
            "mean_safety_gain_when_differing": round(float(np.mean(differing_safety_gains)) if differing_safety_gains else 0.0, 2),
            "mean_worst_segment_gain": round(float(np.mean(worst_gains)), 2),
        },
    }

    return {
        "experiment_name": "SafePath Routing: Safest vs Fastest Trade-Off Evaluation",
        "timestamp_iso": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "city": "Prayagraj, Uttar Pradesh, India",
        "random_seed": seed,
        "methodology": {
            "baseline": "Fastest Route (Dijkstra on travel time from edge length and road speeds)",
            "proposed": "Safety-Aware Route (Dijkstra on safety-penalized time cost)",
            "aggregation": "Length-weighted segment safety score plus worst-segment minimum",
            "scores_provenance": router.scores_provenance,
        },
        "summary_metrics": summary,
        "pairs_detail": pairs_evaluated,
    }


def main():
    """Execute evaluation and export metrics and routes.json."""
    logger.info("Starting Task D3 routing trade-off experiment...")

    router = SafePathRouter.from_files()
    results = run_tradeoff_experiment(router=router, num_pairs=20, seed=42)

    # 1. Save evaluation results JSON per Rule 4
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULTS_DIR / "tradeoff_fastest_vs_safest.json"
    with open(out_json, mode="w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved trade-off metrics to %s", out_json)

    # 2. Pick a representative OD pair that differs and save Section 8 routes.json
    differing_pairs = [p for p in results["pairs_detail"] if p["safest_differs_from_fastest"]]
    demo_pair = differing_pairs[0] if differing_pairs else results["pairs_detail"][0]

    demo_routes = router.route_by_nodes(
        orig_node=demo_pair["origin_node"],
        dest_node=demo_pair["destination_node"],
        max_routes=3,
    )
    routes_json_path = PROCESSED_DIR / "routes.json"
    router.export_routes_json(demo_routes, routes_json_path)
    logger.info("Exported representative routes.json to %s (OD Pair #%d)", routes_json_path, demo_pair["pair_index"])

    # Print executive summary to stdout
    summary = results["summary_metrics"]
    print("\n" + "=" * 65)
    print(" SAFEPATH ROUTING TRADE-OFF EXPERIMENT (20 OD PAIRS, SEED 42)")
    print("=" * 65)
    print(" City:                         Prayagraj, Uttar Pradesh, India")
    print(f" How often safest differs:     {summary['safest_differs_count']} / {summary['num_pairs_evaluated']} ({summary['safest_differs_percentage']}%)")
    print(f" Mean extra time when differing: {summary['extra_travel_time_minutes']['mean_when_differing']} minutes")
    print(f" Max extra time:               {summary['extra_travel_time_minutes']['max_overall']} minutes")
    print(f" Mean safety score gain:       +{summary['safety_improvement']['mean_safety_gain_when_differing']} pts")
    print(f" Mean worst-segment gain:      +{summary['safety_improvement']['mean_worst_segment_gain']} pts")
    print(f" Output JSON:                  {out_json}")
    print(f" Processed routes.json:        {routes_json_path}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()

