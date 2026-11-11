"""SafePath Routing & Aggregation Engine (Role D).

Responsible for:
1. Multi-route generation (Fastest, Safest, Balanced) over the OSMnx road network.
2. Segment safety score ingestion with documented heuristic fallback (Rule 3).
3. Length-weighted safety score aggregation:
       aggregate_safety = sum(length_i * score_i) / sum(length_i)
   and worst_segment_score identification:
       worst_segment_score = min(score_i)
4. Plain-language comparative explanation synthesis.
5. Exporting routes adhering strictly to Section 8 routes.json contract:
       [
         {
           "route_id": "...",
           "segment_ids": [...],
           "total_time": 0.0,
           "total_length": 0.0,
           "aggregate_safety": 0.0,
           "worst_segment_score": 0.0,
           "explanation": "..."
         }
       ]
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from routing.build_segment_features import ROAD_TYPE_FALLBACK_SPEED_KMH
from routing.osm_network import make_segment_id
from routing.snap import SegmentSnapper

logger = logging.getLogger(__name__)

DEFAULT_SPEED_LIMIT_KMH = ROAD_TYPE_FALLBACK_SPEED_KMH
FALLBACK_SPEED_KMH = 30.0

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_DIR = DATA_DIR / "sample"
PROCESSED_DIR = DATA_DIR / "processed"


@dataclass
class SegmentScore:
    """Per-segment safety assessment record."""

    segment_id: str
    safety_score: float
    level: str  # 'safe' | 'caution' | 'risky'
    top_factors: str
    fired_rules: str
    model_name: str


@dataclass
class RouteAlternative:
    """Single evaluated route alternative matching Section 8 contract."""

    route_id: str
    segment_ids: List[str]
    total_time: float  # seconds
    total_length: float  # metres
    aggregate_safety: float  # 0 to 100
    worst_segment_score: float  # 0 to 100
    explanation: str

    def to_contract_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching Section 8 contract exactly."""
        return {
            "route_id": self.route_id,
            "segment_ids": list(self.segment_ids),
            "total_time": round(float(self.total_time), 1),
            "total_length": round(float(self.total_length), 1),
            "aggregate_safety": round(float(self.aggregate_safety), 1),
            "worst_segment_score": round(float(self.worst_segment_score), 1),
            "explanation": str(self.explanation),
        }


# =============================================================================
# 1. Segment Score Ingestion & Fallback
# =============================================================================


def load_segment_scores(
    scores_path: Optional[Path] = None,
    features_path: Optional[Path] = None,
) -> Tuple[Dict[str, SegmentScore], str]:
    """Load segment safety scores from CSV, falling back gracefully per Rule 3.

    Resolution strategy:
    1. If `features_path` or `data/processed/segment_features.csv` exists, compute
       the deterministic baseline heuristic across all network segments so every edge
       in the city graph has an empirical safety score based on isolation, road class,
       and lighting.
    2. Overlay scores from `scores_path`, `data/processed/segment_scores.csv` (Module C),
       or `data/sample/segment_scores.csv`.

    Returns:
        (scores_dict, provenance_label)
    """
    scores: Dict[str, SegmentScore] = {}
    provenance = "none"

    # Step 1: Base scores from segment_features.csv (covers all 43,328 network edges)
    features_csv = features_path or (PROCESSED_DIR / "segment_features.csv")
    if not features_csv.exists():
        features_csv = SAMPLE_DIR / "segment_features.csv"

    if features_csv.exists():
        scores = _compute_substitute_baseline_scores(features_csv)
        provenance = "substitute_baseline_heuristic (features)"

    # Step 2: Overlay specialized model scores from Module C or sample file
    overlay_candidates = []
    if scores_path is not None:
        overlay_candidates.append((scores_path, "explicit_scores_file"))
    overlay_candidates.append((PROCESSED_DIR / "segment_scores.csv", "processed_module_c"))
    overlay_candidates.append((SAMPLE_DIR / "segment_scores.csv", "sample_segment_scores"))

    overlay_count = 0
    for path, prov in overlay_candidates:
        if path.exists() and path.stat().st_size > 0:
            try:
                df = pd.read_csv(path)
                req_cols = {"segment_id", "safety_score"}
                if req_cols.issubset(df.columns):
                    for _, row in df.iterrows():
                        sid = str(row["segment_id"]).strip()
                        sc = float(row["safety_score"])
                        lvl = str(row.get("level", "caution")).strip()
                        top = str(row.get("top_factors", "")).strip()
                        rules = str(row.get("fired_rules", "")).strip()
                        model = str(row.get("model_name", "unknown")).strip()
                        scores[sid] = SegmentScore(
                            segment_id=sid,
                            safety_score=sc,
                            level=lvl,
                            top_factors=top,
                            fired_rules=rules,
                            model_name=model,
                        )
                        overlay_count += 1
                    if overlay_count > 0:
                        provenance = f"{provenance} + {prov}" if scores else f"real_{prov}"
                    logger.info("Loaded/overlaid %d segment scores from %s (%s)", overlay_count, path, prov)
                    break
            except Exception as e:
                logger.warning("Failed loading scores from %s: %s", path, e)

    return scores, provenance


def _compute_substitute_baseline_scores(features_csv: Path) -> Dict[str, SegmentScore]:
    """Compute transparent baseline scores when Module C output is not yet available.

    Formula (0-100 scale, higher is safer):
      safety = 100 - 45 * isolation_score - 25 * (1 if lit == 'no'/'unknown' else 0)
                   - 15 * (1 if road_type is residential/unclassified else 0)
                   - 15 * hazard_penalty
    This matches the Mamdani rule intent in Section 4.6 and is labelled substitute per Rule 3.
    """
    df = pd.read_csv(features_csv)
    scores: Dict[str, SegmentScore] = {}

    for _, row in df.iterrows():
        sid = str(row["segment_id"]).strip()
        isolation = float(row.get("isolation_score", 0.5) or 0.5)
        lit = str(row.get("lit", "unknown")).strip().lower()
        road_type = str(row.get("road_type", "residential")).strip().lower()

        # Deductions
        isolation_penalty = 45.0 * np.clip(isolation, 0.0, 1.0)
        lit_penalty = 0.0 if lit == "yes" else (20.0 if lit == "unknown" else 30.0)
        road_penalty = 15.0 if road_type in ("residential", "living_street", "unclassified") else 0.0

        hazard_density = float(row.get("hazard_density", 0.0) or 0.0)
        hazard_penalty = min(25.0, hazard_density * 20.0)

        raw_score = 100.0 - (isolation_penalty + lit_penalty + road_penalty + hazard_penalty)
        score = float(np.clip(raw_score, 5.0, 95.0))

        if score >= 70.0:
            lvl = "safe"
        elif score >= 40.0:
            lvl = "caution"
        else:
            lvl = "risky"

        factors = []
        if lit != "yes":
            factors.append(f"lighting {lit}")
        if isolation > 0.6:
            factors.append("high isolation")
        if road_penalty > 0:
            factors.append(f"narrow {road_type} road")
        if hazard_penalty > 0:
            factors.append("reported road hazards")
        top_factors = "; ".join(factors) if factors else "standard urban corridor"

        scores[sid] = SegmentScore(
            segment_id=sid,
            safety_score=round(score, 1),
            level=lvl,
            top_factors=top_factors,
            fired_rules="baseline_heuristic_formula",
            model_name="substitute_baseline_heuristic",
        )

    return scores


# =============================================================================
# 2. Graph Preparation & Edge Attributes
# =============================================================================


def prepare_graph_edge_costs(
    G: nx.MultiDiGraph,
    segment_scores: Optional[Dict[str, SegmentScore]] = None,
    default_safety_score: float = 50.0,
    safety_weight_beta: float = 3.0,
) -> nx.MultiDiGraph:
    """Assign segment_id, travel_time, safety_score, and route costs to every graph edge.

    Parameters:
        G: Road network graph from OSMnx.
        segment_scores: Dict mapping segment_id to SegmentScore.
        default_safety_score: Score to assign if segment_id is unknown (50.0).
        safety_weight_beta: Multiplier on risk in safety-penalized edge routing.
                            Edge cost = travel_time * (1.0 + beta * (100 - score) / 100).
    """
    scores = segment_scores or {}

    for u, v, k, data in G.edges(keys=True, data=True):
        # 1. Deterministic segment_id
        if "segment_id" not in data or not data["segment_id"]:
            data["segment_id"] = make_segment_id(u, v, k)
        sid = data["segment_id"]

        # 2. Edge length in metres
        length_m = float(data.get("length", 10.0))
        if length_m <= 0.0:
            length_m = 10.0
        data["length"] = length_m

        # 3. Speed limit in km/h -> m/s
        speed_kmh = None
        raw_speed = data.get("maxspeed")
        if raw_speed is not None:
            if isinstance(raw_speed, list) and len(raw_speed) > 0:
                raw_speed = raw_speed[0]
            try:
                cleaned = "".join(ch for ch in str(raw_speed) if ch.isdigit() or ch == ".")
                if cleaned:
                    speed_kmh = float(cleaned)
            except ValueError:
                speed_kmh = None

        if speed_kmh is None or speed_kmh <= 5.0 or speed_kmh > 140.0:
            highway = data.get("highway", "residential")
            if isinstance(highway, list) and len(highway) > 0:
                highway = highway[0]
            speed_kmh = DEFAULT_SPEED_LIMIT_KMH.get(str(highway).strip().lower(), FALLBACK_SPEED_KMH)

        speed_mps = max(1.0, speed_kmh * (1000.0 / 3600.0))
        data["speed_kph"] = speed_kmh
        data["speed_mps"] = speed_mps

        # 4. Traversal time in seconds
        travel_time = length_m / speed_mps
        data["travel_time"] = travel_time

        # 5. Segment safety score (0 to 100)
        seg_rec = scores.get(sid)
        if seg_rec is not None:
            sc = float(seg_rec.safety_score)
            lvl = seg_rec.level
            top_factors = seg_rec.top_factors
        else:
            sc = default_safety_score
            lvl = "caution"
            top_factors = "unscored segment"

        data["safety_score"] = sc
        data["safety_level"] = lvl
        data["top_factors"] = top_factors

        # 6. Routing costs
        # Pure time cost (for fastest route)
        data["cost_fastest"] = travel_time

        # Pure length cost (for shortest distance route)
        data["cost_distance"] = length_m

        # Safety-penalized time cost (for safest route)
        # Higher risk (lower score) increases effective cost.
        risk_fraction = (100.0 - sc) / 100.0  # 0.0 (safest) to 1.0 (riskiest)
        data["cost_safest"] = travel_time * (1.0 + safety_weight_beta * (risk_fraction ** 1.5))

        # Balanced cost (mild safety penalty)
        data["cost_balanced"] = travel_time * (1.0 + (safety_weight_beta * 0.4) * risk_fraction)

    return G


# =============================================================================
# 3. Route Calculation & Multi-Route Generation
# =============================================================================


def _path_to_edge_list(
    G: nx.MultiDiGraph,
    node_path: List[int | str],
    weight_key: str = "travel_time",
) -> List[Tuple[Any, Any, Any]]:
    """Convert a sequence of nodes into a sequence of edge keys (u, v, k)."""
    edges = []
    for i in range(len(node_path) - 1):
        u, v = node_path[i], node_path[i + 1]
        edge_dict = G.get_edge_data(u, v)
        if not edge_dict:
            continue
        # Pick the edge with the minimal cost according to weight_key
        best_k = min(edge_dict.keys(), key=lambda k: edge_dict[k].get(weight_key, 0.0))
        edges.append((u, v, best_k))
    return edges


def evaluate_edge_sequence(
    G: nx.MultiDiGraph,
    edge_sequence: List[Tuple[Any, Any, Any]],
) -> Dict[str, Any]:
    """Compute aggregate length, time, length-weighted safety, and worst segment.

    Aggregation formulas (Section 4.7):
      total_length = sum(length_i)
      total_time = sum(time_i)
      aggregate_safety = sum(length_i * score_i) / total_length
      worst_segment_score = min(score_i)
    """
    if not edge_sequence:
        return {
            "segment_ids": [],
            "total_time": 0.0,
            "total_length": 0.0,
            "aggregate_safety": 50.0,
            "worst_segment_score": 50.0,
            "worst_segment_id": None,
            "segments_metadata": [],
        }

    segment_ids: List[str] = []
    total_length = 0.0
    total_time = 0.0
    weighted_score_sum = 0.0
    worst_score = float("inf")
    worst_sid = None
    worst_factors = ""
    metadata = []

    for u, v, k in edge_sequence:
        data = G[u][v][k]
        sid = data.get("segment_id", make_segment_id(u, v, k))
        length = float(data.get("length", 0.0))
        time_s = float(data.get("travel_time", 0.0))
        score = float(data.get("safety_score", 50.0))
        factors = str(data.get("top_factors", ""))

        segment_ids.append(sid)
        total_length += length
        total_time += time_s
        weighted_score_sum += length * score

        if score < worst_score:
            worst_score = score
            worst_sid = sid
            worst_factors = factors

        metadata.append({
            "segment_id": sid,
            "length_m": length,
            "travel_time_s": time_s,
            "safety_score": score,
            "road_type": data.get("highway", "residential"),
            "top_factors": factors,
        })

    agg_safety = (weighted_score_sum / total_length) if total_length > 0 else 50.0
    if worst_score == float("inf"):
        worst_score = 50.0

    return {
        "segment_ids": segment_ids,
        "total_time": total_time,
        "total_length": total_length,
        "aggregate_safety": agg_safety,
        "worst_segment_score": worst_score,
        "worst_segment_id": worst_sid,
        "worst_factors": worst_factors,
        "segments_metadata": metadata,
    }


def find_alternative_routes(
    G: nx.MultiDiGraph,
    orig_node: int | str,
    dest_node: int | str,
    max_routes: int = 3,
) -> List[Dict[str, Any]]:
    """Compute 2-3 diverse alternative routes between origin and destination nodes.

    Strategy:
    1. Route 1: 'Fastest' - Dijkstra on `cost_fastest` (pure travel time).
    2. Route 2: 'Safest' - Dijkstra on `cost_safest` (steep penalty on low safety scores).
    3. Route 3: 'Balanced' - Dijkstra on `cost_balanced` or penalty-diversified route
       avoiding the major segments of Route 1.

    Returns:
        List of raw evaluated route dictionaries.
    """
    candidates = []

    # 1. Fastest Path
    try:
        fastest_nodes = nx.shortest_path(G, source=orig_node, target=dest_node, weight="cost_fastest")
        fastest_edges = _path_to_edge_list(G, fastest_nodes, weight_key="cost_fastest")
        fastest_eval = evaluate_edge_sequence(G, fastest_edges)
        fastest_eval["intent"] = "fastest"
        candidates.append(fastest_eval)
    except nx.NetworkXNoPath:
        logger.warning("No path found between %s and %s", orig_node, dest_node)
        return []

    # 2. Safest Path
    try:
        safest_nodes = nx.shortest_path(G, source=orig_node, target=dest_node, weight="cost_safest")
        safest_edges = _path_to_edge_list(G, safest_nodes, weight_key="cost_safest")
        safest_eval = evaluate_edge_sequence(G, safest_edges)
        safest_eval["intent"] = "safest"
        candidates.append(safest_eval)
    except nx.NetworkXNoPath:
        pass

    # 3. Balanced / Diversified Path
    # Use penalty-based corridor diversification so Route C doesn't just duplicate Route A or B.
    try:
        # Clone temporary weights or penalize edges used in fastest route
        used_fastest_edges = set(fastest_eval["segment_ids"])
        temp_graph = nx.MultiDiGraph(G)
        for u, v, k, data in temp_graph.edges(keys=True, data=True):
            sid = data.get("segment_id", make_segment_id(u, v, k))
            base_cost = data.get("cost_balanced", data.get("travel_time", 1.0))
            if sid in used_fastest_edges:
                # 60% penalty on previously used edges to push router to alternative corridor
                data["cost_temp_diverse"] = base_cost * 1.6
            else:
                data["cost_temp_diverse"] = base_cost

        diverse_nodes = nx.shortest_path(temp_graph, source=orig_node, target=dest_node, weight="cost_temp_diverse")
        diverse_edges = _path_to_edge_list(G, diverse_nodes, weight_key="travel_time")
        diverse_eval = evaluate_edge_sequence(G, diverse_edges)
        diverse_eval["intent"] = "balanced"
        candidates.append(diverse_eval)
    except (nx.NetworkXNoPath, Exception):
        # Fallback to shortest distance if diversification fails
        try:
            dist_nodes = nx.shortest_path(G, source=orig_node, target=dest_node, weight="cost_distance")
            dist_edges = _path_to_edge_list(G, dist_nodes, weight_key="cost_distance")
            dist_eval = evaluate_edge_sequence(G, dist_edges)
            dist_eval["intent"] = "balanced"
            candidates.append(dist_eval)
        except nx.NetworkXNoPath:
            pass

    # Deduplicate candidate routes by segment_ids tuple
    unique_routes: List[Dict[str, Any]] = []
    seen_routes = set()

    for cand in candidates:
        sig = tuple(cand["segment_ids"])
        if sig not in seen_routes and len(cand["segment_ids"]) > 0:
            seen_routes.add(sig)
            unique_routes.append(cand)
            if len(unique_routes) >= max_routes:
                break

    # If deduplication left only 1 route, attempt 2nd shortest path via edge removal
    if len(unique_routes) == 1 and len(fastest_edges) > 2:
        try:
            midpoint_edge = fastest_edges[len(fastest_edges) // 2]
            G_sub = G.copy()
            if G_sub.has_edge(midpoint_edge[0], midpoint_edge[1], midpoint_edge[2]):
                G_sub.remove_edge(midpoint_edge[0], midpoint_edge[1], midpoint_edge[2])
                alt_nodes = nx.shortest_path(G_sub, source=orig_node, target=dest_node, weight="cost_fastest")
                alt_edges = _path_to_edge_list(G, alt_nodes, weight_key="cost_fastest")
                alt_eval = evaluate_edge_sequence(G, alt_edges)
                alt_eval["intent"] = "alternative"
                sig = tuple(alt_eval["segment_ids"])
                if sig not in seen_routes:
                    unique_routes.append(alt_eval)
        except Exception:
            pass

    return unique_routes


# =============================================================================
# 4. Plain-Language Explanation Synthesizer
# =============================================================================


def synthesize_route_explanations(routes: List[Dict[str, Any]]) -> List[RouteAlternative]:
    """Assign route IDs ('route_A', 'route_B', 'route_C') and synthesize clear explanations.

    Ranking Order:
    - If multiple routes exist, route_A is the Safest route (SafePath primary goal),
      followed by the fastest alternative, and balanced alternative.
    """
    if not routes:
        return []

    # Sort routes by aggregate safety descending
    sorted_by_safety = sorted(routes, key=lambda r: r["aggregate_safety"], reverse=True)
    fastest_route = min(routes, key=lambda r: r["total_time"])
    min_time_s = fastest_route["total_time"]

    labeled_results: List[RouteAlternative] = []
    route_letters = ["A", "B", "C", "D"]

    for idx, r in enumerate(sorted_by_safety):
        rid = f"route_{route_letters[idx]}" if idx < len(route_letters) else f"route_{idx + 1}"
        total_time_s = r["total_time"]
        total_len_m = r["total_length"]
        agg_safety = r["aggregate_safety"]
        worst_score = r["worst_segment_score"]
        worst_factors = r.get("worst_factors", "mixed conditions")
        time_diff_min = (total_time_s - min_time_s) / 60.0

        is_fastest = (r is fastest_route) or (abs(total_time_s - min_time_s) < 1.0)
        is_safest = (idx == 0)

        # Generate comparative explanation
        if is_safest and is_fastest:
            exp = (
                f"Optimal route: both safest (score {agg_safety:.1f}) and fastest "
                f"({total_time_s / 60.0:.1f} mins). Avoids severe hazards and isolated stretches "
                f"(lowest segment score: {worst_score:.1f})."
            )
        elif is_safest:
            extra_txt = f"{time_diff_min:.1f} minutes slower than fastest" if time_diff_min >= 0.1 else "comparable travel time"
            exp = (
                f"Safest route (safety score {agg_safety:.1f}): prioritizes higher-grade, lower-isolation roads; "
                f"{extra_txt}, but maintains a higher minimum segment score ({worst_score:.1f})."
            )
        elif is_fastest:
            quicker_min = (sorted_by_safety[0]["total_time"] - total_time_s) / 60.0
            quicker_txt = f" ({quicker_min:.1f} mins quicker)" if quicker_min >= 0.1 else ""
            exp = (
                f"Fastest route{quicker_txt}: minimizes travel time ({total_time_s / 60.0:.1f} mins), "
                f"but has lower overall safety ({agg_safety:.1f}) and traverses stretches with {worst_factors} "
                f"(worst segment score: {worst_score:.1f})."
            )
        else:
            exp = (
                f"Balanced alternative (safety score {agg_safety:.1f}): compromise between safety and travel time "
                f"({total_time_s / 60.0:.1f} mins). Traverses moderate corridors (worst segment score: {worst_score:.1f})."
            )

        alt = RouteAlternative(
            route_id=rid,
            segment_ids=r["segment_ids"],
            total_time=total_time_s,
            total_length=total_len_m,
            aggregate_safety=agg_safety,
            worst_segment_score=worst_score,
            explanation=exp,
        )
        labeled_results.append(alt)

    return labeled_results


# =============================================================================
# 5. High-Level SafePathRouter Class
# =============================================================================


class SafePathRouter:
    """Unified routing interface for SafePath."""

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        segment_scores: Optional[Dict[str, SegmentScore]] = None,
        snapper: Optional[SegmentSnapper] = None,
        scores_provenance: str = "real",
    ):
        self.G = copy.deepcopy(graph)
        self.scores = segment_scores or {}
        self.scores_provenance = scores_provenance
        if segment_scores is not None:
            self.scores = segment_scores
        else:
            # Extract existing edge scores if already populated on graph
            existing_scores = {}
            for u, v, k, data in self.G.edges(keys=True, data=True):
                sid = data.get("segment_id", make_segment_id(u, v, k))
                if "safety_score" in data:
                    existing_scores[sid] = SegmentScore(
                        segment_id=sid,
                        safety_score=float(data["safety_score"]),
                        level=str(data.get("safety_level", "caution")),
                        top_factors=str(data.get("top_factors", "")),
                        fired_rules=str(data.get("fired_rules", "")),
                        model_name=str(data.get("model_name", "graph_edge")),
                    )
            self.scores = existing_scores

        # Prepare edge attributes
        prepare_graph_edge_costs(self.G, self.scores)
        self._snapper = snapper

    @property
    def snapper(self) -> SegmentSnapper:
        """Lazily initialize SegmentSnapper if not already provided."""
        if self._snapper is None:
            self._snapper = SegmentSnapper.from_graph(self.G)
        return self._snapper

    @classmethod
    def from_files(
        cls,
        graph_path: Optional[Path] = None,
        scores_path: Optional[Path] = None,
        features_path: Optional[Path] = None,
    ) -> SafePathRouter:
        """Construct router from standard cached files."""
        import osmnx as ox

        g_path = graph_path or (DATA_DIR / "raw" / "prayagraj_drive.graphml")
        if not g_path.exists():
            raise FileNotFoundError(f"Graph cache not found at {g_path}. Run osm_network.py first.")

        logger.info("Loading road graph from %s...", g_path)
        G = ox.load_graphml(g_path)

        scores, prov = load_segment_scores(scores_path=scores_path, features_path=features_path)
        logger.info("Initialized router with %d nodes, %d edges, %d scores (%s).",
                    len(G.nodes), len(G.edges), len(scores), prov)

        return cls(graph=G, segment_scores=scores, scores_provenance=prov)

    def route_by_nodes(
        self,
        orig_node: int | str,
        dest_node: int | str,
        max_routes: int = 3,
    ) -> List[RouteAlternative]:
        """Compute 2-3 evaluated and explained alternative routes between graph nodes."""
        raw_routes = find_alternative_routes(self.G, orig_node, dest_node, max_routes=max_routes)
        return synthesize_route_explanations(raw_routes)

    def route_by_coordinates(
        self,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        max_routes: int = 3,
    ) -> List[RouteAlternative]:
        """Snap arbitrary coordinates to nearest edges/nodes and compute alternative routes."""
        snap_start = self.snapper.snap(start_lat, start_lon)
        snap_end = self.snapper.snap(end_lat, end_lon)

        # Connect to node u of start segment and node v of end segment
        start_node = snap_start["u"]
        end_node = snap_end["v"]

        if start_node == end_node:
            # Trivial same-node trip: use target segment
            sid = snap_start["segment_id"]
            sc = self.scores.get(sid, SegmentScore(sid, 50.0, "caution", "short hop", "", "")).safety_score
            return [
                RouteAlternative(
                    route_id="route_A",
                    segment_ids=[sid],
                    total_time=15.0,
                    total_length=25.0,
                    aggregate_safety=sc,
                    worst_segment_score=sc,
                    explanation="Short local trip within the same street segment.",
                )
            ]

        return self.route_by_nodes(start_node, end_node, max_routes=max_routes)

    def export_routes_json(
        self,
        routes: List[RouteAlternative],
        output_path: Path,
    ) -> Path:
        """Write routes to JSON adhering to Section 8 contract."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        contract_data = [r.to_contract_dict() for r in routes]
        with open(output_path, mode="w", encoding="utf-8") as f:
            json.dump(contract_data, f, indent=2)
        logger.info("Saved %d routes to %s matching Section 8 contract.", len(routes), output_path)
        return output_path
