"""Unit tests for SafePath Routing, Aggregation & Trade-off Logic (Role D).

Verifies:
1. Exact mathematical accuracy of length-weighted safety aggregation and worst-segment score.
2. Compliance with Section 8 routes.json contract (keys, types, value bounds).
3. Multi-route generation (fastest vs safest trade-off) on controlled synthetic topology.
4. Plain-language explanation generation.
5. Ingestion of segment safety scores and fallback behavior per Rule 3.
"""

import json
from pathlib import Path

import networkx as nx
import pytest

from routing.osm_network import make_segment_id
from routing.router import (
    RouteAlternative,
    SafePathRouter,
    SegmentScore,
    evaluate_edge_sequence,
    load_segment_scores,
    prepare_graph_edge_costs,
)


class TestRoutingAggregation:
    """Test mathematical accuracy of route metrics and aggregation formulas."""

    def test_evaluate_edge_sequence_hand_calculated_math(self):
        """Verify length-weighted safety formula against an exact hand calculation.

        Scenario:
          Edge 1: length = 100.0 m, time = 10.0 s, safety = 80.0
          Edge 2: length = 200.0 m, time = 30.0 s, safety = 50.0
          Edge 3: length = 100.0 m, time = 20.0 s, safety = 20.0

          Total length = 400.0 m
          Total time = 60.0 s
          Weighted safety = (100 * 80 + 200 * 50 + 100 * 20) / 400
                          = (8000 + 10000 + 2000) / 400 = 20000 / 400 = 50.0
          Worst segment = min(80.0, 50.0, 20.0) = 20.0
        """
        G = nx.MultiDiGraph()
        G.add_edge(1, 2, 0, segment_id="seg_1_2_0", length=100.0, travel_time=10.0, safety_score=80.0, top_factors="well lit")
        G.add_edge(2, 3, 0, segment_id="seg_2_3_0", length=200.0, travel_time=30.0, safety_score=50.0, top_factors="moderate traffic")
        G.add_edge(3, 4, 0, segment_id="seg_3_4_0", length=100.0, travel_time=20.0, safety_score=20.0, top_factors="high isolation; unlit")

        edge_sequence = [(1, 2, 0), (2, 3, 0), (3, 4, 0)]
        res = evaluate_edge_sequence(G, edge_sequence)

        assert res["total_length"] == pytest.approx(400.0, abs=1e-3)
        assert res["total_time"] == pytest.approx(60.0, abs=1e-3)
        assert res["aggregate_safety"] == pytest.approx(50.0, abs=1e-3)
        assert res["worst_segment_score"] == pytest.approx(20.0, abs=1e-3)
        assert res["worst_segment_id"] == "seg_3_4_0"
        assert res["segment_ids"] == ["seg_1_2_0", "seg_2_3_0", "seg_3_4_0"]

    def test_routes_json_contract_compliance(self, tmp_path: Path):
        """Verify output dictionary and exported JSON adhere strictly to Section 8 schema."""
        expected_keys = {
            "route_id",
            "segment_ids",
            "total_time",
            "total_length",
            "aggregate_safety",
            "worst_segment_score",
            "explanation",
        }

        alt = RouteAlternative(
            route_id="route_A",
            segment_ids=["seg_101", "seg_102"],
            total_time=450.25,
            total_length=1200.44,
            aggregate_safety=78.62,
            worst_segment_score=65.0,
            explanation="Safest route following main roads.",
        )
        cd = alt.to_contract_dict()

        assert set(cd.keys()) == expected_keys
        assert isinstance(cd["route_id"], str)
        assert isinstance(cd["segment_ids"], list)
        assert all(isinstance(s, str) for s in cd["segment_ids"])
        assert isinstance(cd["total_time"], float)
        assert isinstance(cd["total_length"], float)
        assert isinstance(cd["aggregate_safety"], float)
        assert isinstance(cd["worst_segment_score"], float)
        assert isinstance(cd["explanation"], str)

        # Check export
        out_file = tmp_path / "routes.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump([cd], f, indent=2)

        with open(out_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert len(loaded) == 1
        assert loaded[0]["route_id"] == "route_A"
        assert loaded[0]["total_time"] == 450.2
        assert loaded[0]["total_length"] == 1200.4


class TestMultiRouteRouting:
    """Test multi-route alternatives and trade-off discovery."""

    @pytest.fixture
    def diamond_graph(self) -> nx.MultiDiGraph:
        """Create a controlled diamond graph with two distinct corridors:

            (1) --- [North: Fast & Risky] ---> (3)
             |                                  |
            (0)                                (4)
             |                                  |
            (2) --- [South: Slow & Safe]  ---> (3)

        North Corridor (0 -> 1 -> 3 -> 4):
          - Fast: 60 km/h, length = 600m => time = 36s
          - Risky: safety_score = 25.0
        South Corridor (0 -> 2 -> 3 -> 4):
          - Slow: 30 km/h, length = 800m => time = 96s
          - Safe: safety_score = 88.0
        """
        G = nx.MultiDiGraph()
        # Common start
        G.add_node(0, x=81.84, y=25.43)
        G.add_node(1, x=81.84, y=25.44)
        G.add_node(2, x=81.84, y=25.42)
        G.add_node(3, x=81.86, y=25.43)
        G.add_node(4, x=81.87, y=25.43)

        # North path
        G.add_edge(0, 1, 0, length=200.0, highway="primary", maxspeed="60")
        G.add_edge(1, 3, 0, length=400.0, highway="primary", maxspeed="60")

        # South path
        G.add_edge(0, 2, 0, length=300.0, highway="secondary", maxspeed="30")
        G.add_edge(2, 3, 0, length=500.0, highway="secondary", maxspeed="30")

        # Common end
        G.add_edge(3, 4, 0, length=100.0, highway="residential", maxspeed="25")

        scores = {
            make_segment_id(0, 1, 0): SegmentScore(make_segment_id(0, 1, 0), 25.0, "risky", "accident blackspot", "", ""),
            make_segment_id(1, 3, 0): SegmentScore(make_segment_id(1, 3, 0), 25.0, "risky", "dark corridor", "", ""),
            make_segment_id(0, 2, 0): SegmentScore(make_segment_id(0, 2, 0), 88.0, "safe", "well lit, commercial", "", ""),
            make_segment_id(2, 3, 0): SegmentScore(make_segment_id(2, 3, 0), 88.0, "safe", "police station nearby", "", ""),
            make_segment_id(3, 4, 0): SegmentScore(make_segment_id(3, 4, 0), 75.0, "safe", "residential stretch", "", ""),
        }

        prepare_graph_edge_costs(G, segment_scores=scores)
        return G

    def test_safest_vs_fastest_differentiation(self, diamond_graph):
        """Router should return distinct fastest and safest alternatives on the diamond graph."""
        router = SafePathRouter(diamond_graph)
        routes = router.route_by_nodes(orig_node=0, dest_node=4, max_routes=2)

        assert len(routes) >= 2

        # Route A must be Safest
        safest = routes[0]
        # Route B must be Fastest
        fastest = routes[1]

        assert safest.aggregate_safety > fastest.aggregate_safety
        assert fastest.total_time < safest.total_time

        # Check nodes traversed
        # South path segment: 0->2->0
        assert make_segment_id(0, 2, 0) in safest.segment_ids
        # North path segment: 0->1->0
        assert make_segment_id(0, 1, 0) in fastest.segment_ids

        # Check explanations explain the trade-off
        assert "Safest" in safest.explanation or "slower" in safest.explanation
        assert "Fastest" in fastest.explanation or "quicker" in fastest.explanation

    def test_load_segment_scores_fallback_integrity(self):
        """Verify load_segment_scores loads sample scores without failure."""
        scores, prov = load_segment_scores()
        assert len(scores) > 0
        assert "seg_101" in scores or any(s.startswith("seg_") for s in scores)
        assert isinstance(prov, str)

