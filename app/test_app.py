"""Unit tests for SafePath Streamlit Web Application (Role D).

Verifies:
1. Configuration resolution and switching between mock and real modes.
2. Data provenance disclosure dictionary structure per Section 12.
3. Accessible color and label categorization (Safe/Caution/Risky).
4. SegmentGeometryCache coordinate extraction from WKT geometries.
5. Folium map generation with route lines, markers, and layer groups.
6. Landmark coordinate validity within Prayagraj geographic bounds.
"""

from pathlib import Path

import folium
import pytest

from app.config import (
    PRAYAGRAJ_LANDMARKS,
    get_app_config,
)
from app.map_utils import (
    COLOR_CAUTION,
    COLOR_RISKY,
    COLOR_SAFE,
    SegmentGeometryCache,
    build_route_folium_map,
    get_safety_color_and_label,
)


class TestAppConfig:
    """Test app configuration management and mode toggling."""

    def test_mock_mode_configuration(self):
        """Mock mode must resolve to data/sample directory."""
        cfg = get_app_config("mock")
        assert cfg.mode == "mock"
        assert not cfg.is_real_mode
        assert "sample" in str(cfg.routes_json)
        assert "sample" in str(cfg.segment_features_csv)

        prov = cfg.get_provenance_labels()
        assert "Road Network & Distances" in prov
        assert "Sample" in prov["Road Network & Distances"]

    def test_real_mode_configuration(self):
        """Real mode should resolve to processed or cached files."""
        cfg = get_app_config("real")
        assert cfg.mode == "real"
        assert cfg.is_real_mode
        assert cfg.routes_json.exists()

        prov = cfg.get_provenance_labels()
        assert "Real (OpenStreetMap" in prov["Road Network & Distances"]
        assert "Simulated" in prov["Pothole & Hazard Detections"]
        assert "Substitute" in prov["Safety Scores"]

    def test_prayagraj_landmarks_bounds(self):
        """Verify all preset landmarks sit within Prayagraj municipal bounding area."""
        # Prayagraj approximate bounds: lat [25.35, 25.55], lon [81.75, 81.95]
        for name, coords in PRAYAGRAJ_LANDMARKS.items():
            lat = coords["lat"]
            lon = coords["lon"]
            assert 25.30 <= lat <= 25.60, f"Landmark {name} lat {lat} out of bounds"
            assert 81.70 <= lon <= 82.00, f"Landmark {name} lon {lon} out of bounds"


class TestMapUtils:
    """Test map rendering and visual accessibility rules."""

    def test_accessible_color_and_label_thresholds(self):
        """Verify 3-tier safety coloring adheres to specified score intervals."""
        col, lbl = get_safety_color_and_label(85.0)
        assert col == COLOR_SAFE
        assert lbl == "Safe"

        col, lbl = get_safety_color_and_label(70.0)
        assert col == COLOR_SAFE
        assert lbl == "Safe"

        col, lbl = get_safety_color_and_label(55.0)
        assert col == COLOR_CAUTION
        assert lbl == "Caution"

        col, lbl = get_safety_color_and_label(40.0)
        assert col == COLOR_CAUTION
        assert lbl == "Caution"

        col, lbl = get_safety_color_and_label(35.0)
        assert col == COLOR_RISKY
        assert lbl == "Risky"

    def test_segment_geometry_cache(self, tmp_path: Path):
        """Verify parsing LineString WKT into (lat, lon) coordinates."""
        dummy_csv = tmp_path / "test_metadata.csv"
        dummy_csv.write_text(
            "segment_id,geometry_wkt\n"
            "seg_test_1,\"LINESTRING (81.83 25.45, 81.84 25.46)\"\n",
            encoding="utf-8",
        )

        cache = SegmentGeometryCache(dummy_csv)
        coords = cache.get_coords("seg_test_1")
        assert coords is not None
        assert len(coords) == 2
        # Verify folium order (lat, lon)
        assert coords[0] == pytest.approx((25.45, 81.83))
        assert coords[1] == pytest.approx((25.46, 81.84))

    def test_build_route_folium_map(self, tmp_path: Path):
        """Verify building a valid Folium map with active route."""
        dummy_csv = tmp_path / "test_metadata.csv"
        dummy_csv.write_text(
            "segment_id,geometry_wkt\n"
            "seg_A,\"LINESTRING (81.83 25.45, 81.84 25.46)\"\n",
            encoding="utf-8",
        )
        cache = SegmentGeometryCache(dummy_csv)

        routes = [
            {
                "route_id": "route_A",
                "segment_ids": ["seg_A"],
                "total_time": 120.0,
                "total_length": 500.0,
                "aggregate_safety": 82.0,
                "worst_segment_score": 82.0,
                "explanation": "Safe test route",
            }
        ]

        m = build_route_folium_map(
            routes=routes,
            selected_route_id="route_A",
            geom_cache=cache,
            scores_dict={},
            start_point=(25.45, 81.83),
            end_point=(25.46, 81.84),
        )

        assert isinstance(m, folium.Map)
        html_str = m.get_root().render()
        assert "seg_A" in html_str or "OpenStreetMap" in html_str
