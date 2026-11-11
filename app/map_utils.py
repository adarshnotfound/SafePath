"""Geospatial visualization utilities for SafePath Streamlit App (Role D).

Responsible for:
1. Creating Folium interactive map centered on Prayagraj.
2. Color-coding route segments accessibly:
   - Safe (>= 70): Dark Green (#2e7d32)
   - Caution (40 - 69): Deep Amber (#ef6c00)
   - Risky (< 40): Crimson Red (#c62828)
   with explicit text badges and hover tooltips (accessible by text, not color alone).
3. Adding toggleable Hazard Layer (potholes, speed breakers, animals) and Hotspots.
4. Start / Destination landmark markers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import folium
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString

logger = logging.getLogger(__name__)

# Accessibility-tuned colors (high contrast against standard OpenStreetMap basemap)
COLOR_SAFE = "#2e7d32"     # Dark Green (score >= 70)
COLOR_CAUTION = "#ef6c00"  # Deep Amber (score 40 - 69)
COLOR_RISKY = "#c62828"    # Crimson Red (score < 40)
COLOR_DEFAULT = "#1565c0"  # Deep Blue

HAZARD_ICONS = {
    "pothole": {"icon": "exclamation-triangle", "color": "orange"},
    "speed_breaker": {"icon": "minus-circle", "color": "blue"},
    "animal": {"icon": "paw", "color": "purple"},
}


def get_safety_color_and_label(score: float) -> Tuple[str, str]:
    """Return accessible hex color and human-readable level label."""
    if score >= 70.0:
        return COLOR_SAFE, "Safe"
    elif score >= 40.0:
        return COLOR_CAUTION, "Caution"
    else:
        return COLOR_RISKY, "Risky"


class SegmentGeometryCache:
    """Fast in-memory cache mapping segment_id to coordinate points list [(lat, lon), ...]."""

    def __init__(self, metadata_csv: Path) -> None:
        self.segment_coords: Dict[str, List[Tuple[float, float]]] = {}
        if metadata_csv.exists():
            try:
                df = pd.read_csv(metadata_csv)
                if "segment_id" in df.columns and "geometry_wkt" in df.columns:
                    for _, row in df.iterrows():
                        sid = str(row["segment_id"]).strip()
                        wkt_str = str(row["geometry_wkt"]).strip()
                        try:
                            geom = wkt.loads(wkt_str)
                            if isinstance(geom, LineString):
                                # shapely coords are (lon, lat) -> folium needs (lat, lon)
                                coords = [(lat, lon) for lon, lat in geom.coords]
                                self.segment_coords[sid] = coords
                        except Exception:
                            continue
                logger.info("Loaded %d segment geometries from %s", len(self.segment_coords), metadata_csv)
            except Exception as e:
                logger.warning("Could not read segment metadata: %s", e)

    def get_coords(self, segment_id: str) -> Optional[List[Tuple[float, float]]]:
        return self.segment_coords.get(segment_id)


def build_route_folium_map(
    routes: List[Dict[str, Any]],
    selected_route_id: str,
    geom_cache: SegmentGeometryCache,
    scores_dict: Dict[str, Any],
    hazards_df: Optional[pd.DataFrame] = None,
    hotspots_df: Optional[pd.DataFrame] = None,
    show_hazards: bool = True,
    show_hotspots: bool = True,
    start_point: Optional[Tuple[float, float]] = None,
    end_point: Optional[Tuple[float, float]] = None,
) -> folium.Map:
    """Build interactive Folium map rendering selected route and overlay layers."""
    # Default center: Prayagraj Civil Lines
    center_lat, center_lon = 25.4484, 81.8340
    if start_point:
        center_lat, center_lon = start_point

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # 1. Start and Destination Markers
    if start_point:
        folium.Marker(
            location=start_point,
            popup="<b>Start Origin (Point A)</b>",
            tooltip="Origin",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(m)

    if end_point:
        folium.Marker(
            location=end_point,
            popup="<b>Destination (Point B)</b>",
            tooltip="Destination",
            icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
        ).add_to(m)

    # 2. Render Inactive Alternative Routes (subtle gray background paths)
    for r in routes:
        rid = r.get("route_id", "")
        if rid == selected_route_id:
            continue
        seg_ids = r.get("segment_ids", [])
        alt_points = []
        for sid in seg_ids:
            c = geom_cache.get_coords(sid)
            if c:
                alt_points.extend(c)

        if alt_points:
            folium.PolyLine(
                locations=alt_points,
                color="#78909c",
                weight=4,
                opacity=0.6,
                dash_array="5, 8",
                tooltip=f"Alternative: {rid} (Click to switch)",
            ).add_to(m)

    # 3. Render Selected Active Route with Color-Coded Segments
    selected_route = next((r for r in routes if r.get("route_id") == selected_route_id), None)
    if selected_route:
        seg_ids = selected_route.get("segment_ids", [])
        all_active_points = []

        for sid in seg_ids:
            coords = geom_cache.get_coords(sid)
            if not coords:
                continue
            all_active_points.extend(coords)

            # Retrieve segment score
            seg_info = scores_dict.get(sid)
            if seg_info:
                score = getattr(seg_info, "safety_score", 50.0)
                factors = getattr(seg_info, "top_factors", "standard corridor")
            else:
                score = 50.0
                factors = "unscored segment"

            color, label = get_safety_color_and_label(score)

            popup_html = (
                f"<div style='font-family:sans-serif; min-width:160px;'>"
                f"<b>Segment:</b> {sid}<br/>"
                f"<b>Safety Score:</b> <span style='color:{color}; font-weight:bold;'>{score:.1f}/100 ({label})</span><br/>"
                f"<b>Factors:</b> {factors}"
                f"</div>"
            )

            folium.PolyLine(
                locations=coords,
                color=color,
                weight=6,
                opacity=0.9,
                tooltip=f"{sid} | Score: {score:.1f} [{label}]",
                popup=folium.Popup(popup_html, max_width=300),
            ).add_to(m)

        # Fit map bounds to active route if coordinates exist
        if all_active_points:
            lats = [p[0] for p in all_active_points]
            lons = [p[1] for p in all_active_points]
            m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    # 4. Optional Hazard Layer
    if show_hazards and hazards_df is not None and not hazards_df.empty:
        hazard_group = folium.FeatureGroup(name="Road Hazards (Living Map)")
        for _, row in hazards_df.iterrows():
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
                htype = str(row.get("type", "pothole")).strip().lower()
                conf = float(row.get("confidence", 0.8))
                status = str(row.get("status", "active"))
                seen = str(row.get("last_seen", "today"))

                icon_cfg = HAZARD_ICONS.get(htype, {"icon": "warning", "color": "orange"})

                popup_txt = (
                    f"<b>Hazard:</b> {htype.replace('_', ' ').title()}<br/>"
                    f"<b>Confidence:</b> {conf * 100:.0f}%<br/>"
                    f"<b>Status:</b> {status}<br/>"
                    f"<b>Last Seen:</b> {seen}"
                )

                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    color=icon_cfg["color"],
                    fill=True,
                    fill_color=icon_cfg["color"],
                    fill_opacity=0.8,
                    tooltip=f"{htype.title()} ({conf * 100:.0f}%)",
                    popup=folium.Popup(popup_txt, max_width=250),
                ).add_to(hazard_group)
            except Exception:
                continue
        hazard_group.add_to(m)

    # 5. Optional Accident Hotspots Layer
    if show_hotspots and hotspots_df is not None and not hotspots_df.empty:
        hotspot_group = folium.FeatureGroup(name="Accident Hotspots")
        for _, row in hotspots_df.iterrows():
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
                rad = float(row.get("radius_m", 150.0))
                count = int(row.get("accident_count", 5))

                folium.Circle(
                    location=[lat, lon],
                    radius=rad,
                    color="#b71c1c",
                    weight=2,
                    fill=True,
                    fill_color="#b71c1c",
                    fill_opacity=0.2,
                    tooltip=f"Accident Hotspot ({count} historical accidents)",
                    popup=f"<b>Hotspot</b><br/>Accidents: {count}<br/>Radius: {rad:.0f}m",
                ).add_to(hotspot_group)
            except Exception:
                continue
        hotspot_group.add_to(m)

    folium.LayerControl().add_to(m)
    return m
