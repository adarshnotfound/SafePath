"""Geospatial visualization utilities for SafePath Streamlit App (Role D).

Responsible for:
1. Creating Folium interactive map centered on Prayagraj.
2. Color-coding ALL route alternatives accessibly with distinct, vibrant colors:
   - Safest (Route A): Emerald Green (#2e7d32)
   - Fastest (Route B): Vibrant Orange (#e65100)
   - Balanced (Route C): Royal Blue (#1565c0)
   - Active/Selected route is prominently emphasized with a high-contrast casing.
3. Color-coding route segments accessibly by safety score (Safe/Caution/Risky).
4. Adding toggleable Hazard Layer (potholes, speed breakers, animals) and Hotspots.
5. Start / Destination landmark markers.
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

# Distinct color palette for alternative routes so all paths are colored simultaneously
ALTERNATIVE_ROUTE_COLORS = [
    "#2e7d32",  # Emerald Green (Route A / Safest)
    "#e65100",  # Deep Orange (Route B / Fastest)
    "#1565c0",  # Royal Blue (Route C / Balanced)
    "#6a1b9a",  # Deep Purple
    "#00838f",  # Dark Teal
]

HAZARD_ICONS = {
    "pothole": {"icon": "exclamation-triangle", "color": "orange"},
    "speed_breaker": {"icon": "minus-circle", "color": "blue"},
    "animal": {"icon": "paw", "color": "purple"},
}

# Empirical default road segment coordinates along Prayagraj corridor
DEFAULT_SAMPLE_SEGMENTS: Dict[str, List[Tuple[float, float]]] = {
    "seg_101": [(25.4526, 81.8345), (25.4500, 81.8390)],
    "seg_102": [(25.4500, 81.8390), (25.4420, 81.8540)],
    "seg_103": [(25.4420, 81.8540), (25.4330, 81.8720), (25.4290, 81.8840)],
    "seg_104": [(25.4500, 81.8390), (25.4440, 81.8480), (25.4380, 81.8620)],
    "seg_105": [(25.4526, 81.8345), (25.4550, 81.8430), (25.4580, 81.8520)],
    "seg_106": [(25.4580, 81.8520), (25.4460, 81.8700), (25.4290, 81.8840)],
    "seg_107": [(25.4550, 81.8430), (25.4510, 81.8580), (25.4410, 81.8680)],
    "seg_108": [(25.4410, 81.8680), (25.4330, 81.8780), (25.4290, 81.8840)],
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
        # Prepopulate with standard sample segments so contract fixtures always render
        self.segment_coords: Dict[str, List[Tuple[float, float]]] = dict(DEFAULT_SAMPLE_SEGMENTS)

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


def extract_route_points(route: Dict[str, Any], geom_cache: SegmentGeometryCache) -> List[Tuple[float, float]]:
    """Extract ordered (lat, lon) points for a route from its direct coords or segments."""
    # 1. Direct coordinates on route dictionary
    if "coordinates" in route and isinstance(route["coordinates"], list) and len(route["coordinates"]) > 0:
        return [(float(p[0]), float(p[1])) for p in route["coordinates"]]

    # 2. Reconstruct from segment geometry cache
    seg_ids = route.get("segment_ids", [])
    points: List[Tuple[float, float]] = []
    for sid in seg_ids:
        c = geom_cache.get_coords(sid)
        if c:
            points.extend(c)
    return points


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
    """Build interactive Folium map rendering ALL colored alternative routes and overlays."""
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

    all_map_points: List[Tuple[float, float]] = []
    if start_point:
        all_map_points.append(start_point)
    if end_point:
        all_map_points.append(end_point)

    # 1. Start and Destination Markers
    if start_point:
        folium.Marker(
            location=start_point,
            popup="<b>Start Origin (Point A)</b>",
            tooltip="Origin (Point A)",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(m)

    if end_point:
        folium.Marker(
            location=end_point,
            popup="<b>Destination (Point B)</b>",
            tooltip="Destination (Point B)",
            icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
        ).add_to(m)

    # 2. Render ALL Alternative Routes with distinct vibrant colors
    for idx, r in enumerate(routes):
        rid = r.get("route_id", f"route_{idx + 1}")
        is_selected = (rid == selected_route_id)
        route_color = ALTERNATIVE_ROUTE_COLORS[idx % len(ALTERNATIVE_ROUTE_COLORS)]

        route_pts = extract_route_points(r, geom_cache)
        if not route_pts:
            continue
        all_map_points.extend(route_pts)

        agg_sc = r.get("aggregate_safety", 50.0)
        time_min = r.get("total_time", 0.0) / 60.0
        dist_km = r.get("total_length", 0.0) / 1000.0
        _, label = get_safety_color_and_label(agg_sc)
        expl = r.get("explanation", "")

        popup_html = (
            f"<div style='font-family:sans-serif; min-width:180px;'>"
            f"<h4 style='margin:0 0 4px 0; color:{route_color};'>{rid.replace('_', ' ').title()}</h4>"
            f"<b>Safety Rating:</b> {agg_sc:.1f}/100 ({label})<br/>"
            f"<b>Travel Time:</b> {time_min:.1f} mins<br/>"
            f"<b>Distance:</b> {dist_km:.2f} km<br/>"
            f"<p style='margin:6px 0 0 0; font-size:12px;'><i>{expl}</i></p>"
            f"</div>"
        )

        if not is_selected:
            # Render non-selected alternative routes as clearly visible colored paths
            folium.PolyLine(
                locations=route_pts,
                color=route_color,
                weight=5,
                opacity=0.85,
                dash_array="8, 6",
                tooltip=f"Alternative: {rid} | Safety: {agg_sc:.1f} [{label}] | {time_min:.1f}m ({dist_km:.2f}km)",
                popup=folium.Popup(popup_html, max_width=320),
            ).add_to(m)

    # 3. Render SELECTED Active Route prominently on top
    selected_route = next((r for r in routes if r.get("route_id") == selected_route_id), None)
    if selected_route:
        sel_idx = next((i for i, r in enumerate(routes) if r.get("route_id") == selected_route_id), 0)
        sel_color = ALTERNATIVE_ROUTE_COLORS[sel_idx % len(ALTERNATIVE_ROUTE_COLORS)]
        selected_pts = extract_route_points(selected_route, geom_cache)

        if selected_pts:
            # Outer casing outline to make selected route pop and glow over all layers
            folium.PolyLine(
                locations=selected_pts,
                color="#ffffff",
                weight=12,
                opacity=0.9,
            ).add_to(m)
            folium.PolyLine(
                locations=selected_pts,
                color="#212121",
                weight=10,
                opacity=0.75,
            ).add_to(m)

            # Selected route stroke
            agg_sc = selected_route.get("aggregate_safety", 50.0)
            time_min = selected_route.get("total_time", 0.0) / 60.0
            dist_km = selected_route.get("total_length", 0.0) / 1000.0
            _, label = get_safety_color_and_label(agg_sc)
            expl = selected_route.get("explanation", "")

            popup_html = (
                f"<div style='font-family:sans-serif; min-width:200px;'>"
                f"<h3 style='margin:0 0 4px 0; color:{sel_color};'>★ Selected: {selected_route_id.replace('_', ' ').title()}</h3>"
                f"<b>Safety Rating:</b> <span style='font-size:14px; font-weight:bold;'>{agg_sc:.1f}/100 ({label})</span><br/>"
                f"<b>Travel Time:</b> {time_min:.1f} mins<br/>"
                f"<b>Distance:</b> {dist_km:.2f} km<br/>"
                f"<p style='margin:6px 0 0 0; font-size:12px;'><i>{expl}</i></p>"
                f"</div>"
            )

            # Draw per-segment breakdown if segments are cached
            has_segments = False
            for sid in selected_route.get("segment_ids", []):
                coords = geom_cache.get_coords(sid)
                if coords:
                    has_segments = True
                    seg_info = scores_dict.get(sid)
                    score = getattr(seg_info, "safety_score", agg_sc) if seg_info else agg_sc
                    factors = getattr(seg_info, "top_factors", "corridor") if seg_info else "corridor"
                    seg_color, seg_label = get_safety_color_and_label(score)

                    folium.PolyLine(
                        locations=coords,
                        color=seg_color,
                        weight=7,
                        opacity=0.95,
                        tooltip=f"{selected_route_id} • {sid} | Score: {score:.1f} [{seg_label}]",
                        popup=folium.Popup(
                            f"<b>Segment:</b> {sid}<br/>"
                            f"<b>Score:</b> {score:.1f}/100 ({seg_label})<br/>"
                            f"<b>Factors:</b> {factors}",
                            max_width=280,
                        ),
                    ).add_to(m)

            if not has_segments:
                # Direct continuous polyline if segments are not partitioned
                folium.PolyLine(
                    locations=selected_pts,
                    color=sel_color,
                    weight=7,
                    opacity=0.95,
                    tooltip=f"★ Selected: {selected_route_id} | Safety: {agg_sc:.1f} [{label}]",
                    popup=folium.Popup(popup_html, max_width=320),
                ).add_to(m)

    # Auto-fit bounds
    if all_map_points:
        lats = [p[0] for p in all_map_points]
        lons = [p[1] for p in all_map_points]
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
