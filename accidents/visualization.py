"""Interactive Folium Map Visualization for Accident Hotspots - SafePath Module B3.

Renders an interactive geospatial Folium map displaying:
1. Individual historical accident points (styled with severity-based radius and color).
2. Clustered accident blackspot zones (concentric circular buffers styled by severity weight).
3. Hotspot markers with interactive HTML popups summarizing accident counts and segment IDs.
4. Road network segment polylines for context.
5. Interactive layer toggles for blackspots, raw accidents, and road corridors.
6. Clear data provenance badges (India vs. substitute non-Indian data).

Owner: Member B (Hazard map and accident-data lead)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import folium
from folium import FeatureGroup, LayerControl

from accidents.clustering import AccidentHotspot
from accidents.data_loader import DEFAULT_ROUTE_SEGMENTS, AccidentPoint


def get_severity_color(severity: float) -> str:
    """Return hex color reflecting accident severity."""
    if severity >= 4.0:
        return "#E74C3C"  # Dark Red (High / Fatal severity)
    elif severity >= 2.8:
        return "#E67E22"  # Orange (Medium severity)
    else:
        return "#F1C40F"  # Yellow (Minor severity)


def plot_accident_hotspots_folium(
    hotspots: List[AccidentHotspot],
    accidents: Optional[List[AccidentPoint]] = None,
    segments: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    output_html_path: Optional[Path] = None,
    source_country: str = "India",
) -> folium.Map:
    """Generate an interactive Folium map visualizing accident hotspots and incidents.

    Args:
        hotspots: List of clustered AccidentHotspot objects.
        accidents: Optional list of raw individual AccidentPoint events.
        segments: Road network segment waypoints.
        output_html_path: Destination path for HTML map file.
        source_country: Country label ('India' or substitute e.g. 'United Kingdom').

    Returns:
        folium.Map instance.
    """
    repo_root = Path(__file__).resolve().parent.parent

    if output_html_path is None:
        output_html_path = repo_root / "evaluation" / "plots" / "accident_hotspots_map.html"

    # Default center to Bengaluru corridor coordinates
    if hotspots:
        center_lat = sum(h.lat for h in hotspots) / len(hotspots)
        center_lon = sum(h.lon for h in hotspots) / len(hotspots)
    else:
        center_lat, center_lon = 12.9760, 77.6000

    # Initialize Folium base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # 1. Road Network Segment Layer
    seg_dict = segments if segments is not None else DEFAULT_ROUTE_SEGMENTS
    layer_roads = FeatureGroup(name="Road Network Segments", show=True)
    for seg_id, waypoints in seg_dict.items():
        folium.PolyLine(
            locations=waypoints,
            color="#34495E",
            weight=3.5,
            opacity=0.7,
            dash_array="5, 5",
            tooltip=f"Road Segment: {seg_id}",
        ).add_to(layer_roads)
    layer_roads.add_to(m)

    # 2. Raw Accident Events Layer
    if accidents:
        layer_accidents = FeatureGroup(name=f"Accident Events ({len(accidents)})", show=True)
        for acc in accidents:
            color = get_severity_color(acc.severity)
            tooltip_html = (
                f"<b>Accident:</b> {acc.accident_id}<br>"
                f"<b>Severity:</b> {acc.severity:.1f}/5.0<br>"
                f"<b>Time:</b> {acc.timestamp}<br>"
                f"<b>Weather:</b> {acc.weather} | <b>Lighting:</b> {acc.lighting}<br>"
                f"<b>Country:</b> {acc.source_country}"
            )
            folium.CircleMarker(
                location=[acc.lat, acc.lon],
                radius=3.5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                weight=1,
                tooltip=tooltip_html,
            ).add_to(layer_accidents)
        layer_accidents.add_to(m)

    # 3. Hotspot Blackspot Zones Layer
    layer_hotspots = FeatureGroup(name=f"Accident Blackspots ({len(hotspots)})", show=True)
    is_indian = source_country.lower() == "india"
    provenance_badge = (
        "<span style='color: green; font-weight: bold;'>[Native Indian Data]</span>"
        if is_indian
        else "<span style='color: red; font-weight: bold;'>[Substitute Data - Non-India]</span>"
    )

    for h in hotspots:
        color = get_severity_color(h.severity_weight)

        # Danger zone circular buffer
        folium.Circle(
            location=[h.lat, h.lon],
            radius=h.radius_m,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.25,
            tooltip=f"Blackspot Zone: {h.hotspot_id} (Radius: {h.radius_m:.1f}m)",
        ).add_to(layer_hotspots)

        # Centroid marker with informative popup table
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; min-width: 200px;">
            <h4 style="margin: 0 0 8px 0; color: #2C3E50;">Hotspot {h.hotspot_id}</h4>
            <p style="margin: 2px 0;">{provenance_badge}</p>
            <table style="width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 6px;">
                <tr><td style="font-weight: bold;">Accident Count:</td><td>{h.accident_count}</td></tr>
                <tr><td style="font-weight: bold;">Severity Weight:</td><td>{h.severity_weight:.1f} / 5.0</td></tr>
                <tr><td style="font-weight: bold;">Hazard Radius:</td><td>{h.radius_m:.1f} m</td></tr>
                <tr><td style="font-weight: bold;">Nearest Segment:</td><td>{h.segment_id}</td></tr>
                <tr><td style="font-weight: bold;">Coordinates:</td><td>{h.lat:.5f}, {h.lon:.5f}</td></tr>
                <tr><td style="font-weight: bold;">Source Country:</td><td>{h.source_country}</td></tr>
            </table>
        </div>
        """

        folium.Marker(
            location=[h.lat, h.lon],
            icon=folium.Icon(color="red" if h.severity_weight >= 3.5 else "orange", icon="warning-sign"),
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{h.hotspot_id}: {h.accident_count} accidents (Severity {h.severity_weight:.1f})",
        ).add_to(layer_hotspots)

    layer_hotspots.add_to(m)

    # 4. Add interactive layer control
    LayerControl(collapsed=False).add_to(m)

    # Save to HTML
    output_html_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_html_path))

    # Also save copy in evaluation/results for unified reporting
    results_copy_path = repo_root / "evaluation" / "results" / "accident_hotspots_map.html"
    results_copy_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(results_copy_path))

    return m
