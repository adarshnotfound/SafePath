"""Build Section 8 segment_features.csv from an OpenStreetMap road network.

Extracts per-segment road attributes (`length_m`, `road_type`, `speed_limit`,
`lit` with explicit `'unknown'`), computes a metric-CRS `isolation_score`
from distance to the nearest police station, hospital, petrol pump, and shop,
and leaves downstream-owned columns empty (`""`) for Modules 2-3 to populate.
"""

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.neighbors import NearestNeighbors

from routing.osm_network import (
    DEFAULT_PROCESSED_DIR,
    POI_TAG_FILTERS,
    graph_to_edge_geodataframe,
    pois_dataframe_to_geodataframe,
)

# Exact Section 8 file contract column order for segment_features.csv
SECTION_8_SEGMENT_COLUMNS: List[str] = [
    "segment_id",
    "length_m",
    "road_type",
    "speed_limit",
    "lit",
    "hazard_density",
    "mean_hazard_confidence",
    "mean_freshness",
    "accident_score",
    "isolation_score",
    "time_of_day",
]

# Ordered highway priority when an OSM edge carries multiple highway tags
ROAD_TYPE_PRIORITY: List[str] = [
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "service",
]

# Conservative Indian urban road-type fallback speeds (km/h) for metadata / routing
ROAD_TYPE_FALLBACK_SPEED_KMH: Dict[str, float] = {
    "motorway": 80.0,
    "trunk": 60.0,
    "primary": 50.0,
    "secondary": 40.0,
    "tertiary": 30.0,
    "unclassified": 25.0,
    "residential": 25.0,
    "living_street": 15.0,
    "service": 20.0,
}

LIT_TRUE_VALUES = {"yes", "true", "1", "24/7", "automatic", "limited", "sunset-sunrise", "dusk-dawn", "interval"}
LIT_FALSE_VALUES = {"no", "false", "0", "disused"}
MPH_TO_KMH = 1.60934


def normalize_road_type(raw_highway: Any) -> str:
    """Normalize an OSM `highway` tag (string, list, or NaN) into one class string."""
    if raw_highway is None or (isinstance(raw_highway, float) and math.isnan(raw_highway)):
        return "unknown"

    if isinstance(raw_highway, (list, tuple, set)):
        candidates = [str(item).strip().lower() for item in raw_highway if item is not None and str(item).strip()]
    else:
        text = str(raw_highway).strip().lower()
        if not text or text == "nan":
            return "unknown"
        candidates = [part.strip() for part in text.split(";") if part.strip()]

    if not candidates:
        return "unknown"

    # Prefer higher-hierarchy class when multiple tags exist on one merged edge
    for priority_cls in ROAD_TYPE_PRIORITY:
        for cand in candidates:
            base = cand.removesuffix("_link")
            if cand == priority_cls or base == priority_cls:
                return cand
    return candidates[0]


def normalize_lit_tag(raw_lit: Any) -> str:
    """Normalize an OSM `lit` tag into `'yes'`, `'no'`, or explicit `'unknown'`."""
    if raw_lit is None or (isinstance(raw_lit, float) and math.isnan(raw_lit)):
        return "unknown"

    if isinstance(raw_lit, (list, tuple, set)):
        values = [str(item).strip().lower() for item in raw_lit if item is not None and str(item).strip()]
    else:
        text = str(raw_lit).strip().lower()
        if not text or text == "nan":
            return "unknown"
        values = [part.strip() for part in text.split(";") if part.strip()]

    if any(v in LIT_TRUE_VALUES for v in values):
        return "yes"
    if any(v in LIT_FALSE_VALUES for v in values):
        return "no"
    return "unknown"


def parse_speed_limit(raw_maxspeed: Any) -> Optional[float]:
    """Parse an OSM `maxspeed` tag into numeric km/h, or None if unavailable/invalid."""
    if raw_maxspeed is None or (isinstance(raw_maxspeed, float) and math.isnan(raw_maxspeed)):
        return None

    if isinstance(raw_maxspeed, (list, tuple, set)):
        parsed = [parse_speed_limit(item) for item in raw_maxspeed]
        valid = [val for val in parsed if val is not None]
        return round(float(np.mean(valid)), 1) if valid else None

    if isinstance(raw_maxspeed, (int, float)):
        val = float(raw_maxspeed)
        return round(val, 1) if 0 < val <= 160 else None

    text = str(raw_maxspeed).strip().lower()
    if not text or text in {"nan", "none", "signals", "variable", "implicit"}:
        return None

    if ";" in text:
        parts = [parse_speed_limit(part) for part in text.split(";")]
        valid = [val for val in parts if val is not None]
        return round(float(np.mean(valid)), 1) if valid else None

    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return None

    value = float(match.group(1))
    if "mph" in text:
        value *= MPH_TO_KMH

    if value <= 0 or value > 160:
        return None
    return round(value, 1)


def estimate_speed_limit_fallback(road_type: str) -> float:
    """Return a conservative urban Indian speed fallback (km/h) from `road_type`."""
    cleaned = str(road_type).strip().lower().removesuffix("_link")
    return ROAD_TYPE_FALLBACK_SPEED_KMH.get(cleaned, 30.0)


def compute_isolation_scores(
    edges_proj: gpd.GeoDataFrame,
    pois_proj: gpd.GeoDataFrame,
    categories: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compute metric distance to nearest POIs and normalized `isolation_score`.

    Distances are computed in metres on projected geometries (`edges_proj` and
    `pois_proj` must share the same projected metric CRS). Each edge is
    represented by the midpoint along its polyline. The mean distance across
    available POI categories is min-max normalized using the 95th percentile as
    the upper cap so far peripheral outliers do not compress urban scores.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Per-edge distance table (`dist_<cat>_m`, `mean_poi_distance_m`,
        `isolation_score`) aligned to `edges_proj.index`, and summary statistics.
    """
    cats = categories or list(POI_TAG_FILTERS.keys())
    if edges_proj.empty:
        empty_cols = [f"dist_{c}_m" for c in cats] + ["mean_poi_distance_m", "isolation_score"]
        return pd.DataFrame(columns=empty_cols), {"poi_counts": {}, "categories_available": 0}

    # Representative midpoint along each edge polyline in projected metres
    midpoints = edges_proj.geometry.interpolate(0.5, normalized=True)
    edge_xy = np.column_stack([midpoints.x.to_numpy(dtype=float), midpoints.y.to_numpy(dtype=float)])

    dist_df = pd.DataFrame(index=edges_proj.index)
    poi_counts: Dict[str, int] = {}
    category_medians_m: Dict[str, Optional[float]] = {}

    for cat in cats:
        col_name = f"dist_{cat}_m"
        if pois_proj.empty or "poi_category" not in pois_proj.columns:
            cat_pois = gpd.GeoDataFrame()
        else:
            cat_pois = pois_proj[pois_proj["poi_category"] == cat]

        poi_counts[cat] = int(len(cat_pois))
        if cat_pois.empty:
            dist_df[col_name] = np.nan
            category_medians_m[cat] = None
            continue

        poi_xy = np.column_stack(
            [cat_pois.geometry.x.to_numpy(dtype=float), cat_pois.geometry.y.to_numpy(dtype=float)]
        )
        nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        nn.fit(poi_xy)
        distances_m, _ = nn.kneighbors(edge_xy)
        flat_dists = distances_m[:, 0]
        dist_df[col_name] = np.round(flat_dists, 2)
        category_medians_m[cat] = round(float(np.median(flat_dists)), 2)

    dist_cols = [f"dist_{c}_m" for c in cats]
    available_cols = [col for col in dist_cols if not dist_df[col].isna().all()]

    if not available_cols:
        # Neutral fallback if zero POIs were returned across all categories
        dist_df["mean_poi_distance_m"] = np.nan
        dist_df["isolation_score"] = 0.5
        summary = {
            "poi_counts": poi_counts,
            "category_medians_m": category_medians_m,
            "categories_available": 0,
            "normalization": "neutral_fallback_no_pois",
        }
        return dist_df, summary

    mean_dist = dist_df[available_cols].mean(axis=1, skipna=True).to_numpy(dtype=float)
    dist_df["mean_poi_distance_m"] = np.round(mean_dist, 2)

    d_min = float(np.nanmin(mean_dist))
    d_p95 = float(np.nanpercentile(mean_dist, 95))
    d_max = float(np.nanmax(mean_dist))

    denom = d_p95 - d_min
    if denom <= 1e-6:
        denom = d_max - d_min

    if denom <= 1e-6:
        norm_scores = np.zeros_like(mean_dist, dtype=float)
    else:
        clipped = np.clip(mean_dist, d_min, d_min + denom)
        norm_scores = np.clip((clipped - d_min) / denom, 0.0, 1.0)

    dist_df["isolation_score"] = np.round(norm_scores, 4)

    summary = {
        "poi_counts": poi_counts,
        "category_medians_m": category_medians_m,
        "categories_available": len(available_cols),
        "normalization": "minmax_p95_clipped",
        "mean_poi_distance_m_stats": {
            "min": round(d_min, 2),
            "median": round(float(np.nanmedian(mean_dist)), 2),
            "mean": round(float(np.nanmean(mean_dist)), 2),
            "p95": round(d_p95, 2),
            "max": round(d_max, 2),
        },
        "isolation_score_stats": {
            "min": round(float(np.min(norm_scores)), 4),
            "median": round(float(np.median(norm_scores)), 4),
            "mean": round(float(np.mean(norm_scores)), 4),
            "max": round(float(np.max(norm_scores)), 4),
        },
    }
    return dist_df, summary


def compute_tag_coverage(
    features_df: pd.DataFrame,
    raw_speed_present_mask: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """Compute honest OSM tag coverage counts and percentages across all edges."""
    total_edges = int(len(features_df))
    if total_edges == 0:
        return {"total_edges": 0}

    total_length_km = round(float(features_df["length_m"].sum()) / 1000.0, 3)

    lit_counts = features_df["lit"].value_counts(dropna=False).to_dict()
    lit_yes = int(lit_counts.get("yes", 0))
    lit_no = int(lit_counts.get("no", 0))
    lit_unknown = int(lit_counts.get("unknown", 0))
    lit_tagged = lit_yes + lit_no

    if raw_speed_present_mask is not None:
        speed_present = int(raw_speed_present_mask.sum())
    else:
        speed_series = features_df["speed_limit"].replace("", np.nan)
        speed_present = int(speed_series.notna().sum())
    speed_missing = total_edges - speed_present

    road_type_counts = {str(k): int(v) for k, v in features_df["road_type"].value_counts().items()}
    road_type_known = total_edges - int(road_type_counts.get("unknown", 0))

    return {
        "total_edges": total_edges,
        "total_length_km": total_length_km,
        "lit_coverage": {
            "yes_count": lit_yes,
            "no_count": lit_no,
            "unknown_count": lit_unknown,
            "explicitly_tagged_count": lit_tagged,
            "explicitly_tagged_pct": round(100.0 * lit_tagged / total_edges, 2),
            "yes_pct": round(100.0 * lit_yes / total_edges, 2),
            "no_pct": round(100.0 * lit_no / total_edges, 2),
            "unknown_pct": round(100.0 * lit_unknown / total_edges, 2),
        },
        "speed_limit_coverage": {
            "osm_tagged_count": speed_present,
            "missing_count": speed_missing,
            "osm_tagged_pct": round(100.0 * speed_present / total_edges, 2),
        },
        "road_type_coverage": {
            "known_count": road_type_known,
            "known_pct": round(100.0 * road_type_known / total_edges, 2),
            "distribution": road_type_counts,
        },
    }


def build_segment_features(
    graph: nx.MultiDiGraph,
    pois_df: pd.DataFrame,
    impute_missing_speed: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Build Section 8 `segment_features` DataFrame and sidecar `segment_metadata`.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        Drivable road network from OSMnx (or synthetic test graph).
    pois_df : pd.DataFrame
        Safety-proxy POIs (`poi_category`, `lat`, `lon`).
    impute_missing_speed : bool
        If False (default, strictest data honesty), missing OSM `maxspeed` stays
        empty (`""`) in `segment_features.csv` while fallback speeds are recorded
        in `segment_metadata.csv`. If True, uses `ROAD_TYPE_FALLBACK_SPEED_KMH`.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict]
        `(features_df, metadata_df, summary_dict)`.
    """
    edges_gdf = graph_to_edge_geodataframe(graph)
    if edges_gdf.crs is None:
        edges_gdf = edges_gdf.set_crs("EPSG:4326")

    pois_gdf = pois_dataframe_to_geodataframe(pois_df)

    # Project both road segments and POIs to the same local metric UTM CRS
    utm_crs = edges_gdf.estimate_utm_crs()
    edges_proj = edges_gdf.to_crs(utm_crs)
    pois_proj = pois_gdf.to_crs(utm_crs) if not pois_gdf.empty else pois_gdf

    isolation_df, isolation_summary = compute_isolation_scores(edges_proj, pois_proj)

    # Compute WGS84 midpoints for map plotting and snapping metadata
    midpoints_proj = edges_proj.geometry.interpolate(0.5, normalized=True)
    transformer = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    mid_lons, mid_lats = transformer.transform(
        midpoints_proj.x.to_numpy(dtype=float),
        midpoints_proj.y.to_numpy(dtype=float),
    )

    # Edge length: prefer OSMnx `length` tag when valid, otherwise projected length
    proj_lengths = edges_proj.geometry.length.to_numpy(dtype=float)
    if "length" in edges_gdf.columns:
        raw_lengths = pd.to_numeric(edges_gdf["length"], errors="coerce").to_numpy(dtype=float)
        lengths_m = np.where(np.isfinite(raw_lengths) & (raw_lengths > 0), raw_lengths, proj_lengths)
    else:
        lengths_m = proj_lengths

    highway_col = edges_gdf["highway"] if "highway" in edges_gdf.columns else pd.Series([None] * len(edges_gdf))
    lit_col = edges_gdf["lit"] if "lit" in edges_gdf.columns else pd.Series([None] * len(edges_gdf))
    maxspeed_col = edges_gdf["maxspeed"] if "maxspeed" in edges_gdf.columns else pd.Series([None] * len(edges_gdf))

    road_types = [normalize_road_type(val) for val in highway_col]
    lit_values = [normalize_lit_tag(val) for val in lit_col]
    parsed_speeds = [parse_speed_limit(val) for val in maxspeed_col]
    fallback_speeds = [estimate_speed_limit_fallback(rt) for rt in road_types]

    raw_speed_mask = pd.Series([sp is not None for sp in parsed_speeds], index=edges_gdf.index)
    speed_sources = ["osm" if sp is not None else ("road_type_heuristic" if impute_missing_speed else "missing") for sp in parsed_speeds]

    output_speeds: List[Union[float, str]] = []
    for sp, fb in zip(parsed_speeds, fallback_speeds):
        if sp is not None:
            output_speeds.append(sp)
        elif impute_missing_speed:
            output_speeds.append(fb)
        else:
            output_speeds.append("")

    # Build strict Section 8 contract DataFrame
    features_df = pd.DataFrame(
        {
            "segment_id": edges_gdf["segment_id"].astype(str).tolist(),
            "length_m": np.round(lengths_m, 2),
            "road_type": road_types,
            "speed_limit": output_speeds,
            "lit": lit_values,
            "hazard_density": "",
            "mean_hazard_confidence": "",
            "mean_freshness": "",
            "accident_score": "",
            "isolation_score": isolation_df["isolation_score"].to_numpy(dtype=float),
            "time_of_day": "",
        },
        columns=SECTION_8_SEGMENT_COLUMNS,
    )

    osmid_col = [_stringify_osm_value(v) for v in edges_gdf["osmid"]] if "osmid" in edges_gdf.columns else [""] * len(edges_gdf)
    name_col = [_stringify_osm_value(v) for v in edges_gdf["name"]] if "name" in edges_gdf.columns else [""] * len(edges_gdf)

    metadata_df = pd.DataFrame(
        {
            "segment_id": edges_gdf["segment_id"].astype(str).tolist(),
            "u": edges_gdf["u"].astype(int).tolist(),
            "v": edges_gdf["v"].astype(int).tolist(),
            "key": edges_gdf["key"].astype(int).tolist(),
            "osmid": osmid_col,
            "name": name_col,
            "mid_lat": np.round(mid_lats, 7),
            "mid_lon": np.round(mid_lons, 7),
            "speed_limit_osm_kmh": ["" if sp is None else sp for sp in parsed_speeds],
            "speed_limit_fallback_kmh": fallback_speeds,
            "speed_limit_source": speed_sources,
            "dist_police_m": isolation_df["dist_police_m"].to_numpy(),
            "dist_hospital_m": isolation_df["dist_hospital_m"].to_numpy(),
            "dist_petrol_pump_m": isolation_df["dist_petrol_pump_m"].to_numpy(),
            "dist_shop_m": isolation_df["dist_shop_m"].to_numpy(),
            "mean_poi_distance_m": isolation_df["mean_poi_distance_m"].to_numpy(),
            "geometry_wkt": [geom.wkt for geom in edges_gdf.geometry],
        }
    )

    tag_coverage = compute_tag_coverage(features_df, raw_speed_present_mask=raw_speed_mask)
    summary = {
        "utm_crs": str(utm_crs),
        "speed_limit_imputed_in_features_csv": impute_missing_speed,
        "tag_coverage": tag_coverage,
        "isolation_summary": isolation_summary,
    }
    return features_df, metadata_df, summary


def save_segment_outputs(
    features_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    output_dir: Union[str, Path] = DEFAULT_PROCESSED_DIR,
    features_filename: str = "segment_features.csv",
    metadata_filename: str = "segment_metadata.csv",
) -> Tuple[Path, Path]:
    """Save Section 8 `segment_features.csv` and sidecar `segment_metadata.csv`."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    features_path = out_path / features_filename
    metadata_path = out_path / metadata_filename

    # Ensure exact column order before writing
    features_df[SECTION_8_SEGMENT_COLUMNS].to_csv(features_path, index=False)
    metadata_df.to_csv(metadata_path, index=False)
    return features_path, metadata_path


def _stringify_osm_value(value: Any) -> str:
    """Safely convert an OSM scalar or list attribute into a clean string."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = [str(item).strip() for item in value if item is not None and str(item).strip()]
        return ";".join(parts)
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text
