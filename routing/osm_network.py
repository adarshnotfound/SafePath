"""OpenStreetMap network and safety-proxy POI downloader for SafePath.

Handles downloading, caching, and loading the drivable road graph and
observable safety proxies (police stations, hospitals, petrol pumps, shops)
using OSMnx for a target city (default: Prayagraj, India).
"""

import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import geopandas as gpd
import networkx as nx
import pandas as pd
import requests
from shapely.geometry import LineString, Point

# Default city configuration for SafePath
DEFAULT_PLACE_NAME = "Prayagraj, Uttar Pradesh, India"
# In OSM Nominatim, the municipal city boundary of Prayagraj (~316.5 km2,
# relation 10029487) is indexed under its historic name while "Prayagraj,
# Uttar Pradesh, India" alone matches the 5,471 km2 rural district.
CITY_GEOCODE_ALIASES: Dict[str, str] = {
    "prayagraj, uttar pradesh, india": "Allahabad, Prayagraj, Uttar Pradesh, India",
    "prayagraj": "Allahabad, Prayagraj, Uttar Pradesh, India",
}
# Approximate city center of Prayagraj used only if no boundary polygon exists
PRAYAGRAJ_CENTER_LATLON: Tuple[float, float] = (25.4358, 81.8463)

# Public Overpass API base URLs (OSMnx appends "/interpreter")
OVERPASS_ENDPOINTS: List[str] = [
    "https://overpass-api.de/api",
    "https://overpass.private.coffee/api",
]

# Mapping from SafePath proxy category name to OSM tag filters
POI_TAG_FILTERS: Dict[str, Dict[str, Union[bool, str, list]]] = {
    "police": {"amenity": "police"},
    "hospital": {"amenity": ["hospital", "clinic"]},
    "petrol_pump": {"amenity": "fuel"},
    "shop": {"shop": True},
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def make_segment_id(u: int, v: int, key: int = 0) -> str:
    """Create a stable, deterministic segment identifier for an OSMnx edge.

    Uses the directed MultiDiGraph key tuple (u, v, key), where u and v are
    OpenStreetMap node IDs and key distinguishes parallel edges between the
    same intersection pair. This guarantees uniqueness per directed segment
    and allows direct lookup back into the NetworkX graph.
    """
    return f"seg_{int(u)}_{int(v)}_{int(key)}"


def assign_segment_ids(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Attach a deterministic `segment_id` attribute to every edge in-place."""
    for u, v, key, data in graph.edges(keys=True, data=True):
        data["segment_id"] = make_segment_id(u, v, key)
    return graph


def download_city_network(
    place_name: str = DEFAULT_PLACE_NAME,
    raw_dir: Union[str, Path] = DEFAULT_RAW_DIR,
    force_refresh: bool = False,
    fallback_radius_m: int = 12000,
) -> Tuple[nx.MultiDiGraph, Dict[str, Any]]:
    """Download or load the cached drivable OSM road network for a city.

    Parameters
    ----------
    place_name : str
        City name (e.g. `"Prayagraj, Uttar Pradesh, India"`).
    raw_dir : Union[str, Path]
        Directory where raw GraphML and OSMnx HTTP cache files are stored.
    force_refresh : bool
        If True, re-downloads from Overpass even if a local GraphML cache exists.
    fallback_radius_m : int
        Fallback radius around `PRAYAGRAJ_CENTER_LATLON` used only if Nominatim
        returns no administrative boundary polygon for `place_name`.

    Returns
    -------
    tuple[nx.MultiDiGraph, dict[str, Any]]
        The drivable MultiDiGraph with `segment_id` assigned, and provenance
        metadata describing how the graph was obtained.
    """
    import osmnx as ox

    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    slug = _slugify_place(place_name)
    graphml_path = raw_path / f"{slug}_drive.graphml"

    if graphml_path.exists() and not force_refresh:
        graph = ox.load_graphml(graphml_path)
        assign_segment_ids(graph)
        return graph, {
            "source": "OpenStreetMap (cached GraphML)",
            "place_name": place_name,
            "geocode_query": _resolve_geocode_query(place_name),
            "query_mode": "cached",
            "provenance": "real",
            "cache_path": str(graphml_path),
        }

    _configure_osmnx(ox, raw_path)
    geocode_query, polygon, boundary_area_km2 = _resolve_city_polygon(ox, place_name)

    if polygon is not None:
        query_mode = "graph_from_bbox_then_polygon_clip"
        graph, endpoint_used = _call_overpass_with_mirrors(
            ox, lambda: ox.graph_from_bbox(polygon.bounds, network_type="drive", simplify=True)
        )
        graph = ox.truncate.truncate_graph_polygon(graph, polygon)
        graph = ox.truncate.largest_component(graph, strongly=False)
    else:
        query_mode = f"graph_from_point_{fallback_radius_m}m"
        graph, endpoint_used = _call_overpass_with_mirrors(
            ox,
            lambda: ox.graph_from_point(
                PRAYAGRAJ_CENTER_LATLON,
                dist=fallback_radius_m,
                network_type="drive",
                simplify=True,
            ),
        )

    assign_segment_ids(graph)
    ox.save_graphml(graph, graphml_path)

    return graph, {
        "source": "OpenStreetMap via OSMnx",
        "place_name": place_name,
        "geocode_query": geocode_query,
        "query_mode": query_mode,
        "boundary_area_km2": boundary_area_km2,
        "overpass_endpoint": endpoint_used,
        "provenance": "real",
        "cache_path": str(graphml_path),
    }


def download_safety_pois(
    place_name: str = DEFAULT_PLACE_NAME,
    raw_dir: Union[str, Path] = DEFAULT_RAW_DIR,
    force_refresh: bool = False,
    fallback_radius_m: int = 12000,
    tag_filters: Optional[Dict[str, Dict[str, Union[bool, str, list]]]] = None,
) -> pd.DataFrame:
    """Download or load cached safety-proxy POIs (police, hospital, fuel, shop).

    Polygonal OSM features (such as hospital campuses or mall footprints) are
    projected to the local UTM CRS before computing centroids, then transformed
    back to WGS84 (lat/lon) so coordinates stay accurate without degree distortion.

    Returns
    -------
    pd.DataFrame
        Columns: `poi_id`, `poi_category`, `name`, `lat`, `lon`, `source`.
    """
    import osmnx as ox

    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    slug = _slugify_place(place_name)
    poi_csv_path = raw_path / f"{slug}_pois.csv"

    if poi_csv_path.exists() and not force_refresh:
        return pd.read_csv(poi_csv_path)

    _configure_osmnx(ox, raw_path)
    _, polygon, _ = _resolve_city_polygon(ox, place_name)

    records = []
    if tag_filters is None:
        records.extend(_fetch_pois_overpass_center(polygon, fallback_radius_m))
    else:
        for category, tags in tag_filters.items():
            gdf = _fetch_pois_for_category(ox, polygon, tags, fallback_radius_m)
            records.extend(_extract_poi_records_for_category(gdf, category))

    pois_df = pd.DataFrame(
        records,
        columns=["poi_id", "poi_category", "name", "lat", "lon", "source"],
    )
    pois_df.to_csv(poi_csv_path, index=False)
    return pois_df


def pois_dataframe_to_geodataframe(pois_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert a POI DataFrame with `lat` and `lon` columns into a GeoDataFrame."""
    clean_df = pois_df.dropna(subset=["lat", "lon"]).copy() if not pois_df.empty else pois_df.copy()
    if clean_df.empty:
        return gpd.GeoDataFrame(
            clean_df,
            geometry=gpd.GeoSeries([], crs="EPSG:4326"),
            crs="EPSG:4326",
        )
    geometries = [Point(float(lon), float(lat)) for lat, lon in zip(clean_df["lat"], clean_df["lon"])]
    return gpd.GeoDataFrame(clean_df, geometry=geometries, crs="EPSG:4326")


def graph_to_edge_geodataframe(graph: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """Extract an edges GeoDataFrame from a MultiDiGraph with `segment_id` ensured.

    Works on both real OSMnx graphs and lightweight synthetic test graphs
    where edge geometries may need to be built from node coordinates.
    """
    assign_segment_ids(graph)
    rows = []
    geometries = []

    for u, v, key, data in graph.edges(keys=True, data=True):
        geom = data.get("geometry")
        if geom is None:
            u_node = graph.nodes[u]
            v_node = graph.nodes[v]
            geom = LineString([(float(u_node["x"]), float(u_node["y"])), (float(v_node["x"]), float(v_node["y"]))])
        row = dict(data)
        row["u"] = int(u)
        row["v"] = int(v)
        row["key"] = int(key)
        row["segment_id"] = data.get("segment_id", make_segment_id(u, v, key))
        rows.append(row)
        geometries.append(geom)

    crs = graph.graph.get("crs", "EPSG:4326")
    return gpd.GeoDataFrame(rows, geometry=geometries, crs=crs)


def _configure_osmnx(ox_module: Any, raw_path: Path) -> None:
    """Keep OSMnx HTTP cache inside `data/raw/`, preserve `lit`, and set timeouts."""
    ox_module.settings.use_cache = True
    ox_module.settings.cache_folder = str(raw_path / "osmnx_cache")
    ox_module.settings.requests_timeout = 120
    ox_module.settings.overpass_rate_limit = False
    if "lit" not in ox_module.settings.useful_tags_way:
        ox_module.settings.useful_tags_way = list(ox_module.settings.useful_tags_way) + ["lit"]


def _resolve_geocode_query(place_name: str) -> str:
    """Map city names with district collisions to their municipal boundary query."""
    return CITY_GEOCODE_ALIASES.get(place_name.strip().lower(), place_name)


def _resolve_city_polygon(ox_module: Any, place_name: str) -> Tuple[str, Any, Optional[float]]:
    """Geocode `place_name` to a municipal boundary polygon and its area in km2."""
    geocode_query = _resolve_geocode_query(place_name)
    try:
        gdf = ox_module.geocode_to_gdf(geocode_query)
        polygon = gdf.geometry.iloc[0]
        utm_crs = gdf.estimate_utm_crs()
        area_km2 = round(float(gdf.to_crs(utm_crs).geometry.area.iloc[0]) / 1e6, 2)
        return geocode_query, polygon, area_km2
    except (TypeError, ValueError):
        # Nominatim did not return a polygon geometry (caller falls back to radius)
        return geocode_query, None, None


def _call_overpass_with_mirrors(ox_module: Any, fn: Callable[[], Any]) -> Tuple[Any, str]:
    """Try Overpass endpoints sequentially on transient gateway/rate-limit errors."""
    last_exc: Optional[Exception] = None
    for endpoint in OVERPASS_ENDPOINTS:
        ox_module.settings.overpass_url = endpoint
        try:
            return fn(), endpoint
        except Exception as exc:
            # Propagate genuine "no features matched" immediately without retrying mirrors
            if exc.__class__.__name__ == "InsufficientResponseError":
                raise
            last_exc = exc
            time.sleep(1.5)
    assert last_exc is not None
    raise last_exc


def _fetch_pois_for_category(
    ox_module: Any,
    polygon: Any,
    tags: Dict[str, Union[bool, str, Iterable[str]]],
    fallback_radius_m: int,
) -> gpd.GeoDataFrame:
    """Query OSMnx features for one POI category, returning empty GDF if none found."""
    try:
        if polygon is not None:
            gdf, _ = _call_overpass_with_mirrors(
                ox_module, lambda: ox_module.features_from_bbox(polygon.bounds, tags=tags)
            )
            time.sleep(1.0)
            if gdf.empty:
                return gdf
            return gdf[gdf.geometry.intersects(polygon)].copy()
        gdf, _ = _call_overpass_with_mirrors(
            ox_module,
            lambda: ox_module.features_from_point(
                PRAYAGRAJ_CENTER_LATLON,
                tags=tags,
                dist=fallback_radius_m,
            ),
        )
        time.sleep(1.0)
        return gdf
    except Exception as exc:
        if exc.__class__.__name__ == "InsufficientResponseError":
            return gpd.GeoDataFrame()
        raise


def _fetch_pois_overpass_center(polygon: Any, fallback_radius_m: int) -> List[Dict[str, Any]]:
    """Fetch safety-proxy POIs via fast Overpass `out center;` and clip to polygon.

    Using `out center;` avoids downloading full member geometries of large ways
    and relations, keeping Overpass queries fast and lightweight while still
    capturing nodes, buildings, and campus polygons.
    """
    if polygon is not None:
        west, south, east, north = polygon.bounds
        area_filter = f"({south:.6f},{west:.6f},{north:.6f},{east:.6f})"
    else:
        lat0, lon0 = PRAYAGRAJ_CENTER_LATLON
        area_filter = f"(around:{int(fallback_radius_m)},{lat0:.6f},{lon0:.6f})"

    query = (
        "[out:json][timeout:60];"
        "("
        f'nwr["amenity"~"^(police|hospital|clinic|fuel)$"]{area_filter};'
        f'nwr["shop"]{area_filter};'
        ");"
        "out center;"
    )

    last_exc: Optional[Exception] = None
    payload: Optional[Dict[str, Any]] = None
    headers = {"User-Agent": "SafePath-Academic-Research/1.0"}

    for base_url in OVERPASS_ENDPOINTS:
        interpreter_url = base_url.rstrip("/") + "/interpreter"
        try:
            resp = requests.post(interpreter_url, data={"data": query}, headers=headers, timeout=60)
            resp.raise_for_status()
            payload = resp.json()
            break
        except Exception as exc:
            last_exc = exc
            time.sleep(1.5)

    if payload is None:
        assert last_exc is not None
        raise last_exc

    counters: Dict[str, int] = {"police": 0, "hospital": 0, "petrol_pump": 0, "shop": 0}
    records: List[Dict[str, Any]] = []

    for elem in payload.get("elements", []):
        tags = elem.get("tags", {})
        if not tags:
            continue

        if "lat" in elem and "lon" in elem:
            lat_val = float(elem["lat"])
            lon_val = float(elem["lon"])
        elif "center" in elem:
            lat_val = float(elem["center"]["lat"])
            lon_val = float(elem["center"]["lon"])
        else:
            continue

        if polygon is not None and not Point(lon_val, lat_val).intersects(polygon):
            continue

        clean_name = _clean_poi_name(tags.get("name", ""))
        amenity_parts = {p.strip() for p in str(tags.get("amenity", "")).lower().split(";") if p.strip()}
        shop_str = str(tags.get("shop", "")).strip().lower()

        matched_cats: List[str] = []
        if "police" in amenity_parts:
            matched_cats.append("police")
        if amenity_parts & {"hospital", "clinic"}:
            matched_cats.append("hospital")
        if "fuel" in amenity_parts:
            matched_cats.append("petrol_pump")
        if shop_str and shop_str != "no":
            matched_cats.append("shop")

        for cat in matched_cats:
            idx = counters[cat]
            counters[cat] += 1
            records.append(
                {
                    "poi_id": f"{cat}_{idx}",
                    "poi_category": cat,
                    "name": clean_name,
                    "lat": round(lat_val, 7),
                    "lon": round(lon_val, 7),
                    "source": "OpenStreetMap (real)",
                }
            )
    return records


def _extract_poi_records_for_category(gdf: gpd.GeoDataFrame, category: str) -> List[Dict[str, Any]]:
    """Compute UTM centroids and extract standardized POI records for one category."""
    if gdf.empty:
        return []
    valid = gdf[gdf.geometry.notna()].copy()
    if valid.empty:
        return []
    if valid.crs is None:
        valid = valid.set_crs("EPSG:4326")

    utm_crs = valid.estimate_utm_crs()
    centroids_proj = valid.to_crs(utm_crs).geometry.centroid
    centroids_wgs84 = gpd.GeoSeries(centroids_proj, crs=utm_crs).to_crs("EPSG:4326")
    names = valid["name"] if "name" in valid.columns else pd.Series([""] * len(valid), index=valid.index)

    records: List[Dict[str, Any]] = []
    for idx, (geom, raw_name) in enumerate(zip(centroids_wgs84, names)):
        if geom is None or geom.is_empty:
            continue
        records.append(
            {
                "poi_id": f"{category}_{idx}",
                "poi_category": category,
                "name": _clean_poi_name(raw_name),
                "lat": round(float(geom.y), 7),
                "lon": round(float(geom.x), 7),
                "source": "OpenStreetMap (real)",
            }
        )
    return records


def _clean_poi_name(raw_val: Any) -> str:
    """Safely convert an OSM scalar or list tag value into a clean string."""
    if raw_val is None:
        return ""
    if isinstance(raw_val, (list, tuple, set)):
        return ";".join(str(x).strip() for x in raw_val if x is not None and str(x).strip())
    if isinstance(raw_val, float) and pd.isna(raw_val):
        return ""
    text = str(raw_val).strip()
    return "" if text.lower() == "nan" else text


def _slugify_place(place_name: str) -> str:
    """Convert a place string like 'Prayagraj, Uttar Pradesh, India' to a filename slug."""
    head = place_name.split(",")[0].strip().lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in head)
    return "_".join(part for part in cleaned.split("_") if part) or "city"
