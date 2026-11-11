"""Snap geographic coordinates (lat, lon) to the nearest OSM road segment.

Uses a projected metric Coordinate Reference System (UTM) and a Shapely
STRtree spatial index over edge LineString geometries so distances are always
measured in metres rather than WGS84 degrees.
"""

from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkt
from shapely.geometry import Point
from shapely.strtree import STRtree
from sklearn.neighbors import NearestNeighbors

from routing.osm_network import graph_to_edge_geodataframe, make_segment_id


class SegmentSnapper:
    """Fast spatial-index snapper mapping `(lat, lon)` points to nearest road edge."""

    def __init__(self, edges_gdf: gpd.GeoDataFrame) -> None:
        """Build a projected STRtree spatial index from an edges GeoDataFrame."""
        if edges_gdf.empty:
            raise ValueError("Cannot initialize SegmentSnapper with an empty edges GeoDataFrame.")

        gdf = edges_gdf.copy().reset_index(drop=True)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")

        self.utm_crs = gdf.estimate_utm_crs()
        self.edges_proj = gdf.to_crs(self.utm_crs)
        self.geoms_proj = list(self.edges_proj.geometry)
        self.tree = STRtree(self.geoms_proj)

        self.segment_ids: List[str] = self.edges_proj["segment_id"].astype(str).tolist()
        self.u_ids: List[int] = [int(val) for val in self.edges_proj["u"]]
        self.v_ids: List[int] = [int(val) for val in self.edges_proj["v"]]
        self.keys: List[int] = [int(val) for val in self.edges_proj["key"]]

        self._to_proj = Transformer.from_crs("EPSG:4326", self.utm_crs, always_xy=True)
        self._to_wgs84 = Transformer.from_crs(self.utm_crs, "EPSG:4326", always_xy=True)

    @classmethod
    def from_graph(cls, graph: nx.MultiDiGraph) -> "SegmentSnapper":
        """Create a SegmentSnapper directly from an OSMnx / NetworkX MultiDiGraph."""
        edges_gdf = graph_to_edge_geodataframe(graph)
        return cls(edges_gdf)

    @classmethod
    def from_metadata_csv(cls, metadata_path: Union[str, Path]) -> "SegmentSnapper":
        """Create a SegmentSnapper from `data/processed/segment_metadata.csv`."""
        df = pd.read_csv(metadata_path)
        required = {"segment_id", "u", "v", "key", "geometry_wkt"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Metadata CSV is missing required columns: {sorted(missing)}")

        geometries = [wkt.loads(text) for text in df["geometry_wkt"]]
        edges_gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")
        return cls(edges_gdf)

    def snap(self, lat: float, lon: float) -> Dict[str, Any]:
        """Snap a single `(lat, lon)` coordinate to the nearest road segment.

        Returns
        -------
        dict
            Keys: `segment_id`, `u`, `v`, `key`, `distance_m`, `snapped_lat`,
            `snapped_lon`.
        """
        _validate_lat_lon(lat, lon)
        x_proj, y_proj = self._to_proj.transform(float(lon), float(lat))
        query_pt = Point(x_proj, y_proj)

        idx = int(self.tree.nearest(query_pt))
        edge_geom = self.geoms_proj[idx]

        distance_m = float(query_pt.distance(edge_geom))
        snapped_proj = edge_geom.interpolate(edge_geom.project(query_pt))
        snapped_lon, snapped_lat = self._to_wgs84.transform(snapped_proj.x, snapped_proj.y)

        return {
            "segment_id": self.segment_ids[idx],
            "u": self.u_ids[idx],
            "v": self.v_ids[idx],
            "key": self.keys[idx],
            "distance_m": round(distance_m, 3),
            "snapped_lat": round(float(snapped_lat), 7),
            "snapped_lon": round(float(snapped_lon), 7),
        }

    def snap_many(self, coords: Sequence[Tuple[float, float]]) -> pd.DataFrame:
        """Snap a sequence of `(lat, lon)` tuples and return a DataFrame."""
        results = [self.snap(lat, lon) for lat, lon in coords]
        return pd.DataFrame(results)


class NearestNodeSnapBaseline:
    """Baseline snapper that maps `(lat, lon)` to the nearest intersection node.

    Used as the baseline in `evaluate_features.py` to demonstrate why snapping
    to full edge geometries outperforms node-only snapping on long road segments.
    """

    def __init__(self, graph: nx.MultiDiGraph, utm_crs: Any) -> None:
        """Index graph nodes in projected metric coordinates."""
        node_records = []
        for node_id, data in graph.nodes(data=True):
            # Pick a deterministic incident edge for the node
            incident_edge = None
            for u, v, key in graph.out_edges(node_id, keys=True):
                incident_edge = (int(u), int(v), int(key))
                break
            if incident_edge is None:
                for u, v, key in graph.in_edges(node_id, keys=True):
                    incident_edge = (int(u), int(v), int(key))
                    break
            if incident_edge is None:
                continue

            node_records.append(
                {
                    "node_id": int(node_id),
                    "lon": float(data["x"]),
                    "lat": float(data["y"]),
                    "segment_id": make_segment_id(*incident_edge),
                }
            )

        if not node_records:
            raise ValueError("Graph has no connected nodes for NearestNodeSnapBaseline.")

        self._to_proj = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        df = pd.DataFrame(node_records)
        xs, ys = self._to_proj.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
        self.node_xy = np.column_stack([xs, ys])
        self.segment_ids = df["segment_id"].tolist()

        self.nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
        self.nn.fit(self.node_xy)

    def snap(self, lat: float, lon: float) -> Dict[str, Any]:
        """Snap `(lat, lon)` via nearest graph node and return segment_id + distance_m."""
        _validate_lat_lon(lat, lon)
        x_proj, y_proj = self._to_proj.transform(float(lon), float(lat))
        distances_m, indices = self.nn.kneighbors(np.array([[x_proj, y_proj]]))
        idx = int(indices[0, 0])
        return {
            "segment_id": self.segment_ids[idx],
            "distance_m": round(float(distances_m[0, 0]), 3),
        }


def snap_to_segment(
    lat: float,
    lon: float,
    target: Union[SegmentSnapper, nx.MultiDiGraph, gpd.GeoDataFrame],
) -> Dict[str, Any]:
    """Convenience function to snap a single `(lat, lon)` to the nearest road segment.

    Parameters
    ----------
    lat : float
        Latitude in WGS84 decimal degrees.
    lon : float
        Longitude in WGS84 decimal degrees.
    target : SegmentSnapper | nx.MultiDiGraph | gpd.GeoDataFrame
        A pre-built `SegmentSnapper` (fastest for repeated queries), an OSMnx
        `MultiDiGraph`, or an edges `GeoDataFrame`.
    """
    if isinstance(target, SegmentSnapper):
        return target.snap(lat, lon)
    if isinstance(target, nx.MultiDiGraph):
        return SegmentSnapper.from_graph(target).snap(lat, lon)
    if isinstance(target, gpd.GeoDataFrame):
        return SegmentSnapper(target).snap(lat, lon)
    raise TypeError(f"Unsupported target type for snap_to_segment: {type(target).__name__}")


def _validate_lat_lon(lat: float, lon: float) -> None:
    """Validate geographic coordinates are finite and in WGS84 bounds."""
    if not (-90.0 <= float(lat) <= 90.0):
        raise ValueError(f"Latitude out of range [-90, 90]: {lat}")
    if not (-180.0 <= float(lon) <= 180.0):
        raise ValueError(f"Longitude out of range [-180, 180]: {lon}")

