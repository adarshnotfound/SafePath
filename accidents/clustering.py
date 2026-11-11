"""Accident Hotspot Clustering and Algorithm Comparison - SafePath Module B3.

Clusters historical road accident locations into high-risk blackspots using:
1. DBSCAN: Density-Based Spatial Clustering of Applications with Noise (scikit-learn).
2. Fuzzy C-Means (FCM): Soft computing fuzzy clustering with continuous membership degrees (scikit-fuzzy).

Performs systematic parameter sweeps across:
- DBSCAN: eps distance (meters) and min_samples.
- FCM: cluster count (c) and fuzziness coefficient (m).

Evaluates and compares both algorithms using:
- Silhouette Scores (cluster separation and cohesion).
- Fuzzy Partition Coefficient (FPC) for FCM.
- Noise filtering ratio and geographic compactness (radius in meters).
- Outputs accident_hotspots.csv strictly per Section 8 data contract.
- Documents source country provenance (India vs substitute non-Indian data per Section 12).

Owner: Member B (Hazard map and accident-data lead)
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import skfuzzy as fuzz
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

from accidents.data_loader import (
    DEFAULT_ROUTE_SEGMENTS,
    AccidentPoint,
    generate_empirical_accident_dataset,
    load_accidents_from_csv,
    save_accidents_to_csv,
)


@dataclass
class AccidentHotspot:
    """Accident blackspot entity conforming to Section 8 accident_hotspots.csv schema."""

    hotspot_id: str
    lat: float
    lon: float
    radius_m: float
    accident_count: int
    severity_weight: float
    segment_id: str
    source_country: str

    def to_contract_dict(self) -> Dict[str, object]:
        """Format strictly per Section 8 contract columns."""
        return {
            "hotspot_id": self.hotspot_id,
            "lat": f"{self.lat:.5f}",
            "lon": f"{self.lon:.5f}",
            "radius_m": f"{round(self.radius_m, 1):.1f}",
            "accident_count": self.accident_count,
            "severity_weight": f"{round(self.severity_weight, 1):.1f}",
            "segment_id": self.segment_id,
            "source_country": self.source_country,
        }


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two GPS coordinates in meters."""
    r_earth = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r_earth * c


def project_coordinates_to_xy(
    lats: np.ndarray,
    lons: np.ndarray,
    lat0: float,
    lon0: float,
) -> np.ndarray:
    """Project latitude/longitude coordinates to local Cartesian (x, y) in meters."""
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(lat0))
    x = (lons - lon0) * meters_per_deg_lon
    y = (lats - lat0) * meters_per_deg_lat
    return np.column_stack([x, y])


def project_xy_to_latlon(
    x: float,
    y: float,
    lat0: float,
    lon0: float,
) -> Tuple[float, float]:
    """Inverse project local Cartesian meters back to decimal degrees (lat, lon)."""
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = meters_per_deg_lat * math.cos(math.radians(lat0))
    lat = lat0 + (y / meters_per_deg_lat)
    lon = lon0 + (x / meters_per_deg_lon)
    return lat, lon


def find_nearest_segment_id(
    lat: float,
    lon: float,
    segments: Optional[Dict[str, List[Tuple[float, float]]]] = None,
) -> str:
    """Find the closest road segment ID for given hotspot coordinates."""
    seg_dict = segments if segments is not None else DEFAULT_ROUTE_SEGMENTS
    best_seg = "seg_101"
    best_dist = float("inf")

    for seg_id, waypoints in seg_dict.items():
        for wpt_lat, wpt_lon in waypoints:
            dist = haversine_distance_meters(lat, lon, wpt_lat, wpt_lon)
            if dist < best_dist:
                best_dist = dist
                best_seg = seg_id
    return best_seg


def run_dbscan(
    points_xy: np.ndarray,
    eps_m: float = 80.0,
    min_samples: int = 4,
) -> Tuple[np.ndarray, Optional[float], int]:
    """Execute DBSCAN on projected metric coordinates.

    Args:
        points_xy: (N, 2) array of coordinates in meters.
        eps_m: Maximum distance (meters) between two samples to be considered neighbors.
        min_samples: Minimum samples in a neighborhood for a core point.

    Returns:
        (labels, silhouette_score, num_clusters)
    """
    db = DBSCAN(eps=eps_m, min_samples=min_samples, metric="euclidean")
    labels = db.fit_predict(points_xy)

    unique_clusters = [lbl for lbl in set(labels) if lbl != -1]
    num_clusters = len(unique_clusters)

    sil_score = None
    if num_clusters >= 2:
        mask = labels != -1
        if np.sum(mask) >= num_clusters * 2:
            sil_score = float(silhouette_score(points_xy[mask], labels[mask]))

    return labels, sil_score, num_clusters


def run_fcm(
    points_xy: np.ndarray,
    c: int = 6,
    m: float = 1.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float, Optional[float]]:
    """Execute Fuzzy C-Means (FCM) using scikit-fuzzy on projected coordinates.

    Args:
        points_xy: (N, 2) array of coordinates in meters.
        c: Number of clusters.
        m: Array exponentiation fuzziness index (typically 1.2 to 2.0).
        seed: Random seed.

    Returns:
        (centers_xy, membership_matrix_u, fpc, silhouette_score)
    """
    data = points_xy.T  # skfuzzy expects shape (features, samples)
    cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
        data,
        c=c,
        m=m,
        error=0.005,
        maxiter=1000,
        seed=seed,
    )
    hard_labels = np.argmax(u, axis=0)
    unique_labels = len(set(hard_labels))

    sil_score = None
    if unique_labels >= 2:
        sil_score = float(silhouette_score(points_xy, hard_labels))

    return cntr, u, float(fpc), sil_score


def parameter_sweep_dbscan(
    points_xy: np.ndarray,
    eps_values: Optional[List[float]] = None,
    min_samples_values: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """Perform systematic grid search over DBSCAN hyperparameter space."""
    if eps_values is None:
        eps_values = [40.0, 60.0, 80.0, 100.0, 120.0, 150.0]
    if min_samples_values is None:
        min_samples_values = [3, 4, 5]

    results = []
    n_points = len(points_xy)

    for eps_m in eps_values:
        for min_samples in min_samples_values:
            labels, sil_score, n_clusters = run_dbscan(points_xy, eps_m=eps_m, min_samples=min_samples)
            noise_count = int(np.sum(labels == -1))
            noise_ratio = noise_count / n_points

            results.append({
                "algorithm": "DBSCAN",
                "eps_meters": eps_m,
                "min_samples": min_samples,
                "num_clusters": n_clusters,
                "noise_count": noise_count,
                "noise_ratio": round(noise_ratio, 4),
                "silhouette_score": round(sil_score, 4) if sil_score is not None else None,
            })

    # Sort primarily by silhouette score, then reasonable cluster count (5-8)
    def score_key(r: Dict[str, Any]) -> float:
        sil = r["silhouette_score"] if r["silhouette_score"] is not None else -1.0
        # Penalize degenerate cluster counts (<3 or >10)
        c = r["num_clusters"]
        penalty = 0.0 if 5 <= c <= 8 else 0.2
        return sil - penalty

    results.sort(key=score_key, reverse=True)
    return results


def parameter_sweep_fcm(
    points_xy: np.ndarray,
    c_values: Optional[List[int]] = None,
    m_values: Optional[List[float]] = None,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Perform systematic grid search over Fuzzy C-Means hyperparameter space."""
    if c_values is None:
        c_values = [4, 5, 6, 7, 8]
    if m_values is None:
        m_values = [1.2, 1.5, 1.8, 2.0]

    results = []
    for c in c_values:
        for m in m_values:
            cntr, u, fpc, sil_score = run_fcm(points_xy, c=c, m=m, seed=seed)
            hard_labels = np.argmax(u, axis=0)
            cluster_sizes = [int(np.sum(hard_labels == k)) for k in range(c)]

            results.append({
                "algorithm": "FCM",
                "c_clusters": c,
                "m_fuzziness": m,
                "fpc": round(fpc, 4),
                "silhouette_score": round(sil_score, 4) if sil_score is not None else None,
                "min_cluster_size": min(cluster_sizes),
                "max_cluster_size": max(cluster_sizes),
            })

    # Sort primarily by silhouette score and FPC
    def score_key(r: Dict[str, Any]) -> float:
        sil = r["silhouette_score"] if r["silhouette_score"] is not None else -1.0
        return sil + 0.2 * r["fpc"]

    results.sort(key=score_key, reverse=True)
    return results


def build_hotspot_catalog(
    labels: np.ndarray,
    accidents: List[AccidentPoint],
    points_xy: np.ndarray,
    lat0: float,
    lon0: float,
    source_country: str = "India",
    segments: Optional[Dict[str, List[Tuple[float, float]]]] = None,
) -> List[AccidentHotspot]:
    """Construct Section 8 AccidentHotspot records from cluster assignments."""
    hotspots: List[AccidentHotspot] = []
    unique_labels = sorted([lbl for lbl in set(labels) if lbl != -1])

    for idx, cluster_label in enumerate(unique_labels, start=1):
        member_indices = np.where(labels == cluster_label)[0]
        cluster_accidents = [accidents[i] for i in member_indices]
        cluster_coords = points_xy[member_indices]

        # Centroid in projected meters
        mean_x = float(np.mean(cluster_coords[:, 0]))
        mean_y = float(np.mean(cluster_coords[:, 1]))
        c_lat, c_lon = project_xy_to_latlon(mean_x, mean_y, lat0, lon0)

        # Distances from centroid to member points
        dists = np.hypot(cluster_coords[:, 0] - mean_x, cluster_coords[:, 1] - mean_y)
        # Radius: 90th percentile distance bounded between [30m, 90m]
        radius_m = float(np.percentile(dists, 90)) if len(dists) > 1 else 35.0
        radius_m = max(30.0, min(90.0, radius_m))

        accident_count = len(cluster_accidents)
        avg_severity = float(np.mean([a.severity for a in cluster_accidents]))
        seg_id = find_nearest_segment_id(c_lat, c_lon, segments)

        hotspots.append(AccidentHotspot(
            hotspot_id=f"spot_{idx:03d}",
            lat=c_lat,
            lon=c_lon,
            radius_m=radius_m,
            accident_count=accident_count,
            severity_weight=avg_severity,
            segment_id=seg_id,
            source_country=source_country,
        ))

    # Sort descending by accident count (most critical blackspot first)
    hotspots.sort(key=lambda h: h.accident_count, reverse=True)
    # Re-index nicely spot_001, spot_002...
    for i, h in enumerate(hotspots, start=1):
        h.hotspot_id = f"spot_{i:03d}"

    return hotspots


def compare_and_export_hotspots(
    accidents_csv_path: Optional[Path] = None,
    output_hotspots_csv_path: Optional[Path] = None,
    output_metrics_json_path: Optional[Path] = None,
    source_country: str = "India",
    seed: int = 42,
) -> Tuple[List[AccidentHotspot], Dict[str, Any]]:
    """Execute full clustering comparison pipeline (DBSCAN vs FCM) and export Section 8 CSV."""
    repo_root = Path(__file__).resolve().parent.parent

    if accidents_csv_path is None:
        accidents_csv_path = repo_root / "data" / "processed" / "accidents_raw.csv"
    if output_hotspots_csv_path is None:
        output_hotspots_csv_path = repo_root / "data" / "processed" / "accident_hotspots.csv"
    if output_metrics_json_path is None:
        output_metrics_json_path = repo_root / "evaluation" / "results" / "accident_clustering_comparison.json"

    # Generate or load accident records
    if accidents_csv_path.exists():
        accidents = load_accidents_from_csv(accidents_csv_path)
    else:
        accidents = generate_empirical_accident_dataset(
            seed=seed,
            source_country=source_country,
            include_background_noise=True,
        )
        save_accidents_to_csv(accidents, accidents_csv_path)

    lats = np.array([a.lat for a in accidents])
    lons = np.array([a.lon for a in accidents])
    lat0 = float(np.mean(lats))
    lon0 = float(np.mean(lons))
    points_xy = project_coordinates_to_xy(lats, lons, lat0, lon0)

    # 1. Parameter sweeps
    dbscan_sweep = parameter_sweep_dbscan(points_xy)
    fcm_sweep = parameter_sweep_fcm(points_xy, seed=seed)

    best_dbscan_cfg = dbscan_sweep[0]
    best_fcm_cfg = fcm_sweep[0]

    # 2. Run best models
    dbscan_labels, dbscan_sil, dbscan_k = run_dbscan(
        points_xy,
        eps_m=best_dbscan_cfg["eps_meters"],
        min_samples=best_dbscan_cfg["min_samples"],
    )

    fcm_cntr, fcm_u, fcm_fpc, fcm_sil = run_fcm(
        points_xy,
        c=best_fcm_cfg["c_clusters"],
        m=best_fcm_cfg["m_fuzziness"],
        seed=seed,
    )
    fcm_labels = np.argmax(fcm_u, axis=0)

    # 3. Build candidate hotspots from both algorithms
    dbscan_hotspots = build_hotspot_catalog(
        dbscan_labels, accidents, points_xy, lat0, lon0, source_country=source_country
    )
    fcm_hotspots = build_hotspot_catalog(
        fcm_labels, accidents, points_xy, lat0, lon0, source_country=source_country
    )

    # Choose winning model for Section 8 output:
    # DBSCAN filters background noise and yields isolated blackspot clusters with high silhouette
    chosen_hotspots = dbscan_hotspots if (dbscan_sil or -1) >= (fcm_sil or -1) else fcm_hotspots
    winning_algo = "DBSCAN" if chosen_hotspots is dbscan_hotspots else "Fuzzy C-Means"

    # Export accident_hotspots.csv strictly adhering to Section 8 schema
    output_hotspots_csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "hotspot_id",
        "lat",
        "lon",
        "radius_m",
        "accident_count",
        "severity_weight",
        "segment_id",
        "source_country",
    ]
    with open(output_hotspots_csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for h in chosen_hotspots:
            writer.writerow(h.to_contract_dict())

    # Compile comprehensive comparison report
    comparison_report = {
        "dataset_metadata": {
            "total_accidents": len(accidents),
            "source_country": source_country,
            "provenance": "real" if source_country.lower() == "india" else "substitute",
            "is_indian_data": source_country.lower() == "india",
            "geographic_center": {"lat": round(lat0, 5), "lon": round(lon0, 5)},
        },
        "dbscan": {
            "best_params": {
                "eps_meters": best_dbscan_cfg["eps_meters"],
                "min_samples": best_dbscan_cfg["min_samples"],
            },
            "num_hotspots_found": len(dbscan_hotspots),
            "silhouette_score": best_dbscan_cfg["silhouette_score"],
            "noise_points": best_dbscan_cfg["noise_count"],
            "noise_ratio": best_dbscan_cfg["noise_ratio"],
            "mean_hotspot_radius_m": round(float(np.mean([h.radius_m for h in dbscan_hotspots])), 1),
            "sweep_top_5": dbscan_sweep[:5],
        },
        "fuzzy_c_means": {
            "best_params": {
                "c_clusters": best_fcm_cfg["c_clusters"],
                "m_fuzziness": best_fcm_cfg["m_fuzziness"],
            },
            "num_hotspots_found": len(fcm_hotspots),
            "silhouette_score": best_fcm_cfg["silhouette_score"],
            "fuzzy_partition_coefficient": best_fcm_cfg["fpc"],
            "mean_hotspot_radius_m": round(float(np.mean([h.radius_m for h in fcm_hotspots])), 1),
            "sweep_top_5": fcm_sweep[:5],
        },
        "soft_computing_insights": {
            "selected_production_algorithm": winning_algo,
            "dbscan_advantages": "Excludes dispersed background accidents as noise; robust boundary detection without pre-fixing k.",
            "fcm_advantages": "Soft computing continuous membership degrees allow modeling gradual safety risk transitions instead of abrupt step functions.",
        },
    }

    # Save comparison metrics to JSON per Rule 4
    output_metrics_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_json_path, mode="w", encoding="utf-8") as f:
        json.dump(comparison_report, f, indent=2)

    return chosen_hotspots, comparison_report


if __name__ == "__main__":
    from accidents.visualization import plot_accident_hotspots_folium

    hotspots, report = compare_and_export_hotspots()
    repo_root = Path(__file__).resolve().parent.parent
    acc_path = repo_root / "data" / "processed" / "accidents_raw.csv"
    accidents = load_accidents_from_csv(acc_path) if acc_path.exists() else []

    # Generate interactive Folium map
    map_path = repo_root / "evaluation" / "plots" / "accident_hotspots_map.html"
    plot_accident_hotspots_folium(
        hotspots=hotspots,
        accidents=accidents,
        output_html_path=map_path,
        source_country=report["dataset_metadata"]["source_country"],
    )

    print("=" * 65)
    print("SafePath Accident Hotspot Clustering & Comparison (Module B3)")
    print("=" * 65)
    print(f"Data Origin              : {report['dataset_metadata']['source_country']} ({report['dataset_metadata']['provenance']})")
    print(f"Total Accidents          : {report['dataset_metadata']['total_accidents']}")
    print("-" * 65)
    print("DBSCAN Clustering:")
    print(f"  Best Parameters        : eps = {report['dbscan']['best_params']['eps_meters']}m, min_samples = {report['dbscan']['best_params']['min_samples']}")
    print(f"  Hotspots Found         : {report['dbscan']['num_hotspots_found']}")
    print(f"  Silhouette Score       : {report['dbscan']['silhouette_score']}")
    print(f"  Noise Points Excluded  : {report['dbscan']['noise_points']} ({report['dbscan']['noise_ratio'] * 100:.1f}%)")
    print("-" * 65)
    print("Fuzzy C-Means (FCM):")
    print(f"  Best Parameters        : c = {report['fuzzy_c_means']['best_params']['c_clusters']}, m = {report['fuzzy_c_means']['best_params']['m_fuzziness']}")
    print(f"  Hotspots Found         : {report['fuzzy_c_means']['num_hotspots_found']}")
    print(f"  Silhouette Score       : {report['fuzzy_c_means']['silhouette_score']}")
    print(f"  FPC (Fuzzy Partition)  : {report['fuzzy_c_means']['fuzzy_partition_coefficient']}")
    print("-" * 65)
    print(f"Selected Production Model : {report['soft_computing_insights']['selected_production_algorithm']}")
    print(f"Generated Hotspots CSV    : data/processed/accident_hotspots.csv")
    print(f"Metrics Report Saved      : evaluation/results/accident_clustering_comparison.json")
    print(f"Interactive Folium Map    : evaluation/plots/accident_hotspots_map.html")
    print("=" * 65)

