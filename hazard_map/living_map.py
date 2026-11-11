"""Living Hazard Map - SafePath Module B2.

Implements spatial-temporal memory for road hazards:
1. Spatial clustering with DBSCAN on projected metric coordinates (default radius 7m)
   for intra-class detection merging.
2. Incremental per-drive updates:
   - Confidence rise upon confirmed re-detection.
   - Delayed decay only when a drive passes the hazard location and misses it,
     requiring at least K misses (configurable) to filter sensor noise.
3. Type-specific persistence (speed breaker very slow, pothole medium, animal fast)
   configured via external config file.
4. Export strictly conforming to Section 8 hazards.csv data contract.

Owner: Member B (Hazard map and accident-data lead)
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Try to import sklearn DBSCAN, provide pure-Python fallback for resilient portability
try:
    from sklearn.cluster import DBSCAN
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Default route segments matching Bengaluru reference graph
DEFAULT_ROUTE_SEGMENTS: Dict[str, List[Tuple[float, float]]] = {
    "seg_101": [(12.97160, 77.59460), (12.97210, 77.59510)],
    "seg_102": [(12.97210, 77.59510), (12.97305, 77.59600)],
    "seg_103": [(12.97305, 77.59600), (12.97450, 77.59750)],
    "seg_104": [(12.97210, 77.59510), (12.97380, 77.59685)],
    "seg_105": [(12.97050, 77.59320), (12.97160, 77.59460)],
    "seg_106": [(12.97450, 77.59750), (12.97600, 77.59910)],
    "seg_107": [(12.97160, 77.59460), (12.97250, 77.59580)],
    "seg_108": [(12.97250, 77.59580), (12.97380, 77.59685)],
}

DEFAULT_ROUTES: Dict[str, List[str]] = {
    "route_A": ["seg_105", "seg_107", "seg_108"],
    "route_B": ["seg_101", "seg_102", "seg_103"],
    "route_C": ["seg_101", "seg_104", "seg_108"],
}


def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two GPS coordinates in meters."""
    r_earth = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * (math.sin(dlambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r_earth * c


def project_latlon_to_xy(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """Project latitude/longitude to local metric Cartesian coordinates (x, y) in meters."""
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
    x = (lon - lon0) * meters_per_deg_lon
    y = (lat - lat0) * meters_per_deg_lat
    return x, y


def project_xy_to_latlon(x: float, y: float, lat0: float, lon0: float) -> Tuple[float, float]:
    """Inverse project local Cartesian coordinates in meters back to decimal degrees (lat, lon)."""
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat0))
    lat = lat0 + (y / meters_per_deg_lat)
    lon = lon0 + (x / meters_per_deg_lon)
    return lat, lon


def find_nearest_segment(
    lat: float,
    lon: float,
    segments: Optional[Dict[str, List[Tuple[float, float]]]] = None,
) -> str:
    """Find the closest road segment ID for given coordinates."""
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


@dataclass
class TypePersistenceConfig:
    """Type-specific persistence, decay, and miss threshold parameters."""

    decay_rate: float
    freshness_decay_rate: float
    miss_threshold_k: int
    description: str = ""


@dataclass
class LivingMapConfig:
    """Configuration container for clustering, matching, and living-map updates."""

    dbscan_eps_meters: float = 7.0
    dbscan_min_samples: int = 1
    association_radius_meters: float = 12.0
    coverage_radius_meters: float = 45.0
    default_miss_threshold_k: int = 2
    confidence_rise_factor: float = 0.6
    max_confidence: float = 0.99
    min_active_confidence: float = 0.35
    min_active_freshness: float = 0.20
    type_persistence: Dict[str, TypePersistenceConfig] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> LivingMapConfig:
        """Load configuration from JSON file or return defaults."""
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "config.json"

        if not config_path.exists():
            # Return standard defaults
            return cls(
                dbscan_eps_meters=7.0,
                dbscan_min_samples=1,
                association_radius_meters=12.0,
                coverage_radius_meters=45.0,
                default_miss_threshold_k=2,
                type_persistence={
                    "speed_breaker": TypePersistenceConfig(0.02, 0.04, 3, "Permanent"),
                    "pothole": TypePersistenceConfig(0.30, 0.35, 1, "Medium decay"),
                    "animal": TypePersistenceConfig(0.75, 0.80, 1, "Fast decay"),
                },
            )

        with open(config_path, mode="r", encoding="utf-8") as f:
            raw = json.load(f)

        type_persist = {}
        for k, v in raw.get("type_persistence", {}).items():
            type_persist[k] = TypePersistenceConfig(
                decay_rate=float(v.get("decay_rate", 0.2)),
                freshness_decay_rate=float(v.get("freshness_decay_rate", 0.2)),
                miss_threshold_k=int(v.get("miss_threshold_k", 2)),
                description=str(v.get("description", "")),
            )

        return cls(
            dbscan_eps_meters=float(raw.get("dbscan_eps_meters", 7.0)),
            dbscan_min_samples=int(raw.get("dbscan_min_samples", 1)),
            association_radius_meters=float(raw.get("association_radius_meters", 12.0)),
            coverage_radius_meters=float(raw.get("coverage_radius_meters", 45.0)),
            default_miss_threshold_k=int(raw.get("default_miss_threshold_k", 2)),
            confidence_rise_factor=float(raw.get("confidence_rise_factor", 0.6)),
            max_confidence=float(raw.get("max_confidence", 0.99)),
            min_active_confidence=float(raw.get("min_active_confidence", 0.35)),
            min_active_freshness=float(raw.get("min_active_freshness", 0.20)),
            type_persistence=type_persist,
        )

    def get_persistence(self, hazard_type: str) -> TypePersistenceConfig:
        """Get persistence configuration for given hazard type with graceful fallback."""
        if hazard_type in self.type_persistence:
            return self.type_persistence[hazard_type]
        return TypePersistenceConfig(
            decay_rate=0.25,
            freshness_decay_rate=0.30,
            miss_threshold_k=self.default_miss_threshold_k,
            description="Default fallback persistence",
        )


@dataclass
class DetectionItem:
    """Clean detection representation for clustering input."""

    detection_id: str
    drive_id: str
    class_name: str
    confidence: float
    lat: float
    lon: float
    timestamp: str
    is_simulated: bool = True


@dataclass
class HazardRecord:
    """Living hazard map entry conforming to Section 8 hazards.csv schema."""

    hazard_id: str
    type: str
    lat: float
    lon: float
    first_seen: str
    last_seen: str
    observation_count: int
    drive_count: int
    confidence: float
    freshness: float
    status: str  # 'active', 'decayed', 'repaired'
    segment_id: str
    is_simulated: bool = True
    consecutive_misses: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_contract_dict(self) -> Dict[str, object]:
        """Format strictly per Section 8 hazards.csv specification."""
        return {
            "hazard_id": self.hazard_id,
            "type": self.type,
            "lat": f"{self.lat:.5f}",
            "lon": f"{self.lon:.5f}",
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "observation_count": self.observation_count,
            "drive_count": self.drive_count,
            "confidence": f"{round(self.confidence, 2):.2f}",
            "freshness": f"{round(self.freshness, 2):.2f}",
            "status": self.status,
            "segment_id": self.segment_id,
            "is_simulated": self.is_simulated,
        }


def _pure_python_dbscan(points_xy: List[Tuple[float, float]], eps: float, min_samples: int) -> List[int]:
    """Pure Python DBSCAN fallback for environments without scikit-learn."""
    n = len(points_xy)
    labels = [-1] * n
    cluster_id = 0

    def region_query(p_idx: int) -> List[int]:
        px, py = points_xy[p_idx]
        neighbors = []
        for i, (qx, qy) in enumerate(points_xy):
            dist = math.hypot(px - qx, py - qy)
            if dist <= eps:
                neighbors.append(i)
        return neighbors

    for i in range(n):
        if labels[i] != -1:
            continue
        neighbors = region_query(i)
        if len(neighbors) < min_samples:
            continue
        labels[i] = cluster_id
        seeds = list(neighbors)
        seeds.remove(i)
        while seeds:
            curr = seeds.pop(0)
            if labels[curr] == -1:
                labels[curr] = cluster_id
            elif labels[curr] != -1:
                continue
            labels[curr] = cluster_id
            curr_neighbors = region_query(curr)
            if len(curr_neighbors) >= min_samples:
                for nb in curr_neighbors:
                    if nb not in seeds and labels[nb] == -1:
                        seeds.append(nb)
        cluster_id += 1

    return labels


def merge_detections_dbscan(
    detections: List[DetectionItem],
    eps_meters: float = 7.0,
    min_samples: int = 1,
    lat0: Optional[float] = None,
    lon0: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Cluster detections of the same class within eps_meters using DBSCAN on projected metric coordinates.

    Args:
        detections: List of detection items to cluster.
        eps_meters: Clustering distance radius in meters (default: 7.0m).
        min_samples: Minimum points to form dense cluster (default: 1).
        lat0: Origin latitude for metric projection (defaults to mean latitude).
        lon0: Origin longitude for metric projection (defaults to mean longitude).

    Returns:
        List of cluster summary dictionaries.
    """
    if not detections:
        return []

    if lat0 is None:
        lat0 = sum(d.lat for d in detections) / len(detections)
    if lon0 is None:
        lon0 = sum(d.lon for d in detections) / len(detections)

    # Group detections by hazard class (never merge across different classes)
    by_class: Dict[str, List[DetectionItem]] = {}
    for d in detections:
        by_class.setdefault(d.class_name, []).append(d)

    merged_clusters: List[Dict[str, Any]] = []

    for cls_name, class_dets in by_class.items():
        coords_xy = [
            project_latlon_to_xy(d.lat, d.lon, lat0, lon0)
            for d in class_dets
        ]

        if SKLEARN_AVAILABLE and len(coords_xy) > 0:
            arr = np.array(coords_xy)
            db = DBSCAN(eps=eps_meters, min_samples=min_samples, metric="euclidean")
            labels = db.fit_predict(arr).tolist()
        else:
            labels = _pure_python_dbscan(coords_xy, eps=eps_meters, min_samples=min_samples)

        # Group by cluster label
        clusters_by_label: Dict[int, List[DetectionItem]] = {}
        for label, det in zip(labels, class_dets):
            # Noise points (label -1) with min_samples > 1 are treated as singletons
            clusters_by_label.setdefault(label, []).append(det)

        for label, cluster_dets in clusters_by_label.items():
            # Weighted average coordinates based on detection confidence
            total_weight = sum(d.confidence for d in cluster_dets)
            if total_weight > 0:
                avg_lat = sum(d.lat * d.confidence for d in cluster_dets) / total_weight
                avg_lon = sum(d.lon * d.confidence for d in cluster_dets) / total_weight
            else:
                avg_lat = sum(d.lat for d in cluster_dets) / len(cluster_dets)
                avg_lon = sum(d.lon for d in cluster_dets) / len(cluster_dets)

            # Combined confidence: max confidence plus small reinforcement for repeated sightings
            base_conf = max(d.confidence for d in cluster_dets)
            if len(cluster_dets) > 1:
                combined_conf = min(0.99, base_conf + 0.05 * (len(cluster_dets) - 1))
            else:
                combined_conf = base_conf

            sorted_ts = sorted(d.timestamp for d in cluster_dets)
            first_seen = sorted_ts[0]
            last_seen = sorted_ts[-1]
            unique_drives = len(set(d.drive_id for d in cluster_dets))

            merged_clusters.append({
                "class_name": cls_name,
                "lat": avg_lat,
                "lon": avg_lon,
                "confidence": combined_conf,
                "observation_count": len(cluster_dets),
                "drive_count": unique_drives,
                "first_seen": first_seen,
                "last_seen": last_seen,
                "member_detection_ids": [d.detection_id for d in cluster_dets],
                "detections": cluster_dets,
            })

    return merged_clusters


class LivingHazardMap:
    """Living road hazard memory engine maintaining persistent spatial-temporal state."""

    def __init__(self, config: Optional[LivingMapConfig] = None, segments: Optional[Dict[str, List[Tuple[float, float]]]] = None):
        self.config = config if config is not None else LivingMapConfig.load()
        self.segments = segments if segments is not None else DEFAULT_ROUTE_SEGMENTS
        self.hazards: Dict[str, HazardRecord] = {}
        self.hazard_counter = 1
        self.processed_drives: List[str] = []

    def _generate_hazard_id(self) -> str:
        hid = f"haz_{self.hazard_counter:03d}"
        self.hazard_counter += 1
        return hid

    def _drive_passed_hazard(
        self,
        hazard: HazardRecord,
        drive_detections: List[DetectionItem],
        traversed_segments: Optional[Set[str]] = None,
    ) -> bool:
        """Check if a drive physically passed near enough to observe a hazard."""
        # 1. Traversed segment match
        if traversed_segments is not None and hazard.segment_id in traversed_segments:
            return True

        # 2. Spatial proximity to any detection made during the drive
        if drive_detections:
            cov_radius = self.config.coverage_radius_meters
            for d in drive_detections:
                if haversine_distance_meters(hazard.lat, hazard.lon, d.lat, d.lon) <= cov_radius:
                    return True

        # 3. Proximity to road segment geometry if drive segment is known
        return False

    def update_with_drive(
        self,
        drive_id: str,
        detections: List[DetectionItem],
        traversed_segments: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Update living map with observations from a single drive run.

        Rules applied:
        - Detections of the same class within 7m are merged with DBSCAN.
        - Existing hazards matching a detection have confidence raised, observation_count increased,
          and consecutive misses reset to 0.
        - Unmatched existing hazards passed by the drive accumulate consecutive misses;
          decay is applied only when consecutive misses >= K (type-specific).
        - Type-specific persistence governs decay rates (speed_breaker slow, pothole medium, animal fast).

        Args:
            drive_id: Identifier of the drive (e.g. 'drive_01').
            detections: List of detections from this drive.
            traversed_segments: Optional explicit set of road segment IDs visited on this drive.

        Returns:
            Dictionary summarizing updates made during this drive.
        """
        self.processed_drives.append(drive_id)

        # Infer traversed segments if not explicitly supplied
        if traversed_segments is None:
            traversed_segments = {
                find_nearest_segment(d.lat, d.lon, self.segments)
                for d in detections
            }

        # Step 1: Intra-drive clustering with DBSCAN on projected metric coordinates
        drive_clusters = merge_detections_dbscan(
            detections,
            eps_meters=self.config.dbscan_eps_meters,
            min_samples=self.config.dbscan_min_samples,
        )

        detected_hazard_ids: Set[str] = set()
        matched_cluster_indices: Set[int] = set()

        # Step 2: Associate drive clusters to existing active/known hazards of SAME class
        for c_idx, cluster in enumerate(drive_clusters):
            best_hid: Optional[str] = None
            min_dist = float("inf")

            for hid, h_rec in self.hazards.items():
                if h_rec.type != cluster["class_name"]:
                    continue
                dist = haversine_distance_meters(h_rec.lat, h_rec.lon, cluster["lat"], cluster["lon"])
                if dist <= self.config.association_radius_meters and dist < min_dist:
                    min_dist = dist
                    best_hid = hid

            if best_hid is not None:
                # Re-detection: update hazard
                h = self.hazards[best_hid]
                matched_cluster_indices.add(c_idx)
                detected_hazard_ids.add(best_hid)

                # Centroid running update weighted by observations
                old_weight = h.observation_count
                new_weight = cluster["observation_count"]
                total_w = old_weight + new_weight
                h.lat = (h.lat * old_weight + cluster["lat"] * new_weight) / total_w
                h.lon = (h.lon * old_weight + cluster["lon"] * new_weight) / total_w

                # Confidence rise (probabilistic fusion)
                # c_new = 1 - (1 - c_old) * (1 - rise_factor * c_obs)
                c_obs = cluster["confidence"]
                c_prev = h.confidence
                rise_factor = self.config.confidence_rise_factor
                c_new = 1.0 - (1.0 - c_prev) * (1.0 - rise_factor * c_obs)
                h.confidence = min(self.config.max_confidence, c_new)

                h.observation_count += new_weight
                h.drive_count += 1
                h.last_seen = max(h.last_seen, cluster["last_seen"])
                h.consecutive_misses = 0
                h.freshness = 1.0
                h.status = "active"

                h.history.append({
                    "drive_id": drive_id,
                    "event": "re_detected",
                    "confidence": h.confidence,
                    "freshness": h.freshness,
                    "consecutive_misses": 0,
                    "distance_to_match": round(min_dist, 2),
                })

        # Step 3: Add new hazards for unmatched drive clusters
        for c_idx, cluster in enumerate(drive_clusters):
            if c_idx in matched_cluster_indices:
                continue

            new_hid = self._generate_hazard_id()
            detected_hazard_ids.add(new_hid)
            seg_id = find_nearest_segment(cluster["lat"], cluster["lon"], self.segments)

            new_record = HazardRecord(
                hazard_id=new_hid,
                type=cluster["class_name"],
                lat=cluster["lat"],
                lon=cluster["lon"],
                first_seen=cluster["first_seen"],
                last_seen=cluster["last_seen"],
                observation_count=cluster["observation_count"],
                drive_count=1,
                confidence=min(self.config.max_confidence, cluster["confidence"]),
                freshness=1.0,
                status="active",
                segment_id=seg_id,
                is_simulated=True,
                consecutive_misses=0,
                history=[{
                    "drive_id": drive_id,
                    "event": "initial_detection",
                    "confidence": cluster["confidence"],
                    "freshness": 1.0,
                    "consecutive_misses": 0,
                }],
            )
            self.hazards[new_hid] = new_record

        # Step 4: Process misses and type-specific decay for hazards NOT detected in this drive
        decayed_count = 0
        repaired_count = 0

        for hid, h in self.hazards.items():
            if hid in detected_hazard_ids:
                continue

            # Verify if this drive passed near the hazard
            passed = self._drive_passed_hazard(h, detections, traversed_segments)
            if not passed:
                # Drive did not pass location: no miss incremented, no decay
                h.history.append({
                    "drive_id": drive_id,
                    "event": "unvisited",
                    "confidence": h.confidence,
                    "freshness": h.freshness,
                    "consecutive_misses": h.consecutive_misses,
                })
                continue

            # Drive passed the hazard location and missed it!
            h.consecutive_misses += 1
            persist_cfg = self.config.get_persistence(h.type)
            k_threshold = persist_cfg.miss_threshold_k

            if h.consecutive_misses < k_threshold:
                # Noisy single miss: DO NOT apply decay yet
                h.history.append({
                    "drive_id": drive_id,
                    "event": "miss_buffered",
                    "confidence": h.confidence,
                    "freshness": h.freshness,
                    "consecutive_misses": h.consecutive_misses,
                    "k_threshold": k_threshold,
                })
            else:
                # Consecutive misses reached threshold K: apply type-specific decay
                h.confidence = max(0.05, h.confidence * (1.0 - persist_cfg.decay_rate))
                h.freshness = max(0.01, h.freshness * (1.0 - persist_cfg.freshness_decay_rate))

                # Status transition logic
                if h.confidence < self.config.min_active_confidence or h.freshness < self.config.min_active_freshness:
                    if h.type == "animal":
                        h.status = "decayed"
                        decayed_count += 1
                    else:
                        h.status = "repaired"
                        repaired_count += 1

                h.history.append({
                    "drive_id": drive_id,
                    "event": "decay_applied",
                    "confidence": h.confidence,
                    "freshness": h.freshness,
                    "status": h.status,
                    "consecutive_misses": h.consecutive_misses,
                })

        return {
            "drive_id": drive_id,
            "detections_count": len(detections),
            "clusters_count": len(drive_clusters),
            "re_detected_count": len(matched_cluster_indices),
            "new_hazards_count": len(drive_clusters) - len(matched_cluster_indices),
            "total_hazards_tracked": len(self.hazards),
            "newly_decayed_count": decayed_count,
            "newly_repaired_count": repaired_count,
        }

    def process_all_drives(
        self,
        all_detections: List[DetectionItem],
        drive_segments_map: Optional[Dict[str, Set[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Process multiple drives in chronological order."""
        # Partition detections by drive_id, preserving drive order
        drives_order: List[str] = []
        drives_dict: Dict[str, List[DetectionItem]] = {}
        for d in all_detections:
            if d.drive_id not in drives_dict:
                drives_order.append(d.drive_id)
                drives_dict[d.drive_id] = []
            drives_dict[d.drive_id].append(d)

        summaries = []
        for drive_id in drives_order:
            dets = drives_dict[drive_id]
            segs = drive_segments_map.get(drive_id) if drive_segments_map else None
            summary = self.update_with_drive(drive_id, dets, traversed_segments=segs)
            summaries.append(summary)
        return summaries

    def save_to_csv(self, output_path: Path) -> None:
        """Export hazards strictly according to Section 8 hazards.csv schema."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "hazard_id",
            "type",
            "lat",
            "lon",
            "first_seen",
            "last_seen",
            "observation_count",
            "drive_count",
            "confidence",
            "freshness",
            "status",
            "segment_id",
            "is_simulated",
        ]
        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for h in self.hazards.values():
                writer.writerow(h.to_contract_dict())

    def get_active_hazards(self) -> List[HazardRecord]:
        """Return list of currently active hazards."""
        return [h for h in self.hazards.values() if h.status == "active"]


def load_detections_from_csv(csv_path: Path) -> List[DetectionItem]:
    """Load detections from Section 8 detections.csv file."""
    detections: List[DetectionItem] = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            detections.append(DetectionItem(
                detection_id=row["detection_id"],
                drive_id=row["drive_id"],
                class_name=row["class"],
                confidence=float(row["confidence"]),
                lat=float(row["lat"]),
                lon=float(row["lon"]),
                timestamp=row["timestamp"],
                is_simulated=row.get("is_simulated", "True").lower() in ("true", "1"),
            ))
    return detections

