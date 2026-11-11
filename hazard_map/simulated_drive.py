"""Simulated Drive Generator - SafePath Module B1.

Generates realistic simulated drives along road network routes with:
- Ground-truth hazards (potholes, speed breakers, animals)
- Life-cycle transitions (new hazards appearing, repaired hazards disappearing, transient animals)
- Type-dependent detection probabilities
- 5-15m Gaussian GPS noise
- False positive detections
- Exact adherence to Section 8 detections.csv schema

Owner: Member B (Hazard map and accident-data lead)
"""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class GroundTruthHazard:
    """Represents a persistent or transient ground-truth hazard on the road."""

    hazard_id: str
    type: str  # 'pothole', 'speed_breaker', 'animal'
    lat: float
    lon: float
    segment_id: str
    created_drive: str  # Drive ID when hazard first appeared
    repaired_drive: Optional[str]  # Drive ID when hazard was repaired/expired, or None if active
    status: str  # 'active', 'repaired', 'expired'


@dataclass
class DetectionRecord:
    """Represents a single visual detection row conforming to section 8 contract."""

    detection_id: str
    drive_id: str
    frame_id: str
    class_name: str
    confidence: float
    bbox: str
    lat: float
    lon: float
    timestamp: str
    is_simulated: bool = True

    def to_contract_dict(self) -> Dict[str, object]:
        """Convert to exact column names specified in section 8 file contract."""
        return {
            "detection_id": self.detection_id,
            "drive_id": self.drive_id,
            "frame_id": self.frame_id,
            "class": self.class_name,
            "confidence": f"{self.confidence:.2f}",
            "bbox": self.bbox,
            "lat": f"{self.lat:.5f}",
            "lon": f"{self.lon:.5f}",
            "timestamp": self.timestamp,
            "is_simulated": self.is_simulated,
        }


# Default route waypoints and segment IDs (matching Bengaluru sample graph)
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

# Type-dependent detection parameters
TYPE_DETECTION_PROBABILITY: Dict[str, float] = {
    "pothole": 0.85,
    "speed_breaker": 0.95,
    "animal": 0.75,
}


def add_gps_noise(lat: float, lon: float, std_meters: float = 8.0, rng: Optional[random.Random] = None) -> Tuple[float, float]:
    """Add realistic 5-15m Gaussian GPS noise to coordinates.

    Args:
        lat: Ground-truth latitude in decimal degrees.
        lon: Ground-truth longitude in decimal degrees.
        std_meters: Standard deviation of GPS noise in meters (default 8m).
        rng: Random generator instance for reproducible experiments.

    Returns:
        (noisy_lat, noisy_lon) tuple in decimal degrees.
    """
    r = rng if rng is not None else random
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * math.cos(math.radians(lat))

    # Gaussian noise in local North/East coordinates
    noise_north = r.gauss(0.0, std_meters)
    noise_east = r.gauss(0.0, std_meters)

    noisy_lat = lat + (noise_north / meters_per_deg_lat)
    noisy_lon = lon + (noise_east / meters_per_deg_lon)
    return noisy_lat, noisy_lon


def generate_ground_truth_hazards(seed: int = 42) -> List[GroundTruthHazard]:
    """Create a reproducible catalog of ground-truth hazards with lifecycles.

    Lifecycles:
    - Speed breakers: permanent (active across all drives)
    - Potholes: persistent, but some are repaired or appear later
    - Animals: highly transient (present only on specific drives)

    Args:
        seed: Random seed for repeatability.

    Returns:
        List of GroundTruthHazard records.
    """
    hazards = [
        # Baseline hazards present in drive_01
        GroundTruthHazard(
            hazard_id="gt_haz_001",
            type="pothole",
            lat=12.97160,
            lon=77.59460,
            segment_id="seg_101",
            created_drive="drive_01",
            repaired_drive=None,  # Persistent across drives 1, 2, 3
            status="active",
        ),
        GroundTruthHazard(
            hazard_id="gt_haz_002",
            type="speed_breaker",
            lat=12.97210,
            lon=77.59510,
            segment_id="seg_101",
            created_drive="drive_01",
            repaired_drive=None,  # Speed breaker is permanent
            status="active",
        ),
        GroundTruthHazard(
            hazard_id="gt_haz_003",
            type="animal",
            lat=12.97305,
            lon=77.59600,
            segment_id="seg_102",
            created_drive="drive_01",
            repaired_drive="drive_02",  # Animal moved away before drive 2
            status="expired",
        ),
        GroundTruthHazard(
            hazard_id="gt_haz_004",
            type="pothole",
            lat=12.97380,
            lon=77.59685,
            segment_id="seg_102",
            created_drive="drive_01",
            repaired_drive="drive_03",  # Repaired before drive 3
            status="repaired",
        ),
        # New hazard appearing in drive_02
        GroundTruthHazard(
            hazard_id="gt_haz_005",
            type="pothole",
            lat=12.97450,
            lon=77.59750,
            segment_id="seg_103",
            created_drive="drive_02",
            repaired_drive=None,  # Appears in drive 2, active in drive 3
            status="active",
        ),
        # New transient animal appearing in drive_03
        GroundTruthHazard(
            hazard_id="gt_haz_006",
            type="animal",
            lat=12.97230,
            lon=77.59550,
            segment_id="seg_107",
            created_drive="drive_03",
            repaired_drive=None,
            status="active",
        ),
        # Speed breaker on route C
        GroundTruthHazard(
            hazard_id="gt_haz_007",
            type="speed_breaker",
            lat=12.97290,
            lon=77.59590,
            segment_id="seg_104",
            created_drive="drive_01",
            repaired_drive=None,
            status="active",
        ),
    ]
    return hazards


def is_hazard_present_on_drive(hazard: GroundTruthHazard, drive_id: str) -> bool:
    """Determine if a ground-truth hazard physically exists during a given drive."""
    drive_order = ["drive_01", "drive_02", "drive_03"]
    curr_idx = drive_order.index(drive_id)
    created_idx = drive_order.index(hazard.created_drive)

    if curr_idx < created_idx:
        return False  # Not yet appeared

    if hazard.repaired_drive is not None:
        repaired_idx = drive_order.index(hazard.repaired_drive)
        if curr_idx >= repaired_idx:
            return False  # Already repaired or departed

    return True


def simulate_multi_drive_detections(
    hazards: List[GroundTruthHazard],
    seed: int = 42,
    gps_noise_std_meters: float = 8.0,
    include_false_positives: bool = True,
) -> Tuple[List[DetectionRecord], Dict[str, object]]:
    """Simulate detections across multiple drives on consecutive days.

    Args:
        hazards: Ground truth hazard list.
        seed: Random seed for full reproducibility.
        gps_noise_std_meters: Noise standard deviation in meters (5-15m).
        include_false_positives: Whether to inject false detections.

    Returns:
        Tuple of (list of DetectionRecords, metadata dictionary).
    """
    rng = random.Random(seed)
    detections: List[DetectionRecord] = []
    det_counter = 1

    drive_configs = [
        {"drive_id": "drive_01", "start_time": datetime(2026, 11, 1, 10, 0, 0, tzinfo=timezone.utc)},
        {"drive_id": "drive_02", "start_time": datetime(2026, 11, 2, 14, 15, 0, tzinfo=timezone.utc)},
        {"drive_id": "drive_03", "start_time": datetime(2026, 11, 3, 9, 30, 0, tzinfo=timezone.utc)},
    ]

    total_possible_detections = 0
    actual_true_detections = 0
    false_positives_count = 0

    for drive in drive_configs:
        drive_id = drive["drive_id"]
        base_time = drive["start_time"]
        frame_seq = 10

        for hazard in hazards:
            if not is_hazard_present_on_drive(hazard, drive_id):
                continue

            total_possible_detections += 1
            detect_prob = TYPE_DETECTION_PROBABILITY.get(hazard.type, 0.85)

            # Check if this hazard is detected in this drive
            if rng.random() <= detect_prob:
                actual_true_detections += 1
                noisy_lat, noisy_lon = add_gps_noise(
                    hazard.lat, hazard.lon, std_meters=gps_noise_std_meters, rng=rng
                )

                # Confidence: high for true hazards
                conf = min(0.98, max(0.60, rng.gauss(0.88, 0.06)))

                # Synthesize reasonable bounding box [x1, y1, x2, y2]
                box_w = rng.randint(50, 120)
                box_h = rng.randint(40, 90)
                x1 = rng.randint(100, 350)
                y1 = rng.randint(180, 360)
                bbox_str = f"[{x1}, {y1}, {x1 + box_w}, {y1 + box_h}]"

                # Timestamp progression
                ts = base_time + timedelta(seconds=frame_seq * 3)
                iso_ts = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

                record = DetectionRecord(
                    detection_id=f"det_{det_counter:03d}",
                    drive_id=drive_id,
                    frame_id=f"frame_{frame_seq:04d}",
                    class_name=hazard.type,
                    confidence=conf,
                    bbox=bbox_str,
                    lat=noisy_lat,
                    lon=noisy_lon,
                    timestamp=iso_ts,
                    is_simulated=True,
                )
                detections.append(record)
                det_counter += 1
                frame_seq += rng.randint(15, 35)

        # Inject 1 false positive per drive if enabled
        if include_false_positives:
            fp_lat, fp_lon = add_gps_noise(12.9735, 77.5965, std_meters=15.0, rng=rng)
            fp_conf = round(rng.uniform(0.50, 0.68), 2)
            fp_class = "pothole" if rng.random() > 0.4 else "animal"
            ts = base_time + timedelta(seconds=frame_seq * 4)
            iso_ts = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

            fp_record = DetectionRecord(
                detection_id=f"det_{det_counter:03d}",
                drive_id=drive_id,
                frame_id=f"frame_{frame_seq:04d}",
                class_name=fp_class,
                confidence=fp_conf,
                bbox="[220, 290, 275, 345]",
                lat=fp_lat,
                lon=fp_lon,
                timestamp=iso_ts,
                is_simulated=True,
            )
            detections.append(fp_record)
            det_counter += 1
            false_positives_count += 1

    metadata = {
        "random_seed": seed,
        "gps_noise_std_meters": gps_noise_std_meters,
        "num_drives": len(drive_configs),
        "total_ground_truth_hazards": len(hazards),
        "total_possible_detections": total_possible_detections,
        "actual_true_detections": actual_true_detections,
        "false_positives_count": false_positives_count,
        "total_detections_generated": len(detections),
    }

    return detections, metadata


def save_detections_to_csv(detections: List[DetectionRecord], output_path: Path) -> None:
    """Save detections strictly according to the Section 8 file contract."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "detection_id",
        "drive_id",
        "frame_id",
        "class",
        "confidence",
        "bbox",
        "lat",
        "lon",
        "timestamp",
        "is_simulated",
    ]
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for det in detections:
            writer.writerow(det.to_contract_dict())


def save_ground_truth_to_csv(hazards: List[GroundTruthHazard], output_path: Path) -> None:
    """Save ground truth hazards to CSV for evaluation of merge and decay algorithms."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "hazard_id",
        "type",
        "lat",
        "lon",
        "segment_id",
        "created_drive",
        "repaired_drive",
        "status",
    ]
    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for h in hazards:
            writer.writerow(asdict(h))


def run_generator(
    output_detections_path: Optional[Path] = None,
    output_gt_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
    seed: int = 42,
) -> Tuple[List[DetectionRecord], List[GroundTruthHazard], Dict[str, object]]:
    """Run full simulated drive generation pipeline and write contract files."""
    repo_root = Path(__file__).resolve().parent.parent

    if output_detections_path is None:
        output_detections_path = repo_root / "data" / "processed" / "simulated_detections.csv"
    if output_gt_path is None:
        output_gt_path = repo_root / "data" / "processed" / "ground_truth_hazards.csv"
    if metrics_path is None:
        metrics_path = repo_root / "evaluation" / "results" / "simulated_drives_metrics.json"

    hazards = generate_ground_truth_hazards(seed=seed)
    detections, metadata = simulate_multi_drive_detections(hazards, seed=seed)

    save_detections_to_csv(detections, output_detections_path)
    save_ground_truth_to_csv(hazards, output_gt_path)

    # Save metrics to evaluation/results per Rule 4
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, mode="w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return detections, hazards, metadata


if __name__ == "__main__":
    dets, gts, meta = run_generator()
    print(f"Generated {len(dets)} simulated detections across {meta['num_drives']} drives.")
    print(f"Ground-truth hazards tracked: {len(gts)}")
    print("Metrics written to evaluation/results/simulated_drives_metrics.json")
