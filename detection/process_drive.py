"""End-to-end video and drive processing pipeline for SafePath (Module 1).

Follows Section 4.1, 4.2, & Section 8 of the SafePath Specification:
- Samples 1 frame every 1-2 seconds using OpenCV.
- Estimates per-frame lighting condition ('well lit', 'dim', 'dark').
- Executes hazard detector (YOLOv8-nano / Baseline).
- Projects detections along a route polyline with simulated 5-15m GPS noise (Section 4.2).
- Formats timestamps and exports to Section 8 compliant `detections.csv`.
- Discloses all synthetic positions with `is_simulated = True`.
"""

import math
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from detection.detector_base import BaseHazardDetector, DetectionResult
from detection.export import save_detections_to_csv
from detection.frame_extractor import sample_frames_from_video
from detection.yolo_detector import YOLOHazardDetector

# Default Bangalore Route Polyline (M.G. Road to Indiranagar) for coordinate projection
DEFAULT_ROUTE_WAYPOINTS: List[Tuple[float, float]] = [
    (12.9716, 77.5946),  # Waypoint 1 (M.G. Road)
    (12.9735, 77.6075),  # Waypoint 2 (Trinity)
    (12.9754, 77.6200),  # Waypoint 3 (Halasuru)
    (12.9783, 77.6408),  # Waypoint 4 (Indiranagar 100ft Rd)
]


def interpolate_route_position(
    progress_fraction: float,
    waypoints: List[Tuple[float, float]],
) -> Tuple[float, float]:
    """Interpolate latitude and longitude along a sequence of GPS waypoints.

    Parameters
    ----------
    progress_fraction : float
        Value between 0.0 (start) and 1.0 (end of route).
    waypoints : List[Tuple[float, float]]
        List of (lat, lon) coordinates defining the route polyline.

    Returns
    -------
    Tuple[float, float]
        Interpolated (lat, lon) on the trajectory.
    """
    if not waypoints:
        return (12.9716, 77.5946)
    if len(waypoints) == 1 or progress_fraction <= 0.0:
        return waypoints[0]
    if progress_fraction >= 1.0:
        return waypoints[-1]

    # Calculate segment distances
    total_segments = len(waypoints) - 1
    scaled_progress = progress_fraction * total_segments
    seg_idx = min(int(scaled_progress), total_segments - 1)
    seg_fraction = scaled_progress - seg_idx

    p1 = waypoints[seg_idx]
    p2 = waypoints[seg_idx + 1]

    interp_lat = p1[0] + (p2[0] - p1[0]) * seg_fraction
    interp_lon = p1[1] + (p2[1] - p1[1]) * seg_fraction

    return interp_lat, interp_lon


def apply_gps_noise(
    lat: float,
    lon: float,
    noise_meters: float,
    angle_rad: float,
) -> Tuple[float, float]:
    """Apply realistic 5-15m random GPS error to coordinates (Section 4.2).

    1 degree latitude ~= 111,139 meters.
    1 degree longitude ~= 111,139 * cos(lat) meters.
    """
    meters_per_deg_lat = 111139.0
    meters_per_deg_lon = 111139.0 * math.cos(math.radians(lat))

    delta_lat = (noise_meters * math.sin(angle_rad)) / meters_per_deg_lat
    delta_lon = (noise_meters * math.cos(angle_rad)) / meters_per_deg_lon

    return round(lat + delta_lat, 5), round(lon + delta_lon, 5)


def process_drive_video(
    video_path: Union[str, Path],
    drive_id: str = "drive_01",
    detector: Optional[BaseHazardDetector] = None,
    waypoints: Optional[List[Tuple[float, float]]] = None,
    sample_interval_sec: float = 1.0,
    start_time_iso: Optional[str] = None,
    output_csv_path: Optional[Union[str, Path]] = None,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Execute end-to-end drive processing on a video file.

    Parameters
    ----------
    video_path : Union[str, Path]
        Path to road drive footage (.mp4, .avi).
    drive_id : str
        Drive session identifier (e.g., 'drive_01').
    detector : Optional[BaseHazardDetector]
        Detector model instance (defaults to YOLOHazardDetector).
    waypoints : Optional[List[Tuple[float, float]]]
        Route coordinate polyline for GPS projection (Section 4.2).
    sample_interval_sec : float
        Frame sampling interval in seconds (default 1.0s, per Section 4.1).
    start_time_iso : Optional[str]
        Base UTC start timestamp in ISO 8601.
    output_csv_path : Optional[Union[str, Path]]
        Path to save Section 8 compliant detections.csv.
    random_seed : int
        Fixed seed for reproducible GPS noise and simulation.

    Returns
    -------
    dict
        Processing summary and list of detection results.
    """
    rng = random.Random(random_seed)

    if detector is None:
        detector = YOLOHazardDetector(confidence_threshold=0.5, random_seed=random_seed)

    if waypoints is None:
        waypoints = DEFAULT_ROUTE_WAYPOINTS

    if start_time_iso:
        base_time = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00"))
    else:
        base_time = datetime.now(timezone.utc)

    all_detections: List[DetectionResult] = []
    frames_processed = 0
    lighting_counts = {"well lit": 0, "dim": 0, "dark": 0}

    # Stream frames through OpenCV sampler
    for frame_data in sample_frames_from_video(video_path, sample_interval_sec=sample_interval_sec):
        frames_processed += 1
        frame_id = str(frame_data["frame_id"])
        timestamp_sec = float(frame_data["timestamp_sec"])
        frame_img = frame_data["frame"]
        condition = str(frame_data["lighting_condition"])
        lighting_counts[condition] = lighting_counts.get(condition, 0) + 1

        # Calculate timestamp for current frame
        frame_dt = base_time + timedelta(seconds=timestamp_sec)
        iso_timestamp = frame_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        # Run hazard detection
        detections = detector.predict(frame_img, frame_id=frame_id, drive_id=drive_id)

        # Place detections on the route polyline with simulated GPS drift (Section 4.2)
        # Approximate progress along route assuming 60s total nominal video drive length
        progress = min(1.0, timestamp_sec / 60.0)
        base_lat, base_lon = interpolate_route_position(progress, waypoints)

        for det in detections:
            # 5-15m random GPS error to mimic phone GPS inaccuracy
            gps_noise_m = rng.uniform(5.0, 15.0)
            noise_angle = rng.uniform(0.0, 2 * math.pi)
            placed_lat, placed_lon = apply_gps_noise(base_lat, base_lon, gps_noise_m, noise_angle)

            det.lat = placed_lat
            det.lon = placed_lon
            det.timestamp = iso_timestamp
            det.is_simulated = True  # Strict Section 4.2 data honesty labeling
            all_detections.append(det)

    if output_csv_path:
        save_detections_to_csv(all_detections, output_csv_path)

    return {
        "drive_id": drive_id,
        "frames_sampled": frames_processed,
        "total_detections": len(all_detections),
        "lighting_distribution": lighting_counts,
        "detector_model": detector.model_name,
        "output_csv": str(output_csv_path) if output_csv_path else None,
        "detections": all_detections,
    }


def create_mock_video_sequence(
    output_mp4_path: Path,
    num_frames: int = 15,
    fps: int = 5,
) -> Path:
    """Generate a lightweight synthetic MP4 video file to test drive processing."""
    try:
        import cv2

        output_mp4_path = Path(output_mp4_path)
        output_mp4_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_mp4_path), fourcc, fps, (640, 360))

        for i in range(num_frames):
            # Generate simulated asphalt road gradient
            frame = np.full((360, 640, 3), 70, dtype=np.uint8)
            # Add road horizon and lane markers
            frame[240:, :] = 45
            frame[290:310, 310:330] = 230  # Lane dash
            writer.write(frame)

        writer.release()
        return output_mp4_path
    except Exception as e:
        raise RuntimeError(f"OpenCV video writing failed: {e}")


def main():
    """Demo runner executing simulated drive processing."""
    temp_dir = Path(__file__).resolve().parent.parent / "data" / "processed" / "sim_drives"
    temp_dir.mkdir(parents=True, exist_ok=True)
    video_file = temp_dir / "demo_drive.mp4"
    output_csv = temp_dir / "demo_detections.csv"

    print("Generating demo road video...")
    create_mock_video_sequence(video_file, num_frames=10, fps=2)

    print("Executing SafePath Drive Processor (Section 4.1 & 4.2)...")
    summary = process_drive_video(
        video_path=video_file,
        drive_id="drive_demo_01",
        sample_interval_sec=1.0,
        output_csv_path=output_csv,
    )

    print("=" * 60)
    print(f"Drive Processing Complete: {summary['drive_id']}")
    print(f"Frames Sampled:          {summary['frames_sampled']}")
    print(f"Total Hazards Detected:  {summary['total_detections']}")
    print(f"Lighting Profile:        {summary['lighting_distribution']}")
    print(f"Section 8 CSV Export:    {summary['output_csv']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
