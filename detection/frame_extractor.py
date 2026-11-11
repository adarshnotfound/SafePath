"""Frame extraction and lighting condition estimation for SafePath (Module 1).

Follows Section 4.1 of the SafePath Specification:
- Samples 1 frame every 1-2 seconds from video using OpenCV.
- Provides a baseline brightness-based score per frame (mean luminance bucketed
  into 'well lit', 'dim', 'dark').
"""

from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np

# Default luminance thresholds (0-255 scale)
# Can be adjusted based on camera sensor exposure profile
DARK_THRESHOLD = 50.0
DIM_THRESHOLD = 120.0


def estimate_lighting(
    frame: np.ndarray,
    dark_thresh: float = DARK_THRESHOLD,
    dim_thresh: float = DIM_THRESHOLD,
) -> Tuple[str, float]:
    """Estimate lighting conditions of an image frame based on mean luminance.

    Converts the frame to grayscale to calculate average pixel intensity
    (0.0 - 255.0) and assigns a qualitative bucket:
    - 'dark': luminance < dark_thresh
    - 'dim': dark_thresh <= luminance < dim_thresh
    - 'well lit': luminance >= dim_thresh

    Parameters
    ----------
    frame : np.ndarray
        Input image in BGR or grayscale format.
    dark_thresh : float
        Upper luminance boundary for 'dark' category (default 50.0).
    dim_thresh : float
        Upper luminance boundary for 'dim' category (default 120.0).

    Returns
    -------
    Tuple[str, float]
        (lighting_condition, mean_luminance)
    """
    if frame is None or frame.size == 0:
        return "dark", 0.0

    if len(frame.shape) == 3 and frame.shape[2] == 3:
        # Standard conversion to grayscale: Y = 0.299*R + 0.587*G + 0.114*B
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    mean_luminance = float(np.mean(gray))

    if mean_luminance < dark_thresh:
        condition = "dark"
    elif mean_luminance < dim_thresh:
        condition = "dim"
    else:
        condition = "well lit"

    return condition, round(mean_luminance, 2)


def sample_frames_from_video(
    video_path: Union[str, Path],
    sample_interval_sec: float = 1.0,
    output_dir: Optional[Union[str, Path]] = None,
) -> Generator[Dict[str, Union[str, int, float, np.ndarray]], None, None]:
    """Sample video frames at fixed time intervals (e.g., 1 frame per 1-2 seconds).

    Parameters
    ----------
    video_path : Union[str, Path]
        Path to the target video file (.mp4, .avi, etc.).
    sample_interval_sec : float
        Interval in seconds between sampled frames (Section 4.1 specifies 1-2s).
    output_dir : Optional[Union[str, Path]]
        Optional directory to save sampled frames as PNG images.

    Yields
    ------
    dict
        Metadata for each sampled frame:
        - 'frame_id': e.g., 'frame_0001'
        - 'timestamp_sec': timestamp within video
        - 'lighting_condition': 'well lit' | 'dim' | 'dark'
        - 'mean_luminance': float
        - 'frame': numpy array of the frame
        - 'saved_path': file path if saved, else None
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Fallback for streams/metadata issues

    frame_step = max(1, int(round(fps * sample_interval_sec)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    sampled_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_step == 0:
                sampled_count += 1
                frame_id = f"frame_{sampled_count:04d}"
                timestamp_sec = round(frame_index / fps, 2)

                condition, luminance = estimate_lighting(frame)
                saved_path = None

                if output_dir:
                    saved_path = str(output_dir / f"{frame_id}.jpg")
                    cv2.imwrite(saved_path, frame)

                yield {
                    "frame_id": frame_id,
                    "frame_index": frame_index,
                    "timestamp_sec": timestamp_sec,
                    "lighting_condition": condition,
                    "mean_luminance": luminance,
                    "frame": frame,
                    "saved_path": saved_path,
                }

            frame_index += 1
            if total_frames > 0 and frame_index >= total_frames:
                break
    finally:
        cap.release()


def sample_frames_from_image_list(
    image_paths: List[Union[str, Path]],
) -> List[Dict[str, Union[str, int, float, np.ndarray]]]:
    """Process a list of static images (e.g., from RDD2022 or benchmark datasets).

    Parameters
    ----------
    image_paths : List[Union[str, Path]]
        List of image file paths.

    Returns
    -------
    List[dict]
        List of frame metadata dicts with lighting estimates.
    """
    results = []
    for idx, path in enumerate(image_paths, start=1):
        p = Path(path)
        if not p.exists():
            continue
        frame = cv2.imread(str(p))
        if frame is None:
            continue

        frame_id = f"frame_{idx:04d}"
        condition, luminance = estimate_lighting(frame)
        results.append(
            {
                "frame_id": frame_id,
                "frame_index": idx - 1,
                "timestamp_sec": float(idx - 1),
                "lighting_condition": condition,
                "mean_luminance": luminance,
                "frame": frame,
                "saved_path": str(p),
            }
        )
    return results
