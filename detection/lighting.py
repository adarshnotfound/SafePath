"""Mean luminance lighting classifier with custom user thresholds for SafePath (Module 1).

Follows Section 4.1 & User Specification:
- Classifies a frame into: 'well_lit', 'dim', or 'dark' from mean luminance.
- Supports user-chosen thresholds tuned from a small labeled set of frames.
- Provides a calibration function to compute threshold midpoints from user-labeled examples.
"""

from typing import List, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

DEFAULT_DARK_THRESHOLD = 50.0
DEFAULT_DIM_THRESHOLD = 120.0


def compute_frame_luminance(frame: np.ndarray) -> float:
    """Compute scalar mean luminance (0.0 to 255.0) of an image frame."""
    if frame is None or frame.size == 0:
        return 0.0

    if len(frame.shape) == 3 and frame.shape[2] == 3:
        if cv2 is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            # Standard Rec. 601 luma weights: Y = 0.114*B + 0.587*G + 0.299*R
            gray = 0.114 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.299 * frame[:, :, 2]
    else:
        gray = frame

    return float(np.mean(gray))


def classify_lighting(
    frame: np.ndarray,
    dark_threshold: float = DEFAULT_DARK_THRESHOLD,
    dim_threshold: float = DEFAULT_DIM_THRESHOLD,
) -> Tuple[str, float]:
    """Classify an image frame as 'well_lit', 'dim', or 'dark' from mean luminance.

    Parameters
    ----------
    frame : np.ndarray
        Input image in BGR or grayscale format.
    dark_threshold : float
        Upper luminance boundary for 'dark' (e.g. 50.0).
    dim_threshold : float
        Upper luminance boundary for 'dim' (e.g. 120.0).

    Returns
    -------
    Tuple[str, float]
        (label, mean_luminance) where label is 'dark', 'dim', or 'well_lit'.
    """
    mean_lum = compute_frame_luminance(frame)

    if mean_lum < dark_threshold:
        label = "dark"
    elif mean_lum < dim_threshold:
        label = "dim"
    else:
        label = "well_lit"

    return label, round(mean_lum, 2)


def calibrate_lighting_thresholds(
    labeled_frames: List[Tuple[np.ndarray, str]],
) -> Tuple[float, float]:
    """Calculate recommended dark and dim thresholds from a small user-labeled frame set.

    Parameters
    ----------
    labeled_frames : List[Tuple[np.ndarray, str]]
        List of pairs: (image_array, label), where label is 'dark', 'dim', or 'well_lit'.

    Returns
    -------
    Tuple[float, float]
        (recommended_dark_threshold, recommended_dim_threshold)
    """
    luminance_by_label = {"dark": [], "dim": [], "well_lit": []}

    for frame, label in labeled_frames:
        norm_label = label.lower().strip().replace(" ", "_")
        if norm_label in luminance_by_label:
            lum = compute_frame_luminance(frame)
            luminance_by_label[norm_label].append(lum)

    # Calculate cluster medians or midpoints
    dark_vals = luminance_by_label["dark"]
    dim_vals = luminance_by_label["dim"]
    well_vals = luminance_by_label["well_lit"]

    dark_mean = float(np.mean(dark_vals)) if dark_vals else 25.0
    dim_mean = float(np.mean(dim_vals)) if dim_vals else 85.0
    well_mean = float(np.mean(well_vals)) if well_vals else 170.0

    # Decision boundary midpoints
    recommended_dark = round((dark_mean + dim_mean) / 2.0, 1)
    recommended_dim = round((dim_mean + well_mean) / 2.0, 1)

    return recommended_dark, recommended_dim
