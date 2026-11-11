"""Helper to generate 3 sample test images for lighting and batch detection tests."""

from pathlib import Path
from typing import List

import cv2
import numpy as np


def create_three_sample_images(output_dir: Path) -> List[Path]:
    """Create 3 distinct sample road images: daylight (well_lit), dusk (dim), night (dark)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Daylight image (well_lit: mean luminance ~170)
    img_day = np.full((360, 640, 3), 165, dtype=np.uint8)
    cv2.circle(img_day, (320, 240), 40, (50, 50, 50), -1)  # Simulated pothole
    path_day = output_dir / "sample_01_daylight.jpg"
    cv2.imwrite(str(path_day), img_day)

    # 2. Dusk image (dim: mean luminance ~85)
    img_dusk = np.full((360, 640, 3), 85, dtype=np.uint8)
    cv2.rectangle(img_dusk, (200, 260), (440, 290), (35, 35, 35), -1)  # Speed bump
    path_dusk = output_dir / "sample_02_dusk.jpg"
    cv2.imwrite(str(path_dusk), img_dusk)

    # 3. Night image (dark: mean luminance ~25)
    img_night = np.full((360, 640, 3), 22, dtype=np.uint8)
    # Headlight spot
    cv2.circle(img_night, (320, 300), 70, (80, 80, 85), -1)
    path_night = output_dir / "sample_03_night.jpg"
    cv2.imwrite(str(path_night), img_night)

    return [path_day, path_dusk, path_night]


if __name__ == "__main__":
    sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample" / "test_images"
    created = create_three_sample_images(sample_dir)
    print(f"Created {len(created)} sample test images in {sample_dir}")
