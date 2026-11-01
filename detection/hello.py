"""Detection Module - SafePath.

Responsible for road hazard detection (potholes, speed breakers, animals)
from road images and video frames. (Owner A)
"""

from pathlib import Path
import csv


def hello_detection() -> str:
    """Print status and read sample detections."""
    module_name = "Detection Module (Owner A)"
    sample_file = Path(__file__).resolve().parent.parent / "data" / "sample" / "detections.csv"
    count = 0
    if sample_file.exists():
        with open(sample_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = sum(1 for _ in reader)
    return f"[{module_name}] Ready. Found {count} sample detections in {sample_file.name}."


def main():
    print(hello_detection())


if __name__ == "__main__":
    main()

