"""Accidents Module - SafePath.

Responsible for clustering historical accident data into hotspots (FCM / DBSCAN)
and training accident risk neural networks (MLP vs Logistic Regression). (Owner B)
"""

from pathlib import Path
import csv


def hello_accidents() -> str:
    """Print status and read sample accident hotspots."""
    module_name = "Accidents Module (Owner B)"
    sample_file = Path(__file__).resolve().parent.parent / "data" / "sample" / "accident_hotspots.csv"
    count = 0
    if sample_file.exists():
        with open(sample_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            count = sum(1 for _ in reader)
    return f"[{module_name}] Ready. Found {count} sample accident hotspots in {sample_file.name}."


def main():
    print(hello_accidents())


if __name__ == "__main__":
    main()

