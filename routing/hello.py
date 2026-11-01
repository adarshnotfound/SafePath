"""Routing Module - SafePath.

Responsible for OSM graph handling (OSMnx), multi-route generation
(OSRM / k-shortest paths), length-weighted safety aggregation, and route ranking. (Owner D)
"""

from pathlib import Path
import json


def hello_routing() -> str:
    """Print status and read sample routes."""
    module_name = "Routing Module (Owner D)"
    sample_file = Path(__file__).resolve().parent.parent / "data" / "sample" / "routes.json"
    count = 0
    if sample_file.exists():
        with open(sample_file, mode="r", encoding="utf-8") as f:
            data = json.load(f)
            count = len(data) if isinstance(data, list) else 1
    return f"[{module_name}] Ready. Found {count} sample routes in {sample_file.name}."


def main():
    print(hello_routing())


if __name__ == "__main__":
    main()

