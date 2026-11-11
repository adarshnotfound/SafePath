"""Routing Module - SafePath.

Responsible for OSM graph handling (OSMnx), multi-route generation
(OSRM / k-shortest paths), length-weighted safety aggregation, and route ranking. (Owner D)
"""

import json
from pathlib import Path


def hello_routing() -> str:
    """Print status and read available routes."""
    module_name = "Routing Module (Owner D)"
    base = Path(__file__).resolve().parent.parent / "data"
    proc_file = base / "processed" / "routes.json"
    sample_file = base / "sample" / "routes.json"

    target_file = proc_file if proc_file.exists() else sample_file
    count = 0
    if target_file.exists():
        with open(target_file, mode="r", encoding="utf-8") as f:
            data = json.load(f)
            count = len(data) if isinstance(data, list) else 1
    return f"[{module_name}] Ready. Found {count} routes in {target_file.relative_to(base.parent)}."


def main():
    print(hello_routing())


if __name__ == "__main__":
    main()

