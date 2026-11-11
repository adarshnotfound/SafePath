"""SafePath Application Configuration & Data Provenance (Role D).

Provides a single configuration path to switch between:
1. "mock" mode (loading from `data/sample/`)
2. "real" mode (loading from `data/processed/` and `data/raw/`)

Adheres to Section 4.10, Section 8, and Section 12 (Data Provenance & Honesty).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DIR = DATA_DIR / "sample"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"
EVAL_DIR = BASE_DIR / "evaluation" / "results"

# Standard landmarks in Prayagraj for quick route selection
PRAYAGRAJ_LANDMARKS: Dict[str, Dict[str, float]] = {
    "Civil Lines (Central Commercial)": {"lat": 25.4526, "lon": 81.8345},
    "Prayagraj Junction Railway Station": {"lat": 25.4463, "lon": 81.8286},
    "Triveni Sangam (Kumbh Mela Grounds)": {"lat": 25.4290, "lon": 81.8840},
    "Allahabad High Court": {"lat": 25.4497, "lon": 81.8172},
    "University of Allahabad (Senate Hall)": {"lat": 25.4590, "lon": 81.8540},
    "Naini Industrial Area / Bridge": {"lat": 25.4180, "lon": 81.8600},
    "Subedarganj Station": {"lat": 25.4410, "lon": 81.8020},
    "Company Bagh (Alfred Park)": {"lat": 25.4542, "lon": 81.8520},
}


@dataclass
class AppConfig:
    """Configuration paths and metadata for the current execution mode."""

    mode: str  # "mock" or "real"
    routes_json: Path
    segment_features_csv: Path
    segment_scores_csv: Path
    segment_metadata_csv: Path
    hazards_csv: Path
    accident_hotspots_csv: Path
    graph_path: Path
    city_name: str = "Prayagraj, Uttar Pradesh, India"

    @property
    def is_real_mode(self) -> bool:
        return self.mode == "real"

    def get_provenance_labels(self) -> Dict[str, str]:
        """Return explicit provenance classification per Section 12 of spec."""
        if self.is_real_mode:
            return {
                "Road Network & Distances": "Real (OpenStreetMap via OSMnx)",
                "Safety POIs (Police, Hospitals)": "Real (OpenStreetMap via Overpass API)",
                "Pothole & Hazard Detections": "Simulated (RDD2022 trained model on simulated drive)",
                "Accident Hotspots": "Substitute (Historical clustering proxy)",
                "Safety Scores": "Substitute / Heuristic (Mamdani rule base & survey proxy)",
            }
        return {
            "Road Network & Distances": "Sample (Fixed demonstration fixtures)",
            "Safety POIs (Police, Hospitals)": "Sample (Small test subset)",
            "Pothole & Hazard Detections": "Sample / Simulated (Mock detections)",
            "Accident Hotspots": "Sample (Demonstration records)",
            "Safety Scores": "Sample (Pre-scored sample segments)",
        }


def get_app_config(mode: str = "real") -> AppConfig:
    """Retrieve app config for given mode ('mock' or 'real')."""
    normalized_mode = mode.lower().strip()

    if normalized_mode == "mock":
        return AppConfig(
            mode="mock",
            routes_json=SAMPLE_DIR / "routes.json",
            segment_features_csv=SAMPLE_DIR / "segment_features.csv",
            segment_scores_csv=SAMPLE_DIR / "segment_scores.csv",
            segment_metadata_csv=PROCESSED_DIR / "segment_metadata.csv",  # Fallback for geometry
            hazards_csv=SAMPLE_DIR / "hazards.csv",
            accident_hotspots_csv=SAMPLE_DIR / "accident_hotspots.csv",
            graph_path=RAW_DIR / "prayagraj_drive.graphml",
        )

    # Real pipeline mode
    proc_routes = PROCESSED_DIR / "routes.json"
    proc_features = PROCESSED_DIR / "segment_features.csv"
    proc_scores = PROCESSED_DIR / "segment_scores.csv"
    proc_metadata = PROCESSED_DIR / "segment_metadata.csv"
    proc_hazards = PROCESSED_DIR / "hazards.csv"
    proc_hotspots = PROCESSED_DIR / "accident_hotspots.csv"

    return AppConfig(
        mode="real",
        routes_json=proc_routes if proc_routes.exists() else SAMPLE_DIR / "routes.json",
        segment_features_csv=proc_features if proc_features.exists() else SAMPLE_DIR / "segment_features.csv",
        segment_scores_csv=proc_scores if proc_scores.exists() else (
            SAMPLE_DIR / "segment_scores.csv"
        ),
        segment_metadata_csv=proc_metadata if proc_metadata.exists() else SAMPLE_DIR / "segment_features.csv",
        hazards_csv=proc_hazards if proc_hazards.exists() else SAMPLE_DIR / "hazards.csv",
        accident_hotspots_csv=proc_hotspots if proc_hotspots.exists() else SAMPLE_DIR / "accident_hotspots.csv",
        graph_path=RAW_DIR / "prayagraj_drive.graphml",
    )

