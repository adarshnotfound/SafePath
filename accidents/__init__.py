"""SafePath Accidents Module - Modules B3 & B4.

Includes:
- Accident data loader and empirical dataset generator (data_loader.py)
- Spatial clustering and comparison: DBSCAN vs. Fuzzy C-Means (clustering.py)
- Interactive Folium map generation (visualization.py)
- Accident risk neural network: MLP vs. Logistic Regression baseline (risk_model.py)
- Module hello-world sanity check (hello.py)
"""

from accidents.clustering import (
    AccidentHotspot,
    build_hotspot_catalog,
    compare_and_export_hotspots,
    haversine_distance_meters,
    parameter_sweep_dbscan,
    parameter_sweep_fcm,
    project_coordinates_to_xy,
    project_xy_to_latlon,
    run_dbscan,
    run_fcm,
)
from accidents.data_loader import (
    AccidentPoint,
    generate_empirical_accident_dataset,
    load_accidents_from_csv,
    save_accidents_to_csv,
)
from accidents.hello import hello_accidents
from accidents.risk_model import (
    AccidentRiskModelPipeline,
    SegmentRiskFeatures,
    generate_segment_risk_dataset,
    plot_risk_comparison_roc,
    run_accident_risk_pipeline,
)
from accidents.visualization import plot_accident_hotspots_folium

__all__ = [
    # Data & Clustering (B3)
    "AccidentPoint",
    "AccidentHotspot",
    "generate_empirical_accident_dataset",
    "load_accidents_from_csv",
    "save_accidents_to_csv",
    "project_coordinates_to_xy",
    "project_xy_to_latlon",
    "haversine_distance_meters",
    "run_dbscan",
    "run_fcm",
    "parameter_sweep_dbscan",
    "parameter_sweep_fcm",
    "build_hotspot_catalog",
    "compare_and_export_hotspots",
    "plot_accident_hotspots_folium",
    "hello_accidents",
    # Accident Risk Neural Network (B4)
    "AccidentRiskModelPipeline",
    "SegmentRiskFeatures",
    "generate_segment_risk_dataset",
    "run_accident_risk_pipeline",
    "plot_risk_comparison_roc",
]
