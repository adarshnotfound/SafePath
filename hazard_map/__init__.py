"""SafePath Hazard Map Module - Modules B1 & B2.

Includes:
- Simulated drive generator with lifecycles and realistic noise (B1)
- Living hazard map memory with DBSCAN projected clustering and type-specific decay (B2)
- Genetic Algorithm decay rate optimization using DEAP (B2-GA)
"""

from hazard_map.evaluate import (
    LivingMapEvaluationMetrics,
    evaluate_living_map,
    run_pipeline_and_evaluate,
)
from hazard_map.ga_optimizer import (
    evaluate_scenario,
    individual_to_config,
    plot_ga_convergence_curve,
    run_full_ga_pipeline,
    run_genetic_algorithm,
)
from hazard_map.living_map import (
    HazardRecord,
    LivingHazardMap,
    LivingMapConfig,
    haversine_distance_meters,
    load_detections_from_csv,
    merge_detections_dbscan,
    project_latlon_to_xy,
    project_xy_to_latlon,
)
from hazard_map.simulated_drive import (
    DetectionRecord,
    GroundTruthHazard,
    add_gps_noise,
    generate_ground_truth_hazards,
    run_generator,
    simulate_multi_drive_detections,
)

__all__ = [
    # Module B1
    "GroundTruthHazard",
    "DetectionRecord",
    "add_gps_noise",
    "generate_ground_truth_hazards",
    "simulate_multi_drive_detections",
    "run_generator",
    # Module B2
    "HazardRecord",
    "LivingHazardMap",
    "LivingMapConfig",
    "LivingMapEvaluationMetrics",
    "merge_detections_dbscan",
    "evaluate_living_map",
    "run_pipeline_and_evaluate",
    "load_detections_from_csv",
    "project_latlon_to_xy",
    "project_xy_to_latlon",
    "haversine_distance_meters",
    # Module B2 - GA Optimization
    "run_genetic_algorithm",
    "run_full_ga_pipeline",
    "plot_ga_convergence_curve",
    "individual_to_config",
    "evaluate_scenario",
]
