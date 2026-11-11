"""Genetic Algorithm Decay Rate Optimizer - SafePath Module B2.

Replaces hand-set temporal decay parameters with values learned by a Genetic Algorithm (DEAP):
- Chromosome: [decay_rate_sb, k_sb, decay_rate_ph, k_ph, decay_rate_animal, k_animal]
- Fitness: Harmonic mean (F1) of repair detection recall and active new-hazard retention
- Generalization: Trained on Seed Set 1 (e.g. [42, 101, 202]) and evaluated on independent Seed Set 2 (e.g. [303, 404, 505])
- Generates side-by-side comparison tables and matplotlib convergence plots.

Owner: Member B (Hazard map and accident-data lead)
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.pyplot as plt
import numpy as np
from deap import base, creator, tools

from hazard_map.living_map import (
    DetectionItem,
    HazardRecord,
    LivingHazardMap,
    LivingMapConfig,
    TypePersistenceConfig,
    haversine_distance_meters,
)
from hazard_map.simulated_drive import (
    GroundTruthHazard,
    generate_ground_truth_hazards,
    simulate_multi_drive_detections,
)

# Chromosome parameter definitions: (min_bound, max_bound, is_integer)
GENE_SPECS = [
    {"name": "speed_breaker_decay", "min": 0.01, "max": 0.20, "is_int": False, "hand_val": 0.02},
    {"name": "speed_breaker_k",     "min": 1,    "max": 4,    "is_int": True,  "hand_val": 3},
    {"name": "pothole_decay",       "min": 0.20, "max": 0.75, "is_int": False, "hand_val": 0.55},
    {"name": "pothole_k",           "min": 1,    "max": 3,    "is_int": True,  "hand_val": 1},
    {"name": "animal_decay",        "min": 0.50, "max": 0.95, "is_int": False, "hand_val": 0.85},
    {"name": "animal_k",           "min": 1,    "max": 2,    "is_int": True,  "hand_val": 1},
]


def clamp_gene(val: float, spec: Dict[str, Any]) -> float:
    """Clamp individual gene to allowed domain and enforce integer rounding where required."""
    if spec["is_int"]:
        rounded = int(round(val))
        return float(max(int(spec["min"]), min(int(spec["max"]), rounded)))
    else:
        return float(max(spec["min"], min(spec["max"], val)))


def individual_to_config(individual: List[float], base_config: Optional[LivingMapConfig] = None) -> LivingMapConfig:
    """Map GA chromosome to a validated LivingMapConfig."""
    cfg = LivingMapConfig.load() if base_config is None else copy.deepcopy(base_config)

    sb_decay = clamp_gene(individual[0], GENE_SPECS[0])
    sb_k = int(clamp_gene(individual[1], GENE_SPECS[1]))
    ph_decay = clamp_gene(individual[2], GENE_SPECS[2])
    ph_k = int(clamp_gene(individual[3], GENE_SPECS[3]))
    an_decay = clamp_gene(individual[4], GENE_SPECS[4])
    an_k = int(clamp_gene(individual[5], GENE_SPECS[5]))

    cfg.type_persistence["speed_breaker"] = TypePersistenceConfig(
        decay_rate=sb_decay,
        freshness_decay_rate=min(0.95, sb_decay * 1.5),
        miss_threshold_k=sb_k,
        description="GA-learned speed breaker persistence",
    )
    cfg.type_persistence["pothole"] = TypePersistenceConfig(
        decay_rate=ph_decay,
        freshness_decay_rate=min(0.95, ph_decay * 1.2),
        miss_threshold_k=ph_k,
        description="GA-learned pothole persistence",
    )
    cfg.type_persistence["animal"] = TypePersistenceConfig(
        decay_rate=an_decay,
        freshness_decay_rate=min(0.95, an_decay * 1.1),
        miss_threshold_k=an_k,
        description="GA-learned animal persistence",
    )
    return cfg


def evaluate_scenario(config: LivingMapConfig, seed: int) -> Dict[str, float]:
    """Simulate drives for a seed and compute repair detection and active retention metrics."""
    hazards = generate_ground_truth_hazards(seed=seed)
    raw_dets, _ = simulate_multi_drive_detections(hazards, seed=seed)

    dets = [
        DetectionItem(
            detection_id=d.detection_id,
            drive_id=d.drive_id,
            class_name=d.class_name,
            confidence=d.confidence,
            lat=d.lat,
            lon=d.lon,
            timestamp=d.timestamp,
            is_simulated=d.is_simulated,
        )
        for d in raw_dets
    ]

    living_map = LivingHazardMap(config=config)
    living_map.process_all_drives(dets)

    repaired_gt = [h for h in hazards if h.repaired_drive is not None or h.status in ("repaired", "expired")]
    active_gt = [h for h in hazards if h.repaired_drive is None and h.status == "active"]

    # Evaluate repair detection
    tp_repair = 0
    fn_repair = 0
    for r in repaired_gt:
        matched = None
        best_d = float("inf")
        for mh in living_map.hazards.values():
            if mh.type == r.type:
                d = haversine_distance_meters(r.lat, r.lon, mh.lat, mh.lon)
                if d <= 30.0 and d < best_d:
                    best_d = d
                    matched = mh
        if matched is not None and (
            matched.status in ("repaired", "decayed")
            or matched.confidence < config.min_active_confidence
        ):
            tp_repair += 1
        else:
            fn_repair += 1

    # Evaluate active hazard retention
    tp_active = 0
    fn_active = 0  # actively mistakenly decayed (false decay)
    for a in active_gt:
        matched = None
        best_d = float("inf")
        for mh in living_map.hazards.values():
            if mh.type == a.type:
                d = haversine_distance_meters(a.lat, a.lon, mh.lat, mh.lon)
                if d <= 30.0 and d < best_d:
                    best_d = d
                    matched = mh
        if matched is not None and matched.status == "active" and matched.confidence >= config.min_active_confidence:
            tp_active += 1
        else:
            fn_active += 1

    recall_repair = tp_repair / max(1, (tp_repair + fn_repair))
    fp_repair = fn_active
    precision_repair = tp_repair / max(1, (tp_repair + fp_repair))

    if precision_repair + recall_repair > 0:
        f1_repair = 2.0 * precision_repair * recall_repair / (precision_repair + recall_repair)
    else:
        f1_repair = 0.0

    retention_active = tp_active / max(1, (tp_active + fn_active))

    # Overall Fitness: Harmonic mean of repair F1 and active retention recall
    if f1_repair + retention_active > 0:
        fitness = 2.0 * f1_repair * retention_active / (f1_repair + retention_active)
    else:
        fitness = 0.0

    return {
        "fitness": fitness,
        "f1_repair": f1_repair,
        "retention_active": retention_active,
        "recall_repair": recall_repair,
        "precision_repair": precision_repair,
    }


def evaluate_individual(individual: List[float], seeds: List[int]) -> Tuple[float]:
    """DEAP fitness evaluation function over training seeds."""
    cfg = individual_to_config(individual)
    scores = [evaluate_scenario(cfg, s)["fitness"] for s in seeds]
    mean_fitness = sum(scores) / len(scores)
    return (mean_fitness,)


def custom_crossover(ind1: List[float], ind2: List[float], alpha: float = 0.5) -> Tuple[List[float], List[float]]:
    """Hybrid crossover: Blend crossover for continuous rates, uniform crossover for integer thresholds."""
    for i in range(len(ind1)):
        spec = GENE_SPECS[i]
        if spec["is_int"]:
            if random.random() < 0.5:
                ind1[i], ind2[i] = ind2[i], ind1[i]
        else:
            # Blend crossover (BLX-alpha)
            x1, x2 = ind1[i], ind2[i]
            diff = abs(x1 - x2)
            low = min(x1, x2) - alpha * diff
            high = max(x1, x2) + alpha * diff
            ind1[i] = clamp_gene(random.uniform(low, high), spec)
            ind2[i] = clamp_gene(random.uniform(low, high), spec)
    return ind1, ind2


def custom_mutation(individual: List[float], sigma: float = 0.08, indpb: float = 0.3) -> Tuple[List[float]]:
    """Hybrid mutation: Gaussian perturbation for continuous genes, step changes for integer genes."""
    for i in range(len(individual)):
        if random.random() < indpb:
            spec = GENE_SPECS[i]
            if spec["is_int"]:
                step = random.choice([-1, 1])
                individual[i] = clamp_gene(individual[i] + step, spec)
            else:
                perturbation = random.gauss(0.0, sigma)
                individual[i] = clamp_gene(individual[i] + perturbation, spec)
    return (individual,)


# Ensure DEAP classes are registered cleanly once
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMax)


def run_genetic_algorithm(
    train_seeds: Optional[List[int]] = None,
    test_seeds: Optional[List[int]] = None,
    pop_size: int = 30,
    generations: int = 25,
    crossover_prob: float = 0.7,
    mutation_prob: float = 0.3,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Execute Genetic Algorithm optimization of decay rates using DEAP.

    GA Settings & Design Rationale:
    - Population (N=30): Balances exploration of 6D parameter space with fast evaluation time.
    - Generations (G=25): Sufficient to achieve fitness convergence without stagnation.
    - Selection (Tournament, k=3): Preserves selection pressure while preventing premature convergence.
    - Crossover (Blend BLX-0.5 + Uniform): Smoothly interpolates continuous rates and swaps discrete thresholds.
    - Mutation (Gaussian sigma=0.08, p=0.3): Provides fine-grained local hill-climbing around candidate optima.
    - Elitism (Hall of Fame, size=2): Guarantees the best-found chromosome is never lost across generations.

    Args:
        train_seeds: Seed set 1 for training fitness evaluation.
        test_seeds: Seed set 2 for independent validation.
        pop_size: Population size (default 30).
        generations: Generation count (default 25).
        crossover_prob: Crossover probability (default 0.7).
        mutation_prob: Mutation probability (default 0.3).
        random_seed: Seed for GA reproducibility.

    Returns:
        Dictionary containing best chromosome, convergence logs, and test set comparison.
    """
    if train_seeds is None:
        train_seeds = [42, 101, 202]
    if test_seeds is None:
        test_seeds = [303, 404, 505]

    rng = random.Random(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    toolbox = base.Toolbox()

    def create_random_individual() -> creator.Individual:
        genes = [
            clamp_gene(rng.uniform(spec["min"], spec["max"]), spec)
            for spec in GENE_SPECS
        ]
        return creator.Individual(genes)

    toolbox.register("individual", create_random_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual, seeds=train_seeds)
    toolbox.register("mate", custom_crossover, alpha=0.5)
    toolbox.register("mutate", custom_mutation, sigma=0.08, indpb=0.3)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(maxsize=2)

    # Initial evaluation
    fitnesses = [toolbox.evaluate(ind) for ind in population]
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit
    hof.update(population)

    history_stats: List[Dict[str, float]] = []

    def record_stats(gen_idx: int, pop: List[creator.Individual]):
        fits = [ind.fitness.values[0] for ind in pop]
        stats = {
            "generation": gen_idx,
            "max": float(np.max(fits)),
            "mean": float(np.mean(fits)),
            "min": float(np.min(fits)),
            "std": float(np.std(fits)),
        }
        history_stats.append(stats)

    record_stats(0, population)

    # Evolution loop
    for gen in range(1, generations + 1):
        # Selection
        offspring = toolbox.select(population, len(population) - len(hof))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Crossover
        for i in range(1, len(offspring), 2):
            if random.random() < crossover_prob:
                toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

        # Mutation
        for ind in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(ind)
                del ind.fitness.values

        # Evaluate invalid offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = [toolbox.evaluate(ind) for ind in invalid_ind]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Elitism: retain Hall of Fame individuals
        population = offspring + [toolbox.clone(ind) for ind in hof]
        hof.update(population)
        record_stats(gen, population)

    best_individual = list(hof[0])

    # Convert hand-set and GA-fitted to configs for evaluation
    handset_genes = [spec["hand_val"] for spec in GENE_SPECS]
    cfg_handset = individual_to_config(handset_genes)
    cfg_ga = individual_to_config(best_individual)

    # Evaluate on Training Set (Seed Set 1)
    train_handset_eval = [evaluate_scenario(cfg_handset, s) for s in train_seeds]
    train_ga_eval = [evaluate_scenario(cfg_ga, s) for s in train_seeds]

    # Evaluate on Independent Test Set (Seed Set 2)
    test_handset_eval = [evaluate_scenario(cfg_handset, s) for s in test_seeds]
    test_ga_eval = [evaluate_scenario(cfg_ga, s) for s in test_seeds]

    def aggregate_eval(evals: List[Dict[str, float]]) -> Dict[str, float]:
        return {
            "fitness_f1": round(float(np.mean([e["fitness"] for e in evals])), 4),
            "repair_f1": round(float(np.mean([e["f1_repair"] for e in evals])), 4),
            "retention_active": round(float(np.mean([e["retention_active"] for e in evals])), 4),
            "repair_recall": round(float(np.mean([e["recall_repair"] for e in evals])), 4),
            "repair_precision": round(float(np.mean([e["precision_repair"] for e in evals])), 4),
        }

    comparison_results = {
        "parameters": [
            {
                "parameter": spec["name"],
                "hand_set_value": spec["hand_val"],
                "ga_fitted_value": int(best_individual[i]) if spec["is_int"] else round(float(best_individual[i]), 4),
                "type": "integer" if spec["is_int"] else "float",
            }
            for i, spec in enumerate(GENE_SPECS)
        ],
        "training_performance_seed_set_1": {
            "train_seeds": train_seeds,
            "hand_set": aggregate_eval(train_handset_eval),
            "ga_fitted": aggregate_eval(train_ga_eval),
        },
        "generalization_test_seed_set_2": {
            "test_seeds": test_seeds,
            "hand_set": aggregate_eval(test_handset_eval),
            "ga_fitted": aggregate_eval(test_ga_eval),
        },
        "ga_settings": {
            "population_size": pop_size,
            "generations": generations,
            "crossover_probability": crossover_prob,
            "mutation_probability": mutation_prob,
            "crossover_operator": "Blend Crossover (BLX-0.5) + Discrete Swap",
            "mutation_operator": "Gaussian Perturbation (sigma=0.08) + Step Mutation",
            "selection_operator": "Tournament Selection (tournsize=3)",
            "elitism": "Hall of Fame (size=2)",
            "random_seed": random_seed,
        },
        "convergence_history": history_stats,
    }

    return comparison_results


def plot_ga_convergence_curve(
    history: List[Dict[str, float]],
    output_png_path: Path,
) -> None:
    """Render and save a publication-quality GA convergence plot."""
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    gens = [h["generation"] for h in history]
    max_fits = [h["max"] for h in history]
    mean_fits = [h["mean"] for h in history]
    min_fits = [h["min"] for h in history]

    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=300)

    # Shaded envelope between min and max fitness
    ax.fill_between(gens, min_fits, max_fits, color="#3498DB", alpha=0.18, label="Fitness Envelope (Min-Max)")

    # Mean fitness curve
    ax.plot(gens, mean_fits, color="#E67E22", linestyle="--", linewidth=2.0, label="Mean Population Fitness")

    # Best fitness curve
    ax.plot(gens, max_fits, color="#2980B9", linewidth=2.5, marker="o", markersize=4, label="Best Individual Fitness")

    # Annotate final best fitness
    final_best = max_fits[-1]
    ax.annotate(
        f"Optimal Fitness: {final_best:.4f}",
        xy=(gens[-1], final_best),
        xytext=(gens[-1] - 6, final_best - 0.08),
        arrowprops=dict(facecolor="#2C3E50", arrowstyle="->", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.4", fc="#EAEDED", ec="#2C3E50", lw=1.2),
        fontweight="bold",
        fontsize=10,
    )

    ax.set_title("Genetic Algorithm Convergence: Hazard Decay Rate Optimization", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Generation", fontsize=11, fontweight="bold")
    ax.set_ylabel("Fitness (F1 Harmonic Mean)", fontsize=11, fontweight="bold")
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0, len(gens) - 1])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_png_path)
    plt.close()


def run_full_ga_pipeline(
    output_metrics_path: Optional[Path] = None,
    output_plot_path: Optional[Path] = None,
    output_config_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute complete GA optimization pipeline, generate plot, and update config."""
    repo_root = Path(__file__).resolve().parent.parent

    if output_metrics_path is None:
        output_metrics_path = repo_root / "evaluation" / "results" / "ga_decay_optimization.json"
    if output_plot_path is None:
        output_plot_path = repo_root / "evaluation" / "plots" / "ga_decay_convergence.png"
    if output_config_path is None:
        output_config_path = repo_root / "hazard_map" / "config_ga_fitted.json"

    print("Running Genetic Algorithm optimization (DEAP)...")
    results = run_genetic_algorithm(
        train_seeds=[42, 101, 202],
        test_seeds=[303, 404, 505],
        pop_size=30,
        generations=25,
        random_seed=42,
    )

    # 1. Save convergence curve plot
    plot_ga_convergence_curve(results["convergence_history"], output_plot_path)

    # Also save copy in evaluation/results for unified reporting
    results_copy_plot = repo_root / "evaluation" / "results" / "ga_decay_convergence.png"
    results_copy_plot.parent.mkdir(parents=True, exist_ok=True)
    plot_ga_convergence_curve(results["convergence_history"], results_copy_plot)

    # 2. Save JSON metrics
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_metrics_path, mode="w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # 3. Export fitted config
    best_genes = [p["ga_fitted_value"] for p in results["parameters"]]
    ga_config = individual_to_config(best_genes)

    config_dict = {
        "dbscan_eps_meters": ga_config.dbscan_eps_meters,
        "dbscan_min_samples": ga_config.dbscan_min_samples,
        "association_radius_meters": ga_config.association_radius_meters,
        "coverage_radius_meters": ga_config.coverage_radius_meters,
        "default_miss_threshold_k": ga_config.default_miss_threshold_k,
        "confidence_rise_factor": ga_config.confidence_rise_factor,
        "max_confidence": ga_config.max_confidence,
        "min_active_confidence": ga_config.min_active_confidence,
        "min_active_freshness": ga_config.min_active_freshness,
        "type_persistence": {
            k: asdict(v) for k, v in ga_config.type_persistence.items()
        },
    }

    output_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config_path, mode="w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    return results


if __name__ == "__main__":
    res = run_full_ga_pipeline()
    print("=" * 70)
    print("SafePath GA Decay Rate Optimization Results (DEAP)")
    print("=" * 70)
    print(f"{'Parameter':<24} | {'Hand-Set':<12} | {'GA-Fitted':<12}")
    print("-" * 70)
    for p in res["parameters"]:
        print(f"{p['parameter']:<24} | {str(p['hand_set_value']):<12} | {str(p['ga_fitted_value']):<12}")
    print("-" * 70)
    t1 = res["training_performance_seed_set_1"]
    t2 = res["generalization_test_seed_set_2"]
    print(f"Train Set 1 F1 (Hand-set): {t1['hand_set']['fitness_f1']}  -->  (GA-Fitted): {t1['ga_fitted']['fitness_f1']}")
    print(f"Test Set 2 F1  (Hand-set): {t2['hand_set']['fitness_f1']}  -->  (GA-Fitted): {t2['ga_fitted']['fitness_f1']}")
    print("-" * 70)
    print("Metrics written to evaluation/results/ga_decay_optimization.json")
    print("Plot written to    evaluation/plots/ga_decay_convergence.png")
    print("Config written to  hazard_map/config_ga_fitted.json")
    print("=" * 70)
