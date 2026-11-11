"""Genetic Algorithm (GA) Tuner for SafePath Fuzzy Engine (Proposed Learned Method).

Tunes fuzzy rule weights and membership function parameters against human survey ratings
using cross-validation to minimize Mean Absolute Error (MAE).
Uses fixed random seeds (seed 42) for strict reproducibility.
"""

import copy
import random
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

try:
    from deap import base, creator, tools, algorithms
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False

from fuzzy.config import (
    DEFAULT_MAMDANI_RULES,
    DEFAULT_MEMBERSHIP_CONFIG,
)
from fuzzy.mamdani_engine import MamdaniFuzzyEngine


def survey_rating_to_safety_score(rating_1_to_5: float) -> float:
    """Map human 1-5 Likert safety rating to 0-100 continuous safety scale."""
    return float(np.clip((rating_1_to_5 - 1.0) * 25.0, 0.0, 100.0))


def safety_score_to_survey_rating(score_0_to_100: float) -> float:
    """Map 0-100 safety score back to 1-5 Likert rating."""
    return float(np.clip(1.0 + (score_0_to_100 / 25.0), 1.0, 5.0))


class FuzzyGeneticTuner:
    """Genetic Algorithm Optimizer for tuning Fuzzy Rule Weights and MF boundaries."""

    def __init__(
        self,
        segment_features: List[Dict[str, Any]],
        survey_targets: Dict[str, float],
        random_seed: int = 42,
    ):
        """
        Parameters:
            segment_features: List of feature dicts for segments.
            survey_targets: Dict mapping segment_id -> target 0-100 score (or mean rating).
            random_seed: Fixed random seed for reproducibility.
        """
        self.segments = segment_features
        self.targets = survey_targets
        self.random_seed = random_seed
        self.num_rules = len(DEFAULT_MAMDANI_RULES)
        # Total genes: num_rules (weights) + 4 MF tuning multipliers
        self.num_genes = self.num_rules + 4

    def chromosome_to_engine(self, individual: List[float]) -> MamdaniFuzzyEngine:
        """Instantiate a MamdaniFuzzyEngine with parameters decoded from the chromosome."""
        rule_weights = individual[:self.num_rules]
        mf_multipliers = individual[self.num_rules:]

        # Deep copy base membership configuration to avoid mutation side effects
        tuned_mf = copy.deepcopy(DEFAULT_MEMBERSHIP_CONFIG)

        # Multiplier 0: scales high hazard density threshold
        m_haz = max(0.5, min(2.0, mf_multipliers[0]))
        tuned_mf["hazard_density"]["high"] = [
            0.85 * m_haz, 1.30 * m_haz, 5.0, 5.0
        ]

        # Multiplier 1: scales accident score high threshold
        m_acc = max(0.6, min(1.5, mf_multipliers[1]))
        tuned_mf["accident_score"]["high"] = [
            0.60 * m_acc, 0.80 * m_acc, 1.0, 1.0
        ]

        # Multiplier 2: scales isolation score high threshold
        m_iso = max(0.6, min(1.5, mf_multipliers[2]))
        tuned_mf["isolation_score"]["high"] = [
            0.60 * m_iso, 0.80 * m_iso, 1.0, 1.0
        ]

        # Multiplier 3: shift lighting well_lit boundary
        m_lit = max(0.7, min(1.3, mf_multipliers[3]))
        tuned_mf["lighting"]["well_lit"] = [
            0.60 * m_lit, 0.80 * m_lit, 1.0, 1.0
        ]

        engine = MamdaniFuzzyEngine(
            membership_config=tuned_mf,
            rules=DEFAULT_MAMDANI_RULES,
            rule_weights=rule_weights,
            model_name="ga_tuned_fuzzy",
        )
        return engine

    def evaluate_fitness(self, individual: List[float], eval_segments: List[Dict[str, Any]] = None) -> Tuple[float]:
        """Compute Mean Absolute Error (MAE) for an individual chromosome."""
        segs = eval_segments if eval_segments is not None else self.segments
        engine = self.chromosome_to_engine(individual)

        errors = []
        for seg in segs:
            seg_id = seg.get("segment_id")
            if seg_id not in self.targets:
                continue
            pred_score = engine.score_segment(seg)["safety_score"]
            target_score = self.targets[seg_id]
            errors.append(abs(pred_score - target_score))

        if not errors:
            return (100.0,)
        return (float(np.mean(errors)),)

    def run_optimization(
        self,
        population_size: int = 35,
        generations: int = 20,
        cx_prob: float = 0.7,
        mut_prob: float = 0.25,
        train_segments: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute Genetic Algorithm using DEAP (or fallback vector GA) with seed 42."""
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        eval_segs = train_segments if train_segments is not None else self.segments

        # Gene bounds:
        # Rule weights: [0.2, 1.5], default 1.0
        # MF multipliers: [0.7, 1.3], default 1.0
        def create_individual():
            ind = [random.uniform(0.6, 1.2) for _ in range(self.num_rules)]
            ind.extend([random.uniform(0.85, 1.15) for _ in range(4)])
            return ind

        # Initialize Population
        population = [create_individual() for _ in range(population_size)]
        # Insert default baseline individual (weights = 1.0, multipliers = 1.0)
        default_ind = [1.0] * self.num_rules + [1.0] * 4
        population[0] = default_ind

        fitnesses = [self.evaluate_fitness(ind, eval_segs)[0] for ind in population]
        best_idx = int(np.argmin(fitnesses))
        best_fitness = fitnesses[best_idx]
        best_individual = list(population[best_idx])

        convergence_curve = [round(best_fitness, 3)]

        # Evolution loop
        for gen in range(1, generations + 1):
            next_generation = [best_individual]  # Elitism: retain top individual

            while len(next_generation) < population_size:
                # Tournament Selection
                c1, c2 = random.sample(population, 2)
                p1 = c1 if self.evaluate_fitness(c1, eval_segs)[0] < self.evaluate_fitness(c2, eval_segs)[0] else c2
                c3, c4 = random.sample(population, 2)
                p2 = c3 if self.evaluate_fitness(c3, eval_segs)[0] < self.evaluate_fitness(c4, eval_segs)[0] else c4

                child1, child2 = list(p1), list(p2)

                # Blend Crossover (BLX-alpha)
                if random.random() < cx_prob:
                    alpha = 0.3
                    for i in range(self.num_genes):
                        gamma = (1.0 + 2.0 * alpha) * random.random() - alpha
                        val1 = (1.0 - gamma) * p1[i] + gamma * p2[i]
                        val2 = gamma * p1[i] + (1.0 - gamma) * p2[i]
                        child1[i] = val1
                        child2[i] = val2

                # Gaussian Mutation
                for ch in (child1, child2):
                    if random.random() < mut_prob:
                        idx = random.randint(0, self.num_genes - 1)
                        sigma = 0.15 if idx < self.num_rules else 0.08
                        ch[idx] += random.gauss(0, sigma)
                        # Clip to bounds
                        if idx < self.num_rules:
                            ch[idx] = max(0.1, min(1.8, ch[idx]))
                        else:
                            ch[idx] = max(0.5, min(1.6, ch[idx]))

                next_generation.append(child1)
                if len(next_generation) < population_size:
                    next_generation.append(child2)

            population = next_generation
            fitnesses = [self.evaluate_fitness(ind, eval_segs)[0] for ind in population]
            gen_best_idx = int(np.argmin(fitnesses))
            if fitnesses[gen_best_idx] < best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_individual = list(population[gen_best_idx])

            convergence_curve.append(round(best_fitness, 3))

        tuned_engine = self.chromosome_to_engine(best_individual)

        return {
            "best_fitness_mae": round(best_fitness, 3),
            "convergence_curve": convergence_curve,
            "best_chromosome": [round(x, 4) for x in best_individual],
            "rule_weights": [round(x, 4) for x in best_individual[:self.num_rules]],
            "mf_multipliers": [round(x, 4) for x in best_individual[self.num_rules:]],
            "tuned_engine": tuned_engine,
        }
