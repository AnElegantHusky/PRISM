"""PRISM evaluators used after instance selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeanFeedback:
    """EoH strategy: return the mean fitness over selected instances."""

    def transform(self, selected_fitness: np.ndarray) -> float:
        values = np.asarray(selected_fitness, dtype=float)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("selected_fitness must be a finite non-empty vector")
        return float(np.mean(values))


class AugmentedCPIFeedback:
    """EoH-S strategy: k identity objectives plus simplex directions.

    Fitness is maximized. The first ``k`` transformed objectives are exactly
    the selected instances. The remaining objectives are reproducible
    Dirichlet(1) convex combinations, for 300 objectives by default.
    """

    def __init__(self, k: int, directions: int = 300, seed: int = 2026082501):
        if k < 2 or directions < k:
            raise ValueError("directions must be at least k and k must be >= 2")
        rng = np.random.default_rng(int(seed))
        interior = rng.dirichlet(np.ones(k), size=directions - k)
        self.weights = np.vstack((np.eye(k), interior))
        self.k = int(k)
        self.directions = int(directions)
        self.seed = int(seed)

    def transform(self, selected_fitness: np.ndarray) -> np.ndarray:
        values = np.asarray(selected_fitness, dtype=float)
        if values.shape != (self.k,) or not np.all(np.isfinite(values)):
            raise ValueError(f"selected_fitness must have shape ({self.k},)")
        return self.weights @ values

    @staticmethod
    def population_cpi(population_fitness: np.ndarray) -> float:
        """Mean best directional fitness; larger is better."""
        population = np.asarray(population_fitness, dtype=float)
        if population.ndim != 2 or population.shape[0] == 0 or not np.all(np.isfinite(population)):
            raise ValueError("population_fitness must be a finite population-by-direction matrix")
        return float(np.mean(np.max(population, axis=0)))
