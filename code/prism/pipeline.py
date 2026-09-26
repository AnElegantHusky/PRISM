"""Minimal phase-aware PRISM integration independent of the AAD framework."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal

import numpy as np

from .evaluator import AugmentedCPIFeedback, MeanFeedback
from .selector import PureRMSESelector, SelectionResult


@dataclass
class PRISMInitializer:
    """Collect distinct full-instance responses and fit the selector once."""

    num_instances: int
    target_algorithms: int = 10

    def __post_init__(self) -> None:
        self._ids: list[Hashable] = []
        self._fitness: list[np.ndarray] = []

    @property
    def ready(self) -> bool:
        return len(self._fitness) >= self.target_algorithms

    def add(self, algorithm_id: Hashable, full_fitness: np.ndarray) -> bool:
        values = np.asarray(full_fitness, dtype=float)
        if values.shape != (self.num_instances,) or not np.all(np.isfinite(values)):
            raise ValueError(f"full_fitness must have shape ({self.num_instances},)")
        if algorithm_id in self._ids:
            return False
        if any(np.array_equal(values, previous) for previous in self._fitness):
            return False
        self._ids.append(algorithm_id)
        self._fitness.append(values.copy())
        return True

    def response_matrix(self) -> np.ndarray:
        if not self.ready:
            raise RuntimeError("initialization is not complete")
        return np.vstack(self._fitness[: self.target_algorithms]).T

    def algorithms(self) -> tuple[Hashable, ...]:
        return tuple(self._ids[: self.target_algorithms])


class PRISM:
    """Instance selection plus EoH- or EoH-S-compatible feedback adaptation."""

    def __init__(
        self,
        num_instances: int,
        *,
        k: int = 5,
        target_algorithms: int = 10,
        feedback: Literal["mean", "cpi_augmented"] = "cpi_augmented",
        directions: int = 300,
        direction_seed: int = 2026082501,
    ) -> None:
        self.initializer = PRISMInitializer(num_instances, target_algorithms)
        self.selector = PureRMSESelector(k=k)
        self.feedback_name = feedback
        self.feedback = (
            MeanFeedback()
            if feedback == "mean"
            else AugmentedCPIFeedback(k, directions=directions, seed=direction_seed)
        )
        self.selection: SelectionResult | None = None

    @property
    def selected_indices(self) -> tuple[int, ...]:
        if self.selection is None:
            raise RuntimeError("PRISM has not been activated")
        return self.selection.selected_indices

    def observe_initial(self, algorithm_id: Hashable, full_fitness: np.ndarray) -> bool:
        admitted = self.initializer.add(algorithm_id, full_fitness)
        if self.selection is None and self.initializer.ready:
            self.selection = self.selector.select(self.initializer.response_matrix())
        return admitted

    def transform_selected(self, selected_fitness: np.ndarray) -> float | np.ndarray:
        if self.selection is None:
            raise RuntimeError("PRISM has not been activated")
        return self.feedback.transform(np.asarray(selected_fitness, dtype=float))

    def project_initial_population(self) -> dict[Hashable, float | np.ndarray]:
        """Project the already measured initial algorithms into search space."""
        if self.selection is None:
            raise RuntimeError("PRISM has not been activated")
        response = self.initializer.response_matrix()
        rows = np.asarray(self.selected_indices, dtype=int)
        return {
            algorithm_id: self.transform_selected(response[rows, column])
            for column, algorithm_id in enumerate(self.initializer.algorithms())
        }
