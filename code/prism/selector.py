"""Pure-RMSE instance selection used by PRISM.

The response matrix has shape ``(num_instances, num_initial_algorithms)``.
For a subset J, every response row is reconstructed as an affine ridge
combination of the rows in J.  Candidate subsets are scored by leave-one-
algorithm-out (LOAO) RMSE, which prevents an algorithm column from explaining
itself during subset selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


DEFAULT_ALPHAS = (1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0)


@dataclass(frozen=True)
class SelectionResult:
    selected_indices: tuple[int, ...]
    alpha: float
    loao_rmse: float
    reconstruction_weights: np.ndarray


def _validate(matrix: np.ndarray, selected: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    response = np.asarray(matrix, dtype=float)
    rows = np.asarray(selected, dtype=int)
    if response.ndim != 2 or response.shape[1] < 3:
        raise ValueError("response_matrix must be 2-D with at least three algorithms")
    if not np.all(np.isfinite(response)):
        raise ValueError("response_matrix must be finite")
    if rows.ndim != 1 or rows.size == 0 or len(set(rows.tolist())) != rows.size:
        raise ValueError("selected indices must be non-empty and unique")
    if np.any(rows < 0) or np.any(rows >= response.shape[0]):
        raise ValueError("selected index outside response_matrix")
    return response, rows


def fit_affine_ridge(
    matrix: np.ndarray,
    selected: Iterable[int],
    alpha: float,
) -> np.ndarray:
    """Return W with ``R ~= W @ R[J]`` and every row of W summing to one."""
    response, rows = _validate(matrix, tuple(selected))
    x = response[rows].T
    y = response.T
    k = rows.size
    scale = float(np.trace(x.T @ x) / k)
    ridge = float(alpha) * scale if scale > 0.0 else float(alpha)
    system = x.T @ x + ridge * np.eye(k)
    rhs = x.T @ y
    ones = np.ones(k)
    try:
        unconstrained = np.linalg.solve(system, rhs)
        inverse_ones = np.linalg.solve(system, ones)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(system)
        unconstrained = inverse @ rhs
        inverse_ones = inverse @ ones
    denominator = float(ones @ inverse_ones)
    if abs(denominator) < 1e-14:
        raise np.linalg.LinAlgError("affine ridge system is numerically singular")
    multiplier = (ones @ unconstrained - 1.0) / denominator
    weights = (unconstrained - inverse_ones[:, None] * multiplier[None, :]).T
    weights[rows] = np.eye(k)
    return weights


def loao_rmse(matrix: np.ndarray, selected: Iterable[int], alpha: float) -> float:
    """Leave-one-algorithm-out reconstruction RMSE."""
    response = np.asarray(matrix, dtype=float)
    rows = tuple(selected)
    squared_errors = []
    for algorithm in range(response.shape[1]):
        keep = np.arange(response.shape[1]) != algorithm
        weights = fit_affine_ridge(response[:, keep], rows, alpha)
        prediction = weights @ response[np.asarray(rows), algorithm]
        squared_errors.append((response[:, algorithm] - prediction) ** 2)
    return float(np.sqrt(np.mean(np.concatenate(squared_errors))))


class PureRMSESelector:
    """Deterministic greedy minimization of LOAO affine-ridge RMSE."""

    def __init__(self, k: int = 5, alphas: Sequence[float] = DEFAULT_ALPHAS):
        if k < 1:
            raise ValueError("k must be positive")
        values = tuple(float(value) for value in alphas)
        if not values or any(value < 0.0 for value in values):
            raise ValueError("alphas must be non-empty and non-negative")
        self.k = int(k)
        self.alphas = values

    def select(self, response_matrix: np.ndarray) -> SelectionResult:
        response = np.asarray(response_matrix, dtype=float)
        if response.ndim != 2 or response.shape[0] < self.k or response.shape[1] < 3:
            raise ValueError("response_matrix must contain at least k instances and three algorithms")
        if not np.all(np.isfinite(response)):
            raise ValueError("response_matrix must be finite")
        selected: list[int] = []
        best_alpha = self.alphas[0]
        best_rmse = float("inf")
        for _ in range(self.k):
            candidates = []
            for instance in range(response.shape[0]):
                if instance in selected:
                    continue
                trial = (*selected, instance)
                rmse, alpha = min(
                    ((loao_rmse(response, trial, alpha), alpha) for alpha in self.alphas),
                    key=lambda pair: (pair[0], pair[1]),
                )
                candidates.append((rmse, instance, alpha))
            best_rmse, winner, best_alpha = min(candidates)
            selected.append(int(winner))
        weights = fit_affine_ridge(response, selected, best_alpha)
        return SelectionResult(tuple(selected), float(best_alpha), float(best_rmse), weights)
