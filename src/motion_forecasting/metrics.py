"""Displacement metrics for single-trajectory forecasting."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _validate_trajectories(pred: ArrayLike, target: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(pred, dtype=float)
    ground_truth = np.asarray(target, dtype=float)
    if prediction.ndim != 2 or prediction.shape[1] != 2:
        raise ValueError("pred must have shape (T, 2)")
    if ground_truth.shape != prediction.shape:
        raise ValueError("target must have the same (T, 2) shape as pred")
    if len(prediction) == 0:
        raise ValueError("Trajectories must contain at least one future timestep")
    if not np.isfinite(prediction).all() or not np.isfinite(ground_truth).all():
        raise ValueError("Trajectories must contain only finite coordinates")
    return prediction, ground_truth


def ade(pred: ArrayLike, target: ArrayLike) -> float:
    """Average Euclidean displacement error across all future timesteps."""
    prediction, ground_truth = _validate_trajectories(pred, target)
    distances = np.linalg.norm(prediction - ground_truth, axis=1)
    return float(distances.mean())


def fde(pred: ArrayLike, target: ArrayLike) -> float:
    """Euclidean displacement error at the final future timestep."""
    prediction, ground_truth = _validate_trajectories(pred, target)
    return float(np.linalg.norm(prediction[-1] - ground_truth[-1]))
