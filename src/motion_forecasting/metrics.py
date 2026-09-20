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


def multimodal_metrics(
    trajectories: ArrayLike,
    logits: ArrayLike,
    target: ArrayLike,
    *,
    miss_threshold: float = 2.0,
) -> dict[str, float]:
    """Calculate top-1 and oracle K-mode trajectory metrics for a batch.

    Shapes are ``[B, K, T, 2]``, ``[B, K]``, and ``[B, T, 2]``.
    minADE@K and minFDE@K select their best mode independently per sample.
    MissRate@K is the fraction whose best final displacement exceeds threshold.
    """
    prediction = np.asarray(trajectories, dtype=float)
    scores = np.asarray(logits, dtype=float)
    ground_truth = np.asarray(target, dtype=float)
    if prediction.ndim != 4 or prediction.shape[-1] != 2:
        raise ValueError("trajectories must have shape (B, K, T, 2)")
    if scores.shape != prediction.shape[:2]:
        raise ValueError("logits must have shape (B, K)")
    if ground_truth.shape != (prediction.shape[0], prediction.shape[2], 2):
        raise ValueError("target must have shape (B, T, 2) matching trajectories")
    if prediction.shape[0] < 1 or prediction.shape[1] < 1 or prediction.shape[2] < 1:
        raise ValueError("batch, modes, and future timesteps must be non-empty")
    if miss_threshold < 0:
        raise ValueError("miss_threshold must be non-negative")
    if not (
        np.isfinite(prediction).all()
        and np.isfinite(scores).all()
        and np.isfinite(ground_truth).all()
    ):
        raise ValueError("trajectories, logits, and targets must be finite")

    errors = np.linalg.norm(prediction - ground_truth[:, None, :, :], axis=-1)
    ade_per_mode = errors.mean(axis=-1)
    fde_per_mode = errors[:, :, -1]
    top1 = scores.argmax(axis=1)
    batch_indices = np.arange(prediction.shape[0])
    top1_ade = ade_per_mode[batch_indices, top1]
    top1_fde = fde_per_mode[batch_indices, top1]
    min_ade = ade_per_mode.min(axis=1)
    min_fde = fde_per_mode.min(axis=1)
    return {
        "top1_ade": float(top1_ade.mean()),
        "top1_fde": float(top1_fde.mean()),
        "min_ade": float(min_ade.mean()),
        "min_fde": float(min_fde.mean()),
        "miss_rate": float(np.mean(min_fde > miss_threshold)),
    }
