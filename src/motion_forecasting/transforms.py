"""Coordinate transforms and input features for focal-agent trajectories."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_positions(positions: ArrayLike) -> NDArray[np.float64]:
    result = np.asarray(positions, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2 or len(result) == 0:
        raise ValueError("positions must have shape (N, 2) with N > 0")
    if not np.isfinite(result).all():
        raise ValueError("positions must contain only finite coordinates")
    return result


def estimate_agent_angle(positions: ArrayLike, lookback: int = 5) -> float:
    """Estimate the direction of travel from recent observed displacements."""
    points = _as_positions(positions)
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    start = max(0, len(points) - lookback)
    direction = points[-1] - points[start]
    if np.linalg.norm(direction) < 1e-6:
        recent_steps = np.diff(points, axis=0)[-lookback:]
        direction = recent_steps.sum(axis=0)
    if np.linalg.norm(direction) < 1e-6:
        return 0.0
    return float(np.arctan2(direction[1], direction[0]))


def transform_to_agent_frame(
    positions: ArrayLike,
    *,
    origin: ArrayLike | None = None,
    angle: float | None = None,
    lookback: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Translate and rotate world positions so the agent is at the origin.

    When ``origin`` or ``angle`` is omitted, it is inferred from the final
    point and recent direction of the provided positions. The returned angle
    is the world-frame heading in radians and is needed for the inverse.
    """
    points = _as_positions(positions)
    origin_array = (
        points[-1].copy()
        if origin is None
        else np.asarray(origin, dtype=np.float64)
    )
    if origin_array.shape != (2,) or not np.isfinite(origin_array).all():
        raise ValueError("origin must be a finite (2,) coordinate")
    angle_value = estimate_agent_angle(points, lookback) if angle is None else float(angle)
    if not np.isfinite(angle_value):
        raise ValueError("angle must be finite")

    delta = points - origin_array
    cosine, sine = np.cos(angle_value), np.sin(angle_value)
    rotation = np.array([[cosine, sine], [-sine, cosine]])
    local_positions = delta @ rotation.T
    return local_positions, origin_array, angle_value


def transform_to_world_frame(
    local_positions: ArrayLike,
    origin: ArrayLike,
    angle: float,
) -> NDArray[np.float64]:
    """Invert ``transform_to_agent_frame`` for positions in the local frame."""
    points = _as_positions(local_positions)
    origin_array = np.asarray(origin, dtype=np.float64)
    if origin_array.shape != (2,) or not np.isfinite(origin_array).all():
        raise ValueError("origin must be a finite (2,) coordinate")
    if not np.isfinite(angle):
        raise ValueError("angle must be finite")
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return points @ rotation.T + origin_array


def build_history_features(
    observed_positions: ArrayLike,
    representation: str = "agent-centric",
    *,
    lookback: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """Build model input features and return the frame metadata.

    ``basic`` reproduces MLP v1 (translated XY). ``agent-centric`` rotates XY
    to align heading with +x and appends per-timestep local displacement.
    """
    points = _as_positions(observed_positions)
    if representation == "agent-centric":
        local_positions, origin, angle = transform_to_agent_frame(
            points, lookback=lookback
        )
        velocity = np.zeros_like(local_positions)
        velocity[1:] = np.diff(local_positions, axis=0)
        features = np.concatenate((local_positions, velocity), axis=1)
    elif representation == "basic":
        origin = points[-1].copy()
        angle = 0.0
        features = points - origin
    else:
        raise ValueError("representation must be 'basic' or 'agent-centric'")
    return features, origin, angle
