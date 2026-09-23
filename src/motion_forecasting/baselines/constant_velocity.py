"""Constant-velocity trajectory extrapolation baseline."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


class ConstantVelocityPredictor:
    """Continue motion using a short average of recent observed velocities.

    AV2 tracks are sampled at 10 Hz by default, so one timestep represents
    0.1 seconds. The trajectory coordinates are metric map coordinates.
    """

    def __init__(
        self,
        *,
        step_seconds: float = 0.1,
        velocity_window: int = 5,
    ) -> None:
        if step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        if velocity_window < 1:
            raise ValueError("velocity_window must be at least 1")
        self.step_seconds = step_seconds
        self.velocity_window = velocity_window

    def predict(
        self,
        observed_positions: ArrayLike,
        observed_timesteps: ArrayLike,
        future_steps: int,
        *,
        future_timesteps: ArrayLike | None = None,
    ) -> NDArray[np.float64]:
        """Predict future ``(x, y)`` positions from observed coordinates.

        The average velocity is computed from up to ``velocity_window`` of the
        most recent displacement estimates. ``observed_timesteps`` and optional
        ``future_timesteps`` are AV2 frame indices, not seconds.
        """
        positions = np.asarray(observed_positions, dtype=float)
        timesteps = np.asarray(observed_timesteps, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 2:
            raise ValueError("observed_positions must have shape (N, 2)")
        if len(positions) < 2:
            raise ValueError("At least two observed positions are needed to estimate velocity")
        if timesteps.ndim != 1 or len(timesteps) != len(positions):
            raise ValueError("observed_timesteps must be a 1D array matching observed_positions")
        if not np.isfinite(positions).all() or not np.isfinite(timesteps).all():
            raise ValueError("Observed positions and timesteps must be finite")
        if np.any(np.diff(timesteps) <= 0):
            raise ValueError("observed_timesteps must be strictly increasing")
        if future_steps < 0:
            raise ValueError("future_steps cannot be negative")
        if future_timesteps is not None and future_steps != len(future_timesteps):
            raise ValueError("future_steps must match the length of future_timesteps")
        if future_steps == 0:
            return np.empty((0, 2), dtype=float)

        # Convert frame gaps to seconds before calculating each local velocity.
        frame_intervals_seconds = np.diff(timesteps) * self.step_seconds
        estimates = np.diff(positions, axis=0) / frame_intervals_seconds[:, None]
        velocity = estimates[-self.velocity_window :].mean(axis=0)

        last_timestep = timesteps[-1]
        if future_timesteps is None:
            target_timesteps = last_timestep + np.arange(1, future_steps + 1)
        else:
            target_timesteps = np.asarray(future_timesteps, dtype=float)
            if target_timesteps.ndim != 1 or not np.isfinite(target_timesteps).all():
                raise ValueError("future_timesteps must be a finite 1D array")
            if np.any(target_timesteps <= last_timestep):
                raise ValueError("All future_timesteps must follow the last observed timestep")

        elapsed_seconds = (target_timesteps - last_timestep) * self.step_seconds
        return positions[-1] + elapsed_seconds[:, None] * velocity
