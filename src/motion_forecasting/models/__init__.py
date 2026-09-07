"""Neural forecasting models."""

from motion_forecasting.models.mlp import TrajectoryMLP
from motion_forecasting.models.gru import TrajectoryGRU

__all__ = ["TrajectoryMLP", "TrajectoryGRU"]
