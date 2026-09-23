"""PyTorch datasets for trajectory forecasting."""

from motion_forecasting.datasets.trajectory_dataset import TrajectoryDataset
from motion_forecasting.datasets.social_trajectory_dataset import SocialTrajectoryDataset

__all__ = ["TrajectoryDataset", "SocialTrajectoryDataset"]
