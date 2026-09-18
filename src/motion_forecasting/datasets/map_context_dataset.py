"""Attach a precomputed map-context archive to social trajectory samples."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from motion_forecasting.datasets.social_trajectory_dataset import SocialTrajectoryDataset


class MapContextDataset(Dataset[dict[str, torch.Tensor]]):
    """Join cached lane tensors to social samples by scenario ID."""

    def __init__(self, trajectories: SocialTrajectoryDataset, cache_path: str | Path) -> None:
        path = Path(cache_path)
        if not path.is_file():
            raise FileNotFoundError(f"Map cache not found: {path}")
        with np.load(path, allow_pickle=False) as cache:
            cache_ids = cache["scenario_ids"].astype(str)
            all_lanes = cache["lanes"].astype(np.float32)
            all_masks = cache["lane_mask"].astype(np.bool_)

        if all_lanes.ndim != 4 or all_lanes.shape[-2:] != (20, 4):
            raise ValueError(f"Expected cached lanes [scenarios, lanes, 20, 4], got {all_lanes.shape}")
        if all_masks.shape != all_lanes.shape[:2]:
            raise ValueError("lane_mask dimensions do not match cached lane tensors")
        if len(cache_ids) != len(all_lanes) or len(set(cache_ids.tolist())) != len(cache_ids):
            raise ValueError("scenario_ids must be unique and align with cached lane tensors")

        index = {scenario_id: i for i, scenario_id in enumerate(cache_ids.tolist())}
        missing = [scenario_id for scenario_id in trajectories.scenario_ids if scenario_id not in index]
        if missing:
            raise ValueError(
                f"Map cache {path} is missing {len(missing)} scenarios; first missing ID: {missing[0]}"
            )
        self.trajectories = trajectories
        self.map_indices = [index[scenario_id] for scenario_id in trajectories.scenario_ids]
        self.lanes = all_lanes
        self.lane_masks = all_masks

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = dict(self.trajectories[index])
        map_index = self.map_indices[index]
        sample["lanes"] = torch.from_numpy(self.lanes[map_index])
        sample["lane_mask"] = torch.from_numpy(self.lane_masks[map_index])
        return sample
