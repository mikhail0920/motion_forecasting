"""Build fixed-length focal-agent history and future pairs from AV2 Parquet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    """Return relative focal-agent history and future tensors.

    The default lengths match AV2's 5 s observed and 6 s prediction windows at
    10 Hz. Scenarios without exactly these numbers of focal states are skipped
    and listed in ``skipped_scenarios`` instead of silently padding labels.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        past_steps: int = 50,
        future_steps: int = 60,
    ) -> None:
        if past_steps < 1 or future_steps < 1:
            raise ValueError("past_steps and future_steps must be positive")
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.samples: list[dict[str, torch.Tensor]] = []
        self.scenario_ids: list[str] = []
        self.skipped_scenarios: list[str] = []

        root = Path(data_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Scenario directory not found: {root}")
        paths = sorted(root.rglob("scenario_*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No scenario_*.parquet files found under {root}")

        columns = [
            "track_id",
            "focal_track_id",
            "timestep",
            "position_x",
            "position_y",
            "observed",
            "scenario_id",
        ]
        for path in paths:
            frame = pd.read_parquet(path, columns=columns)
            scenario_id = str(frame["scenario_id"].iloc[0])
            focal_id = frame["focal_track_id"].iloc[0]
            focal = frame.loc[frame["track_id"] == focal_id].sort_values("timestep")
            observed = focal["observed"].astype(bool).to_numpy()
            past = focal.loc[observed, ["position_x", "position_y"]].to_numpy(dtype=np.float32)
            future = focal.loc[~observed, ["position_x", "position_y"]].to_numpy(dtype=np.float32)

            if len(past) != past_steps or len(future) != future_steps:
                self.skipped_scenarios.append(scenario_id)
                continue
            if not np.isfinite(past).all() or not np.isfinite(future).all():
                self.skipped_scenarios.append(scenario_id)
                continue

            # Translation normalization makes the last observed point the origin.
            origin = past[-1].copy()
            self.samples.append(
                {
                    "history": torch.from_numpy(past - origin),
                    "future": torch.from_numpy(future - origin),
                }
            )
            self.scenario_ids.append(scenario_id)

        if not self.samples:
            raise ValueError(
                f"No usable scenarios under {root}; expected {past_steps} past and "
                f"{future_steps} future focal states per scenario"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]
