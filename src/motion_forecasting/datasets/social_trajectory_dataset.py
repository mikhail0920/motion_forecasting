"""Focal-agent and nearby-actor samples for social forecasting models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from motion_forecasting.transforms import build_history_features, transform_to_agent_frame


class SocialTrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    """Build fixed-length focal history, nearby observed actors, and future.

    Neighbors are chosen from actors present at the last observed timestep,
    sorted by distance to the focal agent. Missing neighbor timesteps are zero
    padded and identified by the last ``valid`` feature.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        past_steps: int = 50,
        future_steps: int = 60,
        max_neighbors: int = 8,
        max_scenarios: int | None = None,
    ) -> None:
        if past_steps < 1 or future_steps < 1 or max_neighbors < 1:
            raise ValueError("past_steps, future_steps, and max_neighbors must be positive")
        if max_scenarios is not None and max_scenarios < 1:
            raise ValueError("max_scenarios must be positive when specified")
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.max_neighbors = max_neighbors
        self.input_dim = 4
        self.neighbor_dim = 5
        self.samples: list[dict[str, torch.Tensor]] = []
        self.scenario_ids: list[str] = []
        self.skipped_scenarios: list[str] = []

        root = Path(data_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Scenario directory not found: {root}")
        paths = sorted(root.rglob("scenario_*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No scenario_*.parquet files found under {root}")
        if max_scenarios is not None:
            if max_scenarios > len(paths):
                raise ValueError(
                    f"Requested {max_scenarios} scenarios, but only {len(paths)} files "
                    f"are available under {root}"
                )
            paths = paths[:max_scenarios]

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
            if frame.empty:
                self.skipped_scenarios.append(path.stem.removeprefix("scenario_"))
                continue
            scenario_id = str(frame["scenario_id"].iloc[0])
            focal_id = frame["focal_track_id"].iloc[0]
            focal = frame.loc[frame["track_id"] == focal_id].sort_values("timestep")
            focal_observed = focal.loc[focal["observed"].astype(bool)]
            focal_future = focal.loc[~focal["observed"].astype(bool)]
            past = focal_observed[["position_x", "position_y"]].to_numpy(dtype=np.float32)
            future = focal_future[["position_x", "position_y"]].to_numpy(dtype=np.float32)

            if len(past) != past_steps or len(future) != future_steps:
                self.skipped_scenarios.append(scenario_id)
                continue
            if not np.isfinite(past).all() or not np.isfinite(future).all():
                self.skipped_scenarios.append(scenario_id)
                continue

            history, origin, angle = build_history_features(past, "agent-centric")
            future_local, _, _ = transform_to_agent_frame(future, origin=origin, angle=angle)
            past_timesteps = focal_observed["timestep"].to_numpy()
            last_timestep = past_timesteps[-1]
            focal_last = past[-1].astype(np.float64)
            neighbors = np.zeros((max_neighbors, past_steps, self.neighbor_dim), dtype=np.float32)
            neighbor_mask = np.zeros(max_neighbors, dtype=np.bool_)

            observed_scene = frame.loc[
                frame["observed"].astype(bool) & (frame["timestep"] <= last_timestep)
            ]
            at_last = observed_scene.loc[observed_scene["timestep"] == last_timestep]
            candidates: list[tuple[float, str, object, pd.DataFrame]] = []
            for track_id, actor in observed_scene.groupby("track_id", sort=False):
                if track_id == focal_id:
                    continue
                final_state = at_last.loc[at_last["track_id"] == track_id]
                if final_state.empty:
                    continue
                actor_last = final_state[["position_x", "position_y"]].iloc[-1].to_numpy(dtype=np.float64)
                if not np.isfinite(actor_last).all():
                    continue
                distance = float(np.linalg.norm(actor_last - focal_last))
                candidates.append((distance, str(track_id), track_id, actor))

            candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
            for neighbor_index, (_, _, track_id, actor) in enumerate(candidates[:max_neighbors]):
                neighbor_mask[neighbor_index] = True
                states_by_timestep = {
                    timestep: (x, y)
                    for timestep, x, y in actor[["timestep", "position_x", "position_y"]]
                    .itertuples(index=False, name=None)
                }
                valid_indices: list[int] = []
                world_positions: list[tuple[float, float]] = []
                for history_index, timestep in enumerate(past_timesteps):
                    point = states_by_timestep.get(timestep)
                    if point is None or not np.isfinite(point).all():
                        continue
                    valid_indices.append(history_index)
                    world_positions.append(point)

                if not valid_indices:
                    # A selected neighbor must be present at last_timestep, which
                    # is in focal history, but keep the guard for malformed files.
                    neighbor_mask[neighbor_index] = False
                    continue
                local_positions, _, _ = transform_to_agent_frame(
                    np.asarray(world_positions, dtype=np.float64), origin=origin, angle=angle
                )
                row = neighbors[neighbor_index]
                row[valid_indices, :2] = local_positions.astype(np.float32)
                row[valid_indices, 4] = 1.0
                for history_index in range(1, past_steps):
                    if row[history_index - 1, 4] and row[history_index, 4]:
                        row[history_index, 2:4] = (
                            row[history_index, :2] - row[history_index - 1, :2]
                        )

            self.samples.append(
                {
                    "history": torch.from_numpy(history.astype(np.float32)),
                    "neighbors": torch.from_numpy(neighbors),
                    "neighbor_mask": torch.from_numpy(neighbor_mask),
                    "future": torch.from_numpy(future_local.astype(np.float32)),
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
