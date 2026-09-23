"""Load one Argoverse 2 Motion Forecasting scenario from Parquet."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "track_id",
    "object_type",
    "timestep",
    "position_x",
    "position_y",
    "velocity_x",
    "velocity_y",
    "heading",
    "observed",
}


def load_scenario(path: str | Path) -> dict[str, Any]:
    """Return scenario metadata and tracks from an AV2 scenario Parquet file.

    AV2 stores a scenario as a table with one row per observed actor state.
    The scenario-level fields (including the focal track ID) are repeated in
    the table, so we read them from its first row and group states by track.
    """
    scenario_path = Path(path)
    if not scenario_path.is_file():
        raise FileNotFoundError(f"Scenario Parquet file not found: {scenario_path}")

    frame = pd.read_parquet(scenario_path)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Parquet file is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Scenario Parquet file is empty: {scenario_path}")

    first = frame.iloc[0]
    tracks: list[dict[str, Any]] = []
    for track_id, states in frame.groupby("track_id", sort=False):
        states = states.sort_values("timestep")
        tracks.append(
            {
                "track_id": str(track_id),
                "object_type": str(states["object_type"].iloc[0]),
                "timestep": states["timestep"].to_numpy(),
                "position_x": states["position_x"].to_numpy(),
                "position_y": states["position_y"].to_numpy(),
                "velocity_x": states["velocity_x"].to_numpy(),
                "velocity_y": states["velocity_y"].to_numpy(),
                "heading": states["heading"].to_numpy(),
                "observed": states["observed"].astype(bool).to_numpy(),
            }
        )

    return {
        "scenario_id": str(first["scenario_id"]),
        "focal_track_id": str(first["focal_track_id"]),
        "city": str(first["city"]) if "city" in frame.columns else "unknown",
        "tracks": tracks,
    }


def find_scenario(path: str | Path) -> Path:
    """Resolve a Parquet file or a directory containing scenario Parquet files."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        scenarios = sorted(candidate.rglob("scenario_*.parquet"))
        if scenarios:
            return scenarios[0]
    raise FileNotFoundError(
        f"No scenario_*.parquet file found at {candidate}. "
        "Pass a scenario file or a directory containing downloaded AV2 data."
    )
