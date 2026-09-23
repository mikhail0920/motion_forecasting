"""Evaluate the constant-velocity baseline on AV2 focal tracks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from motion_forecasting.baselines import ConstantVelocityPredictor
from motion_forecasting.data import load_scenario
from motion_forecasting.metrics import ade, fde


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/val"), help="AV2 validation directory")
    parser.add_argument("--limit", type=int, help="evaluate at most this many scenarios")
    args = parser.parse_args()

    if not args.data.is_dir():
        parser.error(f"data directory does not exist: {args.data}")
    scenario_paths = sorted(args.data.rglob("scenario_*.parquet"))
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be a positive integer")
        scenario_paths = scenario_paths[: args.limit]
    if not scenario_paths:
        parser.error(f"no scenario_*.parquet files found under {args.data}")

    predictor = ConstantVelocityPredictor()
    ade_scores: list[float] = []
    fde_scores: list[float] = []
    skipped = 0

    for path in scenario_paths:
        scenario = load_scenario(path)
        focal = next(
            (track for track in scenario["tracks"] if track["track_id"] == scenario["focal_track_id"]),
            None,
        )
        if focal is None:
            skipped += 1
            continue

        observed = focal["observed"]
        positions = np.column_stack((focal["position_x"], focal["position_y"]))
        observed_positions = positions[observed]
        observed_timesteps = focal["timestep"][observed]
        future_positions = positions[~observed]
        future_timesteps = focal["timestep"][~observed]
        if len(future_positions) == 0 or len(observed_positions) < 2:
            skipped += 1
            continue

        prediction = predictor.predict(
            observed_positions=observed_positions,
            observed_timesteps=observed_timesteps,
            future_steps=len(future_positions),
            future_timesteps=future_timesteps,
        )
        ade_scores.append(ade(prediction, future_positions))
        fde_scores.append(fde(prediction, future_positions))

    if not ade_scores:
        raise RuntimeError("No scenarios had a usable focal track with past and future states")

    print("Constant Velocity Baseline\n")
    print(f"Scenarios: {len(ade_scores)}")
    print(f"ADE: {np.mean(ade_scores):.3f} m")
    print(f"FDE: {np.mean(fde_scores):.3f} m")
    if skipped:
        print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
