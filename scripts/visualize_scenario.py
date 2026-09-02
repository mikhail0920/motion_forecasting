"""Plot past and future actor trajectories for an AV2 scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from motion_forecasting.baselines import ConstantVelocityPredictor
from motion_forecasting.data import find_scenario, load_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        nargs="?",
        default="data/val",
        help="scenario Parquet file or directory (default: data/val)",
    )
    parser.add_argument("--output", type=Path, help="image output path (default: outputs/<scenario-id>.png)")
    args = parser.parse_args()

    scenario_path = find_scenario(args.scenario)
    scenario = load_scenario(scenario_path)
    focal_id = scenario["focal_track_id"]
    focal_track = next(
        (track for track in scenario["tracks"] if track["track_id"] == focal_id),
        None,
    )
    if focal_track is None:
        raise ValueError(f"Focal track {focal_id} is missing from the scenario")
    focal_observed = focal_track["observed"]
    focal_positions = np.column_stack((focal_track["position_x"], focal_track["position_y"]))
    past_positions = focal_positions[focal_observed]
    future_positions = focal_positions[~focal_observed]
    prediction = ConstantVelocityPredictor().predict(
        observed_positions=past_positions,
        observed_timesteps=focal_track["timestep"][focal_observed],
        future_steps=len(future_positions),
        future_timesteps=focal_track["timestep"][~focal_observed],
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    for track in scenario["tracks"]:
        past = track["observed"]
        x, y = track["position_x"], track["position_y"]
        focal = track["track_id"] == focal_id
        if focal:
            ax.plot(x[past], y[past], color="tab:blue", linewidth=2.8, label="Focal past")
            ax.plot(x[~past], y[~past], color="tab:orange", linewidth=2.8, linestyle="--", label="Focal future")
            ax.plot(prediction[:, 0], prediction[:, 1], color="tab:green", linewidth=2.4, linestyle=":", label="Constant velocity")
            ax.scatter(x[past][-1], y[past][-1], color="black", s=35, zorder=5, label="Current position")
        else:
            ax.plot(x[past], y[past], color="0.65", linewidth=0.8, alpha=0.75)
            if (~past).any():
                ax.plot(x[~past], y[~past], color="0.75", linewidth=0.7, linestyle=":", alpha=0.65)

    ax.set_title(f"AV2 scenario {scenario['scenario_id']} · {scenario['city']}")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.legend(loc="best")
    fig.tight_layout()

    output = args.output or Path("outputs") / f"{scenario['scenario_id']}.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"Saved visualization to {output.resolve()}")


if __name__ == "__main__":
    main()
