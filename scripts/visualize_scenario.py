"""Plot past and future actor trajectories for an AV2 scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from motion_forecasting.baselines import ConstantVelocityPredictor
from motion_forecasting.data import find_scenario, load_scenario
from motion_forecasting.transforms import build_history_features, transform_to_agent_frame, transform_to_world_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        nargs="?",
        default="data/val",
        help="scenario Parquet file or directory (default: data/val)",
    )
    parser.add_argument("--output", type=Path, help="image output path (default: outputs/<scenario-id>.png)")
    parser.add_argument("--checkpoint", type=Path, help="optional trained MLP checkpoint to draw")
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
    mlp_prediction = None
    if args.checkpoint is not None:
        import torch

        from motion_forecasting.models import TrajectoryMLP

        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        mlp = TrajectoryMLP(
            past_steps=int(checkpoint["past_steps"]),
            future_steps=int(checkpoint["future_steps"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            input_dim=int(checkpoint.get("input_dim", 2)),
        )
        mlp.load_state_dict(checkpoint["model_state_dict"])
        mlp.eval()
        representation = checkpoint.get("representation", "basic")
        history_features, origin, angle = build_history_features(
            past_positions,
            representation=representation,
        )
        with torch.inference_mode():
            local_prediction = mlp(
                torch.from_numpy(history_features.astype("float32")).unsqueeze(0)
            )
        local_prediction = local_prediction.squeeze(0).numpy()
        mlp_prediction = transform_to_world_frame(local_prediction, origin, angle)

    fig, ax = plt.subplots(figsize=(10, 10))
    for track in scenario["tracks"]:
        past = track["observed"]
        x, y = track["position_x"], track["position_y"]
        focal = track["track_id"] == focal_id
        if focal:
            ax.plot(x[past], y[past], color="tab:blue", linewidth=2.8, label="Focal past")
            ax.plot(x[~past], y[~past], color="tab:orange", linewidth=2.8, linestyle="--", label="Focal future")
            ax.plot(prediction[:, 0], prediction[:, 1], color="tab:green", linewidth=2.4, linestyle=":", label="Constant velocity")
            if mlp_prediction is not None:
                ax.plot(mlp_prediction[:, 0], mlp_prediction[:, 1], color="tab:purple", linewidth=2.2, linestyle="-.", label="MLP")
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
