"""Plot representative straight, turning, braking, and sharp AV2 scenes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from motion_forecasting.baselines import ConstantVelocityPredictor
from motion_forecasting.data import load_scenario
from motion_forecasting.metrics import ade, fde
from motion_forecasting.models import TrajectoryGRU, TrajectoryMLP
from motion_forecasting.transforms import build_history_features, transform_to_world_frame


def _case_measurements(path: Path) -> tuple[str, float, float, float]:
    columns = [
        "track_id", "focal_track_id", "timestep", "position_x", "position_y",
        "velocity_x", "velocity_y", "observed", "scenario_id",
    ]
    frame = pd.read_parquet(path, columns=columns)
    focal_id = frame["focal_track_id"].iloc[0]
    focal = frame.loc[frame["track_id"] == focal_id].sort_values("timestep")
    observed = focal["observed"].astype(bool).to_numpy()
    positions = focal[["position_x", "position_y"]].to_numpy(dtype=float)
    velocities = focal[["velocity_x", "velocity_y"]].to_numpy(dtype=float)
    future = positions[~observed]
    future_velocity = velocities[~observed]
    if int(observed.sum()) < 5 or len(future) < 20:
        return str(frame["scenario_id"].iloc[0]), 180.0, 0.0, 0.0

    first_direction = future[9] - future[0]
    last_direction = future[-1] - future[-10]
    cross = first_direction[0] * last_direction[1] - first_direction[1] * last_direction[0]
    dot = float(np.dot(first_direction, last_direction))
    turn_degrees = abs(float(np.degrees(np.arctan2(cross, dot))))
    path_length = np.linalg.norm(np.diff(future, axis=0), axis=1).sum()
    straightness = float(np.linalg.norm(future[-1] - future[0]) / max(path_length, 1e-6))
    speeds = np.linalg.norm(future_velocity, axis=1)
    deceleration = float(np.mean(speeds[:10]) - np.mean(speeds[-10:]))
    return str(frame["scenario_id"].iloc[0]), turn_degrees, straightness, deceleration


def _select_cases(data_dir: Path) -> dict[str, Path]:
    candidates: dict[str, list[tuple[Path, float, float, float]]] = {
        "Straight": [], "Turn": [], "Braking": [], "Sharp maneuver": []
    }
    for path in sorted(data_dir.rglob("scenario_*.parquet")):
        _, turn, straightness, deceleration = _case_measurements(path)
        if turn < 12 and straightness > 0.95 and abs(deceleration) < 2:
            candidates["Straight"].append((path, turn, straightness, deceleration))
        if 20 <= turn <= 55 and straightness > 0.6:
            candidates["Turn"].append((path, turn, straightness, deceleration))
        if deceleration > 3 and turn < 25:
            candidates["Braking"].append((path, turn, straightness, deceleration))
        if turn > 60 or straightness < 0.75:
            candidates["Sharp maneuver"].append((path, turn, straightness, deceleration))

    selected: dict[str, Path] = {}
    used: set[Path] = set()
    for label, examples in candidates.items():
        for path, _, _, _ in examples:
            if path not in used:
                selected[label] = path
                used.add(path)
                break
    missing = set(candidates) - set(selected)
    if missing:
        raise RuntimeError(f"No representative validation scenes found for: {', '.join(sorted(missing))}")
    return selected


def _load_model(checkpoint_path: Path, model_type: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if model_type == "mlp":
        model = TrajectoryMLP(
            past_steps=int(checkpoint["past_steps"]),
            future_steps=int(checkpoint["future_steps"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            input_dim=int(checkpoint.get("input_dim", 2)),
        )
    else:
        model = TrajectoryGRU(
            input_dim=int(checkpoint.get("input_dim", 4)),
            hidden_dim=int(checkpoint["hidden_dim"]),
            future_steps=int(checkpoint["future_steps"]),
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _predict(model, checkpoint: dict, past: np.ndarray) -> np.ndarray:
    history, origin, angle = build_history_features(
        past,
        representation=checkpoint.get("representation", "basic"),
    )
    with torch.inference_mode():
        prediction = model(torch.from_numpy(history.astype(np.float32)).unsqueeze(0))
    return transform_to_world_frame(prediction.squeeze(0).numpy(), origin, angle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/val"))
    parser.add_argument("--mlp-checkpoint", type=Path, required=True)
    parser.add_argument("--gru-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/model_cases.png"))
    args = parser.parse_args()
    mlp, mlp_checkpoint = _load_model(args.mlp_checkpoint, "mlp")
    gru, gru_checkpoint = _load_model(args.gru_checkpoint, "gru")
    selected = _select_cases(args.data)

    figure, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (label, path) in zip(axes.flat, selected.items()):
        scenario = load_scenario(path)
        focal = next(track for track in scenario["tracks"] if track["track_id"] == scenario["focal_track_id"])
        observed = focal["observed"]
        positions = np.column_stack((focal["position_x"], focal["position_y"]))
        past, future = positions[observed], positions[~observed]
        future_timesteps = focal["timestep"][~observed]

        cv_prediction = ConstantVelocityPredictor().predict(
            observed_positions=past,
            observed_timesteps=focal["timestep"][observed],
            future_steps=len(future),
            future_timesteps=future_timesteps,
        )
        mlp_prediction = _predict(mlp, mlp_checkpoint, past)
        gru_prediction = _predict(gru, gru_checkpoint, past)

        for track in scenario["tracks"]:
            if track["track_id"] != focal["track_id"]:
                ax.plot(track["position_x"], track["position_y"], color="0.78", linewidth=0.6, alpha=0.55)
        ax.plot(past[:, 0], past[:, 1], color="tab:blue", linewidth=2, label="Past")
        ax.plot(future[:, 0], future[:, 1], color="black", linewidth=2, linestyle="--", label="Ground truth")
        ax.plot(cv_prediction[:, 0], cv_prediction[:, 1], color="tab:green", linewidth=1.7, linestyle=":", label="Constant Velocity")
        ax.plot(mlp_prediction[:, 0], mlp_prediction[:, 1], color="tab:purple", linewidth=1.7, linestyle="-.", label=f"MLP ADE {ade(mlp_prediction, future):.1f}")
        ax.plot(gru_prediction[:, 0], gru_prediction[:, 1], color="tab:red", linewidth=1.7, linestyle="-.", label=f"GRU ADE {ade(gru_prediction, future):.1f}")
        ax.scatter(past[-1, 0], past[-1, 1], color="black", s=22, zorder=4)
        ax.set_title(f"{label} · {scenario['scenario_id'][:8]}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, linewidth=0.4, alpha=0.35)
        ax.legend(fontsize=8, loc="best")

    figure.suptitle("Validation examples (categories selected using future ground truth)")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"Saved comparison cases to {args.output.resolve()}")
    for label, path in selected.items():
        print(f"{label}: {path.parent.name}")


if __name__ == "__main__":
    main()
