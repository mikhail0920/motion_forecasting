"""Scenario preprocessing, inference, and animated bird's-eye visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from motion_forecasting.baselines import ConstantVelocityPredictor
from motion_forecasting.data import load_scenario
from motion_forecasting.metrics import ade, fde
from motion_forecasting.models import (
    LaneConditionedForecaster,
    MapAwareInteractionGRU,
    TrajectoryGRU,
    TrajectoryAwareModeReranker,
)
from motion_forecasting.transforms import (
    build_history_features,
    transform_to_world_frame,
)


def load_map_cache(path: str | Path) -> dict[str, Any]:
    """Load map tensors and index them by AV2 scenario ID."""
    with np.load(Path(path), allow_pickle=False) as archive:
        scenario_ids = archive["scenario_ids"].astype(str)
        lanes = archive["lanes"].astype(np.float32)
        masks = archive["lane_mask"].astype(bool)
    return {
        "indices": {scenario_id: index for index, scenario_id in enumerate(scenario_ids)},
        "lanes": lanes,
        "masks": masks,
    }


def _track_positions(track: dict[str, Any]) -> np.ndarray:
    return np.column_stack((track["position_x"], track["position_y"])).astype(np.float64)


def _prepare_social_inputs(
    scenario: dict[str, Any],
    map_cache: dict[str, Any],
    *,
    past_steps: int,
    max_neighbors: int,
) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray, list[str | None], np.ndarray, np.ndarray, float]:
    focal = next(
        (track for track in scenario["tracks"] if track["track_id"] == scenario["focal_track_id"]),
        None,
    )
    if focal is None:
        raise ValueError("The focal actor is missing from this scenario")
    observed = np.asarray(focal["observed"], dtype=bool)
    future_mask = ~observed
    positions = _track_positions(focal)
    past_positions = positions[observed]
    future_positions = positions[future_mask]
    past_timesteps = np.asarray(focal["timestep"])[observed]
    future_timesteps = np.asarray(focal["timestep"])[future_mask]
    if len(past_positions) != past_steps:
        raise ValueError(f"Expected {past_steps} observed focal states, found {len(past_positions)}")

    history, origin, angle = build_history_features(past_positions, "agent-centric")
    neighbor_tensor = np.zeros((max_neighbors, past_steps, 5), dtype=np.float32)
    neighbor_mask = np.zeros(max_neighbors, dtype=bool)
    candidates: list[tuple[float, str, dict[str, Any]]] = []
    last_timestep = past_timesteps[-1]
    focal_now = past_positions[-1]
    for track in scenario["tracks"]:
        if track["track_id"] == scenario["focal_track_id"]:
            continue
        actor_observed = np.asarray(track["observed"], dtype=bool)
        actor_timesteps = np.asarray(track["timestep"])[actor_observed]
        actor_positions = _track_positions(track)[actor_observed]
        at_current = np.flatnonzero(actor_timesteps == last_timestep)
        if not len(at_current):
            continue
        actor_now = actor_positions[at_current[-1]]
        candidates.append((float(np.linalg.norm(actor_now - focal_now)), track["track_id"], track))
    candidates.sort(key=lambda item: (item[0], item[1]))

    neighbor_ids: list[str | None] = [None] * max_neighbors
    aligned_times = np.asarray(past_timesteps)
    for index, (_, track_id, track) in enumerate(candidates[:max_neighbors]):
        neighbor_ids[index] = track_id
        neighbor_mask[index] = True
        actor_observed = np.asarray(track["observed"], dtype=bool)
        actor_timesteps = np.asarray(track["timestep"])[actor_observed]
        actor_positions = _track_positions(track)[actor_observed]
        timestep_index = {int(t): i for i, t in enumerate(actor_timesteps)}
        valid_indices: list[int] = []
        world_positions: list[np.ndarray] = []
        for history_index, timestep in enumerate(aligned_times):
            actor_index = timestep_index.get(int(timestep))
            if actor_index is not None and np.isfinite(actor_positions[actor_index]).all():
                valid_indices.append(history_index)
                world_positions.append(actor_positions[actor_index])
        if not valid_indices:
            neighbor_mask[index] = False
            neighbor_ids[index] = None
            continue
        local_positions = (np.asarray(world_positions) - origin) @ np.array(
            [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
        ).T
        neighbor_tensor[index, valid_indices, :2] = local_positions.astype(np.float32)
        neighbor_tensor[index, valid_indices, 4] = 1.0
        for history_index in range(1, past_steps):
            if neighbor_tensor[index, history_index - 1, 4] and neighbor_tensor[index, history_index, 4]:
                neighbor_tensor[index, history_index, 2:4] = (
                    neighbor_tensor[index, history_index, :2]
                    - neighbor_tensor[index, history_index - 1, :2]
                )

    cache_index = map_cache["indices"].get(scenario["scenario_id"])
    if cache_index is None:
        raise KeyError(f"Scenario {scenario['scenario_id']} is not present in the selected map cache")
    lanes_local = map_cache["lanes"][cache_index]
    lane_mask = map_cache["masks"][cache_index]
    lanes_world = np.zeros_like(lanes_local)
    for lane_index in np.flatnonzero(lane_mask):
        lanes_world[lane_index, :, :2] = transform_to_world_frame(
            lanes_local[lane_index, :, :2], origin, angle
        ).astype(np.float32)

    inputs = {
        "history": torch.from_numpy(history.astype(np.float32)).unsqueeze(0),
        "neighbors": torch.from_numpy(neighbor_tensor).unsqueeze(0),
        "neighbor_mask": torch.from_numpy(neighbor_mask).unsqueeze(0),
        "lanes": torch.from_numpy(lanes_local.astype(np.float32)).unsqueeze(0),
        "lane_mask": torch.from_numpy(lane_mask).unsqueeze(0),
    }
    return inputs, past_positions, future_positions, neighbor_ids, lanes_world, past_timesteps, angle


def load_predictor(model_name: str, checkpoint_dir: str | Path) -> dict[str, Any]:
    """Load one inference-only model and its checkpoint metadata."""
    root = Path(checkpoint_dir)
    if model_name == "GRU":
        path = root / "gru_20k_cpu.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = TrajectoryGRU(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            future_steps=int(checkpoint["future_steps"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return {"model": model, "checkpoint": checkpoint, "path": path}
    if model_name == "Map-aware":
        path = root / "map_aware_interaction_gru_20k.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = MapAwareInteractionGRU(
            input_dim=int(checkpoint["input_dim"]),
            neighbor_dim=int(checkpoint["neighbor_dim"]),
            lane_dim=int(checkpoint["lane_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            future_steps=int(checkpoint["future_steps"]),
            max_neighbors=int(checkpoint["max_neighbors"]),
            max_lanes=int(checkpoint["max_lanes"]),
            points_per_lane=int(checkpoint["points_per_lane"]),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return {"model": model, "checkpoint": checkpoint, "path": path}
    if model_name == "Final multimodal":
        path = root / "lane_conditioned_20k.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = LaneConditionedForecaster(**checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        reranker_path = root / "trajectory_reranker_lane_conditioned_20k.pt"
        reranker_checkpoint = torch.load(
            reranker_path, map_location="cpu", weights_only=False
        )
        reranker = TrajectoryAwareModeReranker(
            scene_dim=int(reranker_checkpoint["scene_dim"]),
            future_steps=int(reranker_checkpoint["future_steps"]),
            hidden_dim=int(reranker_checkpoint["hidden_dim"]),
        )
        reranker.load_state_dict(reranker_checkpoint["model_state_dict"])
        reranker.eval()
        return {
            "model": model,
            "checkpoint": checkpoint,
            "path": path,
            "reranker": reranker,
            "reranker_checkpoint": reranker_checkpoint,
        }
    raise ValueError(f"Unknown model: {model_name}")


def predict_scenario(
    scenario: dict[str, Any],
    model_name: str,
    predictor: dict[str, Any] | None,
    map_cache: dict[str, Any] | None,
) -> dict[str, Any]:
    """Run selected predictor and return trajectories, scores, and diagnostics."""
    focal = next(track for track in scenario["tracks"] if track["track_id"] == scenario["focal_track_id"])
    observed = np.asarray(focal["observed"], dtype=bool)
    positions = _track_positions(focal)
    past = positions[observed]
    future = positions[~observed]
    past_steps = len(past)
    past_timesteps = np.asarray(focal["timestep"])[observed]
    future_timesteps = np.asarray(focal["timestep"])[~observed]
    future = future[np.argsort(future_timesteps)]
    future_timesteps = np.sort(future_timesteps)

    if model_name == "Constant Velocity":
        predicted = ConstantVelocityPredictor().predict(
            past, past_timesteps, len(future), future_timesteps=future_timesteps
        )[None, :, :]
        return {
            "trajectories": predicted,
            "scores": np.ones(1, dtype=float),
            "top_index": 0,
            "future": future,
            "past": past,
            "neighbor_ids": [],
            "neighbor_weights": {},
            "lanes_world": np.empty((0, 20, 2), dtype=float),
            "angle": 0.0,
        }

    if predictor is None:
        raise ValueError(f"{model_name} checkpoint has not been loaded")
    if model_name == "GRU":
        representation = predictor["checkpoint"].get("representation", "agent-centric")
        history, origin, angle = build_history_features(past, representation)
        tensor = torch.from_numpy(history.astype(np.float32)).unsqueeze(0)
        with torch.inference_mode():
            local_prediction = predictor["model"](tensor)[0].numpy()
        predictions = transform_to_world_frame(local_prediction, origin, angle)[None, :, :]
        return {
            "trajectories": predictions,
            "scores": np.ones(1, dtype=float),
            "top_index": 0,
            "future": future,
            "past": past,
            "neighbor_ids": [],
            "neighbor_weights": {},
            "lanes_world": np.empty((0, 20, 2), dtype=float),
            "angle": angle,
        }

    if map_cache is None:
        raise ValueError("Map-aware models require a matching precomputed map cache")
    config = predictor["checkpoint"].get(
        "model_config", predictor["checkpoint"]
    )
    inputs, past, future, neighbor_ids, lanes_world, _, angle = _prepare_social_inputs(
        scenario,
        map_cache,
        past_steps=int(config.get("past_steps", 50)),
        max_neighbors=int(config["max_neighbors"]),
    )
    with torch.inference_mode():
        if model_name == "Map-aware":
            local_prediction, neighbor_weights, _ = predictor["model"](
                inputs["history"],
                inputs["neighbors"],
                inputs["neighbor_mask"],
                inputs["lanes"],
                inputs["lane_mask"],
                return_attentions=True,
            )
            world_prediction = transform_to_world_frame(
                local_prediction[0].numpy(), past[-1], angle
            )[None, :, :]
            scores = np.ones(1, dtype=float)
            top_index = 0
        else:
            candidates, _, scene_embedding = predictor["model"].generate_with_context(
                inputs["history"],
                inputs["neighbors"],
                inputs["neighbor_mask"],
                inputs["lanes"],
                inputs["lane_mask"],
            )
            scores_tensor = predictor["reranker"](scene_embedding, candidates)
            scores = torch.softmax(scores_tensor, dim=1)[0].numpy()
            top_index = int(scores.argmax())
            world_prediction = np.stack(
                [transform_to_world_frame(path.numpy(), past[-1], angle) for path in candidates[0]]
            )
            _, neighbor_weights, _ = predictor["model"].encode_scene(
                inputs["history"],
                inputs["neighbors"],
                inputs["neighbor_mask"],
                inputs["lanes"],
                inputs["lane_mask"],
            )
    weights = neighbor_weights[0].numpy()
    neighbor_weight_map = {
        neighbor_id: float(weights[index])
        for index, neighbor_id in enumerate(neighbor_ids)
        if neighbor_id is not None
    }
    return {
        "trajectories": world_prediction,
        "scores": scores,
        "top_index": top_index,
        "future": future,
        "past": past,
        "neighbor_ids": neighbor_ids,
        "neighbor_weights": neighbor_weight_map,
        "lanes_world": lanes_world[:, :, :2],
        "angle": angle,
    }


def prediction_metrics(result: dict[str, Any]) -> dict[str, float | bool]:
    """Compute reranked top-1 and oracle metrics after the user reveals GT."""
    target = result["future"]
    trajectories = result["trajectories"]
    top_prediction = trajectories[result["top_index"]]
    top_ade = ade(top_prediction, target)
    top_fde = fde(top_prediction, target)
    per_mode_ade = [ade(path, target) for path in trajectories]
    per_mode_fde = [fde(path, target) for path in trajectories]
    oracle_fde = min(per_mode_fde)
    return {
        "top1_ade": top_ade,
        "top1_fde": top_fde,
        "oracle_min_ade": min(per_mode_ade),
        "oracle_min_fde": oracle_fde,
        "miss": oracle_fde > 2.0,
    }


def animated_figure(
    scenario: dict[str, Any],
    result: dict[str, Any],
    *,
    reveal_ground_truth: bool,
    start_at_cutoff: bool = True,
) -> go.Figure:
    """Build Plotly frames that stop actors at the observation cutoff."""
    focal_id = scenario["focal_track_id"]
    focal = next(track for track in scenario["tracks"] if track["track_id"] == focal_id)
    obs_mask = np.asarray(focal["observed"], dtype=bool)
    obs_steps = np.asarray(focal["timestep"])[obs_mask].astype(int)
    start_timestep = int(np.asarray(focal["timestep"]).min())
    cutoff_timestep = int(obs_steps.max())
    all_timesteps = np.concatenate(
        [np.asarray(track["timestep"], dtype=int) for track in scenario["tracks"]]
    )
    last_timestep = int(all_timesteps.max())
    step_values = np.arange(start_timestep, last_timestep + 1)
    seconds = (step_values - start_timestep) * 0.1
    first_future_timestep = cutoff_timestep + 1
    cutoff_elapsed = (first_future_timestep - start_timestep) * 0.1
    cutoff_index = int(np.where(step_values == first_future_timestep)[0][0])

    figure = go.Figure()
    for lane_index, lane in enumerate(result["lanes_world"]):
        if not np.isfinite(lane).all() or not np.any(lane):
            continue
        figure.add_trace(
            go.Scatter(
                x=lane[:, 0], y=lane[:, 1], mode="lines",
                line={"color": "rgba(100,116,139,0.38)", "width": 2},
                name="HD map lane", legendgroup="map", showlegend=(lane_index == 0),
                hoverinfo="skip",
            )
        )

    dynamic_indices: list[int] = []
    actor_tracks: list[dict[str, Any]] = []
    neighbor_weights = result["neighbor_weights"]
    for track in scenario["tracks"]:
        if track["track_id"] == focal_id:
            continue
        actor_observed = np.asarray(track["observed"], dtype=bool)
        actor_steps = np.asarray(track["timestep"], dtype=int)[actor_observed]
        actor_xy = _track_positions(track)[actor_observed]
        weight = neighbor_weights.get(track["track_id"])
        if weight is not None:
            line_color = f"rgba(234, 88, 12, {0.22 + 0.78 * weight:.3f})"
            line_width = 1.2 + 5.0 * weight
            name = f"Neighbor {track['track_id']} · attention {weight:.2f}"
        else:
            line_color = "rgba(100,116,139,0.26)"
            line_width = 1.0
            name = f"Actor {track['track_id']}"
        dynamic_indices.append(len(figure.data))
        figure.add_trace(
            go.Scatter(
                x=[], y=[], mode="lines", line={"color": line_color, "width": line_width},
                name=name, showlegend=False,
                hovertemplate=f"{name}<extra></extra>",
            )
        )
        actor_tracks.append({"steps": actor_steps, "xy": actor_xy})

    focal_xy = _track_positions(focal)[obs_mask]
    dynamic_indices.append(len(figure.data))
    focal_trace = len(figure.data)
    figure.add_trace(
        go.Scatter(
            x=[], y=[], mode="lines", line={"color": "#2563eb", "width": 3},
            name="Focal past", hovertemplate="Focal past<extra></extra>",
        )
    )
    dynamic_indices.append(len(figure.data))
    current_trace = len(figure.data)
    figure.add_trace(
        go.Scatter(
            x=[], y=[], mode="markers", marker={"color": "#111827", "size": 9, "symbol": "circle"},
            name="Current focal position", hoverinfo="skip",
        )
    )

    palette = ["#dc2626", "#2563eb", "#7c3aed", "#059669", "#d97706", "#0891b2"]
    prediction_traces: list[int] = []
    for mode_index, trajectory in enumerate(result["trajectories"]):
        dynamic_indices.append(len(figure.data))
        prediction_traces.append(len(figure.data))
        probability = float(result["scores"][mode_index])
        label = "Top-1" if mode_index == result["top_index"] else f"Mode {mode_index + 1}"
        figure.add_trace(
            go.Scatter(
                x=[], y=[], mode="lines+markers",
                line={
                    "color": palette[mode_index % len(palette)],
                    "width": 3.8 if mode_index == result["top_index"] else 1.6,
                    "dash": "solid" if mode_index == result["top_index"] else "dot",
                },
                marker={"size": 3, "opacity": 0.65},
                opacity=1.0 if mode_index == result["top_index"] else 0.62,
                name=f"{label} · {probability:.0%}",
                hovertemplate=f"{label} · {probability:.1%}<extra></extra>",
            )
        )

    dynamic_indices.append(len(figure.data))
    ground_truth_trace = len(figure.data)
    figure.add_trace(
        go.Scatter(
            x=[], y=[], mode="lines+markers",
            line={"color": "#111827", "width": 2.4, "dash": "dash"},
            marker={"size": 3}, name="Ground truth",
        )
    )

    def updates_for_frame(frame_index: int) -> list[go.Scatter]:
        timestep = int(step_values[frame_index])
        post_cutoff = timestep > cutoff_timestep
        shown_timestep = min(timestep, cutoff_timestep)
        updates: list[go.Scatter] = []
        for actor in actor_tracks:
            valid = actor["steps"] <= shown_timestep
            xy = actor["xy"][valid]
            updates.append(go.Scatter(x=xy[:, 0], y=xy[:, 1]))
        focal_valid = obs_steps <= shown_timestep
        shown_focal = focal_xy[focal_valid]
        updates.append(go.Scatter(x=shown_focal[:, 0], y=shown_focal[:, 1]))
        if len(shown_focal):
            updates.append(go.Scatter(x=[shown_focal[-1, 0]], y=[shown_focal[-1, 1]]))
        else:
            updates.append(go.Scatter(x=[], y=[]))
        for trajectory in result["trajectories"]:
            if post_cutoff:
                updates.append(go.Scatter(x=trajectory[:, 0], y=trajectory[:, 1]))
            else:
                updates.append(go.Scatter(x=[], y=[]))
        if post_cutoff and reveal_ground_truth:
            truth = result["future"]
            updates.append(go.Scatter(x=truth[:, 0], y=truth[:, 1]))
        else:
            updates.append(go.Scatter(x=[], y=[]))
        return updates

    frames = [
        go.Frame(
            name=str(index),
            data=updates_for_frame(index),
            traces=dynamic_indices,
        )
        for index in range(len(step_values))
    ]
    initial_index = cutoff_index if start_at_cutoff else 0
    initial_data = updates_for_frame(initial_index)
    for trace_index, update in zip(dynamic_indices, initial_data):
        figure.data[trace_index].x = update.x
        figure.data[trace_index].y = update.y
    slider_steps = [
        {
            "method": "animate",
            "args": [[str(index)], {"mode": "immediate", "frame": {"duration": 80, "redraw": False}, "transition": {"duration": 0}}],
            "label": f"{seconds[index]:.1f}s",
        }
        for index in range(len(step_values))
        if index % 5 == 0 or index in (cutoff_index, len(step_values) - 1)
    ]
    figure.frames = frames
    figure.update_layout(
        title=f"{scenario['city']} · {scenario['scenario_id']} · cutoff {cutoff_elapsed:.1f}s",
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
        xaxis={"showgrid": True, "gridcolor": "#e2e8f0", "zeroline": False},
        yaxis_gridcolor="#e2e8f0",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin={"l": 10, "r": 10, "t": 55, "b": 20},
        legend={"orientation": "h", "y": -0.12, "x": 0},
        updatemenus=[
            {
                "type": "buttons", "showactive": False, "x": 0.0, "y": 1.13,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [None, {"frame": {"duration": 100, "redraw": False}, "fromcurrent": True, "transition": {"duration": 0}}],
                    },
                    {
                        "label": "Ⅱ Pause",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}],
                    },
                ],
            }
        ],
        sliders=[{"active": next((i for i, step in enumerate(slider_steps) if step["label"] == f"{seconds[initial_index]:.1f}s"), 0), "steps": slider_steps, "currentvalue": {"prefix": "Timeline · "}, "pad": {"t": 35}}],
    )
    return figure


def focal_state(scenario: dict[str, Any]) -> dict[str, float | int]:
    focal = next(track for track in scenario["tracks"] if track["track_id"] == scenario["focal_track_id"])
    observed = np.asarray(focal["observed"], dtype=bool)
    observed_positions = _track_positions(focal)[observed]
    speed = float(np.linalg.norm(np.array([focal["velocity_x"][observed][-1], focal["velocity_y"][observed][-1]], dtype=float)))
    heading = float(focal["heading"][observed][-1])
    current_timestep = float(np.asarray(focal["timestep"])[observed][-1])
    neighbor_count = 0
    for track in scenario["tracks"]:
        if track["track_id"] == scenario["focal_track_id"]:
            continue
        actor_observed = np.asarray(track["observed"], dtype=bool)
        if np.any(np.asarray(track["timestep"])[actor_observed] == current_timestep):
            neighbor_count += 1
    return {
        "speed": speed,
        "heading": heading,
        "neighbors": neighbor_count,
        "current_x": float(observed_positions[-1, 0]),
        "current_y": float(observed_positions[-1, 1]),
    }
