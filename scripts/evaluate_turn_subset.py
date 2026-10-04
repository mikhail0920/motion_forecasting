"""Compare lane-conditioned checkpoints on held-out turning scenarios."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from motion_forecasting.datasets import MapContextDataset, SocialTrajectoryDataset
from motion_forecasting.metrics import multimodal_metrics
from motion_forecasting.models import LaneConditionedForecaster


def _predict(
    checkpoint_path: Path,
    dataset: MapContextDataset,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = LaneConditionedForecaster(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    predictions: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.inference_mode():
        for batch in loader:
            prediction, logits = model(
                batch["history"].to(device),
                batch["neighbors"].to(device),
                batch["neighbor_mask"].to(device),
                batch["lanes"].to(device),
                batch["lane_mask"].to(device),
            )
            predictions.append(prediction.cpu().numpy())
            logits_all.append(logits.cpu().numpy())
            targets.append(batch["future"].numpy())
    return np.concatenate(predictions), np.concatenate(logits_all), np.concatenate(targets)


def _metrics(
    predictions: np.ndarray,
    logits: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    return multimodal_metrics(
        predictions[mask], logits[mask], targets[mask], miss_threshold=2.0
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--baseline-map-cache", type=Path, required=True)
    parser.add_argument("--route-map-cache", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--route-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turn-threshold-degrees", type=float, default=20.0)
    parser.add_argument("--min-displacement-meters", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.batch_size < 1 or args.turn_threshold_degrees <= 0 or args.min_displacement_meters < 0:
        parser.error("batch size and turn threshold must be positive; displacement must be non-negative")
    for path in (
        args.baseline_map_cache,
        args.route_map_cache,
        args.baseline_checkpoint,
        args.route_checkpoint,
    ):
        if not path.is_file():
            parser.error(f"file not found: {path}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")

    baseline_checkpoint = torch.load(
        args.baseline_checkpoint, map_location="cpu", weights_only=False
    )
    route_checkpoint = torch.load(
        args.route_checkpoint, map_location="cpu", weights_only=False
    )
    baseline_config = baseline_checkpoint["model_config"]
    route_config = route_checkpoint["model_config"]
    if baseline_config["max_neighbors"] != route_config["max_neighbors"]:
        parser.error("baseline and route models must use the same neighbor count")

    trajectories = SocialTrajectoryDataset(
        args.data,
        past_steps=int(baseline_config.get("past_steps", 50)),
        future_steps=int(baseline_config["future_steps"]),
        max_neighbors=int(baseline_config["max_neighbors"]),
    )
    baseline_data = MapContextDataset(trajectories, args.baseline_map_cache)
    route_data = MapContextDataset(trajectories, args.route_map_cache)
    device = torch.device(args.device)
    baseline_pred, baseline_logits, targets = _predict(
        args.baseline_checkpoint, baseline_data,
        batch_size=args.batch_size, device=device,
    )
    route_pred, route_logits, _ = _predict(
        args.route_checkpoint, route_data,
        batch_size=args.batch_size, device=device,
    )

    start = targets[:, 0]
    end = targets[:, -1]
    displacement = np.linalg.norm(end - start, axis=1)
    bearing = np.abs(np.arctan2(end[:, 1], end[:, 0]))
    turn_mask = (np.degrees(bearing) >= args.turn_threshold_degrees) & (
        displacement >= args.min_displacement_meters
    )
    if not turn_mask.any() or turn_mask.all():
        parser.error(
            f"turn split must contain both groups; found {int(turn_mask.sum())} turn scenes "
            f"out of {len(turn_mask)}"
        )

    result = {
        "scenarios": len(trajectories),
        "turn_definition": {
            "future_endpoint_bearing_degrees_at_least": args.turn_threshold_degrees,
            "future_displacement_meters_at_least": args.min_displacement_meters,
            "selection_uses_future_only_for_reporting": True,
        },
        "turn_scenarios": int(turn_mask.sum()),
        "straight_or_mild_scenarios": int((~turn_mask).sum()),
        "baseline": {
            "checkpoint": str(args.baseline_checkpoint),
            "all": _metrics(baseline_pred, baseline_logits, targets, np.ones(len(targets), dtype=bool)),
            "turns": _metrics(baseline_pred, baseline_logits, targets, turn_mask),
            "straight_or_mild": _metrics(baseline_pred, baseline_logits, targets, ~turn_mask),
        },
        "routes": {
            "checkpoint": str(args.route_checkpoint),
            "all": _metrics(route_pred, route_logits, targets, np.ones(len(targets), dtype=bool)),
            "turns": _metrics(route_pred, route_logits, targets, turn_mask),
            "straight_or_mild": _metrics(route_pred, route_logits, targets, ~turn_mask),
        },
        "turn_scenario_ids": [
            scenario_id
            for scenario_id, is_turn in zip(trajectories.scenario_ids, turn_mask)
            if is_turn
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Turn scenarios: {result['turn_scenarios']} / {result['scenarios']}")
    for key in ("baseline", "routes"):
        for split in ("all", "turns", "straight_or_mild"):
            metrics = result[key][split]
            print(
                f"{key} {split}: top1 ADE/FDE="
                f"{metrics['top1_ade']:.3f}/{metrics['top1_fde']:.3f} m; "
                f"minADE/minFDE={metrics['min_ade']:.3f}/{metrics['min_fde']:.3f} m; "
                f"MissRate={metrics['miss_rate']:.3f}"
            )
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
