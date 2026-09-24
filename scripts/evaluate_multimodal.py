"""Evaluate top-1 and oracle K-mode trajectory metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from motion_forecasting.datasets import MapContextDataset, SocialTrajectoryDataset
from motion_forecasting.metrics import multimodal_metrics
from motion_forecasting.models import MultimodalForecaster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--map-cache", type=Path, default=Path("cache/val_500_maps.npz"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--miss-threshold", type=float)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint not found: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    threshold = (
        float(args.miss_threshold)
        if args.miss_threshold is not None
        else float(checkpoint.get("miss_threshold", 2.0))
    )
    if threshold < 0:
        parser.error("miss-threshold must be non-negative")

    trajectories = SocialTrajectoryDataset(
        args.data,
        past_steps=int(checkpoint["model_config"].get("past_steps", 50)),
        future_steps=int(checkpoint["model_config"]["future_steps"]),
        max_neighbors=int(checkpoint["model_config"]["max_neighbors"]),
    )
    dataset = MapContextDataset(trajectories, args.map_cache)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    model = MultimodalForecaster(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(args.device)
    model.to(device).eval()

    predictions: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            predicted, logits = model(
                batch["history"].to(device),
                batch["neighbors"].to(device),
                batch["neighbor_mask"].to(device),
                batch["lanes"].to(device),
                batch["lane_mask"].to(device),
            )
            predictions.append(predicted.cpu().numpy())
            logits_list.append(logits.cpu().numpy())
            targets.append(batch["future"].numpy())
    metrics = multimodal_metrics(
        np.concatenate(predictions),
        np.concatenate(logits_list),
        np.concatenate(targets),
        miss_threshold=threshold,
    )
    result = {
        "scenarios": len(dataset),
        "num_modes": int(checkpoint["model_config"]["num_modes"]),
        "miss_threshold_m": threshold,
        **metrics,
        "checkpoint": str(args.checkpoint),
        "best_epoch": int(checkpoint["epoch"]),
        "train_scenarios": int(checkpoint["train_scenarios"]),
    }
    print("Multimodal Forecasting")
    print(f"Scenarios: {result['scenarios']}")
    print(f"Top-1 ADE/FDE: {metrics['top1_ade']:.3f} / {metrics['top1_fde']:.3f} m")
    print(
        f"minADE@{result['num_modes']}/minFDE@{result['num_modes']}: "
        f"{metrics['min_ade']:.3f} / {metrics['min_fde']:.3f} m"
    )
    print(
        f"MissRate@{result['num_modes']} (>{threshold:.2f} m): "
        f"{metrics['miss_rate']:.3f}"
    )
    metrics_path = args.checkpoint.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
