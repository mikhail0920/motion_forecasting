"""Evaluate a trained social GRU on an AV2 scenario subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from motion_forecasting.datasets import SocialTrajectoryDataset
from motion_forecasting.metrics import ade, fde
from motion_forecasting.models import SocialTrajectoryGRU


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--metrics-output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers cannot be negative")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available in this PyTorch environment")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    dataset = SocialTrajectoryDataset(
        args.data,
        past_steps=int(checkpoint["past_steps"]),
        future_steps=int(checkpoint["future_steps"]),
        max_neighbors=int(checkpoint["max_neighbors"]),
    )
    model = SocialTrajectoryGRU(
        input_dim=int(checkpoint.get("input_dim", 4)),
        neighbor_dim=int(checkpoint.get("neighbor_dim", 5)),
        hidden_dim=int(checkpoint["hidden_dim"]),
        future_steps=int(checkpoint["future_steps"]),
        max_neighbors=int(checkpoint["max_neighbors"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    ade_scores: list[float] = []
    fde_scores: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            prediction = model(
                batch["history"].to(device),
                batch["neighbors"].to(device),
                batch["neighbor_mask"].to(device),
            )
            target = batch["future"].to(device)
            for pred, truth in zip(prediction.cpu().numpy(), target.cpu().numpy()):
                ade_scores.append(ade(pred, truth))
                fde_scores.append(fde(pred, truth))

    ade_value = float(np.mean(ade_scores))
    fde_value = float(np.mean(fde_scores))
    metrics_output = args.metrics_output or args.checkpoint.with_suffix(".metrics.json")
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(
        json.dumps(
            {
                "scenarios": len(dataset),
                "ade_m": ade_value,
                "fde_m": fde_value,
                "checkpoint": str(args.checkpoint),
                "train_scenarios": checkpoint.get("train_scenarios"),
                "seed": checkpoint.get("seed"),
                "best_epoch": checkpoint.get("epoch"),
                "representation": checkpoint.get("representation", "agent-centric"),
                "max_neighbors": checkpoint["max_neighbors"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Social GRU Trajectory Forecasting")
    print(f"Scenarios: {len(dataset)}")
    print(f"ADE: {ade_value:.3f} m")
    print(f"FDE: {fde_value:.3f} m")
    print(f"Saved metrics to {metrics_output}")
    if dataset.skipped_scenarios:
        print(f"Skipped: {len(dataset.skipped_scenarios)}")


if __name__ == "__main__":
    main()
