"""Evaluate a trained trajectory MLP on AV2 focal tracks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from motion_forecasting.datasets import TrajectoryDataset
from motion_forecasting.metrics import ade, fde
from motion_forecasting.models import TrajectoryMLP


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available in this PyTorch environment")

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    dataset = TrajectoryDataset(
        args.data,
        past_steps=int(checkpoint["past_steps"]),
        future_steps=int(checkpoint["future_steps"]),
        representation=checkpoint.get("representation", "basic"),
    )
    model = TrajectoryMLP(
        past_steps=int(checkpoint["past_steps"]),
        future_steps=int(checkpoint["future_steps"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        input_dim=int(checkpoint.get("input_dim", 2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    ade_scores: list[float] = []
    fde_scores: list[float] = []
    with torch.inference_mode():
        for batch in loader:
            history = batch["history"].to(device)
            target = batch["future"].to(device)
            prediction = model(history)
            for pred, truth in zip(prediction.cpu().numpy(), target.cpu().numpy()):
                ade_scores.append(ade(pred, truth))
                fde_scores.append(fde(pred, truth))

    print("MLP Trajectory Forecasting")
    print(f"Scenarios: {len(dataset)}")
    print(f"ADE: {np.mean(ade_scores):.3f} m")
    print(f"FDE: {np.mean(fde_scores):.3f} m")
    if dataset.skipped_scenarios:
        print(f"Skipped: {len(dataset.skipped_scenarios)}")


if __name__ == "__main__":
    main()
