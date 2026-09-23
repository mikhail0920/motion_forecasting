"""Evaluate an interaction-aware GRU and optionally show neighbor attention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from motion_forecasting.datasets import SocialTrajectoryDataset
from motion_forecasting.metrics import ade, fde
from motion_forecasting.models import InteractionTrajectoryGRU


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--attention-scenarios", type=int, default=3)
    parser.add_argument("--metrics-output", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.attention_scenarios < 0:
        parser.error("batch-size must be positive; num-workers and attention-scenarios cannot be negative")
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
    model = InteractionTrajectoryGRU(
        input_dim=int(checkpoint.get("input_dim", 4)),
        neighbor_dim=int(checkpoint.get("neighbor_dim", 5)),
        hidden_dim=int(checkpoint["hidden_dim"]),
        future_steps=int(checkpoint["future_steps"]),
        max_neighbors=int(checkpoint["max_neighbors"]),
        interaction_features=bool(checkpoint.get("interaction_features", False)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    ade_scores: list[float] = []
    fde_scores: list[float] = []
    scenario_offset = 0
    printed_scenarios = 0
    with torch.inference_mode():
        for batch in loader:
            history = batch["history"].to(device)
            neighbors = batch["neighbors"].to(device)
            neighbor_mask = batch["neighbor_mask"].to(device)
            prediction, attention = model(
                history, neighbors, neighbor_mask, return_attention=True
            )
            target = batch["future"].to(device)
            for local_index, (pred, truth) in enumerate(
                zip(prediction.cpu().numpy(), target.cpu().numpy())
            ):
                ade_scores.append(ade(pred, truth))
                fde_scores.append(fde(pred, truth))
                if printed_scenarios >= args.attention_scenarios:
                    continue
                batch_index = local_index
                valid_indices = torch.where(neighbor_mask[batch_index])[0].tolist()
                weights = attention[batch_index].cpu().numpy()
                neighbor_positions = neighbors[batch_index, :, -1, :2].cpu().numpy()
                ranked = sorted(valid_indices, key=lambda idx: float(weights[idx]), reverse=True)
                scenario_id = dataset.scenario_ids[scenario_offset + local_index]
                print(f"Neighbor attention — scenario {scenario_id}:")
                for rank, neighbor_index in enumerate(ranked[:8], start=1):
                    distance = float(np.linalg.norm(neighbor_positions[neighbor_index]))
                    print(
                        f"  #{rank} distance={distance:.1f}m "
                        f"weight={float(weights[neighbor_index]):.3f}"
                    )
                if not ranked:
                    print("  no observed neighbors")
                printed_scenarios += 1
            scenario_offset += len(prediction)

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
                "interaction_features": bool(checkpoint.get("interaction_features", False)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Interaction GRU Trajectory Forecasting")
    print(f"Scenarios: {len(dataset)}")
    print(f"Interaction features: {bool(checkpoint.get('interaction_features', False))}")
    print(f"ADE: {ade_value:.3f} m")
    print(f"FDE: {fde_value:.3f} m")
    print(f"Saved metrics to {metrics_output}")
    if dataset.skipped_scenarios:
        print(f"Skipped: {len(dataset.skipped_scenarios)}")


if __name__ == "__main__":
    main()
