"""Evaluate a frozen multimodal generator with its candidate reranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from motion_forecasting.datasets import MapContextDataset, SocialTrajectoryDataset
from motion_forecasting.metrics import multimodal_metrics
from motion_forecasting.models import MultimodalForecaster, TrajectoryAwareModeReranker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--map-cache", type=Path, default=Path("cache/val_500_maps.npz"))
    parser.add_argument("--generator-checkpoint", type=Path, required=True)
    parser.add_argument("--reranker-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.miss_threshold < 0:
        parser.error("batch-size must be positive; workers and miss threshold non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if not args.generator_checkpoint.is_file():
        parser.error(f"generator checkpoint not found: {args.generator_checkpoint}")
    if not args.reranker_checkpoint.is_file():
        parser.error(f"reranker checkpoint not found: {args.reranker_checkpoint}")

    generator_state = torch.load(
        args.generator_checkpoint, map_location="cpu", weights_only=False
    )
    reranker_state = torch.load(
        args.reranker_checkpoint, map_location="cpu", weights_only=False
    )
    if Path(reranker_state["generator_checkpoint"]).name != args.generator_checkpoint.name:
        raise ValueError(
            "reranker was trained with a different generator checkpoint: "
            f"{reranker_state['generator_checkpoint']}"
        )
    trajectories = SocialTrajectoryDataset(
        args.data,
        past_steps=int(generator_state["model_config"].get("past_steps", 50)),
        future_steps=int(generator_state["model_config"]["future_steps"]),
        max_neighbors=int(generator_state["model_config"]["max_neighbors"]),
    )
    dataset = MapContextDataset(trajectories, args.map_cache)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    device = torch.device(args.device)
    generator = MultimodalForecaster(**generator_state["model_config"])
    generator.load_state_dict(generator_state["model_state_dict"])
    generator.to(device).eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    reranker = TrajectoryAwareModeReranker(
        scene_dim=int(reranker_state["scene_dim"]),
        future_steps=int(reranker_state["future_steps"]),
        hidden_dim=int(reranker_state["hidden_dim"]),
    )
    reranker.load_state_dict(reranker_state["model_state_dict"])
    reranker.to(device).eval()

    predictions: list[np.ndarray] = []
    original_scores: list[np.ndarray] = []
    reranked_scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            predicted, original_logits, scene_embedding = generator.generate_with_context(
                batch["history"].to(device),
                batch["neighbors"].to(device),
                batch["neighbor_mask"].to(device),
                batch["lanes"].to(device),
                batch["lane_mask"].to(device),
            )
            scores = reranker(scene_embedding, predicted)
            predictions.append(predicted.cpu().numpy())
            original_scores.append(original_logits.cpu().numpy())
            reranked_scores.append(scores.cpu().numpy())
            targets.append(batch["future"].numpy())

    candidates = np.concatenate(predictions)
    ground_truth = np.concatenate(targets)
    original_metrics = multimodal_metrics(
        candidates,
        np.concatenate(original_scores),
        ground_truth,
        miss_threshold=args.miss_threshold,
    )
    reranked_metrics = multimodal_metrics(
        candidates,
        np.concatenate(reranked_scores),
        ground_truth,
        miss_threshold=args.miss_threshold,
    )
    result = {
        "scenarios": len(dataset),
        "num_modes": candidates.shape[1],
        "miss_threshold_m": args.miss_threshold,
        "generator_checkpoint": str(args.generator_checkpoint),
        "reranker_checkpoint": str(args.reranker_checkpoint),
        "reranker_best_epoch": int(reranker_state["epoch"]),
        "original": original_metrics,
        "reranked": reranked_metrics,
    }
    print("Frozen multimodal generator")
    print(
        f"  top-1 ADE/FDE: {original_metrics['top1_ade']:.3f} / "
        f"{original_metrics['top1_fde']:.3f} m"
    )
    print(
        f"  minADE@{result['num_modes']}/minFDE@{result['num_modes']}: "
        f"{original_metrics['min_ade']:.3f} / {original_metrics['min_fde']:.3f} m"
    )
    print("Trajectory-aware reranker")
    print(
        f"  top-1 ADE/FDE: {reranked_metrics['top1_ade']:.3f} / "
        f"{reranked_metrics['top1_fde']:.3f} m"
    )
    print(
        f"  minADE@{result['num_modes']}/minFDE@{result['num_modes']}: "
        f"{reranked_metrics['min_ade']:.3f} / {reranked_metrics['min_fde']:.3f} m"
    )
    print(
        f"  MissRate@{result['num_modes']}: {reranked_metrics['miss_rate']:.3f} "
        f"(threshold {args.miss_threshold:.2f} m)"
    )
    print(f"Best reranker epoch: {result['reranker_best_epoch']}")
    metrics_path = args.reranker_checkpoint.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
