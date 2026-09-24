"""Evaluate lane-conditioned hypotheses, optionally with a trained reranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from motion_forecasting.datasets import MapContextDataset, SocialTrajectoryDataset
from motion_forecasting.metrics import multimodal_metrics
from motion_forecasting.models import LaneConditionedForecaster, TrajectoryAwareModeReranker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--map-cache", type=Path, default=Path("cache/val_500_maps.npz"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reranker-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--miss-threshold", type=float)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch-size must be positive and num-workers non-negative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    for path in (args.checkpoint, args.map_cache):
        if not path.is_file():
            parser.error(f"file not found: {path}")
    if args.reranker_checkpoint is not None and not args.reranker_checkpoint.is_file():
        parser.error(f"reranker checkpoint not found: {args.reranker_checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    threshold = (
        float(args.miss_threshold)
        if args.miss_threshold is not None
        else float(checkpoint.get("miss_threshold", 2.0))
    )
    if threshold < 0:
        parser.error("miss-threshold must be non-negative")
    model_config = checkpoint["model_config"]
    trajectories = SocialTrajectoryDataset(
        args.data,
        past_steps=int(model_config.get("past_steps", 50)),
        future_steps=int(model_config["future_steps"]),
        max_neighbors=int(model_config["max_neighbors"]),
    )
    dataset = MapContextDataset(trajectories, args.map_cache)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    device = torch.device(args.device)
    model = LaneConditionedForecaster(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    reranker = None
    if args.reranker_checkpoint is not None:
        reranker_state = torch.load(
            args.reranker_checkpoint, map_location="cpu", weights_only=False
        )
        reranker = TrajectoryAwareModeReranker(
            scene_dim=int(reranker_state["scene_dim"]),
            future_steps=int(reranker_state["future_steps"]),
            hidden_dim=int(reranker_state["hidden_dim"]),
        )
        reranker.load_state_dict(reranker_state["model_state_dict"])
        reranker.to(device).eval()

    predictions: list[np.ndarray] = []
    logits_all: list[np.ndarray] = []
    reranked_scores: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            predicted, logits, scene_embedding = model.generate_with_context(
                batch["history"].to(device),
                batch["neighbors"].to(device),
                batch["neighbor_mask"].to(device),
                batch["lanes"].to(device),
                batch["lane_mask"].to(device),
            )
            predictions.append(predicted.cpu().numpy())
            logits_all.append(logits.cpu().numpy())
            targets.append(batch["future"].numpy())
            if reranker is not None:
                reranked_scores.append(reranker(scene_embedding, predicted).cpu().numpy())

    candidates = np.concatenate(predictions)
    target = np.concatenate(targets)
    baseline_metrics = multimodal_metrics(
        candidates,
        np.concatenate(logits_all),
        target,
        miss_threshold=threshold,
    )
    result: dict[str, object] = {
        "scenarios": len(dataset),
        "num_modes": int(model_config["num_modes"]),
        "num_lane_modes": int(model_config["num_lane_modes"]),
        "miss_threshold_m": threshold,
        "checkpoint": str(args.checkpoint),
        "best_epoch": int(checkpoint["epoch"]),
        "train_scenarios": int(checkpoint["train_scenarios"]),
        "generator": baseline_metrics,
    }
    if reranker is not None:
        result["reranker_checkpoint"] = str(args.reranker_checkpoint)
        result["reranker_best_epoch"] = int(reranker_state["epoch"])
    print("Lane-conditioned multimodal generator")
    print(f"Scenarios: {len(dataset)}; modes: {model_config['num_modes']} ")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(
        f"Top-1 ADE/FDE: {baseline_metrics['top1_ade']:.3f} / "
        f"{baseline_metrics['top1_fde']:.3f} m"
    )
    print(
        f"minADE@{model_config['num_modes']}/minFDE@{model_config['num_modes']}: "
        f"{baseline_metrics['min_ade']:.3f} / {baseline_metrics['min_fde']:.3f} m"
    )
    print(
        f"MissRate@{model_config['num_modes']} (>{threshold:.2f} m): "
        f"{baseline_metrics['miss_rate']:.3f}"
    )
    if reranker is not None:
        reranked_metrics = multimodal_metrics(
            candidates,
            np.concatenate(reranked_scores),
            target,
            miss_threshold=threshold,
        )
        result["reranked"] = reranked_metrics
        print("Existing trajectory-aware reranker over lane-conditioned candidates")
        print(
            f"Top-1 ADE/FDE: {reranked_metrics['top1_ade']:.3f} / "
            f"{reranked_metrics['top1_fde']:.3f} m"
        )
        print(
            f"Oracle minADE/minFDE: {reranked_metrics['min_ade']:.3f} / "
            f"{reranked_metrics['min_fde']:.3f} m; "
            f"MissRate: {reranked_metrics['miss_rate']:.3f}"
        )
    metrics_path = args.checkpoint.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
