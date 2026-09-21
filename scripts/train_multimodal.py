"""Train a K-mode decoder on the map-aware interaction GRU encoder."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from motion_forecasting.datasets import MapContextDataset, SocialTrajectoryDataset
from motion_forecasting.metrics import multimodal_metrics
from motion_forecasting.models import MultimodalForecaster


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _predict_metrics(
    model: MultimodalForecaster,
    loader: DataLoader,
    device: torch.device,
    miss_threshold: float,
) -> dict[str, float]:
    model.eval()
    predictions: list[np.ndarray] = []
    logits_list: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            trajectories, logits = model(
                batch["history"].to(device),
                batch["neighbors"].to(device),
                batch["neighbor_mask"].to(device),
                batch["lanes"].to(device),
                batch["lane_mask"].to(device),
            )
            predictions.append(trajectories.cpu().numpy())
            logits_list.append(logits.cpu().numpy())
            targets.append(batch["future"].numpy())
    return multimodal_metrics(
        np.concatenate(predictions),
        np.concatenate(logits_list),
        np.concatenate(targets),
        miss_threshold=miss_threshold,
    )


def _plot_history(rows: list[dict[str, float]], path: Path) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    plots = (
        ("train_loss", "Training total loss", "Loss"),
        ("val_min_ade", "Validation minADE@K", "Meters"),
        ("val_min_fde", "Validation minFDE@K", "Meters"),
    )
    for axis, (column, title, ylabel) in zip(axes, plots):
        axis.plot(epochs, [row[column] for row in rows], marker="o")
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--train-map-cache", type=Path, default=Path("cache/train_20k_maps.npz"))
    parser.add_argument("--val-map-cache", type=Path, default=Path("cache/val_500_maps.npz"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-neighbors", type=int, default=8)
    parser.add_argument("--num-modes", type=int, default=6)
    parser.add_argument("--classification-weight", type=float, default=0.5)
    parser.add_argument("--scoring-loss", choices=("hard", "soft"), default="hard")
    parser.add_argument("--scoring-temperature", type=float, default=1.0)
    parser.add_argument("--diversity-weight", type=float, default=0.05)
    parser.add_argument("--diversity-margin", type=float, default=2.0)
    parser.add_argument("--miss-threshold", type=float, default=2.0)
    parser.add_argument("--train-scenarios", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="multimodal")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/multimodal.pt"))
    args = parser.parse_args()
    if (
        args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0
        or args.hidden_dim < 1 or args.num_neighbors < 1 or args.num_modes < 1
        or args.classification_weight < 0 or args.diversity_weight < 0
        or args.scoring_temperature <= 0
        or args.diversity_margin < 0 or args.miss_threshold < 0
        or args.num_workers < 0
    ):
        parser.error("dimensions and epochs must be positive, temperature must be positive, and loss weights and thresholds must be non-negative")
    if args.train_scenarios is not None and args.train_scenarios < 1:
        parser.error("train-scenarios must be positive")
    if not args.run_name.strip() or Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        parser.error("run-name must be a non-empty directory name without path components")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    for cache_path in (args.train_map_cache, args.val_map_cache):
        if not cache_path.is_file():
            parser.error(f"map cache not found: {cache_path}; run scripts/precompute_map_context.py first")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)

    train_trajectories = SocialTrajectoryDataset(
        args.train_data, max_neighbors=args.num_neighbors, max_scenarios=args.train_scenarios
    )
    val_trajectories = SocialTrajectoryDataset(
        args.val_data, past_steps=train_trajectories.past_steps,
        future_steps=train_trajectories.future_steps, max_neighbors=args.num_neighbors,
    )
    train_data = MapContextDataset(train_trajectories, args.train_map_cache)
    val_data = MapContextDataset(val_trajectories, args.val_map_cache)
    max_lanes, points_per_lane, lane_dim = train_data.lanes.shape[1:]
    if train_data.lanes.shape[1:] != val_data.lanes.shape[1:]:
        parser.error("train/validation map-cache tensor shapes differ")

    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "num_workers": args.num_workers,
        "worker_init_fn": _seed_worker if args.num_workers else None,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True,
        generator=generator, **loader_options,
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, **loader_options
    )
    model_config = {
        "input_dim": train_trajectories.input_dim,
        "neighbor_dim": train_trajectories.neighbor_dim,
        "lane_dim": lane_dim,
        "hidden_dim": args.hidden_dim,
        "future_steps": train_trajectories.future_steps,
        "max_neighbors": args.num_neighbors,
        "max_lanes": max_lanes,
        "points_per_lane": points_per_lane,
        "num_modes": args.num_modes,
    }
    model = MultimodalForecaster(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    cross_entropy = nn.CrossEntropyLoss()
    best_min_ade = float("inf")
    run_dir = Path("runs") / args.run_name
    metrics_path = run_dir / "metrics.csv"
    curves_path = run_dir / "learning_curves.png"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "epoch", "train_loss", "train_trajectory_loss", "train_classification_loss",
        "train_diversity_loss", "val_top1_ade", "val_top1_fde", "val_min_ade",
        "val_min_fde", "val_miss_rate",
    ]
    rows: list[dict[str, float]] = []
    pair_indices = torch.triu_indices(args.num_modes, args.num_modes, offset=1, device=device)

    print(f"Training scenarios: {len(train_data)}; validation scenarios: {len(val_data)}", flush=True)
    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_losses: list[list[float]] = []
            for batch in train_loader:
                history = batch["history"].to(device)
                neighbors = batch["neighbors"].to(device)
                neighbor_mask = batch["neighbor_mask"].to(device)
                lanes = batch["lanes"].to(device)
                lane_mask = batch["lane_mask"].to(device)
                target = batch["future"].to(device)
                trajectories, logits = model(history, neighbors, neighbor_mask, lanes, lane_mask)
                errors = (trajectories - target.unsqueeze(1)).square().mean(dim=(2, 3))
                best_modes = errors.argmin(dim=1)
                trajectory_loss = errors.gather(1, best_modes.unsqueeze(1)).mean()
                if args.scoring_loss == "hard":
                    classification_loss = cross_entropy(logits, best_modes)
                else:
                    displacement = torch.linalg.vector_norm(
                        trajectories - target.unsqueeze(1), dim=-1
                    )
                    quality_error = (
                        displacement.mean(dim=-1) + 0.5 * displacement[:, :, -1]
                    ).detach()
                    target_probs = torch.softmax(
                        -quality_error / args.scoring_temperature, dim=1
                    )
                    classification_loss = F.kl_div(
                        F.log_softmax(logits, dim=1),
                        target_probs,
                        reduction="batchmean",
                    )
                if args.num_modes > 1:
                    endpoints = trajectories[:, :, -1]
                    distances = torch.linalg.vector_norm(
                        endpoints[:, pair_indices[0]] - endpoints[:, pair_indices[1]], dim=-1
                    )
                    diversity_loss = torch.relu(args.diversity_margin - distances).square().mean()
                else:
                    diversity_loss = trajectories.new_zeros(())
                loss = (
                    trajectory_loss
                    + args.classification_weight * classification_loss
                    + args.diversity_weight * diversity_loss
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append([
                    float(loss.detach()), float(trajectory_loss.detach()),
                    float(classification_loss.detach()), float(diversity_loss.detach()),
                ])

            metrics = _predict_metrics(model, val_loader, device, args.miss_threshold)
            mean_losses = np.asarray(epoch_losses).mean(axis=0)
            row: dict[str, float] = {
                "epoch": epoch,
                "train_loss": float(mean_losses[0]),
                "train_trajectory_loss": float(mean_losses[1]),
                "train_classification_loss": float(mean_losses[2]),
                "train_diversity_loss": float(mean_losses[3]),
                "val_top1_ade": metrics["top1_ade"],
                "val_top1_fde": metrics["top1_fde"],
                "val_min_ade": metrics["min_ade"],
                "val_min_fde": metrics["min_fde"],
                "val_miss_rate": metrics["miss_rate"],
            }
            rows.append(row)
            writer.writerow(row)
            csv_file.flush()
            print(
                f"Epoch {epoch:02d}/{args.epochs}: loss={row['train_loss']:.4f}, "
                f"top1 ADE/FDE={metrics['top1_ade']:.3f}/{metrics['top1_fde']:.3f} m, "
                f"minADE@{args.num_modes}/minFDE@{args.num_modes}="
                f"{metrics['min_ade']:.3f}/{metrics['min_fde']:.3f} m, "
                f"MissRate@{args.num_modes}={metrics['miss_rate']:.3f}",
                flush=True,
            )
            if metrics["min_ade"] < best_min_ade:
                best_min_ade = metrics["min_ade"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": model_config,
                        "representation": "agent-centric",
                        "train_scenarios": len(train_data),
                        "seed": args.seed,
                        "epoch": epoch,
                        "classification_weight": args.classification_weight,
                        "scoring_loss": args.scoring_loss,
                        "scoring_temperature": args.scoring_temperature,
                        "diversity_weight": args.diversity_weight,
                        "diversity_margin": args.diversity_margin,
                        "miss_threshold": args.miss_threshold,
                        "validation_metrics": metrics,
                    },
                    args.output,
                )
                print(f"  saved best minADE checkpoint: {args.output}", flush=True)

    _plot_history(rows, curves_path)
    print(f"Saved metrics to {metrics_path}", flush=True)
    print(f"Saved learning curve to {curves_path}", flush=True)


if __name__ == "__main__":
    main()
