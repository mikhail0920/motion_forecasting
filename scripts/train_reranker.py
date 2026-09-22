"""Freeze a trained multimodal generator and train a trajectory-aware scorer."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from motion_forecasting.datasets import MapContextDataset, SocialTrajectoryDataset
from motion_forecasting.metrics import multimodal_metrics
from motion_forecasting.models import MultimodalForecaster, TrajectoryAwareModeReranker


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _generate_cache(
    generator: MultimodalForecaster,
    loader: DataLoader,
    device: torch.device,
) -> TensorDataset:
    """Generate immutable candidates and scene features once for reranker training."""
    trajectories_all: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []
    scene_all: list[torch.Tensor] = []
    target_all: list[torch.Tensor] = []
    generator.eval()
    with torch.inference_mode():
        for batch in loader:
            trajectories, logits, scene_embedding = generator.generate_with_context(
                batch["history"].to(device),
                batch["neighbors"].to(device),
                batch["neighbor_mask"].to(device),
                batch["lanes"].to(device),
                batch["lane_mask"].to(device),
            )
            trajectories_all.append(trajectories.cpu())
            logits_all.append(logits.cpu())
            scene_all.append(scene_embedding.cpu())
            target_all.append(batch["future"].cpu())
    return TensorDataset(
        torch.cat(scene_all),
        torch.cat(trajectories_all),
        torch.cat(logits_all),
        torch.cat(target_all),
    )


def _calculate_metrics(
    reranker: TrajectoryAwareModeReranker,
    cached: TensorDataset,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    loader = DataLoader(cached, batch_size=batch_size, shuffle=False)
    reranker.eval()
    scores_all: list[np.ndarray] = []
    trajectories_all: list[np.ndarray] = []
    original_logits_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    with torch.inference_mode():
        for scene, trajectories, original_logits, targets in loader:
            scores = reranker(scene.to(device), trajectories.to(device))
            scores_all.append(scores.cpu().numpy())
            trajectories_all.append(trajectories.numpy())
            original_logits_all.append(original_logits.numpy())
            targets_all.append(targets.numpy())
    trajectories = np.concatenate(trajectories_all)
    targets = np.concatenate(targets_all)
    original_metrics = multimodal_metrics(
        trajectories,
        np.concatenate(original_logits_all),
        targets,
        miss_threshold=2.0,
    )
    reranked_metrics = multimodal_metrics(
        trajectories,
        np.concatenate(scores_all),
        targets,
        miss_threshold=2.0,
    )
    return original_metrics, reranked_metrics


def _save_curves(rows: list[dict[str, float]], output: Path) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, column, title in zip(
        axes,
        ("train_ce", "val_top1_ade"),
        ("Reranker training CE", "Validation top-1 ADE"),
    ):
        axis.plot(epochs, [row[column] for row in rows], marker="o")
        axis.set(
            title=title,
            xlabel="Epoch",
            ylabel="Loss" if column == "train_ce" else "Meters",
        )
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument(
        "--generator-checkpoint",
        type=Path,
        default=Path("checkpoints/multimodal_20k.pt"),
    )
    parser.add_argument(
        "--train-map-cache", type=Path, default=Path("cache/train_20k_maps.npz")
    )
    parser.add_argument(
        "--val-map-cache", type=Path, default=Path("cache/val_500_maps.npz")
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--reranker-hidden-dim", type=int, default=128)
    parser.add_argument("--train-scenarios", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="trajectory-reranker")
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/trajectory_reranker.pt")
    )
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.learning_rate <= 0
        or args.reranker_hidden_dim < 1
        or args.num_workers < 0
    ):
        parser.error(
            "epochs, batch-size, learning-rate, and reranker-hidden-dim must be positive"
        )
    if args.train_scenarios is not None and args.train_scenarios < 1:
        parser.error("train-scenarios must be positive")
    if (
        not args.run_name.strip()
        or Path(args.run_name).name != args.run_name
        or args.run_name in {".", ".."}
    ):
        parser.error("run-name must be a non-empty directory name without path components")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if not args.generator_checkpoint.is_file():
        parser.error(f"generator checkpoint not found: {args.generator_checkpoint}")
    for cache_path in (args.train_map_cache, args.val_map_cache):
        if not cache_path.is_file():
            parser.error(f"map cache not found: {cache_path}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    generator_checkpoint = torch.load(
        args.generator_checkpoint, map_location="cpu", weights_only=False
    )
    generator = MultimodalForecaster(**generator_checkpoint["model_config"])
    generator.load_state_dict(generator_checkpoint["model_state_dict"])
    generator.to(device).eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)

    train_trajectories = SocialTrajectoryDataset(
        args.train_data,
        max_neighbors=int(generator_checkpoint["model_config"]["max_neighbors"]),
        max_scenarios=args.train_scenarios,
    )
    val_trajectories = SocialTrajectoryDataset(
        args.val_data,
        past_steps=train_trajectories.past_steps,
        future_steps=train_trajectories.future_steps,
        max_neighbors=int(generator_checkpoint["model_config"]["max_neighbors"]),
    )
    train_data = MapContextDataset(train_trajectories, args.train_map_cache)
    val_data = MapContextDataset(val_trajectories, args.val_map_cache)
    loader_options = {
        "num_workers": args.num_workers,
        "worker_init_fn": _seed_worker if args.num_workers else None,
        "persistent_workers": args.num_workers > 0,
    }
    train_generation_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=False, **loader_options
    )
    val_generation_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, **loader_options
    )
    print(
        f"Frozen generator checkpoint: {args.generator_checkpoint} "
        f"(epoch {generator_checkpoint['epoch']})",
        flush=True,
    )
    print("Caching frozen generator outputs for train and validation...", flush=True)
    train_cached = _generate_cache(generator, train_generation_loader, device)
    val_cached = _generate_cache(generator, val_generation_loader, device)
    if len(train_cached) != len(train_data) or len(val_cached) != len(val_data):
        raise RuntimeError("generated candidate cache does not align with its dataset")

    scene_dim = int(generator_checkpoint["model_config"]["hidden_dim"]) * 3
    reranker = TrajectoryAwareModeReranker(
        scene_dim=scene_dim,
        future_steps=int(generator_checkpoint["model_config"]["future_steps"]),
        hidden_dim=args.reranker_hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(reranker.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    original_metrics, _ = _calculate_metrics(
        reranker, val_cached, args.batch_size, device
    )
    print(
        "Frozen generator baseline top1 ADE/FDE: "
        f"{original_metrics['top1_ade']:.3f}/{original_metrics['top1_fde']:.3f} m",
        flush=True,
    )
    generator = None  # all candidates and scene embeddings are now immutable tensors

    shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_cached,
        batch_size=args.batch_size,
        shuffle=True,
        generator=shuffle_generator,
    )
    run_dir = Path("runs") / args.run_name
    metrics_path = run_dir / "metrics.csv"
    curves_path = run_dir / "learning_curves.png"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []
    best_top1_ade = float("inf")
    fieldnames = [
        "epoch",
        "train_ce",
        "val_top1_ade",
        "val_top1_fde",
        "val_min_ade",
        "val_min_fde",
        "val_miss_rate",
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            reranker.train()
            losses: list[float] = []
            for scene, candidates, _, target in train_loader:
                scene = scene.to(device)
                candidates = candidates.to(device)
                target = target.to(device)
                displacement = torch.linalg.vector_norm(
                    candidates - target.unsqueeze(1), dim=-1
                )
                quality = displacement.mean(dim=-1) + 0.5 * displacement[:, :, -1]
                best_candidate = quality.argmin(dim=1)
                scores = reranker(scene, candidates)
                loss = criterion(scores, best_candidate)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))

            _, val_metrics = _calculate_metrics(
                reranker, val_cached, args.batch_size, device
            )
            row = {
                "epoch": epoch,
                "train_ce": float(np.mean(losses)),
                "val_top1_ade": val_metrics["top1_ade"],
                "val_top1_fde": val_metrics["top1_fde"],
                "val_min_ade": val_metrics["min_ade"],
                "val_min_fde": val_metrics["min_fde"],
                "val_miss_rate": val_metrics["miss_rate"],
            }
            rows.append(row)
            writer.writerow(row)
            csv_file.flush()
            print(
                f"Epoch {epoch:02d}/{args.epochs}: train_CE={row['train_ce']:.4f}, "
                f"top1 ADE/FDE={row['val_top1_ade']:.3f}/{row['val_top1_fde']:.3f} m, "
                f"minADE@K/minFDE@K={row['val_min_ade']:.3f}/{row['val_min_fde']:.3f} m",
                flush=True,
            )
            if row["val_top1_ade"] < best_top1_ade:
                best_top1_ade = row["val_top1_ade"]
                torch.save(
                    {
                        "model_state_dict": reranker.state_dict(),
                        "scene_dim": scene_dim,
                        "future_steps": int(
                            generator_checkpoint["model_config"]["future_steps"]
                        ),
                        "hidden_dim": args.reranker_hidden_dim,
                        "generator_checkpoint": str(args.generator_checkpoint),
                        "generator_epoch": int(generator_checkpoint["epoch"]),
                        "train_scenarios": len(train_cached),
                        "seed": args.seed,
                        "epoch": epoch,
                        "validation_metrics": val_metrics,
                    },
                    args.output,
                )
                print(f"  saved best top1 ADE checkpoint: {args.output}", flush=True)

    _save_curves(rows, curves_path)
    print(f"Saved metrics to {metrics_path}", flush=True)
    print(f"Saved learning curve to {curves_path}", flush=True)


if __name__ == "__main__":
    main()
