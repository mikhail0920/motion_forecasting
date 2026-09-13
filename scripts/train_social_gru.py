"""Train a focal-plus-neighbors social GRU on AV2 motion scenarios."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from motion_forecasting.datasets import SocialTrajectoryDataset
from motion_forecasting.metrics import ade, fde
from motion_forecasting.models import SocialTrajectoryGRU


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _evaluate(
    model: SocialTrajectoryGRU,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
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
    return float(np.mean(ade_scores)), float(np.mean(fde_scores))


def _save_learning_curves(rows: list[dict[str, float]], path: Path) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, column, title, ylabel in zip(
        axes,
        ("train_loss", "val_ade", "val_fde"),
        ("Training MSE", "Validation ADE", "Validation FDE"),
        ("MSE", "Meters", "Meters"),
    ):
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
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-neighbors", type=int, default=8)
    parser.add_argument("--train-scenarios", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="social-gru")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/social_gru.pt"))
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.learning_rate <= 0
        or args.hidden_dim < 1
        or args.max_neighbors < 1
        or args.num_workers < 0
    ):
        parser.error("epochs, batch-size, hidden-dim, max-neighbors and learning-rate must be positive")
    if args.train_scenarios is not None and args.train_scenarios < 1:
        parser.error("train-scenarios must be positive")
    if not args.run_name.strip() or Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        parser.error("run-name must be a non-empty directory name without path components")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available in this PyTorch environment")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)

    train_data = SocialTrajectoryDataset(
        args.train_data,
        max_neighbors=args.max_neighbors,
        max_scenarios=args.train_scenarios,
    )
    val_data = SocialTrajectoryDataset(
        args.val_data,
        past_steps=train_data.past_steps,
        future_steps=train_data.future_steps,
        max_neighbors=args.max_neighbors,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "num_workers": args.num_workers,
        "worker_init_fn": _seed_worker if args.num_workers else None,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, **loader_options)
    model = SocialTrajectoryGRU(
        input_dim=train_data.input_dim,
        neighbor_dim=train_data.neighbor_dim,
        hidden_dim=args.hidden_dim,
        future_steps=train_data.future_steps,
        max_neighbors=args.max_neighbors,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()
    best_val_ade = float("inf")
    run_dir = Path("runs") / args.run_name
    metrics_csv = run_dir / "metrics.csv"
    curves_path = run_dir / "learning_curves.png"
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []

    print(f"Training scenarios: {len(train_data)} (skipped {len(train_data.skipped_scenarios)})")
    print(f"Validation scenarios: {len(val_data)} (skipped {len(val_data.skipped_scenarios)})")
    with metrics_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["epoch", "train_loss", "val_ade", "val_fde"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses: list[float] = []
            for batch in train_loader:
                history = batch["history"].to(device)
                neighbors = batch["neighbors"].to(device)
                neighbor_mask = batch["neighbor_mask"].to(device)
                target = batch["future"].to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(history, neighbors, neighbor_mask)
                loss = loss_function(prediction, target)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            val_ade, val_fde = _evaluate(model, val_loader, device)
            train_loss = float(np.mean(losses))
            row = {"epoch": epoch, "train_loss": train_loss, "val_ade": val_ade, "val_fde": val_fde}
            rows.append(row)
            writer.writerow(row)
            csv_file.flush()
            print(
                f"Epoch {epoch:02d}/{args.epochs}: train_mse={train_loss:.4f}, "
                f"val_ADE={val_ade:.3f} m, val_FDE={val_fde:.3f} m"
            )
            if val_ade < best_val_ade:
                best_val_ade = val_ade
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "input_dim": train_data.input_dim,
                        "neighbor_dim": train_data.neighbor_dim,
                        "hidden_dim": args.hidden_dim,
                        "future_steps": train_data.future_steps,
                        "past_steps": train_data.past_steps,
                        "max_neighbors": args.max_neighbors,
                        "representation": "agent-centric",
                        "train_scenarios": len(train_data),
                        "seed": args.seed,
                        "epoch": epoch,
                        "val_ade": val_ade,
                        "val_fde": val_fde,
                    },
                    args.output,
                )
                print(f"  saved best checkpoint: {args.output}")

    _save_learning_curves(rows, curves_path)
    print(f"Saved per-epoch metrics to {metrics_csv}")
    print(f"Saved learning curve to {curves_path}")


if __name__ == "__main__":
    main()
