"""Train a focal-agent trajectory MLP on AV2 scenarios."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from motion_forecasting.datasets import TrajectoryDataset
from motion_forecasting.metrics import ade, fde
from motion_forecasting.models import TrajectoryMLP


def evaluate(model: TrajectoryMLP, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
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
    return float(np.mean(ade_scores)), float(np.mean(fde_scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/mlp.pt"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available in this PyTorch environment")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    train_data = TrajectoryDataset(args.train_data)
    val_data = TrajectoryDataset(
        args.val_data,
        past_steps=train_data.past_steps,
        future_steps=train_data.future_steps,
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    model = TrajectoryMLP(
        past_steps=train_data.past_steps,
        future_steps=train_data.future_steps,
        hidden_dim=args.hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()
    best_val_ade = float("inf")

    print(f"Training scenarios: {len(train_data)} (skipped {len(train_data.skipped_scenarios)})")
    print(f"Validation scenarios: {len(val_data)} (skipped {len(val_data.skipped_scenarios)})")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in train_loader:
            history = batch["history"].to(device)
            target = batch["future"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(history)
            loss = loss_function(prediction, target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_ade, val_fde = evaluate(model, val_loader, device)
        train_mse = float(np.mean(losses))
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train_mse={train_mse:.4f}, val_ADE={val_ade:.3f} m, val_FDE={val_fde:.3f} m"
        )
        if val_ade < best_val_ade:
            best_val_ade = val_ade
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "past_steps": train_data.past_steps,
                    "future_steps": train_data.future_steps,
                    "hidden_dim": args.hidden_dim,
                    "epoch": epoch,
                    "val_ade": val_ade,
                    "val_fde": val_fde,
                },
                args.output,
            )
            print(f"  saved best checkpoint: {args.output}")


if __name__ == "__main__":
    main()
