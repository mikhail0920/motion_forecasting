"""A feed-forward model that predicts an entire future trajectory at once."""

from __future__ import annotations

import torch
from torch import nn


class TrajectoryMLP(nn.Module):
    """Map a fixed-length relative history to a fixed-length future path."""

    def __init__(
        self,
        past_steps: int = 50,
        future_steps: int = 60,
        hidden_dim: int = 256,
        input_dim: int = 4,
    ) -> None:
        super().__init__()
        if past_steps < 1 or future_steps < 1 or hidden_dim < 1 or input_dim < 1:
            raise ValueError("all model dimensions must be positive")
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.input_dim = input_dim
        self.network = nn.Sequential(
            nn.Linear(past_steps * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_steps * 2),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[1:] != (self.past_steps, self.input_dim):
            raise ValueError(
                f"history must have shape (batch, {self.past_steps}, {self.input_dim}), "
                f"got {tuple(history.shape)}"
            )
        prediction = self.network(history.flatten(start_dim=1))
        return prediction.view(-1, self.future_steps, 2)
