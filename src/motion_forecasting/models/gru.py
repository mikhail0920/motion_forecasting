"""Sequence model for focal-agent trajectory forecasting."""

from __future__ import annotations

import torch
from torch import nn


class TrajectoryGRU(nn.Module):
    """Encode the full observed sequence and decode all future points at once."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        future_steps: int = 60,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or future_steps < 1:
            raise ValueError("input_dim, hidden_dim, and future_steps must be positive")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.future_steps = future_steps
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_steps * 2),
        )

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        if history.ndim != 3 or history.shape[2] != self.input_dim:
            raise ValueError(
                f"history must have shape (batch, timesteps, {self.input_dim}), "
                f"got {tuple(history.shape)}"
            )
        _, hidden = self.gru(history)
        future = self.head(hidden[-1])
        return future.view(history.shape[0], self.future_steps, 2)
