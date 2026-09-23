"""GRU trajectory forecaster with a pooled social context encoder."""

from __future__ import annotations

import torch
from torch import nn


class SocialTrajectoryGRU(nn.Module):
    """Encode focal and neighboring histories, then predict the full future."""

    def __init__(
        self,
        *,
        input_dim: int = 4,
        neighbor_dim: int = 5,
        hidden_dim: int = 128,
        future_steps: int = 60,
        max_neighbors: int = 8,
    ) -> None:
        super().__init__()
        if min(input_dim, neighbor_dim, hidden_dim, future_steps, max_neighbors) < 1:
            raise ValueError("all model dimensions must be positive")
        self.input_dim = input_dim
        self.neighbor_dim = neighbor_dim
        self.hidden_dim = hidden_dim
        self.future_steps = future_steps
        self.max_neighbors = max_neighbors
        self.focal_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.neighbor_gru = nn.GRU(neighbor_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_steps * 2),
        )

    def forward(
        self,
        history: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
    ) -> torch.Tensor:
        if history.ndim != 3 or history.shape[-1] != self.input_dim:
            raise ValueError(
                f"history must have shape (batch, timesteps, {self.input_dim}), "
                f"got {tuple(history.shape)}"
            )
        if (
            neighbors.ndim != 4
            or neighbors.shape[0] != history.shape[0]
            or neighbors.shape[1] != self.max_neighbors
            or neighbors.shape[2] != history.shape[1]
            or neighbors.shape[-1] != self.neighbor_dim
        ):
            raise ValueError(
                "neighbors must have shape "
                f"(batch, {self.max_neighbors}, timesteps, {self.neighbor_dim}), "
                f"got {tuple(neighbors.shape)}"
            )
        if neighbor_mask.shape != (history.shape[0], self.max_neighbors):
            raise ValueError(
                f"neighbor_mask must have shape (batch, {self.max_neighbors}), "
                f"got {tuple(neighbor_mask.shape)}"
            )

        _, focal_hidden = self.focal_gru(history)
        batch_size, neighbor_count, timesteps, feature_dim = neighbors.shape
        neighbor_sequence = neighbors.reshape(batch_size * neighbor_count, timesteps, feature_dim)
        _, neighbor_hidden = self.neighbor_gru(neighbor_sequence)
        neighbor_embeddings = neighbor_hidden[-1].reshape(
            batch_size, neighbor_count, self.hidden_dim
        )

        mask = neighbor_mask.to(
            device=neighbor_embeddings.device, dtype=neighbor_embeddings.dtype
        ).unsqueeze(-1)
        pooled = (neighbor_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        combined = torch.cat((focal_hidden[-1], pooled), dim=-1)
        future = self.head(combined)
        return future.view(batch_size, self.future_steps, 2)
