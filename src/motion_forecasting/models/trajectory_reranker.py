"""Candidate-conditioned confidence scorer for multimodal futures."""

from __future__ import annotations

import torch
from torch import nn


class TrajectoryAwareModeReranker(nn.Module):
    """Score each generated trajectory using its geometry and scene context."""

    def __init__(
        self,
        *,
        scene_dim: int = 384,
        future_steps: int = 60,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if min(scene_dim, future_steps, hidden_dim) < 1:
            raise ValueError("scene_dim, future_steps, and hidden_dim must be positive")
        self.scene_dim = scene_dim
        self.future_steps = future_steps
        self.hidden_dim = hidden_dim
        self.trajectory_encoder = nn.Sequential(
            nn.Linear(future_steps * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(scene_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2 or 1),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2 or 1, 1),
        )

    def forward(self, scene_embedding: torch.Tensor, trajectories: torch.Tensor) -> torch.Tensor:
        if scene_embedding.ndim != 2 or scene_embedding.shape[-1] != self.scene_dim:
            raise ValueError(
                f"scene_embedding must have shape (batch, {self.scene_dim}), "
                f"got {tuple(scene_embedding.shape)}"
            )
        if (
            trajectories.ndim != 4
            or trajectories.shape[0] != scene_embedding.shape[0]
            or trajectories.shape[2:] != (self.future_steps, 2)
            or trajectories.shape[1] < 1
        ):
            raise ValueError(
                "trajectories must have shape "
                f"(batch, modes, {self.future_steps}, 2), got {tuple(trajectories.shape)}"
            )
        batch_size, num_modes = trajectories.shape[:2]
        candidate_embeddings = self.trajectory_encoder(
            trajectories.reshape(batch_size * num_modes, self.future_steps * 2)
        ).reshape(batch_size, num_modes, self.hidden_dim)
        repeated_scene = scene_embedding.unsqueeze(1).expand(-1, num_modes, -1)
        scoring_input = torch.cat((repeated_scene, candidate_embeddings), dim=-1)
        return self.scorer(scoring_input).squeeze(-1)
