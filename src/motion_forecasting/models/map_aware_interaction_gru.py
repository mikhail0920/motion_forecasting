"""Social-attention GRU augmented with learned attention over lane geometry."""

from __future__ import annotations

import torch
from torch import nn


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=scores.device, dtype=torch.bool)
    any_valid = mask.any(dim=1, keepdim=True)
    masked_scores = scores.masked_fill(~mask, -torch.inf)
    masked_scores = torch.where(any_valid, masked_scores, torch.zeros_like(masked_scores))
    weights = torch.softmax(masked_scores, dim=1) * mask.to(scores.dtype)
    return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)


class MapAwareInteractionGRU(nn.Module):
    """Encode focal, nearby actors, and lane polylines to predict a trajectory."""

    def __init__(
        self,
        *,
        input_dim: int = 4,
        neighbor_dim: int = 5,
        lane_dim: int = 4,
        hidden_dim: int = 128,
        future_steps: int = 60,
        max_neighbors: int = 8,
        max_lanes: int = 16,
        points_per_lane: int = 20,
    ) -> None:
        super().__init__()
        if min(
            input_dim,
            neighbor_dim,
            lane_dim,
            hidden_dim,
            future_steps,
            max_neighbors,
            max_lanes,
            points_per_lane,
        ) < 1:
            raise ValueError("all model dimensions must be positive")
        self.input_dim = input_dim
        self.neighbor_dim = neighbor_dim
        self.lane_dim = lane_dim
        self.hidden_dim = hidden_dim
        self.future_steps = future_steps
        self.max_neighbors = max_neighbors
        self.max_lanes = max_lanes
        self.points_per_lane = points_per_lane

        self.focal_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.neighbor_gru = nn.GRU(neighbor_dim, hidden_dim, batch_first=True)
        self.neighbor_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.lane_point_mlp = nn.Sequential(
            nn.Linear(lane_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.map_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_steps * 2),
        )

    def forward(
        self,
        history: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
        lanes: torch.Tensor,
        lane_mask: torch.Tensor,
        *,
        return_attentions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scene_embedding, social_weights, map_weights = self.encode_scene(
            history, neighbors, neighbor_mask, lanes, lane_mask
        )
        future = self.decoder(scene_embedding).view(
            history.shape[0], self.future_steps, 2
        )
        if return_attentions:
            return future, social_weights, map_weights
        return future

    def encode_scene(
        self,
        history: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
        lanes: torch.Tensor,
        lane_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the unchanged focal/social/map representation and attentions."""
        batch_size = history.shape[0]
        if history.ndim != 3 or history.shape[-1] != self.input_dim:
            raise ValueError(
                f"history must have shape (batch, timesteps, {self.input_dim}), "
                f"got {tuple(history.shape)}"
            )
        if (
            neighbors.ndim != 4
            or neighbors.shape[:2] != (batch_size, self.max_neighbors)
            or neighbors.shape[2] != history.shape[1]
            or neighbors.shape[-1] != self.neighbor_dim
        ):
            raise ValueError(
                "neighbors must have shape "
                f"(batch, {self.max_neighbors}, timesteps, {self.neighbor_dim}), "
                f"got {tuple(neighbors.shape)}"
            )
        if neighbor_mask.shape != (batch_size, self.max_neighbors):
            raise ValueError(
                f"neighbor_mask must have shape (batch, {self.max_neighbors}), "
                f"got {tuple(neighbor_mask.shape)}"
            )
        if (
            lanes.ndim != 4
            or lanes.shape[:2] != (batch_size, self.max_lanes)
            or lanes.shape[2] != self.points_per_lane
            or lanes.shape[-1] != self.lane_dim
        ):
            raise ValueError(
                "lanes must have shape "
                f"(batch, {self.max_lanes}, {self.points_per_lane}, {self.lane_dim}), "
                f"got {tuple(lanes.shape)}"
            )
        if lane_mask.shape != (batch_size, self.max_lanes):
            raise ValueError(
                f"lane_mask must have shape (batch, {self.max_lanes}), "
                f"got {tuple(lane_mask.shape)}"
            )

        _, focal_hidden = self.focal_gru(history)
        focal_embedding = focal_hidden[-1]

        _, neighbor_hidden = self.neighbor_gru(
            neighbors.reshape(
                batch_size * self.max_neighbors,
                history.shape[1],
                self.neighbor_dim,
            )
        )
        neighbor_embeddings = neighbor_hidden[-1].reshape(
            batch_size, self.max_neighbors, self.hidden_dim
        )
        repeated_focal = focal_embedding.unsqueeze(1).expand(-1, self.max_neighbors, -1)
        social_scores = self.neighbor_attention(
            torch.cat((repeated_focal, neighbor_embeddings), dim=-1)
        ).squeeze(-1)
        social_weights = _masked_softmax(social_scores, neighbor_mask)
        social_context = (neighbor_embeddings * social_weights.unsqueeze(-1)).sum(dim=1)

        lane_count = lanes.shape[1]
        point_embeddings = self.lane_point_mlp(
            lanes.reshape(batch_size * lane_count, self.points_per_lane, self.lane_dim)
        )
        lane_embeddings = point_embeddings.max(dim=1).values.reshape(
            batch_size, lane_count, self.hidden_dim
        )
        repeated_map_focal = focal_embedding.unsqueeze(1).expand(-1, lane_count, -1)
        map_scores = self.map_attention(
            torch.cat((repeated_map_focal, lane_embeddings), dim=-1)
        ).squeeze(-1)
        map_weights = _masked_softmax(map_scores, lane_mask)
        map_context = (lane_embeddings * map_weights.unsqueeze(-1)).sum(dim=1)

        scene_embedding = torch.cat((focal_embedding, social_context, map_context), dim=-1)
        return scene_embedding, social_weights, map_weights
