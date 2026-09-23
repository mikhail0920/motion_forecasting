"""GRU forecaster with attention over neighbors and optional physics features."""

from __future__ import annotations

import torch
from torch import nn


class InteractionTrajectoryGRU(nn.Module):
    """Encode focal and neighbor tracks, attend to relevant neighbors, forecast."""

    def __init__(
        self,
        *,
        input_dim: int = 4,
        neighbor_dim: int = 5,
        hidden_dim: int = 128,
        future_steps: int = 60,
        max_neighbors: int = 8,
        interaction_features: bool = False,
        timestep_seconds: float = 0.1,
    ) -> None:
        super().__init__()
        if min(input_dim, neighbor_dim, hidden_dim, future_steps, max_neighbors) < 1:
            raise ValueError("all model dimensions must be positive")
        if timestep_seconds <= 0:
            raise ValueError("timestep_seconds must be positive")
        self.input_dim = input_dim
        self.neighbor_dim = neighbor_dim
        self.hidden_dim = hidden_dim
        self.future_steps = future_steps
        self.max_neighbors = max_neighbors
        self.use_interaction_features = interaction_features
        self.timestep_seconds = timestep_seconds

        self.focal_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.neighbor_gru = nn.GRU(neighbor_dim, hidden_dim, batch_first=True)
        score_input_dim = hidden_dim * 2 + (7 if interaction_features else 0)
        self.attention = nn.Sequential(
            nn.Linear(score_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, future_steps * 2),
        )

    def _physics_features(
        self, history: torch.Tensor, neighbors: torch.Tensor
    ) -> torch.Tensor:
        """Return [rel x/y, rel vx/vy, distance, closing speed, CPA time]."""
        relative_position = neighbors[:, :, -1, :2]
        focal_velocity = history[:, -1, 2:4] / self.timestep_seconds

        # Estimate each neighbor's latest velocity from its two latest valid
        # states. This handles actors that appear after the focal history begins.
        valid = neighbors[:, :, :, 4] > 0.5
        time_index = torch.arange(
            neighbors.shape[2], device=neighbors.device
        ).view(1, 1, -1)
        previous_index = torch.where(valid & (time_index < neighbors.shape[2] - 1), time_index, -1)
        previous_index = previous_index.max(dim=-1).values
        previous_index = previous_index.clamp_min(0)
        gather_index = previous_index[..., None, None].expand(-1, -1, 1, 2)
        previous_position = neighbors[:, :, :, :2].gather(2, gather_index).squeeze(2)
        elapsed_steps = (neighbors.shape[2] - 1 - previous_index).clamp_min(1).to(neighbors.dtype)
        neighbor_velocity = (
            relative_position - previous_position
        ) / (elapsed_steps.unsqueeze(-1) * self.timestep_seconds)
        neighbor_has_previous = (valid & (time_index < neighbors.shape[2] - 1)).any(dim=-1)
        neighbor_velocity = neighbor_velocity * neighbor_has_previous.unsqueeze(-1)

        relative_velocity = neighbor_velocity - focal_velocity.unsqueeze(1)
        distance = torch.linalg.vector_norm(relative_position, dim=-1, keepdim=True)
        closing_speed = -(
            relative_position * relative_velocity
        ).sum(dim=-1, keepdim=True) / distance.clamp_min(1e-6)
        relative_speed_sq = relative_velocity.square().sum(dim=-1, keepdim=True)
        time_to_cpa = -(
            relative_position * relative_velocity
        ).sum(dim=-1, keepdim=True) / relative_speed_sq.clamp_min(1e-6)
        time_to_cpa = time_to_cpa.clamp(min=0.0, max=60.0)
        time_to_cpa = torch.where(relative_speed_sq > 1e-6, time_to_cpa, 0.0)
        return torch.cat(
            (relative_position, relative_velocity, distance, closing_speed, time_to_cpa),
            dim=-1,
        )

    def forward(
        self,
        history: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
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
        neighbor_sequence = neighbors.reshape(
            batch_size * neighbor_count, timesteps, feature_dim
        )
        _, neighbor_hidden = self.neighbor_gru(neighbor_sequence)
        neighbor_embeddings = neighbor_hidden[-1].reshape(
            batch_size, neighbor_count, self.hidden_dim
        )
        focal_embedding = focal_hidden[-1]
        score_inputs = [
            focal_embedding.unsqueeze(1).expand(-1, neighbor_count, -1),
            neighbor_embeddings,
        ]
        if self.use_interaction_features:
            score_inputs.append(self._physics_features(history, neighbors))
        scores = self.attention(torch.cat(score_inputs, dim=-1)).squeeze(-1)

        mask = neighbor_mask.to(device=scores.device, dtype=torch.bool)
        any_neighbor = mask.any(dim=1, keepdim=True)
        masked_scores = scores.masked_fill(~mask, -torch.inf)
        masked_scores = torch.where(any_neighbor, masked_scores, torch.zeros_like(masked_scores))
        weights = torch.softmax(masked_scores, dim=1) * mask.to(scores.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        social_context = (neighbor_embeddings * weights.unsqueeze(-1)).sum(dim=1)

        prediction = self.decoder(torch.cat((focal_embedding, social_context), dim=-1))
        prediction = prediction.view(batch_size, self.future_steps, 2)
        if return_attention:
            return prediction, weights
        return prediction
