"""K-hypothesis decoder on the map-aware interaction GRU scene encoder."""

from __future__ import annotations

import torch
from torch import nn

from motion_forecasting.models.map_aware_interaction_gru import MapAwareInteractionGRU


class MultimodalForecaster(MapAwareInteractionGRU):
    """Predict multiple futures and a categorical probability for each future."""

    def __init__(self, *, num_modes: int = 6, **kwargs: int) -> None:
        if num_modes < 1:
            raise ValueError("num_modes must be positive")
        super().__init__(**kwargs)
        self.num_modes = num_modes
        # Keep every focal/social/map encoder layer identical to the single-path model.
        # The inherited single-trajectory decoder is unused by this variant.
        self.decoder = None
        self.mode_embeddings = nn.Parameter(torch.empty(num_modes, self.hidden_dim))
        nn.init.normal_(self.mode_embeddings, mean=0.0, std=0.02)
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.future_steps * 2),
        )
        self.probability_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, num_modes),
        )

    def forward(
        self,
        history: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
        lanes: torch.Tensor,
        lane_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scene_embedding, _, _ = self.encode_scene(
            history, neighbors, neighbor_mask, lanes, lane_mask
        )
        batch_size = history.shape[0]
        mode_embeddings = self.mode_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        repeated_scene = scene_embedding.unsqueeze(1).expand(-1, self.num_modes, -1)
        mode_input = torch.cat((repeated_scene, mode_embeddings), dim=-1)
        trajectories = self.trajectory_decoder(mode_input).view(
            batch_size, self.num_modes, self.future_steps, 2
        )
        logits = self.probability_head(scene_embedding)
        return trajectories, logits
