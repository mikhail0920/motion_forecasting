"""Lane-conditioned multimodal decoder with route and free hypotheses."""

from __future__ import annotations

import torch
from torch import nn

from motion_forecasting.models.multimodal_forecaster import MultimodalForecaster


class LaneConditionedForecaster(MultimodalForecaster):
    """Decode lane-anchored futures plus free modes from the existing scene encoder."""

    def __init__(
        self,
        *,
        num_modes: int = 6,
        num_lane_modes: int = 4,
        sampling_interval: float = 0.1,
        **kwargs: int,
    ) -> None:
        if num_modes < 1 or num_lane_modes < 0 or num_lane_modes > num_modes:
            raise ValueError("num_modes must be positive and num_lane_modes in [0, num_modes]")
        if sampling_interval <= 0:
            raise ValueError("sampling_interval must be positive")
        super().__init__(num_modes=num_modes, **kwargs)
        if num_lane_modes > self.max_lanes:
            raise ValueError("num_lane_modes cannot exceed max_lanes")
        self.num_lane_modes = num_lane_modes
        self.sampling_interval = sampling_interval
        self.trajectory_decoder = nn.Sequential(
            nn.Linear(
                self.hidden_dim * 5 + self.points_per_lane * 2,
                self.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.future_steps * 2),
        )

    def _lane_anchor(
        self,
        history: torch.Tensor,
        lane_xy: torch.Tensor,
    ) -> torch.Tensor:
        """Sample a route from the current lane projection at focal speed."""
        batch_size, lane_count, point_count, _ = lane_xy.shape
        if lane_count == 0 or self.future_steps == 0:
            return lane_xy.new_zeros((batch_size, lane_count, self.future_steps, 2))

        vectors = lane_xy[:, :, 1:] - lane_xy[:, :, :-1]
        lengths = torch.linalg.vector_norm(vectors, dim=-1)
        cumulative = torch.cat(
            (lengths.new_zeros((batch_size, lane_count, 1)), lengths.cumsum(dim=-1)),
            dim=-1,
        )

        starts = lane_xy[:, :, :-1]
        denom = lengths.square().clamp_min(1e-9)
        projection_fraction = (
            (-starts * vectors).sum(dim=-1) / denom
        ).clamp(0.0, 1.0)
        projections = starts + projection_fraction.unsqueeze(-1) * vectors
        nearest_segment = torch.linalg.vector_norm(projections, dim=-1).argmin(dim=-1)
        gather_segment = nearest_segment.unsqueeze(-1)
        current_progress = cumulative[:, :, :-1].gather(-1, gather_segment).squeeze(-1)
        current_progress = current_progress + (
            projection_fraction.gather(-1, gather_segment).squeeze(-1)
            * lengths.gather(-1, gather_segment).squeeze(-1)
        )

        history_velocity = history[:, -min(5, history.shape[1]) :, 2:4]
        speed = torch.linalg.vector_norm(history_velocity, dim=-1).mean(dim=-1) / self.sampling_interval
        speed = speed.clamp(min=0.5, max=20.0)
        future_time = (
            torch.arange(
                1,
                self.future_steps + 1,
                device=history.device,
                dtype=history.dtype,
            )
            * self.sampling_interval
        )
        target_progress = (
            current_progress.unsqueeze(-1)
            + speed[:, None, None] * future_time[None, None, :]
        )

        first_larger = target_progress.unsqueeze(-1) <= cumulative.unsqueeze(-2)
        upper_index = first_larger.to(torch.int64).argmax(dim=-1)
        past_end = target_progress > cumulative[:, :, -1:].expand_as(target_progress)
        upper_index = torch.where(
            past_end,
            torch.full_like(upper_index, point_count - 1),
            upper_index,
        ).clamp(min=1, max=point_count - 1)
        lower_index = upper_index - 1

        def gather_points(index: torch.Tensor) -> torch.Tensor:
            return lane_xy.gather(
                2, index.unsqueeze(-1).expand(-1, -1, -1, 2)
            )

        start = gather_points(lower_index)
        end = gather_points(upper_index)
        start_distance = cumulative.gather(-1, lower_index)
        segment_length = lengths.gather(-1, lower_index).clamp_min(1e-9)
        fraction = ((target_progress - start_distance) / segment_length).clamp_min(0.0)
        return start + fraction.unsqueeze(-1) * (end - start)

    def generate_with_context(
        self,
        history: torch.Tensor,
        neighbors: torch.Tensor,
        neighbor_mask: torch.Tensor,
        lanes: torch.Tensor,
        lane_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        (
            focal_embedding,
            social_context,
            map_context,
            _,
            map_weights,
            lane_embeddings,
        ) = self.encode_scene_components(history, neighbors, neighbor_mask, lanes, lane_mask)
        batch_size = history.shape[0]
        scene_embedding = torch.cat(
            (focal_embedding, social_context, map_context), dim=-1
        )

        if self.num_lane_modes:
            lane_scores = map_weights.masked_fill(~lane_mask.to(torch.bool), -1.0)
            route_indices = lane_scores.topk(self.num_lane_modes, dim=1).indices
            route_mask = lane_mask.gather(1, route_indices).to(torch.bool)
            route_embeddings = lane_embeddings.gather(
                1,
                route_indices.unsqueeze(-1).expand(
                    -1, -1, self.hidden_dim
                ),
            )
            route_xy = lanes[:, :, :, :2].gather(
                1,
                route_indices[:, :, None, None].expand(
                    -1, -1, self.points_per_lane, 2
                ),
            )
            route_embeddings = route_embeddings * route_mask.unsqueeze(-1)
            route_xy = route_xy * route_mask[:, :, None, None]
            route_anchors = self._lane_anchor(history, route_xy)
            route_anchors = route_anchors * route_mask[:, :, None, None]
            route_geometry = route_xy.flatten(start_dim=2)
        else:
            route_mask = lane_mask.new_zeros((batch_size, 0), dtype=torch.bool)
            route_embeddings = lane_embeddings.new_zeros(
                (batch_size, 0, self.hidden_dim)
            )
            route_geometry = lanes.new_zeros((batch_size, 0, self.points_per_lane * 2))
            route_anchors = lanes.new_zeros(
                (batch_size, 0, self.future_steps, 2)
            )

        free_modes = self.num_modes - self.num_lane_modes
        free_embeddings = lane_embeddings.new_zeros(
            (batch_size, free_modes, self.hidden_dim)
        )
        free_geometry = lanes.new_zeros(
            (batch_size, free_modes, self.points_per_lane * 2)
        )
        free_anchors = lanes.new_zeros(
            (batch_size, free_modes, self.future_steps, 2)
        )
        candidate_lane_embeddings = torch.cat((route_embeddings, free_embeddings), dim=1)
        candidate_geometry = torch.cat((route_geometry, free_geometry), dim=1)
        anchors = torch.cat((route_anchors, free_anchors), dim=1)
        modes = self.mode_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        repeated_scene = scene_embedding.unsqueeze(1).expand(-1, self.num_modes, -1)
        decoder_input = torch.cat(
            (repeated_scene, candidate_lane_embeddings, modes, candidate_geometry),
            dim=-1,
        )
        residual = self.trajectory_decoder(decoder_input).view(
            batch_size, self.num_modes, self.future_steps, 2
        )
        trajectories = anchors + residual
        logits = self.probability_head(scene_embedding)
        return trajectories, logits, scene_embedding

