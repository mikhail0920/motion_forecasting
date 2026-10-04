"""Precompute nearby AV2 lane polylines into a compact NPZ cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from motion_forecasting.transforms import estimate_agent_angle, transform_to_agent_frame


def _resample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"expected lane centerline shaped (N, >=2), got {points.shape}")
    xy = points[:, :2]
    xy = xy[np.isfinite(xy).all(axis=1)]
    if not len(xy):
        raise ValueError("lane centerline has no finite XY coordinates")
    if len(xy) == 1:
        return np.repeat(xy, count, axis=0)

    lengths = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    keep = np.concatenate(([True], lengths > 1e-8))
    xy = xy[keep]
    if len(xy) == 1:
        return np.repeat(xy, count, axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))))
    if cumulative[-1] <= 1e-8:
        return np.repeat(xy[:1], count, axis=0)
    sample_distances = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        [np.interp(sample_distances, cumulative, xy[:, axis]) for axis in range(2)]
    )


def _point_to_polyline_distance(point: np.ndarray, polyline: np.ndarray) -> float:
    starts = polyline[:-1]
    vectors = polyline[1:] - starts
    lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
    fractions = np.divide(
        np.einsum("ij,ij->i", point - starts, vectors),
        lengths_sq,
        out=np.zeros_like(lengths_sq),
        where=lengths_sq > 1e-12,
    )
    projections = starts + np.clip(fractions, 0.0, 1.0)[:, None] * vectors
    return float(np.linalg.norm(projections - point, axis=1).min())


def _scenario_lanes(
    map_path: Path,
    current_position: np.ndarray,
    angle: float,
    *,
    max_lanes: int,
    points_per_lane: int,
    route_length_meters: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from av2.map.map_api import ArgoverseStaticMap
    except ImportError as exc:
        raise RuntimeError(
            "Map preprocessing requires the official AV2 API. Install it with: "
            'python -m pip install -e ".[maps]"'
        ) from exc

    static_map = ArgoverseStaticMap.from_json(map_path)
    lane_ids = static_map.get_scenario_lane_segment_ids()
    centerlines: dict[int, np.ndarray] = {}
    for lane_id in lane_ids:
        centerline = np.asarray(static_map.get_lane_segment_centerline(lane_id), dtype=np.float64)
        if centerline.ndim != 2 or centerline.shape[1] < 2 or len(centerline) < 2:
            continue
        try:
            resampled = _resample_polyline(centerline, points_per_lane)
        except ValueError:
            continue
        centerlines[int(lane_id)] = resampled

    candidates: list[tuple[float, int, np.ndarray]] = []
    if route_length_meters is None:
        for lane_id, centerline in centerlines.items():
            candidates.append(
                (_point_to_polyline_distance(current_position, centerline), lane_id, centerline)
            )
    else:
        # Follow map topology from each nearby lane. At junctions, enumerate
        # successor branches so each candidate is a connected route, rather
        # than one short segment that the decoder must extrapolate beyond.
        segment_lengths = {
            lane_id: float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
            for lane_id, points in centerlines.items()
        }
        successors = {
            int(lane_id): sorted(
                int(next_id)
                for next_id in static_map.vector_lane_segments[lane_id].successors or []
                if int(next_id) in centerlines
            )
            for lane_id in centerlines
        }

        def route_paths(start_id: int) -> list[np.ndarray]:
            found: list[np.ndarray] = []
            stack: list[tuple[list[int], float]] = [([start_id], segment_lengths[start_id])]
            while stack and len(found) < 16:
                path, length = stack.pop()
                next_ids = [
                    next_id
                    for next_id in successors[path[-1]]
                    if next_id not in path
                ]
                if length >= route_length_meters or not next_ids:
                    found.append(np.concatenate(
                        [centerlines[lane_id] if index == 0 else centerlines[lane_id][1:]
                         for index, lane_id in enumerate(path)], axis=0
                    ))
                    continue
                for next_id in reversed(next_ids):
                    stack.append((path + [next_id], length + segment_lengths[next_id]))
            return found

        start_candidates = sorted(
            (
                _point_to_polyline_distance(current_position, points),
                lane_id,
            )
            for lane_id, points in centerlines.items()
        )[: max(max_lanes, 1)]
        for _, lane_id in start_candidates:
            for route in route_paths(lane_id):
                # A heading penalty prevents a nearby parallel/opposite lane
                # from ranking above a plausible forward route.
                relative = route - current_position
                ahead = float(np.max(relative @ np.array([
                    np.cos(angle), np.sin(angle),
                ])))
                behind = float(np.min(relative @ np.array([
                    np.cos(angle), np.sin(angle),
                ])))
                if ahead < 0.0 or behind > 35.0:
                    continue
                distance = _point_to_polyline_distance(current_position, route)
                forward_penalty = max(0.0, -behind - 3.0) * 0.25
                candidates.append((distance + forward_penalty, lane_id, route))
    candidates.sort(key=lambda item: (item[0], item[1]))

    lanes = np.zeros((max_lanes, points_per_lane, 4), dtype=np.float32)
    mask = np.zeros(max_lanes, dtype=np.bool_)
    lane_ids = np.full(max_lanes, -1, dtype=np.int64)
    for lane_index, (_, lane_id, centerline) in enumerate(candidates[:max_lanes]):
        local = _resample_polyline(centerline, points_per_lane)
        local, _, _ = transform_to_agent_frame(local, origin=current_position, angle=angle)
        lanes[lane_index, :, :2] = local.astype(np.float32)
        lanes[lane_index, 1:, 2:4] = np.diff(local, axis=0).astype(np.float32)
        mask[lane_index] = True
        lane_ids[lane_index] = lane_id
    return lanes, mask, lane_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-lanes", type=int, default=16)
    parser.add_argument("--points-per-lane", type=int, default=20)
    parser.add_argument(
        "--route-length-meters",
        type=float,
        help="Build connected successor routes to this approximate length instead of individual segments.",
    )
    parser.add_argument("--max-scenarios", type=int)
    args = parser.parse_args()
    if args.max_lanes < 1 or args.points_per_lane < 2:
        parser.error("max-lanes must be positive and points-per-lane must be at least 2")
    if args.route_length_meters is not None and args.route_length_meters <= 0:
        parser.error("route-length-meters must be positive")
    if args.max_scenarios is not None and args.max_scenarios < 1:
        parser.error("max-scenarios must be positive when specified")
    if not args.data.is_dir():
        parser.error(f"scenario directory not found: {args.data}")

    paths = sorted(args.data.rglob("scenario_*.parquet"))
    if not paths:
        parser.error(f"no scenario_*.parquet files found under {args.data}")
    if args.max_scenarios is not None:
        paths = paths[: args.max_scenarios]

    # Import lazily so the command can produce a useful install message without
    # making the base forecasting package depend on all AV2 API requirements.
    try:
        import pandas as pd
        from av2.map.map_api import ArgoverseStaticMap  # noqa: F401
    except ImportError as exc:
        parser.error(
            'map preprocessing requires AV2 API and pandas; run "python -m pip install -e .[maps]" '
            f"first ({exc})"
        )

    scenario_ids: list[str] = []
    all_lanes: list[np.ndarray] = []
    all_masks: list[np.ndarray] = []
    all_lane_ids: list[np.ndarray] = []
    for index, scenario_path in enumerate(paths, start=1):
        scenario_id = scenario_path.stem.removeprefix("scenario_")
        map_path = scenario_path.parent / f"log_map_archive_{scenario_id}.json"
        if not map_path.is_file():
            raise FileNotFoundError(
                f"Map JSON missing for scenario {scenario_id}: {map_path}. "
                "Download scenarios with scripts/download_av2_subset.py --include-maps."
            )
        frame = pd.read_parquet(
            scenario_path,
            columns=["track_id", "focal_track_id", "timestep", "position_x", "position_y", "observed"],
        )
        focal_id = frame["focal_track_id"].iloc[0]
        focal = frame.loc[(frame["track_id"] == focal_id) & frame["observed"].astype(bool)]
        focal = focal.sort_values("timestep")
        observed_xy = focal[["position_x", "position_y"]].to_numpy(dtype=np.float64)
        if len(observed_xy) < 2 or not np.isfinite(observed_xy).all():
            raise ValueError(f"Scenario {scenario_id} has an invalid focal history")
        current_position = observed_xy[-1]
        angle = estimate_agent_angle(observed_xy)
        lanes, mask, lane_ids = _scenario_lanes(
            map_path,
            current_position,
            angle,
            max_lanes=args.max_lanes,
            points_per_lane=args.points_per_lane,
            route_length_meters=args.route_length_meters,
        )
        scenario_ids.append(scenario_id)
        all_lanes.append(lanes)
        all_masks.append(mask)
        all_lane_ids.append(lane_ids)
        if index % 100 == 0 or index == len(paths):
            print(f"  {index:,}/{len(paths):,} maps precomputed", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        scenario_ids=np.asarray(scenario_ids),
        lanes=np.stack(all_lanes),
        lane_mask=np.stack(all_masks),
        lane_ids=np.stack(all_lane_ids),
        max_lanes=np.asarray(args.max_lanes),
        points_per_lane=np.asarray(args.points_per_lane),
    )
    print(f"Saved map tensors for {len(scenario_ids):,} scenarios to {args.output}")


if __name__ == "__main__":
    main()
