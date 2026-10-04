"""Produce per-scenario diagnostics for the held-out turning subset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from motion_forecasting.datasets import MapContextDataset, SocialTrajectoryDataset
from motion_forecasting.models import LaneConditionedForecaster, TrajectoryAwareModeReranker


def predict(generator_path: Path, dataset: MapContextDataset, device: torch.device,
            reranker_path: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    state = torch.load(generator_path, map_location="cpu", weights_only=False)
    model = LaneConditionedForecaster(**state["model_config"])
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    reranker = None
    if reranker_path:
        rstate = torch.load(reranker_path, map_location="cpu", weights_only=False)
        reranker = TrajectoryAwareModeReranker(
            scene_dim=int(rstate["scene_dim"]), future_steps=int(rstate["future_steps"]),
            hidden_dim=int(rstate["hidden_dim"]),
        )
        reranker.load_state_dict(rstate["model_state_dict"])
        reranker.to(device).eval()
    preds, logits, reranked, targets = [], [], [], []
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    with torch.inference_mode():
        for batch in loader:
            trajectory, score, scene = model.generate_with_context(
                batch["history"].to(device), batch["neighbors"].to(device),
                batch["neighbor_mask"].to(device), batch["lanes"].to(device),
                batch["lane_mask"].to(device),
            )
            preds.append(trajectory.cpu().numpy())
            logits.append(score.cpu().numpy())
            if reranker:
                reranked.append(reranker(scene, trajectory).cpu().numpy())
            targets.append(batch["future"].numpy())
    return (np.concatenate(preds), np.concatenate(logits),
            np.concatenate(reranked) if reranked else np.concatenate(logits),
            np.concatenate(targets))


def errors(pred: np.ndarray, score: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    dist = np.linalg.norm(pred - target[:, None], axis=-1)
    ade, fde = dist.mean(axis=-1), dist[:, :, -1]
    selected = score.argmax(axis=1)
    batch = np.arange(len(target))
    return {
        "top_ade": ade[batch, selected], "top_fde": fde[batch, selected],
        "oracle_ade": ade.min(axis=1), "oracle_fde": fde.min(axis=1),
        "top_mode": selected, "oracle_ade_mode": ade.argmin(axis=1),
        "oracle_fde_mode": fde.argmin(axis=1),
        "selection_regret": ade[batch, selected] - ade.min(axis=1),
        "ade_by_mode": ade, "fde_by_mode": fde,
        "diversity": np.linalg.norm(pred[:, :, -1, None] - pred[:, None, :, -1], axis=-1).mean(axis=(1, 2)),
    }


def bootstrap_ci(delta: np.ndarray, seed: int = 19, samples: int = 10000) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    means = delta[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarize(x: dict[str, np.ndarray], mask: np.ndarray | None = None) -> dict[str, float | int]:
    if mask is None:
        mask = np.ones(len(x["top_ade"]), dtype=bool)
    return {
        "n": int(mask.sum()),
        "top1_ADE_m": float(x["top_ade"][mask].mean()),
        "top1_FDE_m": float(x["top_fde"][mask].mean()),
        "oracle_minADE_m": float(x["oracle_ade"][mask].mean()),
        "oracle_minFDE_m": float(x["oracle_fde"][mask].mean()),
        "miss_rate_FDE_gt_2m": float(np.mean(x["oracle_fde"][mask] > 2.0)),
        "selection_regret_ADE_m": float(x["selection_regret"][mask].mean()),
        "mean_endpoint_diversity_m": float(x["diversity"][mask].mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data/val"))
    p.add_argument("--baseline-cache", type=Path, default=Path("cache/val_500_maps.npz"))
    p.add_argument("--route-cache", type=Path, default=Path("cache/val_500_routes.npz"))
    p.add_argument("--baseline", type=Path, default=Path("checkpoints/lane_conditioned_20k.pt"))
    p.add_argument("--reranker", type=Path, default=Path("checkpoints/trajectory_reranker_lane_conditioned_20k.pt"))
    p.add_argument("--routes", type=Path, default=Path("checkpoints/lane_routes_20k.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/turn-diagnostics-20k"))
    args = p.parse_args()
    for path in [args.data, args.baseline_cache, args.route_cache, args.baseline, args.reranker, args.routes]:
        if not path.exists():
            p.error(f"missing input: {path}")
    torch.set_num_threads(4)
    bstate = torch.load(args.baseline, map_location="cpu", weights_only=False)
    ds = SocialTrajectoryDataset(
        args.data, past_steps=int(bstate["model_config"].get("past_steps", 50)),
        future_steps=int(bstate["model_config"]["future_steps"]),
        max_neighbors=int(bstate["model_config"]["max_neighbors"]),
    )
    base_data, route_data = MapContextDataset(ds, args.baseline_cache), MapContextDataset(ds, args.route_cache)
    device = torch.device("cpu")
    base_pred, base_score, base_rerank_score, target = predict(args.baseline, base_data, device, args.reranker)
    route_pred, route_score, _, _ = predict(args.routes, route_data, device)

    # The dataset future starts after the origin; prepend the origin for path shape measures.
    path = np.concatenate([np.zeros((len(ds), 1, 2), dtype=target.dtype), target], axis=1)
    increments = np.diff(path, axis=1)
    displacement = np.linalg.norm(path[:, -1], axis=1)
    endpoint_bearing = np.degrees(np.abs(np.arctan2(path[:, -1, 1], path[:, -1, 0])))
    headings = np.unwrap(np.arctan2(increments[:, :, 1], increments[:, :, 0]), axis=1)
    net_heading_change = np.abs(np.degrees(headings[:, -1] - headings[:, 9]))
    absolute_heading_change = np.degrees(np.abs(np.diff(headings, axis=1)).sum(axis=1))
    path_length = np.linalg.norm(increments, axis=-1).sum(axis=1)
    straightness = displacement / np.maximum(path_length, 1e-6)
    turn_mask = (endpoint_bearing >= 20.0) & (displacement >= 10.0)
    ids = np.asarray(ds.scenario_ids)
    if turn_mask.sum() != 79:
        raise RuntimeError(f"Expected the previous 79-scenario turn subset, found {turn_mask.sum()}")

    models = {
        "lane_generator": errors(base_pred, base_score, target),
        "lane_generator_plus_reranker": errors(base_pred, base_rerank_score, target),
        "connected_routes": errors(route_pred, route_score, target),
    }
    lane, reranked, route = (models[k] for k in models)
    bmask = turn_mask
    masks = {
        "turns_all": bmask,
        "endpoint_bearing_20_35": bmask & (endpoint_bearing < 35),
        "endpoint_bearing_35_60": bmask & (endpoint_bearing < 60) & (endpoint_bearing >= 35),
        "endpoint_bearing_60_plus": bmask & (endpoint_bearing >= 60),
        "net_heading_change_under_15": bmask & (net_heading_change < 15),
        "net_heading_change_15_45": bmask & (net_heading_change >= 15) & (net_heading_change < 45),
        "net_heading_change_45_plus": bmask & (net_heading_change >= 45),
        "low_straightness_under_0_8": bmask & (straightness < 0.8),
        "high_straightness_0_8_plus": bmask & (straightness >= 0.8),
    }
    metrics = {name: {key: summarize(value, mask) for key, value in models.items()} for name, mask in masks.items()}
    turn_idx = np.flatnonzero(turn_mask)
    pair_route_lane = {
        "top1_ADE_delta_route_minus_lane_mean_m": float(np.mean(route["top_ade"][bmask] - lane["top_ade"][bmask])),
        "top1_ADE_delta_bootstrap_95pct_m": bootstrap_ci(route["top_ade"][bmask] - lane["top_ade"][bmask]),
        "oracle_ADE_delta_route_minus_lane_mean_m": float(np.mean(route["oracle_ade"][bmask] - lane["oracle_ade"][bmask])),
        "oracle_ADE_delta_bootstrap_95pct_m": bootstrap_ci(route["oracle_ade"][bmask] - lane["oracle_ade"][bmask]),
        "oracle_FDE_delta_route_minus_lane_mean_m": float(np.mean(route["oracle_fde"][bmask] - lane["oracle_fde"][bmask])),
        "route_wins_oracle_ADE_by_0_25m": int(np.sum(route["oracle_ade"][bmask] + 0.25 < lane["oracle_ade"][bmask])),
        "lane_wins_oracle_ADE_by_0_25m": int(np.sum(lane["oracle_ade"][bmask] + 0.25 < route["oracle_ade"][bmask])),
        "ties_oracle_ADE_within_0_25m": int(np.sum(np.abs(route["oracle_ade"][bmask] - lane["oracle_ade"][bmask]) <= 0.25)),
        "coverage_transitions": {
            "both_cover_FDE_le_2m": int(np.sum((lane["oracle_fde"][bmask] <= 2) & (route["oracle_fde"][bmask] <= 2))),
            "lane_only_covers": int(np.sum((lane["oracle_fde"][bmask] <= 2) & (route["oracle_fde"][bmask] > 2))),
            "route_only_covers": int(np.sum((lane["oracle_fde"][bmask] > 2) & (route["oracle_fde"][bmask] <= 2))),
            "neither_covers": int(np.sum((lane["oracle_fde"][bmask] > 2) & (route["oracle_fde"][bmask] > 2))),
        },
        "lane_selection_success_to_failure": int(np.sum((lane["top_fde"][bmask] > 2) & (lane["oracle_fde"][bmask] <= 2))),
        "lane_selection_failure_to_success_with_reranker": int(np.sum((lane["top_fde"][bmask] > 2) & (reranked["top_fde"][bmask] <= 2))),
        "lane_selection_success_to_failure_with_reranker": int(np.sum((lane["top_fde"][bmask] <= 2) & (reranked["top_fde"][bmask] > 2))),
        "reranker_top1_ADE_delta_vs_lane_mean_m": float(np.mean(reranked["top_ade"][bmask] - lane["top_ade"][bmask])),
        "reranker_top1_ADE_delta_bootstrap_95pct_m": bootstrap_ci(reranked["top_ade"][bmask] - lane["top_ade"][bmask]),
        "reranker_improves_top1_ADE_by_at_least_0_5m": int(np.sum(lane["top_ade"][bmask] - reranked["top_ade"][bmask] >= 0.5)),
        "reranker_worsens_top1_ADE_by_at_least_0_5m": int(np.sum(reranked["top_ade"][bmask] - lane["top_ade"][bmask] >= 0.5)),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "n_validation": len(ds), "n_turns": int(turn_mask.sum()),
        "turn_definition": "absolute endpoint bearing >=20 degrees and displacement >=10m, using future only for offline subgroup reporting",
        "feature_summary_turns": {
            "endpoint_bearing_deg_median": float(np.median(endpoint_bearing[bmask])),
            "endpoint_bearing_deg_p90": float(np.quantile(endpoint_bearing[bmask], 0.9)),
            "net_heading_change_deg_median": float(np.median(net_heading_change[bmask])),
            "absolute_heading_variation_deg_median": float(np.median(absolute_heading_change[bmask])),
            "path_straightness_median": float(np.median(straightness[bmask])),
            "displacement_median": float(np.median(displacement[bmask])),
        },
        "metrics_by_slice": metrics, "paired_comparison": pair_route_lane,
        "case_csv": "turn-cases.csv",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    columns = ["scenario_id", "endpoint_bearing_deg", "net_heading_change_deg", "absolute_heading_variation_deg",
               "straightness", "path_length_m", "displacement_m", "neighbor_count"]
    for model_name, values in models.items():
        columns += [f"{model_name}_{key}" for key in ("top_ade", "top_fde", "oracle_ade", "oracle_fde", "selection_regret", "top_mode", "oracle_ade_mode", "oracle_fde_mode", "diversity")]
    columns += ["lane_oracle_miss", "route_oracle_miss", "lane_selected_miss", "reranked_selected_miss", "route_selected_miss"]
    csv_path = args.out_dir / "turn-cases.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for i in turn_idx:
            row: dict[str, object] = {
                "scenario_id": ids[i], "endpoint_bearing_deg": endpoint_bearing[i],
                "net_heading_change_deg": net_heading_change[i],
                "absolute_heading_variation_deg": absolute_heading_change[i],
                "straightness": straightness[i], "path_length_m": path_length[i],
                "displacement_m": displacement[i], "neighbor_count": int(ds[i]["neighbor_mask"].sum()),
                "lane_oracle_miss": lane["oracle_fde"][i] > 2,
                "route_oracle_miss": route["oracle_fde"][i] > 2,
                "lane_selected_miss": lane["top_fde"][i] > 2,
                "reranked_selected_miss": reranked["top_fde"][i] > 2,
                "route_selected_miss": route["top_fde"][i] > 2,
            }
            for name, values in models.items():
                for key in ("top_ade", "top_fde", "oracle_ade", "oracle_fde", "selection_regret", "top_mode", "oracle_ade_mode", "oracle_fde_mode", "diversity"):
                    row[f"{name}_{key}"] = values[key][i]
            writer.writerow(row)
    # A small, deterministic case panel makes the main failure modes inspectable.
    route_minus_lane = route["oracle_ade"] - lane["oracle_ade"]
    candidates = [
        ("Lane covers; route misses", np.flatnonzero(turn_mask & (lane["oracle_fde"] <= 2) & (route["oracle_fde"] > 2))),
        ("Route covers; lane misses", np.flatnonzero(turn_mask & (lane["oracle_fde"] > 2) & (route["oracle_fde"] <= 2))),
        ("Route worst oracle-ADE regression", np.flatnonzero(turn_mask)),
        ("Route best oracle-ADE gain", np.flatnonzero(turn_mask)),
        ("Largest endpoint bearing", np.flatnonzero(turn_mask)),
        ("Largest reranker gain", np.flatnonzero(turn_mask)),
    ]
    chosen: list[tuple[str, int]] = []
    selectors = [
        lambda ix: ix[0] if len(ix) else None,
        lambda ix: ix[0] if len(ix) else None,
        lambda ix: ix[np.argmax(route_minus_lane[ix])] if len(ix) else None,
        lambda ix: ix[np.argmin(route_minus_lane[ix])] if len(ix) else None,
        lambda ix: ix[np.argmax(endpoint_bearing[ix])] if len(ix) else None,
        lambda ix: ix[np.argmax(lane["top_ade"][ix] - reranked["top_ade"][ix])] if len(ix) else None,
    ]
    for (label, ix), select in zip(candidates, selectors):
        i = select(ix)
        if i is not None and all(old_i != i for _, old_i in chosen):
            chosen.append((label, int(i)))
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, (label, i) in zip(axes.flat, chosen):
        sample = ds[i]
        past = sample["history"][:, :2].numpy()
        ax.plot(past[:, 0], past[:, 1], color="0.55", linewidth=1.2, label="Observed history")
        ax.plot(path[i, :, 0], path[i, :, 1], color="black", linewidth=2.2, label="Ground truth")
        mode = int(base_score[i].argmax())
        ax.plot(base_pred[i, mode, :, 0], base_pred[i, mode, :, 1], color="tab:red", linewidth=1.4, label="Lane selected")
        mode = int(base_rerank_score[i].argmax())
        ax.plot(base_pred[i, mode, :, 0], base_pred[i, mode, :, 1], color="tab:green", linewidth=1.4, label="Lane + reranker")
        mode = int(route_score[i].argmax())
        ax.plot(route_pred[i, mode, :, 0], route_pred[i, mode, :, 1], color="tab:purple", linewidth=1.4, label="Route selected")
        mode = int(lane["oracle_ade_mode"][i])
        ax.plot(base_pred[i, mode, :, 0], base_pred[i, mode, :, 1], color="tab:orange", linestyle="--", linewidth=1.2, label="Lane oracle (ADE)")
        mode = int(route["oracle_ade_mode"][i])
        ax.plot(route_pred[i, mode, :, 0], route_pred[i, mode, :, 1], color="tab:cyan", linestyle=":", linewidth=1.5, label="Route oracle (ADE)")
        ax.set_title(f"{label} · {ids[i][:8]} · bearing {endpoint_bearing[i]:.0f}°")
        ax.set_xlabel("Forward (m)"); ax.set_ylabel("Lateral (m)")
        ax.set_aspect("equal", adjustable="datalim"); ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Held-out turning cases: lane segments vs connected routes", fontsize=14)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    figure_path = args.out_dir / "turn-case-examples.png"
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    print(f"Saved per-case diagnostics to {csv_path} and examples to {figure_path}")


if __name__ == "__main__":
    main()
