"""Print the temporal structure of an AV2 scenario."""

from __future__ import annotations

import argparse
from pathlib import Path

from motion_forecasting.data import find_scenario, load_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        nargs="?",
        default="data/val",
        help="scenario Parquet file or directory (default: data/val)",
    )
    args = parser.parse_args()

    scenario_path = find_scenario(args.scenario)
    scenario = load_scenario(scenario_path)
    focal = next(
        (track for track in scenario["tracks"] if track["track_id"] == scenario["focal_track_id"]),
        None,
    )
    if focal is None:
        raise ValueError(f"Focal track {scenario['focal_track_id']} is missing from the scenario")

    past = focal["observed"]
    future = ~past
    x, y = focal["position_x"], focal["position_y"]
    print(f"Scenario: {scenario['scenario_id']}")
    print(f"City: {scenario['city']}")
    print(f"Tracks: {len(scenario['tracks'])}")
    print("\nFocal agent:")
    print(f"type: {focal['object_type'].upper()}")
    print(f"observed states: {int(past.sum())}")
    print(f"future states: {int(future.sum())}")
    print(f"Start position: ({x[0]:.2f}, {y[0]:.2f})")
    print(f"Last observed position: ({x[past][-1]:.2f}, {y[past][-1]:.2f})")
    print(f"Final position: ({x[-1]:.2f}, {y[-1]:.2f})")
    print(f"\nSource: {Path(scenario_path)}")


if __name__ == "__main__":
    main()
