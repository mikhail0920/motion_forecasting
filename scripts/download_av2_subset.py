"""Download a deterministic subset of AV2 motion-forecasting scenarios."""

from __future__ import annotations

import argparse
import shutil
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


S3_ENDPOINT = "https://argoverse.s3.amazonaws.com/"
S3_PREFIX = "datasets/av2/motion-forecasting"


def list_scenario_ids(split: str) -> list[str]:
    """List all scenario directory IDs for a split using anonymous S3 listing."""
    scenario_ids: set[str] = set()
    continuation_token: str | None = None
    prefix = f"{S3_PREFIX}/{split}/"

    while True:
        params = {"list-type": "2", "prefix": prefix, "delimiter": "/", "max-keys": "1000"}
        if continuation_token:
            params["continuation-token"] = continuation_token
        url = f"{S3_ENDPOINT}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(
            url, headers={"User-Agent": "motion-forecasting-av2-subset/1.0"}
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                root = ET.fromstring(response.read())
        except (urllib.error.URLError, ET.ParseError) as exc:
            raise RuntimeError(f"Failed to list AV2 {split} scenarios from public S3: {exc}") from exc

        for common_prefix in root.findall(".//{*}CommonPrefixes/{*}Prefix"):
            value = common_prefix.text or ""
            scenario_id = value.removeprefix(prefix).strip("/")
            if scenario_id and "/" not in scenario_id:
                scenario_ids.add(scenario_id)

        if root.findtext(".//{*}IsTruncated") != "true":
            break
        continuation_token = root.findtext(".//{*}NextContinuationToken")
        if not continuation_token:
            raise RuntimeError("S3 marked the listing as truncated but did not return a continuation token")

    return sorted(scenario_ids)


def download_scenario(scenario_id: str, split: str, output_dir: Path) -> Path:
    """Download only one scenario Parquet, preserving the AV2 directory layout."""
    filename = f"scenario_{scenario_id}.parquet"
    target_dir = output_dir / scenario_id
    target = target_dir / filename
    if target.is_file() and target.stat().st_size > 0:
        return target

    target_dir.mkdir(parents=True, exist_ok=True)
    key = f"{S3_PREFIX}/{split}/{scenario_id}/{filename}"
    url = f"{S3_ENDPOINT}{urllib.parse.quote(key, safe='/')}"
    temporary = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "motion-forecasting-av2-subset/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as file:
            shutil.copyfileobj(response, file)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Downloaded an empty scenario file: {url}")
        temporary.replace(target)
    except (urllib.error.URLError, OSError, RuntimeError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download scenario {scenario_id}: {exc}") from exc
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--num-scenarios", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_scenarios < 1:
        parser.error("--num-scenarios must be positive")

    print(f"Listing AV2 {args.split} scenario IDs from public S3...")
    scenario_ids = list_scenario_ids(args.split)
    if args.num_scenarios > len(scenario_ids):
        parser.error(
            f"requested {args.num_scenarios:,} scenarios, but split {args.split!r} contains "
            f"only {len(scenario_ids):,} scenario directories"
        )

    selected = scenario_ids[: args.num_scenarios]
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(selected):,} sorted {args.split} scenarios to {args.output}...")
    for index, scenario_id in enumerate(selected, start=1):
        download_scenario(scenario_id, args.split, args.output)
        if index % 100 == 0 or index == len(selected):
            print(f"  {index:,}/{len(selected):,}")
    print(f"Finished. Downloaded only scenario_*.parquet files under {args.output}.")


if __name__ == "__main__":
    main()
