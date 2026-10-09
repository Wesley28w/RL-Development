# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Small, framework-independent helpers for fixed-budget policy evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path


def output_paths(output_dir: str, train_seed: int) -> tuple[Path, Path]:
    """Return the summary and episode-table paths for one trained seed."""
    root = Path(output_dir).expanduser().resolve()
    return root / f"seed_{train_seed}_summary.json", root / f"seed_{train_seed}_episodes.csv"


def refuse_overwrite(summary_path: Path, episodes_path: Path, overwrite: bool) -> None:
    """Fail before Isaac Sim starts if an evaluation would overwrite evidence."""
    existing = [str(path) for path in (summary_path, episodes_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Evaluation output already exists; pass --overwrite to replace it:\n  " + "\n  ".join(existing)
        )


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Hash a checkpoint without loading another copy into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(successes: int, episodes: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Compute the two-sided 95% Wilson interval for a binomial proportion."""
    if episodes <= 0:
        return math.nan, math.nan
    proportion = successes / episodes
    denominator = 1.0 + z * z / episodes
    center = (proportion + z * z / (2.0 * episodes)) / denominator
    half_width = (
        z * math.sqrt(proportion * (1.0 - proportion) / episodes + z * z / (4.0 * episodes * episodes)) / denominator
    )
    return center - half_width, center + half_width


def write_results(
    *,
    summary_path: Path,
    episodes_path: Path,
    checkpoint: str,
    task: str,
    framework: str,
    train_seed: int,
    evaluation_seed: int,
    num_envs: int,
    requested_episodes: int,
    deterministic: bool,
    success_definition: str,
    records: list[dict[str, int | float | bool]],
    invocation: list[str] | None = None,
    additional_summary: dict[str, int | float | str | bool] | None = None,
    metric_definitions: dict[str, str] | None = None,
) -> dict:
    """Write auditable per-episode data and a compact run summary."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    returns = [float(record["return"]) for record in records]
    lengths = [int(record["length"]) for record in records]
    successes = sum(bool(record["success"]) for record in records)
    ci_low, ci_high = wilson_interval(successes, len(records))

    core_fields = ["episode", "env_id", "return", "length", "success"]
    extra_fields = sorted({key for record in records for key in record}.difference(core_fields))
    with episodes_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=core_fields + extra_fields)
        writer.writeheader()
        writer.writerows(records)

    checkpoint_path = str(Path(checkpoint).resolve())
    result = {
        "schema_version": 2,
        "task": task,
        "framework": framework,
        "checkpoint": checkpoint_path,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "train_seed": train_seed,
        "evaluation_seed": evaluation_seed,
        "num_envs": num_envs,
        "requested_episodes": requested_episodes,
        "completed_episodes": len(records),
        "deterministic_policy": deterministic,
        "reset_state_curriculum_enabled": False,
        "reset_distribution": "original/default task reset distribution",
        "success_definition": success_definition,
        "successes": successes,
        "success_rate": successes / len(records),
        "success_rate_wilson_95_low": ci_low,
        "success_rate_wilson_95_high": ci_high,
        "mean_episode_return": statistics.fmean(returns),
        "sample_std_episode_return": statistics.stdev(returns) if len(returns) > 1 else 0.0,
        "mean_episode_length": statistics.fmean(lengths),
        "sample_std_episode_length": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
        "command": subprocess.list2cmdline(invocation if invocation is not None else sys.argv),
        "episodes_file": str(episodes_path.resolve()),
    }
    if additional_summary:
        collisions = set(result).intersection(additional_summary)
        if collisions:
            raise ValueError(f"Additional summary keys collide with core result keys: {sorted(collisions)}")
        result.update(additional_summary)
    if metric_definitions:
        result["metric_definitions"] = metric_definitions
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return result
