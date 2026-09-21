#!/usr/bin/env python3
"""Build the immutable ODEWorld diagnostic sampling manifest.

Classification: mock test. This reads the existing offline reproduction
manifest and metric-shard identity tables. It does not create MuJoCo, execute
actions, or run ODEWorld inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "odeworld_diagnostic_sampling_v1"
CLASSIFICATION = "mock test"
REQUIRED_SUITE_COUNTS = {
    "libero_10": 10,
    "libero_90": 90,
    "libero_goal": 10,
    "libero_object": 10,
    "libero_spatial": 10,
}
REQUIRED_METHODS = {
    "image_goal",
    "language_goal",
    "rae_reconstruction",
    "pixel_linear",
}
IDENTITY_FIELDS = ("suite", "task", "demo")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Expected regular non-symlink file: {path}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(_regular_file(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def _rank(seed: str, values: Iterable[str]) -> str:
    payload = "|".join((seed, *values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_metric_rows(paths: Sequence[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with _regular_file(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"suite", "task", "task_file", "demo", "instruction", "method", "status"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise RuntimeError(f"Metric shard {path} is missing fields: {sorted(missing)}")
            rows.extend(dict(row) for row in reader)
    return rows


def _validate_reproduction(
    source_root: Path,
    reproduction: Mapping[str, Any],
    *,
    expected_suite_counts: Mapping[str, int],
) -> None:
    dataset = reproduction.get("dataset")
    if not isinstance(dataset, Mapping):
        raise TypeError("reproduction_manifest.dataset must be an object")
    suites = dataset.get("suites")
    if dict(suites or {}) != dict(expected_suite_counts):
        raise RuntimeError(
            f"Suite inventory mismatch: expected {dict(expected_suite_counts)}, got {suites}"
        )
    expected_tasks = sum(expected_suite_counts.values())
    if dataset.get("task_files") != expected_tasks:
        raise RuntimeError(
            f"Expected {expected_tasks} task files, got {dataset.get('task_files')}"
        )
    if not source_root.is_dir() or source_root.is_symlink():
        raise RuntimeError(f"Expected regular source directory: {source_root}")


def _identity(row: Mapping[str, str]) -> tuple[str, str, str]:
    values = tuple(str(row.get(field, "")).strip() for field in IDENTITY_FIELDS)
    if any(not value for value in values):
        raise RuntimeError(f"Metric row has empty identity fields: {row}")
    return values


def _select_demo_rows(
    rows_by_task: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    selected_tasks: Sequence[tuple[str, str]],
    *,
    demos_per_task: int,
    seed: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for suite, task in selected_tasks:
        rows = rows_by_task[(suite, task)]
        demos = sorted(
            {str(row["demo"]) for row in rows},
            key=lambda demo: _rank(seed, (suite, task, demo)),
        )
        if len(demos) < demos_per_task:
            raise RuntimeError(f"Task {suite}/{task} has fewer than {demos_per_task} demos")
        for demo in demos[:demos_per_task]:
            source = next(row for row in rows if str(row["demo"]) == demo)
            selected.append(
                {
                    "suite": suite,
                    "task": task,
                    "demo": demo,
                    "task_file": str(source["task_file"]),
                    "instruction": str(source["instruction"]),
                }
            )
    return selected


def build_manifest(
    source_root: Path,
    *,
    selection_seed: str,
    expected_suite_counts: Mapping[str, int] = REQUIRED_SUITE_COUNTS,
    expected_demos_per_task: int = 50,
    metric_shard_count: int = 3,
) -> dict[str, Any]:
    source_root = source_root.absolute()
    reproduction_path = source_root / "reproduction_manifest.json"
    reproduction = _load_json(reproduction_path)
    _validate_reproduction(
        source_root, reproduction, expected_suite_counts=expected_suite_counts
    )
    shard_paths = sorted(source_root.glob("metrics_shard_*.csv"))
    if len(shard_paths) != metric_shard_count:
        raise RuntimeError(
            f"Expected {metric_shard_count} metrics shards, found {len(shard_paths)}"
        )
    rows = _read_metric_rows(shard_paths)
    rows_by_identity: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    rows_by_task: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        identity = _identity(row)
        if row.get("status") != "ok":
            raise RuntimeError(f"Non-ok metric row for {identity}: {row.get('status')}")
        rows_by_identity[identity].append(row)
        rows_by_task[(identity[0], identity[1])].append(row)

    duplicate_identities = [
        key
        for key, values in rows_by_identity.items()
        if len(values) != len(REQUIRED_METHODS)
        or {str(row["method"]) for row in values} != REQUIRED_METHODS
    ]
    if duplicate_identities:
        raise RuntimeError(
            "Each demo must have exactly one row per required method; bad identities: "
            + repr(duplicate_identities[:5])
        )
    expected_tasks = sum(expected_suite_counts.values())
    expected_population = expected_tasks * expected_demos_per_task
    if len(rows_by_identity) != expected_population:
        raise RuntimeError(
            f"Expected {expected_population} unique demos, found {len(rows_by_identity)}"
        )
    task_counts = {suite: 0 for suite in expected_suite_counts}
    for suite, _task in rows_by_task:
        if suite not in task_counts:
            raise RuntimeError(f"Unknown suite in metrics: {suite}")
        task_counts[suite] += 1
    if task_counts != dict(expected_suite_counts):
        raise RuntimeError(f"Task counts mismatch: expected {dict(expected_suite_counts)}, got {task_counts}")
    for task_key, task_rows in rows_by_task.items():
        demo_count = len({str(row["demo"]) for row in task_rows})
        if demo_count != expected_demos_per_task:
            raise RuntimeError(f"Expected {expected_demos_per_task} demos in {task_key}, got {demo_count}")
        for row in task_rows:
            if not str(row["task_file"]).endswith(".hdf5"):
                raise RuntimeError(f"Task file is not HDF5 in {task_key}: {row['task_file']}")

    tasks_by_suite: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for suite, task in rows_by_task:
        tasks_by_suite[suite].append((suite, task))
    pilot_tasks: list[tuple[str, str]] = []
    confirm_tasks: list[tuple[str, str]] = []
    b2_tasks: list[tuple[str, str]] = []
    for suite in expected_suite_counts:
        ranked = sorted(tasks_by_suite[suite], key=lambda key: _rank(selection_seed, key))
        if len(ranked) < 8:
            raise RuntimeError(f"Suite {suite} needs at least 8 tasks for the frozen design")
        pilot_tasks.append(ranked[0])
        confirm_tasks.extend(ranked[1:6])
        b2_tasks.extend(ranked[6:8])

    population = []
    for identity in sorted(rows_by_identity, key=lambda key: (key[0], key[1], int(key[2].split("_")[-1]))):
        source = rows_by_identity[identity][0]
        population.append(
            {
                "suite": identity[0],
                "task": identity[1],
                "demo": identity[2],
                "task_file": source["task_file"],
                "instruction": source["instruction"],
            }
        )

    return {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "source": {
            "root": str(source_root),
            "reproduction_manifest": {
                "path": str(reproduction_path),
                "sha256": sha256_file(reproduction_path),
            },
            "metric_shards": [
                {"path": str(path), "sha256": sha256_file(path)} for path in shard_paths
            ],
        },
        "population": {
            "suite_counts": dict(expected_suite_counts),
            "task_count": expected_tasks,
            "demos_per_task": expected_demos_per_task,
            "demo_count": len(population),
            "rows": population,
        },
        "splits": {
            "b1_pilot": _select_demo_rows(rows_by_task, pilot_tasks, demos_per_task=3, seed=selection_seed),
            "b1_confirmatory": _select_demo_rows(rows_by_task, confirm_tasks, demos_per_task=4, seed=selection_seed),
            "b2_confirmatory": _select_demo_rows(rows_by_task, b2_tasks, demos_per_task=3, seed=selection_seed),
        },
        "color_state_pairs": {
            "pilot": [f"pair_{index:02d}" for index in range(5)],
            "confirmatory": [f"pair_{index:02d}" for index in range(5, 25)],
        },
        "selection": {
            "seed": selection_seed,
            "algorithm": "sha256(seed|suite|task|demo), identity fields only",
            "metric_values_used_for_selection": False,
            "pilot_tasks_disjoint_from_b1_confirmatory": True,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection-seed", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = build_manifest(args.source_root, selection_seed=args.selection_seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        "population={demo_count} b1_pilot={pilot} b1_confirmatory={confirm} b2_confirmatory={b2}".format(
            demo_count=manifest["population"]["demo_count"],
            pilot=len(manifest["splits"]["b1_pilot"]),
            confirm=len(manifest["splits"]["b1_confirmatory"]),
            b2=len(manifest["splits"]["b2_confirmatory"]),
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
