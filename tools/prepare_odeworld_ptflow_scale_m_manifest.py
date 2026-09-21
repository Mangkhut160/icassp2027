#!/usr/bin/env python3
"""Prepare the task-balanced PT-Flow Scale-M offline split manifest.

Classification: mock test. This tool only reads JSON metadata. It does not
start LIBERO/MuJoCo, call env.step, execute actions, or measure success rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "odeworld_ptflow_scale_m_manifest_v1"
SCALE_L_SCHEMA = "odeworld_ptflow_scale_l_manifest_v1"
SOURCE_SCHEMA = "odeworld_diagnostic_sampling_v1"
HELD_OUT_SPLITS = ("b1_pilot", "b1_confirmatory", "b2_confirmatory")


def identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["suite"]), str(row["task"]), str(row["demo"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_seed(data_seed: int, suite: str, task: str) -> int:
    payload = f"{int(data_seed)}\0{suite}\0{task}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _validate_row(row: Mapping[str, Any]) -> dict[str, Any]:
    required = {"suite", "task", "demo", "task_file", "instruction"}
    if set(row) != required:
        raise RuntimeError(f"Unexpected population row fields: {sorted(row)}")
    result = {key: str(row[key]) for key in sorted(required)}
    if any(not result[key] for key in required):
        raise RuntimeError(f"Empty population row field: {row}")
    return result


def build_scale_manifest(
    source: Mapping[str, Any],
    *,
    source_path: str,
    source_sha256: str,
    data_seed: int,
    train_per_task: int = 20,
    validation_per_task: int = 5,
    test_per_task: int = 5,
    variant: str = "scale_m",
) -> dict[str, Any]:
    if source.get("schema") != SOURCE_SCHEMA:
        raise RuntimeError(f"Expected source schema {SOURCE_SCHEMA}")
    requested = (int(train_per_task), int(validation_per_task), int(test_per_task))
    if any(value <= 0 for value in requested):
        raise ValueError("Per-task split counts must be positive")

    population = [_validate_row(row) for row in source.get("population", {}).get("rows", [])]
    population_by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    tasks: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in population:
        key = identity(row)
        if key in population_by_identity:
            raise RuntimeError(f"Duplicate population identity: {key}")
        population_by_identity[key] = row
        tasks[key[:2]].append(row)
    if not tasks:
        raise RuntimeError("Source population is empty")

    held_out: set[tuple[str, str, str]] = set()
    for split in HELD_OUT_SPLITS:
        for raw_row in source.get("splits", {}).get(split, []):
            key = identity(raw_row)
            if key not in population_by_identity:
                raise RuntimeError(f"Held-out identity is absent from population: {key}")
            held_out.add(key)

    if variant not in {"scale_m", "scale_l"}:
        raise ValueError(f"Unsupported scale variant: {variant}")
    prefix = variant
    result_splits: dict[str, list[dict[str, Any]]] = {
        f"{prefix}_train": [],
        f"{prefix}_validation": [],
        f"{prefix}_test": [],
        f"{prefix}_unused": [],
    }
    for task_key in sorted(tasks):
        suite, task = task_key
        rows = sorted(tasks[task_key], key=identity)
        if len(rows) < sum(requested):
            raise RuntimeError(f"Task has too few demos for requested split: {task_key} count={len(rows)}")
        task_held_out = [row for row in rows if identity(row) in held_out]
        if len(task_held_out) > test_per_task:
            raise RuntimeError(
                f"Task has more held-out rows than the test allocation: {task_key} "
                f"held-out={len(task_held_out)} test={test_per_task}"
            )
        available = [row for row in rows if identity(row) not in held_out]
        random.Random(_task_seed(data_seed, suite, task)).shuffle(available)
        test_fill = test_per_task - len(task_held_out)
        test_rows = sorted(task_held_out + available[:test_fill], key=identity)
        cursor = test_fill
        validation_rows = sorted(available[cursor : cursor + validation_per_task], key=identity)
        cursor += validation_per_task
        train_rows = sorted(available[cursor : cursor + train_per_task], key=identity)
        cursor += train_per_task
        unused_rows = sorted(available[cursor:], key=identity)
        result_splits[f"{prefix}_train"].extend(train_rows)
        result_splits[f"{prefix}_validation"].extend(validation_rows)
        result_splits[f"{prefix}_test"].extend(test_rows)
        result_splits[f"{prefix}_unused"].extend(unused_rows)

    split_identities = {name: {identity(row) for row in rows} for name, rows in result_splits.items()}
    split_names = list(split_identities)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = split_identities[left] & split_identities[right]
            if overlap:
                raise RuntimeError(f"Scale-M splits overlap: {left}/{right}: {sorted(overlap)[:3]}")
    if not held_out <= split_identities[f"{prefix}_test"]:
        raise RuntimeError(f"The {variant} test split does not preserve every prior held-out row")

    task_count = len(tasks)
    expected_counts = {
            f"{prefix}_train": task_count * train_per_task,
        f"{prefix}_validation": task_count * validation_per_task,
        f"{prefix}_test": task_count * test_per_task,
    }
    for name, expected in expected_counts.items():
        if len(result_splits[name]) != expected:
            raise RuntimeError(f"Unexpected {name} count: {len(result_splits[name])} != {expected}")

    return {
        "schema": SCHEMA if variant == "scale_m" else SCALE_L_SCHEMA,
        "classification": "mock test",
        "source": {
            "path": str(source_path),
            "sha256": str(source_sha256),
            "schema": SOURCE_SCHEMA,
            "held_out_splits": list(HELD_OUT_SPLITS),
            "held_out_union_count": len(held_out),
        },
        "selection": {
            "kind": "task_balanced_deterministic_shuffle",
            "data_seed": int(data_seed),
            "held_out_rows_are_test_only": True,
        },
        "counts": {
            "tasks": task_count,
            "population": len(population),
            "train": len(result_splits[f"{prefix}_train"]),
            "validation": len(result_splits[f"{prefix}_validation"]),
            "test": len(result_splits[f"{prefix}_test"]),
            "unused": len(result_splits[f"{prefix}_unused"]),
            "train_per_task": int(train_per_task),
            "validation_per_task": int(validation_per_task),
            "test_per_task": int(test_per_task),
        },
        "splits": result_splits,
        "scope": {
            "env_step_calls": 0,
            "libero_or_mujoco_started": False,
            "success_rate_measured": False,
        },
    }


def build_scale_m_manifest(
    source: Mapping[str, Any],
    *,
    source_path: str,
    source_sha256: str,
    data_seed: int,
    train_per_task: int = 20,
    validation_per_task: int = 5,
    test_per_task: int = 5,
) -> dict[str, Any]:
    return build_scale_manifest(
        source,
        source_path=source_path,
        source_sha256=source_sha256,
        data_seed=data_seed,
        train_per_task=train_per_task,
        validation_per_task=validation_per_task,
        test_per_task=test_per_task,
        variant="scale_m",
    )


def write_json(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-seed", type=int, default=20260903)
    parser.add_argument("--train-per-task", type=int, default=20)
    parser.add_argument("--validation-per-task", type=int, default=5)
    parser.add_argument("--test-per-task", type=int, default=5)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.source.is_symlink() or not args.source.is_file():
        raise RuntimeError(f"Expected regular source manifest: {args.source}")
    source_sha256 = sha256_file(args.source)
    source = json.loads(args.source.read_text(encoding="utf-8"))
    manifest = build_scale_m_manifest(
        source,
        source_path=str(args.source.resolve()),
        source_sha256=source_sha256,
        data_seed=args.data_seed,
        train_per_task=args.train_per_task,
        validation_per_task=args.validation_per_task,
        test_per_task=args.test_per_task,
    )
    write_json(args.output, manifest)
    print(json.dumps({"classification": "mock test", "output": str(args.output), "counts": manifest["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
