#!/usr/bin/env python3
"""Evaluate a PT-Flow checkpoint with the frozen D107 offline protocol.

Classification: mock test. This reads offline LIBERO HDF5 RGB frames and runs
ODEWorld inference only. It does not create LIBERO/MuJoCo, call env.step,
execute robot actions, or measure task success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import Dinov2GoalPred, Dinov2PTflowImgoal, Dinov2RAE
from tools.run_libero_framewise_odeworld import ModelBundle
from tools.run_libero_language_color_ablation import ColorAblationBackend
from tools.run_odeworld_color_grounding_pilot import _seed_everything
from tools.run_odeworld_refresh_decomposition import (
    BRANCHES,
    CLASSIFICATION,
    HORIZON,
    SAMPLING_MANIFEST,
    STRATEGIES,
    run_trial,
)
from tools.libero_manifest import resolve_task_file_rows
from tools.run_odeworld_temporal_diagnostic_shard import _load_sampling_manifest, _regular_file

RAE_CHECKPOINT = PROJECT_ROOT / "assets" / "pretrained" / "ODEWorld-RAE-LIBERO"
GOAL_CHECKPOINT = PROJECT_ROOT / "assets" / "pretrained" / "ODEWorld-Goal-Predictor-LIBERO"
SCHEMA = "odeworld_ptflow_checkpoint_eval_v1"
SCALE_M_SCHEMA = "odeworld_ptflow_scale_m_manifest_v1"
SCALE_L_SCHEMA = "odeworld_ptflow_scale_l_manifest_v1"
FROZEN_SPLITS = ("b1_pilot", "b1_confirmatory", "b2_confirmatory")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _regular_file(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_dir(path: Path) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"Expected regular non-symlink directory: {path}")
    return path


def write_json(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_bundle(flow_checkpoint: Path, device: Any) -> ModelBundle:
    flow = Dinov2PTflowImgoal.from_pretrained(str(regular_dir(flow_checkpoint))).to(device).eval()
    rae = Dinov2RAE.from_pretrained(str(regular_dir(RAE_CHECKPOINT))).to(device).eval()
    goal = Dinov2GoalPred.from_pretrained(str(regular_dir(GOAL_CHECKPOINT))).to(device).eval()
    if int(flow.max_time_length) != 50:
        raise RuntimeError(f"Expected PT-Flow max_time_length=50, got {flow.max_time_length}")
    return ModelBundle(flow=flow, rae=rae, goal_predictor=goal, device=device)


def aggregate(trials: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values: dict[tuple[str, str], list[float]] = {
        (strategy, branch): [] for strategy in STRATEGIES for branch in BRANCHES
    }
    for trial in trials:
        for row in trial["metric_rows"]:
            values[(str(row["strategy"]), str(row["branch"]))].append(float(row["l1"]))
    result: dict[str, Any] = {}
    for key, entries in values.items():
        result[f"{key[0]}_{key[1]}"] = {
            "count": len(entries),
            "mean_l1": float(np.mean(entries)),
            "median_l1": float(np.median(entries)),
        }
    return result


def _row_identity(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["suite"]), str(row["task"]), str(row["demo"])


def load_evaluation_rows(
    path: Path, *, split: str, max_demos: int, libero_root: str | None = None
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if max_demos <= 0:
        raise ValueError("max_demos must be positive")
    if path.absolute() == SAMPLING_MANIFEST.absolute():
        if split not in FROZEN_SPLITS:
            raise RuntimeError(f"Frozen diagnostic manifest does not define evaluation split: {split}")
        manifest = _load_sampling_manifest(path)
        rows = resolve_task_file_rows(manifest["splits"][split][:max_demos], libero_root)
        return manifest, rows

    manifest = json.loads(_regular_file(path).read_text(encoding="utf-8"))
    manifest_schema = str(manifest.get("schema"))
    if manifest_schema not in {SCALE_M_SCHEMA, SCALE_L_SCHEMA}:
        raise RuntimeError(f"Unsupported evaluation manifest: {path}")
    source = manifest.get("source", {})
    if Path(str(source.get("path", ""))).resolve() != SAMPLING_MANIFEST.resolve():
        raise RuntimeError("Scale-M source path does not match the frozen diagnostic manifest")
    if source.get("sha256") != sha256_file(SAMPLING_MANIFEST):
        raise RuntimeError("Scale-M source hash does not match the frozen diagnostic manifest")
    expected_split = "scale_l_test" if manifest_schema == SCALE_L_SCHEMA else "scale_m_test"
    if split != expected_split:
        raise RuntimeError(f"{manifest_schema} evaluation requires split={expected_split}")
    counts = manifest.get("counts", {})
    if (
        int(counts.get("tasks", -1)) != 130
        or int(counts.get("test", -1)) != 650
        or int(counts.get("test_per_task", -1)) != 5
    ):
        raise RuntimeError("Balanced manifest does not declare the exact 130-task/650-demo test contract")
    if max_demos != 650:
        raise RuntimeError("Balanced evaluation requires the full 650-demo test split")
    rows = [dict(row) for row in manifest.get("splits", {}).get(split, [])]
    identities = [_row_identity(row) for row in rows]
    task_counts = Counter(key[:2] for key in identities)
    if len(rows) != 650 or len(set(identities)) != 650:
        raise RuntimeError("Balanced test rows must contain 650 unique identities")
    if len(task_counts) != 130 or set(task_counts.values()) != {5}:
        raise RuntimeError("Balanced test rows are not balanced at five demos per task")
    return manifest, resolve_task_file_rows(rows, libero_root)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("SLURM_JOB_ID"):
        print(
            "warning: SLURM_JOB_ID is not set; running outside Slurm", file=sys.stderr
        )
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(f"Output root must be absent: {args.output_root}")
    manifest, rows = load_evaluation_rows(
        args.sampling_manifest,
        split=args.split,
        max_demos=args.max_demos,
        libero_root=args.libero_root,
    )
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    _seed_everything(args.seed)
    args.output_root.mkdir(parents=True)
    bundle = load_bundle(args.flow_checkpoint, torch.device(args.device))
    backend = ColorAblationBackend(
        bundle,
        decode_chunk_size=args.decode_chunk_size,
        ode_solver=args.ode_solver,
        ode_rtol=args.ode_rtol,
        ode_atol=args.ode_atol,
    )
    trials = []
    for row in rows:
        trial = run_trial(
            row,
            args.output_root,
            backend,
            ode_solver=args.ode_solver,
            ode_rtol=args.ode_rtol,
            ode_atol=args.ode_atol,
        )
        trial_dir = args.output_root / "trials" / str(row["suite"]) / str(row["task"]) / str(row["demo"])
        metric_rows = []
        with (trial_dir / "metrics.jsonl").open("r", encoding="utf-8") as handle:
            metric_rows = [json.loads(line) for line in handle if line.strip()]
        trials.append({**trial, "metric_rows": metric_rows})
    config = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "flow_checkpoint": str(args.flow_checkpoint),
        "flow_config_sha256": sha256_file(args.flow_checkpoint / "config.json"),
        "flow_weights_sha256": sha256_file(args.flow_checkpoint / "model.safetensors"),
        "rae_checkpoint": str(RAE_CHECKPOINT),
        "goal_predictor_checkpoint": str(GOAL_CHECKPOINT),
        "sampling_manifest": str(args.sampling_manifest),
        "sampling_manifest_sha256": sha256_file(args.sampling_manifest),
        "sampling_manifest_schema": manifest.get("schema"),
        "split": args.split,
        "demo_count": len(rows),
        "strategies": list(STRATEGIES),
        "branches": list(BRANCHES),
        "horizon": HORIZON,
        "ode_solver": args.ode_solver,
        "ode_rtol": args.ode_rtol,
        "ode_atol": args.ode_atol,
        "seed": args.seed,
        "device": args.device,
        "scope": {"env_step_calls": 0, "libero_or_mujoco_started": False, "success_rate_measured": False},
    }
    write_json(args.output_root / "config.json", config)
    summary = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "all_checks_passed": True,
        "trial_count": len(trials),
        "metric_rows": sum(len(trial["metric_rows"]) for trial in trials),
        "ode_solver": args.ode_solver,
        "ode_rtol": args.ode_rtol,
        "ode_atol": args.ode_atol,
        "aggregate": aggregate(trials),
        "scope": config["scope"],
    }
    write_json(args.output_root / "summary.json", summary)
    print(json.dumps({"classification": CLASSIFICATION, "completed": True, **summary}, indent=2), flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow-checkpoint", type=Path, required=True)
    parser.add_argument("--sampling-manifest", type=Path, default=SAMPLING_MANIFEST)
    parser.add_argument(
        "--libero-root",
        default=None,
        help="LIBERO HDF5 root for manifests with relative task_file paths "
        "(defaults to the LIBERO_ROOT environment variable)",
    )
    parser.add_argument(
        "--split",
        default="b2_confirmatory",
    )
    parser.add_argument("--max-demos", type=int, default=5)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--decode-chunk-size", type=int, default=16)
    parser.add_argument("--ode-solver", choices=("rk4", "dopri5"), default="rk4")
    parser.add_argument("--ode-rtol", type=float, default=1e-5)
    parser.add_argument("--ode-atol", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args(argv)


if __name__ == "__main__":
    print("classification=mock test env_step_calls=0 libero_or_mujoco_started=false success_rate_measured=false", flush=True)
    run(parse_args())
