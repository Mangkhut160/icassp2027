#!/usr/bin/env python3
"""Run the D105 generated-frame refresh decomposition diagnostic.

Classification: mock test. This reads offline HDF5 RGB frames and frozen
ODEWorld checkpoints only. It does not create a simulator, call env.step,
execute robot actions, or measure task success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np

from tools.libero_manifest import resolve_task_file_rows
from tools.run_libero_framewise_odeworld import (
    GOAL_PREDICTOR_CHECKPOINT,
    PT_FLOW_CHECKPOINT,
    RAE_CHECKPOINT,
    load_models,
    rgb_to_tensor,
)
from tools.run_libero_language_color_ablation import (
    ColorAblationBackend,
    stable_frame_metrics,
)
from tools.run_odeworld_color_grounding_pilot import (
    _checkpoint_provenance,
    _seed_everything,
)
from tools.run_odeworld_temporal_diagnostic_shard import (
    _image_key,
    _load_sampling_manifest,
    _parse_instruction,
    _regular_file,
    sample_demo_frames,
)

SCHEMA = "odeworld_refresh_decomposition_v1"
CLASSIFICATION = "mock test"
STEPS = 64
HORIZON = 1.0
RESOLUTION = 256
TRANSFORM = "rgb_vhflip"
SEGMENT_TIME_RULE = "restart_zero_each_segment"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLING_MANIFEST = PROJECT_ROOT / "manifests" / "odeworld_diagnostic_sampling_v1.json"
BRANCHES = ("language", "oracle")
STRATEGIES = (
    "open_loop_k64",
    "image_local_k8",
    "image_global_k8",
    "latent_global_k8",
    "true_refresh_k8",
)
STRATEGY_PROTOCOL = {
    "open_loop_k64": {"period": 64, "boundary_source": "original_start", "time_rule": "global"},
    "image_local_k8": {"period": 8, "boundary_source": "previous_generated_frame", "time_rule": "restart_zero"},
    "image_global_k8": {"period": 8, "boundary_source": "previous_generated_frame", "time_rule": "global"},
    "latent_global_k8": {"period": 8, "boundary_source": "carried_dynamic_latent", "saved_boundary_image_role": "decoded_observation_only_not_input", "time_rule": "global"},
    "true_refresh_k8": {"period": 8, "boundary_source": "true_hdf5_frame", "time_rule": "restart_zero"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def condition_name(strategy: str, branch: str) -> str:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    if branch not in BRANCHES:
        raise ValueError(f"Unknown branch: {branch}")
    return f"{strategy}_{branch}"


def _image_segment(
    backend: Any,
    current: np.ndarray,
    goal_latent: np.ndarray,
    *,
    start_time: float,
    end_time: float,
    steps: int,
    ode_solver: str = "rk4",
    ode_rtol: float = 1e-5,
    ode_atol: float = 1e-6,
) -> np.ndarray:
    """Integrate one image-conditioned segment on an explicit time interval."""
    import torch
    from torchdiffeq import odeint

    current_tensor = rgb_to_tensor(current, backend.bundle.device)
    flow = backend.bundle.flow
    latent = torch.as_tensor(goal_latent, dtype=torch.float32, device=backend.bundle.device)
    if latent.ndim == 2:
        latent = latent.unsqueeze(0)
    with torch.inference_mode():
        state_zero = flow.latent_encode(current_tensor)
        dynamic_zero = flow.delta_decouple(state_zero, state_zero)
        dynamic_goal = flow.delta_decouple(state_zero, latent)

        def ode_func(time_scalar: Any, dynamic_latent: Any) -> Any:
            physical_time = time_scalar.view(1, 1).expand(1, 1)
            return flow.forward_vmodel(
                dynamic_zero, dynamic_latent, dynamic_goal, physical_time
            ) * flow.max_time_length

        grid = torch.linspace(start_time, end_time, steps + 1, device=backend.bundle.device)
        ode_options = {}
        if ode_solver != "rk4":
            ode_options = {"rtol": ode_rtol, "atol": ode_atol}
        dynamic = odeint(
            ode_func,
            dynamic_zero,
            grid,
            method=ode_solver,
            **ode_options,
        )[1:]
        dynamic = dynamic.permute(1, 0, 2, 3)[0]
        decoded = []
        for start in range(0, steps, backend.decode_chunk_size):
            chunk = dynamic[start : start + backend.decode_chunk_size]
            decoded.append(flow.delta_decode(state_zero.expand(chunk.shape[0], -1, -1), chunk))
        reconstructed = torch.cat(decoded, dim=0)
        from tools.run_libero_framewise_odeworld import decode_patch_latents
        return np.stack(decode_patch_latents(backend.bundle.rae, reconstructed, chunk_size=backend.decode_chunk_size))


def _latent_global_rollout(
    backend: Any,
    start: np.ndarray,
    goal_latent: np.ndarray,
    *,
    period: int = 8,
    ode_solver: str = "rk4",
    ode_rtol: float = 1e-5,
    ode_atol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Carry PT-Flow dynamic state across global-time segments without RGB re-encoding."""
    import torch
    from torchdiffeq import odeint
    flow = backend.bundle.flow
    start_tensor = rgb_to_tensor(start, backend.bundle.device)
    latent = torch.as_tensor(goal_latent, dtype=torch.float32, device=backend.bundle.device)
    if latent.ndim == 2:
        latent = latent.unsqueeze(0)
    with torch.inference_mode():
        state_zero = flow.latent_encode(start_tensor)
        dynamic_zero = flow.delta_decouple(state_zero, state_zero)
        dynamic_goal = flow.delta_decouple(state_zero, latent)
        current_dynamic = dynamic_zero
        outputs = []
        boundaries = []
        for cursor in range(0, STEPS, period):
            count = min(period, STEPS - cursor)
            boundaries.append(start.copy() if cursor == 0 else np.asarray(outputs[-1]).copy())
            t0, t1 = cursor / STEPS, (cursor + count) / STEPS

            def ode_func(time_scalar: Any, dynamic_latent: Any) -> Any:
                physical_time = time_scalar.view(1, 1).expand(1, 1)
                return flow.forward_vmodel(
                    dynamic_zero, dynamic_latent, dynamic_goal, physical_time
                ) * flow.max_time_length

            grid = torch.linspace(t0, t1, count + 1, device=backend.bundle.device)
            ode_options = {}
            if ode_solver != "rk4":
                ode_options = {"rtol": ode_rtol, "atol": ode_atol}
            trajectory = odeint(
                ode_func,
                current_dynamic,
                grid,
                method=ode_solver,
                **ode_options,
            )[1:]
            current_dynamic = trajectory[-1]
            dynamic = trajectory.permute(1, 0, 2, 3)[0]
            decoded = []
            for offset in range(0, count, backend.decode_chunk_size):
                chunk = dynamic[offset : offset + backend.decode_chunk_size]
                decoded.append(flow.delta_decode(state_zero.expand(chunk.shape[0], -1, -1), chunk))
            from tools.run_libero_framewise_odeworld import decode_patch_latents
            outputs.extend(decode_patch_latents(backend.bundle.rae, torch.cat(decoded, dim=0), chunk_size=backend.decode_chunk_size))
        return np.stack(outputs), np.stack(boundaries)


def rollout_strategy(
    backend: Any,
    start: np.ndarray,
    goal_latent: np.ndarray,
    targets: np.ndarray,
    *,
    strategy: str,
    horizon: float = HORIZON,
    ode_solver: str = "rk4",
    ode_rtol: float = 1e-5,
    ode_atol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one frozen D105 strategy and return predictions and boundary inputs."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    if targets.ndim != 4 or targets.shape[0] != STEPS:
        raise ValueError(f"Expected targets [64,H,W,3], got {targets.shape}")
    if start.shape != targets.shape[1:] or start.dtype != np.uint8 or targets.dtype != np.uint8:
        raise ValueError("Start and targets must be shape-compatible uint8 images")
    if horizon <= 0.0:
        raise ValueError("horizon must be positive")
    if ode_solver not in {"rk4", "dopri5"}:
        raise ValueError(f"Unsupported ODE solver: {ode_solver}")
    if ode_rtol <= 0.0 or ode_atol <= 0.0:
        raise ValueError("ODE tolerances must be positive")

    if strategy == "latent_global_k8":
        return _latent_global_rollout(
            backend,
            start,
            goal_latent,
            period=8,
            ode_solver=ode_solver,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
        )
    spec = STRATEGY_PROTOCOL[strategy]
    period = int(spec["period"])
    source = str(spec["boundary_source"])
    output = np.empty_like(targets)
    boundary_inputs: list[np.ndarray] = []
    cursor = 0
    while cursor < STEPS:
        segment_steps = min(period, STEPS - cursor)
        if cursor == 0:
            current = start
        elif source == "previous_generated_frame":
            current = output[cursor - 1]
        elif source == "true_hdf5_frame":
            current = targets[cursor - 1]
        elif source == "original_start":
            raise RuntimeError("open_loop_k64 unexpectedly requested another segment")
        else:
            raise RuntimeError(f"Unsupported boundary source: {source}")
        boundary_inputs.append(np.asarray(current).copy())
        if strategy == "image_global_k8":
            generated = _image_segment(
                backend, current, goal_latent,
                start_time=cursor / STEPS,
                end_time=(cursor + segment_steps) / STEPS,
                steps=segment_steps,
                ode_solver=ode_solver,
                ode_rtol=ode_rtol,
                ode_atol=ode_atol,
            )
        else:
            generated = np.stack(backend.rollout_from_goal_latent(
                current, goal_latent,
                horizon=horizon * segment_steps / STEPS,
                steps=segment_steps,
            ))
        expected_shape = (segment_steps, *targets.shape[1:])
        if generated.shape != expected_shape or generated.dtype != np.uint8:
            raise RuntimeError(
                f"Invalid segment output for {strategy} at {cursor}: {generated.shape}"
            )
        output[cursor : cursor + segment_steps] = generated
        cursor += segment_steps
    return output, np.stack(boundary_inputs)


def _metric_rows(
    identity: Mapping[str, str],
    targets: np.ndarray,
    target_indices: np.ndarray,
    predictions: Mapping[tuple[str, str], np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        for branch in BRANCHES:
            for frame_index, (target, prediction, raw_index) in enumerate(
                zip(targets, predictions[(strategy, branch)], target_indices)
            ):
                metric = stable_frame_metrics(target, prediction)
                rows.append(
                    {
                        **identity,
                        "strategy": strategy,
                        "branch": branch,
                        "frame_index": frame_index,
                        "raw_target_index": int(raw_index),
                        "psnr_db": None
                        if not np.isfinite(metric["psnr_db"])
                        else float(metric["psnr_db"]),
                        "l1": float(metric["l1"]),
                        "mse": float(metric["mse"]),
                    }
                )
    return rows


def run_trial(
    row: Mapping[str, Any],
    output_root: Path,
    backend: Any,
    *,
    ode_solver: str = "rk4",
    ode_rtol: float = 1e-5,
    ode_atol: float = 1e-6,
) -> dict[str, Any]:
    task_file = _regular_file(Path(str(row["task_file"])))
    identity = {
        "suite": str(row["suite"]),
        "task": str(row["task"]),
        "demo": str(row["demo"]),
    }
    trial_dir = (
        output_root
        / "trials"
        / identity["suite"]
        / identity["task"]
        / identity["demo"]
    )
    if trial_dir.exists() or trial_dir.is_symlink():
        raise FileExistsError(f"Refusing existing trial directory: {trial_dir}")
    trial_dir.mkdir(parents=True)

    with h5py.File(task_file, "r") as handle:
        data = handle["data"]
        demo_group = data[identity["demo"]]
        start, _goal, targets, indices = sample_demo_frames(
            demo_group["obs"][_image_key(demo_group["obs"])],
            steps=STEPS,
            transform=TRANSFORM,
            resolution=RESOLUTION,
        )
        instruction = str(row.get("instruction") or _parse_instruction(data, task_file))

    _seed_everything(42)
    language_latent, _ = backend.predict_goal(start, instruction)
    import torch

    with torch.inference_mode():
        oracle_latent = (
            backend.bundle.flow.latent_encode(
                rgb_to_tensor(targets[-1], backend.bundle.device)
            )[0]
            .detach()
            .float()
            .cpu()
            .numpy()
        )
    latents = {
        "language": np.asarray(language_latent, dtype=np.float32),
        "oracle": np.asarray(oracle_latent, dtype=np.float32),
    }
    if latents["language"].shape != latents["oracle"].shape:
        raise RuntimeError("language/oracle latent shapes differ")

    predictions: dict[tuple[str, str], np.ndarray] = {}
    boundaries: dict[tuple[str, str], np.ndarray] = {}
    for strategy in STRATEGIES:
        for branch in BRANCHES:
            key = (strategy, branch)
            predictions[key], boundaries[key] = rollout_strategy(
                backend,
                start,
                latents[branch],
                targets,
                strategy=strategy,
                ode_solver=ode_solver,
                ode_rtol=ode_rtol,
                ode_atol=ode_atol,
            )

    np.savez_compressed(
        trial_dir / "reference_frames.npz",
        start=start,
        targets=targets,
        target_indices=indices,
    )
    np.savez_compressed(trial_dir / "goal_latents.npz", **latents)
    prediction_arrays = {
        condition_name(strategy, branch): predictions[(strategy, branch)]
        for strategy in STRATEGIES
        for branch in BRANCHES
    }
    boundary_arrays = {
        condition_name(strategy, branch): boundaries[(strategy, branch)]
        for strategy in STRATEGIES
        for branch in BRANCHES
    }
    np.savez_compressed(trial_dir / "predictions.npz", **prediction_arrays)
    np.savez_compressed(trial_dir / "boundary_inputs.npz", **boundary_arrays)
    rows = _metric_rows(identity, targets, indices, predictions)
    with (trial_dir / "metrics.jsonl").open("x", encoding="utf-8") as handle:
        for metric_row in rows:
            handle.write(json.dumps(metric_row, allow_nan=False) + "\n")

    _write_json(
        trial_dir / "trial_manifest.json",
        {
            "schema": SCHEMA,
            "classification": CLASSIFICATION,
            "identity": identity,
            "instruction": instruction,
            "source": {
                "task_file": str(task_file),
                "task_file_sha256": sha256_file(task_file),
                "start_sha256": sha256_array(start),
                "targets_sha256": sha256_array(targets),
            },
            "protocol": {
                "strategies": list(STRATEGIES),
                "strategy_protocol": STRATEGY_PROTOCOL,
                "branches": list(BRANCHES),
                "steps": STEPS,
                "horizon": HORIZON,
                "ode_solver": "rk4",
                "ode_rtol": 1e-5,
                "ode_atol": 1e-6,
                "frame_transform": TRANSFORM,
                "resolution": RESOLUTION,
                "target_index_rule": "round(linspace(1,num_raw_frames-1,steps))",
                "segment_time_rule": SEGMENT_TIME_RULE,
            },
            "latents": {
                "path": "goal_latents.npz",
                "sha256": sha256_file(trial_dir / "goal_latents.npz"),
                "arrays": {key: sha256_array(value) for key, value in latents.items()},
            },
            "predictions": {
                "path": "predictions.npz",
                "sha256": sha256_file(trial_dir / "predictions.npz"),
                "arrays": {
                    key: sha256_array(value) for key, value in prediction_arrays.items()
                },
            },
            "boundary_inputs": {
                "path": "boundary_inputs.npz",
                "sha256": sha256_file(trial_dir / "boundary_inputs.npz"),
                "arrays": {
                    key: sha256_array(value) for key, value in boundary_arrays.items()
                },
            },
            "metric_rows": len(rows),
        },
    )
    _write_json(
        trial_dir / "result.json",
        {
            "schema": SCHEMA,
            "classification": CLASSIFICATION,
            "status": "completed",
            "identity": identity,
            "metric_rows": len(rows),
        },
    )
    return {"identity": identity, "metric_rows": len(rows)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("D105 must run through Slurm")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(f"Output root must be absent: {args.output_root}")
    if args.sampling_manifest.absolute() != SAMPLING_MANIFEST.absolute():
        raise RuntimeError(
            f"D105 requires the frozen sampling manifest: {SAMPLING_MANIFEST}"
        )
    manifest = _load_sampling_manifest(args.sampling_manifest)
    rows = resolve_task_file_rows(manifest["splits"][args.split], args.libero_root)
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_root.mkdir(parents=True)
    backend = ColorAblationBackend(
        load_models(torch.device(args.device)),
        decode_chunk_size=args.decode_chunk_size,
    )
    completed = [run_trial(row, args.output_root, backend) for row in rows]
    _write_json(
        args.output_root / "config.json",
        {
            "schema": SCHEMA,
            "classification": CLASSIFICATION,
            "sampling_manifest": str(args.sampling_manifest),
            "sampling_manifest_sha256": sha256_file(args.sampling_manifest),
            "split": args.split,
            "strategies": list(STRATEGIES),
            "strategy_protocol": STRATEGY_PROTOCOL,
            "branches": list(BRANCHES),
            "steps": STEPS,
            "horizon": HORIZON,
            "frame_transform": TRANSFORM,
            "resolution": RESOLUTION,
            "segment_time_rule": SEGMENT_TIME_RULE,
            "device": args.device,
            "source_files": {"producer": sha256_file(Path(__file__).resolve())},
            "checkpoints": {
                "pt_flow": _checkpoint_provenance(PT_FLOW_CHECKPOINT),
                "rae": _checkpoint_provenance(RAE_CHECKPOINT),
                "goal_predictor": _checkpoint_provenance(GOAL_PREDICTOR_CHECKPOINT),
            },
        },
    )
    _write_json(
        args.output_root / "result.json",
        {
            "schema": SCHEMA,
            "classification": CLASSIFICATION,
            "split": args.split,
            "trial_count": len(completed),
            "metric_rows": sum(int(item["metric_rows"]) for item in completed),
            "trials": completed,
        },
    )
    return {
        "classification": CLASSIFICATION,
        "trial_count": len(completed),
        "output_root": str(args.output_root),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-manifest", type=Path, default=SAMPLING_MANIFEST)
    parser.add_argument(
        "--libero-root",
        default=None,
        help="LIBERO HDF5 root for manifests with relative task_file paths "
        "(defaults to the LIBERO_ROOT environment variable)",
    )
    parser.add_argument(
        "--split", choices=("b1_pilot", "b2_confirmatory"), required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--decode-chunk-size", type=int, default=16)
    args = parser.parse_args(argv)
    print(
        "classification=mock test env_step_calls=0 "
        "robot_actions_executed=0 success_rate_measured=false",
        flush=True,
    )
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
