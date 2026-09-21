#!/usr/bin/env python3
"""Run one shard of the ODEWorld temporal diagnostic.

Classification: mock test. This reads offline HDF5 RGB frames and runs ODEWorld
inference only. It does not create LIBERO/MuJoCo, execute robot actions, or
measure task success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.libero_manifest import resolve_task_file_rows

SCHEMA = "odeworld_temporal_diagnostic_v1"
CLASSIFICATION = "mock test"
METHODS = (
    "pixel_linear",
    "rae_reconstruction",
    "ptflow_identity",
    "ode_oracle_goal",
    "ode_language_goal",
    "ode_oracle_goal_teacher",
    "ode_language_goal_teacher",
    "hold_last_frame",
)
DEFAULT_STEPS = 64
DEFAULT_FRAME_TRANSFORM = "rgb_vhflip"
DEFAULT_HORIZON = 1.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _regular_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Expected regular non-symlink file: {path}")
    return path


def numeric_demo_sort(name: str) -> int:
    return int(name.rsplit("_", 1)[-1])


def target_indices(frame_count: int, steps: int) -> np.ndarray:
    if frame_count < 2:
        raise ValueError(f"Expected at least two raw frames, got {frame_count}")
    if steps <= 0:
        raise ValueError("steps must be positive")
    return np.rint(np.linspace(1, frame_count - 1, steps)).astype(np.int64)


def transform_rgb(frame: np.ndarray, transform: str) -> np.ndarray:
    if transform == "rgb":
        output = frame
    elif transform == "rgb_vflip":
        output = frame[::-1]
    elif transform == "rgb_vhflip":
        output = frame[::-1, ::-1]
    else:
        raise ValueError(f"Unsupported frame transform: {transform}")
    return np.ascontiguousarray(output)


def resize_rgb(frame: np.ndarray, size: int = 256) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected RGB frame [H,W,3], got {frame.shape}")
    if frame.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB frame, got {frame.dtype}")
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def sample_demo_frames(
    frames: h5py.Dataset,
    *,
    steps: int,
    transform: str,
    resolution: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = target_indices(int(frames.shape[0]), steps)
    start = resize_rgb(transform_rgb(np.asarray(frames[0], dtype=np.uint8), transform), resolution)
    goal = resize_rgb(transform_rgb(np.asarray(frames[-1], dtype=np.uint8), transform), resolution)
    targets = np.stack(
        [resize_rgb(transform_rgb(np.asarray(frames[index], dtype=np.uint8), transform), resolution) for index in indices],
        axis=0,
    )
    return start, goal, targets, indices


def pixel_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float | None]:
    if reference.shape != prediction.shape or reference.dtype != np.uint8 or prediction.dtype != np.uint8:
        raise ValueError("Reference and prediction must be same-shape uint8 arrays")
    difference = prediction.astype(np.float32) - reference.astype(np.float32)
    mse = float(np.mean(np.square(difference)))
    return {
        # JSON has no representation for infinity; mse=0 is the exact-match sentinel.
        "psnr_db": None if mse == 0.0 else float(10.0 * math.log10((255.0**2) / mse)),
        "l1": float(np.abs(difference).mean()),
        "mse": mse,
    }


def pixel_linear_prediction(start: np.ndarray, goal: np.ndarray, steps: int) -> np.ndarray:
    if start.shape != goal.shape or start.dtype != np.uint8 or goal.dtype != np.uint8:
        raise ValueError("start and goal must be matching uint8 RGB frames")
    if start.ndim != 3 or start.shape[2] != 3:
        raise ValueError("start and goal must have shape [H,W,3]")
    alphas = np.linspace(1.0 / steps, 1.0, steps, dtype=np.float32)
    prediction = start[None].astype(np.float32) * (1.0 - alphas[:, None, None, None])
    prediction += goal[None].astype(np.float32) * alphas[:, None, None, None]
    return np.rint(np.clip(prediction, 0.0, 255.0)).astype(np.uint8)


def hold_last_prediction(start: np.ndarray, steps: int) -> np.ndarray:
    if steps <= 0:
        raise ValueError("steps must be positive")
    return np.repeat(start[None], steps, axis=0)


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _parse_instruction(data_group: h5py.Group, task_file: Path) -> str:
    raw = data_group.attrs.get("problem_info")
    if raw is not None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            info = json.loads(raw)
            instruction = info.get("language_instruction", "")
            if isinstance(instruction, list):
                instruction = "".join(str(part) for part in instruction)
            if str(instruction).strip():
                return str(instruction).strip().strip('"')
        except (TypeError, json.JSONDecodeError):
            pass
    return task_file.stem.removesuffix("_demo").replace("_", " ")


def _image_key(obs: h5py.Group) -> str:
    for key in ("agentview_rgb", "agentview_image"):
        if key in obs:
            return key
    raise KeyError(f"No agent-view image in {list(obs.keys())}")


def _load_models(device: str):
    import torch

    from models import Dinov2GoalPred, Dinov2PTflowImgoal, Dinov2RAE

    root = Path(__file__).resolve().parents[1]
    pretrained = root / "assets" / "pretrained"
    rae = Dinov2RAE.from_pretrained(str(pretrained / "ODEWorld-RAE-LIBERO")).to(device).eval()
    flow = Dinov2PTflowImgoal.from_pretrained(str(pretrained / "ODEWorld-PT-Flow-LIBERO")).to(device).eval()
    goal = Dinov2GoalPred.from_pretrained(str(pretrained / "ODEWorld-Goal-Predictor-LIBERO")).to(device).eval()
    return torch, rae, flow, goal


def _to_tensor(array: np.ndarray, device: str):
    import torch

    return torch.from_numpy(array).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32).div_(255.0)


def _decode(rae, latents, chunk_size: int):
    import torch

    if latents.ndim != 3:
        raise ValueError(f"RAE decode expects [frames,tokens,channels], got {tuple(latents.shape)}")
    chunks = []
    for start in range(0, latents.shape[0], chunk_size):
        chunks.append(rae.decode(latents[start : start + chunk_size]).clamp_(0, 1))
    return torch.cat(chunks, dim=0)


def _rgb_from_tensor(tensor) -> np.ndarray:
    return np.rint(tensor.detach().float().cpu().clamp(0, 1).permute(0, 2, 3, 1).mul(255)).byte().numpy()


def _model_predictions(
    start: np.ndarray,
    goal: np.ndarray,
    targets: np.ndarray,
    instruction: str,
    *,
    device: str,
    horizon: float,
    steps: int,
    decode_chunk_size: int,
) -> dict[str, np.ndarray]:
    torch, rae, flow, goal_predictor = _load_models(device)
    start_tensor = _to_tensor(start[None], device)
    goal_tensor = _to_tensor(goal[None], device)
    target_tensor = _to_tensor(targets, device)
    with torch.inference_mode():
        rae_prediction = _decode(rae, rae.encoder(target_tensor), decode_chunk_size)
        s0 = flow.latent_encode(start_tensor)
        identity = _decode(rae, s0.expand(steps, -1, -1), decode_chunk_size)
        oracle_frames, _ = flow.rollout_ode(start_tensor, goal_tensor, horizon=horizon, steps=steps)
        oracle = _decode(rae, oracle_frames[0], decode_chunk_size)
        language_frames, _ = flow.rollout_ode_lang(start_tensor, [instruction], goal_predictor, horizon=horizon, steps=steps)
        language = _decode(rae, language_frames[0], decode_chunk_size)
        teacher_oracle = []
        teacher_language = []
        for index in range(steps):
            current = target_tensor[index : index + 1]
            one_oracle, _ = flow.rollout_ode(current, goal_tensor, horizon=horizon / steps, steps=1)
            one_language, _ = flow.rollout_ode_lang(current, [instruction], goal_predictor, horizon=horizon / steps, steps=1)
            teacher_oracle.append(_decode(rae, one_oracle[0], decode_chunk_size)[0])
            teacher_language.append(_decode(rae, one_language[0], decode_chunk_size)[0])
    return {
        "pixel_linear": pixel_linear_prediction(start, goal, steps),
        "rae_reconstruction": _rgb_from_tensor(rae_prediction),
        "ptflow_identity": _rgb_from_tensor(identity),
        "ode_oracle_goal": _rgb_from_tensor(oracle),
        "ode_language_goal": _rgb_from_tensor(language),
        "ode_oracle_goal_teacher": _rgb_from_tensor(torch.stack(teacher_oracle)),
        "ode_language_goal_teacher": _rgb_from_tensor(torch.stack(teacher_language)),
        "hold_last_frame": hold_last_prediction(start, steps),
    }


def _load_sampling_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(_regular_file(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "odeworld_diagnostic_sampling_v1":
        raise RuntimeError(f"Unexpected sampling manifest: {path}")
    return value


def _metric_rows(
    base: Mapping[str, Any], targets: np.ndarray, predictions: Mapping[str, np.ndarray], target_indices_array: np.ndarray
) -> list[dict[str, Any]]:
    rows = []
    for method in METHODS:
        frames = predictions[method]
        if frames.shape != targets.shape:
            raise ValueError(f"Prediction shape mismatch for {method}: {frames.shape} vs {targets.shape}")
        for frame_index, (target, prediction, raw_index) in enumerate(zip(targets, frames, target_indices_array)):
            rows.append({
                **base,
                "method": method,
                "frame_index": frame_index,
                "raw_target_index": int(raw_index),
                **pixel_metrics(target, prediction),
            })
    return rows


def run_trial(row: Mapping[str, Any], output_dir: Path, *, device: str, steps: int, horizon: float, transform: str, decode_chunk_size: int) -> dict[str, Any]:
    task_file = _regular_file(Path(str(row["task_file"])))
    suite, task, demo = str(row["suite"]), str(row["task"]), str(row["demo"])
    trial_dir = output_dir / "trials" / suite / task / demo
    if trial_dir.exists() or trial_dir.is_symlink():
        raise FileExistsError(f"Refusing existing trial directory: {trial_dir}")
    trial_dir.mkdir(parents=True, exist_ok=False)
    with h5py.File(task_file, "r") as handle:
        data = handle["data"]
        if demo not in data:
            raise KeyError(f"Missing {demo} in {task_file}")
        demo_group = data[demo]
        obs = demo_group["obs"]
        start, goal, targets, indices = sample_demo_frames(obs[_image_key(obs)], steps=steps, transform=transform)
        instruction = str(row.get("instruction") or _parse_instruction(data, task_file))
        start_hash, goal_hash = sha256_array(start), sha256_array(goal)
    predictions = _model_predictions(start, goal, targets, instruction, device=device, horizon=horizon, steps=steps, decode_chunk_size=decode_chunk_size)
    base = {
        "suite": suite,
        "task": task,
        "demo": demo,
        "instruction": instruction,
        "steps": steps,
        "horizon": horizon,
        "frame_transform": transform,
        "resolution": int(start.shape[0]),
    }
    rows = _metric_rows(base, targets, predictions, indices)
    np.savez_compressed(trial_dir / "reference_frames.npz", start=start, goal=goal, targets=targets, target_indices=indices)
    method_hashes = {}
    method_dir = trial_dir / "method_frames"
    method_dir.mkdir()
    for method, frames in predictions.items():
        path = method_dir / f"{method}.npz"
        np.savez_compressed(path, frames=frames)
        method_hashes[method] = {"path": str(path.relative_to(output_dir)), "sha256": sha256_file(path), "array_sha256": sha256_array(frames)}
    with (trial_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for metric_row in rows:
            handle.write(json.dumps(metric_row, allow_nan=False) + "\n")
    _write_json(trial_dir / "trial_manifest.json", {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "identity": {"suite": suite, "task": task, "demo": demo},
        "source": {"task_file": str(task_file), "task_file_sha256": sha256_file(task_file), "start_sha256": start_hash, "goal_sha256": goal_hash},
        "protocol": {"methods": list(METHODS), "steps": steps, "horizon": horizon, "frame_transform": transform, "resolution": int(start.shape[0]), "target_index_rule": "round(linspace(1,num_raw_frames-1,steps))"},
        "method_files": method_hashes,
        "metric_rows": len(rows),
    })
    _write_json(trial_dir / "result.json", {"status": "completed", "classification": CLASSIFICATION, "identity": {"suite": suite, "task": task, "demo": demo}, "metric_rows": len(rows)})
    return {"identity": {"suite": suite, "task": task, "demo": demo}, "metric_rows": len(rows), "methods": list(METHODS)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sampling-manifest", type=Path, required=True)
    parser.add_argument(
        "--libero-root",
        default=None,
        help="LIBERO HDF5 root for manifests with relative task_file paths "
        "(defaults to the LIBERO_ROOT environment variable)",
    )
    parser.add_argument("--split", choices=("b1_pilot", "b1_confirmatory", "b2_confirmatory"), required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--horizon", type=float, default=DEFAULT_HORIZON)
    parser.add_argument("--frame-transform", default=DEFAULT_FRAME_TRANSFORM, choices=("rgb", "rgb_vflip", "rgb_vhflip"))
    parser.add_argument("--decode-chunk-size", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; submit this command through Slurm")
    if (args.output_root.exists() or args.output_root.is_symlink()) and any(args.output_root.iterdir()):
        raise FileExistsError(f"Output root must be absent or empty: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_sampling_manifest(args.sampling_manifest)
    rows = resolve_task_file_rows(manifest["splits"][args.split], args.libero_root)
    shard_rows = [row for index, row in enumerate(rows) if index % args.shard_count == args.shard_index]
    config = {"schema": SCHEMA, "classification": CLASSIFICATION, "sampling_manifest": str(args.sampling_manifest), "sampling_manifest_sha256": sha256_file(args.sampling_manifest), "split": args.split, "shard_index": args.shard_index, "shard_count": args.shard_count, "steps": args.steps, "horizon": args.horizon, "frame_transform": args.frame_transform, "device": args.device}
    _write_json(args.output_root / f"config_shard_{args.shard_index:02d}.json", config)
    completed = []
    for row in shard_rows:
        completed.append(run_trial(row, args.output_root, device=args.device, steps=args.steps, horizon=args.horizon, transform=args.frame_transform, decode_chunk_size=args.decode_chunk_size))
    _write_json(args.output_root / f"result_shard_{args.shard_index:02d}.json", {"schema": SCHEMA, "classification": CLASSIFICATION, "split": args.split, "shard_index": args.shard_index, "trial_count": len(completed), "trials": completed})
    print(json.dumps({"classification": CLASSIFICATION, "split": args.split, "shard_index": args.shard_index, "trial_count": len(completed)}), flush=True)


if __name__ == "__main__":
    main()
