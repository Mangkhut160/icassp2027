#!/usr/bin/env python3
"""Evaluate public ODEWorld LIBERO checkpoints on official LIBERO HDF5 demos.

This is a transparent local evaluation protocol. The public ODEWorld repository does
not include the authors' dataset conversion or benchmark scripts, so results from this
tool must not be presented as an exact reproduction of the paper's hidden protocol.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable

import cv2
import h5py
import lpips
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models import Dinov2GoalPred, Dinov2PTflowImgoal, Dinov2RAE  # noqa: E402


PRETRAINED_ROOT = ROOT / "assets" / "pretrained"
MODEL_DIRS = {
    "pt_flow": PRETRAINED_ROOT / "ODEWorld-PT-Flow-LIBERO",
    "rae": PRETRAINED_ROOT / "ODEWorld-RAE-LIBERO",
    "goal_predictor": PRETRAINED_ROOT / "ODEWorld-Goal-Predictor-LIBERO",
}
SUITES = ("libero_10", "libero_90", "libero_goal", "libero_object", "libero_spatial")
TRANSFORMS = ("rgb", "rgb_vflip", "rgb_vhflip")
CSV_FIELDS = (
    "suite",
    "task",
    "task_file",
    "demo",
    "instruction",
    "num_raw_frames",
    "steps",
    "horizon",
    "frame_transform",
    "method",
    "psnr_mean_db",
    "psnr_endpoint_db",
    "lpips_mean",
    "lpips_endpoint",
    "inference_seconds",
    "frames_per_second",
    "peak_memory_mib",
    "status",
    "error",
)


def numeric_demo_sort(name: str) -> int:
    return int(name.rsplit("_", 1)[-1])


def discover_tasks(dataset_root: Path, suites: Iterable[str]) -> list[Path]:
    tasks = []
    for suite in suites:
        tasks.extend(sorted((dataset_root / suite).glob("*.hdf5")))
    return sorted(tasks, key=lambda path: (path.parent.name, path.name))


def parse_instruction(data_group: h5py.Group, task_path: Path) -> str:
    raw = data_group.attrs.get("problem_info")
    if raw is not None:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        info = json.loads(raw)
        instruction = info.get("language_instruction", "")
        if isinstance(instruction, list):
            instruction = "".join(str(part) for part in instruction)
        instruction = str(instruction).strip().strip('"')
        if instruction:
            return instruction
    return task_path.stem.removesuffix("_demo").replace("_", " ")


def image_key(obs_group: h5py.Group) -> str:
    for key in ("agentview_rgb", "agentview_image"):
        if key in obs_group:
            return key
    raise KeyError(f"No agent-view image in {list(obs_group.keys())}")


def transform_rgb(frame: np.ndarray, transform: str) -> np.ndarray:
    if transform == "rgb":
        output = frame
    elif transform == "rgb_vflip":
        output = frame[::-1]
    elif transform == "rgb_vhflip":
        output = frame[::-1, ::-1]
    else:
        raise ValueError(f"Unsupported transform: {transform}")
    return np.ascontiguousarray(output)


def resize_rgb(frame: np.ndarray, size: int = 256) -> np.ndarray:
    return cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)


def rgb_batch_to_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
    return tensor.to(device=device, dtype=torch.float32).div_(255.0)


def sample_demo_frames(
    frames: h5py.Dataset,
    steps: int,
    transform: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frames.shape[0] < 2:
        raise ValueError(f"Demo must have at least two frames, got {frames.shape[0]}")
    indices = np.rint(np.linspace(1, frames.shape[0] - 1, steps)).astype(np.int64)
    start = resize_rgb(transform_rgb(np.asarray(frames[0], dtype=np.uint8), transform))
    goal = resize_rgb(transform_rgb(np.asarray(frames[-1], dtype=np.uint8), transform))
    targets = np.stack(
        [
            resize_rgb(transform_rgb(np.asarray(frames[index], dtype=np.uint8), transform))
            for index in indices
        ],
        axis=0,
    )
    return start[None], goal[None], targets


@torch.inference_mode()
def decode_latents(rae: Dinov2RAE, latents: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunks = []
    for start in range(0, latents.shape[0], chunk_size):
        chunks.append(rae.decode(latents[start : start + chunk_size]).clamp_(0, 1))
    return torch.cat(chunks, dim=0)


@torch.inference_mode()
def reconstruct_images(rae: Dinov2RAE, images: torch.Tensor, chunk_size: int) -> torch.Tensor:
    chunks = []
    for start in range(0, images.shape[0], chunk_size):
        image_chunk = images[start : start + chunk_size]
        chunks.append(rae(image_chunk).clamp_(0, 1))
    return torch.cat(chunks, dim=0)


def psnr_per_frame(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction.float() - target.float()).square().flatten(1).mean(1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


@torch.inference_mode()
def lpips_per_frame(
    metric: torch.nn.Module,
    prediction: torch.Tensor,
    target: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    values = []
    for start in range(0, prediction.shape[0], chunk_size):
        pred_chunk = prediction[start : start + chunk_size].mul(2).sub(1)
        target_chunk = target[start : start + chunk_size].mul(2).sub(1)
        values.append(metric(pred_chunk, target_chunk, normalize=False).flatten())
    return torch.cat(values)


def metric_row(
    base: dict[str, object],
    method: str,
    prediction: torch.Tensor,
    target: torch.Tensor,
    lpips_metric: torch.nn.Module,
    lpips_chunk_size: int,
    inference_seconds: float,
    peak_memory_mib: float,
) -> dict[str, object]:
    psnr = psnr_per_frame(prediction, target)
    perceptual = lpips_per_frame(
        lpips_metric, prediction, target, chunk_size=lpips_chunk_size
    )
    steps = prediction.shape[0]
    return {
        **base,
        "method": method,
        "psnr_mean_db": float(psnr.mean().item()),
        "psnr_endpoint_db": float(psnr[-1].item()),
        "lpips_mean": float(perceptual.mean().item()),
        "lpips_endpoint": float(perceptual[-1].item()),
        "inference_seconds": inference_seconds,
        "frames_per_second": steps / inference_seconds if inference_seconds > 0 else math.inf,
        "peak_memory_mib": peak_memory_mib,
        "status": "ok",
        "error": "",
    }


def tensor_to_bgr(image: torch.Tensor) -> np.ndarray:
    rgb = image.detach().float().cpu().clamp(0, 1)
    rgb = rgb.permute(1, 2, 0).mul(255).round().byte().numpy()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_comparison_video(
    path: Path,
    target: torch.Tensor,
    goal_prediction: torch.Tensor,
    language_prediction: torch.Tensor,
    fps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = target.shape[-2:]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 3, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    for index in range(target.shape[0]):
        panels = [
            tensor_to_bgr(target[index]),
            tensor_to_bgr(goal_prediction[index]),
            tensor_to_bgr(language_prediction[index]),
        ]
        labels = ("Ground truth", "Image-goal", "Language-goal")
        for panel, label in zip(panels, labels):
            cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
            cv2.putText(
                panel,
                label,
                (7, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        writer.write(np.concatenate(panels, axis=1))
    writer.release()


def timed_cuda_call(function):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak_mib = torch.cuda.max_memory_allocated() / (1024**2)
    return result, elapsed, peak_mib


def append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()


def completed_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {
            (row["suite"], row["task"], row["demo"])
            for row in csv.DictReader(handle)
            if row["status"] == "ok" and row["method"] == "language_goal"
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=list(SUITES))
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--horizon", type=float, default=1.0)
    parser.add_argument("--frame-transform", choices=TRANSFORMS, default="rgb_vhflip")
    parser.add_argument("--max-demos-per-task", type=int, default=1)
    parser.add_argument("--task-limit", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--decode-chunk-size", type=int, default=16)
    parser.add_argument("--lpips-chunk-size", type=int, default=16)
    parser.add_argument("--save-videos-per-task", type=int, default=1)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This evaluation requires a Slurm-allocated CUDA GPU")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_csv = args.output_dir / f"metrics_shard_{args.shard_index:02d}.csv"
    failures_path = args.output_dir / f"failures_shard_{args.shard_index:02d}.jsonl"
    run_config = vars(args).copy()
    run_config["dataset_root"] = str(args.dataset_root)
    run_config["output_dir"] = str(args.output_dir)
    (args.output_dir / f"config_shard_{args.shard_index:02d}.json").write_text(
        json.dumps(run_config, indent=2) + "\n"
    )

    all_tasks = discover_tasks(args.dataset_root, args.suites)
    if args.task_limit is not None:
        all_tasks = all_tasks[: args.task_limit]
    tasks = [task for index, task in enumerate(all_tasks) if index % args.num_shards == args.shard_index]
    done = completed_keys(result_csv)
    failed_tasks = 0
    completed_demos = 0

    device = torch.device(args.device)
    load_started = time.perf_counter()
    rae = Dinov2RAE.from_pretrained(str(MODEL_DIRS["rae"])).to(device).eval()
    pt_flow = Dinov2PTflowImgoal.from_pretrained(str(MODEL_DIRS["pt_flow"])).to(device).eval()
    goal_predictor = Dinov2GoalPred.from_pretrained(
        str(MODEL_DIRS["goal_predictor"])
    ).to(device).eval()
    lpips_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "event": "models_loaded",
                "seconds": time.perf_counter() - load_started,
                "allocated_mib": torch.cuda.memory_allocated() / (1024**2),
                "gpu": torch.cuda.get_device_name(),
                "tasks": len(tasks),
            }
        ),
        flush=True,
    )

    for task_index, task_path in enumerate(tasks):
        suite = task_path.parent.name
        task_name = task_path.stem.removesuffix("_demo")
        try:
            with h5py.File(task_path, "r") as handle:
                data = handle["data"]
                instruction = parse_instruction(data, task_path)
                demos = sorted(data.keys(), key=numeric_demo_sort)[: args.max_demos_per_task]
                for demo_position, demo_name in enumerate(demos):
                    key = (suite, task_name, demo_name)
                    if key in done:
                        continue
                    obs = data[demo_name]["obs"]
                    frames = obs[image_key(obs)]
                    start_np, goal_np, target_np = sample_demo_frames(
                        frames, args.steps, args.frame_transform
                    )
                    start_image = rgb_batch_to_tensor(start_np, device)
                    goal_image = rgb_batch_to_tensor(goal_np, device)
                    target = rgb_batch_to_tensor(target_np, device)

                    base = {
                        "suite": suite,
                        "task": task_name,
                        "task_file": str(task_path),
                        "demo": demo_name,
                        "instruction": instruction,
                        "num_raw_frames": int(frames.shape[0]),
                        "steps": args.steps,
                        "horizon": args.horizon,
                        "frame_transform": args.frame_transform,
                    }

                    def image_goal_rollout() -> torch.Tensor:
                        latents, _ = pt_flow.rollout_ode(
                            start_image,
                            goal_image,
                            horizon=args.horizon,
                            steps=args.steps,
                        )
                        return decode_latents(rae, latents[0], args.decode_chunk_size)

                    image_prediction, image_seconds, image_peak = timed_cuda_call(
                        image_goal_rollout
                    )

                    def language_goal_rollout() -> torch.Tensor:
                        latents, _ = pt_flow.rollout_ode_lang(
                            start_image,
                            [instruction],
                            goal_predictor,
                            horizon=args.horizon,
                            steps=args.steps,
                        )
                        return decode_latents(rae, latents[0], args.decode_chunk_size)

                    language_prediction, language_seconds, language_peak = timed_cuda_call(
                        language_goal_rollout
                    )

                    rae_prediction, rae_seconds, rae_peak = timed_cuda_call(
                        lambda: reconstruct_images(rae, target, args.decode_chunk_size)
                    )
                    interpolation = torch.stack(
                        [
                            start_image[0].lerp(goal_image[0], alpha)
                            for alpha in torch.linspace(
                                1.0 / args.steps, 1.0, args.steps, device=device
                            )
                        ]
                    )

                    rows = [
                        metric_row(
                            base,
                            "image_goal",
                            image_prediction,
                            target,
                            lpips_metric,
                            args.lpips_chunk_size,
                            image_seconds,
                            image_peak,
                        ),
                        metric_row(
                            base,
                            "language_goal",
                            language_prediction,
                            target,
                            lpips_metric,
                            args.lpips_chunk_size,
                            language_seconds,
                            language_peak,
                        ),
                        metric_row(
                            base,
                            "rae_reconstruction",
                            rae_prediction,
                            target,
                            lpips_metric,
                            args.lpips_chunk_size,
                            rae_seconds,
                            rae_peak,
                        ),
                        metric_row(
                            base,
                            "pixel_linear",
                            interpolation,
                            target,
                            lpips_metric,
                            args.lpips_chunk_size,
                            0.0,
                            0.0,
                        ),
                    ]
                    append_rows(result_csv, rows)
                    completed_demos += 1

                    if demo_position < args.save_videos_per_task:
                        save_comparison_video(
                            args.output_dir / "videos" / suite / task_name / f"{demo_name}.mp4",
                            target,
                            image_prediction,
                            language_prediction,
                            args.fps,
                        )
                    print(
                        json.dumps(
                            {
                                "event": "demo_complete",
                                "task_index": task_index,
                                "task_count": len(tasks),
                                "suite": suite,
                                "task": task_name,
                                "demo": demo_name,
                                "image_goal_psnr": rows[0]["psnr_mean_db"],
                                "language_goal_psnr": rows[1]["psnr_mean_db"],
                            }
                        ),
                        flush=True,
                    )
                    del target, image_prediction, language_prediction, rae_prediction, interpolation
        except Exception as error:
            failed_tasks += 1
            failure = {
                "suite": suite,
                "task": task_name,
                "task_file": str(task_path),
                "error": repr(error),
                "traceback": traceback.format_exc(),
            }
            with failures_path.open("a") as handle:
                handle.write(json.dumps(failure) + "\n")
            print(json.dumps({"event": "task_failed", **failure}), flush=True)

    print(
        json.dumps(
            {
                "event": "shard_complete",
                "tasks": len(tasks),
                "completed_demos": completed_demos,
                "failed_tasks": failed_tasks,
            }
        ),
        flush=True,
    )
    if failed_tasks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
