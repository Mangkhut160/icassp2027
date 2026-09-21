#!/usr/bin/env python3
"""Run an offline ODEWorld language-color prompt ablation.

Classification: mock test. The script reads saved frames from a verified
partial rollout, but does not create LIBERO or call env.step().
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tools.run_libero_framewise_odeworld import (
    GOAL_PREDICTOR_CHECKPOINT,
    PT_FLOW_CHECKPOINT,
    RAE_CHECKPOINT,
    ModelBundle,
    VideoSink,
    decode_patch_latents,
    fixed_error_heatmap,
    full_future_schedule,
    load_models,
    make_labelled_panel,
    one_step_horizon,
    rgb_to_tensor,
    tensor_to_rgb,
    validate_dino_cache,
    validate_promoted_result_source,
    write_csv,
    write_gif,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESULT = (
    Path(os.environ.get("ODEWORLD_OUTPUTS_ROOT", str(ROOT / "outputs")))
    / "libero_moka_red_framewise_odeworld_20260824"
)
PROMPTS = {
    "neutral": "put the moka pot on the stove",
    "red": "put the red moka pot on the stove",
    "silver": "put the silver moka pot on the stove",
}
CONDITIONS = tuple(PROMPTS)
CLASSIFICATION = "mock test"
SCOPE_DESCRIPTION = (
    "offline language-color ablation on frames from a verified partial rollout"
)
RESOLUTION = 256
FPS = 10
SEED = 42
EXPECTED_SOURCE_FRAMES = 152
MODE_ORDER = ("goal_probe", "teacher_forced", "full_future", "autoregressive")
PAIR_DEFINITIONS = {
    "red_minus_neutral": ("red", "neutral"),
    "red_minus_silver": ("red", "silver"),
    "silver_minus_neutral": ("silver", "neutral"),
}
LATENT_PAIR_DEFINITIONS = {
    "neutral_vs_red": ("neutral", "red"),
    "neutral_vs_silver": ("neutral", "silver"),
    "red_vs_silver": ("red", "silver"),
}


def _validate_rgb(frame: np.ndarray, *, name: str = "frame") -> None:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected {name} shape [H,W,3], got {frame.shape}")
    if frame.dtype != np.uint8:
        raise ValueError(f"Expected {name} dtype uint8, got {frame.dtype}")


def build_red_mask(
    frame: np.ndarray,
    *,
    min_pixels: int = 25,
) -> np.ndarray | None:
    """Find the largest rendered red component in the lower 85% of a frame."""
    _validate_rgb(frame)
    if min_pixels <= 0:
        raise ValueError("min_pixels must be positive")
    rgb = frame.astype(np.int16)
    red = rgb[..., 0]
    mask = (
        (red >= 100)
        & (red - rgb[..., 1] >= 30)
        & (red - rgb[..., 2] >= 30)
    )
    mask[: round(frame.shape[0] * 0.15)] = False
    kernel = np.ones((3, 3), dtype=np.uint8)
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    if count <= 1:
        return None
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[component, cv2.CC_STAT_AREA])
    if area < min_pixels:
        return None
    return labels == component


def color_scores(frame: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    """Measure red chromaticity and bright neutral color inside a fixed mask."""
    _validate_rgb(frame)
    if mask.shape != frame.shape[:2] or mask.dtype != np.bool_:
        raise ValueError("mask must be a bool array matching the frame")
    if not mask.any():
        raise ValueError("mask must be non-empty")
    pixels = frame[mask].astype(np.float64)
    red, green, blue = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    red_score = np.mean((red - (green + blue) / 2.0) / 255.0)
    red_fraction = np.mean(
        (red >= 100) & (red >= 1.25 * green) & (red >= 1.25 * blue)
    )
    chroma = pixels.max(axis=1) - pixels.min(axis=1)
    neutrality = 1.0 - float(np.mean(chroma / 255.0))
    brightness = float(np.mean(pixels) / 255.0)
    return {
        "mask_pixels": int(mask.sum()),
        "red_score": float(red_score),
        "red_fraction": float(red_fraction),
        "neutrality": neutrality,
        "brightness": brightness,
        "silver_score": neutrality * brightness,
    }


def pairwise_rgb_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    metrics = stable_frame_metrics(first, second)
    changed = np.any(first != second, axis=2)
    return {
        "rgb_l1": metrics["l1"],
        "psnr_db": metrics["psnr_db"],
        "changed_fraction": float(changed.mean()),
    }


def stable_frame_metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """Use float64 reductions so persisted raw frames reproduce metrics exactly."""
    _validate_rgb(reference, name="reference")
    _validate_rgb(prediction, name="prediction")
    if reference.shape != prediction.shape:
        raise ValueError("Reference and prediction shapes differ")
    difference = prediction.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(np.square(difference), dtype=np.float64))
    return {
        "psnr_db": math.inf if mse == 0.0 else 10.0 * math.log10((255.0**2) / mse),
        "l1": float(np.mean(np.abs(difference), dtype=np.float64)),
        "mse": mse,
    }


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def latent_pair_metrics(first: Any, second: Any) -> dict[str, float]:
    """Compute mean per-token cosine and normalized-L2 distances."""
    left = _as_numpy(first)
    right = _as_numpy(second)
    if left.shape != right.shape or left.ndim < 2:
        raise ValueError(f"Latent shapes differ or are invalid: {left.shape} vs {right.shape}")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Latents must be finite")
    left = left.reshape(-1, left.shape[-1])
    right = right.reshape(-1, right.shape[-1])
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    denominator = np.maximum(left_norm * right_norm, 1e-12)
    cosine = np.sum(left * right, axis=1) / denominator
    both_zero = (left_norm < 1e-12) & (right_norm < 1e-12)
    cosine[both_zero] = 1.0
    normalized_left = left / np.maximum(left_norm[:, None], 1e-12)
    normalized_right = right / np.maximum(right_norm[:, None], 1e-12)
    return {
        "cosine_distance": float(np.mean(1.0 - cosine)),
        "normalized_l2": float(
            np.mean(np.linalg.norm(normalized_left - normalized_right, axis=1))
        ),
    }


def moving_block_interval(
    grouped_values: Mapping[int, Sequence[float]],
    *,
    block_length: int,
    samples: int,
    seed: int,
) -> dict[str, float | int]:
    """Bootstrap ordered groups while preserving all values within each group."""
    if not grouped_values:
        raise ValueError("grouped_values must be non-empty")
    if block_length <= 0 or samples <= 0:
        raise ValueError("block_length and samples must be positive")
    keys = sorted(grouped_values)
    groups = [np.asarray(grouped_values[key], dtype=np.float64) for key in keys]
    if any(group.size == 0 for group in groups):
        raise ValueError("each bootstrap group must be non-empty")
    flat = np.concatenate(groups)
    if not np.isfinite(flat).all():
        raise ValueError("bootstrap values must be finite")
    rng = np.random.default_rng(seed)
    group_count = len(groups)
    block = min(block_length, group_count)
    estimates = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        selected: list[np.ndarray] = []
        selected_groups = 0
        while selected_groups < group_count:
            start = int(rng.integers(0, group_count))
            for offset in range(block):
                selected.append(groups[(start + offset) % group_count])
                selected_groups += 1
                if selected_groups == group_count:
                    break
        estimates[sample_index] = float(np.mean(np.concatenate(selected)))
    return {
        "count": int(flat.size),
        "group_count": group_count,
        "block_length": block,
        "samples": samples,
        "seed": seed,
        "mean": float(np.mean(flat)),
        "median": float(np.median(flat)),
        "positive_fraction": float(np.mean(flat > 0)),
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
    }


@dataclass(frozen=True)
class GoalCache:
    latents: Mapping[str, np.ndarray]
    images: Mapping[str, np.ndarray]


class ColorAblationBackend:
    """ODEWorld adapter that separates goal prediction from PT-Flow rollout."""

    def __init__(
        self,
        bundle: ModelBundle,
        *,
        decode_chunk_size: int = 16,
        ode_solver: str = "rk4",
        ode_rtol: float = 1e-5,
        ode_atol: float = 1e-6,
    ) -> None:
        if decode_chunk_size <= 0:
            raise ValueError("decode_chunk_size must be positive")
        if ode_solver not in {"rk4", "dopri5"}:
            raise ValueError(f"Unsupported ODE solver: {ode_solver}")
        if ode_rtol <= 0.0 or ode_atol <= 0.0:
            raise ValueError("ODE tolerances must be positive")
        self.bundle = bundle
        self.decode_chunk_size = decode_chunk_size
        self.ode_solver = ode_solver
        self.ode_rtol = float(ode_rtol)
        self.ode_atol = float(ode_atol)
        self.max_time_length = int(bundle.flow.max_time_length)
        if self.max_time_length != 50:
            raise RuntimeError(
                f"Expected PT-Flow max_time_length=50, got {self.max_time_length}"
            )

    def predict_goal(
        self,
        current: np.ndarray,
        instruction: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        import torch

        current_tensor = rgb_to_tensor(current, self.bundle.device)
        with torch.inference_mode():
            latent = self.bundle.goal_predictor.predict(
                current_tensor,
                [instruction],
            )
            if latent.ndim != 3 or latent.shape[0] != 1:
                raise RuntimeError(
                    f"Unexpected Goal Predictor latent shape: {tuple(latent.shape)}"
                )
            decoded = self.bundle.rae.decode(latent).clamp(0, 1)
            image = tensor_to_rgb(decoded[0])
        return latent[0].detach().float().cpu().numpy(), image

    def rollout_from_goal_latent(
        self,
        current: np.ndarray,
        goal_latent: Any,
        *,
        horizon: float,
        steps: int,
    ) -> list[np.ndarray]:
        import torch
        from torchdiffeq import odeint

        if steps <= 0 or horizon <= 0:
            raise ValueError("steps and horizon must be positive")
        current_tensor = rgb_to_tensor(current, self.bundle.device)
        flow = self.bundle.flow
        latent = torch.as_tensor(
            goal_latent,
            dtype=torch.float32,
            device=self.bundle.device,
        )
        if latent.ndim == 2:
            latent = latent.unsqueeze(0)
        if latent.ndim != 3 or latent.shape[0] != 1:
            raise ValueError(f"Unexpected cached goal latent shape: {tuple(latent.shape)}")

        with torch.inference_mode():
            state_zero = flow.latent_encode(current_tensor)
            if tuple(latent.shape) != tuple(state_zero.shape):
                raise ValueError(
                    "Goal/current latent shapes differ: "
                    f"goal={tuple(latent.shape)} current={tuple(state_zero.shape)}"
                )
            dynamic_zero = flow.delta_decouple(state_zero, state_zero)
            dynamic_goal = flow.delta_decouple(state_zero, latent)

            def ode_func(time_scalar: Any, dynamic_latent: Any) -> Any:
                physical_time = time_scalar.view(1, 1).expand(
                    current_tensor.shape[0],
                    1,
                )
                return (
                    flow.forward_vmodel(
                        dynamic_zero,
                        dynamic_latent,
                        dynamic_goal,
                        physical_time,
                    )
                    * flow.max_time_length
                )

            time_grid = torch.linspace(
                0,
                horizon,
                steps + 1,
                device=self.bundle.device,
            )
            ode_options = {}
            if self.ode_solver != "rk4":
                ode_options = {"rtol": self.ode_rtol, "atol": self.ode_atol}
            dynamic = odeint(
                ode_func,
                dynamic_zero,
                time_grid,
                method=self.ode_solver,
                **ode_options,
            )[1:]
            dynamic = dynamic.permute(1, 0, 2, 3)[0]
            reconstructed_chunks = []
            for start in range(0, steps, self.decode_chunk_size):
                chunk = dynamic[start : start + self.decode_chunk_size]
                state_chunk = state_zero.expand(chunk.shape[0], -1, -1)
                reconstructed_chunks.append(flow.delta_decode(state_chunk, chunk))
            reconstructed = torch.cat(reconstructed_chunks, dim=0)
            return decode_patch_latents(
                self.bundle.rae,
                reconstructed,
                chunk_size=self.decode_chunk_size,
            )


def _validate_actual_frames(frames: np.ndarray) -> None:
    expected_tail = (RESOLUTION, RESOLUTION, 3)
    if frames.ndim != 4 or frames.shape[1:] != expected_tail:
        raise ValueError(
            f"Expected source frames [N,{RESOLUTION},{RESOLUTION},3], got {frames.shape}"
        )
    if frames.dtype != np.uint8:
        raise ValueError(f"Expected source frames dtype uint8, got {frames.dtype}")
    if len(frames) < 2:
        raise ValueError("At least two source frames are required")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_source_frames(
    source_result: Path,
    *,
    max_transitions: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load only a previously verified framewise result; never launch LIBERO."""
    source_result = source_result.expanduser().resolve()
    frames_path = source_result / "actual_replay" / "frames.npz"
    diagnostics_path = source_result / "actual_replay" / "replay_diagnostics.json"
    for path in (frames_path, diagnostics_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required source artifact: {path}")
    promotion_evidence = validate_promoted_result_source(
        source_result,
        expected_classification="partial rollout",
        expected_transitions=EXPECTED_SOURCE_FRAMES - 1,
    )
    verification_path = promotion_evidence["verification_path"]
    verification = promotion_evidence["verification"]
    if verification.get("external_renderer_verified") is not True:
        raise ValueError("Source was not generated by a verified external renderer")
    if verification.get("external_renderer_required") is not True:
        raise ValueError("Source verification did not require the external renderer")
    frames_sha256 = _sha256_file(frames_path)
    if verification.get("actual_replay_frames_sha256") != frames_sha256:
        raise ValueError(
            "Current source frames do not match the upstream framewise verification"
        )
    artifact_hashes = verification.get("artifact_hashes")
    if not isinstance(artifact_hashes, Mapping) or artifact_hashes.get(
        "actual_replay/frames.npz"
    ) != frames_sha256:
        raise ValueError("Source frames are not covered by the upstream artifact seal")
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    source_env_steps = int(diagnostics.get("env_step_calls", -1))
    if source_env_steps != EXPECTED_SOURCE_FRAMES - 1:
        raise ValueError(
            f"Expected source env_step_calls=151, got {source_env_steps}"
        )
    with np.load(frames_path, allow_pickle=False) as archive:
        if set(archive.files) != {"frames"}:
            raise ValueError(f"Unexpected source NPZ keys: {archive.files}")
        full_frames = np.asarray(archive["frames"])
    _validate_actual_frames(full_frames)
    if full_frames.shape != (
        EXPECTED_SOURCE_FRAMES,
        RESOLUTION,
        RESOLUTION,
        3,
    ):
        raise ValueError(f"Unexpected verified source frame shape: {full_frames.shape}")
    if max_transitions < 1 or max_transitions > EXPECTED_SOURCE_FRAMES - 1:
        raise ValueError("max_transitions must be in [1,151]")
    return full_frames[: max_transitions + 1].copy(), {
        "result_dir": str(source_result),
        "frames_path": str(frames_path),
        "frames_sha256": frames_sha256,
        "verification_path": str(verification_path),
        "verification_sha256": promotion_evidence["verification_sha256"],
        "promotion_path": str(promotion_evidence["promotion_path"]),
        "promotion_sha256": promotion_evidence["promotion_sha256"],
        "promotion_verified": True,
        "verification_all_checks_passed": True,
        "source_classification": verification["classification"],
        "source_env_step_calls": source_env_steps,
        "source_frame_count": int(full_frames.shape[0]),
    }


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing NPZ: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _optional_finite(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _relative_output(path: Path, output_dir: Path) -> str:
    return path.relative_to(output_dir).as_posix()


def _color_fields(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> dict[str, Any]:
    mask = build_red_mask(reference)
    if mask is None:
        return {
            "mask_valid": False,
            "mask_pixels": None,
            "red_score": None,
            "red_fraction": None,
            "neutrality": None,
            "brightness": None,
            "silver_score": None,
        }
    return {"mask_valid": True, **color_scores(prediction, mask)}


def prediction_metric_row(
    *,
    mode: str,
    condition: str,
    start_index: int,
    lead_steps: int,
    target_index: int,
    reference: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    _validate_rgb(prediction, name="prediction")
    metrics = stable_frame_metrics(reference, prediction)
    return {
        "mode": mode,
        "condition": condition,
        "start_index": start_index,
        "lead_steps": lead_steps,
        "target_index": target_index,
        "psnr_db": _optional_finite(metrics["psnr_db"]),
        "l1": metrics["l1"],
        "mse": metrics["mse"],
        "lpips": None,
        **_color_fields(prediction, reference),
    }


def _validate_predictions(
    predictions: Sequence[np.ndarray],
    *,
    expected: int,
    condition: str,
) -> None:
    if len(predictions) != expected:
        raise RuntimeError(
            f"{condition} returned {len(predictions)} frames, expected {expected}"
        )
    for frame in predictions:
        _validate_rgb(frame, name=f"{condition} prediction")
        if frame.shape != (RESOLUTION, RESOLUTION, 3):
            raise ValueError(f"Unexpected prediction shape: {frame.shape}")


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        condition_result: dict[str, Any] = {"count": len(selected)}
        for metric in (
            "psnr_db",
            "l1",
            "mse",
            "red_score",
            "red_fraction",
            "neutrality",
            "brightness",
            "silver_score",
        ):
            values = [
                float(row[metric])
                for row in selected
                if row.get(metric) is not None
                and math.isfinite(float(row[metric]))
            ]
            condition_result[metric] = (
                {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
                if values
                else None
            )
        result[condition] = condition_result
    return result


def paired_effect_rows_for_key(
    *,
    mode: str,
    start_index: int,
    lead_steps: int,
    target_index: int,
    condition_rows: Mapping[str, Mapping[str, Any]],
    condition_images: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    if set(condition_rows) != set(CONDITIONS) or set(condition_images) != set(CONDITIONS):
        raise ValueError("Paired effects require all three conditions")
    result = []
    for comparison, (positive_name, baseline_name) in PAIR_DEFINITIONS.items():
        positive = condition_rows[positive_name]
        baseline = condition_rows[baseline_name]
        rgb = pairwise_rgb_metrics(
            condition_images[positive_name],
            condition_images[baseline_name],
        )
        result.append(
            {
                "mode": mode,
                "comparison": comparison,
                "start_index": start_index,
                "lead_steps": lead_steps,
                "target_index": target_index,
                "red_score_delta": (
                    float(positive["red_score"]) - float(baseline["red_score"])
                    if positive["red_score"] is not None
                    and baseline["red_score"] is not None
                    else None
                ),
                "silver_score_delta": (
                    float(positive["silver_score"])
                    - float(baseline["silver_score"])
                    if positive["silver_score"] is not None
                    and baseline["silver_score"] is not None
                    else None
                ),
                "rgb_l1": rgb["rgb_l1"],
                "psnr_db": _optional_finite(rgb["psnr_db"]),
                "changed_fraction": rgb["changed_fraction"],
            }
        )
    return result


def run_goal_probe(
    backend: Any,
    actual_frames: np.ndarray,
    output_dir: Path,
    *,
    fps: int,
) -> tuple[
    GoalCache,
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Predict and cache all three language goals for every real current frame."""
    _validate_actual_frames(actual_frames)
    if int(backend.max_time_length) != 50:
        raise RuntimeError("Goal probe requires max_time_length=50")
    transition_count = len(actual_frames) - 1
    mode_dir = output_dir / "goal_probe"
    mode_dir.mkdir(parents=True, exist_ok=True)
    video_path = mode_dir / "goal_comparison.mp4"
    gif_path = mode_dir / "goal_comparison.gif"
    latent_path = mode_dir / "goal_latents.npz"
    image_path = mode_dir / "goal_images.npz"
    latent_metrics_path = mode_dir / "latent_metrics.csv"
    metrics_path = mode_dir / "metrics.csv"
    summary_path = mode_dir / "summary.json"
    latent_lists: dict[str, list[np.ndarray]] = {name: [] for name in CONDITIONS}
    image_lists: dict[str, list[np.ndarray]] = {name: [] for name in CONDITIONS}
    metric_rows: list[dict[str, Any]] = []
    latent_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    panels: list[np.ndarray] = []
    goal_reference = actual_frames[-1]
    started = time.perf_counter()

    with VideoSink(
        video_path,
        fps=fps,
        frame_size=(RESOLUTION * 4, RESOLUTION),
    ) as sink:
        for start_index in range(transition_count):
            current = actual_frames[start_index]
            current_latents: dict[str, np.ndarray] = {}
            current_images: dict[str, np.ndarray] = {}
            current_metric_rows: dict[str, dict[str, Any]] = {}
            for condition in CONDITIONS:
                latent, image = backend.predict_goal(current, PROMPTS[condition])
                latent = np.asarray(latent, dtype=np.float32)
                if latent.ndim != 2 or not np.isfinite(latent).all():
                    raise ValueError(
                        f"Invalid {condition} goal latent at {start_index}: {latent.shape}"
                    )
                _validate_rgb(image, name=f"{condition} decoded goal")
                if image.shape != (RESOLUTION, RESOLUTION, 3):
                    raise ValueError(f"Unexpected decoded goal shape: {image.shape}")
                current_latents[condition] = latent
                current_images[condition] = image
                latent_lists[condition].append(latent)
                image_lists[condition].append(image)
                metric_row = prediction_metric_row(
                        mode="goal_probe",
                        condition=condition,
                        start_index=start_index,
                        lead_steps=transition_count - start_index,
                        target_index=transition_count,
                        reference=goal_reference,
                        prediction=image,
                    )
                metric_rows.append(metric_row)
                current_metric_rows[condition] = metric_row

            latent_shapes = {value.shape for value in current_latents.values()}
            if len(latent_shapes) != 1:
                raise ValueError(
                    f"Goal latent shapes differ at start {start_index}: {latent_shapes}"
                )
            for pair_name, (first_name, second_name) in LATENT_PAIR_DEFINITIONS.items():
                latent_metrics = latent_pair_metrics(
                    current_latents[first_name],
                    current_latents[second_name],
                )
                rgb_metrics = pairwise_rgb_metrics(
                    current_images[first_name],
                    current_images[second_name],
                )
                latent_rows.append(
                    {
                        "start_index": start_index,
                        "comparison": pair_name,
                        **latent_metrics,
                        "rgb_l1": rgb_metrics["rgb_l1"],
                        "psnr_db": _optional_finite(rgb_metrics["psnr_db"]),
                        "changed_fraction": rgb_metrics["changed_fraction"],
                    }
                )
            effect_rows.extend(
                paired_effect_rows_for_key(
                    mode="goal_probe",
                    start_index=start_index,
                    lead_steps=transition_count - start_index,
                    target_index=transition_count,
                    condition_rows=current_metric_rows,
                    condition_images=current_images,
                )
            )
            panel = make_labelled_panel(
                [current, *(current_images[name] for name in CONDITIONS)],
                [
                    f"Actual current t={start_index}",
                    "Neutral goal",
                    "Red goal",
                    "Silver goal",
                ],
            )
            sink.write(panel)
            panels.append(panel)

    latents = {
        name: np.stack(latent_lists[name], axis=0).astype(np.float32, copy=False)
        for name in CONDITIONS
    }
    images = {
        name: np.stack(image_lists[name], axis=0)
        for name in CONDITIONS
    }
    _write_npz(
        latent_path,
        {
            "start_indices": np.arange(transition_count, dtype=np.int32),
            **latents,
        },
    )
    _write_npz(image_path, images)
    write_gif(gif_path, panels, fps=fps)
    write_csv(latent_metrics_path, latent_rows)
    write_csv(metrics_path, metric_rows)
    summary = {
        "classification": CLASSIFICATION,
        "scope": {
            "description": SCOPE_DESCRIPTION,
            "env_step_calls": 0,
            "robot_actions_executed": False,
            "success_rate_measured": False,
        },
        "mode": "goal_probe",
        "transition_count": transition_count,
        "prompts": PROMPTS,
        "goal_count": transition_count * len(CONDITIONS),
        "goals_per_condition": transition_count,
        "latent_metric_rows": len(latent_rows),
        "metric_rows": len(metric_rows),
        "video_frames": len(panels),
        "latent_shape_per_goal": list(next(iter(latents.values())).shape[1:]),
        "timing_seconds": time.perf_counter() - started,
        "metrics": aggregate_metrics(metric_rows),
        "pair_metrics": {
            pair: {
                "count": sum(row["comparison"] == pair for row in latent_rows),
                "cosine_distance_mean": float(
                    np.mean(
                        [
                            row["cosine_distance"]
                            for row in latent_rows
                            if row["comparison"] == pair
                        ]
                    )
                ),
                "normalized_l2_mean": float(
                    np.mean(
                        [
                            row["normalized_l2"]
                            for row in latent_rows
                            if row["comparison"] == pair
                        ]
                    )
                ),
            }
            for pair in LATENT_PAIR_DEFINITIONS
        },
        "outputs": {
            "latents": _relative_output(latent_path, output_dir),
            "raw_predictions": _relative_output(image_path, output_dir),
            "video": _relative_output(video_path, output_dir),
            "gif": _relative_output(gif_path, output_dir),
            "latent_metrics": _relative_output(latent_metrics_path, output_dir),
            "metrics": _relative_output(metrics_path, output_dir),
        },
    }
    write_json(summary_path, summary)
    return GoalCache(latents=latents, images=images), summary, metric_rows, effect_rows


def run_teacher_forced(
    backend: Any,
    actual_frames: np.ndarray,
    goal_cache: GoalCache,
    output_dir: Path,
    *,
    fps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run three one-step prompt conditions from the same actual current frame."""
    _validate_actual_frames(actual_frames)
    transition_count = len(actual_frames) - 1
    mode_dir = output_dir / "teacher_forced"
    mode_dir.mkdir(parents=True, exist_ok=True)
    video_path = mode_dir / "prompt_comparison.mp4"
    gif_path = mode_dir / "prompt_comparison.gif"
    metrics_path = mode_dir / "metrics.csv"
    summary_path = mode_dir / "summary.json"
    horizon = one_step_horizon(int(backend.max_time_length))
    rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    prediction_lists: dict[str, list[np.ndarray]] = {
        name: [] for name in CONDITIONS
    }
    panels: list[np.ndarray] = []
    started = time.perf_counter()

    with VideoSink(
        video_path,
        fps=fps,
        frame_size=(RESOLUTION * 5, RESOLUTION),
    ) as sink:
        for start_index in range(transition_count):
            current = actual_frames[start_index]
            target = actual_frames[start_index + 1]
            predictions: dict[str, np.ndarray] = {}
            current_metric_rows: dict[str, dict[str, Any]] = {}
            for condition in CONDITIONS:
                generated = backend.rollout_from_goal_latent(
                    current,
                    goal_cache.latents[condition][start_index],
                    horizon=horizon,
                    steps=1,
                )
                _validate_predictions(generated, expected=1, condition=condition)
                prediction = generated[0]
                predictions[condition] = prediction
                prediction_lists[condition].append(prediction)
                metric_row = prediction_metric_row(
                        mode="teacher_forced",
                        condition=condition,
                        start_index=start_index,
                        lead_steps=1,
                        target_index=start_index + 1,
                        reference=target,
                        prediction=prediction,
                    )
                rows.append(metric_row)
                current_metric_rows[condition] = metric_row
            effect_rows.extend(
                paired_effect_rows_for_key(
                    mode="teacher_forced",
                    start_index=start_index,
                    lead_steps=1,
                    target_index=start_index + 1,
                    condition_rows=current_metric_rows,
                    condition_images=predictions,
                )
            )
            panel = make_labelled_panel(
                [
                    current,
                    *(predictions[name] for name in CONDITIONS),
                    target,
                ],
                [
                    f"Actual current t={start_index}",
                    "Neutral next",
                    "Red next",
                    "Silver next",
                    f"Actual next t={start_index + 1}",
                ],
            )
            sink.write(panel)
            panels.append(panel)

    write_gif(gif_path, panels, fps=fps)
    write_csv(metrics_path, rows)
    predictions_path = mode_dir / "predictions.npz"
    _write_npz(
        predictions_path,
        {name: np.stack(prediction_lists[name], axis=0) for name in CONDITIONS},
    )
    summary = {
        "classification": CLASSIFICATION,
        "scope": {
            "description": SCOPE_DESCRIPTION,
            "env_step_calls": 0,
            "robot_actions_executed": False,
            "success_rate_measured": False,
        },
        "mode": "teacher_forced",
        "transition_count": transition_count,
        "prompts": PROMPTS,
        "metric_rows": len(rows),
        "video_frames": len(panels),
        "horizon": horizon,
        "steps_per_prediction": 1,
        "timing_seconds": time.perf_counter() - started,
        "metrics": aggregate_metrics(rows),
        "outputs": {
            "video": _relative_output(video_path, output_dir),
            "gif": _relative_output(gif_path, output_dir),
            "metrics": _relative_output(metrics_path, output_dir),
            "raw_predictions": _relative_output(predictions_path, output_dir),
        },
    }
    write_json(summary_path, summary)
    return summary, rows, effect_rows


def run_full_future(
    backend: Any,
    actual_frames: np.ndarray,
    goal_cache: GoalCache,
    output_dir: Path,
    *,
    fps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run all remaining lead times from every actual current frame."""
    _validate_actual_frames(actual_frames)
    transition_count = len(actual_frames) - 1
    schedule = full_future_schedule(
        transition_count,
        int(backend.max_time_length),
    )
    mode_dir = output_dir / "full_future"
    condition_dirs = {name: mode_dir / name for name in CONDITIONS}
    for directory in condition_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    video_path = mode_dir / "endpoint_comparison.mp4"
    gif_path = mode_dir / "endpoint_comparison.gif"
    metrics_path = mode_dir / "metrics.csv"
    summary_path = mode_dir / "summary.json"
    rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    panels: list[np.ndarray] = []
    timing = {name: 0.0 for name in CONDITIONS}
    final_target = actual_frames[-1]
    started = time.perf_counter()

    with VideoSink(
        video_path,
        fps=fps,
        frame_size=(RESOLUTION * 5, RESOLUTION),
    ) as endpoint_sink:
        for start_index, steps, horizon in schedule:
            current = actual_frames[start_index]
            endpoints: dict[str, np.ndarray] = {}
            condition_predictions: dict[str, list[np.ndarray]] = {}
            condition_metric_rows: dict[str, list[dict[str, Any]]] = {}
            for condition in CONDITIONS:
                condition_started = time.perf_counter()
                predictions = backend.rollout_from_goal_latent(
                    current,
                    goal_cache.latents[condition][start_index],
                    horizon=horizon,
                    steps=steps,
                )
                timing[condition] += time.perf_counter() - condition_started
                _validate_predictions(
                    predictions,
                    expected=steps,
                    condition=condition,
                )
                condition_predictions[condition] = predictions
                condition_metric_rows[condition] = []
                clip_path = condition_dirs[condition] / f"frame_{start_index:03d}.mp4"
                with VideoSink(
                    clip_path,
                    fps=fps,
                    frame_size=(RESOLUTION, RESOLUTION),
                ) as clip_sink:
                    for lead_steps, prediction in enumerate(predictions, start=1):
                        clip_sink.write(prediction)
                        target_index = start_index + lead_steps
                        metric_row = prediction_metric_row(
                                mode="full_future",
                                condition=condition,
                                start_index=start_index,
                                lead_steps=lead_steps,
                                target_index=target_index,
                                reference=actual_frames[target_index],
                                prediction=prediction,
                            )
                        rows.append(metric_row)
                        condition_metric_rows[condition].append(metric_row)
                _write_npz(
                    clip_path.with_suffix(".npz"),
                    {"frames": np.stack(predictions, axis=0)},
                )
                endpoints[condition] = predictions[-1]

            for lead_index in range(steps):
                effect_rows.extend(
                    paired_effect_rows_for_key(
                        mode="full_future",
                        start_index=start_index,
                        lead_steps=lead_index + 1,
                        target_index=start_index + lead_index + 1,
                        condition_rows={
                            condition: condition_metric_rows[condition][lead_index]
                            for condition in CONDITIONS
                        },
                        condition_images={
                            condition: condition_predictions[condition][lead_index]
                            for condition in CONDITIONS
                        },
                    )
                )

            panel = make_labelled_panel(
                [
                    current,
                    *(endpoints[name] for name in CONDITIONS),
                    final_target,
                ],
                [
                    f"Actual current t={start_index}",
                    "Neutral endpoint",
                    "Red endpoint",
                    "Silver endpoint",
                    f"Actual final t={transition_count}",
                ],
            )
            endpoint_sink.write(panel)
            panels.append(panel)

    write_gif(gif_path, panels, fps=fps)
    write_csv(metrics_path, rows)
    frames_per_condition = sum(steps for _, steps, _ in schedule)
    summary = {
        "classification": CLASSIFICATION,
        "scope": {
            "description": SCOPE_DESCRIPTION,
            "env_step_calls": 0,
            "robot_actions_executed": False,
            "success_rate_measured": False,
        },
        "stress_test_label": "long-integration stress test",
        "mode": "full_future",
        "transition_count": transition_count,
        "prompts": PROMPTS,
        "clips_per_condition": transition_count,
        "total_clips": transition_count * len(CONDITIONS),
        "predicted_frames_per_condition": frames_per_condition,
        "metric_rows": len(rows),
        "endpoint_video_frames": len(panels),
        "maximum_horizon": schedule[0][2],
        "timing_seconds": {
            "total": time.perf_counter() - started,
            **timing,
        },
        "metrics": aggregate_metrics(rows),
        "outputs": {
            "condition_clip_directories": {
                name: _relative_output(path, output_dir)
                for name, path in condition_dirs.items()
            },
            "video": _relative_output(video_path, output_dir),
            "gif": _relative_output(gif_path, output_dir),
            "metrics": _relative_output(metrics_path, output_dir),
            "raw_predictions": "full_future/<condition>/frame_<start>.npz",
        },
    }
    write_json(summary_path, summary)
    return summary, rows, effect_rows


def _color_response_panel(
    neutral: np.ndarray,
    red: np.ndarray,
    silver: np.ndarray,
) -> np.ndarray:
    red_silver = fixed_error_heatmap(red, silver)
    neutral_red = fixed_error_heatmap(neutral, red)
    top = cv2.resize(
        neutral_red,
        (RESOLUTION, RESOLUTION // 2),
        interpolation=cv2.INTER_AREA,
    )
    bottom = cv2.resize(
        red_silver,
        (RESOLUTION, RESOLUTION // 2),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.concatenate([top, bottom], axis=0)
    cv2.putText(
        panel,
        "Neutral vs red",
        (7, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "Red vs silver",
        (7, RESOLUTION // 2 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.47,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def run_autoregressive(
    backend: Any,
    actual_frames: np.ndarray,
    output_dir: Path,
    *,
    fps: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Feed three prompt-specific decoded predictions back independently."""
    _validate_actual_frames(actual_frames)
    transition_count = len(actual_frames) - 1
    mode_dir = output_dir / "autoregressive"
    mode_dir.mkdir(parents=True, exist_ok=True)
    video_path = mode_dir / "prompt_comparison.mp4"
    gif_path = mode_dir / "prompt_comparison.gif"
    metrics_path = mode_dir / "metrics.csv"
    summary_path = mode_dir / "summary.json"
    horizon = one_step_horizon(int(backend.max_time_length))
    states = {name: actual_frames[0].copy() for name in CONDITIONS}
    rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    prediction_lists: dict[str, list[np.ndarray]] = {
        name: [] for name in CONDITIONS
    }
    panels: list[np.ndarray] = []
    timing = {name: 0.0 for name in CONDITIONS}
    initial_response = _color_response_panel(
        states["neutral"],
        states["red"],
        states["silver"],
    )
    initial_panel = make_labelled_panel(
        [
            actual_frames[0],
            *(states[name] for name in CONDITIONS),
            initial_response,
        ],
        ["Actual t=0", "Neutral AR t=0", "Red AR t=0", "Silver AR t=0", "Prompt response"],
    )
    started = time.perf_counter()

    with VideoSink(
        video_path,
        fps=fps,
        frame_size=(RESOLUTION * 5, RESOLUTION),
    ) as sink:
        sink.write(initial_panel)
        panels.append(initial_panel)
        for index in range(transition_count):
            target = actual_frames[index + 1]
            next_states: dict[str, np.ndarray] = {}
            current_metric_rows: dict[str, dict[str, Any]] = {}
            for condition in CONDITIONS:
                condition_started = time.perf_counter()
                goal_latent, _ = backend.predict_goal(
                    states[condition],
                    PROMPTS[condition],
                )
                predictions = backend.rollout_from_goal_latent(
                    states[condition],
                    goal_latent,
                    horizon=horizon,
                    steps=1,
                )
                timing[condition] += time.perf_counter() - condition_started
                _validate_predictions(
                    predictions,
                    expected=1,
                    condition=condition,
                )
                prediction = predictions[0]
                next_states[condition] = prediction
                prediction_lists[condition].append(prediction)
                metric_row = prediction_metric_row(
                        mode="autoregressive",
                        condition=condition,
                        start_index=0,
                        lead_steps=index + 1,
                        target_index=index + 1,
                        reference=target,
                        prediction=prediction,
                    )
                rows.append(metric_row)
                current_metric_rows[condition] = metric_row
            states = next_states
            effect_rows.extend(
                paired_effect_rows_for_key(
                    mode="autoregressive",
                    start_index=0,
                    lead_steps=index + 1,
                    target_index=index + 1,
                    condition_rows=current_metric_rows,
                    condition_images=states,
                )
            )
            response = _color_response_panel(
                states["neutral"],
                states["red"],
                states["silver"],
            )
            panel = make_labelled_panel(
                [
                    target,
                    *(states[name] for name in CONDITIONS),
                    response,
                ],
                [
                    f"Actual t={index + 1}",
                    f"Neutral AR t={index + 1}",
                    f"Red AR t={index + 1}",
                    f"Silver AR t={index + 1}",
                    "Prompt response",
                ],
            )
            sink.write(panel)
            panels.append(panel)

    write_gif(gif_path, panels, fps=fps)
    write_csv(metrics_path, rows)
    predictions_path = mode_dir / "predictions.npz"
    _write_npz(
        predictions_path,
        {name: np.stack(prediction_lists[name], axis=0) for name in CONDITIONS},
    )
    summary = {
        "classification": CLASSIFICATION,
        "scope": {
            "description": SCOPE_DESCRIPTION,
            "env_step_calls": 0,
            "robot_actions_executed": False,
            "success_rate_measured": False,
        },
        "mode": "autoregressive",
        "transition_count": transition_count,
        "prompts": PROMPTS,
        "metric_rows": len(rows),
        "video_frames": len(panels),
        "horizon": horizon,
        "rgb_feedback": True,
        "timing_seconds": {
            "total": time.perf_counter() - started,
            **timing,
        },
        "metrics": aggregate_metrics(rows),
        "outputs": {
            "video": _relative_output(video_path, output_dir),
            "gif": _relative_output(gif_path, output_dir),
            "metrics": _relative_output(metrics_path, output_dir),
            "raw_predictions": _relative_output(predictions_path, output_dir),
        },
    }
    write_json(summary_path, summary)
    return summary, rows, effect_rows


def build_paired_prompt_effects(
    mode_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    effects: list[dict[str, Any]] = []
    for mode in MODE_ORDER:
        rows = mode_rows[mode]
        grouped: dict[tuple[int, int, int], dict[str, Mapping[str, Any]]] = {}
        for row in rows:
            key = (
                int(row["start_index"]),
                int(row["lead_steps"]),
                int(row["target_index"]),
            )
            condition = str(row["condition"])
            if condition in grouped.setdefault(key, {}):
                raise ValueError(f"Duplicate condition row for {mode} {key} {condition}")
            grouped[key][condition] = row
        for key in sorted(grouped):
            by_condition = grouped[key]
            if set(by_condition) != set(CONDITIONS):
                raise ValueError(
                    f"Incomplete prompt pairing for {mode} {key}: {sorted(by_condition)}"
                )
            predictions = by_condition
            for comparison, (positive_name, baseline_name) in PAIR_DEFINITIONS.items():
                positive = predictions[positive_name]
                baseline = predictions[baseline_name]
                positive_image = positive.get("_prediction")
                baseline_image = baseline.get("_prediction")
                rgb = (
                    pairwise_rgb_metrics(positive_image, baseline_image)
                    if isinstance(positive_image, np.ndarray)
                    and isinstance(baseline_image, np.ndarray)
                    else None
                )
                red_delta = (
                    float(positive["red_score"]) - float(baseline["red_score"])
                    if positive.get("red_score") is not None
                    and baseline.get("red_score") is not None
                    else None
                )
                silver_delta = (
                    float(positive["silver_score"])
                    - float(baseline["silver_score"])
                    if positive.get("silver_score") is not None
                    and baseline.get("silver_score") is not None
                    else None
                )
                effects.append(
                    {
                        "mode": mode,
                        "comparison": comparison,
                        "start_index": key[0],
                        "lead_steps": key[1],
                        "target_index": key[2],
                        "red_score_delta": red_delta,
                        "silver_score_delta": silver_delta,
                        "rgb_l1": rgb["rgb_l1"] if rgb is not None else None,
                        "psnr_db": (
                            _optional_finite(rgb["psnr_db"])
                            if rgb is not None
                            else None
                        ),
                        "changed_fraction": (
                            rgb["changed_fraction"] if rgb is not None else None
                        ),
                    }
                )
    return effects


def bootstrap_effects(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for mode in MODE_ORDER:
        mode_summary: dict[str, Any] = {}
        for comparison in PAIR_DEFINITIONS:
            selected = [
                row
                for row in rows
                if row["mode"] == mode and row["comparison"] == comparison
            ]
            comparison_summary: dict[str, Any] = {}
            for metric in (
                "red_score_delta",
                "silver_score_delta",
                "rgb_l1",
                "psnr_db",
                "changed_fraction",
            ):
                grouped: dict[int, list[float]] = {}
                for row in selected:
                    if row[metric] is None:
                        continue
                    group_key = (
                        int(row["start_index"])
                        if mode in {"goal_probe", "full_future"}
                        else int(row["target_index"])
                    )
                    grouped.setdefault(group_key, []).append(float(row[metric]))
                comparison_summary[metric] = (
                    moving_block_interval(
                        grouped,
                        block_length=10,
                        samples=2000,
                        seed=SEED,
                    )
                    if grouped
                    else None
                )
            mode_summary[comparison] = comparison_summary
        summaries[mode] = mode_summary
    return summaries


def _bounded_transitions(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > EXPECTED_SOURCE_FRAMES - 1:
        raise argparse.ArgumentTypeError("max transitions must be in [1,151]")
    return parsed


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-result", type=Path, default=SOURCE_RESULT)
    parser.add_argument(
        "--max-transitions",
        type=_bounded_transitions,
        default=EXPECTED_SOURCE_FRAMES - 1,
    )
    parser.add_argument("--fps", type=_positive_integer, default=FPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--decode-chunk-size",
        type=_positive_integer,
        default=16,
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def validate_output_directory(output_dir: Path) -> None:
    if not output_dir.exists():
        return
    conflicts = [
        path
        for path in output_dir.iterdir()
        if not (
            path.is_file()
            and path.name.startswith("slurm-")
            and path.suffix in {".out", ".err"}
        )
    ]
    if conflicts:
        raise FileExistsError(
            f"Refusing to overwrite existing result artifacts: {conflicts}"
        )


def _checkpoint_provenance(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    weights_path = path / "model.safetensors"
    return {
        "path": str(path),
        "config": json.loads(config_path.read_text(encoding="utf-8")),
        "config_sha256": _sha256_file(config_path),
        "weights_bytes": weights_path.stat().st_size,
        "weights_sha256": _sha256_file(weights_path),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_metadata() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {"commit": commit, "dirty": bool(status.strip())}


def collect_provenance(
    args: argparse.Namespace,
    source: Mapping[str, Any],
    *,
    peak_gpu_memory_bytes: int,
) -> dict[str, Any]:
    import torch

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "classification": CLASSIFICATION,
        "scope": {
            "description": SCOPE_DESCRIPTION,
            "env_step_calls": 0,
            "robot_actions_executed": False,
            "libero_environment_created": False,
            "success_rate_measured": False,
            "source_is_verified_partial_rollout": True,
            "long_integration_stress_test": True,
        },
        "prompts": PROMPTS,
        "source": dict(source),
        "checkpoints": {
            "pt_flow": _checkpoint_provenance(PT_FLOW_CHECKPOINT),
            "rae": _checkpoint_provenance(RAE_CHECKPOINT),
            "goal_predictor": _checkpoint_provenance(GOAL_PREDICTOR_CHECKPOINT),
        },
        "inference": {
            "dtype": "float32",
            "ode_solver": "rk4",
            "max_time_length": 50,
            "one_step_horizon": 0.02,
            "prompt_conditions_batched": False,
            "goal_cache_reused_by": ["teacher_forced", "full_future"],
            "image_goal_rerun": False,
            "lpips": {
                "enabled": False,
                "reason": "not requested for this color-response ablation",
            },
        },
        "git": _git_metadata(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "host": platform.node(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "torchdiffeq": _package_version("torchdiffeq"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_node": os.environ.get("SLURMD_NODENAME"),
            "gpu": {
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
                "peak_allocated_memory_bytes": peak_gpu_memory_bytes,
            },
        },
        "arguments": {
            "execution_output_dir": str(args.output_dir),
            "artifact_paths_are_relative": True,
            "source_result": str(args.source_result),
            "max_transitions": args.max_transitions,
            "fps": args.fps,
            "seed": args.seed,
            "decode_chunk_size": args.decode_chunk_size,
            "device": args.device,
        },
    }


def validate_runtime(args: argparse.Namespace) -> None:
    import torch

    if args.device != "cuda":
        raise RuntimeError("GPU inference must use --device cuda")
    if args.seed != SEED:
        raise RuntimeError(f"This paired ablation requires seed={SEED}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; submit this runner through Slurm")
    checkpoint_files = []
    for checkpoint in (
        PT_FLOW_CHECKPOINT,
        RAE_CHECKPOINT,
        GOAL_PREDICTOR_CHECKPOINT,
    ):
        checkpoint_files.extend(
            [checkpoint / "config.json", checkpoint / "model.safetensors"]
        )
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch"))
    missing = [path for path in checkpoint_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing local checkpoint/cache files: {missing}")
    validate_dino_cache(torch_home)


def _set_run_state(
    output_dir: Path,
    *,
    status: str,
    args: argparse.Namespace,
    error: BaseException | None = None,
) -> None:
    if status not in {"running", "completed", "failed"}:
        raise ValueError(f"Unsupported run state: {status}")
    path = output_dir / "run_state.json"
    now = datetime.now(timezone.utc).isoformat()
    previous: Mapping[str, Any] = {}
    if path.is_file():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, Mapping):
            previous = loaded
    write_json(
        path,
        {
            "classification": CLASSIFICATION,
            "scope": SCOPE_DESCRIPTION,
            "status": status,
            "env_step_calls": 0,
            "success_rate_measured": False,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_node": os.environ.get("SLURMD_NODENAME"),
            "started_at_utc": previous.get("started_at_utc", now),
            "completed_at_utc": now if status == "completed" else None,
            "failed_at_utc": now if status == "failed" else None,
            "max_transitions": args.max_transitions,
            "prompts": PROMPTS,
            "error": (
                {"type": type(error).__name__, "message": str(error)}
                if error is not None
                else None
            ),
        },
        overwrite=True,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(f"classification={CLASSIFICATION}", flush=True)
    print(f"scope={SCOPE_DESCRIPTION}; env.step=0; no success rate", flush=True)
    print(f"prompts={json.dumps(PROMPTS)}", flush=True)
    validate_output_directory(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _set_run_state(args.output_dir, status="running", args=args)
    started = time.perf_counter()
    try:
        validate_runtime(args)
        import torch

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
        actual_frames, source = load_source_frames(
            args.source_result,
            max_transitions=args.max_transitions,
        )
        backend = ColorAblationBackend(
            load_models(torch.device(args.device)),
            decode_chunk_size=args.decode_chunk_size,
        )
        print("models_loaded=3 once=true", flush=True)

        goal_cache, goal_summary, goal_rows, goal_effects = run_goal_probe(
            backend,
            actual_frames,
            args.output_dir,
            fps=args.fps,
        )
        print("completed_mode=goal_probe", flush=True)
        teacher_summary, teacher_rows, teacher_effects = run_teacher_forced(
            backend,
            actual_frames,
            goal_cache,
            args.output_dir,
            fps=args.fps,
        )
        print("completed_mode=teacher_forced", flush=True)
        full_summary, full_rows, full_effects = run_full_future(
            backend,
            actual_frames,
            goal_cache,
            args.output_dir,
            fps=args.fps,
        )
        print("completed_mode=full_future", flush=True)
        autoregressive_summary, autoregressive_rows, autoregressive_effects = (
            run_autoregressive(
                backend,
                actual_frames,
                args.output_dir,
                fps=args.fps,
            )
        )
        print("completed_mode=autoregressive", flush=True)
        effect_rows = [
            *goal_effects,
            *teacher_effects,
            *full_effects,
            *autoregressive_effects,
        ]
        write_csv(args.output_dir / "paired_prompt_effects.csv", effect_rows)
        effect_bootstrap = bootstrap_effects(effect_rows)
        torch.cuda.synchronize()
        peak_memory = int(torch.cuda.max_memory_allocated())
        provenance = collect_provenance(
            args,
            source,
            peak_gpu_memory_bytes=peak_memory,
        )
        write_json(args.output_dir / "provenance.json", provenance)
        mode_summaries = {
            "goal_probe": goal_summary,
            "teacher_forced": teacher_summary,
            "full_future": full_summary,
            "autoregressive": autoregressive_summary,
        }
        summary = {
            "classification": CLASSIFICATION,
            "scope": {
                "description": SCOPE_DESCRIPTION,
                "env_step_calls": 0,
                "robot_actions_executed": False,
                "success_rate_measured": False,
            },
            "transition_count": args.max_transitions,
            "source_frame_count_used": len(actual_frames),
            "completed_modes": list(MODE_ORDER),
            "prompts": PROMPTS,
            "bootstrap": {
                "block_length": 10,
                "samples": 2000,
                "seed": SEED,
            },
            "paired_effect_summaries": effect_bootstrap,
            "paired_effect_rows": len(effect_rows),
            "lpips": provenance["inference"]["lpips"],
            "runtime": {
                key: provenance["runtime"][key]
                for key in ("host", "slurm_job_id", "slurm_node")
            },
            "timing_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": peak_memory,
            "mode_summaries": mode_summaries,
        }
        write_json(args.output_dir / "summary.json", summary)
        _set_run_state(args.output_dir, status="completed", args=args)
        print(
            f"completed=true transitions={args.max_transitions} "
            f"paired_effect_rows={len(effect_rows)}",
            flush=True,
        )
    except BaseException as error:
        _set_run_state(args.output_dir, status="failed", args=args, error=error)
        raise


if __name__ == "__main__":
    main()
