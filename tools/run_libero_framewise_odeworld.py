#!/usr/bin/env python3
"""Run framewise ODEWorld predictions against a real LIBERO expert replay.

Classification: partial rollout. LIBERO executes recorded expert actions while
ODEWorld predicts images only; this script does not evaluate policy success.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import h5py
import numpy as np

from tools.render_libero_material_inputs import (
    LIBERO_SOURCE,
    SOURCE_RESOLUTION,
    find_task,
    material_names,
    physics_hashes,
    rewrite_chiliocosm_asset_paths,
    select_exact_material,
    transform_agent_view,
)
from tools.verify_libero_red_replay import validate_source_files


ROOT = Path(__file__).resolve().parents[1]
LIBERO_SOURCE = Path(os.environ.get("LIBERO_SOURCE", "libero_source")).resolve()
RENDERER_PYTHON = os.environ.get("RENDERER_PYTHON", sys.executable)
HDF5_PATH = (
    Path(os.environ.get("LIBERO_ROOT", "."))
    / "libero_90"
    / "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove_demo.hdf5"
)
DEMO_NAME = "demo_42"
TASK_STEM = "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove"
INSTRUCTION = "put the moka pot on the stove"
MATERIAL_NAME = "moka_pot_1_moka_pot_body"
TARGET_RGBA = (0.95, 0.04, 0.04, 1.0)
RESOLUTION = 256
FPS = 10
SEED = 42
EXPECTED_STATES = 152
EXPECTED_ACTION_DIM = 7
EXPECTED_STATE_DIM = 47
BUNDLE_CLASSIFICATION = "partial rollout"
MODEL_STAGE_CLASSIFICATION = "mock test"
MODEL_STAGE_SCOPE = {
    "description": "offline ODEWorld visual prediction from saved replay frames",
    "env_step_calls": 0,
    "robot_actions_executed": False,
    "libero_environment_created": False,
    "success_rate_measured": False,
}
PT_FLOW_CHECKPOINT = Path(
    os.environ.get(
        "ODEWORLD_PT_FLOW_CHECKPOINT",
        str(ROOT / "assets/pretrained/ODEWorld-PT-Flow-LIBERO"),
    )
).resolve()
RAE_CHECKPOINT = ROOT / "assets/pretrained/ODEWorld-RAE-LIBERO"
GOAL_PREDICTOR_CHECKPOINT = ROOT / "assets/pretrained/ODEWorld-Goal-Predictor-LIBERO"
LPIPS_ALEXNET_CHECKPOINT = "alexnet-owt-7be5be79.pth"


def build_transitions(
    num_states: int,
    num_actions: int,
) -> list[tuple[int, int, int]]:
    """Return (state, action, next-state) indices with recorded targets."""
    if num_states < 2:
        raise ValueError(f"Expected at least 2 states, got {num_states}")
    expected = num_states - 1
    if num_actions < expected:
        raise ValueError(
            f"Expected at least {expected} actions for {expected} transitions, "
            f"got {num_actions}"
        )
    return [(index, index, index + 1) for index in range(expected)]


def one_step_horizon(max_time_length: int) -> float:
    """Map one recorded transition to PT-Flow physical time."""
    if max_time_length <= 0:
        raise ValueError("max_time_length must be positive")
    return 1.0 / max_time_length


def full_future_schedule(
    num_transitions: int,
    max_time_length: int,
) -> list[tuple[int, int, float]]:
    """Return (start index, steps, horizon) for every replanning point."""
    if num_transitions <= 0:
        raise ValueError("num_transitions must be positive")
    step_horizon = one_step_horizon(max_time_length)
    return [
        (start, num_transitions - start, (num_transitions - start) * step_horizon)
        for start in range(num_transitions)
    ]


def _validate_rgb_frame(frame: np.ndarray, *, name: str = "frame") -> None:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"Expected {name} shape [H,W,3], got {frame.shape}")
    if frame.dtype != np.uint8:
        raise ValueError(f"Expected {name} dtype uint8, got {frame.dtype}")


def frame_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Compute pixel metrics for two RGB uint8 frames."""
    _validate_rgb_frame(reference, name="reference")
    _validate_rgb_frame(prediction, name="prediction")
    if reference.shape != prediction.shape:
        raise ValueError(
            f"Frame shapes differ: reference={reference.shape}, prediction={prediction.shape}"
        )
    difference = prediction.astype(np.float32) - reference.astype(np.float32)
    absolute = np.abs(difference)
    mse = float(np.mean(np.square(difference)))
    psnr = math.inf if mse == 0.0 else 10.0 * math.log10((255.0**2) / mse)
    return {
        "psnr_db": float(psnr),
        "l1": float(absolute.mean()),
        "mse": mse,
    }


def labelled(frame: np.ndarray, text: str) -> np.ndarray:
    """Add a compact label without changing an RGB frame's dimensions."""
    _validate_rgb_frame(frame)
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        panel,
        text,
        (7, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def make_labelled_panel(
    frames: Sequence[np.ndarray],
    labels: Sequence[str],
) -> np.ndarray:
    if not frames or len(frames) != len(labels):
        raise ValueError("frames and labels must have the same nonzero length")
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise ValueError("All panel frames must have the same shape")
    return np.concatenate(
        [labelled(frame, label) for frame, label in zip(frames, labels)],
        axis=1,
    )


def fixed_error_heatmap(reference: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Render mean absolute RGB error with a fixed 0..255 scale."""
    _validate_rgb_frame(reference, name="reference")
    _validate_rgb_frame(prediction, name="prediction")
    if reference.shape != prediction.shape:
        raise ValueError(
            f"Frame shapes differ: reference={reference.shape}, prediction={prediction.shape}"
        )
    error = np.abs(
        prediction.astype(np.float32) - reference.astype(np.float32)
    ).mean(axis=2)
    fixed_scale = np.clip(np.rint(error), 0, 255).astype(np.uint8)
    bgr = cv2.applyColorMap(fixed_scale, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class VideoSink:
    """Strict RGB-to-MP4 streaming writer."""

    def __init__(
        self,
        path: Path,
        *,
        fps: int,
        frame_size: tuple[int, int],
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing video: {path}")
        width, height = frame_size
        if width <= 0 or height <= 0:
            raise ValueError("frame_size must contain positive dimensions")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.frame_size = frame_size
        self.frame_count = 0
        self._writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            frame_size,
        )
        if not self._writer.isOpened():
            self._writer.release()
            raise RuntimeError(f"Failed to open video writer: {path}")

    def __enter__(self) -> "VideoSink":
        return self

    def write(self, frame_rgb: np.ndarray) -> None:
        _validate_rgb_frame(frame_rgb)
        expected = (self.frame_size[1], self.frame_size[0], 3)
        if frame_rgb.shape != expected:
            raise ValueError(f"Expected video frame shape {expected}, got {frame_rgb.shape}")
        self._writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        self.frame_count += 1

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._writer.release()
        if exc_type is None and self.frame_count == 0:
            self.path.unlink(missing_ok=True)
            raise ValueError(f"Refusing to keep empty video: {self.path}")
        return False


def _temporary_sibling(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def _prepare_atomic_path(path: Path, *, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    temporary = _temporary_sibling(path)
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    return temporary


def write_json(path: Path, value: Any, *, overwrite: bool = False) -> None:
    temporary = _prepare_atomic_path(path, overwrite=overwrite)
    try:
        temporary.write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool = False,
) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV")
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise ValueError("All CSV rows must have the same ordered fields")
    temporary = _prepare_atomic_path(path, overwrite=overwrite)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_gif(
    path: Path,
    frames: Sequence[np.ndarray],
    *,
    fps: int,
    overwrite: bool = False,
) -> None:
    if not frames:
        raise ValueError("Cannot write an empty GIF")
    if fps <= 0:
        raise ValueError("fps must be positive")
    from PIL import Image

    shape = frames[0].shape
    for frame in frames:
        _validate_rgb_frame(frame)
        if frame.shape != shape:
            raise ValueError("All GIF frames must have the same shape")
    temporary = _prepare_atomic_path(path, overwrite=overwrite)
    images = [Image.fromarray(frame, mode="RGB") for frame in frames]
    try:
        images[0].save(
            temporary,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=round(1000 / fps),
            loop=0,
            optimize=False,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class LpipsMetric:
    """Optional LPIPS metric that never initiates a weight download."""

    def __init__(self, weights_path: Path | None, *, device: str = "cpu") -> None:
        self.enabled = False
        self.reason = (
            "local LPIPS dependency or weights unavailable; no network download attempted"
        )
        self._metric = None
        self._device = device
        if weights_path is None or not weights_path.is_file():
            return
        torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch"))
        trunk_path = torch_home / "hub/checkpoints" / LPIPS_ALEXNET_CHECKPOINT
        if not trunk_path.is_file():
            return
        try:
            import lpips

            metric = lpips.LPIPS(
                net="alex",
                model_path=str(weights_path),
                verbose=False,
            ).to(device).eval()
        except (ImportError, OSError, RuntimeError):
            return
        self._metric = metric
        self.enabled = True
        self.reason = ""

    def status(self) -> dict[str, str | bool]:
        return {"enabled": self.enabled, "reason": self.reason}

    def __call__(self, reference: np.ndarray, prediction: np.ndarray) -> float:
        if not self.enabled or self._metric is None:
            raise RuntimeError(self.reason)
        _validate_rgb_frame(reference, name="reference")
        _validate_rgb_frame(prediction, name="prediction")
        if reference.shape != prediction.shape:
            raise ValueError(
                f"Frame shapes differ: reference={reference.shape}, prediction={prediction.shape}"
            )
        import torch

        batch = np.stack([reference, prediction]).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self._device)
        with torch.inference_mode():
            value = self._metric(tensor[0:1], tensor[1:2], normalize=False)
        return float(value.item())


@dataclass(frozen=True)
class DemoData:
    actions: np.ndarray
    states: np.ndarray
    recorded_agentview: np.ndarray
    model_xml: str


@dataclass(frozen=True)
class ReplayFrame:
    index: int
    image_rgb: np.ndarray
    sim_state: np.ndarray


@dataclass(frozen=True)
class ReplayResult:
    frames: np.ndarray
    replay_frames: tuple[ReplayFrame, ...]
    state_diagnostics: tuple[dict[str, Any], ...]
    env_step_calls: int


def _resolve_replay_root(source: Path) -> Path:
    """Resolve either a replay root or its ``frames.npz`` file."""
    source = source.expanduser().resolve()
    if source.is_file():
        if source.name != "frames.npz":
            raise ValueError(f"Replay source file must be frames.npz, got {source}")
        return source.parent.parent
    if (source / "actual_replay" / "frames.npz").is_file():
        return source
    if (source / "frames.npz").is_file():
        return source.parent
    raise FileNotFoundError(
        f"Could not find actual_replay/frames.npz below replay source {source}"
    )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def validate_promoted_result_source(
    source_root: Path,
    *,
    expected_classification: str,
    expected_transitions: int,
) -> dict[str, Any]:
    """Validate generic-promoter evidence and return its preserved verification."""
    source_root = source_root.expanduser().resolve()
    promotion_path = source_root / "promotion.json"
    summary_path = source_root / "summary.json"
    run_state_path = source_root / "run_state.json"
    for path in (promotion_path, summary_path, run_state_path):
        if not path.is_file():
            raise FileNotFoundError(f"Promoted source lacks required evidence: {path}")
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
    required = (
        "classification",
        "scope",
        "staging_source",
        "final_destination",
        "atomic_rename",
        "slurm_job_id",
        "pre_promotion_verification_path",
        "pre_promotion_verification_sha256",
        "pre_promotion_all_checks_passed",
        "expected_transitions",
        "promoted_at",
    )
    missing = [key for key in required if key not in promotion]
    if missing:
        raise RuntimeError(f"Source promotion lacks required fields: {missing}")
    if promotion["classification"] != expected_classification:
        raise RuntimeError("Source promotion classification mismatch")
    if summary.get("classification") != expected_classification:
        raise RuntimeError("Source summary classification mismatch")
    if run_state.get("classification") != expected_classification:
        raise RuntimeError("Source run-state classification mismatch")
    if run_state.get("status") != "completed" or run_state.get("error") is not None:
        raise RuntimeError("Source run state is not cleanly completed")
    if promotion["scope"] != summary.get("scope"):
        raise RuntimeError("Source promotion scope differs from its summary")
    final = Path(str(promotion["final_destination"])).resolve()
    staging = Path(str(promotion["staging_source"])).resolve()
    if final != source_root:
        raise RuntimeError("Source promotion destination is not the selected result")
    if staging.exists() or staging.parent != final.parent:
        raise RuntimeError("Source promotion does not prove a completed same-parent rename")
    if promotion["atomic_rename"] is not True:
        raise RuntimeError("Source promotion is not recorded as atomic")
    job_id = str(promotion["slurm_job_id"])
    if not job_id or job_id != str(run_state.get("slurm_job_id", "")):
        raise RuntimeError("Source promotion Slurm job differs from its run state")
    if int(promotion["expected_transitions"]) != expected_transitions:
        raise RuntimeError("Source promotion transition count mismatch")
    promoted_at = datetime.fromisoformat(str(promotion["promoted_at"]))
    if promoted_at.utcoffset() != timedelta(0):
        raise RuntimeError("Source promotion timestamp is not UTC")
    relative_verification = Path(str(promotion["pre_promotion_verification_path"]))
    if relative_verification.is_absolute() or ".." in relative_verification.parts:
        raise RuntimeError("Source pre-promotion verification path is not result-relative")
    preserved = source_root / relative_verification
    try:
        resolved_preserved = preserved.resolve(strict=True)
        resolved_preserved.relative_to(source_root)
    except (FileNotFoundError, ValueError) as error:
        raise RuntimeError(
            "Source pre-promotion verification is missing or outside the result"
        ) from error
    cursor = source_root
    for part in relative_verification.parts:
        cursor /= part
        if cursor.is_symlink():
            raise RuntimeError("Source pre-promotion verification uses a symlink")
    recorded_hash = promotion["pre_promotion_verification_sha256"]
    if not _is_sha256(recorded_hash) or _sha256_file(preserved) != recorded_hash:
        raise RuntimeError("Source pre-promotion verification hash mismatch")
    if promotion["pre_promotion_all_checks_passed"] is not True:
        raise RuntimeError("Source promotion does not record a passing verification")
    verification = json.loads(preserved.read_text(encoding="utf-8"))
    if (
        verification.get("classification") != expected_classification
        or verification.get("all_checks_passed") is not True
        or int(verification.get("expected_transitions", -1)) != expected_transitions
    ):
        raise RuntimeError("Source preserved verification is not a matching pass")
    return {
        "promotion": promotion,
        "promotion_path": promotion_path,
        "promotion_sha256": _sha256_file(promotion_path),
        "verification": verification,
        "verification_path": preserved,
        "verification_sha256": recorded_hash,
        "slurm_job_id": job_id,
    }


def load_external_replay(
    source: Path,
    output_dir: Path,
    *,
    max_transitions: int,
) -> tuple[ReplayResult, dict[str, Any]]:
    """Import a replay rendered by the canonical FAST-WAM environment.

    The model stage runs in the ODEWorld environment, so the replay artifacts
    are copied into the model-stage output and then treated as immutable input.
    """
    source_root = _resolve_replay_root(source)
    source_replay = source_root / "actual_replay"
    required = (
        source_replay / "frames.npz",
        source_replay / "actual_replay.mp4",
        source_replay / "replay_diagnostics.json",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"External replay is incomplete: {missing}")
    diagnostics = json.loads(
        (source_replay / "replay_diagnostics.json").read_text(encoding="utf-8")
    )
    renderer_provenance_path = source_root / "renderer_provenance.json"
    for required_path in (renderer_provenance_path,):
        if not required_path.is_file():
            raise FileNotFoundError(f"External replay lacks verified provenance: {required_path}")
    promotion_evidence = validate_promoted_result_source(
        source_root,
        expected_classification=BUNDLE_CLASSIFICATION,
        expected_transitions=max_transitions,
    )
    verification_path = promotion_evidence["verification_path"]
    renderer_provenance = json.loads(
        renderer_provenance_path.read_text(encoding="utf-8")
    )
    source_verification = promotion_evidence["verification"]
    renderer = renderer_provenance.get("renderer", {})
    expected_renderer = {
        "environment": "FAST-WAM",
        "mujoco": "3.3.2",
        "robosuite": "1.4.0",
        "numpy": "1.26.4",
        "opencv": "4.6.0.66",
        "mujoco_gl": "egl",
        "pyopengl_platform": "egl",
        "camera_transform": "rgb_vhflip",
        "source_camera_resolution": 128,
        "model_resolution": 256,
        "resize_interpolation": "INTER_AREA",
    }
    mismatches = {
        key: {"expected": value, "actual": renderer.get(key)}
        for key, value in expected_renderer.items()
        if renderer.get(key) != value
    }
    if mismatches:
        raise ValueError(f"External renderer contract mismatch: {mismatches}")
    if Path(str(renderer.get("python"))).resolve() != Path(RENDERER_PYTHON).resolve():
        raise ValueError(f"Unexpected external renderer Python: {renderer.get('python')}")
    if Path(str(renderer.get("libero_source_root"))).resolve() != LIBERO_SOURCE:
        raise ValueError(
            f"Unexpected external LIBERO source: {renderer.get('libero_source_root')}"
        )
    if not str(renderer.get("libero_module", "")).startswith(str(LIBERO_SOURCE) + "/"):
        raise ValueError(f"Unexpected external LIBERO module: {renderer.get('libero_module')}")
    if renderer_provenance.get("runtime", {}).get("slurm_job_id") in (None, ""):
        raise ValueError("External renderer provenance lacks a Slurm job ID")
    if str(renderer_provenance["runtime"]["slurm_job_id"]) != promotion_evidence[
        "slurm_job_id"
    ]:
        raise ValueError("External renderer and promotion Slurm jobs differ")
    validate_source_files(renderer.get("source_files"))
    if diagnostics.get("classification") != "partial rollout":
        raise ValueError("External replay is not labelled partial rollout")
    if not diagnostics.get("invariance", {}).get("physics_arrays_unchanged"):
        raise ValueError("External replay does not prove physics invariance")
    material = diagnostics.get("material", {})
    if material.get("material_name") != MATERIAL_NAME or not np.allclose(
        np.asarray(material.get("rgba_after")),
        np.asarray(TARGET_RGBA),
        atol=1e-6,
    ):
        raise ValueError("External replay does not use the canonical red moka material")
    with np.load(source_replay / "frames.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"frames"}:
            raise ValueError(f"Unexpected external replay arrays: {archive.files}")
        frames = np.asarray(archive["frames"])
    source_hash_checks = {
        "frames_sha256": source_replay / "frames.npz",
        "replay_diagnostics_sha256": source_replay / "replay_diagnostics.json",
        "renderer_provenance_sha256": renderer_provenance_path,
        "run_state_sha256": source_root / "run_state.json",
    }
    for key, path in source_hash_checks.items():
        if source_verification.get(key) != _sha256_file(path):
            raise ValueError(f"External replay verification hash mismatch: {key}")
    video_record = source_verification.get("video", {})
    if video_record.get("sha256") != _sha256_file(source_replay / "actual_replay.mp4"):
        raise ValueError("External replay verification video hash mismatch")
    _validate_rgb_frame(frames[0], name="external replay frame")
    if frames.ndim != 4 or frames.shape[1:] != (RESOLUTION, RESOLUTION, 3):
        raise ValueError(f"Unexpected external replay shape: {frames.shape}")
    if frames.dtype != np.uint8 or not np.isfinite(frames).all():
        raise ValueError("External replay must contain finite uint8 RGB frames")
    expected_frames = max_transitions + 1
    if frames.shape[0] != expected_frames:
        raise ValueError(
            f"External replay has {frames.shape[0]} frames; expected {expected_frames} "
            f"for {max_transitions} transitions"
        )
    if int(diagnostics.get("env_step_calls", -1)) != max_transitions:
        raise ValueError("External replay environment-step count does not match request")
    state_diagnostics = diagnostics.get("state_diagnostics")
    if not isinstance(state_diagnostics, list) or len(state_diagnostics) != max_transitions:
        raise ValueError("External replay state diagnostics count does not match request")

    destination_replay = output_dir / "actual_replay"
    destination_replay.mkdir(parents=True, exist_ok=True)
    for path in required:
        target = destination_replay / path.name
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite imported replay artifact: {target}")
        shutil.copy2(path, target)
    copied_hashes = {
        "frames_sha256": _sha256_file(destination_replay / "frames.npz"),
        "video_sha256": _sha256_file(destination_replay / "actual_replay.mp4"),
    }
    expected_copied_hashes = {
        "frames_sha256": _sha256_file(source_replay / "frames.npz"),
        "video_sha256": _sha256_file(source_replay / "actual_replay.mp4"),
    }
    if copied_hashes != expected_copied_hashes:
        raise RuntimeError(
            "External replay copy hash mismatch: "
            f"expected={expected_copied_hashes}, actual={copied_hashes}"
        )
    imported_diagnostics = {
        **diagnostics,
        "external_source": {
            "root": str(source_root),
            "frames_sha256": _sha256_file(source_replay / "frames.npz"),
            "video_sha256": _sha256_file(source_replay / "actual_replay.mp4"),
            "diagnostics_sha256": _sha256_file(source_replay / "replay_diagnostics.json"),
            "renderer_provenance_sha256": _sha256_file(renderer_provenance_path),
            "verification_sha256": _sha256_file(verification_path),
            "promotion_sha256": promotion_evidence["promotion_sha256"],
            "verification_all_checks_passed": True,
            "promotion_verified": True,
            "source_slurm_job_id": promotion_evidence["slurm_job_id"],
            "renderer": renderer,
        },
    }
    write_json(destination_replay / "replay_diagnostics.json", imported_diagnostics, overwrite=True)
    replay = ReplayResult(
        frames=np.ascontiguousarray(frames.copy()),
        replay_frames=tuple(),
        state_diagnostics=tuple(state_diagnostics),
        env_step_calls=max_transitions,
    )
    metadata = dict(imported_diagnostics)
    metadata["external_source"] = imported_diagnostics["external_source"]
    return replay, metadata


def _require_shape(name: str, value: np.ndarray, expected: tuple[int, ...]) -> None:
    if value.shape != expected:
        raise ValueError(f"Expected {name} shape {expected}, got {value.shape}")


def load_demo(path: Path = HDF5_PATH, *, demo_name: str = DEMO_NAME) -> DemoData:
    """Load and validate the exact official demonstration used by this run."""
    with h5py.File(path, "r") as handle:
        demo_path = f"data/{demo_name}"
        if demo_path not in handle:
            raise KeyError(f"Missing HDF5 group {demo_path}")
        demo = handle[demo_path]
        actions = np.asarray(demo["actions"]).copy()
        states = np.asarray(demo["states"]).copy()
        recorded_agentview = np.asarray(demo["obs"]["agentview_rgb"]).copy()
        model_xml = demo.attrs["model_file"]

    _require_shape(
        "actions",
        actions,
        (EXPECTED_STATES, EXPECTED_ACTION_DIM),
    )
    _require_shape(
        "states",
        states,
        (EXPECTED_STATES, EXPECTED_STATE_DIM),
    )
    _require_shape(
        "recorded agentview",
        recorded_agentview,
        (EXPECTED_STATES, 128, 128, 3),
    )
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("Demo actions and states must be finite")
    if recorded_agentview.dtype != np.uint8:
        raise ValueError(
            f"Expected recorded agentview dtype uint8, got {recorded_agentview.dtype}"
        )
    if isinstance(model_xml, bytes):
        model_xml = model_xml.decode("utf-8")
    if not isinstance(model_xml, str):
        raise TypeError(f"Expected model_file XML string, got {type(model_xml)}")
    return DemoData(
        actions=actions,
        states=states,
        recorded_agentview=recorded_agentview,
        model_xml=model_xml,
    )


def _agentview_rgb(observation: Mapping[str, Any]) -> np.ndarray:
    if "agentview_image" not in observation:
        raise KeyError("Observation does not contain agentview_image")
    image = np.asarray(observation["agentview_image"])
    if image.dtype != np.uint8:
        raise ValueError(f"Expected agentview_image dtype uint8, got {image.dtype}")
    transformed = transform_agent_view(image, resolution=RESOLUTION)
    _validate_rgb_frame(transformed, name="transformed agentview")
    return transformed


def _validate_replay_arrays(actions: np.ndarray, states: np.ndarray) -> None:
    if actions.ndim != 2 or actions.shape[1] != EXPECTED_ACTION_DIM:
        raise ValueError(
            f"Expected actions shape [N,{EXPECTED_ACTION_DIM}], got {actions.shape}"
        )
    if states.ndim != 2 or states.shape[1] != EXPECTED_STATE_DIM:
        raise ValueError(
            f"Expected states shape [N,{EXPECTED_STATE_DIM}], got {states.shape}"
        )
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("Replay actions and states must be finite")


def replay_expert_actions(
    env: Any,
    actions: np.ndarray,
    states: np.ndarray,
    *,
    max_transitions: int | None,
) -> ReplayResult:
    """Replay recorded actions without treating ODEWorld as a controller."""
    actions = np.asarray(actions)
    states = np.asarray(states)
    _validate_replay_arrays(actions, states)
    transitions = build_transitions(len(states), len(actions))
    if max_transitions is not None:
        if max_transitions <= 0 or max_transitions > len(transitions):
            raise ValueError(
                f"max_transitions must be in [1,{len(transitions)}], got {max_transitions}"
            )
        transitions = transitions[:max_transitions]

    initial_observation = env.set_init_state(states[0])
    initial_state = np.asarray(env.get_sim_state()).copy()
    if initial_state.shape != states[0].shape or not np.isfinite(initial_state).all():
        raise ValueError(
            f"Invalid restored simulator state shape/value: {initial_state.shape}"
        )
    replay_frames = [
        ReplayFrame(
            index=0,
            image_rgb=_agentview_rgb(initial_observation),
            sim_state=initial_state,
        )
    ]
    diagnostics: list[dict[str, Any]] = []

    for state_index, action_index, target_state_index in transitions:
        observation, reward, done, _ = env.step(actions[action_index])
        actual_state = np.asarray(env.get_sim_state()).copy()
        if actual_state.shape != states[target_state_index].shape:
            raise ValueError(
                "Simulator state shape changed after step "
                f"{action_index}: {actual_state.shape}"
            )
        if not np.isfinite(actual_state).all():
            raise ValueError(f"Simulator state after step {action_index} is not finite")
        image_rgb = _agentview_rgb(observation)
        difference = np.abs(
            actual_state.astype(np.float64)
            - states[target_state_index].astype(np.float64)
        )
        diagnostics.append(
            {
                "state_index": state_index,
                "action_index": action_index,
                "target_state_index": target_state_index,
                "reward": float(reward),
                "done": bool(done),
                "mean_abs_state_error": float(difference.mean()),
                "max_abs_state_error": float(difference.max(initial=0.0)),
            }
        )
        replay_frames.append(
            ReplayFrame(
                index=target_state_index,
                image_rgb=image_rgb,
                sim_state=actual_state,
            )
        )

    frames = np.stack([frame.image_rgb for frame in replay_frames])
    return ReplayResult(
        frames=frames,
        replay_frames=tuple(replay_frames),
        state_diagnostics=tuple(diagnostics),
        env_step_calls=len(transitions),
    )


def _json_metric(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _unmodified_alignment(env: Any, demo: DemoData) -> list[dict[str, Any]]:
    rows = []
    for index, state in enumerate(demo.states):
        observation = env.set_init_state(state)
        rendered = _agentview_rgb(observation)
        recorded = transform_agent_view(
            demo.recorded_agentview[index],
            resolution=RESOLUTION,
        )
        metrics = frame_metrics(recorded, rendered)
        rows.append(
            {
                "state_index": index,
                "psnr_db": _json_metric(metrics["psnr_db"]),
                "exact_pixel_match": bool(metrics["mse"] == 0.0),
                "l1": metrics["l1"],
                "mse": metrics["mse"],
            }
        )
    env.set_init_state(demo.states[0])
    return rows


def create_libero_red_replay(
    *,
    max_transitions: int | None,
    seed: int = SEED,
) -> tuple[ReplayResult, dict[str, Any]]:
    """Create LIBERO, apply the runtime material edit, and replay the expert."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero.utils import utils as libero_utils

    demo = load_demo()
    suite = benchmark.get_benchmark_dict()["libero_90"]()
    task_id, task = find_task(suite)
    bddl_path = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = None
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_names=["agentview", "robot0_eye_in_hand"],
            camera_heights=SOURCE_RESOLUTION,
            camera_widths=SOURCE_RESOLUTION,
            camera_depths=False,
        )
        env.seed(seed)
        env.reset()
        model_xml = libero_utils.postprocess_model_xml(demo.model_xml, {})
        model_xml = rewrite_chiliocosm_asset_paths(
            model_xml,
            LIBERO_SOURCE / "libero/libero/assets",
        )
        env.reset_from_xml_string(model_xml)
        env.sim.reset()
        environment_state_length = int(np.asarray(env.get_sim_state()).size)
        if environment_state_length != EXPECTED_STATE_DIM:
            raise RuntimeError(
                f"State length mismatch: environment={environment_state_length}, "
                f"HDF5={EXPECTED_STATE_DIM}"
            )

        alignment = _unmodified_alignment(env, demo)
        model = env.sim.model
        material_id, resolved_material_name = select_exact_material(
            material_names(model),
            MATERIAL_NAME,
        )
        rgba_before = np.asarray(model.mat_rgba[material_id]).copy()
        physics_before = physics_hashes(model)
        model.mat_rgba[material_id] = np.asarray(TARGET_RGBA, dtype=np.float32)
        env.sim.forward()
        physics_after = physics_hashes(model)
        if physics_before != physics_after:
            raise RuntimeError("Material edit changed physical model arrays")
        rgba_after = np.asarray(model.mat_rgba[material_id]).copy()
        replay = replay_expert_actions(
            env,
            demo.actions,
            demo.states,
            max_transitions=max_transitions,
        )
        metadata = {
            "classification": "partial rollout",
            "scope": {
                "suite": "libero_90",
                "task_id": task_id,
                "task_name": TASK_STEM,
                "instruction": task.language,
                "hdf5": str(HDF5_PATH),
                "demo": DEMO_NAME,
                "seed": seed,
                "env_step_calls": replay.env_step_calls,
                "odeworld_generated_actions": False,
                "success_rate_measured": False,
            },
            "material": {
                "material_id": material_id,
                "material_name": resolved_material_name,
                "rgba_before": rgba_before.tolist(),
                "rgba_after": rgba_after.tolist(),
                "target_rgba": list(TARGET_RGBA),
            },
            "invariance": {
                "physics_hashes_before": physics_before,
                "physics_hashes_after": physics_after,
                "physics_arrays_unchanged": physics_before == physics_after,
            },
            "unmodified_recorded_alignment": alignment,
        }
        return replay, metadata
    finally:
        if env is not None:
            env.close()


def save_replay_artifacts(
    output_root: Path,
    replay: ReplayResult,
    metadata: Mapping[str, Any],
    *,
    fps: int = FPS,
) -> dict[str, str]:
    """Persist the shared replay only after all environment steps complete."""
    replay_dir = output_root / "actual_replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    npz_path = replay_dir / "frames.npz"
    if npz_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing replay: {npz_path}")
    temporary_npz = _temporary_sibling(npz_path)
    try:
        with temporary_npz.open("wb") as handle:
            np.savez_compressed(handle, frames=replay.frames)
        os.replace(temporary_npz, npz_path)
    finally:
        temporary_npz.unlink(missing_ok=True)

    video_path = replay_dir / "actual_replay.mp4"
    with VideoSink(
        video_path,
        fps=fps,
        frame_size=(RESOLUTION, RESOLUTION),
    ) as sink:
        for frame in replay.frames:
            sink.write(frame)

    diagnostics_path = replay_dir / "replay_diagnostics.json"
    write_json(
        diagnostics_path,
        {
            **dict(metadata),
            "frame_count": int(replay.frames.shape[0]),
            "env_step_calls": replay.env_step_calls,
            "state_diagnostics": list(replay.state_diagnostics),
        },
    )
    return {
        "frames": str(npz_path),
        "video": str(video_path),
        "diagnostics": str(diagnostics_path),
    }


@dataclass(frozen=True)
class ModelBundle:
    flow: Any
    rae: Any
    goal_predictor: Any
    device: Any


def load_models(device: Any) -> ModelBundle:
    """Load each local ODEWorld checkpoint once."""
    from models import Dinov2GoalPred, Dinov2PTflowImgoal, Dinov2RAE

    flow = Dinov2PTflowImgoal.from_pretrained(str(PT_FLOW_CHECKPOINT)).to(device).eval()
    rae = Dinov2RAE.from_pretrained(str(RAE_CHECKPOINT)).to(device).eval()
    goal_predictor = Dinov2GoalPred.from_pretrained(
        str(GOAL_PREDICTOR_CHECKPOINT)
    ).to(device).eval()
    if flow.max_time_length != 50:
        raise RuntimeError(f"Expected max_time_length=50, got {flow.max_time_length}")
    return ModelBundle(flow, rae, goal_predictor, device)


def rgb_to_tensor(frame: np.ndarray, device: Any) -> Any:
    _validate_rgb_frame(frame)
    import torch

    return (
        torch.from_numpy(np.ascontiguousarray(frame))
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )


def tensor_to_rgb(image: Any) -> np.ndarray:
    return (
        image.detach()
        .float()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .mul(255)
        .round()
        .byte()
        .numpy()
    )


def decode_patch_latents(rae: Any, latents: Any, *, chunk_size: int) -> list[np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("decode chunk_size must be positive")
    frames: list[np.ndarray] = []
    for start in range(0, latents.shape[0], chunk_size):
        decoded = rae.decode(latents[start : start + chunk_size]).clamp(0, 1)
        frames.extend(tensor_to_rgb(frame) for frame in decoded)
    return frames


class OdeWorldBackend:
    """RGB-facing adapter over PT-Flow, RAE, and Goal Predictor."""

    def __init__(self, bundle: ModelBundle, *, decode_chunk_size: int = 16) -> None:
        self.bundle = bundle
        self.decode_chunk_size = decode_chunk_size
        self.max_time_length = int(bundle.flow.max_time_length)

    def image_rollout(
        self,
        current: np.ndarray,
        goal: np.ndarray,
        *,
        horizon: float,
        steps: int,
    ) -> list[np.ndarray]:
        import torch

        current_tensor = rgb_to_tensor(current, self.bundle.device)
        goal_tensor = rgb_to_tensor(goal, self.bundle.device)
        with torch.inference_mode():
            reconstructed, _ = self.bundle.flow.rollout_ode(
                current_tensor,
                goal_tensor,
                horizon=horizon,
                steps=steps,
            )
            return decode_patch_latents(
                self.bundle.rae,
                reconstructed[0],
                chunk_size=self.decode_chunk_size,
            )

    def language_rollout(
        self,
        current: np.ndarray,
        instruction: str,
        *,
        horizon: float,
        steps: int,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        import torch
        from torchdiffeq import odeint

        current_tensor = rgb_to_tensor(current, self.bundle.device)
        flow = self.bundle.flow
        with torch.inference_mode():
            s0 = flow.latent_encode(current_tensor)
            sg = self.bundle.goal_predictor.predict(current_tensor, [instruction])
            z0 = flow.delta_decouple(s0, s0)
            zg = flow.delta_decouple(s0, sg)

            def ode_func(time_scalar, dynamic_latent):
                tau = time_scalar.view(1, 1).expand(current_tensor.shape[0], 1)
                return (
                    flow.forward_vmodel(z0, dynamic_latent, zg, tau)
                    * flow.max_time_length
                )

            time_grid = torch.linspace(
                0,
                horizon,
                steps + 1,
                device=self.bundle.device,
            )
            dynamic = odeint(ode_func, z0, time_grid, method="rk4")[1:]
            dynamic = dynamic.permute(1, 0, 2, 3)[0]
            reconstructed_chunks = []
            for start in range(0, steps, self.decode_chunk_size):
                chunk = dynamic[start : start + self.decode_chunk_size]
                s0_chunk = s0.expand(chunk.shape[0], -1, -1)
                reconstructed_chunks.append(flow.delta_decode(s0_chunk, chunk))
            reconstructed = torch.cat(reconstructed_chunks, dim=0)
            frames = decode_patch_latents(
                self.bundle.rae,
                reconstructed,
                chunk_size=self.decode_chunk_size,
            )
            predicted_goal = tensor_to_rgb(self.bundle.rae.decode(sg).clamp(0, 1)[0])
        return frames, predicted_goal


def _validate_prediction_backend(backend: Any) -> None:
    if int(backend.max_time_length) != 50:
        raise RuntimeError(
            f"Expected max_time_length=50, got {backend.max_time_length}"
        )


def _prediction_metric_row(
    *,
    mode: str,
    branch: str,
    start_index: int,
    lead_steps: int,
    target_index: int,
    reference: np.ndarray,
    prediction: np.ndarray,
    lpips_metric: LpipsMetric,
) -> dict[str, Any]:
    metrics = frame_metrics(reference, prediction)
    lpips_value = lpips_metric(reference, prediction) if lpips_metric.enabled else None
    return {
        "mode": mode,
        "branch": branch,
        "start_index": start_index,
        "lead_steps": lead_steps,
        "target_index": target_index,
        "psnr_db": metrics["psnr_db"],
        "l1": metrics["l1"],
        "mse": metrics["mse"],
        "lpips": lpips_value,
    }


def _aggregate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    branches = sorted({str(row["branch"]) for row in rows})
    for branch in branches:
        selected = [row for row in rows if row["branch"] == branch]
        branch_result: dict[str, Any] = {"count": len(selected)}
        for name in ("psnr_db", "l1", "mse", "lpips"):
            values = [float(row[name]) for row in selected if row[name] is not None]
            branch_result[name] = (
                {
                    "mean": float(np.mean(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
                if values
                else None
            )
        result[branch] = branch_result
    return result


def run_teacher_forced(
    backend: Any,
    actual_frames: np.ndarray,
    output_dir: Path,
    *,
    instruction: str,
    fps: int,
    lpips_metric: LpipsMetric,
) -> dict[str, Any]:
    """Predict each next frame from the corresponding real current frame."""
    _validate_prediction_backend(backend)
    if actual_frames.ndim != 4 or actual_frames.shape[1:] != (
        RESOLUTION,
        RESOLUTION,
        3,
    ):
        raise ValueError(f"Unexpected actual frame array: {actual_frames.shape}")
    transition_count = len(actual_frames) - 1
    if transition_count <= 0:
        raise ValueError("Teacher-forced mode requires at least two frames")
    mode_dir = output_dir / "mode1_teacher_forced"
    mode_dir.mkdir(parents=True, exist_ok=True)
    video_path = mode_dir / "teacher_forced_four_panel.mp4"
    gif_path = mode_dir / "teacher_forced_four_panel.gif"
    metric_path = mode_dir / "metrics.csv"
    summary_path = mode_dir / "summary.json"
    goal = actual_frames[-1]
    horizon = one_step_horizon(backend.max_time_length)
    rows: list[dict[str, Any]] = []
    panels: list[np.ndarray] = []
    image_seconds = 0.0
    language_seconds = 0.0

    with VideoSink(
        video_path,
        fps=fps,
        frame_size=(RESOLUTION * 4, RESOLUTION),
    ) as sink:
        for index in range(transition_count):
            current = actual_frames[index]
            target = actual_frames[index + 1]
            started = time.perf_counter()
            image_predictions = backend.image_rollout(
                current,
                goal,
                horizon=horizon,
                steps=1,
            )
            image_seconds += time.perf_counter() - started
            started = time.perf_counter()
            language_predictions, _ = backend.language_rollout(
                current,
                instruction,
                horizon=horizon,
                steps=1,
            )
            language_seconds += time.perf_counter() - started
            if len(image_predictions) != 1 or len(language_predictions) != 1:
                raise RuntimeError("One-step inference did not return exactly one frame")
            image_prediction = image_predictions[0]
            language_prediction = language_predictions[0]
            rows.extend(
                [
                    _prediction_metric_row(
                        mode="teacher_forced",
                        branch="image_goal",
                        start_index=index,
                        lead_steps=1,
                        target_index=index + 1,
                        reference=target,
                        prediction=image_prediction,
                        lpips_metric=lpips_metric,
                    ),
                    _prediction_metric_row(
                        mode="teacher_forced",
                        branch="language_goal",
                        start_index=index,
                        lead_steps=1,
                        target_index=index + 1,
                        reference=target,
                        prediction=language_prediction,
                        lpips_metric=lpips_metric,
                    ),
                ]
            )
            panel = make_labelled_panel(
                [current, image_prediction, language_prediction, target],
                [
                    f"Actual current t={index}",
                    "Image-goal next",
                    "Language-goal next",
                    f"Actual next t={index + 1}",
                ],
            )
            sink.write(panel)
            panels.append(panel)

    write_gif(gif_path, panels, fps=fps)
    write_csv(metric_path, rows)
    summary = {
        "classification": MODEL_STAGE_CLASSIFICATION,
        "stage_classification": MODEL_STAGE_CLASSIFICATION,
        "scope": dict(MODEL_STAGE_SCOPE),
        "mode": "teacher_forced",
        "transition_count": transition_count,
        "metric_rows": len(rows),
        "video_frames": len(panels),
        "horizon": horizon,
        "steps_per_prediction": 1,
        "lpips": lpips_metric.status(),
        "timing_seconds": {
            "image_goal": image_seconds,
            "language_goal": language_seconds,
        },
        "metrics": _aggregate_metric_rows(rows),
        "outputs": {
            "video": video_path.relative_to(output_dir).as_posix(),
            "gif": gif_path.relative_to(output_dir).as_posix(),
            "metrics": metric_path.relative_to(output_dir).as_posix(),
        },
    }
    write_json(summary_path, summary)
    return summary


def _validate_rollout_frames(
    frames: Sequence[np.ndarray],
    *,
    expected: int,
    branch: str,
) -> None:
    if len(frames) != expected:
        raise RuntimeError(
            f"{branch} returned {len(frames)} frames, expected {expected}"
        )
    for frame in frames:
        _validate_rgb_frame(frame, name=f"{branch} prediction")
        if frame.shape != (RESOLUTION, RESOLUTION, 3):
            raise ValueError(f"Unexpected {branch} prediction shape: {frame.shape}")


def _lead_time_aggregates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    aggregates: dict[str, Any] = {}
    for branch in sorted({str(row["branch"]) for row in rows}):
        branch_rows = [row for row in rows if row["branch"] == branch]
        by_lead = {}
        for lead in sorted({int(row["lead_steps"]) for row in branch_rows}):
            selected = [row for row in branch_rows if int(row["lead_steps"]) == lead]
            by_lead[str(lead)] = {
                "count": len(selected),
                "psnr_db_mean": float(np.mean([row["psnr_db"] for row in selected])),
                "l1_mean": float(np.mean([row["l1"] for row in selected])),
                "mse_mean": float(np.mean([row["mse"] for row in selected])),
                "lpips_mean": (
                    float(np.mean([row["lpips"] for row in selected]))
                    if selected[0]["lpips"] is not None
                    else None
                ),
            }
        aggregates[branch] = by_lead
    return aggregates


def run_full_future(
    backend: Any,
    actual_frames: np.ndarray,
    output_dir: Path,
    *,
    instruction: str,
    fps: int,
    lpips_metric: LpipsMetric,
) -> dict[str, Any]:
    """Replan the complete remaining visual future from every actual frame."""
    _validate_prediction_backend(backend)
    transition_count = len(actual_frames) - 1
    schedule = full_future_schedule(transition_count, backend.max_time_length)
    mode_dir = output_dir / "mode2_full_future"
    image_dir = mode_dir / "image_goal"
    language_dir = mode_dir / "language_goal"
    image_dir.mkdir(parents=True, exist_ok=True)
    language_dir.mkdir(parents=True, exist_ok=True)
    overview_path = mode_dir / "endpoint_overview.mp4"
    overview_gif_path = mode_dir / "endpoint_overview.gif"
    metrics_path = mode_dir / "metrics.csv"
    summary_path = mode_dir / "summary.json"
    fixed_goal = actual_frames[-1]
    rows: list[dict[str, Any]] = []
    overview_panels: list[np.ndarray] = []
    image_seconds = 0.0
    language_seconds = 0.0

    with VideoSink(
        overview_path,
        fps=fps,
        frame_size=(RESOLUTION * 4, RESOLUTION),
    ) as overview_sink:
        for start_index, steps, horizon in schedule:
            current = actual_frames[start_index]
            started = time.perf_counter()
            image_predictions = backend.image_rollout(
                current,
                fixed_goal,
                horizon=horizon,
                steps=steps,
            )
            image_seconds += time.perf_counter() - started
            started = time.perf_counter()
            language_predictions, _ = backend.language_rollout(
                current,
                instruction,
                horizon=horizon,
                steps=steps,
            )
            language_seconds += time.perf_counter() - started
            _validate_rollout_frames(
                image_predictions,
                expected=steps,
                branch="image_goal",
            )
            _validate_rollout_frames(
                language_predictions,
                expected=steps,
                branch="language_goal",
            )

            for branch, predictions, directory in (
                ("image_goal", image_predictions, image_dir),
                ("language_goal", language_predictions, language_dir),
            ):
                clip_path = directory / f"frame_{start_index:03d}.mp4"
                with VideoSink(
                    clip_path,
                    fps=fps,
                    frame_size=(RESOLUTION, RESOLUTION),
                ) as clip_sink:
                    for offset, prediction in enumerate(predictions, start=1):
                        clip_sink.write(prediction)
                        rows.append(
                            _prediction_metric_row(
                                mode="full_future",
                                branch=branch,
                                start_index=start_index,
                                lead_steps=offset,
                                target_index=start_index + offset,
                                reference=actual_frames[start_index + offset],
                                prediction=prediction,
                                lpips_metric=lpips_metric,
                            )
                        )

            overview = make_labelled_panel(
                [
                    current,
                    image_predictions[-1],
                    language_predictions[-1],
                    fixed_goal,
                ],
                [
                    f"Actual current t={start_index}",
                    "Image-goal endpoint",
                    "Language-goal endpoint",
                    f"Actual final t={transition_count}",
                ],
            )
            overview_sink.write(overview)
            overview_panels.append(overview)

    write_gif(overview_gif_path, overview_panels, fps=fps)
    write_csv(metrics_path, rows)
    predicted_frames_per_branch = sum(steps for _, steps, _ in schedule)
    summary = {
        "classification": MODEL_STAGE_CLASSIFICATION,
        "stage_classification": MODEL_STAGE_CLASSIFICATION,
        "scope": dict(MODEL_STAGE_SCOPE),
        "stress_test_label": "long-integration stress test",
        "mode": "full_future",
        "transition_count": transition_count,
        "clips_per_branch": transition_count,
        "predicted_frames_per_branch": predicted_frames_per_branch,
        "metric_rows": len(rows),
        "overview_frames": len(overview_panels),
        "maximum_horizon": schedule[0][2],
        "lpips": lpips_metric.status(),
        "timing_seconds": {
            "image_goal": image_seconds,
            "language_goal": language_seconds,
        },
        "metrics": _aggregate_metric_rows(rows),
        "metrics_by_lead_time": _lead_time_aggregates(rows),
        "outputs": {
            "image_goal_clips": image_dir.relative_to(output_dir).as_posix(),
            "language_goal_clips": language_dir.relative_to(output_dir).as_posix(),
            "overview_video": overview_path.relative_to(output_dir).as_posix(),
            "overview_gif": overview_gif_path.relative_to(output_dir).as_posix(),
            "metrics": metrics_path.relative_to(output_dir).as_posix(),
        },
    }
    write_json(summary_path, summary)
    return summary


def _combined_error_panel(
    reference: np.ndarray,
    image_prediction: np.ndarray,
    language_prediction: np.ndarray,
) -> np.ndarray:
    image_error = fixed_error_heatmap(reference, image_prediction)
    language_error = fixed_error_heatmap(reference, language_prediction)
    image_half = cv2.resize(
        image_error,
        (RESOLUTION // 2, RESOLUTION),
        interpolation=cv2.INTER_AREA,
    )
    language_half = cv2.resize(
        language_error,
        (RESOLUTION // 2, RESOLUTION),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.concatenate([image_half, language_half], axis=1)
    cv2.putText(
        panel,
        "I",
        (8, RESOLUTION - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "L",
        (RESOLUTION // 2 + 8, RESOLUTION - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def _metric_slopes(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for branch in sorted({str(row["branch"]) for row in rows}):
        selected = [row for row in rows if row["branch"] == branch]
        x = np.asarray([row["target_index"] for row in selected], dtype=np.float64)
        branch_result = {}
        for metric in ("psnr_db", "l1", "mse", "lpips"):
            if selected[0][metric] is None:
                branch_result[f"{metric}_per_step"] = None
                continue
            y = np.asarray([row[metric] for row in selected], dtype=np.float64)
            branch_result[f"{metric}_per_step"] = (
                float(np.polyfit(x, y, 1)[0]) if len(x) > 1 else 0.0
            )
        result[branch] = branch_result
    return result


def run_autoregressive(
    backend: Any,
    actual_frames: np.ndarray,
    output_dir: Path,
    *,
    instruction: str,
    fps: int,
    lpips_metric: LpipsMetric,
) -> dict[str, Any]:
    """Feed each decoded ODEWorld prediction back as the next RGB input."""
    _validate_prediction_backend(backend)
    transition_count = len(actual_frames) - 1
    if transition_count <= 0:
        raise ValueError("Autoregressive mode requires at least two frames")
    mode_dir = output_dir / "mode3_autoregressive"
    mode_dir.mkdir(parents=True, exist_ok=True)
    video_path = mode_dir / "autoregressive_four_panel.mp4"
    gif_path = mode_dir / "autoregressive_four_panel.gif"
    metrics_path = mode_dir / "metrics.csv"
    summary_path = mode_dir / "summary.json"
    fixed_goal = actual_frames[-1]
    horizon = one_step_horizon(backend.max_time_length)
    current_image = actual_frames[0].copy()
    current_language = actual_frames[0].copy()
    panels: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    image_seconds = 0.0
    language_seconds = 0.0

    initial_error = _combined_error_panel(
        actual_frames[0],
        current_image,
        current_language,
    )
    initial_panel = make_labelled_panel(
        [actual_frames[0], current_image, current_language, initial_error],
        ["Actual t=0", "Image AR t=0", "Language AR t=0", "Error I | L"],
    )

    with VideoSink(
        video_path,
        fps=fps,
        frame_size=(RESOLUTION * 4, RESOLUTION),
    ) as sink:
        sink.write(initial_panel)
        panels.append(initial_panel)
        for index in range(transition_count):
            target = actual_frames[index + 1]
            started = time.perf_counter()
            image_predictions = backend.image_rollout(
                current_image,
                fixed_goal,
                horizon=horizon,
                steps=1,
            )
            image_seconds += time.perf_counter() - started
            started = time.perf_counter()
            language_predictions, _ = backend.language_rollout(
                current_language,
                instruction,
                horizon=horizon,
                steps=1,
            )
            language_seconds += time.perf_counter() - started
            _validate_rollout_frames(
                image_predictions,
                expected=1,
                branch="image_goal",
            )
            _validate_rollout_frames(
                language_predictions,
                expected=1,
                branch="language_goal",
            )
            current_image = image_predictions[0]
            current_language = language_predictions[0]
            rows.extend(
                [
                    _prediction_metric_row(
                        mode="autoregressive",
                        branch="image_goal",
                        start_index=0,
                        lead_steps=index + 1,
                        target_index=index + 1,
                        reference=target,
                        prediction=current_image,
                        lpips_metric=lpips_metric,
                    ),
                    _prediction_metric_row(
                        mode="autoregressive",
                        branch="language_goal",
                        start_index=0,
                        lead_steps=index + 1,
                        target_index=index + 1,
                        reference=target,
                        prediction=current_language,
                        lpips_metric=lpips_metric,
                    ),
                ]
            )
            error_panel = _combined_error_panel(
                target,
                current_image,
                current_language,
            )
            panel = make_labelled_panel(
                [target, current_image, current_language, error_panel],
                [
                    f"Actual t={index + 1}",
                    f"Image AR t={index + 1}",
                    f"Language AR t={index + 1}",
                    "Error I | L",
                ],
            )
            sink.write(panel)
            panels.append(panel)

    write_gif(gif_path, panels, fps=fps)
    write_csv(metrics_path, rows)
    endpoint_rows = [row for row in rows if row["target_index"] == transition_count]
    summary = {
        "classification": MODEL_STAGE_CLASSIFICATION,
        "stage_classification": MODEL_STAGE_CLASSIFICATION,
        "scope": dict(MODEL_STAGE_SCOPE),
        "mode": "autoregressive",
        "transition_count": transition_count,
        "metric_rows": len(rows),
        "video_frames": len(panels),
        "horizon": horizon,
        "lpips": lpips_metric.status(),
        "timing_seconds": {
            "image_goal": image_seconds,
            "language_goal": language_seconds,
        },
        "metrics": _aggregate_metric_rows(rows),
        "metric_slopes": _metric_slopes(rows),
        "endpoint_metrics": _aggregate_metric_rows(endpoint_rows),
        "outputs": {
            "video": video_path.relative_to(output_dir).as_posix(),
            "gif": gif_path.relative_to(output_dir).as_posix(),
            "metrics": metrics_path.relative_to(output_dir).as_posix(),
        },
    }
    write_json(summary_path, summary)
    return summary


MODE_ORDER = ("teacher_forced", "full_future", "autoregressive")


def _bounded_transitions(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > EXPECTED_STATES - 1:
        raise argparse.ArgumentTypeError(
            f"max transitions must be in [1,{EXPECTED_STATES - 1}]"
        )
    return parsed


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--max-transitions",
        type=_bounded_transitions,
        default=EXPECTED_STATES - 1,
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODE_ORDER,
        default=list(MODE_ORDER),
    )
    parser.add_argument("--fps", type=_positive_integer, default=FPS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--decode-chunk-size",
        type=_positive_integer,
        default=16,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lpips-weights", type=Path)
    replay_group = parser.add_mutually_exclusive_group(required=True)
    replay_group.add_argument(
        "--replay-source",
        type=Path,
        help=(
            "Use a separately rendered and independently verified canonical "
            "replay root (or frames.npz)"
        ),
    )
    replay_group.add_argument(
        "--allow-inline-legacy-renderer",
        action="store_true",
        help=(
            "DANGEROUS legacy compatibility: create LIBERO in the ODEWorld "
            "MuJoCo 2.3.7 process; never use for corrected formal results"
        ),
    )
    return parser.parse_args(argv)


def validate_output_directory(output_dir: Path) -> None:
    """Permit Slurm logs but reject every pre-existing result artifact."""
    if not output_dir.exists():
        return
    allowed = {
        path.name
        for path in output_dir.iterdir()
        if path.is_file()
        and path.name.startswith("slurm-")
        and path.suffix in {".out", ".err"}
    }
    conflicts = [path for path in output_dir.iterdir() if path.name not in allowed]
    if conflicts:
        raise FileExistsError(
            f"Refusing to overwrite existing result artifacts: {conflicts}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    replay_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    runtime = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "mujoco": _package_version("mujoco"),
        "libero": _package_version("libero"),
        "opencv": cv2.__version__,
        "numpy": np.__version__,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "gpu": gpu,
    }
    if args.replay_source is not None:
        renderer_provenance_path = (
            _resolve_replay_root(args.replay_source) / "renderer_provenance.json"
        )
        if not renderer_provenance_path.is_file():
            raise FileNotFoundError(
                f"Missing external renderer provenance: {renderer_provenance_path}"
            )
        runtime["external_renderer"] = json.loads(
            renderer_provenance_path.read_text(encoding="utf-8")
        )["renderer"]
        runtime["external_renderer_provenance"] = str(renderer_provenance_path)
        runtime["external_renderer_provenance_sha256"] = _sha256_file(
            renderer_provenance_path
        )
    return {
        "classification": BUNDLE_CLASSIFICATION,
        "stage_classification": MODEL_STAGE_CLASSIFICATION,
        "scope": {
            **dict(replay_metadata["scope"]),
            "max_transitions": args.max_transitions,
            "modes": args.modes,
            "camera_transform": "rgb_vhflip",
            "source_camera_resolution": SOURCE_RESOLUTION,
            "model_resolution": RESOLUTION,
            "resize_interpolation": "INTER_AREA",
            "replay_source": (
                "external FAST-WAM renderer"
                if args.replay_source is not None
                else "inline renderer"
            ),
            "success_rate_measured": False,
        },
        "material": dict(replay_metadata["material"]),
        "invariance": dict(replay_metadata["invariance"]),
        "checkpoints": {
            "pt_flow": _checkpoint_provenance(PT_FLOW_CHECKPOINT),
            "rae": _checkpoint_provenance(RAE_CHECKPOINT),
            "goal_predictor": _checkpoint_provenance(GOAL_PREDICTOR_CHECKPOINT),
        },
        "git": _git_metadata(),
        "runtime": runtime,
        "arguments": {
            "execution_output_dir": str(args.output_dir),
            "artifact_paths_are_relative": True,
            "max_transitions": args.max_transitions,
            "modes": args.modes,
            "fps": args.fps,
            "seed": args.seed,
            "decode_chunk_size": args.decode_chunk_size,
            "device": args.device,
            "lpips_weights": (
                str(args.lpips_weights) if args.lpips_weights is not None else None
            ),
            "replay_source": (
                str(args.replay_source) if args.replay_source is not None else None
            ),
            "allow_inline_legacy_renderer": args.allow_inline_legacy_renderer,
        },
    }


def required_dino_cache_files(torch_home: Path) -> tuple[Path, ...]:
    hub = torch_home / "hub"
    checkpoints = hub / "checkpoints"
    repository = hub / "facebookresearch_dinov2_main"
    return (
        checkpoints / "dinov2_vitb14_pretrain.pth",
        checkpoints / "dinov2_vitl14_reg4_pretrain.pth",
        checkpoints
        / "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth",
        checkpoints
        / "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth",
        repository / "hubconf.py",
        repository
        / "dinov2/thirdparty/CLIP/clip/bpe_simple_vocab_16e6.txt.gz",
    )


def validate_dino_cache(torch_home: Path) -> None:
    missing = [path for path in required_dino_cache_files(torch_home) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing offline DINO cache files: {missing}")


def validate_runtime(args: argparse.Namespace) -> None:
    import torch

    if args.device != "cuda":
        raise RuntimeError("Formal framewise inference must use --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; submit this runner through Slurm")
    required = [
        HDF5_PATH,
        *(path / "config.json" for path in (
            PT_FLOW_CHECKPOINT,
            RAE_CHECKPOINT,
            GOAL_PREDICTOR_CHECKPOINT,
        )),
        *(path / "model.safetensors" for path in (
            PT_FLOW_CHECKPOINT,
            RAE_CHECKPOINT,
            GOAL_PREDICTOR_CHECKPOINT,
        )),
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing local input/checkpoint files: {missing}")
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch"))
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
    value = {
        "classification": BUNDLE_CLASSIFICATION,
        "stage_classification": MODEL_STAGE_CLASSIFICATION,
        "status": status,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
        "started_at_utc": previous.get("started_at_utc", now),
        "completed_at_utc": now if status == "completed" else None,
        "failed_at_utc": now if status == "failed" else None,
        "max_transitions": args.max_transitions,
        "modes": args.modes,
        "error": (
            {"type": type(error).__name__, "message": str(error)}
            if error is not None
            else None
        ),
    }
    write_json(path, value, overwrite=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print("classification=partial rollout", flush=True)
    print(
        "scope=one LIBERO-90 demo; recorded expert replay input; "
        "ODEWorld visual prediction; no success rate",
        flush=True,
    )
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
        if args.replay_source is None:
            if not args.allow_inline_legacy_renderer:
                raise RuntimeError(
                    "Inline legacy rendering requires --allow-inline-legacy-renderer"
                )
            replay, replay_metadata = create_libero_red_replay(
                max_transitions=args.max_transitions,
                seed=args.seed,
            )
            saved_replay_outputs = save_replay_artifacts(
                args.output_dir,
                replay,
                replay_metadata,
                fps=args.fps,
            )
            replay_outputs = {
                name: Path(path).relative_to(args.output_dir).as_posix()
                for name, path in saved_replay_outputs.items()
            }
        else:
            replay, replay_metadata = load_external_replay(
                args.replay_source,
                args.output_dir,
                max_transitions=args.max_transitions,
            )
            replay_outputs = {
                "frames": "actual_replay/frames.npz",
                "video": "actual_replay/actual_replay.mp4",
                "diagnostics": "actual_replay/replay_diagnostics.json",
            }
        device = torch.device(args.device)
        backend = OdeWorldBackend(
            load_models(device),
            decode_chunk_size=args.decode_chunk_size,
        )
        lpips_metric = LpipsMetric(args.lpips_weights, device=args.device)
        summaries = {}
        for mode in MODE_ORDER:
            if mode not in args.modes:
                continue
            if mode == "teacher_forced":
                summaries[mode] = run_teacher_forced(
                    backend,
                    replay.frames,
                    args.output_dir,
                    instruction=INSTRUCTION,
                    fps=args.fps,
                    lpips_metric=lpips_metric,
                )
            elif mode == "full_future":
                summaries[mode] = run_full_future(
                    backend,
                    replay.frames,
                    args.output_dir,
                    instruction=INSTRUCTION,
                    fps=args.fps,
                    lpips_metric=lpips_metric,
                )
            elif mode == "autoregressive":
                summaries[mode] = run_autoregressive(
                    backend,
                    replay.frames,
                    args.output_dir,
                    instruction=INSTRUCTION,
                    fps=args.fps,
                    lpips_metric=lpips_metric,
                )
        torch.cuda.synchronize()
        provenance = collect_provenance(args, replay_metadata)
        write_json(args.output_dir / "provenance.json", provenance)
        summary = {
            "classification": BUNDLE_CLASSIFICATION,
            "stage_classification": MODEL_STAGE_CLASSIFICATION,
            "scope": replay_metadata["scope"],
            "requested_modes": args.modes,
            "completed_modes": list(summaries),
            "transition_count": args.max_transitions,
            "success_rate_measured": False,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20,
            "lpips": lpips_metric.status(),
            "runtime": {
                key: provenance["runtime"][key]
                for key in ("host", "slurm_job_id", "slurm_node")
            },
            "replay_outputs": replay_outputs,
            "mode_summaries": summaries,
        }
        write_json(args.output_dir / "summary.json", summary)
        _set_run_state(args.output_dir, status="completed", args=args)
        print(json.dumps(summary, indent=2, allow_nan=False), flush=True)
    except BaseException as error:
        _set_run_state(args.output_dir, status="failed", args=args, error=error)
        raise


if __name__ == "__main__":
    main()
