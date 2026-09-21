#!/usr/bin/env python3
"""Independently verify a FAST-WAM-rendered red-moka expert replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Callable, Mapping

import cv2
import numpy as np


LIBERO_SOURCE = Path(os.environ.get("LIBERO_SOURCE", "libero_source")).resolve()
RENDERER_PYTHON = os.environ.get("RENDERER_PYTHON", sys.executable)
_MOKA_POT_ASSETS = LIBERO_SOURCE / "libero/libero/assets/turbosquid_objects/moka_pot"
CANONICAL_SOURCE_PATHS = {
    "hdf5": (
        Path(os.environ.get("LIBERO_ROOT", "."))
        / "libero_90"
        / "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove_demo.hdf5"
    ),
    "moka_pot_xml": _MOKA_POT_ASSETS / "moka_pot.xml",
    "metal_diff": _MOKA_POT_ASSETS / "metal_diff.png",
    "rubber_black": _MOKA_POT_ASSETS / "rubber_black.png",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_files(
    source_files: object,
    *,
    expected_paths: Mapping[str, Path] = CANONICAL_SOURCE_PATHS,
    hash_function: Callable[[Path], str] = sha256_file,
) -> None:
    if not isinstance(source_files, Mapping):
        raise RuntimeError("Renderer does not record structured source-file hashes")
    if set(source_files) != set(expected_paths):
        raise RuntimeError(
            "Renderer source-file set mismatch: "
            f"expected={sorted(expected_paths)}, actual={sorted(source_files)}"
        )
    for name, expected_path in expected_paths.items():
        record = source_files[name]
        if not isinstance(record, Mapping):
            raise RuntimeError(f"Renderer source-file record is malformed: {name}")
        path = Path(str(record.get("path"))).resolve()
        if path != expected_path.resolve():
            raise RuntimeError(
                f"Renderer source-file path mismatch for {name}: "
                f"expected={expected_path.resolve()}, actual={path}"
            )
        if not path.is_file() or record.get("sha256") != hash_function(path):
            raise RuntimeError(f"Renderer source-file hash mismatch: {name}")


def video_info(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open replay video: {path}")
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape != (height, width, 3):
            raise RuntimeError("Replay video dimensions change across frames")
        decoded += 1
    capture.release()
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "declared_frames": declared,
        "decoded_frames": decoded,
        "fps": fps,
        "width": width,
        "height": height,
    }


def verify(output_dir: Path, *, expected_transitions: int) -> dict:
    output_dir = output_dir.resolve()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    run_state = json.loads((output_dir / "run_state.json").read_text(encoding="utf-8"))
    provenance = json.loads(
        (output_dir / "renderer_provenance.json").read_text(encoding="utf-8")
    )
    diagnostics = json.loads(
        (output_dir / "actual_replay/replay_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    for name, document in (
        ("run state", run_state),
        ("summary", summary),
        ("renderer provenance", provenance),
        ("replay diagnostics", diagnostics),
    ):
        if document.get("classification") != "partial rollout":
            raise RuntimeError(f"{name} lacks partial rollout classification")
    if summary.get("status") != "completed":
        raise RuntimeError("Renderer summary is not completed")
    if run_state.get("status") != "completed" or run_state.get("error") is not None:
        raise RuntimeError("Renderer run state is not cleanly completed")
    if run_state.get("slurm_job_id") in (None, ""):
        raise RuntimeError("Renderer run state lacks a Slurm job ID")
    renderer = provenance.get("renderer", {})
    expected_versions = {
        "mujoco": "3.3.2",
        "robosuite": "1.4.0",
        "numpy": "1.26.4",
        "opencv": "4.6.0.66",
    }
    mismatches = {
        key: {"expected": value, "actual": renderer.get(key)}
        for key, value in expected_versions.items()
        if renderer.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Renderer version mismatch: {mismatches}")
    if renderer.get("mujoco_gl") != "egl" or renderer.get("pyopengl_platform") != "egl":
        raise RuntimeError("Renderer did not record EGL for both MuJoCo and PyOpenGL")
    if Path(str(renderer.get("python"))).resolve() != Path(RENDERER_PYTHON).resolve():
        raise RuntimeError(f"Unexpected renderer Python: {renderer.get('python')}")
    if Path(str(renderer.get("libero_source_root"))).resolve() != LIBERO_SOURCE:
        raise RuntimeError(
            f"Unexpected LIBERO source root: {renderer.get('libero_source_root')}"
        )
    if not str(renderer.get("libero_module", "")).startswith(str(LIBERO_SOURCE) + "/"):
        raise RuntimeError(f"Unexpected LIBERO module: {renderer.get('libero_module')}")
    validate_source_files(renderer.get("source_files"))
    if renderer.get("camera_transform") != "rgb_vhflip":
        raise RuntimeError("Renderer camera transform is not rgb_vhflip")
    expected_camera = {
        "source_camera_resolution": 128,
        "model_resolution": 256,
        "resize_interpolation": "INTER_AREA",
    }
    camera_mismatches = {
        key: {"expected": value, "actual": renderer.get(key)}
        for key, value in expected_camera.items()
        if renderer.get(key) != value
    }
    if camera_mismatches:
        raise RuntimeError(f"Renderer camera protocol mismatch: {camera_mismatches}")
    if int(diagnostics.get("env_step_calls", -1)) != expected_transitions:
        raise RuntimeError("Replay env.step count mismatch")
    if diagnostics.get("scope", {}).get("success_rate_measured"):
        raise RuntimeError("Replay unexpectedly reports success rate")
    if not diagnostics.get("invariance", {}).get("physics_arrays_unchanged"):
        raise RuntimeError("Replay physics invariance is not verified")
    material = diagnostics.get("material", {})
    if material.get("material_name") != "moka_pot_1_moka_pot_body":
        raise RuntimeError(f"Unexpected edited material: {material.get('material_name')}")
    target_rgba = np.asarray([0.95, 0.04, 0.04, 1.0], dtype=np.float32)
    if not np.allclose(np.asarray(material.get("rgba_after")), target_rgba, atol=1e-6):
        raise RuntimeError(f"Unexpected edited material RGBA: {material.get('rgba_after')}")
    if diagnostics["invariance"].get("physics_hashes_before") != diagnostics[
        "invariance"
    ].get("physics_hashes_after"):
        raise RuntimeError("Replay physics hashes changed")
    rows = diagnostics.get("state_diagnostics", [])
    if len(rows) != expected_transitions:
        raise RuntimeError("Replay state diagnostic count mismatch")
    for index, row in enumerate(rows):
        if int(row["action_index"]) != index or int(row["target_state_index"]) != index + 1:
            raise RuntimeError(f"Replay state/action index mismatch at {index}")
        for key in ("mean_abs_state_error", "max_abs_state_error"):
            if not math.isfinite(float(row[key])):
                raise RuntimeError(f"Non-finite replay metric at {index}/{key}")
    frames_path = output_dir / "actual_replay/frames.npz"
    with np.load(frames_path, allow_pickle=False) as archive:
        if set(archive.files) != {"frames"}:
            raise RuntimeError(f"Unexpected replay arrays: {archive.files}")
        frames = np.asarray(archive["frames"])
    expected_shape = (expected_transitions + 1, 256, 256, 3)
    if frames.shape != expected_shape or frames.dtype != np.uint8:
        raise RuntimeError(f"Replay frames mismatch: {frames.shape} {frames.dtype}")
    if not np.isfinite(frames).all() or float(frames.std()) <= 1.0:
        raise RuntimeError("Replay frames are invalid or blank")
    video = video_info(output_dir / "actual_replay/actual_replay.mp4")
    if video["decoded_frames"] != expected_transitions + 1:
        raise RuntimeError("Replay video frame count mismatch")
    if (video["width"], video["height"]) != (256, 256) or abs(video["fps"] - 10) > 0.01:
        raise RuntimeError("Replay video format mismatch")
    alignment = diagnostics.get("unmodified_recorded_alignment", [])
    if len(alignment) != 152:
        raise RuntimeError("Expected all 152 HDF5 states in renderer alignment diagnostics")
    psnr = [float(row["psnr_db"]) for row in alignment if row.get("psnr_db") is not None]
    result = {
        "classification": "partial rollout",
        "all_checks_passed": True,
        "scope_verified": True,
        "renderer_versions_verified": True,
        "physics_invariance_verified": True,
        "expected_transitions": expected_transitions,
        "frame_count": int(frames.shape[0]),
        "env_step_calls": int(diagnostics["env_step_calls"]),
        "unmodified_alignment_mean_psnr_db": float(np.mean(psnr)),
        "unmodified_alignment_min_psnr_db": float(np.min(psnr)),
        "frames_sha256": sha256_file(frames_path),
        "replay_diagnostics_sha256": sha256_file(
            output_dir / "actual_replay/replay_diagnostics.json"
        ),
        "renderer_provenance_sha256": sha256_file(
            output_dir / "renderer_provenance.json"
        ),
        "run_state_sha256": sha256_file(output_dir / "run_state.json"),
        "video": video,
    }
    path = output_dir / "verification_summary.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-transitions", type=int, default=151)
    args = parser.parse_args()
    print(json.dumps(verify(args.output_dir, expected_transitions=args.expected_transitions), indent=2))
