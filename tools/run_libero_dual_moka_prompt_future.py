#!/usr/bin/env python3
"""Generate three ODEWorld prompt-conditioned futures from one dual-moka image.

Classification: mock test. This script reads a saved RGB render and never
creates LIBERO/MuJoCo or executes robot actions.
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
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from tools.run_libero_framewise_odeworld import (
    GOAL_PREDICTOR_CHECKPOINT,
    PT_FLOW_CHECKPOINT,
    RAE_CHECKPOINT,
    VideoSink,
    load_models,
    make_labelled_panel,
    write_csv,
    write_gif,
    write_json,
)
from tools.run_libero_language_color_ablation import (
    ColorAblationBackend,
    build_red_mask,
    color_scores,
    latent_pair_metrics,
)
from tools.render_libero_dual_moka_aligned_scene import (
    CANONICAL_ASSET_ROOTS,
    HDF5_PATH,
    LIBERO_SOURCE,
    SOURCE_BDDL,
    sha256_array,
)
from tools.render_libero_material_inputs import validate_canonical_renderer_record


ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {
    "neutral": "put the moka pot on the stove",
    "red": "put the red moka pot on the stove",
    "silver": "put the silver moka pot on the stove",
}
CONDITIONS = tuple(PROMPTS)
PAIR_DEFINITIONS = (
    ("neutral", "red"),
    ("neutral", "silver"),
    ("red", "silver"),
)
CLASSIFICATION = "mock test"
RESOLUTION = 256
DEFAULT_STEPS = 200
DEFAULT_HORIZON = 1.0
DEFAULT_FPS = 10
DEFAULT_SEED = 42


def _validate_rgb(image: np.ndarray, *, name: str = "image") -> None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected {name} shape [H,W,3], got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"Expected {name} dtype uint8, got {image.dtype}")


def preprocess_source_image(image_rgb: np.ndarray) -> np.ndarray:
    """Resize RGB input exactly once without flipping or swapping channels."""
    _validate_rgb(image_rgb, name="source image")
    return cv2.resize(
        image_rgb,
        (RESOLUTION, RESOLUTION),
        interpolation=cv2.INTER_AREA,
    )


def run_metadata(*, steps: int, horizon: float, fps: int) -> dict[str, Any]:
    if steps <= 0 or horizon <= 0 or fps <= 0:
        raise ValueError("steps, horizon, and fps must be positive")
    return {
        "classification": CLASSIFICATION,
        "steps": int(steps),
        "output_frames_per_condition": int(steps),
        "horizon": float(horizon),
        "integrator": "rk4",
        "fps": int(fps),
        "resolution": RESOLUTION,
        "env_step_calls": 0,
        "robot_actions_executed": 0,
        "success_rate_measured": False,
    }


def _stable_pairwise_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    _validate_rgb(first, name="first frame")
    _validate_rgb(second, name="second frame")
    if first.shape != second.shape:
        raise ValueError(f"Frame shapes differ: {first.shape} vs {second.shape}")
    difference = second.astype(np.float64) - first.astype(np.float64)
    mse = float(np.mean(np.square(difference), dtype=np.float64))
    changed = np.any(first != second, axis=2)
    return {
        "rgb_l1": float(np.mean(np.abs(difference), dtype=np.float64)),
        "mse": mse,
        "psnr_db": math.inf if mse == 0.0 else 10.0 * math.log10((255.0**2) / mse),
        "changed_fraction": float(changed.mean()),
    }


def build_pairwise_rollout_rows(
    rollouts: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    if tuple(rollouts) != CONDITIONS:
        raise ValueError(f"Expected rollout conditions {CONDITIONS}, got {tuple(rollouts)}")
    frame_count = len(rollouts[CONDITIONS[0]])
    if frame_count <= 0:
        raise ValueError("Rollouts must be non-empty")
    expected_shape = rollouts[CONDITIONS[0]].shape
    if expected_shape[-1:] != (3,) or any(value.shape != expected_shape for value in rollouts.values()):
        raise ValueError("All rollouts must share shape [T,H,W,3]")
    rows = []
    for frame_index in range(frame_count):
        for first, second in PAIR_DEFINITIONS:
            metrics = _stable_pairwise_metrics(
                rollouts[first][frame_index],
                rollouts[second][frame_index],
            )
            rows.append(
                {
                    "frame_index": frame_index,
                    "first": first,
                    "second": second,
                    **metrics,
                }
            )
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing source JSON artifact: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected source JSON object: {path}")
    return value


def _verification_output_paths(record: Mapping[str, Any]) -> list[Path]:
    return [
        Path(str(record[key])).resolve()
        for key in ("output_dir", "verified_output")
        if record.get(key) not in (None, "")
    ]


def _verification_digest_seal(record: Mapping[str, Any]) -> dict[str, Any]:
    recomputed = record.get("recomputed", {})
    return {
        "summary_sha256": record.get(
            "summary_sha256", recomputed.get("summary_sha256")
        ),
        "run_state_sha256": record.get("run_state_sha256"),
        "artifact_sha256": record.get(
            "artifact_sha256", recomputed.get("artifact_sha256")
        ),
    }


def validate_promotion_lineage(
    output_dir: Path,
    *,
    classification: str,
    run_state: Mapping[str, Any],
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    promotion_path = output_dir / "promotion.json"
    promotion = _load_json_object(promotion_path)
    if (
        promotion.get("classification") != classification
        or promotion.get("atomic_rename") is not True
        or promotion.get("pre_promotion_all_checks_passed") is not True
        or Path(str(promotion.get("final_destination", ""))).resolve() != output_dir
        or str(promotion.get("slurm_job_id", ""))
        != str(run_state.get("slurm_job_id", ""))
    ):
        raise RuntimeError("Promotion record does not match the selected final artifact")
    staging = Path(str(promotion.get("staging_source", ""))).resolve()
    if staging.parent != output_dir.parent or staging == output_dir or staging.exists():
        raise RuntimeError("Promotion record does not prove a completed atomic rename")

    preserved_relative = Path(
        str(promotion.get("pre_promotion_verification_path", ""))
    )
    if preserved_relative.is_absolute():
        raise RuntimeError("Pre-promotion verification path must be relative")
    preserved_path = (output_dir / preserved_relative).resolve()
    if not preserved_path.is_relative_to(output_dir) or not preserved_path.is_file():
        raise RuntimeError("Preserved pre-promotion verification is missing")
    preserved_digest = sha256_file(preserved_path)
    if preserved_digest != promotion.get("pre_promotion_verification_sha256"):
        raise RuntimeError("Preserved pre-promotion verification hash mismatch")
    preserved = _load_json_object(preserved_path)
    if (
        preserved.get("classification") != classification
        or preserved.get("all_checks_passed") is not True
        or staging not in _verification_output_paths(preserved)
    ):
        raise RuntimeError("Preserved pre-promotion verification is not a matching pass")
    current_outputs = _verification_output_paths(verification)
    if not current_outputs or any(path not in {staging, output_dir} for path in current_outputs):
        raise RuntimeError("Current verification is not bound through the promotion record")
    if _verification_digest_seal(preserved) != _verification_digest_seal(verification):
        raise RuntimeError("Artifact digests changed across promotion verification")
    return {
        "path": str(promotion_path),
        "sha256": sha256_file(promotion_path),
        "staging_source": str(staging),
        "final_destination": str(output_dir),
        "pre_promotion_verification": str(preserved_path),
        "pre_promotion_verification_sha256": preserved_digest,
        "promoted_at": promotion.get("promoted_at"),
    }


def validate_external_source_lineage(
    scope: Mapping[str, Any],
    sources: Mapping[str, Any],
    *,
    expected_hdf5: Path = HDF5_PATH,
    expected_bddl: Path = SOURCE_BDDL,
) -> dict[str, Any]:
    hdf5_path = Path(str(scope.get("hdf5", ""))).resolve()
    bddl_path = Path(str(sources.get("source_bddl", ""))).resolve()
    if hdf5_path != expected_hdf5.resolve() or not hdf5_path.is_file():
        raise RuntimeError(f"Source HDF5 is not canonical: {hdf5_path}")
    if bddl_path != expected_bddl.resolve() or not bddl_path.is_file():
        raise RuntimeError(f"Source BDDL is not canonical: {bddl_path}")
    hdf5_digest = sha256_file(hdf5_path)
    bddl_digest = sha256_file(bddl_path)
    if not (
        hdf5_digest == sources.get("hdf5_sha256_before")
        == sources.get("hdf5_sha256_after")
    ):
        raise RuntimeError("Source HDF5 changed after aligned verification")
    if bddl_digest != sources.get("source_bddl_sha256"):
        raise RuntimeError("Source BDDL changed after aligned verification")
    return {
        "hdf5": str(hdf5_path),
        "hdf5_sha256": hdf5_digest,
        "source_bddl": str(bddl_path),
        "source_bddl_sha256": bddl_digest,
    }


def _source_output_path(
    source_dir: Path,
    outputs: Mapping[str, Any],
    key: str,
) -> Path:
    relative = outputs.get(key)
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"Source summary is missing output {key!r}")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise RuntimeError(f"Source output {key!r} must be relative: {candidate}")
    resolved = (source_dir / candidate).resolve()
    if not resolved.is_relative_to(source_dir):
        raise RuntimeError(f"Source output {key!r} escapes source directory: {candidate}")
    if not resolved.is_file():
        raise RuntimeError(f"Missing source output {key!r}: {resolved}")
    return resolved


def validate_source_lineage(
    source_dir: Path,
    *,
    image_name: str = "dual_moka_start.png",
) -> dict[str, Any]:
    """Validate and describe one verified frame-0 aligned dual-moka source."""
    source_dir = source_dir.resolve()
    image_relative = Path(image_name)
    if image_relative.is_absolute() or image_relative.name != image_name:
        raise ValueError("source image name must be a plain filename")

    summary_path = source_dir / "summary.json"
    verification_path = source_dir / "verification_summary.json"
    run_state_path = source_dir / "run_state.json"
    summary = _load_json_object(summary_path)
    verification = _load_json_object(verification_path)
    run_state = _load_json_object(run_state_path)
    if summary.get("classification") != "partial rollout":
        raise RuntimeError("Source summary is not the verified partial rollout")
    if (
        verification.get("classification") != "partial rollout"
        or verification.get("scope", {}).get("verification_only") is not True
    ):
        raise RuntimeError("Source verification has the wrong classification")
    if (
        run_state.get("classification") != "partial rollout"
        or run_state.get("status") != "completed"
        or run_state.get("frame_index") != 0
        or not run_state.get("slurm_job_id")
        or run_state.get("slurm_job_id")
        != summary.get("runtime", {}).get("slurm_job_id")
    ):
        raise RuntimeError("Source run_state is not a completed frame-0 Slurm render")
    summary_checks = summary.get("checks")
    if (
        summary.get("all_checks_passed") is not True
        or not isinstance(summary_checks, dict)
        or not summary_checks
        or not all(bool(value) for value in summary_checks.values())
    ):
        raise RuntimeError("Source generator checks did not all pass")
    verification_checks = verification.get("checks")
    if (
        verification.get("all_checks_passed") is not True
        or not isinstance(verification_checks, dict)
        or not verification_checks
        or not all(bool(value) for value in verification_checks.values())
    ):
        raise RuntimeError("Source verification did not pass")

    promotion_lineage = validate_promotion_lineage(
        source_dir,
        classification="partial rollout",
        run_state=run_state,
        verification=verification,
    )
    if verification.get("recomputed", {}).get("summary_sha256") != sha256_file(
        summary_path
    ):
        raise RuntimeError("Source summary changed after aligned verification")
    if verification.get("run_state_sha256") != sha256_file(run_state_path):
        raise RuntimeError("Source run_state changed after aligned verification")

    source_scope = summary.get("scope", {})
    expected_scope = {
        "suite": "libero_90",
        "task_id": 19,
        "demo": "demo_42",
        "frame": 0,
        "transform": "rgb_vhflip",
        "resolution": 512,
        "env_step_calls": 0,
        "robot_actions_executed": 0,
        "odeworld_inference_run": False,
        "success_rate_measured": False,
    }
    mismatches = {
        key: {"expected": value, "actual": source_scope.get(key)}
        for key, value in expected_scope.items()
        if source_scope.get(key) != value
    }
    if mismatches:
        if "frame" in mismatches:
            raise RuntimeError(f"Source must be aligned to frame 0: {mismatches}")
        raise RuntimeError(f"Source scope does not match the aligned dual-moka render: {mismatches}")

    runtime = summary.get("runtime", {})
    renderer = runtime.get("renderer", {})
    validate_canonical_renderer_record(
        renderer,
        source_camera_resolution=512,
        model_resolution=512,
        resize_interpolation="none",
    )
    if not runtime.get("slurm_job_id"):
        raise RuntimeError("Source renderer record is missing its Slurm job ID")
    if summary.get("sources", {}).get("libero_source") != str(LIBERO_SOURCE):
        raise RuntimeError("Source is not rooted in the configured LIBERO checkout")

    outputs = summary.get("outputs", {})
    if outputs.get("dual_moka_start") != image_relative.name:
        raise RuntimeError(
            "Requested source image is not the verified dual_moka_start output"
        )
    image_path = _source_output_path(source_dir, outputs, "dual_moka_start")
    scene_path = _source_output_path(source_dir, outputs, "scene_xml")
    asset_manifest_path = _source_output_path(source_dir, outputs, "asset_manifest")
    state_evidence_path = _source_output_path(source_dir, outputs, "state_evidence")

    artifact_sha256 = summary.get("artifact_sha256")
    verified_artifact_sha256 = verification.get("recomputed", {}).get(
        "artifact_sha256"
    )
    required_artifacts = {
        str(path.relative_to(source_dir))
        for path in (image_path, scene_path, asset_manifest_path, state_evidence_path)
    }
    if (
        not isinstance(artifact_sha256, dict)
        or not artifact_sha256
        or artifact_sha256 != verified_artifact_sha256
        or not required_artifacts.issubset(artifact_sha256)
    ):
        raise RuntimeError("Source artifact digest map was not independently verified")
    for relative, expected_digest in artifact_sha256.items():
        artifact_path = _source_output_path(
            source_dir, {"artifact": relative}, "artifact"
        )
        if sha256_file(artifact_path) != expected_digest:
            raise RuntimeError(
                f"Source artifact changed after verification: {relative}"
            )

    sources = summary.get("sources", {})
    if sha256_file(scene_path) != sources.get("dual_model_xml_sha256"):
        raise RuntimeError("Source scene.xml hash does not match its summary")
    asset_manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(asset_manifest, list) or not asset_manifest:
        raise RuntimeError("Source asset manifest must be a non-empty list")
    manifest_hash = _canonical_json_sha256(asset_manifest)
    if (
        manifest_hash != sources.get("asset_manifest_sha256")
        or len(asset_manifest) != sources.get("asset_count")
    ):
        raise RuntimeError("Source asset manifest does not match its summary")
    for index, record in enumerate(asset_manifest):
        if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
            raise RuntimeError(f"Invalid source asset manifest record {index}")
        asset_path = Path(str(record["path"])).resolve()
        if not any(asset_path.is_relative_to(root) for root in CANONICAL_ASSET_ROOTS):
            raise RuntimeError(f"Source asset {index} is outside canonical roots")
        digest = str(record["sha256"])
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise RuntimeError(f"Invalid source asset hash in record {index}")
        if not isinstance(record["bytes"], int) or record["bytes"] < 0:
            raise RuntimeError(f"Invalid source asset byte count in record {index}")

    state_mapping = summary.get("state_mapping", {})
    with np.load(state_evidence_path, allow_pickle=False) as archive:
        if "selected_dual_state" not in archive.files:
            raise RuntimeError("Source state evidence has no selected_dual_state")
        selected_state = np.asarray(archive["selected_dual_state"])
        selected_clone_qpos = np.asarray(archive["selected_clone_qpos"])
        selected_clone_qvel = np.asarray(archive["selected_clone_qvel"])
    if (
        list(selected_state.shape) != state_mapping.get("selected_dual_state_shape")
        or selected_state.dtype.str != state_mapping.get("selected_dual_state_dtype")
        or sha256_array(selected_state) != state_mapping.get("selected_dual_state_sha256")
        or not np.isfinite(selected_state).all()
    ):
        raise RuntimeError("Source selected simulator state does not match its summary")
    if (
        selected_clone_qpos.tolist() != state_mapping.get("selected_clone_qpos")
        or selected_clone_qvel.tolist() != state_mapping.get("selected_clone_qvel")
    ):
        raise RuntimeError("Source selected clone pose does not match its summary")

    recomputed = verification.get("recomputed", {})
    if recomputed.get("frame_index") != 0 or recomputed.get("expected_frame_index") != 0:
        raise RuntimeError("Source verification did not independently confirm frame 0")
    if recomputed.get("asset_count") != len(asset_manifest):
        raise RuntimeError("Source verification asset count does not match")
    if recomputed.get("selected_dual_state_sha256") != sha256_array(selected_state):
        raise RuntimeError("Source verification selected-state hash does not match")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to decode source image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    _validate_rgb(image_rgb, name="decoded source image")
    if image_rgb.shape != (512, 512, 3):
        raise RuntimeError(f"Verified source image must be 512x512 RGB, got {image_rgb.shape}")

    return {
        "source_dir": str(source_dir),
        "source_summary": str(summary_path),
        "source_verification": str(verification_path),
        "source_run_state": str(run_state_path),
        "source_image": str(image_path),
        "source_image_name": image_relative.name,
        "source_image_sha256": sha256_file(image_path),
        "source_summary_sha256": sha256_file(summary_path),
        "source_verification_sha256": sha256_file(verification_path),
        "source_run_state_sha256": sha256_file(run_state_path),
        "source_promotion": promotion_lineage,
        "source_classification": summary["classification"],
        "source_env_step_calls": source_scope["env_step_calls"],
        "source_scope": source_scope,
        "source_renderer": renderer,
        "source_sources": sources,
        "source_artifact_sha256": artifact_sha256,
        "source_scene_xml": str(scene_path),
        "source_scene_xml_sha256": sources["dual_model_xml_sha256"],
        "source_asset_manifest": str(asset_manifest_path),
        "source_asset_manifest_file_sha256": sha256_file(asset_manifest_path),
        "source_asset_manifest_sha256": manifest_hash,
        "source_asset_count": len(asset_manifest),
        "source_assets": asset_manifest,
        "source_state_evidence": str(state_evidence_path),
        "source_state_evidence_sha256": sha256_file(state_evidence_path),
        "source_state_mapping": state_mapping,
        "source_selected_state_sha256": sha256_array(selected_state),
        "source_selected_clone_qpos": selected_clone_qpos.tolist(),
        "source_selected_clone_qvel": selected_clone_qvel.tolist(),
        "source_selected_candidate": summary.get("scene", {}).get("selected_candidate"),
    }


def load_verified_source(
    source_dir: Path,
    *,
    image_name: str = "dual_moka_start.png",
) -> tuple[np.ndarray, dict[str, Any]]:
    provenance = validate_source_lineage(source_dir, image_name=image_name)
    image_bgr = cv2.imread(provenance["source_image"], cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to decode source image: {provenance['source_image']}")
    return preprocess_source_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)), provenance


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    _validate_rgb(image_rgb)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write image: {path}")


def _checkpoint_provenance(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    weights_path = path / "model.safetensors"
    return {
        "path": str(path),
        "config_sha256": sha256_file(config_path),
        "weights_sha256": sha256_file(weights_path),
        "weights_bytes": weights_path.stat().st_size,
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


def _validate_runtime(args: argparse.Namespace) -> None:
    import torch

    if (args.steps, args.horizon, args.fps) != (
        DEFAULT_STEPS,
        DEFAULT_HORIZON,
        DEFAULT_FPS,
    ):
        raise RuntimeError("Formal three-prompt inference requires 200 steps, horizon 1.0, 10 FPS")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This inference must run with --device cuda through Slurm")
    if os.environ.get("SLURM_JOB_ID") in (None, ""):
        raise RuntimeError("This inference must run through Slurm")
    if args.seed != DEFAULT_SEED:
        raise RuntimeError(f"Paired comparison requires seed {DEFAULT_SEED}")
    required = []
    for checkpoint in (PT_FLOW_CHECKPOINT, RAE_CHECKPOINT, GOAL_PREDICTOR_CHECKPOINT):
        required.extend((checkpoint / "config.json", checkpoint / "model.safetensors"))
    torch_home = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch"))
    dino_dir = torch_home / "hub" / "checkpoints"
    required.extend(
        (
            dino_dir / "dinov2_vitb14_pretrain.pth",
            dino_dir / "dinov2_vitl14_reg4_pretrain.pth",
            dino_dir
            / "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_text_encoder.pth",
            dino_dir
            / "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l_vision_head.pth",
        )
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing local checkpoint/cache files: {missing}")


def _validate_output_dir(output_dir: Path) -> None:
    if (output_dir / "summary.json").exists() or (output_dir / "run_state.json").exists():
        raise FileExistsError(f"Refusing to overwrite existing experiment: {output_dir}")


def _write_run_state(
    output_dir: Path,
    *,
    status: str,
    args: argparse.Namespace,
    error: BaseException | None = None,
) -> None:
    write_json(
        output_dir / "run_state.json",
        {
            "classification": CLASSIFICATION,
            "scope": "offline three-prompt visual future from a verified dual-moka render",
            "status": status,
            "prompts": PROMPTS,
            "steps": args.steps,
            "horizon": args.horizon,
            "fps": args.fps,
            "seed": args.seed,
            "decode_chunk_size": args.decode_chunk_size,
            "env_step_calls": 0,
            "robot_actions_executed": 0,
            "success_rate_measured": False,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "error": (
                {"type": type(error).__name__, "message": str(error)}
                if error is not None
                else None
            ),
        },
        overwrite=True,
    )


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _aggregate_pairwise_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = {}
    for first, second in PAIR_DEFINITIONS:
        selected = [row for row in rows if row["first"] == first and row["second"] == second]
        if not selected:
            raise ValueError(f"Missing pairwise rows for {first} vs {second}")
        summary[f"{first}_vs_{second}"] = {
            "frames": len(selected),
            "mean_rgb_l1": float(np.mean([float(row["rgb_l1"]) for row in selected])),
            "mean_psnr_db": float(np.mean([float(row["psnr_db"]) for row in selected])),
            "mean_changed_fraction": float(
                np.mean([float(row["changed_fraction"]) for row in selected])
            ),
            "endpoint_rgb_l1": float(selected[-1]["rgb_l1"]),
            "endpoint_psnr_db": _finite_or_none(float(selected[-1]["psnr_db"])),
            "endpoint_changed_fraction": float(selected[-1]["changed_fraction"]),
        }
    return summary


def _build_goal_rows(
    goal_latents: Mapping[str, np.ndarray],
    goal_images: Mapping[str, np.ndarray],
    red_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for first, second in PAIR_DEFINITIONS:
        latent = latent_pair_metrics(goal_latents[first], goal_latents[second])
        rgb = _stable_pairwise_metrics(goal_images[first], goal_images[second])
        rows.append(
            {
                "first": first,
                "second": second,
                "cosine_distance": latent["cosine_distance"],
                "normalized_l2": latent["normalized_l2"],
                "rgb_l1": rgb["rgb_l1"],
                "mse": rgb["mse"],
                "psnr_db": rgb["psnr_db"],
                "changed_fraction": rgb["changed_fraction"],
            }
        )
    for condition in CONDITIONS:
        scores = color_scores(goal_images[condition], red_mask)
        rows.append(
            {
                "first": condition,
                "second": "source_red_mask_diagnostic",
                "cosine_distance": "",
                "normalized_l2": "",
                "rgb_l1": "",
                "mse": "",
                "psnr_db": "",
                "changed_fraction": "",
                "red_score": scores["red_score"],
                "red_fraction": scores["red_fraction"],
                "silver_score": scores["silver_score"],
            }
        )
    normalized = []
    fields = (
        "first",
        "second",
        "cosine_distance",
        "normalized_l2",
        "rgb_l1",
        "mse",
        "psnr_db",
        "changed_fraction",
        "red_score",
        "red_fraction",
        "silver_score",
    )
    for row in rows:
        normalized.append({field: row.get(field, "") for field in fields})
    return normalized


def _build_color_rows(
    rollouts: Mapping[str, np.ndarray],
    red_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for frame_index in range(len(rollouts[CONDITIONS[0]])):
        for condition in CONDITIONS:
            scores = color_scores(rollouts[condition][frame_index], red_mask)
            rows.append(
                {
                    "frame_index": frame_index,
                    "condition": condition,
                    "mask_pixels": scores["mask_pixels"],
                    "red_score": scores["red_score"],
                    "red_fraction": scores["red_fraction"],
                    "neutrality": scores["neutrality"],
                    "brightness": scores["brightness"],
                    "silver_score": scores["silver_score"],
                }
            )
    return rows


def _aggregate_color_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        summary[condition] = {
            "frames": len(selected),
            "mean_red_score": float(np.mean([float(row["red_score"]) for row in selected])),
            "mean_red_fraction": float(
                np.mean([float(row["red_fraction"]) for row in selected])
            ),
            "mean_silver_score": float(
                np.mean([float(row["silver_score"]) for row in selected])
            ),
            "endpoint_red_score": float(selected[-1]["red_score"]),
            "endpoint_red_fraction": float(selected[-1]["red_fraction"]),
            "endpoint_silver_score": float(selected[-1]["silver_score"]),
        }
    return summary


def _save_media(
    output_dir: Path,
    source_image: np.ndarray,
    rollouts: Mapping[str, np.ndarray],
    *,
    fps: int,
) -> dict[str, Any]:
    media = {}
    rollout_dir = output_dir / "rollouts"
    for condition in CONDITIONS:
        frames = list(rollouts[condition])
        mp4_path = rollout_dir / f"{condition}.mp4"
        gif_path = rollout_dir / f"{condition}.gif"
        with VideoSink(mp4_path, fps=fps, frame_size=(RESOLUTION, RESOLUTION)) as sink:
            for frame in frames:
                sink.write(frame)
        write_gif(gif_path, frames, fps=fps)
        media[condition] = {"mp4": str(mp4_path.relative_to(output_dir)), "gif": str(gif_path.relative_to(output_dir))}

    comparison_frames = [
        make_labelled_panel(
            [source_image, *(rollouts[condition][index] for condition in CONDITIONS)],
            ["Fixed start", "Neutral future", "Red future", "Silver future"],
        )
        for index in range(len(rollouts[CONDITIONS[0]]))
    ]
    comparison_mp4 = output_dir / "prompt_comparison.mp4"
    comparison_gif = output_dir / "prompt_comparison.gif"
    with VideoSink(
        comparison_mp4,
        fps=fps,
        frame_size=(RESOLUTION * 4, RESOLUTION),
    ) as sink:
        for frame in comparison_frames:
            sink.write(frame)
    write_gif(comparison_gif, comparison_frames, fps=fps)
    media["comparison"] = {
        "mp4": comparison_mp4.name,
        "gif": comparison_gif.name,
    }
    return media


def _collect_provenance(
    args: argparse.Namespace,
    source: Mapping[str, Any],
    *,
    elapsed_seconds: float,
    peak_gpu_memory_bytes: int,
) -> dict[str, Any]:
    import torch

    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "classification": CLASSIFICATION,
        "scope": {
            "description": "offline three-prompt visual future from a verified dual-moka render",
            "libero_environment_created": False,
            "env_step_calls": 0,
            "robot_actions_executed": 0,
            "success_rate_measured": False,
            "ground_truth_future_available": False,
        },
        "source": dict(source),
        "prompts": PROMPTS,
        "checkpoints": {
            "pt_flow": _checkpoint_provenance(PT_FLOW_CHECKPOINT),
            "rae": _checkpoint_provenance(RAE_CHECKPOINT),
            "goal_predictor": _checkpoint_provenance(GOAL_PREDICTOR_CHECKPOINT),
        },
        "inference": {
            **run_metadata(steps=args.steps, horizon=args.horizon, fps=args.fps),
            "seed": args.seed,
            "dtype": "float32",
            "max_time_length": 50,
            "decode_chunk_size": args.decode_chunk_size,
            "models_loaded_once": True,
            "condition_metrics_are_pairwise_not_accuracy": True,
        },
        "runtime": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "host": platform.node(),
            "python_executable": str(Path(sys.executable).resolve()),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "torchdiffeq": _package_version("torchdiffeq"),
            "elapsed_seconds": elapsed_seconds,
            "gpu": {
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "peak_allocated_memory_bytes": peak_gpu_memory_bytes,
            },
        },
        "git": _git_metadata(),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-image-name", default="dual_moka_start.png")
    parser.add_argument(
        "--steps", type=int, choices=(DEFAULT_STEPS,), default=DEFAULT_STEPS
    )
    parser.add_argument(
        "--horizon", type=float, choices=(DEFAULT_HORIZON,), default=DEFAULT_HORIZON
    )
    parser.add_argument("--fps", type=int, choices=(DEFAULT_FPS,), default=DEFAULT_FPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--decode-chunk-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    print(
        "classification=mock test env_step_calls=0 robot_actions_executed=0 "
        "success_rate_measured=false ground_truth_future_available=false",
        flush=True,
    )
    print(f"prompts={json.dumps(PROMPTS)}", flush=True)
    _validate_output_dir(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_run_state(args.output_dir, status="running", args=args)
    started = time.perf_counter()
    try:
        _validate_runtime(args)
        import torch

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

        source_image, source_provenance = load_verified_source(
            args.source_dir,
            image_name=args.source_image_name,
        )
        red_mask = build_red_mask(source_image)
        if red_mask is None:
            raise RuntimeError("Could not derive a valid red-pot mask from the dual-moka source")
        _save_rgb(args.output_dir / "input.png", source_image)
        np.savez_compressed(args.output_dir / "input.npz", rgb=source_image)
        np.savez_compressed(args.output_dir / "source_red_mask.npz", mask=red_mask)

        backend = ColorAblationBackend(
            load_models(torch.device(args.device)),
            decode_chunk_size=args.decode_chunk_size,
        )
        print("models_loaded=3 once=true", flush=True)
        (args.output_dir / "goals").mkdir(parents=True, exist_ok=True)
        (args.output_dir / "rollouts").mkdir(parents=True, exist_ok=True)
        goal_latents: dict[str, np.ndarray] = {}
        goal_images: dict[str, np.ndarray] = {}
        rollouts: dict[str, np.ndarray] = {}
        timings = {}
        for condition in CONDITIONS:
            condition_started = time.perf_counter()
            latent, goal_image = backend.predict_goal(source_image, PROMPTS[condition])
            frames = backend.rollout_from_goal_latent(
                source_image,
                latent,
                horizon=args.horizon,
                steps=args.steps,
            )
            rollout = np.stack(frames, axis=0)
            if rollout.shape != (args.steps, RESOLUTION, RESOLUTION, 3):
                raise RuntimeError(f"Unexpected {condition} rollout shape: {rollout.shape}")
            if rollout.dtype != np.uint8 or not np.isfinite(rollout).all():
                raise RuntimeError(f"Invalid {condition} rollout dtype or values")
            goal_latents[condition] = np.asarray(latent, dtype=np.float32)
            goal_images[condition] = np.asarray(goal_image, dtype=np.uint8)
            rollouts[condition] = rollout
            timings[condition] = time.perf_counter() - condition_started
            _save_rgb(args.output_dir / "goals" / f"{condition}.png", goal_images[condition])
            np.savez_compressed(
                args.output_dir / "rollouts" / f"{condition}.npz",
                frames=rollout,
            )
            print(f"completed_condition={condition} frames={len(rollout)}", flush=True)

        np.savez_compressed(
            args.output_dir / "goals" / "goals.npz",
            **{f"{condition}_latent": goal_latents[condition] for condition in CONDITIONS},
            **{f"{condition}_rgb": goal_images[condition] for condition in CONDITIONS},
        )
        goal_panel = make_labelled_panel(
            [source_image, *(goal_images[condition] for condition in CONDITIONS)],
            ["Fixed start", "Neutral goal", "Red goal", "Silver goal"],
        )
        _save_rgb(args.output_dir / "goal_comparison.png", goal_panel)

        pairwise_rows = build_pairwise_rollout_rows(rollouts)
        color_rows = _build_color_rows(rollouts, red_mask)
        goal_rows = _build_goal_rows(goal_latents, goal_images, red_mask)
        write_csv(args.output_dir / "goal_metrics.csv", goal_rows)
        write_csv(args.output_dir / "rollout_pairwise_metrics.csv", pairwise_rows)
        write_csv(args.output_dir / "rollout_color_diagnostics.csv", color_rows)
        media = _save_media(args.output_dir, source_image, rollouts, fps=args.fps)

        torch.cuda.synchronize()
        peak_memory = int(torch.cuda.max_memory_allocated())
        elapsed = time.perf_counter() - started
        provenance = _collect_provenance(
            args,
            source_provenance,
            elapsed_seconds=elapsed,
            peak_gpu_memory_bytes=peak_memory,
        )
        summary = {
            "classification": CLASSIFICATION,
            "scope": provenance["scope"],
            "prompts": PROMPTS,
            "run": run_metadata(steps=args.steps, horizon=args.horizon, fps=args.fps),
            "counts": {
                "conditions": len(CONDITIONS),
                "goals": len(goal_images),
                "frames_per_condition": args.steps,
                "total_prediction_frames": args.steps * len(CONDITIONS),
                "pairwise_rows": len(pairwise_rows),
                "color_diagnostic_rows": len(color_rows),
            },
            "goal_metrics": goal_rows,
            "pairwise_rollout_summary": _aggregate_pairwise_rows(pairwise_rows),
            "color_diagnostic_summary": _aggregate_color_rows(color_rows),
            "condition_timing_seconds": timings,
            "elapsed_seconds": elapsed,
            "peak_gpu_memory_bytes": peak_memory,
            "media": media,
            "limitations": [
                "No ground-truth future is available for this static input.",
                "Pairwise PSNR/L1 measure condition differences, not accuracy.",
                "The fixed source red mask is a spatial diagnostic, not predicted-object segmentation.",
                "The outputs do not execute robot actions or establish task success.",
            ],
        }
        write_json(args.output_dir / "provenance.json", provenance)
        write_json(args.output_dir / "summary.json", summary)
        _write_run_state(args.output_dir, status="completed", args=args)
        print(
            f"completed=true frames_per_condition={args.steps} "
            f"pairwise_rows={len(pairwise_rows)} elapsed_seconds={elapsed:.3f}",
            flush=True,
        )
    except BaseException as error:
        _write_run_state(args.output_dir, status="failed", args=args, error=error)
        raise


if __name__ == "__main__":
    main()
