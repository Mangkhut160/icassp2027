"""Render ODEWorld start/goal inputs from real LIBERO states after a material edit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LIBERO_SOURCE = Path(
    os.environ.get(
        "LIBERO_SOURCE",
        "libero_source",
    )
).resolve()
RENDERER_PYTHON = os.environ.get("RENDERER_PYTHON", sys.executable)
HDF5_PATH = (
    Path(os.environ.get("LIBERO_ROOT", "."))
    / "libero_90"
    / "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove_demo.hdf5"
)
TASK_STEM = "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove"
DEMO_NAME = "demo_42"
FRAME_INDICES = {"start": 104, "goal": 139}
INSTRUCTION = "put the moka pot on the stove"
MATERIAL_NAME = "moka_pot_1_moka_pot_body"
TARGET_RGBA = np.asarray([0.95, 0.04, 0.04, 1.0], dtype=np.float32)
SOURCE_RESOLUTION = 128
MODEL_RESOLUTION = 256
RESOLUTION = MODEL_RESOLUTION
MOKA_ASSET_DIR = (
    LIBERO_SOURCE
    / "libero"
    / "libero"
    / "assets"
    / "turbosquid_objects"
    / "moka_pot"
)
SOURCE_ASSETS = ("moka_pot.xml", "metal_diff.png", "rubber_black.png")
# Optional remap prefix for legacy XML asset references recorded by older
# renderers; set CHILIOCOSM_ASSET_PREFIX to enable the remap.
LEGACY_CHILIOCOSM_ASSET_PREFIX = os.environ.get("CHILIOCOSM_ASSET_PREFIX", "")
CANONICAL_LIBERO_SOURCE = LIBERO_SOURCE


def canonical_libero_paths() -> dict[str, str]:
    core_root = CANONICAL_LIBERO_SOURCE / "libero" / "libero"
    return {
        "assets": str((core_root / "assets").resolve()),
        "bddl_files": str((core_root / "bddl_files").resolve()),
        "init_states": str((core_root / "init_files").resolve()),
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def canonical_renderer_fields(
    *,
    source_camera_resolution: int = SOURCE_RESOLUTION,
    model_resolution: int = MODEL_RESOLUTION,
    resize_interpolation: str = "INTER_AREA",
    camera_transform: str = "rgb_vhflip",
) -> dict[str, object]:
    if source_camera_resolution <= 0 or model_resolution <= 0:
        raise ValueError("Renderer resolutions must be positive")
    if not resize_interpolation or not camera_transform:
        raise ValueError("Renderer transform fields must be non-empty")
    return {
        "python": str(Path(RENDERER_PYTHON).resolve()),
        "mujoco": "3.3.2",
        "robosuite": "1.4.0",
        "numpy": "1.26.4",
        "opencv": "4.6.0.66",
        "mujoco_gl": "egl",
        "pyopengl_platform": "egl",
        "camera_transform": camera_transform,
        "source_camera_resolution": source_camera_resolution,
        "model_resolution": model_resolution,
        "resize_interpolation": resize_interpolation,
        "libero_source_root": str(CANONICAL_LIBERO_SOURCE),
        "libero_paths": canonical_libero_paths(),
    }


CANONICAL_RENDERER_FIELDS = canonical_renderer_fields()


def libero_core_module_path() -> Path:
    # ``libero`` is a namespace package in the FAST-WAM checkout and has no
    # top-level __file__. The importable implementation lives one level below.
    from libero import libero as libero_core

    module_file = getattr(libero_core, "__file__", None)
    if module_file is None:
        raise RuntimeError("libero.libero has no module file")
    return Path(module_file).resolve()


def validate_canonical_renderer_record(
    runtime: Mapping[str, object],
    *,
    source_camera_resolution: int = SOURCE_RESOLUTION,
    model_resolution: int = MODEL_RESOLUTION,
    resize_interpolation: str = "INTER_AREA",
    camera_transform: str = "rgb_vhflip",
) -> None:
    expected = canonical_renderer_fields(
        source_camera_resolution=source_camera_resolution,
        model_resolution=model_resolution,
        resize_interpolation=resize_interpolation,
        camera_transform=camera_transform,
    )
    mismatches = {
        key: {"expected": value, "actual": runtime.get(key)}
        for key, value in expected.items()
        if runtime.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Canonical renderer record mismatch: {mismatches}")
    if not str(runtime.get("libero_module", "")).startswith(str(CANONICAL_LIBERO_SOURCE) + "/"):
        raise RuntimeError(f"Unexpected LIBERO module: {runtime.get('libero_module')}")


def renderer_runtime_contract(
    *,
    require_canonical: bool = False,
    source_camera_resolution: int = SOURCE_RESOLUTION,
    model_resolution: int = MODEL_RESOLUTION,
    resize_interpolation: str = "INTER_AREA",
    camera_transform: str = "rgb_vhflip",
) -> dict:
    import mujoco
    from libero.libero import get_libero_path

    runtime = {
        "environment": "FAST-WAM" if require_canonical else "recorded runtime",
        "python": str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "mujoco": getattr(mujoco, "__version__", None) or package_version("mujoco"),
        "robosuite": package_version("robosuite"),
        "libero": package_version("libero"),
        "numpy": np.__version__,
        "opencv": package_version("opencv-python"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "pyopengl_platform": os.environ.get("PYOPENGL_PLATFORM"),
        "camera_transform": camera_transform,
        "source_camera_resolution": source_camera_resolution,
        "model_resolution": model_resolution,
        "resize_interpolation": resize_interpolation,
        "libero_source_root": str(LIBERO_SOURCE),
        "libero_module": str(libero_core_module_path()),
        "libero_paths": {
            name: str(Path(get_libero_path(name)).resolve())
            for name in ("assets", "bddl_files", "init_states")
        },
    }
    if require_canonical:
        validate_canonical_renderer_record(
            runtime,
            source_camera_resolution=source_camera_resolution,
            model_resolution=model_resolution,
            resize_interpolation=resize_interpolation,
            camera_transform=camera_transform,
        )
        if os.environ.get("SLURM_JOB_ID") in (None, ""):
            raise RuntimeError("Canonical renderer must run through Slurm")
    return runtime


def select_exact_material(
    names: Sequence[str | None],
    target_name: str,
) -> tuple[int, str]:
    matches = [(index, name) for index, name in enumerate(names) if name == target_name]
    if len(matches) != 1:
        related = [(index, name) for index, name in enumerate(names) if name and "moka_pot" in name]
        raise RuntimeError(
            f"Expected exactly one material named {target_name!r}, found {matches}. "
            f"Related materials: {related}"
        )
    material_id, material_name = matches[0]
    return material_id, str(material_name)


def select_state_rows(
    states: np.ndarray,
    indices: Mapping[str, int],
) -> dict[str, np.ndarray]:
    if states.ndim != 2:
        raise ValueError(f"Expected a two-dimensional state array, got {states.shape}")
    selected = {}
    for label, index in indices.items():
        if index < 0 or index >= states.shape[0]:
            raise IndexError(
                f"State index for {label} is {index}, but the trajectory has {states.shape[0]} rows"
            )
        selected[label] = np.asarray(states[index]).copy()
    return selected


def transform_agent_view(image_rgb: np.ndarray, *, resolution: int = RESOLUTION) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {image_rgb.shape}")
    flipped = np.ascontiguousarray(image_rgb[::-1, ::-1])
    if flipped.shape[:2] == (resolution, resolution):
        return flipped.copy()
    return cv2.resize(flipped, (resolution, resolution), interpolation=cv2.INTER_AREA)


def image_difference_metrics(original: np.ndarray, changed: np.ndarray) -> dict:
    if original.shape != changed.shape:
        raise ValueError(f"Image shapes differ: {original.shape} vs {changed.shape}")
    delta = np.abs(changed.astype(np.int16) - original.astype(np.int16))
    mask = np.any(delta != 0, axis=2)
    ys, xs = np.nonzero(mask)
    bbox = None if len(xs) == 0 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return {
        "changed_pixels": int(mask.sum()),
        "changed_fraction": float(mask.mean()),
        "changed_bbox_xyxy": bbox,
        "mean_absolute_channel_delta": float(delta.mean()),
        "max_absolute_channel_delta": int(delta.max(initial=0)),
    }


def build_manifest() -> dict:
    return {
        "dataset": "libero",
        "cases": [
            {
                "id": "case_04",
                "start_image": "case_04/start.png",
                "goal_image": "case_04/goal.png",
                "instruction": INSTRUCTION,
            }
        ],
        "provenance": {
            "classification": "partial rollout input-render probe",
            "input_source": "LIBERO/MuJoCo runtime render",
            "hdf5": str(HDF5_PATH),
            "demo": DEMO_NAME,
            "frames": FRAME_INDICES,
            "material": MATERIAL_NAME,
        },
    }


def rewrite_chiliocosm_asset_paths(xml_string: str, asset_root: Path) -> str:
    root = ET.fromstring(xml_string)
    for element in root.iter():
        file_path = element.get("file")
        if file_path and LEGACY_CHILIOCOSM_ASSET_PREFIX and file_path.startswith(LEGACY_CHILIOCOSM_ASSET_PREFIX):
            relative_path = file_path.removeprefix(LEGACY_CHILIOCOSM_ASSET_PREFIX)
            element.set("file", str(asset_root / relative_path))
    return ET.tostring(root, encoding="utf8").decode("utf8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def physics_hashes(model) -> dict[str, str]:
    return {
        "body_mass": sha256_array(np.asarray(model.body_mass)),
        "geom_friction": sha256_array(np.asarray(model.geom_friction)),
        "geom_contype": sha256_array(np.asarray(model.geom_contype)),
        "geom_conaffinity": sha256_array(np.asarray(model.geom_conaffinity)),
    }


def material_names(sim_model) -> list[str | None]:
    import mujoco

    raw_model = getattr(sim_model, "_model", sim_model)
    return [
        mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_MATERIAL, material_id)
        for material_id in range(raw_model.nmat)
    ]


def save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write {path}")


def labelled(image_rgb: np.ndarray, label: str) -> np.ndarray:
    panel = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(
        panel,
        label,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def make_comparison(
    original: Mapping[str, np.ndarray],
    red: Mapping[str, np.ndarray],
) -> np.ndarray:
    rows = []
    for label in ("start", "goal"):
        rows.append(
            np.concatenate(
                [
                    labelled(original[label], f"{label}: original material"),
                    labelled(red[label], f"{label}: red MuJoCo material"),
                ],
                axis=1,
            )
        )
    return np.concatenate(rows, axis=0)


def psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.shape != candidate.shape:
        raise ValueError(f"Image shapes differ: {reference.shape} vs {candidate.shape}")
    mse = float(np.mean((reference.astype(np.float32) - candidate.astype(np.float32)) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def find_task(suite) -> tuple[int, object]:
    matches = []
    for task_id in range(suite.get_num_tasks()):
        task = suite.get_task(task_id)
        if Path(task.bddl_file).stem == TASK_STEM:
            matches.append((task_id, task))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one task named {TASK_STEM}, found {matches}")
    return matches[0]


def capture_state(env, state: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    obs = env.set_init_state(state)
    restored_state = np.asarray(env.get_sim_state()).copy()
    images = {
        "agentview": transform_agent_view(np.asarray(obs["agentview_image"])),
        "wrist": transform_agent_view(np.asarray(obs["robot0_eye_in_hand_image"])),
    }
    return images, restored_state


def _source_hashes() -> dict[str, str]:
    hashes = {"hdf5": sha256_file(HDF5_PATH)}
    hashes.update({name: sha256_file(MOKA_ASSET_DIR / name) for name in SOURCE_ASSETS})
    return hashes


def _assert_clean_output(output_dir: Path) -> None:
    protected = (
        "summary.json",
        "start_original.png",
        "start_red_material.png",
        "goal_original.png",
        "goal_red_material.png",
        "render_comparison.png",
        "dataset",
    )
    conflicts = [output_dir / name for name in protected if (output_dir / name).exists()]
    if conflicts:
        raise RuntimeError(f"Refusing to overwrite existing preview artifacts: {conflicts}")


def run_preview(output_dir: Path, *, seed: int = 42) -> dict:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _assert_clean_output(output_dir)
    started = time.perf_counter()
    renderer_runtime = renderer_runtime_contract(require_canonical=True)
    source_hashes_before = _source_hashes()

    with h5py.File(HDF5_PATH, "r") as handle:
        demo = handle["data"][DEMO_NAME]
        model_xml = demo.attrs["model_file"]
        if isinstance(model_xml, bytes):
            model_xml = model_xml.decode("utf-8")
        if not isinstance(model_xml, str):
            raise TypeError(f"Expected model_file XML string, got {type(model_xml)}")
        states_array = np.asarray(demo["states"])
        states = select_state_rows(states_array, FRAME_INDICES)
        recorded_frames = {
            label: transform_agent_view(np.asarray(demo["obs"]["agentview_rgb"][index]))
            for label, index in FRAME_INDICES.items()
        }

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
        from libero.libero.utils import utils as libero_utils

        processed_model_xml = libero_utils.postprocess_model_xml(model_xml, {})
        processed_model_xml = rewrite_chiliocosm_asset_paths(
            processed_model_xml,
            LIBERO_SOURCE / "libero/libero/assets",
        )
        env.reset_from_xml_string(processed_model_xml)
        env.sim.reset()
        environment_state_length = int(np.asarray(env.get_sim_state()).size)
        hdf5_state_length = int(states_array.shape[1])
        if environment_state_length != hdf5_state_length:
            raise RuntimeError(
                f"State length mismatch: environment={environment_state_length}, "
                f"HDF5={hdf5_state_length}"
            )

        original_images = {}
        original_wrist_images = {}
        original_states = {}
        input_state_deltas = {}
        for label, state in states.items():
            images, restored_state = capture_state(env, state)
            original_images[label] = images["agentview"]
            original_wrist_images[label] = images["wrist"]
            original_states[label] = restored_state
            input_state_deltas[label] = float(
                np.abs(restored_state.astype(np.float64) - state.astype(np.float64)).max(initial=0.0)
            )

        model = env.sim.model
        names = material_names(model)
        material_id, resolved_material_name = select_exact_material(names, MATERIAL_NAME)
        rgba_before = np.asarray(model.mat_rgba[material_id]).copy()
        physics_before = physics_hashes(model)
        model.mat_rgba[material_id] = TARGET_RGBA
        env.sim.forward()

        red_images = {}
        red_wrist_images = {}
        red_states = {}
        material_state_deltas = {}
        for label, state in states.items():
            images, restored_state = capture_state(env, state)
            red_images[label] = images["agentview"]
            red_wrist_images[label] = images["wrist"]
            red_states[label] = restored_state
            material_state_deltas[label] = float(
                np.abs(restored_state.astype(np.float64) - original_states[label].astype(np.float64)).max(
                    initial=0.0
                )
            )

        physics_after = physics_hashes(model)
        rgba_after = np.asarray(model.mat_rgba[material_id]).copy()

        for label in FRAME_INDICES:
            save_rgb(output_dir / f"{label}_original.png", original_images[label])
            save_rgb(output_dir / f"{label}_red_material.png", red_images[label])
        save_rgb(output_dir / "dataset/libero/case_04/start.png", red_images["start"])
        save_rgb(output_dir / "dataset/libero/case_04/goal.png", red_images["goal"])
        manifest_path = output_dir / "dataset/libero/manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(build_manifest(), indent=2) + "\n", encoding="utf-8")
        comparison_path = output_dir / "render_comparison.png"
        comparison = make_comparison(original_images, red_images)
        if not cv2.imwrite(str(comparison_path), comparison):
            raise RuntimeError(f"Failed to write {comparison_path}")

        source_hashes_after = _source_hashes()
        agent_metrics = {
            label: image_difference_metrics(original_images[label], red_images[label])
            for label in FRAME_INDICES
        }
        wrist_metrics = {
            label: image_difference_metrics(original_wrist_images[label], red_wrist_images[label])
            for label in FRAME_INDICES
        }
        recorded_psnr = {
            label: psnr(recorded_frames[label], original_images[label]) for label in FRAME_INDICES
        }
        checks = {
            "input_states_restored_exactly": all(value == 0.0 for value in input_state_deltas.values()),
            "material_edit_preserved_states": all(value == 0.0 for value in material_state_deltas.values()),
            "physics_arrays_unchanged": physics_before == physics_after,
            "source_files_unchanged": source_hashes_before == source_hashes_after,
            "agentview_changed_both_states": all(
                0 < metrics["changed_pixels"] and metrics["changed_fraction"] < 0.10
                for metrics in agent_metrics.values()
            ),
            "wrist_changed_both_states": all(metrics["changed_pixels"] > 0 for metrics in wrist_metrics.values()),
        }
        failed_checks = [name for name, passed in checks.items() if not passed]
        if failed_checks:
            raise RuntimeError(f"Preview verification failed: {failed_checks}")

        summary = {
            "classification": "partial rollout",
            "scope": {
                "suite": "libero_90",
                "task_id": task_id,
                "task_name": TASK_STEM,
                "instruction": task.language,
                "hdf5": str(HDF5_PATH),
                "demo": DEMO_NAME,
                "frames": FRAME_INDICES,
                "environment_xml_source": "data/demo_42.attrs/model_file",
                "transform": "rgb_vhflip",
                "source_camera_resolution": SOURCE_RESOLUTION,
                "model_resolution": MODEL_RESOLUTION,
                "resize_interpolation": "INTER_AREA",
                "seed": seed,
                "env_step_calls": 0,
                "robot_actions_executed": 0,
                "odeworld_inference_run": False,
                "success_rate_measured": False,
            },
            "runtime": {
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "host": os.uname().nodename,
                "elapsed_seconds": time.perf_counter() - started,
                "renderer": renderer_runtime,
            },
            "states": {
                "trajectory_rows": int(states_array.shape[0]),
                "hdf5_state_length": hdf5_state_length,
                "environment_state_length": environment_state_length,
                "input_max_abs_deltas": input_state_deltas,
                "material_max_abs_deltas": material_state_deltas,
            },
            "material": {
                "material_id": material_id,
                "material_name": resolved_material_name,
                "rgba_before": rgba_before.tolist(),
                "rgba_after": rgba_after.tolist(),
                "target_rgba": TARGET_RGBA.tolist(),
            },
            "invariance": {
                "physics_hashes_before": physics_before,
                "physics_hashes_after": physics_after,
                "source_hashes_before": source_hashes_before,
                "source_hashes_after": source_hashes_after,
            },
            "images": {
                "agentview_difference": agent_metrics,
                "wrist_difference": wrist_metrics,
                "recorded_vs_rerendered_original_psnr_db": recorded_psnr,
            },
            "outputs": {
                "comparison": "render_comparison.png",
                "dataset": "dataset/libero",
            },
            "checks": checks,
            "all_checks_passed": True,
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary
    finally:
        if env is not None:
            env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_preview(args.output_dir, seed=args.seed)
    print(json.dumps(result, indent=2))
