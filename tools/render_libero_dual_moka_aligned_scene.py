"""Render a demo-state-aligned LIBERO scene with red and silver moka pots."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import h5py
import numpy as np


SOURCE_INSTANCE = "moka_pot_1"
CLONE_INSTANCE = "moka_pot_2"
SOURCE_BODY_MATERIAL = "moka_pot_1_moka_pot_body"
CLONE_BODY_MATERIAL = "moka_pot_2_moka_pot_body"
MIN_CLEARANCE_M = 0.03
MIN_MASK_PIXELS = 1000
MAX_BBOX_OVERLAP = 0.05
ROOT = Path(__file__).resolve().parents[1]
LIBERO_SOURCE = Path(
    os.environ.get("LIBERO_SOURCE", "libero_source")
).resolve()
HDF5_PATH = (
    Path(os.environ.get("LIBERO_ROOT", "."))
    / "libero_90"
    / "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove_demo.hdf5"
)
SOURCE_BDDL = (
    LIBERO_SOURCE
    / "libero/libero/bddl_files/libero_90/"
    "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove.bddl"
)
DEMO_NAME = "demo_42"
DEFAULT_FRAME_INDEX = 0
RESOLUTION = 512
RED_RGBA = np.asarray([0.95, 0.04, 0.04, 1.0], dtype=np.float32)
# Optional remap prefix for legacy XML asset references recorded by older
# renderers; set CHILIOCOSM_ASSET_PREFIX to enable the remap.
LEGACY_CHILIOCOSM_ASSET_PREFIX = os.environ.get("CHILIOCOSM_ASSET_PREFIX", "")
CLONE_JOINT = "moka_pot_2_joint0"
POT_RADIUS_SUM_M = 0.05
CANDIDATE_XY = (
    (0.20, 0.00),
    (0.20, -0.10),
    (0.20, 0.10),
    (0.20, -0.18),
    (0.20, 0.18),
    (0.08, -0.20),
    (0.08, 0.20),
    (-0.08, -0.20),
    (-0.08, 0.20),
    (0.28, -0.10),
    (0.28, 0.10),
)
PROTECTED_CENTERS_XY = {"stove": [118, 315], "pan": [397, 315]}
# Roots that recorded source assets must live under; override with a
# comma-separated CANONICAL_ASSET_ROOTS list, otherwise LIBERO_SOURCE is used.
CANONICAL_ASSET_ROOTS = tuple(
    Path(item).resolve()
    for item in os.environ.get("CANONICAL_ASSET_ROOTS", str(LIBERO_SOURCE)).split(",")
    if item
)


def _find_parent(root: ET.Element, child: ET.Element) -> ET.Element:
    for parent in root.iter():
        if child in list(parent):
            return parent
    raise ValueError("XML element has no parent")


def clone_moka_in_xml(xml_text: str) -> str:
    """Clone moka_pot_1 as moka_pot_2 without changing the camera or source object."""
    root = ET.fromstring(xml_text)
    source_body_matches = root.findall(f".//body[@name='{SOURCE_INSTANCE}_main']")
    if len(source_body_matches) != 1:
        raise RuntimeError(
            f"Expected one {SOURCE_INSTANCE}_main body, found {len(source_body_matches)}"
        )
    source_material_matches = root.findall(f".//material[@name='{SOURCE_BODY_MATERIAL}']")
    if len(source_material_matches) != 1:
        raise RuntimeError(
            f"Expected one {SOURCE_BODY_MATERIAL} material, found {len(source_material_matches)}"
        )
    if root.findall(f".//*[@name='{CLONE_INSTANCE}_main']"):
        raise RuntimeError(f"XML already contains {CLONE_INSTANCE}")

    source_body = source_body_matches[0]
    cloned_body = copy.deepcopy(source_body)
    for element in cloned_body.iter():
        name = element.get("name")
        if name and name.startswith(SOURCE_INSTANCE):
            element.set("name", CLONE_INSTANCE + name[len(SOURCE_INSTANCE) :])
        if element.get("material") == SOURCE_BODY_MATERIAL:
            element.set("material", CLONE_BODY_MATERIAL)

    source_material = source_material_matches[0]
    cloned_material = copy.deepcopy(source_material)
    cloned_material.set("name", CLONE_BODY_MATERIAL)
    _find_parent(root, source_material).append(cloned_material)
    _find_parent(root, source_body).append(cloned_body)
    return ET.tostring(root, encoding="unicode")


def transform_agent_view(image_rgb: np.ndarray) -> np.ndarray:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image [H,W,3], got {image_rgb.shape}")
    return np.ascontiguousarray(image_rgb[::-1, ::-1])


def red_color_mask(image_rgb: np.ndarray) -> np.ndarray:
    values = image_rgb.astype(np.int16)
    red, green, blue = values[:, :, 0], values[:, :, 1], values[:, :, 2]
    return (red >= 120) & (red >= green + 45) & (red >= blue + 45)


def difference_visibility_mask(
    reference_rgb: np.ndarray,
    candidate_rgb: np.ndarray,
    *,
    threshold: int = 8,
) -> np.ndarray:
    if reference_rgb.shape != candidate_rgb.shape:
        raise ValueError(
            f"Image shapes differ: {reference_rgb.shape} vs {candidate_rgb.shape}"
        )
    delta = np.abs(reference_rgb.astype(np.int16) - candidate_rgb.astype(np.int16))
    return np.any(delta > threshold, axis=2)


def bbox_overlap_fraction(first: Sequence[int], second: Sequence[int]) -> float:
    if len(first) != 4 or len(second) != 4:
        raise ValueError("Bounding boxes must be xyxy sequences")
    ax1, ay1, ax2, ay2 = (int(value) for value in first)
    bx1, by1, bx2, by2 = (int(value) for value in second)
    intersection_width = max(0, min(ax2, bx2) - max(ax1, bx1) + 1)
    intersection_height = max(0, min(ay2, by2) - max(ay1, by1) + 1)
    intersection = intersection_width * intersection_height
    first_area = max(0, ax2 - ax1 + 1) * max(0, ay2 - ay1 + 1)
    second_area = max(0, bx2 - bx1 + 1) * max(0, by2 - by1 + 1)
    smaller = min(first_area, second_area)
    return 0.0 if smaller == 0 else float(intersection / smaller)


def _candidate_passes(record: Mapping[str, object]) -> bool:
    return bool(
        not record["forbidden_contacts"]
        and float(record["clearance_m"]) >= MIN_CLEARANCE_M
        and int(record["red_pixels"]) >= MIN_MASK_PIXELS
        and int(record["silver_pixels"]) >= MIN_MASK_PIXELS
        and not bool(record["red_border"])
        and not bool(record["silver_border"])
        and float(record["bbox_overlap_fraction"]) < MAX_BBOX_OVERLAP
        and not bool(record.get("covers_protected_center", False))
    )


def candidate_failures(record: Mapping[str, object]) -> list[str]:
    failures = []
    if record["forbidden_contacts"]:
        failures.append("forbidden_contacts")
    if float(record["clearance_m"]) < MIN_CLEARANCE_M:
        failures.append("clearance")
    if int(record["red_pixels"]) < MIN_MASK_PIXELS:
        failures.append("red_pixels")
    if int(record["silver_pixels"]) < MIN_MASK_PIXELS:
        failures.append("silver_pixels")
    if bool(record["red_border"]):
        failures.append("red_border")
    if bool(record["silver_border"]):
        failures.append("silver_border")
    if float(record["bbox_overlap_fraction"]) >= MAX_BBOX_OVERLAP:
        failures.append("bbox_overlap")
    if bool(record.get("covers_protected_center", False)):
        failures.append("covers_protected_center")
    return failures


def select_candidate(records: Iterable[Mapping[str, object]]) -> dict[str, object]:
    valid = [dict(record) for record in records if _candidate_passes(record)]
    if not valid:
        raise RuntimeError("No placement candidate satisfies all hard constraints")
    return max(
        valid,
        key=lambda record: (
            int(record["red_pixels"]) + int(record["silver_pixels"]),
            float(record["clearance_m"]),
            str(record["candidate_id"]),
        ),
    )


JointSlices = Mapping[str, tuple[slice, slice]]


def copy_shared_joint_state(
    source_qpos: np.ndarray,
    source_qvel: np.ndarray,
    source_slices: JointSlices,
    target_qpos: np.ndarray,
    target_qvel: np.ndarray,
    target_slices: JointSlices,
    *,
    excluded: set[str],
) -> list[str]:
    missing = sorted(set(source_slices) - set(target_slices) - excluded)
    if missing:
        raise RuntimeError(f"Target simulation is missing shared joints: {missing}")
    copied = sorted(set(source_slices) & set(target_slices) - excluded)
    for name in copied:
        source_qpos_slice, source_qvel_slice = source_slices[name]
        target_qpos_slice, target_qvel_slice = target_slices[name]
        source_qpos_value = source_qpos[source_qpos_slice]
        source_qvel_value = source_qvel[source_qvel_slice]
        if source_qpos_value.shape != target_qpos[target_qpos_slice].shape:
            raise RuntimeError(f"qpos width differs for shared joint {name}")
        if source_qvel_value.shape != target_qvel[target_qvel_slice].shape:
            raise RuntimeError(f"qvel width differs for shared joint {name}")
        target_qpos[target_qpos_slice] = source_qpos_value
        target_qvel[target_qvel_slice] = source_qvel_value
    return copied


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def asset_manifest_from_xml(xml_text: str) -> list[dict[str, object]]:
    root = ET.fromstring(xml_text)
    paths: set[Path] = set()
    for element in root.iter():
        raw_path = element.get("file")
        if raw_path is None:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            raise RuntimeError(f"Compiled scene asset path is not absolute: {path}")
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Compiled scene asset does not exist: {resolved}")
        paths.add(resolved)
    if not paths:
        raise RuntimeError("Compiled scene XML contains no file-backed assets")
    return [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(paths, key=str)
    ]


def asset_paths_are_canonical(manifest: Sequence[Mapping[str, object]]) -> bool:
    return bool(manifest) and all(
        any(Path(str(record["path"])).is_relative_to(root) for root in CANONICAL_ASSET_ROOTS)
        for record in manifest
    )


def rewrite_legacy_asset_paths(xml_text: str) -> str:
    root = ET.fromstring(xml_text)
    asset_root = LIBERO_SOURCE / "libero/libero/assets"
    for element in root.iter():
        file_path = element.get("file")
        if file_path and LEGACY_CHILIOCOSM_ASSET_PREFIX and file_path.startswith(LEGACY_CHILIOCOSM_ASSET_PREFIX):
            relative = file_path.removeprefix(LEGACY_CHILIOCOSM_ASSET_PREFIX)
            element.set("file", str(asset_root / relative))
    return ET.tostring(root, encoding="unicode")


def _mujoco_names(model, object_type) -> list[str | None]:
    import mujoco

    raw_model = getattr(model, "_model", model)
    counts = {
        mujoco.mjtObj.mjOBJ_BODY: raw_model.nbody,
        mujoco.mjtObj.mjOBJ_JOINT: raw_model.njnt,
        mujoco.mjtObj.mjOBJ_GEOM: raw_model.ngeom,
        mujoco.mjtObj.mjOBJ_MATERIAL: raw_model.nmat,
        mujoco.mjtObj.mjOBJ_CAMERA: raw_model.ncam,
    }
    return [mujoco.mj_id2name(raw_model, object_type, index) for index in range(counts[object_type])]


def _exact_name_id(names: Sequence[str | None], target: str, kind: str) -> int:
    matches = [index for index, name in enumerate(names) if name == target]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {kind} named {target!r}, found {matches}")
    return matches[0]


def _address_slice(address: int | tuple[int, int]) -> slice:
    if isinstance(address, (int, np.integer)):
        return slice(int(address), int(address) + 1)
    start, end = address
    return slice(int(start), int(end))


def joint_slices(model) -> dict[str, tuple[slice, slice]]:
    import mujoco

    names = _mujoco_names(model, mujoco.mjtObj.mjOBJ_JOINT)
    result: dict[str, tuple[slice, slice]] = {}
    for name in names:
        if name is None:
            continue
        if name in result:
            raise RuntimeError(f"Duplicate compiled joint name: {name}")
        result[name] = (
            _address_slice(model.get_joint_qpos_addr(name)),
            _address_slice(model.get_joint_qvel_addr(name)),
        )
    return result


def camera_record(model, camera_name: str = "agentview") -> dict[str, object]:
    import mujoco

    names = _mujoco_names(model, mujoco.mjtObj.mjOBJ_CAMERA)
    camera_id = _exact_name_id(names, camera_name, "camera")
    raw_model = getattr(model, "_model", model)
    return {
        "name": camera_name,
        "id": camera_id,
        "position_xyz_m": np.asarray(raw_model.cam_pos[camera_id], dtype=np.float64).tolist(),
        "quaternion_wxyz": np.asarray(raw_model.cam_quat[camera_id], dtype=np.float64).tolist(),
        "fovy_deg": float(raw_model.cam_fovy[camera_id]),
    }


def _render_rgb(env) -> np.ndarray:
    return np.asarray(
        env.sim.render(
            width=RESOLUTION,
            height=RESOLUTION,
            camera_name="agentview",
            segmentation=False,
        )
    ).copy()


def _geom_ids(model, prefix: str) -> set[int]:
    import mujoco

    names = _mujoco_names(model, mujoco.mjtObj.mjOBJ_GEOM)
    return {index for index, name in enumerate(names) if name and name.startswith(prefix)}


def _mask_record(mask: np.ndarray) -> dict[str, object]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return {"pixels": 0, "bbox_xyxy": None, "touches_border": False, "center_xy": None}
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    touches_border = bool(
        bbox[0] == 0
        or bbox[1] == 0
        or bbox[2] == mask.shape[1] - 1
        or bbox[3] == mask.shape[0] - 1
    )
    return {
        "pixels": int(mask.sum()),
        "bbox_xyxy": bbox,
        "touches_border": touches_border,
        "center_xy": [int(round(xs.mean())), int(round(ys.mean()))],
    }


def _contact_records(env, clone_geom_ids: set[int]) -> tuple[list[dict], list[dict]]:
    import mujoco

    names = _mujoco_names(env.sim.model, mujoco.mjtObj.mjOBJ_GEOM)
    all_records: list[dict] = []
    forbidden: list[dict] = []
    for index in range(int(env.sim.data.ncon)):
        contact = env.sim.data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 not in clone_geom_ids and geom2 not in clone_geom_ids:
            continue
        other_id = geom2 if geom1 in clone_geom_ids else geom1
        if other_id in clone_geom_ids:
            continue
        other_name = names[other_id] or f"geom_{other_id}"
        record = {
            "contact_index": index,
            "geom1_id": geom1,
            "geom1_name": names[geom1],
            "geom2_id": geom2,
            "geom2_name": names[geom2],
            "distance_m": float(contact.dist),
            "other_geom_name": other_name,
        }
        all_records.append(record)
        if "table" not in other_name.lower() and "floor" not in other_name.lower():
            forbidden.append(record)
    return all_records, forbidden


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"Failed to write image: {path}")


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise RuntimeError(f"Failed to write mask: {path}")


def _labelled(image_rgb: np.ndarray, label: str) -> np.ndarray:
    panel = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        panel,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def _material_record_and_apply(model, target_name: str, rgba: np.ndarray) -> dict[str, object]:
    import mujoco

    names = _mujoco_names(model, mujoco.mjtObj.mjOBJ_MATERIAL)
    material_id = _exact_name_id(names, target_name, "material")
    before = np.asarray(model.mat_rgba[material_id]).copy()
    model.mat_rgba[material_id] = rgba
    return {
        "id": material_id,
        "name": target_name,
        "rgba_before": before.tolist(),
        "rgba_after": np.asarray(model.mat_rgba[material_id]).tolist(),
    }


def _make_env(xml_text: str, *, seed: int):
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(
        bddl_file_name=str(SOURCE_BDDL),
        camera_names=["agentview"],
        camera_heights=RESOLUTION,
        camera_widths=RESOLUTION,
        camera_depths=False,
    )
    env.seed(seed)
    env.reset()
    env.reset_from_xml_string(xml_text)
    env.sim.reset()
    return env


def select_recorded_state(states: np.ndarray, frame_index: int) -> np.ndarray:
    if states.ndim != 2:
        raise ValueError(f"Expected states [T,D], got {states.shape}")
    if frame_index < 0 or frame_index >= states.shape[0]:
        raise IndexError(f"Frame {frame_index} exceeds trajectory length {states.shape[0]}")
    return np.asarray(states[frame_index]).copy()


def _load_recorded_input(frame_index: int) -> tuple[str, np.ndarray, int]:
    with h5py.File(HDF5_PATH, "r") as handle:
        demo = handle["data"][DEMO_NAME]
        model_xml = demo.attrs["model_file"]
        if isinstance(model_xml, bytes):
            model_xml = model_xml.decode("utf-8")
        if not isinstance(model_xml, str):
            raise TypeError(f"Expected model_file XML string, got {type(model_xml)}")
        states = np.asarray(demo["states"])
        return model_xml, select_recorded_state(states, frame_index), int(states.shape[0])


def run_scene(
    output_dir: Path,
    *,
    seed: int = 42,
    frame_index: int = DEFAULT_FRAME_INDEX,
) -> dict[str, object]:
    from libero.libero.utils import utils as libero_utils
    from tools.render_libero_material_inputs import renderer_runtime_contract

    output_dir = output_dir.resolve()
    if (output_dir / "summary.json").exists():
        raise RuntimeError(f"Refusing to overwrite completed output: {output_dir}")
    renderer_runtime = renderer_runtime_contract(
        require_canonical=True,
        source_camera_resolution=RESOLUTION,
        model_resolution=RESOLUTION,
        resize_interpolation="none",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    hdf5_hash_before = sha256_file(HDF5_PATH)
    source_xml, recorded_state, trajectory_rows = _load_recorded_input(frame_index)
    source_xml_hash = sha256_bytes(source_xml.encode("utf-8"))
    processed_xml = libero_utils.postprocess_model_xml(source_xml, {})
    processed_xml = rewrite_legacy_asset_paths(processed_xml)
    processed_xml_hash = sha256_bytes(processed_xml.encode("utf-8"))
    dual_xml = clone_moka_in_xml(processed_xml)
    dual_xml_hash = sha256_bytes(dual_xml.encode("utf-8"))
    asset_manifest = asset_manifest_from_xml(dual_xml)
    asset_manifest_hash = sha256_bytes(
        json.dumps(asset_manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    (output_dir / "scene.xml").write_text(dual_xml, encoding="utf-8")
    (output_dir / "asset_manifest.json").write_text(
        json.dumps(asset_manifest, indent=2) + "\n", encoding="utf-8"
    )

    reference_env = None
    dual_env = None
    try:
        reference_env = _make_env(processed_xml, seed=seed)
        reference_env.set_init_state(recorded_state)
        reference_state = reference_env.sim.get_state()
        reference_slices = joint_slices(reference_env.sim.model)
        reference_camera = camera_record(reference_env.sim.model)
        reference_material = _material_record_and_apply(
            reference_env.sim.model, SOURCE_BODY_MATERIAL, RED_RGBA
        )
        reference_env.sim.forward()
        reference_raw = _render_rgb(reference_env)
        reference_rgb = transform_agent_view(reference_raw)
        reference_red_mask = red_color_mask(reference_rgb)
        reference_red_record = _mask_record(reference_red_mask)
        protected_centers = dict(PROTECTED_CENTERS_XY)

        reference_qpos = np.asarray(reference_state.qpos).copy()
        reference_qvel = np.asarray(reference_state.qvel).copy()
        reference_time = float(reference_state.time)
        reference_flat_delta = float(
            np.abs(np.asarray(reference_env.get_sim_state()) - recorded_state).max(initial=0.0)
        )

        dual_env = _make_env(dual_xml, seed=seed)
        dual_camera = camera_record(dual_env.sim.model)
        dual_slices = joint_slices(dual_env.sim.model)
        copied_joints = copy_shared_joint_state(
            reference_qpos,
            reference_qvel,
            reference_slices,
            dual_env.sim.data.qpos,
            dual_env.sim.data.qvel,
            dual_slices,
            excluded={CLONE_JOINT},
        )
        dual_env.sim.data.time = reference_time
        dual_env.sim.data.qvel[dual_slices[CLONE_JOINT][1]] = 0.0
        red_material = _material_record_and_apply(dual_env.sim.model, SOURCE_BODY_MATERIAL, RED_RGBA)
        silver_material = _material_record_and_apply(
            dual_env.sim.model,
            CLONE_BODY_MATERIAL,
            np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )

        red_joint_slice = dual_slices[f"{SOURCE_INSTANCE}_joint0"][0]
        red_position = np.asarray(dual_env.sim.data.qpos[red_joint_slice][:3]).copy()
        clone_qpos_slice, clone_qvel_slice = dual_slices[CLONE_JOINT]
        if dual_env.sim.data.qpos[clone_qpos_slice].shape != (7,):
            raise RuntimeError(f"Expected a free joint for {CLONE_JOINT}")
        clone_geom_ids = _geom_ids(dual_env.sim.model, CLONE_INSTANCE)
        red_geom_ids = _geom_ids(dual_env.sim.model, SOURCE_INSTANCE)
        if not clone_geom_ids or not red_geom_ids:
            raise RuntimeError("Compiled dual model is missing moka-pot geoms")

        candidate_records: list[dict[str, object]] = []
        candidate_images: dict[str, np.ndarray] = {}
        for index, (x, y) in enumerate(CANDIDATE_XY):
            candidate_id = f"candidate_{index:02d}"
            dual_env.sim.data.qpos[clone_qpos_slice] = np.asarray(
                [x, y, 0.97, 1.0, 0.0, 0.0, 0.0], dtype=np.float64
            )
            dual_env.sim.data.qvel[clone_qvel_slice] = 0.0
            dual_env.sim.forward()
            raw_rgb = _render_rgb(dual_env)
            rgb = transform_agent_view(raw_rgb)
            red_mask = red_color_mask(rgb)
            silver_mask = difference_visibility_mask(reference_rgb, rgb)
            red_record = _mask_record(red_mask)
            silver_record = _mask_record(silver_mask)
            overlap = (
                1.0
                if red_record["bbox_xyxy"] is None or silver_record["bbox_xyxy"] is None
                else bbox_overlap_fraction(red_record["bbox_xyxy"], silver_record["bbox_xyxy"])
            )
            all_contacts, forbidden_contacts = _contact_records(dual_env, clone_geom_ids)
            clearance = float(np.linalg.norm(np.asarray([x, y]) - red_position[:2]) - POT_RADIUS_SUM_M)
            covers_protected = any(
                bool(silver_mask[center[1], center[0]]) for center in protected_centers.values()
            )
            record = {
                "candidate_id": candidate_id,
                "position_xyz_m": [float(x), float(y), 0.97],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                "clearance_m": clearance,
                "red_pixels": red_record["pixels"],
                "silver_pixels": silver_record["pixels"],
                "red_bbox_xyxy": red_record["bbox_xyxy"],
                "silver_bbox_xyxy": silver_record["bbox_xyxy"],
                "red_border": red_record["touches_border"],
                "silver_border": silver_record["touches_border"],
                "bbox_overlap_fraction": overlap,
                "covers_protected_center": covers_protected,
                "all_contacts": all_contacts,
                "forbidden_contacts": forbidden_contacts,
            }
            record["failed_constraints"] = candidate_failures(record)
            record["passes_hard_constraints"] = _candidate_passes(record)
            candidate_records.append(record)
            candidate_images[candidate_id] = rgb
            _save_rgb(output_dir / "candidates" / f"{candidate_id}.png", rgb)

        (output_dir / "candidate_diagnostics.json").write_text(
            json.dumps(candidate_records, indent=2) + "\n", encoding="utf-8"
        )
        selected = select_candidate(candidate_records)
        selected_id = str(selected["candidate_id"])
        selected_position = np.asarray(selected["position_xyz_m"], dtype=np.float64)
        dual_env.sim.data.qpos[clone_qpos_slice] = np.concatenate(
            [selected_position, np.asarray([1.0, 0.0, 0.0, 0.0])]
        )
        dual_env.sim.data.qvel[clone_qvel_slice] = 0.0
        dual_env.sim.forward()
        selected_raw = _render_rgb(dual_env)
        dual_rgb = transform_agent_view(selected_raw)
        selected_red_mask = red_color_mask(dual_rgb)
        selected_silver_mask = difference_visibility_mask(reference_rgb, dual_rgb)
        selected_dual_state = np.asarray(dual_env.get_sim_state()).copy()
        selected_clone_qpos = np.asarray(dual_env.sim.data.qpos[clone_qpos_slice]).copy()
        selected_clone_qvel = np.asarray(dual_env.sim.data.qvel[clone_qvel_slice]).copy()

        shared_qpos_deltas: dict[str, float] = {}
        shared_qvel_deltas: dict[str, float] = {}
        for name in copied_joints:
            reference_qpos_slice, reference_qvel_slice = reference_slices[name]
            dual_qpos_slice, dual_qvel_slice = dual_slices[name]
            shared_qpos_deltas[name] = float(
                np.abs(
                    reference_qpos[reference_qpos_slice]
                    - np.asarray(dual_env.sim.data.qpos[dual_qpos_slice])
                ).max(initial=0.0)
            )
            shared_qvel_deltas[name] = float(
                np.abs(
                    reference_qvel[reference_qvel_slice]
                    - np.asarray(dual_env.sim.data.qvel[dual_qvel_slice])
                ).max(initial=0.0)
            )

        _save_rgb(output_dir / "reference_start.png", reference_rgb)
        _save_rgb(output_dir / "dual_moka_start.png", dual_rgb)
        _save_rgb(output_dir / "raw_agentview.png", selected_raw)
        comparison = np.concatenate(
            [
                _labelled(reference_rgb, f"reference: demo_42 frame {frame_index}"),
                _labelled(dual_rgb, "dual moka: red + silver"),
            ],
            axis=1,
        )
        comparison_path = output_dir / "scene_comparison.png"
        if not cv2.imwrite(str(comparison_path), comparison):
            raise RuntimeError(f"Failed to write {comparison_path}")

        import mujoco

        body_names = _mujoco_names(dual_env.sim.model, mujoco.mjtObj.mjOBJ_BODY)
        joint_names = _mujoco_names(dual_env.sim.model, mujoco.mjtObj.mjOBJ_JOINT)
        material_names = _mujoco_names(dual_env.sim.model, mujoco.mjtObj.mjOBJ_MATERIAL)
        red_body_id = _exact_name_id(body_names, f"{SOURCE_INSTANCE}_main", "body")
        silver_body_id = _exact_name_id(body_names, f"{CLONE_INSTANCE}_main", "body")
        selected_red_record = _mask_record(selected_red_mask)
        selected_silver_record = _mask_record(selected_silver_mask)
        _save_mask(output_dir / "masks/reference_red.png", reference_red_mask)
        _save_mask(output_dir / "masks/dual_red.png", selected_red_mask)
        _save_mask(output_dir / "masks/dual_silver.png", selected_silver_mask)

        reference_shared_qpos = np.concatenate(
            [reference_qpos[reference_slices[name][0]] for name in copied_joints]
        )
        reference_shared_qvel = np.concatenate(
            [reference_qvel[reference_slices[name][1]] for name in copied_joints]
        )
        dual_shared_qpos = np.concatenate(
            [np.asarray(dual_env.sim.data.qpos[dual_slices[name][0]]) for name in copied_joints]
        )
        dual_shared_qvel = np.concatenate(
            [np.asarray(dual_env.sim.data.qvel[dual_slices[name][1]]) for name in copied_joints]
        )
        np.savez_compressed(
            output_dir / "state_evidence.npz",
            recorded_state=recorded_state,
            restored_reference_state=np.asarray(reference_env.get_sim_state()).copy(),
            reference_shared_qpos=reference_shared_qpos,
            dual_shared_qpos=dual_shared_qpos,
            reference_shared_qvel=reference_shared_qvel,
            dual_shared_qvel=dual_shared_qvel,
            selected_dual_state=selected_dual_state,
            selected_clone_qpos=selected_clone_qpos,
            selected_clone_qvel=selected_clone_qvel,
        )
        camera_equal = reference_camera == dual_camera
        max_qpos_delta = max(shared_qpos_deltas.values(), default=0.0)
        max_qvel_delta = max(shared_qvel_deltas.values(), default=0.0)
        hdf5_hash_after = sha256_file(HDF5_PATH)
        checks = {
            "reference_hdf5_state_restored_exactly": reference_flat_delta == 0.0,
            "camera_parameters_identical": camera_equal,
            "all_shared_joints_copied": set(copied_joints) == set(reference_slices),
            "shared_qpos_identical": max_qpos_delta == 0.0,
            "shared_qvel_identical": max_qvel_delta == 0.0,
            "two_distinct_moka_bodies": red_body_id != silver_body_id,
            "two_distinct_free_joints": (
                f"{SOURCE_INSTANCE}_joint0" in joint_names and CLONE_JOINT in joint_names
            ),
            "independent_body_materials": (
                SOURCE_BODY_MATERIAL in material_names and CLONE_BODY_MATERIAL in material_names
            ),
            "red_material_applied": bool(np.allclose(red_material["rgba_after"], RED_RGBA)),
            "silver_material_preserved": bool(
                np.allclose(silver_material["rgba_after"], [1.0, 1.0, 1.0, 1.0])
            ),
            "selected_candidate_passes": bool(selected["passes_hard_constraints"]),
            "reference_red_visible": int(reference_red_record["pixels"]) >= MIN_MASK_PIXELS,
            "selected_red_visible": int(selected_red_record["pixels"]) >= MIN_MASK_PIXELS,
            "selected_silver_visible": int(selected_silver_record["pixels"]) >= MIN_MASK_PIXELS,
            "transform_exact": bool(np.array_equal(dual_rgb, selected_raw[::-1, ::-1])),
            "source_hdf5_unchanged": hdf5_hash_before == hdf5_hash_after,
            "canonical_asset_paths": asset_paths_are_canonical(asset_manifest),
            "selected_dual_state_is_finite": bool(np.isfinite(selected_dual_state).all()),
            "selected_dual_state_extends_source": selected_dual_state.size > recorded_state.size,
            "selected_clone_qpos_matches_candidate": bool(
                np.array_equal(
                    selected_clone_qpos,
                    np.concatenate(
                        [selected_position, np.asarray([1.0, 0.0, 0.0, 0.0])]
                    ),
                )
            ),
            "selected_clone_qvel_is_zero": bool(np.array_equal(selected_clone_qvel, np.zeros(6))),
        }
        outputs = {
            "reference_start": "reference_start.png",
            "dual_moka_start": "dual_moka_start.png",
            "raw_agentview": "raw_agentview.png",
            "comparison": "scene_comparison.png",
            "scene_xml": "scene.xml",
            "asset_manifest": "asset_manifest.json",
            "candidate_diagnostics": "candidate_diagnostics.json",
            "state_evidence": "state_evidence.npz",
            "masks": {
                "reference_red": "masks/reference_red.png",
                "dual_red": "masks/dual_red.png",
                "dual_silver": "masks/dual_silver.png",
            },
        }
        lineage_artifacts = (
            outputs["reference_start"],
            outputs["dual_moka_start"],
            outputs["raw_agentview"],
            outputs["comparison"],
            outputs["scene_xml"],
            outputs["asset_manifest"],
            outputs["candidate_diagnostics"],
            outputs["state_evidence"],
            *outputs["masks"].values(),
        )
        artifact_sha256 = {
            relative: sha256_file(output_dir / relative)
            for relative in lineage_artifacts
        }
        summary = {
            "classification": "partial rollout",
            "scope": {
                "suite": "libero_90",
                "task_id": 19,
                "task_name": SOURCE_BDDL.stem,
                "instruction": "put the moka pot on the stove",
                "hdf5": str(HDF5_PATH),
                "demo": DEMO_NAME,
                "frame": frame_index,
                "transform": "rgb_vhflip",
                "resolution": RESOLUTION,
                "seed": seed,
                "env_reset_calls": 2,
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
            "sources": {
                "libero_source": str(LIBERO_SOURCE),
                "source_bddl": str(SOURCE_BDDL),
                "source_bddl_sha256": sha256_file(SOURCE_BDDL),
                "hdf5_sha256_before": hdf5_hash_before,
                "hdf5_sha256_after": hdf5_hash_after,
                "source_model_xml_sha256": source_xml_hash,
                "processed_model_xml_sha256": processed_xml_hash,
                "dual_model_xml_sha256": dual_xml_hash,
                "asset_manifest_sha256": asset_manifest_hash,
                "asset_count": len(asset_manifest),
                "trajectory_rows": trajectory_rows,
                "recorded_state_length": int(recorded_state.size),
            },
            "camera": {"reference": reference_camera, "dual": dual_camera},
            "state_mapping": {
                "copied_joint_count": len(copied_joints),
                "copied_joints": copied_joints,
                "reference_flat_state_max_abs_delta": reference_flat_delta,
                "shared_qpos_max_abs_delta": max_qpos_delta,
                "shared_qvel_max_abs_delta": max_qvel_delta,
                "shared_qpos_deltas": shared_qpos_deltas,
                "shared_qvel_deltas": shared_qvel_deltas,
                "selected_dual_state_sha256": sha256_array(selected_dual_state),
                "selected_dual_state_shape": list(selected_dual_state.shape),
                "selected_dual_state_dtype": selected_dual_state.dtype.str,
                "selected_clone_qpos": selected_clone_qpos.tolist(),
                "selected_clone_qvel": selected_clone_qvel.tolist(),
            },
            "materials": {
                "reference_red": reference_material,
                "dual_red": red_material,
                "dual_silver": silver_material,
            },
            "scene": {
                "visibility_method": (
                    "red material color mask + aligned dual-minus-reference RGB difference "
                    "(threshold >8/255)"
                ),
                "red_body_position_xyz_m": np.asarray(
                    dual_env.sim.data.body_xpos[red_body_id], dtype=np.float64
                ).tolist(),
                "silver_body_position_xyz_m": np.asarray(
                    dual_env.sim.data.body_xpos[silver_body_id], dtype=np.float64
                ).tolist(),
                "protected_centers_xy": protected_centers,
                "reference_red_mask": reference_red_record,
                "selected_red_mask": selected_red_record,
                "selected_silver_mask": selected_silver_record,
                "selected_candidate": selected,
                "candidate_count": len(candidate_records),
            },
            "outputs": outputs,
            "artifact_sha256": artifact_sha256,
            "checks": checks,
            "all_checks_passed": all(checks.values()),
        }
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        if not summary["all_checks_passed"]:
            failed = [name for name, passed in checks.items() if not passed]
            raise RuntimeError(f"Aligned dual-moka checks failed: {failed}")
        return summary
    finally:
        if dual_env is not None:
            dual_env.close()
        if reference_env is not None:
            reference_env.close()


def run_state_payload(
    *,
    status: str,
    seed: int,
    frame_index: int,
    error: BaseException | None = None,
) -> dict[str, object]:
    if status not in {"running", "completed", "failed"}:
        raise ValueError(f"Unsupported run status: {status}")
    return {
        "classification": "partial rollout",
        "scope": {
            "description": "state-only aligned dual-moka MuJoCo render",
            "env_step_calls": 0,
            "robot_actions_executed": 0,
            "odeworld_inference_run": False,
            "success_rate_measured": False,
        },
        "status": status,
        "seed": seed,
        "frame_index": frame_index,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "error": (
            {"type": type(error).__name__, "message": str(error)}
            if error is not None
            else None
        ),
    }


def _write_run_state(
    output_dir: Path,
    *,
    status: str,
    seed: int,
    frame_index: int,
    error: BaseException | None = None,
) -> None:
    if status == "running" and (output_dir / "summary.json").exists():
        raise FileExistsError(
            f"Refusing to modify completed output: {output_dir / 'summary.json'}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "run_state.json"
    if status == "running" and path.exists():
        raise FileExistsError(f"Refusing to overwrite existing run state: {path}")
    path.write_text(
        json.dumps(
            run_state_payload(
                status=status,
                seed=seed,
                frame_index=frame_index,
                error=error,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-index", type=int, default=DEFAULT_FRAME_INDEX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "classification=partial rollout env_step_calls=0 robot_actions_executed=0 "
        "success_rate_measured=false odeworld_inference_run=false",
        flush=True,
    )
    output_dir = args.output_dir.resolve()
    _write_run_state(
        output_dir,
        status="running",
        seed=args.seed,
        frame_index=args.frame_index,
    )
    try:
        summary = run_scene(output_dir, seed=args.seed, frame_index=args.frame_index)
        _write_run_state(
            output_dir,
            status="completed",
            seed=args.seed,
            frame_index=args.frame_index,
        )
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        _write_run_state(
            output_dir,
            status="failed",
            seed=args.seed,
            frame_index=args.frame_index,
            error=error,
        )
        raise


if __name__ == "__main__":
    main()
