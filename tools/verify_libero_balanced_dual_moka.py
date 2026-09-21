"""Independently verify balanced dual-moka scene and reachability artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


CLASSIFICATION = "partial rollout"
MIN_DISTANCE_M = 0.22
MAX_DISTANCE_M = 0.28
MAX_ROW_DELTA_PX = 20.0
MAX_SETTLING_DRIFT_M = 0.02
MAX_FINAL_ERROR_M = 0.05
MAX_OBJECT_DISPLACEMENT_M = 0.01
MIN_VISIBLE_PIXELS = 600


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _layout_contract_failures(
    layout: Mapping[str, Any], *, require_white_left: bool = True
) -> list[str]:
    failures = []
    if not MIN_DISTANCE_M <= float(layout.get("center_distance_m", math.inf)) <= MAX_DISTANCE_M:
        failures.append("distance")
    if float(layout.get("bbox_overlap_fraction", math.inf)) != 0.0:
        failures.append("overlap")
    if float(layout.get("row_delta_px", math.inf)) > MAX_ROW_DELTA_PX:
        failures.append("row")
    white = layout.get("white_center_xy")
    red = layout.get("red_center_xy")
    if white is None or red is None:
        failures.append("ordering")
    elif require_white_left and float(white[0]) >= float(red[0]):
        failures.append("ordering")
    if int(layout.get("white_pixels", 0)) < MIN_VISIBLE_PIXELS:
        failures.append("white_visibility")
    if int(layout.get("red_pixels", 0)) < MIN_VISIBLE_PIXELS:
        failures.append("red_visibility")
    if bool(layout.get("white_border", True)) or bool(layout.get("red_border", True)):
        failures.append("border")
    if layout.get("forbidden_contacts"):
        failures.append("contacts")
    if not bool(layout.get("passes_hard_constraints", False)):
        failures.append("generator_gate")
    return failures


def _reach_contract_failures(reach: Mapping[str, Any]) -> list[str]:
    failures = []
    if reach.get("classification") != CLASSIFICATION:
        failures.append("classification")
    if reach.get("controller") != "scripted_osc_hover":
        failures.append("controller")
    actions = int(reach.get("action_count", 0))
    if actions <= 0 or int(reach.get("frame_count", -1)) != actions + 1:
        failures.append("counts")
    if float(reach.get("final_error_m", math.inf)) >= MAX_FINAL_ERROR_M:
        failures.append("final_error")
    if float(reach.get("max_object_displacement_m", math.inf)) > MAX_OBJECT_DISPLACEMENT_M:
        failures.append("displacement")
    if int(reach.get("pot_contact_steps", -1)) != 0:
        failures.append("contacts")
    if not bool(reach.get("all_checks_passed", False)):
        failures.append("generator_gate")
    return failures


def verify_summary_contract(summary: Mapping[str, Any]) -> list[str]:
    """Check the semantic claims without trusting the generator's final boolean."""
    failures = []
    if summary.get("classification") != CLASSIFICATION:
        failures.append("classification")
    scope = summary.get("scope", {})
    if (
        bool(scope.get("fastwam_inference_run", True))
        or bool(scope.get("success_rate_measured", True))
        or bool(scope.get("full_benchmark", True))
        or int(scope.get("reach_targets", 0)) != 2
    ):
        failures.append("scope")
    layout = summary.get("settled_layout", {})
    require_white_left = summary.get("scope", {}).get("orientation", "A") == "A"
    if _layout_contract_failures(layout, require_white_left=require_white_left):
        failures.append("layout")
    drift = layout.get("settling_drift_m", {})
    if len(drift) != 2 or max((float(value) for value in drift.values()), default=math.inf) > MAX_SETTLING_DRIFT_M:
        failures.append("settling_drift")
    reaches = summary.get("reachability", {})
    for label in ("white", "red"):
        if _reach_contract_failures(reaches.get(label, {})):
            failures.append(f"{label}_reachability")
    if not bool(summary.get("all_checks_passed", False)):
        failures.append("generator_all_checks")
    return failures


def _decode_image(path: Path) -> Any:
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or min(image.shape[:2]) <= 0:
        raise RuntimeError(f"Could not decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _decode_video_count(path: Path) -> tuple[int, tuple[int, int], float]:
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(path))
    count = 0
    shape = None
    variances = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        count += 1
        shape = frame.shape[:2]
        variances.append(float(np.var(frame)))
    capture.release()
    if count == 0 or shape is None:
        raise RuntimeError(f"Could not decode video frames: {path}")
    return count, (int(shape[0]), int(shape[1])), min(variances)


def _verify_image_appearance(image: Any, layout: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    red_values = image.astype(np.int16)
    red_mask = (
        (red_values[:, :, 0] >= 120)
        & (red_values[:, :, 0] >= red_values[:, :, 1] + 45)
        & (red_values[:, :, 0] >= red_values[:, :, 2] + 45)
    )
    red_bbox = layout["red_bbox_xyxy"]
    white_bbox = layout["white_bbox_xyxy"]

    def crop(mask_or_image: Any, bbox: list[int]) -> Any:
        x1, y1, x2, y2 = bbox
        return mask_or_image[y1 : y2 + 1, x1 : x2 + 1]

    red_fraction = float(crop(red_mask, red_bbox).mean())
    white_crop = crop(image, white_bbox).astype(np.int16)
    white_neutral = (
        (white_crop.min(axis=2) >= 90)
        & ((white_crop.max(axis=2) - white_crop.min(axis=2)) <= 65)
    )
    white_red_fraction = float(crop(red_mask, white_bbox).mean())
    record = {
        "red_pixels_full_image": int(red_mask.sum()),
        "red_fraction_in_declared_red_bbox": red_fraction,
        "neutral_bright_fraction_in_declared_white_bbox": float(white_neutral.mean()),
        "red_fraction_in_declared_white_bbox": white_red_fraction,
    }
    record["passed"] = bool(
        record["red_pixels_full_image"] >= MIN_VISIBLE_PIXELS
        and red_fraction >= 0.05
        and record["neutral_bright_fraction_in_declared_white_bbox"] >= 0.05
        and white_red_fraction < red_fraction
    )
    return record


def verify(output_dir: Path) -> dict[str, Any]:
    import numpy as np

    output_dir = output_dir.resolve()
    summary_path = output_dir / "layout_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    checks["summary_contract"] = not verify_summary_contract(summary)

    for name in ("agentview.png", "wristview.png", "scene_comparison.png"):
        image = _decode_image(output_dir / name)
        checks[f"decode_{name}"] = bool(image.size and float(np.var(image)) > 1.0)
    agentview = _decode_image(output_dir / "agentview.png")
    appearance = _verify_image_appearance(agentview, summary["settled_layout"])
    details["appearance"] = appearance
    checks["red_white_appearance"] = bool(appearance["passed"])

    claimed_hashes = summary.get("sha256", {})
    hash_results = {}
    for name, expected in claimed_hashes.items():
        path = output_dir / name
        actual = sha256_file(path) if path.is_file() else None
        hash_results[name] = {"expected": expected, "actual": actual, "matches": actual == expected}
    details["hashes"] = hash_results
    checks["all_claimed_hashes_match"] = bool(hash_results and all(item["matches"] for item in hash_results.values()))

    for label in ("white", "red"):
        record = json.loads((output_dir / f"reachability_{label}.json").read_text(encoding="utf-8"))
        npz = np.load(output_dir / f"reachability_{label}.npz")
        target = np.asarray(npz["target_xyz_m"], dtype=np.float64)
        eef = np.asarray(npz["eef_path_xyz_m"], dtype=np.float64)
        actions = np.asarray(npz["actions"])
        white = np.asarray(npz["white_path_xyz_m"], dtype=np.float64)
        red = np.asarray(npz["red_path_xyz_m"], dtype=np.float64)
        recomputed_error = float(np.linalg.norm(eef[-1] - target))
        recomputed_displacement = max(
            float(np.linalg.norm(white - white[0], axis=1).max(initial=0.0)),
            float(np.linalg.norm(red - red[0], axis=1).max(initial=0.0)),
        )
        video_count, video_shape, minimum_variance = _decode_video_count(
            output_dir / f"reachability_{label}.mp4"
        )
        target_details = {
            "json_contract_failures": _reach_contract_failures(record),
            "npz_action_count": int(actions.shape[0]),
            "npz_path_count": int(eef.shape[0]),
            "recomputed_final_error_m": recomputed_error,
            "recomputed_max_object_displacement_m": recomputed_displacement,
            "video_frame_count": video_count,
            "video_shape_hw": list(video_shape),
            "minimum_video_frame_variance": minimum_variance,
        }
        details[f"reachability_{label}"] = target_details
        checks[f"reachability_{label}"] = bool(
            not target_details["json_contract_failures"]
            and actions.ndim == 2
            and actions.shape[1] == 7
            and eef.shape == (actions.shape[0] + 1, 3)
            and white.shape == eef.shape
            and red.shape == eef.shape
            and abs(recomputed_error - float(record["final_error_m"])) < 1e-9
            and abs(recomputed_displacement - float(record["max_object_displacement_m"])) < 1e-9
            and recomputed_error < MAX_FINAL_ERROR_M
            and recomputed_displacement <= MAX_OBJECT_DISPLACEMENT_M
            and video_count == int(record["frame_count"])
            and minimum_variance > 1.0
        )

    verification = {
        "classification": CLASSIFICATION,
        "source_summary": str(summary_path),
        "checks": checks,
        "details": details,
        "all_checks_passed": all(checks.values()),
    }
    verification_path = output_dir / "verification_summary.json"
    verification_path.write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    if not verification["all_checks_passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Independent verification failed: {failed}")
    return verification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(verify(args.output_dir), indent=2), flush=True)


if __name__ == "__main__":
    main()
