#!/usr/bin/env python3
"""Run the five-pair ODEWorld B3 color pilot.

Classification: mock test.  Inputs are verified MuJoCo frame-0 images, but
this producer performs only offline ODEWorld prediction and PT-Flow rollout:
it creates no simulator, calls no ``env.step``, and measures no task success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tools.run_libero_language_color_ablation import ColorAblationBackend, build_red_mask
from tools.run_libero_framewise_odeworld import (
    GOAL_PREDICTOR_CHECKPOINT,
    PT_FLOW_CHECKPOINT,
    RAE_CHECKPOINT,
    load_models,
)
from tools.run_libero_dual_moka_prompt_future import PROMPTS, CONDITIONS, _checkpoint_provenance
from tools.verify_libero_balanced_dual_moka_multidemo import verify_multidemo


ROOT = Path(__file__).resolve().parents[1]
CLASSIFICATION = "mock test"
LAYOUTS = ("layout_a", "layout_b")
DEFAULT_STEPS = 200
DEFAULT_HORIZON = 1.0
DEFAULT_SEEDS = (42, 43, 44)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"{label} must be a regular file: {path}")
    return path


def load_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(regular_file(path, "input image")), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode input image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.shape[:2] == (512, 512):
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    if image.shape != (256, 256, 3) or image.dtype != np.uint8:
        raise RuntimeError(f"Unexpected input image shape/dtype: {path}")
    return image


def build_masks(images: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Return physical red/silver masks separately for each counterbalanced layout."""
    red_masks = {layout: build_red_mask(images[layout]) for layout in LAYOUTS}
    if any(mask is None for mask in red_masks.values()):
        raise RuntimeError("Could not derive red object masks for both layouts")
    red_a = np.asarray(red_masks["layout_a"], dtype=bool)
    red_b = np.asarray(red_masks["layout_b"], dtype=bool)
    if min(int(red_a.sum()), int(red_b.sum())) < 500 or np.any(red_a & red_b):
        raise RuntimeError("Invalid counterbalanced red object masks")
    masks = {
        "layout_a": {"red": red_a, "silver": red_b},
        "layout_b": {"red": red_b, "silver": red_a},
    }
    return masks


def expected_cells(pair_count: int, seeds: tuple[int, ...]) -> list[tuple[int, str, int]]:
    return [(pair, layout, seed) for pair in range(pair_count) for layout in LAYOUTS for seed in seeds]


def _seed_everything(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_pilot(
    output_root: Path,
    input_root: Path,
    source_root: Path,
    *,
    pair_count: int = 5,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    steps: int = DEFAULT_STEPS,
    horizon: float = DEFAULT_HORIZON,
    device: str = "cuda",
) -> dict[str, Any]:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("B3 pilot must run through Slurm")
    if output_root.exists():
        raise FileExistsError(f"Refusing existing pilot root: {output_root}")
    if pair_count <= 0 or steps <= 0 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("pair_count, steps and seeds must be positive and unique")
    input_root = Path(input_root).resolve()
    source_root = Path(source_root).resolve()
    expected_demos = [f"demo_{index}" for index in range(pair_count)]
    input_gate = verify_multidemo(input_root, source_root, expected_demos)
    output_root.mkdir(parents=True)
    images_by_pair: dict[int, dict[str, np.ndarray]] = {}
    masks_by_pair: dict[int, dict[str, np.ndarray]] = {}
    input_records: dict[str, Any] = {}
    for pair in range(pair_count):
        pair_name = f"pair_{pair:02d}"
        images = {layout: load_rgb(input_root / pair_name / layout / "agentview.png") for layout in LAYOUTS}
        masks = build_masks(images)
        images_by_pair[pair] = images
        masks_by_pair[pair] = masks
        input_records[pair_name] = {
            "demo": f"demo_{pair}",
            "layout_a_sha256": sha256_array(images["layout_a"]),
            "layout_b_sha256": sha256_array(images["layout_b"]),
            "layout_a_red_mask_sha256": sha256_array(masks["layout_a"]["red"]),
            "layout_a_silver_mask_sha256": sha256_array(masks["layout_a"]["silver"]),
            "layout_b_red_mask_sha256": sha256_array(masks["layout_b"]["red"]),
            "layout_b_silver_mask_sha256": sha256_array(masks["layout_b"]["silver"]),
        }
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("B3 pilot requires a Slurm GPU")
    backend = ColorAblationBackend(load_models(torch.device(device)), decode_chunk_size=16)
    cells: list[dict[str, Any]] = []
    started = time.perf_counter()
    for pair, layout, seed in expected_cells(pair_count, seeds):
        pair_name = f"pair_{pair:02d}"
        cell_dir = output_root / pair_name / layout / f"seed_{seed}"
        cell_dir.mkdir(parents=True)
        image = images_by_pair[pair][layout]
        masks = masks_by_pair[pair][layout]
        goals: dict[str, np.ndarray] = {}
        rollouts: dict[str, np.ndarray] = {}
        for condition in CONDITIONS:
            _seed_everything(seed)
            latent, goal_rgb = backend.predict_goal(image, PROMPTS[condition])
            frames = np.stack(backend.rollout_from_goal_latent(image, latent, horizon=horizon, steps=steps))
            if frames.shape != (steps, 256, 256, 3) or frames.dtype != np.uint8:
                raise RuntimeError(f"Invalid rollout shape/dtype: {pair_name}/{layout}/seed_{seed}/{condition}")
            goals[f"{condition}_latent"] = np.asarray(latent, dtype=np.float32)
            goals[f"{condition}_rgb"] = np.asarray(goal_rgb, dtype=np.uint8)
            rollouts[condition] = frames
        arrays_path = cell_dir / "arrays.npz"
        np.savez_compressed(
            arrays_path,
            input=image,
            red_mask=masks["red"],
            silver_mask=masks["silver"],
            **goals,
            **{f"{condition}_rollout": value for condition, value in rollouts.items()},
        )
        record = {
            "schema": "odeworld_b3_color_pilot_cell_v1",
            "classification": CLASSIFICATION,
            "pair": pair_name,
            "pair_index": pair,
            "demo": f"demo_{pair}",
            "layout": layout,
            "inference_seed": seed,
            "steps": steps,
            "horizon": horizon,
            "prompts": PROMPTS,
            "input_record": input_records[pair_name],
            "arrays": {"path": "arrays.npz", "sha256": sha256_file(arrays_path)},
        }
        (cell_dir / "cell_manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        cells.append(record)
        print(f"completed pair={pair_name} layout={layout} seed={seed}", flush=True)
    manifest = {
        "schema": "odeworld_b3_color_pilot_v1",
        "classification": CLASSIFICATION,
        "scope": {
            "pairs": pair_count,
            "layouts": 2,
            "prompts": 3,
            "seeds": list(seeds),
            "cells": len(cells),
            "sequences": len(cells) * len(CONDITIONS),
            "steps": steps,
            "env_step_calls": 0,
            "robot_actions_executed": 0,
            "success_rate_measured": False,
            "full_benchmark": False,
        },
        "input_root": str(input_root),
        "source_root": str(source_root),
        "input_gate": input_gate,
        "inputs": input_records,
        "checkpoints": {
            "pt_flow": _checkpoint_provenance(PT_FLOW_CHECKPOINT),
            "rae": _checkpoint_provenance(RAE_CHECKPOINT),
            "goal_predictor": _checkpoint_provenance(GOAL_PREDICTOR_CHECKPOINT),
        },
        "source_files": {
            "producer": sha256_file(Path(__file__).resolve()),
            "counterbalanced_helpers": sha256_file(ROOT / "tools" / "run_odeworld_counterbalanced_color.py"),
        },
        "cells": cells,
        "started_at_host": platform.node(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "elapsed_seconds": time.perf_counter() - started,
    }
    manifest["manifest_sha256"] = sha256_bytes(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    (output_root / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("input_root", type=Path)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--pair-count", type=int, default=5)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--horizon", type=float, default=DEFAULT_HORIZON)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print("classification=mock test env_step_calls=0 robot_actions_executed=0 success_rate_measured=false", flush=True)
    print(json.dumps(run_pilot(args.output_root, args.input_root, args.source_root, pair_count=args.pair_count, seeds=tuple(args.seeds), steps=args.steps, horizon=args.horizon, device=args.device), indent=2), flush=True)


if __name__ == "__main__":
    main()
