#!/usr/bin/env python3
"""Fine-tune PT-Flow with multi-step rollout and stability losses.

Classification: mock test. This trainer reads offline LIBERO HDF5 RGB frames
and frozen ODEWorld checkpoints. It does not create LIBERO/MuJoCo, call
``env.step``, execute robot actions, or measure task success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchdiffeq import odeint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.libero_manifest import resolve_task_file_rows
from tools.run_odeworld_temporal_diagnostic_shard import _image_key, _regular_file

CLASSIFICATION = "mock test"
SCHEMA = "odeworld_ptflow_rollout_stability_train_v1"
CHECKPOINT = PROJECT_ROOT / "assets" / "pretrained" / "ODEWorld-PT-Flow-LIBERO"
SAMPLING_MANIFEST = PROJECT_ROOT / "manifests" / "odeworld_diagnostic_sampling_v1.json"
BALANCED_MANIFEST_SCHEMAS = {
    "odeworld_ptflow_scale_m_manifest_v1": "scale_m",
    "odeworld_ptflow_scale_l_manifest_v1": "scale_l",
}
LOCAL_MANIFEST_SCHEMA = "odeworld_ptflow_local_supervision_manifest_v1"
HELD_OUT_SPLITS = ("b1_pilot", "b1_confirmatory", "b2_confirmatory")
FRAME_TRANSFORM = "rgb_vhflip"
RESOLUTION = 256
SOURCE_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "models" / "DINOv2PTFlow.py",
    PROJECT_ROOT / "tools" / "run_odeworld_temporal_diagnostic_shard.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with _regular_file(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def transform_resize(frame: np.ndarray) -> np.ndarray:
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"Expected uint8 RGB frame, got {frame.shape} {frame.dtype}")
    if FRAME_TRANSFORM == "rgb_vhflip":
        frame = frame[::-1, ::-1]
    return np.ascontiguousarray(cv2.resize(frame, (RESOLUTION, RESOLUTION), interpolation=cv2.INTER_AREA))


def identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["suite"]), str(row["task"]), str(row["demo"])


def _validate_named_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    split_name: str,
    expected_count: int,
    expected_per_task: int,
    expected_tasks: int,
) -> list[dict[str, Any]]:
    result = [dict(row) for row in rows]
    identities = [identity(row) for row in result]
    if len(identities) != len(set(identities)):
        raise RuntimeError(f"Named split contains duplicate identities: {split_name}")
    if len(result) != expected_count:
        raise RuntimeError(f"Named split count mismatch: {split_name}={len(result)} expected={expected_count}")
    task_counts = Counter(key[:2] for key in identities)
    if len(task_counts) != expected_tasks or set(task_counts.values()) != {expected_per_task}:
        raise RuntimeError(
            f"Named split is not task-balanced: {split_name} tasks={len(task_counts)} "
            f"counts={sorted(set(task_counts.values()))}"
        )
    return result


def load_training_rows(
    path: Path,
    *,
    max_train: int,
    max_val: int,
    train_split: str | None = None,
    validation_split: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads(_regular_file(path).read_text(encoding="utf-8"))
    if (train_split is None) != (validation_split is None):
        raise ValueError("train_split and validation_split must be specified together")
    if train_split is not None:
        variant = BALANCED_MANIFEST_SCHEMAS.get(str(manifest.get("schema")))
        counts = manifest.get("counts", {})
        if manifest.get("schema") == LOCAL_MANIFEST_SCHEMA:
            expected_splits = ("train", "validation")
            if (train_split, validation_split) != expected_splits:
                raise RuntimeError("LOCAL supervision requires train and validation splits")
        else:
            if variant is None:
                raise RuntimeError(f"Named training splits require a balanced Scale-M, Scale-L, or local manifest: {path}")
            expected_splits = (f"{variant}_train", f"{variant}_validation")
            if (train_split, validation_split) != expected_splits:
                raise RuntimeError(f"{variant.upper()} training requires {expected_splits[0]} and {expected_splits[1]}")
        expected_tasks = int(counts.get("tasks", 0))
        train_rows = _validate_named_split(
            manifest.get("splits", {}).get(train_split, []),
            split_name=train_split,
            expected_count=int(counts.get("train", -1)),
            expected_per_task=int(counts.get("train_per_task", -1)),
            expected_tasks=expected_tasks,
        )
        val_rows = _validate_named_split(
            manifest.get("splits", {}).get(validation_split, []),
            split_name=validation_split,
            expected_count=int(counts.get("validation", -1)),
            expected_per_task=int(counts.get("validation_per_task", -1)),
            expected_tasks=expected_tasks,
        )
        overlap = {identity(row) for row in train_rows} & {identity(row) for row in val_rows}
        if overlap:
            raise RuntimeError(f"Named train/validation splits overlap: {sorted(overlap)[:3]}")
        return train_rows, val_rows

    if manifest.get("schema") != "odeworld_diagnostic_sampling_v1":
        raise RuntimeError(f"Unexpected sampling manifest schema: {path}")
    held_out = {
        identity(row)
        for split in HELD_OUT_SPLITS
        for row in manifest.get("splits", {}).get(split, [])
    }
    population = [row for row in manifest.get("population", {}).get("rows", []) if identity(row) not in held_out]
    population.sort(key=identity)
    if len(population) < max_train + max_val:
        raise RuntimeError(f"Not enough non-held-out population rows: {len(population)}")
    val_rows = population[:max_val]
    train_rows = population[max_val : max_val + max_train]
    return train_rows, val_rows


class HDF5WindowDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: Sequence[Mapping[str, Any]], window: int) -> None:
        if window < 3 or window % 2 == 0:
            raise ValueError("window must be odd and at least 3")
        self.rows = list(rows)
        self.window = window

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        task_file = _regular_file(Path(str(row["task_file"])))
        demo = str(row["demo"])
        with h5py.File(task_file, "r") as handle:
            data = handle["data"]
            if demo not in data:
                raise KeyError(f"Missing {demo} in {task_file}")
            obs = data[demo]["obs"]
            frames = obs[_image_key(obs)]
            if int(frames.shape[0]) < self.window:
                raise ValueError(f"Demo has fewer frames than window: {task_file}:{demo}")
            if "window_indices" in row:
                indices = np.asarray(row["window_indices"], dtype=np.int64)
                if indices.shape != (self.window,) or np.any(indices < 0) or np.any(indices >= frames.shape[0]):
                    raise ValueError(f"Invalid manifest window_indices for {identity(row)}: {indices.tolist()}")
            else:
                indices = np.rint(np.linspace(0, int(frames.shape[0]) - 1, self.window)).astype(np.int64)
            sampled = np.stack([transform_resize(np.asarray(frames[idx], dtype=np.uint8)) for idx in indices])
        tensor = torch.from_numpy(sampled).permute(0, 3, 1, 2).float().div_(255.0)
        return {"frames": tensor, "identity": identity(row), "task_file": str(task_file)}


def collate_windows(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "frames": torch.stack([item["frames"] for item in batch]),
        "identity": [item["identity"] for item in batch],
        "task_file": [item["task_file"] for item in batch],
    }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def freeze_backbone(flow: nn.Module) -> None:
    backbone = getattr(flow, "backbone", None)
    if backbone is None:
        raise RuntimeError("PT-Flow checkpoint has no backbone")
    backbone.requires_grad_(False)
    backbone.eval()


def resize_delta_tokens(flow: nn.Module, num_delta_tokens: int, *, seed: int) -> dict[str, int]:
    """Expand or shrink the learned dynamic query-token bank deterministically.

    The image patch-token grid is intentionally untouched.  Existing query tokens
    are copied exactly and newly allocated tokens receive a small deterministic
    initialization so they do not remain permutation-symmetric during fine-tuning.
    """
    target = int(num_delta_tokens)
    if target <= 0:
        raise ValueError("num_delta_tokens must be positive")
    current = int(getattr(flow, "num_delta_tokens", 0))
    token = getattr(flow, "delta_token", None)
    if not isinstance(token, nn.Parameter) or token.ndim != 3 or token.shape[0] != 1:
        raise RuntimeError("PT-Flow checkpoint has an invalid delta_token parameter")
    if current != int(token.shape[1]):
        raise RuntimeError(
            f"PT-Flow delta-token metadata mismatch: num_delta_tokens={current}, "
            f"parameter_shape={tuple(token.shape)}"
        )
    if target == current:
        return {"base_num_delta_tokens": current, "num_delta_tokens": target}

    generator = torch.Generator(device=token.device)
    generator.manual_seed(int(seed))
    resized = torch.empty(
        (1, target, int(token.shape[2])),
        device=token.device,
        dtype=token.dtype,
    )
    copy_count = min(current, target)
    resized[:, :copy_count].copy_(token.detach()[:, :copy_count])
    if target > current:
        resized[:, current:].normal_(mean=0.0, std=0.01, generator=generator)
    flow.delta_token = nn.Parameter(resized)
    flow.num_delta_tokens = target
    return {"base_num_delta_tokens": current, "num_delta_tokens": target}


def encode_frames(flow: Any, frames: torch.Tensor) -> torch.Tensor:
    batch, window, channels, height, width = frames.shape
    with torch.no_grad():
        encoded = flow.latent_encode(frames.reshape(batch * window, channels, height, width))
    return encoded.reshape(batch, window, encoded.shape[-2], encoded.shape[-1])


def _velocity(flow: Any, z0: torch.Tensor, z: torch.Tensor, zg: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return flow.forward_vmodel(z0, z, zg, tau) * flow.max_time_length


def _feature_regions(latents: torch.Tensor, clusters: int = 4, iterations: int = 10) -> torch.Tensor:
    """Deterministic k-means region assignment over frozen patch features.

    latents: [samples, patches, channels]; returns integer assignments [samples, patches].
    Used to split supervision into arm/object/background regions (patch level) so small
    objects are not diluted by background pixels."""
    samples, patches, channels = latents.shape
    if clusters < 2 or patches < clusters:
        raise ValueError("Need at least two clusters and as many patches")
    features = torch.nn.functional.normalize(latents, dim=-1)
    centers = features[:, torch.linspace(0, patches - 1, clusters).round().long()]  # [S, K, C]
    for _ in range(iterations):
        similarity = torch.bmm(features, centers.transpose(1, 2))  # cosine, [S, P, K]
        assignment = similarity.argmax(dim=-1)
        onehot = torch.nn.functional.one_hot(assignment, clusters).to(features.dtype)  # [S, P, K]
        counts = onehot.sum(dim=1, keepdim=True).clamp_min(1.0)
        new_centers = torch.bmm(onehot.transpose(1, 2), features) / counts.transpose(1, 2)
        centers = torch.where((counts.transpose(1, 2) > 0), new_centers, centers)
    return torch.bmm(features, centers.transpose(1, 2)).argmax(dim=-1)


def _identity_regions(
    s0: torch.Tensor,
    targets: torch.Tensor,
    clusters: int = 4,
    iterations: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """K-means once on start-frame patches; propagate labels by nearest-centroid
    matching so cluster identity stays consistent across later frames.

    s0: [S, P, C] start-frame patch features; targets: [S, P, C] features of the
    matching later frames (row i of targets follows row i of s0). Returns (target
    assignments [S, P], background cluster index [S], where background is each
    sample's largest start-frame cluster)."""
    features0 = torch.nn.functional.normalize(s0, dim=-1)
    centers = features0[:, torch.linspace(0, s0.shape[1] - 1, clusters).round().long()]
    for _ in range(iterations):
        assignment = torch.bmm(features0, centers.transpose(1, 2)).argmax(dim=-1)
        onehot = torch.nn.functional.one_hot(assignment, clusters).to(features0.dtype)
        counts = onehot.sum(dim=1, keepdim=True).clamp_min(1.0)
        new_centers = torch.bmm(onehot.transpose(1, 2), features0) / counts.transpose(1, 2)
        centers = torch.where(counts.transpose(1, 2) > 0, new_centers, centers)
    target_assignment = torch.bmm(
        torch.nn.functional.normalize(targets, dim=-1), centers.transpose(1, 2)
    ).argmax(dim=-1)
    background = counts.sum(dim=1).reshape(-1, clusters).argmax(dim=-1)
    return target_assignment, background


def rollout_losses(
    flow: Any,
    frames: torch.Tensor,
    *,
    rollout_weight: float,
    stability_weight: float,
    velocity_weight: float = 1.0,
    decode_weight: float = 0.0,
    rae: Any = None,
    region_weight: float = 0.0,
    region_clusters: int = 4,
    region_identity: bool = False,
    partflow_weight: float = 0.0,
    motion_weight: float = 0.0,
    boundary_weight: float = 0.0,
    motion_latent_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    latents = encode_frames(flow, frames)
    batch, window, patches, channels = latents.shape
    s0 = latents[:, 0]
    target_states = latents[:, 1:]
    sg = latents[:, -1]
    z0 = flow.delta_decouple(s0, s0)
    zg = flow.delta_decouple(s0, sg)

    target_s0 = s0[:, None].expand(batch, window - 1, patches, channels).reshape(batch * (window - 1), patches, channels)
    target_flat = target_states.reshape(batch * (window - 1), patches, channels)
    target_z = flow.delta_decouple(target_s0, target_flat).reshape(batch, window - 1, -1, channels).detach()

    time_grid = torch.linspace(0.0, 1.0, window, device=frames.device, dtype=frames.dtype)

    def ode_func(time_scalar: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        tau = time_scalar.reshape(1, 1).expand(batch, 1)
        return _velocity(flow, z0, state, zg, tau)

    trajectory = odeint(ode_func, z0, time_grid, method="rk4")[1:]
    trajectory = trajectory.permute(1, 0, 2, 3)
    predicted_flat = trajectory.reshape(batch * (window - 1), -1, channels)
    s0_flat = s0[:, None].expand(batch, window - 1, patches, channels).reshape(batch * (window - 1), patches, channels)
    reconstructed = flow.delta_decode(s0_flat, predicted_flat)

    dt = 1.0 / float(window - 1)
    target_sequence = torch.cat([z0.detach().unsqueeze(1), target_z], dim=1)
    target_velocity = (target_sequence[:, 1:] - target_sequence[:, :-1]) / dt
    local_velocity = []
    for index in range(window - 1):
        tau = time_grid[index + 1].reshape(1, 1).expand(batch, 1)
        local_velocity.append(_velocity(flow, z0, target_z[:, index], zg, tau))
    local_velocity_tensor = torch.stack(local_velocity, dim=1)
    rollout_velocity = []
    state_sequence = torch.cat([z0.unsqueeze(1), trajectory], dim=1)
    for index in range(window):
        tau = time_grid[index].reshape(1, 1).expand(batch, 1)
        rollout_velocity.append(_velocity(flow, z0, state_sequence[:, index], zg, tau))
    rollout_velocity_tensor = torch.stack(rollout_velocity, dim=1)

    loss_rec = torch.mean((reconstructed - target_flat) ** 2)
    loss_dyn = torch.mean((local_velocity_tensor - target_velocity) ** 2)
    loss_rollout = torch.mean((trajectory - target_z) ** 2)
    if window > 2:
        acceleration = rollout_velocity_tensor[:, 2:] - rollout_velocity_tensor[:, 1:-1]
        loss_stability = torch.mean(acceleration ** 2)
    else:
        loss_stability = torch.zeros((), device=frames.device)
    loss_decode = torch.zeros((), device=frames.device)
    loss_motion = torch.zeros((), device=frames.device)
    loss_boundary = torch.zeros((), device=frames.device)
    loss_motion_latent = torch.zeros((), device=frames.device)
    if motion_latent_weight > 0:
        dz_pred = (trajectory[:, 1:] - trajectory[:, :-1]).norm(dim=-1)
        dz_gt = (target_z.detach()[:, 1:] - target_z.detach()[:, :-1]).norm(dim=-1)
        loss_motion_latent = (dz_pred - dz_gt).abs().mean()
    if decode_weight > 0 or motion_weight > 0 or boundary_weight > 0:
        if rae is None:
            raise ValueError("decode-path losses require the frozen RAE decoder")
        decoded = rae.decode(reconstructed).clamp(0.0, 1.0)
        targets_rgb = frames[:, 1:].reshape(decoded.shape)
        if decode_weight > 0:
            loss_decode = (decoded - targets_rgb).abs().mean()
        if motion_weight > 0:
            decoded_seq = decoded.reshape(batch, window - 1, *decoded.shape[1:])
            target_seq = frames[:, 1:]
            td_pred = decoded_seq[:, 1:] - decoded_seq[:, :-1]
            td_gt = target_seq[:, 1:] - target_seq[:, :-1]
            loss_motion = (td_pred - td_gt).abs().mean()
        if boundary_weight > 0:
            anchor_rgb = rae.decode(flow.delta_decode(s0, z0)).clamp(0.0, 1.0)
            loss_boundary = (anchor_rgb - frames[:, 0]).abs().mean()
    loss_region = torch.zeros((), device=frames.device)
    loss_partflow = torch.zeros((), device=frames.device)
    if region_weight > 0 or partflow_weight > 0:
        patch_error = ((reconstructed - target_flat) ** 2).mean(dim=-1).reshape(
            batch, window - 1, patches)
        flat_targets = target_flat.detach().reshape(batch * (window - 1), patches, channels)
        background = None
        with torch.no_grad():
            if region_identity or partflow_weight > 0:
                s0_expanded = (s0.detach().unsqueeze(1)
                               .expand(-1, window - 1, -1, -1)
                               .reshape(batch * (window - 1), patches, channels))
                assignments_flat, background = _identity_regions(
                    s0_expanded, flat_targets, clusters=region_clusters)
                assignments = assignments_flat.reshape(batch, window - 1, patches)
                background = background.reshape(batch, window - 1, 1)
            if region_weight > 0 and not region_identity:
                assignments = _feature_regions(
                    flat_targets, clusters=region_clusters).reshape(batch, window - 1, patches)
        if region_weight > 0:
            terms = []
            for sample in range(batch):
                for step_index in range(window - 1):
                    for cluster in range(region_clusters):
                        selected = assignments[sample, step_index] == cluster
                        if bool(selected.any()):
                            terms.append(patch_error[sample, step_index][selected].mean())
            loss_region = torch.stack(terms).mean() if terms else patch_error.mean()
        if partflow_weight > 0:
            # Directional part supervision in the per-patch decoded-latent space
            # (the ODE trajectory itself lives in the delta-token bottleneck): the
            # predicted displacement of each identity-consistent non-background
            # patch must align with its GT displacement, so parts travel instead
            # of teleporting.
            s0_b = s0.detach().unsqueeze(1)
            gt_disp = target_flat.reshape(batch, window - 1, patches, channels) - s0_b
            pred_disp = reconstructed.reshape(batch, window - 1, patches, channels) - s0_b
            gt_norm = gt_disp.norm(dim=-1)
            keep = gt_norm > gt_norm.mean() * 0.05
            for sample in range(batch):
                keep[sample] &= assignments[sample] != background[sample]
            cosine = (gt_disp * pred_disp).sum(dim=-1) / (
                gt_norm * pred_disp.norm(dim=-1).clamp_min(1e-8))
            if bool(keep.any()):
                loss_partflow = (1.0 - cosine)[keep].mean()
    total = (
        loss_rec
        + velocity_weight * loss_dyn
        + rollout_weight * loss_rollout
        + stability_weight * loss_stability
        + decode_weight * loss_decode
        + region_weight * loss_region
        + partflow_weight * loss_partflow
        + motion_weight * loss_motion
        + boundary_weight * loss_boundary
        + motion_latent_weight * loss_motion_latent
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "loss_rec": float(loss_rec.detach().cpu()),
        "loss_dyn": float(loss_dyn.detach().cpu()),
        "loss_rollout": float(loss_rollout.detach().cpu()),
        "loss_stability": float(loss_stability.detach().cpu()),
        "loss_decode": float(loss_decode.detach().cpu()),
        "loss_region": float(loss_region.detach().cpu()),
        "loss_partflow": float(loss_partflow.detach().cpu()),
        "loss_motion": float(loss_motion.detach().cpu()),
        "loss_boundary": float(loss_boundary.detach().cpu()),
        "loss_motion_latent": float(loss_motion_latent.detach().cpu()),
        "rollout_weight": float(rollout_weight),
        "stability_weight": float(stability_weight),
        "velocity_weight": float(velocity_weight),
    }


def evaluate_validation(
    flow: Any,
    loader: Any,
    device: torch.device,
    *,
    rollout_weight: float,
    stability_weight: float,
    velocity_weight: float = 1.0,
    decode_weight: float = 0.0,
    rae: Any = None,
    region_weight: float = 0.0,
    region_clusters: int = 4,
    region_identity: bool = False,
    partflow_weight: float = 0.0,
    motion_weight: float = 0.0,
    boundary_weight: float = 0.0,
    motion_latent_weight: float = 0.0,
) -> dict[str, float | int]:
    metric_names = ("loss", "loss_rec", "loss_dyn", "loss_rollout", "loss_stability",
                    "loss_decode", "loss_region", "loss_partflow",
                    "loss_motion", "loss_boundary", "loss_motion_latent")
    totals = {name: 0.0 for name in metric_names}
    sample_count = 0
    batch_count = 0
    was_training = bool(flow.training)
    flow.eval()
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device, non_blocking=True)
            _, metrics = rollout_losses(
                flow,
                frames,
                rollout_weight=rollout_weight,
                stability_weight=stability_weight,
                velocity_weight=velocity_weight,
                decode_weight=decode_weight,
                rae=rae,
                region_weight=region_weight,
                region_clusters=region_clusters,
                region_identity=region_identity,
                partflow_weight=partflow_weight,
                motion_weight=motion_weight,
                boundary_weight=boundary_weight,
                motion_latent_weight=motion_latent_weight,
            )
            batch_size = int(frames.shape[0])
            sample_count += batch_size
            batch_count += 1
            for name in metric_names:
                totals[name] += float(metrics[name]) * batch_size
    if sample_count == 0:
        raise RuntimeError("Validation loader is empty")
    if was_training:
        flow.train()
        flow.backbone.eval()
    result: dict[str, float | int] = {name: value / sample_count for name, value in totals.items()}
    result.update({"sample_count": sample_count, "batch_count": batch_count})
    return result


def _capture_trainable_state(flow: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in flow.named_parameters()
        if parameter.requires_grad
    }


def validate_training_checkpoint(flow, loader, device, args, rae=None):
    """Share the exact training weights across initial and periodic validation."""
    return evaluate_validation(
        flow, loader, device, rollout_weight=args.rollout_weight,
        stability_weight=args.stability_weight, velocity_weight=args.velocity_weight,
        decode_weight=args.decode_supervision_weight, rae=rae,
        region_weight=args.region_loss_weight, region_clusters=args.region_clusters,
        region_identity=getattr(args, "region_identity", False),
        partflow_weight=getattr(args, "partflow_weight", 0.0),
        motion_weight=getattr(args, "motion_weight", 0.0),
        boundary_weight=getattr(args, "boundary_weight", 0.0),
        motion_latent_weight=getattr(args, "motion_latent_weight", 0.0),
    )


def _restore_trainable_state(flow: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    parameters = dict(flow.named_parameters())
    if set(parameters).issuperset(state) is False:
        raise RuntimeError("Selected validation state does not match the model")
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(device=parameters[name].device, dtype=parameters[name].dtype))


def parse_snapshot_steps(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    steps = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if any(step <= 0 for step in steps):
        raise ValueError("snapshot steps must be positive")
    return steps


def select_validation_snapshots(
    validation_history: Sequence[Mapping[str, Any]],
    captured_states: Mapping[int, Mapping[str, torch.Tensor]],
    snapshot_steps: Sequence[int],
) -> dict[int, tuple[float, int, Mapping[str, torch.Tensor]]]:
    selected: dict[int, tuple[float, int, Mapping[str, torch.Tensor]]] = {}
    for budget in sorted({int(step) for step in snapshot_steps}):
        candidates = [
            (float(row["loss"]), int(row["step"]))
            for row in validation_history
            if int(row["step"]) <= budget and int(row["step"]) in captured_states
        ]
        if not candidates:
            raise RuntimeError(f"No validation checkpoint captured through snapshot step {budget}")
        loss, step = min(candidates, key=lambda item: (item[0], item[1]))
        selected[budget] = (loss, step, captured_states[step])
    return selected


def train(args: argparse.Namespace) -> dict[str, Any]:
    schema = json.loads(_regular_file(args.sampling_manifest).read_text()).get("schema", "")
    if schema.startswith("odeworld_ptflow_local_supervision_manifest_"):
        raise ValueError("Local windows require the frame-time local trainer, not the legacy [0,1] trainer")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; submit this trainer through Slurm")
    if args.steps <= 0 or args.max_train_demos <= 0 or args.max_val_demos <= 0:
        raise ValueError("steps and demo counts must be positive")
    if args.validation_every < 0:
        raise ValueError("validation_every must be non-negative")
    snapshot_steps = tuple(sorted({int(step) for step in args.snapshot_steps}))
    if any(step > args.steps for step in snapshot_steps):
        raise ValueError(f"snapshot step exceeds training steps: {snapshot_steps} > {args.steps}")
    if args.rollout_weight < 0.0 or args.stability_weight < 0.0 or args.velocity_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    if args.motion_weight < 0.0 or args.boundary_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    if args.motion_latent_weight < 0.0:
        raise ValueError("loss weights must be non-negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    train_rows, val_rows = load_training_rows(
        args.sampling_manifest,
        max_train=args.max_train_demos,
        max_val=args.max_val_demos,
        train_split=args.train_split,
        validation_split=args.validation_split,
    )
    train_rows = resolve_task_file_rows(train_rows, args.libero_root)
    val_rows = resolve_task_file_rows(val_rows, args.libero_root)
    dataset = HDF5WindowDataset(train_rows, args.window)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_windows,
        generator=torch.Generator().manual_seed(args.seed),
    )
    validation_dataset = HDF5WindowDataset(val_rows, args.window)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_windows,
    )
    from models import Dinov2PTflowImgoal

    device = torch.device(args.device)
    flow = Dinov2PTflowImgoal.from_pretrained(str(args.checkpoint)).to(device)
    rae = None
    if args.decode_supervision_weight > 0 or args.motion_weight > 0 or args.boundary_weight > 0:
        from models import Dinov2RAE

        rae = Dinov2RAE.from_pretrained(
            str(Path(__file__).resolve().parents[1] / 'assets/pretrained/ODEWorld-RAE-LIBERO')
        ).to(device).eval().requires_grad_(False)
    if int(flow.max_time_length) != 50:
        raise RuntimeError(f"Expected PT-Flow max_time_length=50, got {flow.max_time_length}")
    token_config = resize_delta_tokens(flow, args.num_delta_tokens, seed=args.seed + 17)
    freeze_backbone(flow)
    trainable = [parameter for parameter in flow.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable PT-Flow parameters remain")
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    config = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "checkpoint": str(args.checkpoint),
        "checkpoint_config_sha256": sha256_file(args.checkpoint / "config.json"),
        "checkpoint_weights_sha256": sha256_file(args.checkpoint / "model.safetensors"),
        "sampling_manifest": str(args.sampling_manifest),
        "sampling_manifest_sha256": sha256_file(args.sampling_manifest),
        "source_files": {
            str(path.relative_to(PROJECT_ROOT)): {"path": str(path), "sha256": sha256_file(path)}
            for path in SOURCE_FILES
        },
        "held_out_splits": list(HELD_OUT_SPLITS),
        "train_identities": [list(identity(row)) for row in train_rows],
        "validation_identities": [list(identity(row)) for row in val_rows],
        "train_split": args.train_split,
        "validation_split": args.validation_split,
        "window": args.window,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "stability_weight": args.stability_weight,
        "rollout_weight": args.rollout_weight,
        "decode_supervision_weight": args.decode_supervision_weight,
        "region_loss_weight": args.region_loss_weight,
        "region_clusters": args.region_clusters,
        "region_identity": bool(args.region_identity),
        "partflow_weight": args.partflow_weight,
        "motion_weight": args.motion_weight,
        "boundary_weight": args.boundary_weight,
        "motion_latent_weight": args.motion_latent_weight,
        "velocity_weight": args.velocity_weight,
        "base_num_delta_tokens": token_config["base_num_delta_tokens"],
        "num_delta_tokens": token_config["num_delta_tokens"],
        "dynamic_token_experiment": {
            "kind": "delta_query_token_count",
            "image_patch_tokens_fixed": True,
            "initialization": "copy_existing_then_seeded_normal_std_0.01",
            "initialization_seed": args.seed + 17,
        },
        "seed": args.seed,
        "device": args.device,
        "frozen_backbone": True,
        "ode_solver": "rk4",
        "ode_horizon": 1.0,
        "validation_every": args.validation_every,
        "checkpoint_selection": "lowest_validation_total_loss",
        "snapshot_steps": list(snapshot_steps),
        "scope": {"env_step_calls": 0, "libero_or_mujoco_started": False, "success_rate_measured": False},
    }
    write_json(args.output_dir / "config.json", config)
    history: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    iterator = iter(loader)
    started = time.perf_counter()
    flow.train()
    flow.backbone.eval()
    initial_validation = validate_training_checkpoint(flow, validation_loader, device, args, rae=rae)
    initial_validation["step"] = 0
    validation_history.append(initial_validation)
    best_validation_loss = float(initial_validation["loss"])
    best_validation_step = 0
    best_trainable_state = _capture_trainable_state(flow)
    captured_states: dict[int, Mapping[str, torch.Tensor]] = {0: best_trainable_state}
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        frames = batch["frames"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = rollout_losses(
            flow,
            frames,
            rollout_weight=args.rollout_weight,
            stability_weight=args.stability_weight,
            velocity_weight=args.velocity_weight,
            decode_weight=args.decode_supervision_weight,
            rae=rae,
            region_weight=args.region_loss_weight,
            region_clusters=args.region_clusters,
            region_identity=args.region_identity,
            partflow_weight=args.partflow_weight,
            motion_weight=args.motion_weight,
            boundary_weight=args.boundary_weight,
            motion_latent_weight=args.motion_latent_weight,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {metrics}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm).detach().cpu())
        optimizer.step()
        metrics.update({"step": step, "grad_norm": grad_norm, "identities": [list(value) for value in batch["identity"]]})
        history.append(metrics)
        if step == 1 or step == args.steps or step % args.log_every == 0:
            print(json.dumps({key: value for key, value in metrics.items() if key != "identities"}, sort_keys=True), flush=True)
        should_validate = step == args.steps or (args.validation_every > 0 and step % args.validation_every == 0)
        if should_validate:
            validation_metrics = validate_training_checkpoint(flow, validation_loader, device, args, rae=rae)
            validation_metrics["step"] = step
            validation_history.append(validation_metrics)
            captured_states[step] = _capture_trainable_state(flow)
            print(json.dumps({"validation": validation_metrics}, sort_keys=True), flush=True)
            if float(validation_metrics["loss"]) < best_validation_loss:
                best_validation_loss = float(validation_metrics["loss"])
                best_validation_step = step
                best_trainable_state = captured_states[step]
    selected_snapshots = select_validation_snapshots(validation_history, captured_states, snapshot_steps)
    _restore_trainable_state(flow, best_trainable_state)
    output_checkpoint = args.output_dir / "checkpoint"
    def save_checkpoint(path: Path, *, selected_step: int, selected_loss: float, label: str) -> None:
        checkpoint_config = dict(getattr(flow, "_hub_mixin_config", {}) or {})
        checkpoint_config.update(
            {
                "num_delta_tokens": token_config["num_delta_tokens"],
                "dynamic_token_experiment": config["dynamic_token_experiment"],
                "training_sampling_manifest_sha256": config["sampling_manifest_sha256"],
                "training_split": args.train_split,
                "validation_split": args.validation_split,
                "selected_validation_step": selected_step,
                "selected_validation_loss": selected_loss,
                "checkpoint_budget_label": label,
            }
        )
        flow.save_pretrained(str(path), config=checkpoint_config)

    save_checkpoint(
        output_checkpoint,
        selected_step=best_validation_step,
        selected_loss=best_validation_loss,
        label="full_training_selected",
    )
    snapshot_metadata: dict[str, Any] = {}
    for budget, (snapshot_loss, snapshot_step, snapshot_state) in selected_snapshots.items():
        label = f"budget_{budget // 1000}k"
        snapshot_path = args.output_dir / f"checkpoint_{label}"
        _restore_trainable_state(flow, snapshot_state)
        save_checkpoint(
            snapshot_path,
            selected_step=snapshot_step,
            selected_loss=snapshot_loss,
            label=label,
        )
        snapshot_metadata[label] = {
            "budget_steps": budget,
            "selected_validation_loss": snapshot_loss,
            "checkpoint": str(snapshot_path),
        }
    _restore_trainable_state(flow, best_trainable_state)
    summary = {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "all_checks_passed": True,
        "scope": config["scope"],
        "train_steps": args.steps,
        "train_demo_count": len(train_rows),
        "validation_demo_count": len(val_rows),
        "validation_history": validation_history,
        "selected_validation_step": best_validation_step,
        "selected_validation_loss": best_validation_loss,
        "snapshot_checkpoints": snapshot_metadata,
        "base_num_delta_tokens": token_config["base_num_delta_tokens"],
        "num_delta_tokens": token_config["num_delta_tokens"],
        "final_metrics": history[-1],
        "initial_metrics": history[0],
        "history": history,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(output_checkpoint),
    }
    write_json(args.output_dir / "training_summary.json", summary)
    print(json.dumps({"classification": CLASSIFICATION, "completed": True, "steps": args.steps, "output_dir": str(args.output_dir)}), flush=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--sampling-manifest", type=Path, default=SAMPLING_MANIFEST)
    parser.add_argument(
        "--libero-root",
        default=None,
        help="LIBERO HDF5 root for manifests with relative task_file paths "
        "(defaults to the LIBERO_ROOT environment variable)",
    )
    parser.add_argument("--max-train-demos", type=int, default=2)
    parser.add_argument("--max-val-demos", type=int, default=1)
    parser.add_argument("--train-split")
    parser.add_argument("--validation-split")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--validation-every", type=int, default=0)
    parser.add_argument(
        "--snapshot-steps",
        type=parse_snapshot_steps,
        default=(),
        help="Comma-separated validation budgets to save, for example 26000,52000.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--stability-weight", type=float, default=1e-2)
    parser.add_argument("--decode-supervision-weight", type=float, default=0.0,
                        help="T3: L1 between frozen-RAE-decoded anchor states and GT frames")
    parser.add_argument("--region-loss-weight", type=float, default=0.0,
                        help="T3: per-region normalized latent loss over DINOv2-feature clusters")
    parser.add_argument("--region-clusters", type=int, default=4)
    parser.add_argument("--region-identity", action="store_true",
                        help="T4: cluster the start frame once and propagate labels by "
                             "nearest-centroid matching, so regions keep one identity "
                             "across all frames instead of per-frame re-clustering")
    parser.add_argument("--partflow-weight", type=float, default=0.0,
                        help="T4: cosine alignment between predicted and GT per-patch "
                             "latent displacement on identity-consistent non-background "
                             "regions (parts travel instead of teleporting)")
    parser.add_argument("--motion-weight", type=float, default=0.0,
                        help="T5: L1 between consecutive-frame differences of the decoded "
                             "rollout and of the GT frames, so under-displaced trajectories "
                             "are penalized through the dynamics pathway")
    parser.add_argument("--boundary-weight", type=float, default=0.0,
                        help="T5: L1 between the decoded anchor state and the start frame, "
                             "suppressing the spurious first-frame re-render jump")
    parser.add_argument("--motion-latent-weight", type=float, default=0.0,
                        help="T5b: match per-step latent displacement magnitude between the "
                             "rollout and the true trajectory, never touching the decode path")
    parser.add_argument("--rollout-weight", type=float, default=1.0)
    parser.add_argument("--velocity-weight", type=float, default=1.0)
    parser.add_argument(
        "--num-delta-tokens",
        type=int,
        default=1,
        help="Number of learned PT-Flow dynamic query tokens; image patch tokens stay fixed.",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
