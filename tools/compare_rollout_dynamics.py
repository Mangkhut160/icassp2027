"""Compare open-loop rollout dynamics between two checkpoint evaluations.

Answers the teleport question quantitatively: on demos shared by both eval
roots, compute per-step L1, motion profiles (frame-to-frame change), motion
concentration (fraction of total motion in the busiest 3-step window), and
the largest single-step jump; render step curves, trajectory strips, and a
side-by-side GIF.

Both roots must come from tools/evaluate_odeworld_ptflow_checkpoint.py
(trials/<suite>/<task>/<demo>/{predictions,reference_frames}.npz).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

Grid = tuple[int, int]


def discover_trials(root: Path) -> dict[tuple[str, str, str], Path]:
    found: dict[tuple[str, str, str], Path] = {}
    for pred in sorted(root.glob("trials/*/*/*/predictions.npz")):
        demo_dir = pred.parent
        key = (demo_dir.parent.parent.name, demo_dir.parent.name, demo_dir.name)
        found[key] = demo_dir
    return found


def shared_demo_dirs(
    baseline_root: Path,
    candidate_root: Path,
    max_demos: int,
    seed: int,
) -> list[tuple[tuple[str, str, str], Path, Path]]:
    base = discover_trials(baseline_root)
    cand = discover_trials(candidate_root)
    shared = sorted(set(base) & set(cand))
    if not shared:
        raise SystemExit("no shared demos between the two eval roots")
    if max_demos and len(shared) > max_demos:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(shared), size=max_demos, replace=False)
        shared = [shared[i] for i in sorted(pick.tolist())]
    return [(key, base[key], cand[key]) for key in shared]


def _uint(a: np.ndarray) -> np.ndarray:
    return a.astype(np.int16)


def demo_dynamics(demo_dir: Path, condition: str) -> dict:
    pred = np.load(demo_dir / "predictions.npz")[condition]  # (64,256,256,3)
    ref = np.load(demo_dir / "reference_frames.npz")
    start, targets = ref["start"], ref["targets"]

    l1 = np.abs(_uint(pred) - _uint(targets)).mean(axis=(1, 2, 3))

    def motion(frames: np.ndarray) -> np.ndarray:
        seq = np.concatenate([start[None], frames], axis=0)
        return np.abs(_uint(seq[1:]) - _uint(seq[:-1])).mean(axis=(1, 2, 3))

    motion_pred = motion(pred)
    motion_gt = motion(targets)

    def conc3(profile: np.ndarray) -> float:
        window = np.convolve(profile, np.ones(3), mode="valid")
        return float(window.max() / max(profile.sum(), 1e-9))

    return {
        "l1_per_step": l1.tolist(),
        "l1_mean": float(l1.mean()),
        "motion_pred_per_step": motion_pred.tolist(),
        "motion_gt_per_step": motion_gt.tolist(),
        "motion_pred_total": float(motion_pred.sum()),
        "concentration3_pred": conc3(motion_pred),
        "concentration3_gt": conc3(motion_gt),
        "max_jump_pred": float(motion_pred.max()),
        "argmax_jump_pred": int(motion_pred.argmax()),
        "max_jump_gt": float(motion_gt.max()),
        "argmax_jump_gt": int(motion_gt.argmax()),
        "start": start,
        "targets": targets,
        "pred": pred,
    }


def aggregate(runs: list[dict]) -> dict:
    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in runs]))

    curves = {
        "l1_per_step": np.mean([r["l1_per_step"] for r in runs], axis=0).tolist(),
        "motion_pred_per_step": np.mean([r["motion_pred_per_step"] for r in runs], axis=0).tolist(),
        "motion_gt_per_step": np.mean([r["motion_gt_per_step"] for r in runs], axis=0).tolist(),
    }
    return {
        "demo_count": len(runs),
        "l1_mean": mean("l1_mean"),
        "concentration3_pred": mean("concentration3_pred"),
        "concentration3_gt": mean("concentration3_gt"),
        "max_jump_pred": mean("max_jump_pred"),
        "max_jump_gt": mean("max_jump_gt"),
        "argmax_jump_pred_hist": {str(s): int(sum(r["argmax_jump_pred"] == s for r in runs)) for s in range(64)},
        "curves": curves,
    }


def plot_curves(base_agg: dict, cand_agg: dict, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.arange(64)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(steps, base_agg["curves"]["l1_per_step"], label="baseline", color="#888888")
    ax1.plot(steps, cand_agg["curves"]["l1_per_step"], label="candidate", color="#d62728")
    ax1.set_ylabel("per-step L1 (0-255)")
    ax1.set_title(f"Open-loop per-step L1  (baseline {base_agg['l1_mean']:.2f} vs candidate {cand_agg['l1_mean']:.2f})")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(steps, base_agg["curves"]["motion_pred_per_step"], label="baseline pred", color="#888888")
    ax2.plot(steps, cand_agg["curves"]["motion_pred_per_step"], label="candidate pred", color="#d62728")
    ax2.plot(steps, cand_agg["curves"]["motion_gt_per_step"], label="ground truth", color="#1f77b4", linestyle="--")
    ax2.set_ylabel("frame-to-frame motion")
    ax2.set_xlabel("open-loop step")
    ax2.set_title("Motion profile: where does the predicted change happen?")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


STRIP_STEPS = (0, 9, 18, 27, 36, 45, 54, 63)


def to_u8(frame: np.ndarray) -> Image:
    return Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))


def plot_strips(runs: list[dict], out_png: Path, max_rows: int = 6) -> None:
    rows = min(len(runs), max_rows)
    cols = len(STRIP_STEPS)
    cell = 128
    panel_w, panel_h = cols * cell, rows * 3 * cell + rows * 18
    canvas = Image.new("RGB", (panel_w, panel_h), "white")
    draw = ImageDraw.Draw(canvas)
    for r, run in enumerate(runs[:rows]):
        y0 = r * (3 * cell + 18)
        labels = ("GT", "baseline", "candidate")
        for row_i, source in enumerate((run["targets"], run["baseline_pred"], run["pred"])):
            for c, s in enumerate(STRIP_STEPS):
                tile = to_u8(source[s]).resize((cell, cell), Image.BILINEAR)
                canvas.paste(tile, (c * cell, y0 + row_i * cell))
        draw.text((4, y0 + 3 * cell + 2), f"{run['key'][0]}/{run['key'][2]}  candL1={run['l1_mean']:.1f}", fill="black")
        for row_i, label in enumerate(labels):
            draw.text((4, y0 + row_i * cell + 3), label, fill="white")
    canvas.save(out_png)


def make_gif(run: dict, out_gif: Path, duration_ms: int = 250) -> None:
    frames = []
    for s in range(64):
        band = Image.new("RGB", (3 * 256 + 8, 256 + 14), "white")
        draw = ImageDraw.Draw(band)
        for i, source in enumerate((run["targets"], run["baseline_pred"], run["pred"])):
            band.paste(to_u8(source[s]), (i * (256 + 4), 14))
        draw.text((4, 1), "GT | baseline | candidate", fill="black")
        frames.append(band)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--condition", default="open_loop_k64_oracle")
    parser.add_argument("--max-shared-demos", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = shared_demo_dirs(args.baseline_dir, args.candidate_dir, args.max_shared_demos, args.seed)
    report: dict = {
        "condition": args.condition,
        "baseline_dir": str(args.baseline_dir),
        "candidate_dir": str(args.candidate_dir),
        "shared_demo_count": len(pairs),
    }

    base_runs, cand_runs, strips = [], [], []
    for i, (key, base_dir, cand_dir) in enumerate(pairs):
        base_run = demo_dynamics(base_dir, args.condition)
        cand_run = demo_dynamics(cand_dir, args.condition)
        base_run["key"], cand_run["key"] = key, key
        cand_run["baseline_pred"] = base_run.pop("pred")
        base_runs.append(base_run)
        cand_runs.append(cand_run)
        strips.append(cand_run)
        print(f"[{i + 1}/{len(pairs)}] {key}  l1 base={base_run['l1_mean']:.2f} cand={cand_run['l1_mean']:.2f}", flush=True)

    base_agg = aggregate(base_runs)
    cand_agg = aggregate(cand_runs)
    report["baseline"] = {k: v for k, v in base_agg.items() if k != "curves"}
    report["candidate"] = {k: v for k, v in cand_agg.items() if k != "curves"}

    plot_curves(base_agg, cand_agg, args.out_dir / "step_curves.png")
    strips.sort(key=lambda r: r["l1_mean"])
    median_run = strips[len(strips) // 2]
    plot_strips(strips, args.out_dir / "trajectory_strips.png")
    make_gif(median_run, args.out_dir / "rollout_compare.gif")
    report["gif_demo"] = list(median_run["key"])

    (args.out_dir / "dynamics_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("shared_demo_count",)}, indent=0))
    print("baseline:", json.dumps(report["baseline"], indent=1)[:400])
    print("candidate:", json.dumps(report["candidate"], indent=1)[:400])
    print(f"outputs in {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
