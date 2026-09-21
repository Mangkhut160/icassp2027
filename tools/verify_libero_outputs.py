"""Verify the expected LIBERO GPU demo artifacts and write a compact summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def video_info(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {path}")
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    return {
        "bytes": path.stat().st_size,
        "declared_frames": declared_frames,
        "decoded_frames": decoded,
        "fps": fps,
        "width": width,
        "height": height,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-frames", type=int, default=64)
    args = parser.parse_args()
    cases = []
    for case_dir in sorted(path for path in (args.output_dir / "libero").iterdir() if path.is_dir()):
        record = {
            "case": case_dir.name,
            "goal_prediction_comparison": (case_dir / "goal_prediction_comparison.png").stat().st_size,
            "gt_goal_rollout": video_info(case_dir / "gt_goal_rollout.mp4"),
            "language_predicted_rollout": video_info(case_dir / "predicted_goal_rollout.mp4"),
            "pca_field": video_info(case_dir / "pca_field.mp4"),
        }
        for key in ("gt_goal_rollout", "language_predicted_rollout", "pca_field"):
            actual = record[key]
            if actual["decoded_frames"] != args.expected_frames:
                raise RuntimeError(f"{case_dir.name}/{key} decoded {actual['decoded_frames']} frames")
        cases.append(record)
    if len(cases) != 5:
        raise RuntimeError(f"Expected five LIBERO cases, found {len(cases)}")
    summary = {
        "expected_cases": 5,
        "expected_frames_per_video": args.expected_frames,
        "all_cases_verified": True,
        "cases": cases,
    }
    path = args.output_dir / "verification_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Verified {len(cases)} cases; wrote {path}")


if __name__ == "__main__":
    main()
