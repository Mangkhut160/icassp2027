"""Create GIF previews and a contact sheet from LIBERO rollout MP4 files."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_LIBERO = PROJECT_ROOT / "assets" / "libero"


def read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"Could not decode frames from {path}")
    return frames


def label_frame(frame: np.ndarray, label: str, size: int = 144) -> Image.Image:
    image = Image.fromarray(frame).resize((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size + 26), "black")
    canvas.paste(image, (0, 26))
    draw = ImageDraw.Draw(canvas)
    draw.text((5, 5), label, fill="white", font=ImageFont.load_default())
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    root = args.output_dir / "libero"
    preview_dir = args.output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        gt = read_video(case_dir / "gt_goal_rollout.mp4")
        predicted = read_video(case_dir / "predicted_goal_rollout.mp4")
        if len(gt) != len(predicted):
            raise RuntimeError(f"Frame count mismatch in {case_dir}")

        for kind, frames in (("gt_goal", gt), ("language_predicted", predicted)):
            images = [Image.fromarray(frame) for frame in frames]
            images[0].save(
                preview_dir / f"{case_dir.name}_{kind}.gif",
                save_all=True,
                append_images=images[1:],
                duration=125,
                loop=0,
                optimize=False,
            )

        start = cv2.cvtColor(cv2.imread(str(ASSETS_LIBERO / case_dir.name / "start.png")), cv2.COLOR_BGR2RGB)
        goal = cv2.cvtColor(cv2.imread(str(ASSETS_LIBERO / case_dir.name / "goal.png")), cv2.COLOR_BGR2RGB)
        indexes = [0, len(gt) // 2, len(gt) - 1]
        panels = [
            label_frame(start, "start"),
            label_frame(goal, "target"),
            *(label_frame(gt[index], f"goal t={index}") for index in indexes),
            *(label_frame(predicted[index], f"lang t={index}") for index in indexes),
        ]
        row = Image.new("RGB", (len(panels) * 144, 170), "#202124")
        for index, panel in enumerate(panels):
            row.paste(panel, (index * 144, 0))
        draw = ImageDraw.Draw(row)
        draw.text((5, 152), case_dir.name, fill="white", font=ImageFont.load_default())
        rows.append(row)

    sheet = Image.new("RGB", (rows[0].width, len(rows) * rows[0].height), "#202124")
    for index, row in enumerate(rows):
        sheet.paste(row, (0, index * row.height))
    sheet.save(preview_dir / "libero_all_cases_comparison.png")
    print(f"Wrote {len(rows) * 2} GIFs and {preview_dir / 'libero_all_cases_comparison.png'}")


if __name__ == "__main__":
    main()
