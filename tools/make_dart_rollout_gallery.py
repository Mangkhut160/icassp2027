"""Render before/after rollout galleries for the DART retrained PT-Flow.

Reads stored open-loop predictions from two evaluation roots produced by
tools/evaluate_odeworld_ptflow_checkpoint.py — a latent-only baseline root
and the DART root — and renders per-demo GT|baseline|DART trajectory strips
(view-size and large), animated side-by-side GIFs, native-resolution frame
trees, an HTML step viewer, and a contact sheet. CPU-only; no model
inference, no environment.

Demo selection: per-suite median by DART L1, plus the overall best and
worst L1 improvement, so the gallery is representative rather than
cherry-picked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CONDITION = "open_loop_k64_oracle"
STRIP_STEPS = (0, 9, 18, 27, 36, 45, 54, 63)
LARGE_STEPS = (0, 18, 36, 63)
CONTACT_STEPS = (0, 18, 36, 63)
ROW_LABELS = ("GT", "Baseline (before)", "DART (after)")
PANEL_TEXT = ("GT", "Baseline", "DART")

FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
)


def load_font(size: int):
    for path in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def discover_trials(root: Path) -> dict[tuple[str, str, str], Path]:
    found: dict[tuple[str, str, str], Path] = {}
    for pred in sorted(root.glob("trials/*/*/*/predictions.npz")):
        demo_dir = pred.parent
        key = (demo_dir.parent.parent.name, demo_dir.parent.name, demo_dir.name)
        found[key] = demo_dir
    return found


def demo_mean_l1(demo_dir: Path) -> float | None:
    """Per-demo mean L1 from metrics.jsonl if present, else None."""
    path = demo_dir / "metrics.jsonl"
    try:
        values = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("condition") == CONDITION and isinstance(row.get("l1"), (int, float)):
                values.append(float(row["l1"]))
        return float(np.mean(values)) if values else None
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def load_demo(demo_dir: Path) -> dict:
    pred = np.load(demo_dir / "predictions.npz")[CONDITION]  # (64,256,256,3)
    ref = np.load(demo_dir / "reference_frames.npz")
    targets = ref["targets"]
    l1 = np.abs(pred.astype(np.int16) - targets.astype(np.int16)).mean(axis=(1, 2, 3))
    return {"start": ref["start"], "targets": targets, "pred": pred, "l1_per_step": l1, "l1_mean": float(l1.mean())}


def select_demos(base_l1: dict, cand_l1: dict) -> list[tuple[str, str, str]]:
    shared = sorted(set(base_l1) & set(cand_l1))
    if not shared:
        raise SystemExit("no shared demos between the two eval roots")
    by_suite: dict[str, list] = {}
    for key in shared:
        by_suite.setdefault(key[0], []).append(key)
    picked: list[tuple[str, str, str]] = []
    for suite in sorted(by_suite):
        keys = sorted(by_suite[suite], key=lambda k: cand_l1[k])
        picked.append(keys[len(keys) // 2])
    ranked = sorted(shared, key=lambda k: base_l1[k] - cand_l1[k])
    for key in (ranked[0], ranked[-1]):
        if key not in picked:
            picked.append(key)
    return picked


def short_name(key: tuple[str, str, str]) -> str:
    suite, task, demo = key
    return f"{suite}_{task[:36]}_{demo}"


def to_u8(frame: np.ndarray) -> Image:
    return Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))


def scaled(frame: np.ndarray, cell: int) -> Image:
    img = to_u8(frame)
    if img.width != cell:
        img = img.resize((cell, cell), Image.LANCZOS)
    return img


def render_strip(run: dict, base_pred: np.ndarray, key: tuple[str, str, str], out_png: Path) -> None:
    cell, margin = 256, 150
    cols = len(STRIP_STEPS)
    header = 46
    rows = 3
    canvas = Image.new("RGB", (margin + cols * cell, header + rows * cell), "white")
    draw = ImageDraw.Draw(canvas)
    title = f"{key[0]} / {key[1]} / {key[2]}   L1  baseline {run['baseline_l1']:.2f} -> DART {run['l1_mean']:.2f}"
    draw.text((margin + 4, 6), title, fill="black")
    for row_i, source in enumerate((run["targets"], base_pred, run["pred"])):
        y0 = header + row_i * cell
        for c, s in enumerate(STRIP_STEPS):
            tile = to_u8(source[s]).resize((cell, cell), Image.BILINEAR)
            canvas.paste(tile, (margin + c * cell, y0))
        draw.text((6, y0 + cell // 2 - 5), ROW_LABELS[row_i], fill="black")
    canvas.save(out_png)


def render_strip_large(run: dict, base_pred: np.ndarray, key: tuple[str, str, str], out_png: Path, cell: int = 512) -> None:
    font = load_font(30)
    margin = 210
    cols = len(LARGE_STEPS)
    header = 104
    canvas = Image.new("RGB", (margin + cols * cell, header + 3 * cell), "white")
    draw = ImageDraw.Draw(canvas)
    title = f"{key[0]} / {key[1]} / {key[2]}"
    draw.text((margin, 12), title, fill="black", font=font)
    draw.text(
        (margin, 54),
        f"L1  baseline {run['baseline_l1']:.2f} -> DART {run['l1_mean']:.2f}",
        fill="#333333",
        font=font,
    )
    for row_i, source in enumerate((run["targets"], base_pred, run["pred"])):
        y0 = header + row_i * cell
        for c, s in enumerate(LARGE_STEPS):
            canvas.paste(scaled(source[s], cell), (margin + c * cell, y0))
        draw.text((8, y0 + cell // 2 - 18), ROW_LABELS[row_i], fill="black", font=font)
    canvas.save(out_png)


def render_gif(run: dict, base_pred: np.ndarray, out_gif: Path, duration_ms: int = 250) -> None:
    frames = []
    for s in range(64):
        band = Image.new("RGB", (3 * 256 + 8, 256 + 16), "white")
        draw = ImageDraw.Draw(band)
        for i, source in enumerate((run["targets"], base_pred, run["pred"])):
            band.paste(to_u8(source[s]), (i * (256 + 4), 16))
        draw.text((4, 2), "GT | Baseline | DART   step %02d/63" % s, fill="black")
        frames.append(band)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def render_gif_large(run: dict, base_pred: np.ndarray, out_gif: Path, duration_ms: int = 250, cell: int = 512) -> None:
    font = load_font(24)
    gap, header = 8, 42
    frames = []
    for s in range(64):
        band = Image.new("RGB", (3 * cell + 2 * gap, cell + header), "white")
        draw = ImageDraw.Draw(band)
        for i, source in enumerate((run["targets"], base_pred, run["pred"])):
            band.paste(scaled(source[s], cell), (i * (cell + gap), header))
        draw.text((6, 7), f"{PANEL_TEXT[0]} | {PANEL_TEXT[1]} | {PANEL_TEXT[2]}   step {s:02d}/63", fill="black", font=font)
        frames.append(band)
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def write_native_frames(run: dict, base_pred: np.ndarray, frames_dir: Path) -> None:
    for panel, source in zip(("gt", "baseline", "dart"), (run["targets"], base_pred, run["pred"])):
        out = frames_dir / panel
        out.mkdir(parents=True, exist_ok=True)
        for s in range(64):
            to_u8(source[s]).save(out / f"t{s:02d}.png")


VIEWER_TEMPLATE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>DART Rollout 对比</title>
<style>
:root{color-scheme:dark;--cell:256px}
body{margin:0;padding:24px;font:14px/1.6 system-ui,sans-serif;background:#141414;color:#e9e9e9}
h1{font-size:18px;margin:0 0 2px}
.sub{color:#969696;margin:0 0 18px}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-bottom:8px}
select{font-size:14px;max-width:52ch}
button{font-size:14px;padding:2px 10px}
.stepnum{color:#bbb;min-width:7ch;display:inline-block}
.meta{color:#c8c8c8;margin:6px 0 14px}
.grid{display:grid;grid-template-columns:repeat(3,var(--cell));gap:8px}
.panel{position:relative}
.panel img{width:var(--cell);height:var(--cell);display:block;background:#000}
.panel span{position:absolute;top:5px;left:7px;font-size:12px;color:#fff;text-shadow:0 1px 3px #000}
.pixel img{image-rendering:pixelated}
</style>
</head>
<body>
<h1>DART 训练前后 Rollout 对比</h1>
<p class="sub">GT 为数据集真值帧；图像为原始 256×256 分辨率，放大倍数只做最近邻缩放。</p>
<div class="controls">
  <select id="demo"></select>
  <button id="prev" title="上一步">◀</button>
  <input type="range" id="step" min="0" max="63" value="0" style="width:260px">
  <button id="next" title="下一步">▶</button>
  <button id="play">播放</button>
  <span class="stepnum" id="stepval">0 / 63</span>
  <label>放大 <select id="zoom"><option value="1">1x (256px)</option><option value="2">2x (512px)</option><option value="3">3x (768px)</option></select></label>
  <label><input type="checkbox" id="pixel" checked> 最近邻</label>
</div>
<div class="meta" id="meta"></div>
<div class="grid" id="grid">
  <div class="panel"><span>GT 真值</span><img id="img-gt" alt="GT"></div>
  <div class="panel"><span>训练前 Baseline</span><img id="img-base" alt="Baseline"></div>
  <div class="panel"><span>训练后 DART</span><img id="img-dart" alt="DART"></div>
</div>
<script>
const DEMOS = __DEMOS_JSON__;
const $ = (id) => document.getElementById(id);
const sel = $("demo"), step = $("step");
DEMOS.forEach((d, i) => {
  const o = document.createElement("option");
  o.value = i;
  o.textContent = d.suite + " / " + d.task + " (" + d.demo + ")";
  sel.appendChild(o);
});
const pad = (n) => String(n).padStart(2, "0");
function show() {
  const d = DEMOS[sel.value], s = +step.value;
  $("img-gt").src = `frames/${d.dir}/gt/t${pad(s)}.png`;
  $("img-base").src = `frames/${d.dir}/baseline/t${pad(s)}.png`;
  $("img-dart").src = `frames/${d.dir}/dart/t${pad(s)}.png`;
  $("stepval").textContent = s + " / 63";
  $("meta").textContent = "该条 demo 64 步平均 L1（0-255）：训练前 " + d.l1_baseline.toFixed(2) + " → 训练后 " + d.l1_dart.toFixed(2);
}
let timer = null;
$("play").onclick = () => {
  if (timer) { clearInterval(timer); timer = null; $("play").textContent = "播放"; return; }
  $("play").textContent = "暂停";
  timer = setInterval(() => { step.value = (+step.value + 1) % 64; show(); }, 250);
};
sel.onchange = () => { step.value = 0; show(); };
step.oninput = show;
$("prev").onclick = () => { step.value = (+step.value + 63) % 64; show(); };
$("next").onclick = () => { step.value = (+step.value + 1) % 64; show(); };
$("zoom").onchange = (e) => { document.documentElement.style.setProperty("--cell", e.target.value + "px"); show(); };
$("pixel").onchange = (e) => { $("grid").classList.toggle("pixel", e.target.checked); };
show();
</script>
</body>
</html>
"""


def write_index_html(runs: list[dict], out_html: Path) -> None:
    demos = [
        {
            "dir": short_name(run["key"]),
            "suite": run["key"][0],
            "task": run["key"][1],
            "demo": run["key"][2],
            "l1_baseline": run["baseline_l1"],
            "l1_dart": run["l1_mean"],
        }
        for run in runs
    ]
    page = VIEWER_TEMPLATE.replace(
        "__DEMOS_JSON__", json.dumps(demos, ensure_ascii=False).replace("</", "<\\/")
    )
    out_html.write_text(page)


def render_contact_sheet(runs: list[dict], out_png: Path) -> None:
    cell, margin, header = 128, 190, 22
    cols = len(CONTACT_STEPS) * 3
    rows = len(runs)
    canvas = Image.new("RGB", (margin + cols * cell, header + rows * (cell + 16)), "white")
    draw = ImageDraw.Draw(canvas)
    for c, s in enumerate(CONTACT_STEPS):
        for panel, label in enumerate(ROW_LABELS):
            x = margin + (c * 3 + panel) * cell
            draw.text((x + 4, 4), f"{label} t={s}", fill="black")
    for r, run in enumerate(runs):
        y0 = header + r * (cell + 16)
        for c, s in enumerate(CONTACT_STEPS):
            for panel, source in enumerate((run["targets"], run["baseline_pred"], run["pred"])):
                tile = to_u8(source[s]).resize((cell, cell), Image.BILINEAR)
                canvas.paste(tile, (margin + (c * 3 + panel) * cell, y0))
        info = f"{run['key'][0]}/{run['key'][2]}\nL1 {run['baseline_l1']:.2f} -> {run['l1_mean']:.2f}"
        draw.multiline_text((6, y0 + cell // 2 - 14), info, fill="black")
    canvas.save(out_png)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--gif-demos", type=int, default=3, help="how many selected demos also get an animated GIF")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base_trials = discover_trials(args.baseline_dir)
    cand_trials = discover_trials(args.candidate_dir)

    base_l1, cand_l1 = {}, {}
    for key, demo_dir in cand_trials.items():
        l1 = demo_mean_l1(demo_dir)
        if l1 is not None:
            cand_l1[key] = l1
    for key, demo_dir in base_trials.items():
        l1 = demo_mean_l1(demo_dir)
        if l1 is not None:
            base_l1[key] = l1
    missing_array_fallback = len(base_l1) < len(base_trials) or len(cand_l1) < len(cand_trials)
    print(f"shared trials: base={len(base_trials)} cand={len(cand_trials)} metrics-derived L1 base={len(base_l1)} cand={len(cand_l1)}")

    if missing_array_fallback:
        # Metrics incomplete: derive L1 from stored frames for the missing side.
        for label, trials, store in (("base", base_trials, base_l1), ("cand", cand_trials, cand_l1)):
            for key, demo_dir in trials.items():
                if key in store:
                    continue
                pred = np.load(demo_dir / "predictions.npz")[CONDITION]
                targets = np.load(demo_dir / "reference_frames.npz")["targets"]
                store[key] = float(np.abs(pred.astype(np.int16) - targets.astype(np.int16)).mean())
                del pred, targets

    picked = select_demos(base_l1, cand_l1)
    print(f"selected {len(picked)} demos: {picked}")

    runs = []
    for key in picked:
        print(f"loading {key}", flush=True)
        run = load_demo(cand_trials[key])
        base_run = load_demo(base_trials[key])
        run["baseline_pred"] = base_run["pred"]
        run["baseline_l1"] = base_run["l1_mean"]
        run["key"] = key
        runs.append(run)

    name = lambda run: short_name(run["key"])  # noqa: E731
    large_dir = args.out_dir / "large"
    frames_root = args.out_dir / "frames"
    large_dir.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)
    for run in runs:
        render_strip(run, run["baseline_pred"], run["key"], args.out_dir / f"strip_{name(run)}.png")
        render_strip_large(run, run["baseline_pred"], run["key"], large_dir / f"strip_{name(run)}.png")
        write_native_frames(run, run["baseline_pred"], frames_root / name(run))
    for run in runs[: args.gif_demos]:
        render_gif(run, run["baseline_pred"], args.out_dir / f"rollout_{name(run)}.gif")
        render_gif_large(run, run["baseline_pred"], large_dir / f"rollout_{name(run)}.gif")
    render_contact_sheet(runs, args.out_dir / "contact_sheet.png")
    write_index_html(runs, args.out_dir / "index.html")

    summary = {
        "condition": CONDITION,
        "baseline_dir": str(args.baseline_dir),
        "candidate_dir": str(args.candidate_dir),
        "demos": [
            {
                "suite": run["key"][0],
                "task": run["key"][1],
                "demo": run["key"][2],
                "l1_baseline": run["baseline_l1"],
                "l1_dart": run["l1_mean"],
                "strip": f"strip_{name(run)}.png",
                "strip_large": f"large/strip_{name(run)}.png",
                "frame_viewer": "index.html",
            }
            for run in runs
        ],
    }
    (args.out_dir / "gallery_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"gallery written to {args.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
