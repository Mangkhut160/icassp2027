#!/usr/bin/env bash
# Bundled LIBERO image-and-language demo (offline prediction from static
# start/goal images; mock test).
#
# Environment overrides:
#   PYTHON      Python interpreter (default: python3)
#   STEPS, FPS  Rollout length and output frame rate
#   TORCH_HOME  Torch hub cache for the DINOv2 code/weights (default: repo .cache/torch)
#   HF_HOME     Hugging Face cache (default: repo .cache/huggingface)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
OUTPUT_DIR=${1:-"$ROOT/outputs/gpu_libero_steps200"}
STEPS=${STEPS:-200}
FPS=${FPS:-10}

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export TMPDIR="${TMPDIR:-/tmp}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-$TMPDIR}"

cd "$ROOT"
echo "host=$(hostname)"
echo "steps=$STEPS"
echo "fps=$FPS"
"$PYTHON" demo_infer.py \
  --dataset libero \
  --output-dir "$OUTPUT_DIR" \
  --steps "$STEPS" \
  --horizon 1.0 \
  --fps "$FPS" \
  --decode-chunk-size 16 \
  --device cuda
