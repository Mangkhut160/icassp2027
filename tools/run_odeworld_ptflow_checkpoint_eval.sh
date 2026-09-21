#!/usr/bin/env bash
# Wrapper for tools/evaluate_odeworld_ptflow_checkpoint.py.
#
# Environment overrides:
#   PYTHON      Python interpreter (default: python3)
#   TORCH_HOME  Torch hub cache for the DINOv2 code/weights (default: repo .cache/torch)
#   HF_HOME     Hugging Face cache (default: repo .cache/huggingface)
#   LIBERO_ROOT LIBERO HDF5 root used to resolve relative manifest task_file paths
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"

OUTPUT_DIR=${1:?Usage: $0 OUTPUT_DIR --flow-checkpoint CHECKPOINT [args]}
if [[ -e "$OUTPUT_DIR" || -L "$OUTPUT_DIR" ]]; then
  echo "Refusing existing output directory: $OUTPUT_DIR" >&2
  exit 2
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME HF_HOME
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
cd "$ROOT"
echo "classification=mock test env_step_calls=0 libero_or_mujoco_started=false success_rate_measured=false"
exec "$PYTHON" tools/evaluate_odeworld_ptflow_checkpoint.py --output-root "$OUTPUT_DIR" "${@:2}"
