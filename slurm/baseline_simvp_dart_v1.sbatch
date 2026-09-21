#!/bin/bash
#SBATCH --job-name=simvp-dart
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

# DART arm of the SimVP baseline: identical to baseline_simvp_v1 except the loss
# (L1 + 0.5 * DINOv2 feature loss). Offline HDF5 prediction only (mock test).
# Usage: sbatch [--partition=<partition>] [--nodelist=<nodes>] slurm/baseline_simvp_dart_v1.sbatch
# Environment: PYTHON, OUT (run output root, default <repo>/outputs/baseline_simvp_dart_v1),
# DATA (prepared SimVP dataset root, required; see tools/baseline_simvp/prep_data.py),
# TORCH_HOME (DINOv2 hub cache).

set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$REPO/tools/baseline_simvp"
PYTHON="${PYTHON:-python3}"
OUT="${OUT:-$REPO/outputs/baseline_simvp_dart_v1}"
DATA="${DATA:?Set DATA to the prepared SimVP dataset root (tools/baseline_simvp/prep_data.py)}"
mkdir -p "$OUT"

export TORCH_HOME="${TORCH_HOME:-$REPO/.cache/torch}"

echo "=== $(date) job $SLURM_JOB_ID on $(hostname) node-local GPUs: $CUDA_VISIBLE_DEVICES ==="

echo "=== train two DART-arm seeds in parallel (one GPU each, 990-minute budget each) ==="
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$ROOT/train_simvp_dart.py" --data-root "$DATA" \
  --out "$OUT/seed20260914" --seed 20260914 --max-minutes 990 \
  > "$OUT/train_seedA.log" 2>&1 &
PID_A=$!
CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$ROOT/train_simvp_dart.py" --data-root "$DATA" \
  --out "$OUT/seed20260915" --seed 20260915 --max-minutes 990 \
  > "$OUT/train_seedB.log" 2>&1 &
PID_B=$!
wait $PID_A
wait $PID_B
echo "both DART trainings finished"

echo "=== evaluate both DART seeds on the 650-demo test split ==="
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$ROOT/eval_simvp.py" --data-root "$DATA" \
  --ckpt "$OUT/seed20260914/best.pt" --out "$OUT/eval_seed20260914"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$ROOT/eval_simvp.py" --data-root "$DATA" \
  --ckpt "$OUT/seed20260915/best.pt" --out "$OUT/eval_seed20260915"

echo "=== done $(date) ==="
classification=$'mock test\nenv_step_calls=0\nlibero_or_mujoco_started=false\nsuccess_rate_measured=false'
echo "$classification"
