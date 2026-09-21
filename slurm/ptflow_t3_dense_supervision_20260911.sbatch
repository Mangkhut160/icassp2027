#!/usr/bin/env bash
#SBATCH --job-name=ptflow-t3-%j
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=14:00:00

# T3 (DART) training plus a chained b1_confirmatory eval: window 5 with
# decode-path supervision 0.5 and DINOv2 region loss 0.5.
# Usage: sbatch [--partition=<partition>] --export=ALL,RUN_NAME=<run>,SEED=<seed> slurm/ptflow_t3_dense_supervision_20260911.sbatch
# Environment: PYTHON, BASE (output root, default <repo>/outputs), MANIFEST
# (Scale-M manifest, required), LIBERO_ROOT, TORCH_HOME, HF_HOME.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
BASE="${BASE:-$ROOT/outputs}"
MANIFEST="${MANIFEST:?Set MANIFEST to a Scale-M manifest (tools/prepare_odeworld_ptflow_scale_m_manifest.py)}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME}"
SEED="${SEED:?Set SEED}"
OUT=$BASE/$RUN_NAME

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
cd "$ROOT"

"$PYTHON" tools/train_odeworld_ptflow_rollout_stability.py "$OUT" \
  --sampling-manifest "$MANIFEST" \
  --train-split scale_m_train --validation-split scale_m_validation \
  --window 5 --batch-size 2 --steps 26000 \
  --learning-rate 1e-5 --weight-decay 1e-4 \
  --stability-weight 0.0 --rollout-weight 1.0 \
  --decode-supervision-weight 0.5 \
  --region-loss-weight 0.5 --region-clusters 4 \
  --seed "$SEED" --validation-every 6500

CKPT=$("$PYTHON" - "$OUT" <<'PYEOF'
import json, sys
summary = json.loads(open(sys.argv[1] + '/training_summary.json').read())
print(summary['checkpoint'])
PYEOF
)
EVAL_OUT=${OUT}_eval_b1_$(date +%H%M)
"$PYTHON" tools/evaluate_odeworld_ptflow_checkpoint.py \
  --flow-checkpoint "$CKPT" \
  --split b1_confirmatory \
  --output-root "$EVAL_OUT"
echo "TRAIN_OK $RUN_NAME"
echo "EVAL_OK $EVAL_OUT"
