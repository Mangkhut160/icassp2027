#!/usr/bin/env bash
#SBATCH --job-name=ptflow-t2-%j
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

# T2 dense-anchor latent-only training plus a chained b1_confirmatory eval.
# Usage: sbatch [--partition=<partition>] --export=ALL,RUN_NAME=<run>,SEED=<seed>,WINDOW=<w> slurm/ptflow_dense_anchor_t2_20260911.sbatch
# Environment: PYTHON, BASE (output root, default <repo>/outputs), MANIFEST
# (Scale-M manifest, required), LIBERO_ROOT, TORCH_HOME, HF_HOME.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
BASE="${BASE:-$ROOT/outputs}"
MANIFEST="${MANIFEST:?Set MANIFEST to a Scale-M manifest (tools/prepare_odeworld_ptflow_scale_m_manifest.py)}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME}"
SEED="${SEED:?Set SEED}"
WINDOW="${WINDOW:?Set WINDOW}"
OUT=$BASE/$RUN_NAME

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
cd "$ROOT"

"$PYTHON" tools/train_odeworld_ptflow_rollout_stability.py "$OUT" \
  --sampling-manifest "$MANIFEST" \
  --train-split scale_m_train --validation-split scale_m_validation \
  --window "$WINDOW" --batch-size 2 --steps 26000 \
  --learning-rate 1e-5 --weight-decay 1e-4 \
  --stability-weight 0.0 --rollout-weight 1.0 \
  --seed "$SEED" --validation-every 6500

# chained b1_100 evaluation on the selected checkpoint
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
