#!/usr/bin/env bash
#SBATCH --job-name=ptflow-t3eval-%j
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

# Full b1_confirmatory (100-demo) evaluation for a T3 rollout-stability checkpoint.
# Usage: sbatch [--partition=<partition>] --export=ALL,RUN_NAME=<training run dir name> slurm/ptflow_t3_b1_100_eval_20260911.sbatch
# Environment: PYTHON, BASE (output root containing the training run, default
# <repo>/outputs), LIBERO_ROOT, TORCH_HOME, HF_HOME.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
BASE="${BASE:-$ROOT/outputs}"
RUN_NAME="${RUN_NAME:?Set RUN_NAME}"
TRAIN_DIR=$BASE/$RUN_NAME

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
cd "$ROOT"

CKPT=$("$PYTHON" - "$TRAIN_DIR" <<'PYEOF'
import json, sys
summary = json.loads(open(sys.argv[1] + '/training_summary.json').read())
print(summary['checkpoint'])
PYEOF
)
EVAL_OUT=${TRAIN_DIR}_eval_b1_100_$(date +%H%M)
"$PYTHON" tools/evaluate_odeworld_ptflow_checkpoint.py \
  --flow-checkpoint "$CKPT" \
  --split b1_confirmatory \
  --max-demos 100 \
  --output-root "$EVAL_OUT"
echo "EVAL100_OK $EVAL_OUT"
