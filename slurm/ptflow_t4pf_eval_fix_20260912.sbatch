#!/usr/bin/env bash
#SBATCH --job-name=ptflow-t4pfev-%j
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

# Eval-only rerun for the two T4b checkpoints when a previous attempt left
# empty eval dirs behind (the evaluation refuses existing output roots).
# Usage: sbatch [--partition=<partition>] [--nodelist=<nodes>] slurm/ptflow_t4pf_eval_fix_20260912.sbatch
# Environment: PYTHON, BASE (output root containing the training runs, default
# <repo>/outputs), LIBERO_ROOT, TORCH_HOME, HF_HOME.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
BASE="${BASE:-$ROOT/outputs}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
cd "$ROOT"

eval_one () { # name
  local NAME=$1
  local OUT=$BASE/$NAME
  local CKPT
  CKPT=$("$PYTHON" - "$OUT" <<'PYEOF'
import json, sys
print(json.loads(open(sys.argv[1] + '/training_summary.json').read())['checkpoint'])
PYEOF
)
  "$PYTHON" tools/evaluate_odeworld_ptflow_checkpoint.py \
    --flow-checkpoint "$CKPT" --split b1_confirmatory \
    --output-root "${OUT}_eval_b1" > "${OUT}.evallog" 2>&1
  echo "EVAL_DONE $NAME rc=$?"
}

eval_one ptflow_t4w9_idregpf_seed20260910_20260912 &
E1=$!
eval_one ptflow_t4w9_idregpf_seed20260911_20260912 &
E2=$!
wait $E1 $E2
echo "T4PF_EVAL_OK"
