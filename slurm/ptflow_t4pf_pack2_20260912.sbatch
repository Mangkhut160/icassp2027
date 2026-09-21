#!/usr/bin/env bash
#SBATCH --job-name=ptflow-t4pf-%j
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00

# Rerun of the two T4b (identity regions + partflow 0.25) arms: ONE job ID,
# TWO GPUs, one training per GPU, then each GPU's chained 5-demo eval.
# Usage: sbatch [--partition=<partition>] [--nodelist=<nodes>] slurm/ptflow_t4pf_pack2_20260912.sbatch
# Environment: PYTHON, BASE (output root, default <repo>/outputs), MANIFEST
# (Scale-M manifest, required), LIBERO_ROOT, TORCH_HOME, HF_HOME.

set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
BASE="${BASE:-$ROOT/outputs}"
MANIFEST="${MANIFEST:?Set MANIFEST to a Scale-M manifest (tools/prepare_odeworld_ptflow_scale_m_manifest.py)}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="${TORCH_HOME:-$ROOT/.cache/torch}"
export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
cd "$ROOT"

train () { # name seed
  local NAME=$1 SEED=$2
  local OUT=$BASE/$NAME
  "$PYTHON" tools/train_odeworld_ptflow_rollout_stability.py "$OUT" \
    --sampling-manifest "$MANIFEST" \
    --train-split scale_m_train --validation-split scale_m_validation \
    --window 9 --batch-size 2 --steps 20000 \
    --learning-rate 1e-5 --weight-decay 1e-4 \
    --stability-weight 0.0 --rollout-weight 1.0 \
    --decode-supervision-weight 0.5 \
    --region-loss-weight 0.5 --region-clusters 4 --region-identity \
    --partflow-weight 0.25 \
    --seed "$SEED" --validation-every 5000 \
    > "${OUT}.trainlog" 2>&1
  echo "TRAIN_DONE $NAME rc=$?"
}

chained_eval () { # name
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

N1=ptflow_t4w9_idregpf_seed20260910_20260912
N2=ptflow_t4w9_idregpf_seed20260911_20260912

GPU0_STREAM () {
  export CUDA_VISIBLE_DEVICES=0
  train "$N1" 20260910
  chained_eval "$N1"
}

GPU1_STREAM () {
  export CUDA_VISIBLE_DEVICES=1
  train "$N2" 20260911
  chained_eval "$N2"
}

LOGP=${SLURM_JOB_NAME}-${SLURM_JOB_ID}
GPU0_STREAM > "${LOGP}.stream0.log" 2>&1 &
PA=$!
GPU1_STREAM > "${LOGP}.stream1.log" 2>&1 &
PB=$!
wait $PA $PB
echo "T4PF_ALL_DONE"
grep -h "TRAIN_DONE\|EVAL_DONE" "${LOGP}.stream"*.log
