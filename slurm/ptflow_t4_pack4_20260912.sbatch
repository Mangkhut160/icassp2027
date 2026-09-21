#!/usr/bin/env bash
#SBATCH --job-name=ptflow-t4pack-%j
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00

# Packed T4 arms: ONE job ID, TWO GPUs, FOUR concurrent trainings
# (TWO per GPU, each ~11GB/80GB, num_workers=0), then chained 5-demo evals.
# T4 = window 9 (8 intermediate anchors against REAL in-between frames) +
# identity-consistent regions (k-means once on the start frame, labels propagated
# by nearest-centroid). T4b adds the partflow direction loss. Each stream runs its
# two trainings CONCURRENTLY, then its two chained evals concurrently.
# Usage: sbatch [--partition=<partition>] [--nodelist=<nodes>] slurm/ptflow_t4_pack4_20260912.sbatch
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

train () { # name seed partflow
  local NAME=$1 SEED=$2 PF=$3
  local OUT=$BASE/$NAME
  "$PYTHON" tools/train_odeworld_ptflow_rollout_stability.py "$OUT" \
    --sampling-manifest "$MANIFEST" \
    --train-split scale_m_train --validation-split scale_m_validation \
    --window 9 --batch-size 2 --steps 20000 \
    --learning-rate 1e-5 --weight-decay 1e-4 \
    --stability-weight 0.0 --rollout-weight 1.0 \
    --decode-supervision-weight 0.5 \
    --region-loss-weight 0.5 --region-clusters 4 --region-identity \
    --partflow-weight "$PF" \
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

N1=ptflow_t4w9_idreg_seed20260910_20260912
N2=ptflow_t4w9_idregpf_seed20260910_20260912
N3=ptflow_t4w9_idreg_seed20260911_20260912
N4=ptflow_t4w9_idregpf_seed20260911_20260912

GPU0_STREAM () {
  export CUDA_VISIBLE_DEVICES=0
  train "$N1" 20260910 0.0 & local P1=$!
  train "$N2" 20260910 0.25 & local P2=$!
  wait $P1 $P2
  chained_eval "$N1" & local E1=$!
  chained_eval "$N2" & local E2=$!
  wait $E1 $E2
}

GPU1_STREAM () {
  export CUDA_VISIBLE_DEVICES=1
  train "$N3" 20260911 0.0 & local P3=$!
  train "$N4" 20260911 0.25 & local P4=$!
  wait $P3 $P4
  chained_eval "$N3" & local E3=$!
  chained_eval "$N4" & local E4=$!
  wait $E3 $E4
}

LOGP=${SLURM_JOB_NAME}-${SLURM_JOB_ID}
GPU0_STREAM > "${LOGP}.stream0.log" 2>&1 &
PA=$!
GPU1_STREAM > "${LOGP}.stream1.log" 2>&1 &
PB=$!
wait $PA $PB
echo "PACK4_ALL_DONE"
grep -h "TRAIN_DONE\|EVAL_DONE" "${LOGP}.stream"*.log
