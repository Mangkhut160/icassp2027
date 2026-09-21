#!/usr/bin/env bash
#SBATCH --job-name=dart-gallery-%j
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

# Before/after rollout gallery from stored b1_100 evaluations:
# latent-only parent (Scale-M seed20260910) vs DART (ptflow_t3w5 seed20260910).
# CPU-only rendering from saved predictions; no model inference, no environment.
# Usage: sbatch [--partition=<cpu-partition>] --export=ALL,BASELINE_DIR=<dir>,CANDIDATE_DIR=<dir>,OUT_DIR=<dir> slurm/dart_rollout_gallery_20260917.sbatch
# Environment: PYTHON; BASELINE_DIR, CANDIDATE_DIR and OUT_DIR are required.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
BASELINE_DIR="${BASELINE_DIR:?Set BASELINE_DIR to a parent b1_100 eval output}"
CANDIDATE_DIR="${CANDIDATE_DIR:?Set CANDIDATE_DIR to a DART b1_100 eval output}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR to the gallery output directory}"

cd "$ROOT"
"$PYTHON" tools/make_dart_rollout_gallery.py \
  --baseline-dir "$BASELINE_DIR" \
  --candidate-dir "$CANDIDATE_DIR" \
  --out-dir "$OUT_DIR"

echo "GALLERY_DONE"
