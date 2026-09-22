# Frozen Flows Forget: Diagnosing and Restoring Lost Motion in a Latent-Flow World Model

Source code for the ICASSP 2027 paper **"Frozen Flows Forget: Diagnosing and Restoring Lost Motion in a Latent-Flow World Model"**.

Latent world models that integrate a flow in a frozen self-supervised latent space train stably and cheaply, yet silently lose the property manipulation depends on most: motion. The pretrained flow never moves the manipulated object, and retraining it with latent-only losses only trades stillness for teleport-like motion. This repository traces that failure to the training signal — anchor-sparse, latent-only supervision never says where along the horizon change belongs — and repairs it with **DART (Decode-Augmented Rollout Training)**: retraining only the flow with decode-path supervision while keeping the representation frozen.

This work builds on [ODEWorld](https://github.com/Dstate/ODEWorld) ([arXiv:2607.27924](https://arxiv.org/abs/2607.27924)). DART retrains the original PT-Flow checkpoint; the DINOv2 encoder, the RAE decoder, and the goal predictor stay exactly as released.

## Scope

All training and evaluation in this repository is **offline open-loop prediction on official LIBERO demonstrations**. Nothing here starts LIBERO or MuJoCo, executes robot actions, calls `env.step()`, or measures task success rates. Rollout refers to visual latent prediction from demonstration frames, not to closed-loop robot control.

## Repository Layout

```
models/                  ODEWorld model code (PT-Flow, RAE, DINOv2 encoders, goal predictor)
dataloader/              Offline demonstration readers
demo_infer.py            Bundled image-and-language demo inference
scripts/
  evaluate_libero_hdf5.py        Offline evaluation of ODEWorld checkpoints on LIBERO HDF5
tools/
  train_odeworld_ptflow_rollout_stability.py   Trainer: latent-only parent, DART, and ablations
  evaluate_odeworld_ptflow_checkpoint.py       64-step rollout evaluation of a trained checkpoint
  compare_rollout_dynamics.py                  L1 / motion-concentration / total-motion comparison
  make_dart_rollout_gallery.py                 Rollout strip figures and GIFs
  build_odeworld_diagnostic_sampling_manifest.py    Frozen demo-sampling manifest builder
  prepare_odeworld_ptflow_scale_m_manifest.py       Task-balanced Scale-M train/val/test split builder
  libero_manifest.py                           Resolves manifest task_file paths against LIBERO_ROOT
  baseline_simvp/                              SimVP baseline (preparation, training, evaluation)
assets/
  libero/                 Five bundled LIBERO start/goal demo cases
  pretrained/             Destination for the downloaded checkpoints
manifests/                Frozen diagnostic sampling manifest
```

## Installation

```bash
git clone https://github.com/Mangkhut160/icassp2027.git
cd icassp2027
conda create -n dart python=3.10 -y
conda activate dart

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Pretrained Checkpoints

Download the original ODEWorld checkpoints into `assets/pretrained`:

```bash
mkdir -p assets/pretrained

for model in \
  ODEWorld-PT-Flow-LIBERO \
  ODEWorld-Goal-Predictor-LIBERO \
  ODEWorld-RAE-LIBERO
do
  hf download "ldxxx/${model}" --local-dir "assets/pretrained/${model}"
done
```

All three checkpoints are also available on [Hugging Face](https://huggingface.co/collections/ldxxx/odeworld). Each checkpoint bundles the DINOv2 backbone, so no separate encoder download is required. The first run also fetches the [DINOv2](https://github.com/facebookresearch/dinov2) code through `torch.hub`; set `TORCH_HOME` to control where the cache is written.

## Data

The experiments use the official LIBERO demonstration HDF5 files, which contain the raw RGB trajectories. Obtain them from the [LIBERO project](https://libero-project.github.io/) and place the five suites (`libero_10`, `libero_90`, `libero_goal`, `libero_object`, `libero_spatial`) under one root directory, referred to below as `$LIBERO_ROOT`.

## Evaluation Manifests

The pipeline uses two manifests, both generated from an offline evaluation pass of the original checkpoints over the full 6,500-demo corpus.

**Step 1 — evaluate the original checkpoints** (image-goal, language-goal, RAE reconstruction, and linear-interpolation reference rows). The manifest builder in Step 2 expects exactly three metric shards, so run the evaluation as three shards over the full corpus of 50 demos per task:

```bash
for shard in 0 1 2; do
  python scripts/evaluate_libero_hdf5.py \
    --dataset-root "$LIBERO_ROOT" \
    --output-dir outputs/libero_full_evaluation/full_6500_demos \
    --steps 64 --frame-transform rgb_vhflip \
    --max-demos-per-task 50 \
    --num-shards 3 --shard-index "$shard"
done
```

**Step 2 — build the frozen diagnostic sampling manifest** (defines the paper's evaluation splits: `b1_pilot` with 15 demos, `b1_confirmatory` with 100 demos, and `b2_confirmatory` with 30 demos):

```bash
python tools/build_odeworld_diagnostic_sampling_manifest.py \
  --source-root outputs/libero_full_evaluation/full_6500_demos \
  --output manifests/odeworld_diagnostic_sampling_v1.json \
  --selection-seed 20260903
```

**Step 3 — build the task-balanced Scale-M training split** (20 train / 5 validation / 5 test demos per task, 650 test demos over 130 tasks) from the frozen diagnostic manifest:

```bash
python tools/prepare_odeworld_ptflow_scale_m_manifest.py \
  --source manifests/odeworld_diagnostic_sampling_v1.json \
  --output manifests/scale_m_manifest.json
```

The shipped diagnostic manifest stores `task_file` paths relative to the LIBERO HDF5 root, so every tool that consumes manifest rows resolves them against `--libero-root` or the `LIBERO_ROOT` environment variable. Export it once to use the commands below unchanged:

```bash
export LIBERO_ROOT=/path/to/libero_hdf5
```

## Training

A single trainer covers every variant in the paper: `tools/train_odeworld_ptflow_rollout_stability.py`. It fine-tunes the original PT-Flow checkpoint on demonstration windows from the Scale-M manifest; the DINOv2 backbone and the RAE decoder stay frozen.

**Latent-only parent (the baseline DART improves on):**

```bash
python tools/train_odeworld_ptflow_rollout_stability.py outputs/parent_seed20260910 \
  --sampling-manifest manifests/scale_m_manifest.json \
  --train-split scale_m_train --validation-split scale_m_validation \
  --window 5 --batch-size 2 --steps 26000 \
  --learning-rate 1e-5 --weight-decay 1e-4 \
  --stability-weight 0.0 --rollout-weight 1.0 \
  --seed 20260910 --validation-every 6500
```

**DART (the paper method; decode-path and region supervision at 0.5 each):**

```bash
python tools/train_odeworld_ptflow_rollout_stability.py outputs/dart_seed20260910 \
  --sampling-manifest manifests/scale_m_manifest.json \
  --train-split scale_m_train --validation-split scale_m_validation \
  --window 5 --batch-size 2 --steps 26000 \
  --learning-rate 1e-5 --weight-decay 1e-4 \
  --stability-weight 0.0 --rollout-weight 1.0 \
  --decode-supervision-weight 0.5 \
  --region-loss-weight 0.5 --region-clusters 4 \
  --seed 20260910 --validation-every 6500
```

The checkpoint selected by lowest validation loss is recorded in `<output_dir>/training_summary.json`.

**Ablations:**

- Doubled loss weights: set `--decode-supervision-weight 1.0` or `--region-loss-weight 1.0`.
- Part-decomposition variant (identity-consistent parts, window 9, optional part-flow loss): add `--region-identity`, `--window 9`, and optionally `--partflow-weight 0.25`.

## Evaluation

**64-step image-goal rollout evaluation** of a trained checkpoint against the frozen splits:

```bash
python tools/evaluate_odeworld_ptflow_checkpoint.py \
  --flow-checkpoint outputs/dart_seed20260910/checkpoint \
  --split b1_confirmatory \
  --max-demos 100 \
  --output-root outputs/dart_seed20260910_eval_b1_100
```

Use `--split b1_pilot` for the pilot split. For the 650-demo scale benchmark, pass the Scale-M manifest and its test split:

```bash
python tools/evaluate_odeworld_ptflow_checkpoint.py \
  --flow-checkpoint outputs/dart_seed20260910/checkpoint \
  --sampling-manifest manifests/scale_m_manifest.json \
  --split scale_m_test \
  --max-demos 650 \
  --output-root outputs/dart_seed20260910_eval_test650
```

**Dynamics comparison** between a parent run and a DART run on the matched demo set (per-step L1, motion concentration `conc3`, total motion, strips, and GIFs):

```bash
python tools/compare_rollout_dynamics.py \
  --baseline-dir outputs/parent_seed20260910_eval_b1_100 \
  --candidate-dir outputs/dart_seed20260910_eval_b1_100 \
  --out-dir outputs/dynamics_parent_vs_dart
```

**Rollout gallery** (demo strip figures and animated GIFs):

```bash
python tools/make_dart_rollout_gallery.py \
  --baseline-dir outputs/parent_seed20260910_eval_b1_100 \
  --candidate-dir outputs/dart_seed20260910_eval_b1_100 \
  --out-dir outputs/dart_rollout_gallery
```

### Metrics

- **Per-step L1** catches wrong pixels (lower is better).
- **Motion concentration `conc3`** catches mistimed motion: the fraction of total pixel change carried by the three largest steps, compared against the ground truth (lower is better).
- **Total motion** catches missing motion: the sum of mean per-step pixel differences over the horizon (higher is closer to ground truth).

Pixel L1 alone rewards frozen predictions, which is why the concentration target was fixed before any DART training.

## SimVP Baseline

The pixel-space baseline lives in `tools/baseline_simvp/`:

```bash
python tools/baseline_simvp/prep_data.py --manifest manifests/scale_m_manifest.json --out outputs/simvp_data
python tools/baseline_simvp/train_simvp.py      --data-root outputs/simvp_data --out outputs/simvp_base
python tools/baseline_simvp/train_simvp_dart.py --data-root outputs/simvp_data --out outputs/simvp_dart
python tools/baseline_simvp/eval_simvp.py     --data-root outputs/simvp_data --ckpt outputs/simvp_base --out outputs/simvp_base_eval
```

`train_simvp.py` trains the base arm and `train_simvp_dart.py` the DART arm (L1 plus a DINOv2 feature loss); `openstl_provenance.txt` records the OpenSTL modules the implementation follows.

## Demo Inference

Run the bundled LIBERO image-and-language examples with the original public checkpoints:

```bash
python demo_infer.py --dataset libero
```

This produces goal-conditioned and language-predicted rollouts plus a PCA video for each of the five bundled cases from the static start/goal images in `assets/libero/`.

## Citation

If you use this code, please cite the underlying ODEWorld release:

```bibtex
@article{liu-niu2026odeworld,
  title   = {ODEWorld: A Continuous Predictive Architecture via Physical-Time Flow},
  author  = {Liu, Dongxiu and Niu, Haoyi and Cheng, Peng and Gao, Yuan and Kang, Xirui and Teng, Sangli and Sreenath, Koushil and Zhan, Xianyuan},
  journal = {arXiv preprint arXiv:2607.27924},
  year    = {2026}
}
```

## License

The pretrained checkpoints are released under Apache-2.0. LIBERO demonstrations are distributed under their own license by the LIBERO project.
