"""Evaluate the SimVP baseline under the exact ODEWorld open-loop protocol.

For each of the 650 Scale-M test demos: input = linear interpolation of the
start and goal frames (the same conditioning information ODEWorld's oracle
branch receives), prediction = 64 frames, references = the protocol target
frames. Reports per-step L1 (0-255), PSNR, motion concentration conc3, total
motion, and per-demo motion correlation with ground truth. Offline mock test.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from simvp_gsta import SimVP_Model  # noqa: E402


def predict(model, start, goal, steps, device, input_mode='interp'):
    """start/goal: (3,H,W) float [0,1]. Returns (steps,3,H,W) prediction."""
    if input_mode == 'interp':
        weights = (np.arange(steps) / (steps - 1)).astype(np.float32)
        source = (1.0 - weights[:, None, None, None]) * start + weights[:, None, None, None] * goal
    else:  # 'inpaint': endpoints only, middle zero
        source = np.zeros((steps, 3, start.shape[1], start.shape[2]), dtype=np.float32)
        source[0] = start
        source[-1] = goal
    tensor = torch.from_numpy(np.ascontiguousarray(source[None])).to(device)
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        pred = model(tensor)
    return pred[0].float().cpu().numpy().transpose(0, 2, 3, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--batch-model-kwargs', default='32,512')
    parser.add_argument('--input-mode', choices=['interp', 'inpaint'], default='interp')
    args = parser.parse_args()
    hid_s, hid_t = (int(x) for x in args.batch_model_kwargs.split(','))

    device = torch.device('cuda')
    model = SimVP_Model(in_shape=(64, 3, 256, 256), hid_S=hid_s, hid_T=hid_t,
                        N_S=4, N_T=4, model_type='gSTA').to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)['model'])

    split = Path(args.data_root) / 'scale_m_test'
    start = np.load(split / 'start.npy', mmap_mode='r')
    goal = np.load(split / 'goal.npy', mmap_mode='r')
    target = np.load(split / 'target.npy', mmap_mode='r')

    per_step_l1, per_step_psnr, conc3_list, conc3_gt_list, motion_list, motion_gt_list = [], [], [], [], [], []
    for index in range(start.shape[0]):
        start_i = np.asarray(start[index], dtype=np.float32)
        goal_i = np.asarray(goal[index], dtype=np.float32)
        if start_i.ndim == 4:  # prep stored start/goal with a leading [None] axis
            start_i, goal_i = start_i[0], goal_i[0]
        start_i = start_i / 255.0
        goal_i = goal_i / 255.0
        target_i = np.asarray(target[index], dtype=np.float32) / 255.0
        pred = predict(model, start_i.transpose(2, 0, 1), goal_i.transpose(2, 0, 1),
                       target.shape[1], device, args.input_mode)
        diff = np.abs(pred - target_i)                       # (64,H,W,3)
        l1 = diff.mean(axis=(1, 2, 3)) * 255.0                    # per-step L1, 0-255
        mse = (diff ** 2).mean(axis=(1, 2, 3))
        psnr = -10.0 * np.log10(np.maximum(mse, 1e-10))
        per_step_l1.append(l1)
        per_step_psnr.append(psnr)

        d_pred = np.abs(np.diff(pred, axis=0)).mean(axis=(1, 2, 3)) * 255.0
        d_gt = np.abs(np.diff(target_i, axis=0)).mean(axis=(1, 2, 3)) * 255.0
        top3 = np.sort(d_pred)[-3:].sum()
        conc3_list.append(top3 / max(d_pred.sum(), 1e-8))
        conc3_gt_list.append(np.sort(d_gt)[-3:].sum() / max(d_gt.sum(), 1e-8))
        motion_list.append(d_pred.sum())
        motion_gt_list.append(d_gt.sum())
        if (index + 1) % 100 == 0:
            print(f'eval {index + 1}/{start.shape[0]}', flush=True)

    per_step_l1 = np.stack(per_step_l1)
    per_step_psnr = np.stack(per_step_psnr)
    conc3_gt = float(np.mean(conc3_gt_list))
    from scipy import stats as sp_stats
    spearman = sp_stats.spearmanr(motion_list, motion_gt_list)
    pearson = sp_stats.pearsonr(motion_list, motion_gt_list)

    summary = {
        'schema': 'baseline_simvp_eval_v1',
        'classification': 'mock test',
        'checkpoint': args.ckpt,
        'input_mode': args.input_mode,
        'demo_count': int(start.shape[0]),
        'l1_mean': float(per_step_l1.mean()),
        'l1_median': float(np.median(per_step_l1.mean(axis=1))),
        'l1_endpoint': float(per_step_l1[:, -1].mean()),
        'psnr_mean': float(per_step_psnr.mean()),
        'psnr_endpoint': float(per_step_psnr[:, -1].mean()),
        'conc3_pred': float(np.mean(conc3_list)),
        'conc3_gt': conc3_gt,
        'total_motion_pred': float(np.mean(motion_list)),
        'total_motion_gt': float(np.mean(motion_gt_list)),
        'motion_spearman_vs_gt': float(spearman.statistic),
        'motion_pearson_vs_gt': float(pearson[0]),
        'per_step_l1_mean': per_step_l1.mean(axis=0).tolist(),
        'scope': {'env_step_calls': 0, 'libero_or_mujoco_started': False,
                  'success_rate_measured': False},
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2))


if __name__ == '__main__':
    main()
