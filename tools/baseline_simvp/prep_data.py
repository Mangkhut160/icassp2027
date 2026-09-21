"""Preprocess LIBERO demos for the SimVP pixel baseline (offline HDF5, mock test).

For every demo in the Scale-M manifest splits, store exactly the tensors the
ODEWorld open-loop protocol uses: start frame, goal frame (last), and the 64
target frames at indices rint(linspace(1, T-1, 64)), all rgb_vhflip + INTER_AREA
resized to 256x256 uint8. Convention verified against
scripts/evaluate_libero_hdf5.py and the registered 650-demo eval artifacts.
"""

import argparse
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.libero_manifest import resolve_task_file

STEPS = 64


def process_one(args):
    hdf5_path, demo = args
    with h5py.File(hdf5_path, 'r') as handle:
        frames = handle[f'data/{demo}/obs/agentview_rgb'][:]
    t = frames.shape[0]

    def prep(index):
        frame = frames[index][::-1, ::-1]
        return cv2.resize(np.ascontiguousarray(frame), (256, 256), interpolation=cv2.INTER_AREA)

    indices = np.rint(np.linspace(1, t - 1, STEPS)).astype(np.int64)
    return (prep(0)[None], prep(t - 1)[None],
            np.stack([prep(i) for i in indices], axis=0))


def build_split(manifest, split_name, out_dir, workers, libero_root):
    rows = manifest['splits'][split_name]
    jobs = [(resolve_task_file(row['task_file'], libero_root), row['demo']) for row in rows]
    out_dir.mkdir(parents=True, exist_ok=True)
    starts, goals, targets = [], [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (start, goal, target) in enumerate(pool.map(process_one, jobs, chunksize=4)):
            starts.append(start); goals.append(goal); targets.append(target)
            if (i + 1) % 200 == 0:
                print(f'{split_name}: {i + 1}/{len(jobs)}', flush=True)
    np.save(out_dir / 'start.npy', np.stack(starts))
    np.save(out_dir / 'goal.npy', np.stack(goals))
    np.save(out_dir / 'target.npy', np.stack(targets))
    (out_dir / 'ids.json').write_text(json.dumps(
        [f"{r['suite']}/{r['task']}/{r['demo']}" for r in rows]))
    print(f'{split_name}: wrote npy arrays ({len(jobs)} demos)', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--splits', nargs='+', default=['scale_m_train', 'scale_m_validation', 'scale_m_test'])
    parser.add_argument('--workers', type=int, default=32)
    parser.add_argument('--libero-root', default=None,
                        help='LIBERO HDF5 root for manifests with relative task_file paths '
                             '(defaults to the LIBERO_ROOT environment variable)')
    args = parser.parse_args()

    manifest = json.load(open(args.manifest))
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        target = out_root / split
        if (target / 'target.npy').exists():
            print(f'{split}: exists, skipping', flush=True)
            continue
        build_split(manifest, split, target, args.workers, args.libero_root)

    meta = {'manifest': args.manifest,
            'manifest_sha256': hashlib.sha256(open(args.manifest, 'rb').read()).hexdigest(),
            'steps': STEPS, 'resolution': [256, 256],
            'transform': 'rgb_vhflip+INTER_AREA',
            'classification': 'mock test',
            'scope': {'env_step_calls': 0, 'libero_or_mujoco_started': False,
                      'success_rate_measured': False}}
    (out_root / 'prep_meta.json').write_text(json.dumps(meta, indent=2))
    print('prep complete', flush=True)


if __name__ == '__main__':
    main()
