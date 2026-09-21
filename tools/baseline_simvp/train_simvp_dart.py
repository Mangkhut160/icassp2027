"""Train the SimVP-gSTA goal-conditioned pixel baseline, DART arm (offline, mock test).

Identical to train_simvp.py (same data, model, schedule, seeds, wall-clock budget);
the ONLY difference is the loss: L1 + 0.5 * DINOv2 feature loss on 4 random target
frames per step (the DART decode-path supervision idea transferred to pixel space).
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
from simvp_gsta import SimVP_Model  # noqa: E402
from train_simvp_libero import DinoLoss  # noqa: E402


class GoalConditioned(Dataset):
    def __init__(self, split_dir, steps=64):
        split_dir = Path(split_dir)
        self.start = np.load(split_dir / 'start.npy', mmap_mode='r')
        self.goal = np.load(split_dir / 'goal.npy', mmap_mode='r')
        self.target = np.load(split_dir / 'target.npy', mmap_mode='r')
        self.steps = steps

    def __len__(self):
        return self.start.shape[0]

    def __getitem__(self, index):
        weights = (np.arange(self.steps) / (self.steps - 1)).astype(np.float32)
        start = np.asarray(self.start[index], dtype=np.float32) / 255.0
        goal = np.asarray(self.goal[index], dtype=np.float32) / 255.0
        frames = (1.0 - weights[:, None, None, None]) * start \
            + weights[:, None, None, None] * goal
        source = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
        target = np.asarray(self.target[index], dtype=np.float32).transpose(0, 3, 1, 2) / 255.0
        return source, target


def to_tensor(batch, device):
    source, target = batch
    return (source.to(device, non_blocking=True),
            target.to(device, non_blocking=True))


def evaluate_l1(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for batch in loader:
            source, target = to_tensor(batch, device)
            pred = model(source)
            total += (pred.float() - target).abs().mean().item() * source.shape[0]
            count += source.shape[0]
    return total / max(count, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, default=20260914)
    parser.add_argument('--epochs', type=int, default=130)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--hid-s', type=int, default=32)
    parser.add_argument('--hid-t', type=int, default=512)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--max-minutes', type=int, default=660)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda')
    dino_loss = DinoLoss().to(device)
    dino_loss.eval()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_set = GoalConditioned(Path(args.data_root) / 'scale_m_train')
    val_set = GoalConditioned(Path(args.data_root) / 'scale_m_validation')

    def make_loaders(batch):
        train_loader = DataLoader(train_set, batch_size=batch, shuffle=True,
                                  num_workers=args.workers, pin_memory=True, drop_last=True,
                                  persistent_workers=True)
        val_loader = DataLoader(val_set, batch_size=batch, shuffle=False,
                                num_workers=2, pin_memory=True)
        return train_loader, val_loader

    while args.batch_size >= 1:
        try:
            train_loader, val_loader = make_loaders(args.batch_size)
            probe = next(iter(train_loader))
            del probe
            break
        except torch.cuda.OutOfMemoryError:
            print(f'OOM at batch {args.batch_size}, halving', flush=True)
            args.batch_size //= 2

    model = SimVP_Model(in_shape=(64, 3, 256, 256), hid_S=args.hid_s, hid_T=args.hid_t,
                        N_S=4, N_T=4, model_type='gSTA').to(device)
    params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader))

    config = vars(args) | {'params_millions': round(params / 1e6, 2),
                           'model': 'SimVP-gSTA (vendored from OpenSTL, Apache-2.0)',
                           'loss': 'L1 + 0.5 * DINOv2 feature loss on 4 random target frames',
                           'arm': 'dart',
                           'dino_backbone': 'dinov2_vitl14_reg4 (frozen, local cache)',
                           'conditioning': 'start+goal via deterministic 64-step linear interpolation',
                           'classification': 'mock test',
                           'scope': {'env_step_calls': 0, 'libero_or_mujoco_started': False,
                                     'success_rate_measured': False}}
    (out / 'config.json').write_text(json.dumps(config, indent=2))

    best_val, history, started = float('inf'), [], time.time()
    iters = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss, batches = 0.0, 0
        for batch in train_loader:
            source, target = to_tensor(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(source)
                loss = (pred.float() - target).abs().mean()
                loss = loss + 0.5 * dino_loss(pred.float(), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item(); batches += 1; iters += 1
        val = evaluate_l1(model, val_loader, device)
        history.append({'epoch': epoch, 'train_l1': epoch_loss / max(batches, 1),
                        'val_l1_0_1': val, 'val_l1_255': val * 255.0})
        print(f'epoch {epoch:3d} train {epoch_loss / max(batches, 1):.4f} '
              f'val(255) {val * 255.0:.2f} elapsed {(time.time() - started) / 60:.0f}m', flush=True)
        (out / 'history.json').write_text(json.dumps(history, indent=1))
        if val < best_val:
            best_val = val
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_l1_0_1': val},
                       out / 'best.pt')
        if (time.time() - started) / 60 > args.max_minutes:
            print('time budget reached, stopping', flush=True)
            break
    (out / 'final_meta.json').write_text(json.dumps(
        {'best_val_l1_255': best_val * 255.0, 'epochs_done': len(history), 'iters': iters,
         'wall_minutes': round((time.time() - started) / 60, 1),
         'model_sha256': hashlib.sha256((out / 'best.pt').read_bytes()).hexdigest()}, indent=2))
    print('training complete', flush=True)


if __name__ == '__main__':
    main()
