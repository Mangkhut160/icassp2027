#!/usr/bin/env python3
"""Train a SimVP (gSTA, OpenSTL) goal-conditioned video predictor on LIBERO demos.

Two arms:
  base : Charbonnier/L1 pixel loss on the 64 target frames.
  dart : base + 0.5 * DINOv2 feature loss on a random subset of frames
         (the DART decode-path supervision idea, architecture-transferred).

Scope: mock test. Offline HDF5 prediction training; no env, no actions.
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT / 'tools' / 'baseline_simvp'))

from simvp_gsta import SimVP_Model  # noqa: E402


def load_split(split, data_root):
    index = json.load(open(Path(data_root) / 'index.json'))
    entries = [dict(e, split_dirname=split) for e in index[split] if e.get('frames', 0) >= 66]
    return entries


def load_demo(path):
    return np.load(path)


class LiberoDataset(torch.utils.data.Dataset):
    """(frame0, frameT-1) conditioning -> 64 frames at linspace(1, T-1, 64)."""

    def __init__(self, entries, data_root, hflip=0.5):
        self.entries = entries
        self.data_root = Path(data_root)
        self.hflip = hflip

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        frames = load_demo(self.data_root / entry['split_dirname'] / entry['file'])
        t_total = frames.shape[0]
        target_idx = np.rint(np.linspace(1, t_total - 1, 64)).astype(int)
        start = torch.from_numpy(frames[0]).permute(2, 0, 1).float() / 255.0
        goal = torch.from_numpy(frames[-1]).permute(2, 0, 1).float() / 255.0
        targets = torch.from_numpy(frames[target_idx]).permute(0, 3, 1, 2).float() / 255.0
        if random.random() < self.hflip:
            start, goal, targets = start.flip(-1), goal.flip(-1), targets.flip(-1)
        return start, goal, targets


class CondSimVP(nn.Module):
    """SimVP with broadcast (start, goal) conditioning at every input slot."""

    def __init__(self, out_frames=64, size=128, hid_S=16, hid_T=512, N_S=4, N_T=8):
        super().__init__()
        self.out_frames = out_frames
        self.net = SimVP_Model(in_shape=(out_frames, 6, size, size), out_channels=3,
                               hid_S=hid_S, hid_T=hid_T, N_S=N_S, N_T=N_T)

    def forward(self, start, goal):
        cond = torch.cat([start, goal], dim=1)  # (B, 6, H, W)
        x = cond.unsqueeze(1).repeat(1, self.out_frames, 1, 1, 1)
        return self.net(x)


class DinoLoss(nn.Module):
    """L1 between frozen DINOv2 patch features of predicted and target frames.

    Same pretrained backbone family as the stack's encoder (ViT-L/14 reg4),
    loaded from the local torch hub cache (TORCH_HOME).
    """

    def __init__(self, size=112):
        super().__init__()
        hub_dir = Path(torch.hub.get_dir())
        repo = str(hub_dir / 'facebookresearch_dinov2_main')
        ckpt = hub_dir / 'checkpoints' / 'dinov2_vitl14_reg4_pretrain.pth'
        self.encoder = torch.hub.load(repo, 'dinov2_vitl14_reg',
                                      source='local', pretrained=False)
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        assert not missing, f'missing keys: {missing[:5]}'
        self.encoder.requires_grad_(False).eval()
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.size = size

    def features(self, frames):
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b * t, c, h, w)
        flat = F.interpolate(flat, size=(self.size, self.size), mode='bicubic', align_corners=False)
        flat = (flat - self.mean.to(flat.device)) / self.std.to(flat.device)
        z = self.encoder(flat)  # (b*t, dim)
        return F.normalize(z, dim=-1)

    def forward(self, pred, target, n_frames=4):
        t = pred.shape[1]
        idx = sorted(random.sample(range(t), min(n_frames, t)))
        return F.l1_loss(self.features(pred[:, idx]), self.features(target[:, idx]))


def charbonnier(pred, target):
    diff = pred - target
    return torch.sqrt(diff * diff + 1e-6).mean()


def evaluate_val(model, val_loader, device):
    model.eval()
    losses = []
    with torch.inference_mode():
        for start, goal, targets in val_loader:
            pred = model(start.to(device), goal.to(device))
            losses.append(charbonnier(pred, targets.to(device)).item())
    model.train()
    return float(np.mean(losses))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['base', 'dart'], required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--data-root', required=True,
                        help='Directory produced by the SimVP data preparation step '
                             '(contains index.json, train/, eval/)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--val-frac', type=float, default=0.05)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()

    torch.manual_seed(20260914)
    random.seed(20260914)
    np.random.seed(20260914)
    device = torch.device('cuda')

    entries = load_split('train', args.data_root)
    random.shuffle(entries)
    n_val = max(1, int(len(entries) * args.val_frac))
    val_entries, train_entries = entries[:n_val], entries[n_val:]
    print(f'train demos={len(train_entries)} val demos={len(val_entries)}')

    train_set = LiberoDataset(train_entries, args.data_root)
    val_set = LiberoDataset(val_entries, args.data_root, hflip=0.0)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=8,
        pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch, num_workers=4, persistent_workers=True)

    model = CondSimVP().to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'CondSimVP params: {n_params:.1f}M, arm={args.arm}')

    dino_loss = DinoLoss().to(device) if args.arm == 'dart' else None
    if dino_loss is not None:
        dino_loss.eval()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config['params_m'] = round(n_params, 2)
    config['scope'] = 'mock test; offline HDF5 training, no env, no actions'
    json.dump(config, open(out / 'config.json', 'w'), indent=2)

    if args.smoke:
        args.epochs = 1
        iters = 3

    model.train()
    best = float('inf')
    for epoch in range(args.epochs):
        for it, (start, goal, targets) in enumerate(train_loader):
            start = start.to(device, non_blocking=True)
            goal = goal.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = model(start, goal)
                loss = charbonnier(pred.float(), targets)
                if dino_loss is not None:
                    loss = loss + 0.5 * dino_loss(pred.float(), targets)
            loss.backward()
            optimizer.step()
            if it % 50 == 0:
                print(f'epoch {epoch} iter {it}/{len(train_loader)} loss {loss.item():.4f}', flush=True)
            if args.smoke and it >= iters:
                break
        val = evaluate_val(model, val_loader, device)
        scheduler.step()
        print(f'epoch {epoch} val_charbonnier {val:.4f}', flush=True)
        if val < best:
            best = val
            torch.save({'model': model.state_dict(), 'val': val, 'epoch': epoch},
                       out / 'best.pt')
        torch.save({'model': model.state_dict(), 'val': val, 'epoch': epoch},
                   out / 'last.pt')
    json.dump({'best_val': best}, open(out / 'train_summary.json', 'w'))
    print('training done, best val', best)


if __name__ == '__main__':
    main()
