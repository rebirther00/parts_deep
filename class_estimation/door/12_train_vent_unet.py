"""통풍구 세그멘테이션 U-Net 학습.

입력: 11_generate_vent_labels.py 산출물(정규화 정렬 RGB + CAD 슬롯 라벨).
- 여러 라벨 디렉토리(원본 + 극단 RGB 증강셋) 동시 사용 가능
- val은 첫 번째(원본) 디렉토리에서 클래스별 20%, 같은 인덱스는
  증강 디렉토리에서도 학습 제외 (증강 벤치마크 누수 방지)
- 현장풍 증강: 밝기/감마/가우시안 노이즈/블러

실행:
    python 12_train_vent_unet.py \
        --roots vent_labels/datasets vent_labels/datasets_aug \
                vent_labels/datasets_aug2
"""
import argparse
import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from attribute_utils import VentUNet

DOOR_DIR = os.path.dirname(os.path.abspath(__file__))
CROP = 256
SEED = 42

parser = argparse.ArgumentParser()
parser.add_argument('--roots', nargs='+',
                    default=['vent_labels/datasets'],
                    help='라벨 디렉토리 (첫 번째가 원본, val 기준)')
parser.add_argument('--out', default='attribute_models/vent_unet2.pth')
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--min_iou', type=float, default=0.8,
                    help='정합 IoU 미달 라벨 제외 임계')
args = parser.parse_args()
ROOTS = [os.path.join(DOOR_DIR, r) for r in args.roots]
CKPT = os.path.join(DOOR_DIR, args.out)
random.seed(SEED)
torch.manual_seed(SEED)


def build_items():
    # val: 원본 디렉토리에서 클래스별 20% (시드 고정 셔플)
    val_idx = {}
    meta0 = json.load(open(f'{ROOTS[0]}/meta.json'))
    for cls, rows in meta0.items():
        idxs = sorted(r['idx'] for r in rows)
        rng = random.Random(SEED + hash(cls) % 1000)
        rng.shuffle(idxs)
        val_idx[cls] = set(idxs[:max(1, len(idxs) // 5)])
    items = []
    for ri, root in enumerate(ROOTS):
        meta = json.load(open(f'{root}/meta.json'))
        for cls, rows in meta.items():
            good = {r['idx'] for r in rows if r['iou'] >= args.min_iou}
            for f in sorted(glob.glob(f'{root}/{cls}/aligned_*.png')):
                idx = os.path.basename(f)[8:12]
                if idx not in good:
                    continue
                is_val = idx in val_idx.get(cls, set())
                if is_val and ri > 0:
                    continue
                items.append({'img': f, 'cls': cls, 'root': root,
                              'dmask': f.replace('aligned_', 'dmask_'),
                              'split': 'val' if (is_val and ri == 0)
                              else 'train'})
    return items


class VentDataset(Dataset):
    def __init__(self, items, train=True):
        self.items = [i for i in items
                      if i['split'] == ('train' if train else 'val')]
        self.train = train
        self.vents = {}

    def vent(self, root, cls):
        key = (root, cls)
        if key not in self.vents:
            self.vents[key] = cv2.imread(f'{root}/{cls}/vent_label.png',
                                         cv2.IMREAD_GRAYSCALE)
        return self.vents[key]

    def __len__(self):
        return len(self.items) * 2

    def __getitem__(self, i):
        it = self.items[i % len(self.items)]
        img = cv2.imread(it['img'])
        dm = cv2.imread(it['dmask'], cv2.IMREAD_GRAYSCALE)
        vent = self.vent(it['root'], it['cls'])
        H, W = img.shape[:2]
        rng = np.random if self.train else np.random.RandomState(i)
        cy, cx = rng.randint(0, H), rng.randint(0, W)
        y0, x0 = cy - CROP // 2, cx - CROP // 2
        canvas = np.zeros((CROP, CROP, 3), np.uint8)
        lab = np.zeros((CROP, CROP), np.uint8)
        msk = np.zeros((CROP, CROP), np.uint8)
        sy0, sx0 = max(0, y0), max(0, x0)
        sy1, sx1 = min(H, y0 + CROP), min(W, x0 + CROP)
        if sy1 > sy0 and sx1 > sx0:
            canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                img[sy0:sy1, sx0:sx1]
            lab[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                vent[sy0:sy1, sx0:sx1]
            msk[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = dm[sy0:sy1, sx0:sx1]
        if self.train:
            if rng.rand() < 0.5:
                canvas, lab, msk = (np.ascontiguousarray(a[:, ::-1])
                                    for a in (canvas, lab, msk))
            if rng.rand() < 0.5:
                canvas, lab, msk = (np.ascontiguousarray(a[::-1])
                                    for a in (canvas, lab, msk))
            f = canvas.astype(np.float32)
            f = f * (0.6 + 0.8 * rng.rand()) + rng.randint(-40, 40)
            f = np.clip(f, 0, 255)
            f = 255 * (f / 255) ** (0.6 + 1.0 * rng.rand())
            if rng.rand() < 0.5:
                f += rng.randn(*f.shape) * (2 + 6 * rng.rand())
            canvas = np.clip(f, 0, 255).astype(np.uint8)
            if rng.rand() < 0.5:
                k = int(rng.choice([3, 3, 5]))
                canvas = cv2.GaussianBlur(canvas, (k, k), 0)
        x = torch.from_numpy(canvas.transpose(2, 0, 1)).float() / 255.
        y = torch.from_numpy((lab > 0).astype(np.float32))[None]
        m = torch.from_numpy((msk > 0).astype(np.float32))[None]
        return x, y, m


def dice_loss(logit, y, m):
    p = torch.sigmoid(logit) * m
    y = y * m
    inter = (p * y).sum((2, 3))
    return 1 - ((2 * inter + 1) / (p.sum((2, 3)) + y.sum((2, 3)) + 1)).mean()


if __name__ == '__main__':
    dev = 'cuda'
    items = build_items()
    n_tr = sum(1 for i in items if i['split'] == 'train')
    print(f'train={n_tr} val={len(items) - n_tr}', flush=True)
    tr = DataLoader(VentDataset(items, True), batch_size=32, shuffle=True,
                    num_workers=8, drop_last=True)
    va = DataLoader(VentDataset(items, False), batch_size=32, num_workers=8)
    net = VentUNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    best = 0
    for ep in range(args.epochs):
        net.train()
        tl = 0
        for x, y, m in tr:
            x, y, m = x.to(dev), y.to(dev), m.to(dev)
            logit = net(x)
            lb = (bce(logit, y) * m).sum() / m.sum().clamp(min=1)
            loss = lb + dice_loss(logit, y, m)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()
        sched.step()
        net.eval()
        inter = union = 0
        with torch.no_grad():
            for x, y, m in va:
                x, y, m = x.to(dev), y.to(dev), m.to(dev)
                p = (torch.sigmoid(net(x)) > 0.5).float() * m
                y = y * m
                inter += (p * y).sum().item()
                union += (p + y - p * y).sum().item()
        iou = inter / max(union, 1)
        print(f'ep{ep + 1}/{args.epochs} loss={tl / len(tr):.4f} '
              f'valIoU={iou:.4f}', flush=True)
        if iou > best:
            best = iou
            torch.save(net.state_dict(), CKPT)
    print(f'best val IoU={best:.4f} -> {CKPT}')
