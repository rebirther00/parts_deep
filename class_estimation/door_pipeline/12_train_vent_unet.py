"""통풍구 세그멘테이션 U-Net 학습.

입력: 11_generate_vent_labels.py 산출물(정규화 정렬 RGB + CAD 슬롯 라벨).
- 분할: 클래스별 train/val/test = 70/15/15 (레거시 방법론과 동일).
  val = 체크포인트 선택, test = 평가 전용(13번 기본값), 분할은
  split.json에 저장·재사용.
- datasets_aug/aug2는 학습에 사용하지 않는다 — 강건성 평가 전용
  (프로젝트 방법론). 대신 온라인(on-the-fly) 광도 증강(밝기/감마/노이즈/
  블러)으로 현장 화질 변동에 대비한다.

실행:
    python 12_train_vent_unet.py
"""
import argparse
import glob
import json
import time
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
SPLIT_SEED = 42  # 분할 전용 시드 (split.json 재사용 시 무관)

parser = argparse.ArgumentParser()
parser.add_argument('--root', default='vent_labels/datasets',
                    help='라벨 디렉토리 (datasets_aug 계열은 평가 전용 — '
                         '학습에 넣지 말 것)')
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--seed', type=int, default=None,
                    help='가중치 초기화·증강 난수 시드. 미지정 시 랜덤 생성. '
                         '데이터 분할은 split.json에 고정되어 시드와 무관')
parser.add_argument('--min_iou', type=float, default=0.8,
                    help='정합 IoU 미달 라벨 제외 임계')
parser.add_argument('--no_promote', action='store_true',
                    help='배포 포인터(attribute_models/vent_unet.pth) 갱신 안 함')
args = parser.parse_args()
ROOT = os.path.join(DOOR_DIR, args.root)
SEED = SPLIT_SEED
if args.seed is None:
    args.seed = random.randint(0, 99999)
    print(f'시드 미지정 → 랜덤 시드 사용: {args.seed}')
random.seed(args.seed)
torch.manual_seed(args.seed)

# 레거시와 동일하게 run별 폴더로 관리, 배포 경로는 별도 포인터
RUN_NAME = f'vent_unet_seed{args.seed}'
RUN_DIR = os.path.join(DOOR_DIR, 'attribute_models', 'runs', RUN_NAME)
CKPT = os.path.join(RUN_DIR, 'model.pth')
CANONICAL = os.path.join(DOOR_DIR, 'attribute_models', 'vent_unet.pth')
os.makedirs(RUN_DIR, exist_ok=True)


def load_or_make_split(root, meta):
    """클래스별 train/val/test = 70/15/15 분할 (split.json 저장·재사용).

    주의: 내장 hash()는 실행마다 달라지므로 zlib.crc32 사용.
    """
    import zlib
    split_path = f'{root}/split.json'
    if os.path.exists(split_path):
        sp = json.load(open(split_path))
        if all('test' in v for v in sp.values()):
            return {c: {'val': set(v['val']), 'test': set(v['test'])}
                    for c, v in sp.items()}
        print('구버전 split.json(테스트 분할 없음) → 3분할로 재생성')
    split = {}
    for cls, rows in meta.items():
        idxs = sorted(r['idx'] for r in rows)
        rng = random.Random(SEED + zlib.crc32(cls.encode()) % 1000)
        rng.shuffle(idxs)
        n = len(idxs)
        n_val = max(1, int(n * 0.15))
        n_test = max(1, int(n * 0.15))
        split[cls] = {'val': set(idxs[:n_val]),
                      'test': set(idxs[n_val:n_val + n_test])}
    json.dump({c: {'val': sorted(v['val']), 'test': sorted(v['test'])}
               for c, v in split.items()},
              open(split_path, 'w'), indent=1)
    print(f'train/val/test 분할 생성/저장: {split_path}')
    return split


def build_items():
    meta = json.load(open(f'{ROOT}/meta.json'))
    split = load_or_make_split(ROOT, meta)
    items = []
    for cls, rows in meta.items():
        good = {r['idx'] for r in rows if r['iou'] >= args.min_iou}
        for f in sorted(glob.glob(f'{ROOT}/{cls}/aligned_*.png')):
            idx = os.path.basename(f)[8:12]
            if idx not in good:
                continue
            if idx in split[cls]['test']:
                continue  # test는 학습 과정에서 완전 격리
            s = 'val' if idx in split[cls]['val'] else 'train'
            items.append({'img': f, 'cls': cls, 'root': ROOT,
                          'dmask': f.replace('aligned_', 'dmask_'),
                          'split': s})
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
    info = {
        'run_name': RUN_NAME,
        'seed': args.seed, 'split_seed': SPLIT_SEED, 'root': args.root,
        'epochs': args.epochs, 'min_iou': args.min_iou,
        'best_val_iou': round(best, 4),
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    json.dump(info, open(os.path.join(RUN_DIR, 'train_info.json'), 'w'),
              indent=1)
    print(f'best val IoU={best:.4f} -> {CKPT}')
    if args.no_promote:
        print(f'배포 포인터 미갱신 (수동 갱신: cp {CKPT} {CANONICAL})')
    else:
        import shutil
        shutil.copy2(CKPT, CANONICAL)
        json.dump(info, open(CANONICAL.replace('.pth',
                                               '_train_info.json'), 'w'),
                  indent=1)
        print(f'배포 포인터 갱신: {CANONICAL} <- {RUN_NAME}')
