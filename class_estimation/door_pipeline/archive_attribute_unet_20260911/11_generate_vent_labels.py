"""통풍구 세그멘테이션 학습 라벨 자동 생성 (수작업 라벨링 불필요).

각 학습 이미지(rgb/depth/mask)를 렉티파이 → 축 정렬 → CAD 템플릿 프레임에
정합하고, CAD 슬롯 마스크를 라벨로 공유한다. 이미지별 산출물:
- aligned_XXXX.png : 템플릿 프레임으로 정규화된 RGB (학습 입력)
- dmask_XXXX.png   : 도어 마스크
- vent_label.png   : CAD 슬롯 마스크 (클래스당 1장 공유)
- meta.json        : 정합 실루엣 IoU / 플립 판정 (IoU<0.8은 학습 제외 권장)

실행:
    python 11_generate_vent_labels.py                       # datasets
    python 11_generate_vent_labels.py --base datasets_aug \
        --out vent_labels/datasets_aug
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from attribute_utils import (axis_align, load_templates, rectify, CLASSES)

DOOR_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--base', default='datasets',
                    help='입력 데이터셋 디렉토리 (DOOR 기준 상대경로)')
parser.add_argument('--out', default='vent_labels/datasets',
                    help='출력 디렉토리 (DOOR 기준 상대경로)')
args = parser.parse_args()
BASE = os.path.join(DOOR_DIR, args.base)
OUT = os.path.join(DOOR_DIR, args.out)
T = load_templates()


def edge_density(rgb_aligned, mask_aligned):
    gray = cv2.cvtColor(rgb_aligned, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 140)
    m_er = cv2.erode(mask_aligned, np.ones((7, 7), np.uint8))
    edges[m_er == 0] = 0
    return cv2.blur((edges > 0).astype(np.float32), (25, 25))


def register(rgb, depth, mask, cls):
    """렉티파이 → 축정렬 → 템플릿 크기 리사이즈 → 플립 판정(에지밀도 상관)."""
    ortho, omask = rectify(rgb, depth, mask)
    ao, am = axis_align(ortho, omask)
    sil = T[f'{cls}_sil']
    slot = T[f'{cls}_slot']
    th, tw = sil.shape
    if (ao.shape[1] > ao.shape[0]) != (tw > th):
        ao = cv2.rotate(ao, cv2.ROTATE_90_CLOCKWISE)
        am = cv2.rotate(am, cv2.ROTATE_90_CLOCKWISE)
    ao = cv2.resize(ao, (tw, th))
    am = cv2.resize(am, (tw, th), interpolation=cv2.INTER_NEAREST)
    iou = ((am > 0) & (sil > 0)).sum() / ((am > 0) | (sil > 0)).sum()
    dens = edge_density(ao, am)
    slot_b = slot > 0
    best = None
    for fl in ['none', 'h', 'v', 'hv']:
        d = dens
        if 'h' in fl:
            d = d[:, ::-1]
        if 'v' in fl:
            d = d[::-1, :]
        inside = d[slot_b].mean() if slot_b.any() else 0.0
        outside = d[(sil > 0) & ~slot_b].mean()
        score = inside - outside
        if best is None or score > best[1]:
            best = (fl, score)
    fl = best[0]
    if 'h' in fl:
        ao, am = ao[:, ::-1], am[:, ::-1]
    if 'v' in fl:
        ao, am = ao[::-1, :], am[::-1, :]
    return (np.ascontiguousarray(ao), np.ascontiguousarray(am),
            float(iou), fl)


if __name__ == '__main__':
    meta = {}
    for cls in CLASSES:
        os.makedirs(f'{OUT}/{cls}', exist_ok=True)
        rows = []
        for mp in sorted(glob.glob(f'{BASE}/{cls}/mask_*.png')):
            idx = os.path.basename(mp)[5:9]
            rgb = cv2.imread(f'{BASE}/{cls}/rgb_{idx}.png')
            mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            depth = cv2.imread(f'{BASE}/{cls}/depth_{idx}.png',
                               cv2.IMREAD_UNCHANGED)
            if rgb is None or mask is None or depth is None:
                continue
            try:
                ao, am, iou, fl = register(rgb, depth, mask, cls)
            except Exception as e:
                print(f'  {cls} {idx} FAIL {e}')
                continue
            cv2.imwrite(f'{OUT}/{cls}/aligned_{idx}.png', ao)
            cv2.imwrite(f'{OUT}/{cls}/dmask_{idx}.png', am)
            rows.append({'idx': idx, 'iou': round(iou, 3), 'flip': fl})
        cv2.imwrite(f'{OUT}/{cls}/vent_label.png',
                    load_templates()[f'{cls}_slot'])
        meta[cls] = rows
        ious = [r['iou'] for r in rows] or [0]
        print(f'{cls}: n={len(rows)} IoU med={np.median(ious):.3f}',
              flush=True)
    json.dump(meta, open(f'{OUT}/meta.json', 'w'))
