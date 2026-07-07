"""속성 파이프라인 평가 — datasets / datasets_aug / datasets_factory.

배포형 원시 경로(렉티파이→U-Net→방향 정합 템플릿 매칭→종횡비 결합)를
N프레임 부트스트랩(그룹 다수결 + 종횡비 중앙값)으로 평가한다.

현장(datasets_factory)은 마스크가 없으므로 MobileSAM으로 자동 생성
(센터 포인트 프롬프트, factory_masks/에 캐시).

실행:
    python 13_evaluate_attribute_pipeline.py                # datasets
    python 13_evaluate_attribute_pipeline.py --base datasets_factory
    python 13_evaluate_attribute_pipeline.py --n_frames 10
"""
import argparse
import glob
import os

import cv2
import numpy as np

from attribute_utils import (
    CLASSES, GROUP, decide, frame_scores, load_templates, load_vent_unet)
from dimension_utils import _load_mobile_sam, refine_mask

DOOR_DIR = os.path.dirname(os.path.abspath(__file__))
SAM_CKPT = os.path.join(DOOR_DIR, 'sam_models', 'mobile_sam.pt')

parser = argparse.ArgumentParser()
parser.add_argument('--base', default='datasets')
parser.add_argument('--split', choices=['val', 'all'], default=None,
                    help='val: U-Net 학습에 안 쓴 프레임만 평가 (누수 방지). '
                         '기본값: datasets/aug 계열은 val, 그 외(현장)는 all')
parser.add_argument('--n_frames', type=int, default=10)
parser.add_argument('--n_boot', type=int, default=2000)
parser.add_argument('--model', default='attribute_models/vent_unet2.pth')
args = parser.parse_args()
BASE = os.path.join(DOOR_DIR, args.base)
MASK_CACHE = os.path.join(DOOR_DIR, 'factory_masks')

# 학습 데이터 출신 셋(datasets, datasets_aug*)은 기본 val-only —
# 학습에 사용된 프레임을 평가에 넣으면 낙관적 수치가 나온다.
if args.split is None:
    args.split = 'val' if args.base.startswith('datasets') \
        and 'factory' not in args.base else 'all'
VAL_IDX = None
if args.split == 'val':
    import json
    sp = os.path.join(DOOR_DIR, 'vent_labels', 'datasets', 'split.json')
    if not os.path.exists(sp):
        raise SystemExit(f'split.json이 없습니다: {sp} '
                         '(11→12 실행 후 생성됨, 또는 --split all 사용)')
    VAL_IDX = {cls: set(v['val'])
               for cls, v in json.load(open(sp)).items()}
    print(f'평가 분할: val (학습 미사용 프레임만, {sp})')
else:
    print('평가 분할: all')


def ensure_mask(cls, idx, rgb_path):
    """마스크 확보: 데이터셋 동봉 마스크 → 캐시 → MobileSAM 생성."""
    p = f'{BASE}/{cls}/mask_{idx}.png'
    if os.path.exists(p):
        return cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    os.makedirs(MASK_CACHE, exist_ok=True)
    cp = f'{MASK_CACHE}/{cls}_mask_{idx}.png'
    if os.path.exists(cp):
        return cv2.imread(cp, cv2.IMREAD_GRAYSCALE)
    global _sam
    if '_sam' not in globals():
        _sam = _load_mobile_sam(SAM_CKPT, 'cuda')
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    _sam.set_image(rgb)
    h, w = rgb.shape[:2]
    masks, _, _ = _sam.predict(
        point_coords=np.array([[w // 2, h // 2]], np.float32),
        point_labels=np.array([1], np.int32), multimask_output=True)
    best = max(range(len(masks)), key=lambda i: masks[i].sum())
    m = masks[best].astype(np.uint8) * 255
    cv2.imwrite(cp, m)
    return m


if __name__ == '__main__':
    net = load_vent_unet(os.path.join(DOOR_DIR, args.model))
    templates = load_templates()
    frames = {}
    for cls_dir in sorted(glob.glob(f'{BASE}/*/')):
        cls = os.path.basename(cls_dir.rstrip('/'))
        if cls not in CLASSES:
            continue
        fr = []
        for rp in sorted(glob.glob(f'{cls_dir}/rgb_*.png')):
            idx = os.path.basename(rp)[4:8]
            if VAL_IDX is not None and idx not in VAL_IDX.get(cls, set()):
                continue
            rgb = cv2.imread(rp)
            depth = cv2.imread(f'{cls_dir}/depth_{idx}.png',
                               cv2.IMREAD_UNCHANGED)
            mask = ensure_mask(cls, idx, rp)
            if rgb is None or depth is None or mask is None:
                continue
            try:
                fr.append(frame_scores(rgb, depth, refine_mask(mask),
                                       net, templates))
            except Exception:
                pass
        if fr:
            frames[cls] = fr
            print(f'{cls}: {len(fr)} frames', flush=True)

    rng = np.random.default_rng(42)
    print(f'\nN={args.n_frames} 부트스트랩({args.n_boot}회) 최종 판정:')
    tot_g = tot_c = tot = 0
    for cls, fr in frames.items():
        okg = okc = 0
        for _ in range(args.n_boot):
            sel = [fr[i] for i in rng.choice(
                len(fr), size=min(args.n_frames, len(fr)), replace=False)]
            pred, grp, _ = decide(sel)
            okg += (GROUP[pred] == GROUP[cls])
            okc += (pred == cls)
        tot_g += okg
        tot_c += okc
        tot += args.n_boot
        print(f'  {cls:22s} group {100 * okg / args.n_boot:5.1f}%  '
              f'class {100 * okc / args.n_boot:5.1f}%')
    print(f'  {"종합":20s} group {100 * tot_g / tot:5.1f}%  '
          f'class {100 * tot_c / tot:5.1f}%')
