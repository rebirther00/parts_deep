"""CAD STL → 속성 파이프라인 템플릿/스펙 생성.

cad/door_stl/*.stl 을 도어 평면에 직교 투영해서:
- attribute_models/cad_templates.npz : 클래스별 실루엣 + 슬롯 마스크 (2mm/px)
- attribute_models/class_spec.json   : bbox/종횡비/슬롯수 스펙 갱신

신규 부품/리비전 추가 절차: STL을 cad/door_stl/에 넣고 본 스크립트 재실행.
(aspect_center는 실측 캘리브레이션 값이라 기존 값을 보존하며, 신규 클래스는
 aspect_cad로 초기화 — 이후 13_evaluate로 실측 재캘리브레이션 권장)

주의: STL 투영 시 cv2.fillPoly는 even-odd 규칙으로 판금 앞뒷면이 상쇄되므로
반드시 삼각형별 fillConvexPoly(OR 누적)를 사용한다.

실행: python 10_generate_cad_templates.py
"""
import glob
import json
import os

import cv2
import numpy as np
from scipy import ndimage
from stl import mesh

DOOR_DIR = os.path.dirname(os.path.abspath(__file__))
STL_DIR = os.path.normpath(os.path.join(DOOR_DIR, '..', '..', 'cad',
                                        'door_stl'))
OUT_DIR = os.path.join(DOOR_DIR, 'attribute_models')
RES = 2.0  # mm/px

GROUP_OF = (lambda c: 'FRT' if 'FRT' in c
            else 'RH' if c.endswith('RH') else 'RR')


def project(stl_path, res=RES):
    """가장 얇은 축으로 직교 투영한 점유 마스크."""
    m = mesh.Mesh.from_file(stl_path)
    pts = m.vectors.reshape(-1, 3)
    mins, maxs = pts.min(axis=0), pts.max(axis=0)
    dims = maxs - mins
    thin = int(np.argmin(dims))
    axes = [a for a in range(3) if a != thin]
    W = int(dims[axes[0]] / res) + 2
    H = int(dims[axes[1]] / res) + 2
    tri = ((m.vectors[:, :, axes] - [mins[axes[0]], mins[axes[1]]]) / res
           ).astype(np.int32)
    grid = np.zeros((H, W), np.uint8)
    for t in tri:
        cv2.fillConvexPoly(grid, t, 255)
    bbox = sorted(dims, reverse=True)[:2]  # [최장, 차장] mm
    return grid, [round(float(b)) for b in bbox]


def slot_mask_of(grid, res=RES):
    """내부 관통 홀 중 통풍 슬롯(3~30cm²)만 추출."""
    inv = (grid == 0)
    lbl, n = ndimage.label(inv)
    border = set(lbl[0, :]) | set(lbl[-1, :]) | set(lbl[:, 0]) \
        | set(lbl[:, -1])
    slot = np.zeros_like(grid)
    count = 0
    objs = ndimage.find_objects(lbl)
    for i in range(1, n + 1):
        if i in border:
            continue
        sl = objs[i - 1]
        comp = (lbl[sl] == i)
        area = comp.sum() * res * res / 100
        if 3 <= area <= 30:
            slot[sl][comp] = 255
            count += 1
    return slot, count


if __name__ == '__main__':
    os.makedirs(OUT_DIR, exist_ok=True)
    spec_path = os.path.join(OUT_DIR, 'class_spec.json')
    old_spec = (json.load(open(spec_path, encoding='utf-8'))
                if os.path.exists(spec_path) else {'classes': {}})

    templates = {}
    classes = {}
    for p in sorted(glob.glob(f'{STL_DIR}/*.stl')):
        name = os.path.basename(p).replace('.stl', '')
        grid, bbox = project(p)
        slot, n_slot = slot_mask_of(grid)
        sil = ndimage.binary_fill_holes(grid > 0).astype(np.uint8) * 255
        templates[f'{name}_sil'] = sil
        templates[f'{name}_slot'] = slot
        aspect_cad = round(bbox[0] / bbox[1], 4)
        prev = old_spec['classes'].get(name, {})
        classes[name] = {
            'group': GROUP_OF(name),
            'bbox_mm': bbox,
            'aspect_cad': aspect_cad,
            'aspect_center': prev.get('aspect_center', aspect_cad),
            'slot_count_cad': n_slot,
        }
        print(f'{name}: bbox={bbox} slots={n_slot} '
              f'aspect_center={classes[name]["aspect_center"]}')

    np.savez_compressed(os.path.join(OUT_DIR, 'cad_templates.npz'),
                        **templates)
    spec = {
        'version': old_spec.get('version', ''),
        'common_side_mm': 1140,
        'decision': old_spec.get('decision', {
            'eps': 0.01, 'sigma': 0.05, 'n_frames': 10}),
        'classes': classes,
    }
    json.dump(spec, open(spec_path, 'w', encoding='utf-8'),
              indent=2, ensure_ascii=False)
    print(f'저장: {OUT_DIR}/cad_templates.npz, class_spec.json')
