"""Manual-point geometry diagnosis on TRAIN/VAL only; never model test accuracy."""
import argparse
from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from door8.core import (check_source, dump, jsonl, now, read_json, require_fresh_dir,
                       setup_runtime, sha256, snapshot, write_csv)
setup_runtime()
import cv2
import numpy as np
from door8.annotations import KEYS
from door8.field_data import field_partitions, freeze_field_labels, load_field_labels
from door8.geometry import bolt_frame, classify_measurement, depth_measure, fit_scale, geometry_gate


def geometry_features(frame, points, visible):
    check_source(frame)
    image = cv2.imread(frame['rgb'])
    depth = cv2.imread(frame['depth'], cv2.IMREAD_UNCHANGED)
    if image is None or depth is None or depth.shape != image.shape[:2]:
        raise ValueError('Unreadable or unregistered RGB/depth pair')
    bolts = [points[k] for k in KEYS[:4] if visible[k] and points[k] is not None]
    hinge = points['corner_hinge'] if visible['corner_hinge'] else None
    latch = points['corner_latch'] if visible['corner_latch'] else None
    bf = bolt_frame(bolts)
    result = dict(frame_id=frame['frame_id'], rgb_sha256=frame['rgb_sha256'],
                  gate=geometry_gate(bf, hinge, latch, image.shape), span_normalized=None)
    if hinge is not None and latch is not None:
        result['span_normalized'] = float(np.linalg.norm(np.asarray(hinge)-latch)/frame['intrinsics']['fx'])
        measured = depth_measure(depth, bolts+[hinge, latch], hinge, latch, frame['intrinsics'])
        if measured:
            result.update(measured)
    return result


def diagnose(labelset, report, fold='0', name='v1'):
    sp, _, frames, splits = snapshot(name)
    cfg = read_json(sp / 'protocol.json')
    lp, lm, labels = load_field_labels(labelset, sp)
    parts = field_partitions(labels, frames, splits, fold)
    index = {r['frame_id']: r for r in frames}
    dst = require_fresh_dir('reports/' + report)
    features, rows_by_role = {}, {}
    for role in ('train', 'val'):
        rows_by_role[role] = [index[r['frame_id']] for r in parts[role]]
        for label in parts[role]:
            features[label['frame_id']] = geometry_features(index[label['frame_id']], label['points'], label['visible'])
    test_ids = {r['frame_id'] for r in parts['test']}
    if test_ids & set(features):
        raise ValueError('Test coordinates entered development diagnosis')
    jsonl(dst / 'manual_features.jsonl', list(features.values()))
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm']))
    result = dict(created_at=now(), labelset=labelset, labelset_sha256=sha256(lp / 'COMPLETE.json'),
                  fold=str(fold), source='human_manual_coordinates_not_model_predictions',
                  test_images_evaluated=0, reserved_test_images=len(test_ids),
                  scope='Development diagnosis; TRAIN is calibration data; VAL has one image per class.',
                  geometry_code_sha256=sha256(Path(__file__).parents[1]/'door8/geometry.py'),
                  gates={role: dict(Counter(features[r['frame_id']]['gate'] for r in rr)) for role,rr in rows_by_role.items()},
                  modes={})
    for mode in ('pixel', 'depth'):
        cal = fit_scale(rows_by_role['train'], features, cad, mode)
        if set(cal['train_frame_ids']) != {r['frame_id'] for r in rows_by_role['train']}:
            raise ValueError('Calibration leakage')
        dump(dst / (mode + '_calibration.json'), cal)
        result['modes'][mode] = {}
        for role, rows in rows_by_role.items():
            predictions = []
            for row in rows:
                pred, distance, reason = classify_measurement(row, features[row['frame_id']], cal, cfg)
                predictions.append(dict(frame_id=row['frame_id'], session_dir=row['session_dir'],
                                        class_name=row['class_name'], pred=pred, reason=reason,
                                        distance_mm=distance, cad_mm=cad[row['class_name']],
                                        signed_cad_residual_mm=distance-cad[row['class_name']] if distance is not None else None))
            residuals = [abs(p['signed_cad_residual_mm']) for p in predictions if p['signed_cad_residual_mm'] is not None]
            correct = sum(p['pred'] == p['class_name'] for p in predictions)
            judged = sum(p['pred'] is not None for p in predictions)
            result['modes'][mode][role] = dict(n=len(rows), correct=correct, judged=judged,
                abstained=len(rows)-judged, accuracy_all=correct/len(rows), coverage=judged/len(rows),
                median_absolute_cad_residual_mm=float(np.median(residuals)) if residuals else None,
                p95_absolute_cad_residual_mm=float(np.quantile(residuals,.95)) if residuals else None,
                max_absolute_cad_residual_mm=max(residuals) if residuals else None,
                residual_scope='All computable distances, including CAD-distance rejections; not physical metrology errors.')
            jsonl(dst / f'{mode}_{role}_predictions.jsonl', predictions)
            write_csv(dst / f'{mode}_{role}_predictions.csv', predictions)
    dump(dst / 'diagnostic.json', result)
    lines = ['수동 현장 좌표의 기하 판정 진단', '',
             '사람이 지정한 좌표를 입력한 개발용 진단이다. 자동 모델의 시험 정확도가 아니다.',
             f'고정 현장 주석: {len(labels)}장. Fold {fold}: 학습 {len(parts["train"])}장, 검증 {len(parts["val"])}장. 시험 {len(test_ids)}장은 평가하지 않았다.', '',
             '| 방식 | 구분 | 정답/전체 | 판정 보류 | CAD 절대 잔차 중앙값 / 최대 (mm) |',
             '|---|---|---:|---:|---:|']
    for mode in ('pixel', 'depth'):
        for role in ('train','val'):
            x=result['modes'][mode][role]
            lines.append(f'| {mode} | {role} | {x["correct"]}/{x["n"]} | {x["abstained"]} | {x["median_absolute_cad_residual_mm"]:.2f} / {x["max_absolute_cad_residual_mm"]:.2f} |')
    lines += ['', '보정은 학습 24장만 사용했다. 학습 성능은 보정에 사용한 자료의 결과다. 검증은 클래스당 1장으로 표본이 작다.',
              '기존 임계값과 기하 규칙을 변경하지 않았다. 수동 좌표가 정확한 경우의 성립 여부를 확인하며 검출기의 일반화 성능을 뜻하지 않는다.',
              '픽셀 거리 방식이 이 표본에서 성립하므로 현장 좌표를 포함한 검출기 학습을 다음 단계로 진행한다. 깊이 방식의 큰 잔차·보류도 그대로 보존한다.',
              'CAD 잔차는 실물의 직접 측정 오차가 아니다. 시험 자료나 자동 좌표를 이용해 라벨을 수정하지 않았다.', '']
    (dst / '진단결과.md').write_text('\n'.join(lines), encoding='utf8')
    print(result, flush=True)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--labelset',default='field40_20261109_v1')
    p.add_argument('--freeze',action='store_true');p.add_argument('--fold',default='0')
    p.add_argument('--report',default='field40_manual_f0_v1');a=p.parse_args()
    if a.freeze:print(freeze_field_labels(a.labelset),flush=True)
    diagnose(a.labelset,a.report,a.fold)
