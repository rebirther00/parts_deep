"""Frozen human field coordinates and fold-specific supervision budgets."""
from collections import Counter
from copy import deepcopy
import hashlib

from .annotations import KEYS, validate_label
from .core import (ROOT, dump, jsonl, now, outpath, partition, read_json,
                   read_jsonl, require_fresh_dir, sha256, snapshot)

IMMUTABLE = ['annotation_id', 'frame_id', 'session_id', 'session_dir', 'group_id',
             'class_name', 'rgb', 'rgb_sha256', 'snapshot_sha256', 'cv_group', 'roles_by_fold']


def freeze_field_labels(labelset, name='v1'):
    sp, _, frames, splits = snapshot(name)
    queue_path = ROOT / 'data/field40_v2_queue.jsonl'
    queue = read_jsonl(queue_path)
    index = {r['frame_id']: r for r in frames}
    if len(queue) != 40 or len({r['frame_id'] for r in queue}) != 40:
        raise ValueError('Expected the 40 distinct prespecified field candidates')
    role_maps = {fold: {r['frame_id']: role for role, rr in partition(frames, splits, fold).items() for r in rr}
                 for fold in splits['partitions']}
    rows, hashes = [], {}
    for template in queue:
        path = ROOT / 'data/annotations/field40_v2' / (template['annotation_id'] + '.json')
        row = read_json(path)
        validate_label(row, row['annotator'])
        if any(row.get(k) != template.get(k) for k in IMMUTABLE):
            raise ValueError('Annotation metadata changed: ' + template['annotation_id'])
        if row['source'] != 'human_manual_field' or row['annotation_status'] != 'human_verified':
            raise ValueError('Human annotation provenance required')
        if row['snapshot_sha256'] != sha256(sp / 'COMPLETE.json'):
            raise ValueError('Annotation/base snapshot mismatch')
        frame = index[row['frame_id']]
        for key in ['session_id', 'session_dir', 'group_id', 'class_name', 'rgb', 'rgb_sha256']:
            if row[key] != frame[key]:
                raise ValueError('Annotation/frozen frame mismatch: ' + key)
        for fold, roles in role_maps.items():
            if row['roles_by_fold'][fold] != roles[row['frame_id']]:
                raise ValueError('Annotation/fold role mismatch')
        row = deepcopy(row)
        row['source_label_sha256'] = sha256(path)
        hashes[str(path.relative_to(ROOT))] = row['source_label_sha256']
        rows.append(row)
    dst = require_fresh_dir('data/annotation_snapshots/' + labelset)
    jsonl(dst / 'annotations.jsonl', rows)
    meta = dict(status='complete', created_at=now(), labelset=labelset,
                base_snapshot=name, base_snapshot_sha256=sha256(sp / 'COMPLETE.json'),
                queue_sha256=sha256(queue_path), annotations_sha256=sha256(dst / 'annotations.jsonl'),
                source_label_hashes=hashes, images=len(rows), source='human_manual_field',
                visible_points={k: sum(r['visible'][k] for r in rows) for k in KEYS},
                note='Immutable copy of human coordinates. Test coordinates are not development inputs.')
    dump(dst / 'COMPLETE.json', meta)
    return meta


def load_field_labels(labelset, sp):
    path = outpath('data/annotation_snapshots/' + labelset)
    meta = read_json(path / 'COMPLETE.json')
    if meta['status'] != 'complete' or meta['base_snapshot_sha256'] != sha256(sp / 'COMPLETE.json'):
        raise ValueError('Field label snapshot mismatch')
    if sha256(path / 'annotations.jsonl') != meta['annotations_sha256']:
        raise ValueError('Frozen human field coordinates changed')
    rows = read_jsonl(path / 'annotations.jsonl')
    if len(rows) != 40 or len({r['annotation_id'] for r in rows}) != 40:
        raise ValueError('Incomplete or duplicate field annotations')
    return path, meta, rows


def field_partitions(labels, frames, splits, fold):
    parts = partition(frames, splits, str(fold))
    index = {r['frame_id']: r for r in frames}
    role_ids = {role: {r['frame_id'] for r in rr} for role, rr in parts.items()}
    result = {role: [] for role in parts}
    for label in labels:
        frame = index[label['frame_id']]
        if any(frame[k] != label[k] for k in ['group_id', 'class_name', 'rgb_sha256']):
            raise ValueError('Field label/frame mismatch')
        roles = [role for role, ids in role_ids.items() if label['frame_id'] in ids]
        if len(roles) != 1 or label['roles_by_fold'][str(fold)] != roles[0]:
            raise ValueError('Field label partition leakage')
        result[roles[0]].append(label)
    return result


def nested_budget(train_labels, classes, budget, fold, selection_seed=20261109):
    if budget not in (8, 16, 24) or len(classes) != 8:
        raise ValueError('Field training budgets must be 8/16/24 (one/two/three per class)')
    selected = []
    for cls in classes:
        candidates = [r for r in train_labels if r['class_name'] == cls]
        if len(candidates) != 3:
            raise ValueError('Expected three TRAIN field candidates per class')
        def order(row):
            return hashlib.sha256(f'{selection_seed}:{fold}:{row["annotation_id"]}'.encode()).hexdigest()
        selected.extend(sorted(candidates, key=order)[:budget // 8])
    if len(selected) != budget:
        raise ValueError('Unbalanced annotation budget')
    return selected
