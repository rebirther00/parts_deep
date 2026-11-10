"""Human annotations shared by the desktop and local browser interfaces."""
from copy import deepcopy
from datetime import datetime
import math
import re
from threading import Lock

from PIL import Image

from .core import ROOT, dump, now, outpath, read_json, read_jsonl, sha256

KEYS = ['bolt_tl', 'bolt_tr', 'bolt_bl', 'bolt_br', 'corner_hinge', 'corner_latch']


def validate_label(row, annotator):
    if not isinstance(annotator, str) or not annotator.strip():
        raise ValueError('작업자 이름 또는 식별자를 입력하세요.')
    if len(annotator.strip()) > 100:
        raise ValueError('작업자 식별자는 100자 이내로 입력하세요.')
    if sha256(row['rgb']) != row['rgb_sha256']:
        raise ValueError('원본 영상이 고정 스냅샷 이후 변경됐습니다.')
    with Image.open(row['rgb']) as im:
        width, height = im.size
    for key in KEYS:
        visible = row['visible'][key]
        point = row['points'][key]
        if type(visible) is not bool:
            raise ValueError(f'{key}: 가시성 값이 올바르지 않습니다.')
        if not visible:
            if point is not None:
                raise ValueError(f'{key}: 가려진 점의 좌표는 비워야 합니다.')
            continue
        if point is None:
            raise ValueError(f'{key}: 점을 클릭하거나 가려짐(V)을 표시하세요.')
        if (not isinstance(point, (list, tuple)) or len(point) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in point)):
            raise ValueError(f'{key}: 유효한 x, y 좌표가 필요합니다.')
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            raise ValueError(f'{key}: 원본 영상 범위 밖의 좌표입니다.')


class RevisionConflict(ValueError):
    pass


class AnnotationStore:
    def __init__(self, queue=None, labeldir=None):
        self.rows = read_jsonl(queue or ROOT / 'data/field40_v2_queue.jsonl')
        self.labeldir = outpath(labeldir or 'data/annotations/field40_v2')
        self.lock = Lock()
        self.by_id = {}
        for row in self.rows:
            key = row['annotation_id']
            if not re.fullmatch(r'[A-Za-z0-9_.-]+', key) or key in self.by_id:
                raise ValueError('Invalid or repeated annotation ID')
            self.by_id[key] = row
        if not self.rows:
            raise ValueError('주석 후보 목록이 비어 있습니다.')

    def path(self, key):
        if key not in self.by_id:
            raise KeyError('알 수 없는 영상입니다.')
        return outpath(self.labeldir / (key + '.json'))

    def image_path(self, key):
        row = self.by_id[key]
        if sha256(row['rgb']) != row['rgb_sha256']:
            raise ValueError('원본 영상이 고정 스냅샷 이후 변경됐습니다.')
        return row['rgb']

    def items(self):
        return [dict(annotation_id=r['annotation_id'], class_name=r['class_name'],
                     session_dir=r['session_dir'], cv_group=r['cv_group'],
                     saved=self.path(r['annotation_id']).exists()) for r in self.rows]

    def item(self, key):
        with self.lock:
            row = deepcopy(self.by_id[key])
            path = self.path(key)
            revision = sha256(path) if path.exists() else None
            if revision is not None:
                saved = read_json(path)
                for field in ('points', 'visible', 'annotator', 'annotation_time'):
                    if field in saved:
                        row[field] = saved[field]
            with Image.open(self.image_path(key)) as im:
                row['width'], row['height'] = im.size
            row.pop('rgb', None)
            row['revision'] = revision
            return row

    def save(self, payload):
        required = {'annotation_id', 'annotator', 'points', 'visible', 'revision'}
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError('저장 요청 필드가 올바르지 않습니다.')
        key = payload['annotation_id']
        if not isinstance(key, str) or key not in self.by_id:
            raise ValueError('알 수 없는 영상입니다.')
        for name in ('points', 'visible'):
            if not isinstance(payload[name], dict) or set(payload[name]) != set(KEYS):
                raise ValueError('여섯 기준점 모두 표시하거나 가려짐으로 기록하세요.')
        # Only human point data can change; paths, class labels, and splits are fixed.
        row = deepcopy(self.by_id[key])
        row['points'] = deepcopy(payload['points'])
        row['visible'] = deepcopy(payload['visible'])
        validate_label(row, payload['annotator'])
        with self.lock:
            path = self.path(key)
            revision = sha256(path) if path.exists() else None
            if payload['revision'] != revision:
                raise RevisionConflict('다른 창에서 라벨이 변경됐습니다. 영상을 다시 열어 확인하세요.')
            if path.exists():
                stamp = datetime.now().strftime('%Y%m%dT%H%M%S%f')
                dump(self.labeldir / 'history' / (key + '_' + stamp + '.json'), read_json(path))
            row.update(annotation_status='human_verified', source='human_manual_field',
                       annotator=payload['annotator'].strip(), annotation_time=now())
            dump(path, row)
            return dict(revision=sha256(path), annotation_time=row['annotation_time'])
