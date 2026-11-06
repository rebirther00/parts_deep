from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SOURCE = REPO / 'class_estimation/door_pipeline'


def outpath(path):
    """Reject output escapes, including an existing symlink outside this project."""
    p = Path(path)
    p = (p if p.is_absolute() else ROOT / p).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError(f'Output must remain inside {ROOT}: {p}')
    return p


def dump(path, obj):
    p = outpath(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = outpath(p.with_name(p.name + f'.{os.getpid()}.tmp'))
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    tmp.replace(p)


def jsonl(path, rows):
    p = outpath(path); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = outpath(p.with_name(p.name + f'.{os.getpid()}.tmp'))
    with tmp.open('w', encoding='utf8') as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(p)


def write_csv(path, rows, fields=None):
    p = outpath(path); p.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else [])
    with p.open('w', newline='', encoding='utf8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)


def read_json(path): return json.loads(Path(path).read_text(encoding='utf8'))
def read_jsonl(path): return [json.loads(s) for s in Path(path).read_text(encoding='utf8').splitlines() if s.strip()]
def now(): return datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def setup_runtime():
    for name, rel in [('TORCH_HOME','cache/torch'), ('XDG_CACHE_HOME','cache/xdg'), ('MPLCONFIGDIR','cache/matplotlib')]:
        p = outpath(rel); p.mkdir(parents=True, exist_ok=True); os.environ[name] = str(p)
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[name] = '1'


def protocol(): return read_json(ROOT / 'config/protocol.json')


def snapshot(name='v1'):
    p = outpath('data/snapshots/' + name)
    if not (p/'COMPLETE.json').exists(): raise RuntimeError(f'Incomplete or missing snapshot: {p}')
    summary = read_json(p/'summary.json')
    for filename, expected in read_json(p/'COMPLETE.json')['files'].items():
        if sha256(p/filename) != expected: raise RuntimeError(f'Frozen metadata changed: {filename}')
    return p, summary, read_jsonl(p/'frames.jsonl'), read_json(p/'splits.json')


def partition(rows, splits, fold):
    f = splits['partitions'][str(fold)]
    groups = {k: set(f[k]) for k in ('train','val','test')}
    if groups['train'] & groups['val'] or groups['train'] & groups['test'] or groups['val'] & groups['test']:
        raise ValueError('Split group leakage')
    out = {k:[r for r in rows if r['group_id'] in s] for k,s in groups.items()}
    if sum(map(len,out.values())) != len(rows): raise ValueError('Split does not cover every frame')
    for a,b in [('train','val'),('train','test'),('val','test')]:
        if {r['rgb_sha256'] for r in out[a]} & {r['rgb_sha256'] for r in out[b]}:
            raise ValueError(f'Identical RGB content across {a}/{b}')
    return out


def require_fresh_dir(path):
    p = outpath(path)
    if p.exists(): raise FileExistsError(f'Refusing to overwrite experiment: {p}')
    p.mkdir(parents=True)
    return p


def check_source(row, full_hash=False):
    for key in ('rgb','depth'):
        p = Path(row[key])
        if not p.is_file(): raise FileNotFoundError(p)
        st = p.stat()
        if st.st_size != row[key+'_bytes'] or st.st_mtime_ns != row[key+'_mtime_ns']:
            raise RuntimeError(f'Source file changed since snapshot: {p}')
        if full_hash and sha256(p) != row[key+'_sha256']: raise RuntimeError(f'Source content changed: {p}')
