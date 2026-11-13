"""Read-only audit of v5 outputs; write evidence only into the review directory."""
from pathlib import Path
from collections import defaultdict, Counter
from bisect import bisect_left, bisect_right
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'reports/논문초안_v5_리뷰_20261110'
OUT = ROOT / 'reports/extension_v5'


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.open()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(known, unknown):
    k = sorted(known)
    return sum(len(k) - bisect_right(k, u) + (bisect_right(k, u) - bisect_left(k, u)) / 2
               for u in unknown) / (len(k) * len(unknown))


def vote(rs):
    counts = Counter(r['pred'] for r in rs if r['pred'] is not None).most_common(2)
    return None if not counts or (len(counts) == 2 and counts[0][1] == counts[1][1]) else counts[0][0]


def metrics(rs, excluded, score):
    groups = defaultdict(list)
    for r in rs:
        groups[r['group_id']].append(r)
    unknown = [g for g in groups.values() if g[0]['class_name'] == excluded]
    known = [g for g in groups.values() if g[0]['class_name'] != excluded]
    uf = [r for r in rs if r['class_name'] == excluded]
    kf = [r for r in rs if r['class_name'] != excluded]
    return dict(unknown_doors=[sum(vote(g) is None for g in unknown), len(unknown)],
                unknown_frames=[sum(r['pred'] is None for r in uf), len(uf)],
                known=dict(n=len(known), correct=sum(vote(g) == g[0]['class_name'] for g in known),
                           abstained=sum(vote(g) is None for g in known),
                           wrong=sum(vote(g) not in (None, g[0]['class_name']) for g in known)),
                auc=auc([score(r) for r in kf], [score(r) for r in uf]))


def compare(m, saved):
    assert m['unknown_doors'] == [saved['unknown_doors']['rejected'], saved['unknown_doors']['n']]
    assert m['unknown_frames'] == [saved['unknown_frames']['rejected'], saved['unknown_frames']['n']]
    assert m['known'] == saved['known_doors_threshold']
    assert math.isclose(m['auc'], saved['auroc_frame'], abs_tol=1e-12)


def main():
    source_hashes = read(DEST / '원본해시.json')
    for name, expected in source_hashes.items():
        assert digest(ROOT / name) == expected
    sp = ROOT / 'data/snapshots/v1'
    for name, expected in read(sp / 'COMPLETE.json')['files'].items():
        assert digest(sp / name) == expected
    frames = rows(sp / 'frames.jsonl')
    by_id = {r['frame_id']: r for r in frames}
    cfg = read(sp / 'protocol.json')
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm']))
    partitions = read(sp / 'splits.json')['partitions']
    test_groups = set(partitions['0']['test'])
    test_ids = {r['frame_id'] for r in frames if r['group_id'] in test_groups}
    protocol_hash = digest(ROOT / 'config/extension_v5_protocol.json')
    cnn = {}
    for path in sorted(OUT.glob('*_loco_*_metrics.json')):
        saved = read(path)
        run = saved['run']
        meta = read(ROOT / 'runs' / run / 'run.json')
        excluded = saved['excluded_class']
        assert meta['status'] == 'complete'
        assert meta['extension_protocol_sha256'] == protocol_hash
        assert set(meta['classes']) == set(cad) - {excluded}
        for role in ('train', 'val'):
            expected = {r['frame_id'] for r in frames
                        if r['group_id'] in set(partitions['0'][role]) and r['class_name'] != excluded}
            assert set(meta[role + '_frame_ids']) == expected
            assert not set(meta[role + '_group_ids']) & test_groups
        assert not set(meta['train_group_ids']) & set(meta['val_group_ids'])
        pred = rows(OUT / (run + '_predictions.jsonl'))
        assert len(pred) == len(test_ids) and {r['frame_id'] for r in pred} == test_ids
        val_conf = sorted(r['confidence'] for r in rows(ROOT / 'runs' / run / 'val_predictions.jsonl'))
        threshold = val_conf[int(.05 * len(val_conf))]
        assert threshold == saved['threshold']
        for r in pred:
            assert r['accepted'] == (r['confidence'] >= threshold)
            assert r['unknown'] == (r['class_name'] == excluded)
            assert r['class_name'] == by_id[r['frame_id']]['class_name']
            if not r['accepted']:
                r['pred'] = None
        m = metrics(pred, excluded, lambda r: r['confidence'])
        compare(m, saved)
        cnn[run] = dict(**m, threshold=threshold, split_checked=True,
                        unknown_training_class_absent=True, protocol_hash_matches=True)
    assert len(cnn) == 16
    cnn_summary = {}
    for kind in ('rgb', 'rgbe'):
        mm = [m for run, m in cnn.items() if run.startswith(kind + '_')]
        cnn_summary[kind] = dict(unknown_rejected=sum(m['unknown_doors'][0] for m in mm),
             unknown_total=sum(m['unknown_doors'][1] for m in mm),
             known_abstained=sum(m['known']['abstained'] for m in mm),
             known_wrong=sum(m['known']['wrong'] for m in mm),
             known_correct=sum(m['known']['correct'] for m in mm),
             auroc_mean=statistics.mean(m['auc'] for m in mm), auroc_min=min(m['auc'] for m in mm))

    hole = read(OUT / 'hole_loco.json')
    hp = defaultdict(list)
    for r in rows(OUT / 'hole_loco_predictions.jsonl'):
        hp[r['fold'] + ':' + r['excluded']].append(r)
    assert len(hp) == 40
    min_distance = float('inf')
    for key, rr in hp.items():
        excluded = rr[0]['excluded']
        m = metrics(rr, excluded, lambda r: r['score'] if r['score'] is not None else float('-inf'))
        compare(m, hole['runs'][key])
        scores = [r['score'] for r in rr if r['class_name'] == excluded and r['score'] is not None]
        md = -max(scores)
        assert math.isclose(md, hole['runs'][key]['unknown_min_distance_mm'], abs_tol=1e-10)
        min_distance = min(min_distance, md)

    margin = read(OUT / 'margin.json')
    disagreements, per_fold = {}, []
    for mode in ('pixel', 'depth'):
        for split in ('0', '1', '2', '3', '4', 'temporal'):
            if split == '0':
                path = ROOT / f'reports/fold0_test_v3/holes_field8_f0_s42_v2_{mode}_test_predictions.jsonl'
            elif split == 'temporal':
                path = ROOT / f'reports/extension_v3/holes_field_temporal_s42_{mode}_predictions.jsonl'
            else:
                path = ROOT / f'reports/extension_v3/holes_field8_f{split}_s42_v3_{mode}_predictions.jsonl'
            gg = defaultdict(list)
            for r in rows(path):
                gg[r['group_id']].append(r)
            conflicts, us = [], []
            for gid, rr in gg.items():
                true, prediction = rr[0]['class_name'], vote(rr)
                residuals = [r['measurement'] / cad[true] - 1 for r in rr if r['measurement'] is not None]
                if not residuals:
                    continue
                median = statistics.median(residuals)
                u = median / margin['bands'][true][int(median >= 0)]
                us.append(u)
                if (u < 1) != (prediction == true):
                    conflicts.append(dict(group=gid, true=true, predicted=prediction, U=u,
                        n_frames=len(rr), votes=dict(Counter(str(r['pred']) for r in rr))))
            saved = margin['splits'][split + ':' + mode]
            assert len(gg) == saved['test']['doors']
            assert sum(vote(rr) != rr[0]['class_name'] for rr in gg.values()) == saved['test']['failed']
            assert math.isclose(max(us), max(saved['test']['utilisation'].values()), abs_tol=1e-12)
            disagreements[split + ':' + mode] = conflicts
            if mode == 'depth':
                per_fold.append(dict(split=split, train_n=saved['train']['doors'], train_fail=saved['train']['failed'],
                                     test_n=saved['test']['doors'], test_fail=saved['test']['failed']))

    e23 = read(OUT / 'e23_auroc.json')
    e23_computed = {}
    for kind in ('rgb', 'rgbe'):
        known = rows(ROOT / f'reports/{kind}_f0_s42_test/predictions.jsonl')
        unknown = rows(ROOT / f'reports/e23_openset_v3/{kind}_f0_s42_predictions.jsonl')
        value = auc([r['confidence'] for r in known], [r['confidence'] for r in unknown])
        assert math.isclose(value, e23[kind]['auroc'], abs_tol=1e-12)
        e23_computed[kind] = value
    result = dict(review_date='2026-11-10', original_sha256=source_hashes,
                  cnn_runs=cnn, cnn_summary=cnn_summary, all_16_cnn_splits_and_metrics_match=True,
                  hole_runs_checked=40, hole_summary=hole['oof'], hole_min_distance_mm=min_distance,
                  U_disagreements=disagreements, depth_train_test_by_split=per_fold,
                  e23_auroc_recomputed=e23_computed, native_hwpx=dict(top_paragraphs=113, all_paragraphs=510, tables=21, figures=4),
                  scope='Stored-output audit; no training or changes to manuscript or experiment outputs.')
    (DEST / '근거대조.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(cnn_summary=cnn_summary, hole_min_distance_mm=min_distance,
          U_disagreements={k: len(v) for k, v in disagreements.items()}, e23=e23_computed), ensure_ascii=False))


if __name__ == '__main__':
    main()
