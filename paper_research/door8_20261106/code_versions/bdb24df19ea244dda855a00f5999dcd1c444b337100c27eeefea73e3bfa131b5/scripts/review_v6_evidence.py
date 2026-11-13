"""Audit stored v6 evidence without training or changing experiment outputs."""
from pathlib import Path
from collections import defaultdict
from itertools import permutations
import json
import math
import statistics as st

from review_v5_evidence import rows, read, digest, metrics, compare, auc

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'reports/논문초안_v6_리뷰_20261110'
OUT = ROOT / 'reports/extension_v6'


def scale(train, features, cad):
    session = defaultdict(list)
    for r in train:
        f = features[r['frame_id']]
        if f['gate'] == 'ok' and f.get('span_normalized', 0) > 0:
            session[str(r['intrinsics'].get('serial', 'unknown')), r['session_dir']].append(cad[r['class_name']] / f['span_normalized'])
    cameras = defaultdict(list)
    for (cam, _), values in session.items():
        cameras[cam].append(st.median(values))
    return {cam: st.median(values) for cam, values in cameras.items()}


def pooled(mm):
    return dict(unknown_doors=[sum(m['unknown_doors'][i] for m in mm) for i in (0, 1)],
                known_door_decisions={k: sum(m['known'][k] for m in mm) for k in ('n', 'correct', 'abstained', 'wrong')},
                auroc_mean=st.mean(m['auc'] for m in mm), auroc_min=min(m['auc'] for m in mm))


def check_pool(result, expected):
    for k in ('unknown_doors', 'known_door_decisions'):
        assert result[k] == expected[k]
    for k in ('auroc_mean', 'auroc_min'):
        assert math.isclose(result[k], expected[k], abs_tol=1e-12)


def main():
    originals = read(DEST / '원본해시.json')
    for name, value in originals.items():
        assert digest(ROOT / name) == value
    sp = ROOT / 'data/snapshots/v1'
    for name, value in read(sp / 'COMPLETE.json')['files'].items():
        assert digest(sp / name) == value
    cfg = read(sp / 'protocol.json')
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm']))
    frames = rows(sp / 'frames.jsonl')
    partitions = read(sp / 'splits.json')['partitions']
    by_id = {r['frame_id']: r for r in frames}
    parts = {k: {role: [r for r in frames if r['group_id'] in set(gg)] for role, gg in v.items()}
             for k, v in partitions.items()}
    protocol_hash = digest(ROOT / 'config/extension_v6_protocol.json')
    collected, seen = read(OUT / 'collected.json'), read(OUT / 'seen.json')
    assert not collected['missing_strict'] and collected['protocol_sha256'] == seen['protocol_sha256'] == protocol_hash

    hp, cp = defaultdict(list), defaultdict(list)
    for r in rows(OUT / 'seen_hole_predictions.jsonl'):
        hp[r['fold'], r['excluded']].append(r)
    for r in rows(OUT / 'seen_cnn_predictions.jsonl'):
        cp[r['kind'], r['fold'], r['excluded']].append(r)
    assert len(hp) == 40 and len(cp) == 80
    seen_checked = {'hole': {}, 'rgb': {}, 'rgbe': {}}
    stable_mask = dict(changed_frame_acceptances=0, max_score_change=0., max_auroc_change=0.)
    for k in map(str, range(5)):
        if k == '0':
            files = ['features/holes_field8_f0_s42_v2_development/rows.jsonl', 'features/holes_field8_f0_s42_v2_f0_test_v3/rows.jsonl']
        else:
            files = [f'features/holes_field8_f{k}_s42_v3_trainTest/rows.jsonl']
        pool = {r['frame_id']: r for file in files for r in rows(ROOT / file)}
        cal = scale(parts[k]['train'], pool, cad)
        for c in cad:
            rr = hp[k, c]
            assert {r['frame_id'] for r in rr} == {r['frame_id'] for r in parts[k]['test']}
            key = k + ':' + c
            assert cal == seen['hole'][key]['scale']
            for r in rr:
                if r['measurement'] is not None:
                    f = by_id[r['frame_id']]
                    assert math.isclose(r['measurement'], pool[r['frame_id']]['span_normalized'] * cal[str(f['intrinsics']['serial'])], abs_tol=1e-10)
            m = metrics(rr, c, lambda r: r['score'] if r['score'] is not None else float('-inf'))
            compare(m, seen['hole'][key]); seen_checked['hole'][key] = m
        for kind in ('rgb', 'rgbe'):
            meta = read(ROOT / f'runs/{kind}_f{k}_s42/run.json')
            test = rows(ROOT / f'reports/{kind}_f{k}_s42_test/predictions.jsonl')
            val = rows(ROOT / f'runs/{kind}_f{k}_s42/val_predictions.jsonl')
            assert {r['frame_id'] for r in test} == {r['frame_id'] for r in parts[k]['test']}
            assert {r['frame_id'] for r in val} == {r['frame_id'] for r in parts[k]['val']}
            for c in cad:
                ci = meta['classes'].index(c)
                def stable(r):
                    q = [p for i, p in enumerate(r['probabilities']) if i != ci]
                    return max(q) / math.fsum(q)
                vv = sorted(stable(r) for r in val if r['class_name'] != c)
                threshold = vv[int(.05 * len(vv))]
                key, original = k + ':' + c, cp[kind, k, c]
                oi = {r['frame_id']: r for r in original}
                stable_known, stable_unknown = [], []
                for r in test:
                    sc, old = stable(r), oi[r['frame_id']]
                    stable_mask['max_score_change'] = max(stable_mask['max_score_change'], abs(sc - old['confidence']))
                    stable_mask['changed_frame_acceptances'] += (sc >= threshold) != old['accepted']
                    (stable_unknown if r['class_name'] == c else stable_known).append(sc)
                    assert old['accepted'] == (old['confidence'] >= seen['cnn'][kind][key]['threshold'])
                opened = [dict(r, pred=r['pred'] if r['accepted'] else None) for r in original]
                m = metrics(opened, c, lambda r: r['confidence'])
                compare(m, seen['cnn'][kind][key]); seen_checked[kind][key] = m
                stable_mask['max_auroc_change'] = max(stable_mask['max_auroc_change'], abs(auc(stable_known, stable_unknown) - m['auc']))
    seen_summary = {}
    for kind, mm in seen_checked.items():
        seen_summary[kind] = {}
        for subset in ('fold0', 'oof'):
            result = pooled([m for key, m in mm.items() if subset == 'oof' or key.startswith('0:')])
            check_pool(result, collected['seen']['pooled'][kind][subset])
            seen_summary[kind][subset] = result

    base = read(ROOT / 'runs/holes_field8_f0_s42_v2/run.json')
    label_path = ROOT / 'data/annotation_snapshots' / base['labelset']
    field = rows(label_path / 'annotations.jsonl')
    assert digest(label_path / 'COMPLETE.json') == base['labelset_sha256']
    assert digest(label_path / 'annotations.jsonl') == read(label_path / 'COMPLETE.json')['annotations_sha256']
    labels = {r['annotation_id']: r for r in field + rows(sp / 'annotations_lab.jsonl')}
    test_labels = [r for r in field if r['group_id'] in set(partitions['0']['test'])]
    assert len(test_labels) == 8
    strict, points, weight_hashes = {}, [], set()
    for c in cad:
        run = f'holes_strict_loco_{c}_f0_s42'
        rp = ROOT / 'runs' / run
        meta = read(rp / 'run.json')
        assert meta['status'] == 'complete' and meta['extension_protocol_sha256'] == protocol_hash
        assert meta['hyperparameters'] == base['hyperparameters']
        assert digest(rp / 'model.pth') == meta['weights_sha256']
        weight_hashes.add(meta['weights_sha256'])
        for role in ('train_annotation_ids', 'val_annotation_ids', 'lab_monitor_annotation_ids'):
            expected = {i for i in base[role] if labels[i]['class_name'] != c}
            assert set(meta[role]) == expected
            assert all(labels[i]['class_name'] != c for i in meta[role])
            assert not {labels[i].get('group_id') for i in meta[role]} & set(partitions['0']['test'])
        fp = ROOT / 'features' / (run + '_trainTest')
        feature_meta = read(fp / 'features.json')
        assert feature_meta['weights_sha256'] == meta['weights_sha256']
        assert digest(fp / 'rows.jsonl') == feature_meta['rows_sha256']
        features = {r['frame_id']: r for r in rows(fp / 'rows.jsonl')}
        assert set(features) == {r['frame_id'] for role in ('train', 'test') for r in parts['0'][role]}
        cad7 = {key: value for key, value in cad.items() if key != c}
        cal = scale([r for r in parts['0']['train'] if r['class_name'] != c], features, cad7)
        saved = read(OUT / (run + '_metrics.json'))
        assert cal == saved['scale']
        rr = rows(OUT / (run + '_predictions.jsonl'))
        assert {r['frame_id'] for r in rr} == {r['frame_id'] for r in parts['0']['test']}
        for r in rr:
            if r['measurement'] is not None:
                v = features[r['frame_id']]['span_normalized'] * cal[str(by_id[r['frame_id']]['intrinsics']['serial'])]
                assert math.isclose(v, r['measurement'], abs_tol=1e-10)
                closest = min(cad7, key=lambda key: abs(cad7[key] - v))
                expected = closest if abs(cad7[closest] - v) <= 30 else None
                assert r['pred'] == expected
        m = metrics(rr, c, lambda r: r['score'] if r['score'] is not None else float('-inf'))
        compare(m, saved)
        errors, excluded_errors = [], []
        for label in test_labels:
            for channel, keys in [('bolt', ['bolt_tl', 'bolt_tr', 'bolt_bl', 'bolt_br']), ('hinge', ['corner_hinge']), ('latch', ['corner_latch'])]:
                keys = [key for key in keys if label['visible'][key]]
                pred = features[label['frame_id']]['points'][channel]
                candidates = [[math.dist(label['points'][key], p[:2]) for key, p in zip(keys, perm)] for perm in permutations(pred, len(keys))]
                ee = min(candidates, key=sum)
                errors.extend(ee)
                if label['class_name'] == c:
                    excluded_errors.extend(ee)
        assert len(errors) == 48 and all(e <= 8 for e in errors)
        assert math.isclose(st.median(errors), saved['test_role_localization']['all']['median_error_px_when_matched'], abs_tol=1e-8)
        assert len(excluded_errors) == 6
        points.extend(excluded_errors)
        strict[c] = dict(**m, annotation_exclusion_checked=True, checkpoint_images_exclusion_checked=True,
                         lab_train=meta['lab_train_images'], field_train=meta['field_train_images'],
                         field_val=meta['field_val_images'], best_epoch=meta['best_epoch'],
                         min_distance_mm=saved['unknown_min_distance_mm'], test_points=48,
                         localization_scope_string=saved['test_role_localization']['scope'])
    assert len(weight_hashes) == 8
    strict_summary = pooled(list(strict.values()))
    check_pool(strict_summary, collected['unseen']['pooled']['hole'])
    previous = read(ROOT / 'reports/논문초안_v5_리뷰_20261110/근거대조.json')
    conflicts = [dict(split=split, **r) for split, rr in previous['U_disagreements'].items() for r in rr]
    nonmajority_reject = [r for r in conflicts if r['votes'].get('None', 0) <= r['n_frames'] / 2]
    evidence = dict(review_date='2026-11-10', original_sha256=originals,
        native_hwpx=dict(top_paragraphs=121, all_paragraphs=568, tables=22, figures=4, abstract_words=150),
        seen_runs_checked=120, seen_summary=seen_summary, mask_numerical_check=stable_mask,
        strict_runs_checked=8, distinct_weight_hashes=len(weight_hashes), strict_runs=strict,
        strict_summary=strict_summary, strict_excluded_point_summary=dict(n=len(points), median=st.median(points), maximum=max(points)),
        U_nonmajority_abstention_counterexamples=nonmajority_reject,
        scope='Stored predictions, frozen annotations, training metadata, checkpoint/feature hashes; no training or original result changes.')
    (DEST / '근거대조.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(seen_runs=120, strict_runs=8, strict_summary=strict_summary,
                         localization=evidence['strict_excluded_point_summary'], mask=stable_mask,
                         U_explanation_counterexamples=len(nonmajority_reject)), ensure_ascii=False))


if __name__ == '__main__':
    main()
