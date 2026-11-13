"""Read-only checks for manuscript v3; writes a separate review evidence record."""
from pathlib import Path
import sys
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from docx import Document
from door8.core import ROOT, snapshot, partition, read_json, read_jsonl, sha256, dump, now
from door8.field_experiments import check_adaptation_provenance
from door8.metrics import summarize, wilson

DEST = ROOT / 'reports/논문초안_v3_리뷰_20261109'


def main():
    sp, _, frames, splits = snapshot()
    cfg = read_json(sp / 'protocol.json')
    classes = cfg['classes']
    frame_index = {r['frame_id']: r for r in frames}
    all_predictions = {k: [] for k in ['hole_pixel', 'hole_depth', 'rgb', 'rgbe']}
    checks = []
    temporal = {}
    localization = []
    for fold in ['0', '1', '2', '3', '4', 'temporal']:
        parts = partition(frames, splits, fold)
        expected = {r['frame_id'] for r in parts['test']}
        run = ('holes_field8_f0_s42_v2' if fold == '0' else
               'holes_field_temporal_s42' if fold == 'temporal' else f'holes_field8_f{fold}_s42_v3')
        meta = read_json(ROOT / 'runs' / run / 'run.json')
        assert meta['status'] == 'complete' and meta['seed'] == 42
        roles = check_adaptation_provenance(meta, sp, frames, splits, fold)
        train_groups = {r['group_id'] for r in parts['train']}
        val_groups = {r['group_id'] for r in parts['val']}
        assert set(meta['train_group_ids']) <= train_groups
        assert set(meta['val_group_ids']) <= val_groups
        assert not set(meta['train_group_ids']) & {r['group_id'] for r in parts['test']}
        for kind in all_predictions:
            if kind.startswith('hole_'):
                mode = kind.split('_')[1]
                filename = (ROOT / 'reports/fold0_test_v3' / f'{run}_{mode}_test_predictions.jsonl'
                            if fold == '0' else ROOT / 'reports/extension_v3' / f'{run}_{mode}_predictions.jsonl')
            else:
                cnn = f'{kind}_temporal_s42' if fold == 'temporal' else f'{kind}_f{fold}_s42'
                cm = read_json(ROOT / 'runs' / cnn / 'run.json')
                assert set(cm['train_frame_ids']) == {r['frame_id'] for r in parts['train']}
                assert set(cm['val_frame_ids']) == {r['frame_id'] for r in parts['val']}
                filename = ROOT / 'reports' / f'{cnn}_test/predictions.jsonl'
            rows = read_jsonl(filename)
            assert len(rows) == len(expected) and {r['frame_id'] for r in rows} == expected
            assert all(r['class_name'] == frame_index[r['frame_id']]['class_name'] for r in rows)
            if fold == 'temporal':
                temporal[kind] = summarize(rows, classes)
            else:
                all_predictions[kind].extend(rows)
        if fold == '0':
            loc = read_json(ROOT / 'reports/fold0_test_v3/summary.json')['results'][run]['localization_test_role_field8']
            points = read_jsonl(ROOT / 'reports/fold0_test_v3' / f'{run}_test_localization_points.jsonl')
            assert {p['frame_id'] for p in points} == {p['frame_id'] for p in roles['test']}
        else:
            loc = read_json(ROOT / 'reports/extension_v3' / f'{run}_summary.json')['localization_test_role']
        if fold != 'temporal':
            assert loc['all']['visible_points'] == 48 and loc['all']['pck8'] == 1.0
            localization.append(dict(fold=fold, points=48, pck8=1.0, scope_metadata=loc['scope']))
        checks.append(dict(fold=fold, field_training=meta['field_train_budget'],
                           field_validation=meta['field_val_images'], test_frames=len(expected),
                           field_training_disjoint_from_test=True, cnn_training_matches_split=True))
    oof = {}
    for k, rows in all_predictions.items():
        assert len(rows) == 7104 and {r['frame_id'] for r in rows} == set(frame_index)
        m = summarize(rows, classes)
        oof[k] = {unit: {key: m[unit][key] for key in ['n', 'correct', 'abstained', 'coverage', 'accuracy_all']}
                  for unit in ['frame', 'session']}
    assert [oof[k]['session']['correct'] for k in all_predictions] == [361, 332, 360, 359]
    assert [temporal[k]['session']['correct'] for k in all_predictions] == [133, 108, 132, 132]
    original = ROOT / 'manuscript/논문초안_v3.docx'
    doc = Document(original)
    malformed = [{'paragraph': i+1, 'subscript_text': r.text} for i,p in enumerate(doc.paragraphs)
                 for r in p.runs if r.font.subscript and len(r.text) > 30]
    cost = read_json(ROOT / 'reports/annotation_cost_field40_v2.json')
    evidence = dict(created_at=now(), manuscript=str(original.relative_to(ROOT)),
                    original_sha256=sha256(original), original_md_sha256=sha256(ROOT/'manuscript/논문초안_v3.md'),
                    scope='Review of stored artifacts and code; no retraining or new inference; no full external replication.',
                    split_checks=checks, oof=oof,
                    temporal={k: {u: {v:m[u][v] for v in ['n','correct','abstained','coverage','accuracy_all']}
                                  for u in ['frame','session']} for k,m in temporal.items()},
                    test_localization=localization,
                    annotation_time_confirmed_seconds=cost['duration']['confirmed_seconds'],
                    proportional_train8_minutes=cost['duration']['confirmed_seconds']/40*8/60,
                    proportional_train8_plus_val8_minutes=cost['duration']['confirmed_seconds']/40*16/60,
                    depth_error_stats=read_json(ROOT/'reports/extension_v3/analysis.json')['error_depth'],
                    malformed_subscripts=malformed,
                    lab_only_all_frame_geometry_gates=dict(Counter(r['gate'] for r in read_jsonl(ROOT/'features/holes_lab_s42_full/rows.jsonl'))),
                    illustrative_wilson95={'six_of_six':wilson(6,6),'nine_of_nine':wilson(9,9)},
                    note='Correct TEST point inputs were used, but localization scope strings inherit an outdated VAL description. Correct this metadata separately; it is not evidence of split leakage.')
    DEST.mkdir(exist_ok=True)
    dump(DEST / '근거대조.json', evidence)
    print({'split_checks': len(checks), 'oof_frames_per_method':7104, 'oof_sessions':361,
           'temporal_sessions':133, 'manuscript_sha256':evidence['original_sha256'],
           'malformed_subscript_paragraphs':[r['paragraph'] for r in malformed]})


if __name__ == '__main__':
    main()
