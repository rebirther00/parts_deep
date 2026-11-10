"""Read v4's native HWPX and recompute its new descriptive curves; no model runs."""
from pathlib import Path
import re
import sys
import zipfile
from lxml import etree
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from door8.core import ROOT, snapshot, read_json, read_jsonl, sha256, dump, now
import analyze_coverage_v4 as coverage

DEST = ROOT / 'reports/논문초안_v4_리뷰_20261109'
NS = {'hp': 'http://www.hancom.co.kr/hwpml/2011/paragraph'}


def main():
    DEST.mkdir(exist_ok=True)
    original_paths = [ROOT / 'manuscript' / ('논문초안_v4.' + ext) for ext in ['hwpx', 'md', 'docx', 'pdf']]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in original_paths}
    with zipfile.ZipFile(original_paths[0]) as z:
        root = etree.fromstring(z.read('Contents/section0.xml'))
        paragraphs = [{'index': i, 'id': p.get('id'), 'text': ''.join(p.xpath('.//hp:t/text()', namespaces=NS))}
                      for i, p in enumerate(root.findall('hp:p', NS))]
        flat = '\n'.join(p['text'] for p in paragraphs)
        table_count = len(root.xpath('//hp:tbl', namespaces=NS))
        picture_count = len(root.xpath('//hp:pic', namespaces=NS))
    dump(DEST / 'HWPX문단.json', paragraphs)
    (DEST / 'HWPX본문추출.txt').write_text(flat, encoding='utf8')
    md = original_paths[1].read_text()
    abstract = md.split('## ABSTRACT')[1].split('Key Words:')[0].strip()
    assert abstract in flat
    expected_ranges = ['−1.14%에서 +1.11%', '1,474~1,475 mm', '0.99955~1.0']
    assert all(t in flat for t in expected_ranges)
    sp, _, _, _ = snapshot()
    cfg = read_json(sp / 'protocol.json')
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm']))
    saved = read_json(ROOT / 'reports/coverage_v4/coverage.json')
    curve_checks = {}
    for setting in ['oof', 'temporal']:
        curves = coverage.setting_curves(setting, cad)
        summaries = {name: coverage.summarize_curve(c, name, 30 if name.startswith('Hole') else None)
                     for name, c in curves.items()}
        for old in saved['settings'][setting]['curves']:
            assert summaries[old['method']] == old
        curve_checks[setting] = summaries
    thresholds = []
    marker_policies = {}
    for kind in ['rgb', 'rgbe']:
        accepted, rejected = [], []
        for split in ['f0', 'f1', 'f2', 'f3', 'f4', 'temporal']:
            threshold = read_json(ROOT / f'reports/{kind}_{split}_s42_test/report.json')['val95coverage_threshold']
            val = read_jsonl(ROOT / f'runs/{kind}_{split}_s42/val_predictions.jsonl')
            thresholds.append(dict(kind=kind, split=split, threshold=threshold,
                                   achieved_validation_coverage=sum(r['confidence'] >= threshold for r in val) / len(val)))
            if split != 'temporal':
                for row in read_jsonl(ROOT / f'reports/{kind}_{split}_s42_test/predictions_val95coverage.jsonl'):
                    (accepted if row['pred'] is not None else rejected).append(row['confidence'])
        marker_policies[kind] = dict(min_accepted_confidence=min(accepted), max_rejected_confidence=max(rejected),
                                    reproducible_by_single_raw_threshold=max(rejected) < min(accepted))
    rgb = coverage.setting_curves('oof', cad)['RGB CNN']
    oof_first_points = [dict(threshold=float(rgb['threshold'][i]), coverage=float(rgb['coverage'][i]),
                            risk=float(rgb['risk'][i])) for i in range(3)]
    tm = read_json(ROOT / 'runs/holes_field_temporal_s42/run.json')
    f0 = read_json(ROOT / 'runs/holes_field8_f0_s42_v2/run.json')
    evidence = dict(checked_at=now(), original_sha256=hashes, native_hwpx_read=True,
                    top_level_paragraphs=len(paragraphs), tables=table_count, figures=picture_count,
                    abstract_words=len(abstract.split()), range_strings_preserved=expected_ranges,
                    field_supervision={'cv': [f0['field_train_budget'], f0['field_val_images']],
                                       'temporal': [tm['field_train_budget'], tm['field_val_images']]},
                    reproduced_curve_summaries=curve_checks, validation_thresholds=thresholds,
                    pooled_oof_marker_policy=marker_policies, oof_rgb_first_attainable_points=oof_first_points,
                    risk_definition='Misclassified accepted frames / all accepted frames; coverage denominator includes abstentions.',
                    pdf_scope='Supplied PDF is a LibreOffice rendering of companion DOCX; it is not a native Hancom HWPX rendering.',
                    agency_source='https://www.keit.re.kr/', agency_current_name='한국산업기술기획평가원',
                    scientific_scope='Stored-output review; no new training, prospective data collection, or additional seed evaluation.')
    dump(DEST / '근거대조.json', evidence)
    print({'native_hwpx': True, 'abstract_words': len(abstract.split()), 'temporal_supervision': evidence['field_supervision'],
           'curves_match': True, 'oof_marker_policies': marker_policies})


if __name__ == '__main__':
    main()
