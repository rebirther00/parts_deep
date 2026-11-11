"""Threshold-free frame AUROC for the E23 rows of manuscript v5 Table 7 (stored scores only; no model is run).

Same definition as the leave-one-class-out runs (scripts/run_extension_v5.py, auroc):
known = fold-0 test frames (1,478), unknown = six E23 doors (120 frames), eight classes registered.
CNN score = max softmax; hole score = minus the distance to the nearest registered CAD value (gate-failed frames = -inf).
Added after the E23 protocol (e23_openset_v3) as a descriptive metric; it does not change any decision.
Writes reports/extension_v5/e23_auroc.json.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from door8.core import ROOT, dump, now, read_json, read_jsonl, snapshot
from run_extension_v5 import auroc
import analyze_coverage_v5 as cov

F0 = ROOT / 'reports/fold0_test_v3'
E23 = ROOT / 'reports/e23_openset_v3'


def main():
    sp, _, _, _ = snapshot(); cfg = read_json(sp / 'protocol.json'); cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm']))
    out = dict(created_at=now(), known='fold-0 test frames', unknown='E23 LH FRT, 6 doors, 120 frames')
    known = read_jsonl(F0 / 'holes_field8_f0_s42_v2_pixel_test_predictions.jsonl')
    unk = read_jsonl(E23 / 'holes_field8_f0_unregistered_predictions.jsonl')
    ks, _, kv = cov.hole_scores(known, cad); us, _, uv = cov.hole_scores([dict(r, class_name=None) for r in unk], cad)
    ks = np.where(kv, -ks, -np.inf); us = np.where(uv, -us, -np.inf)
    out['hole'] = dict(auroc=auroc(ks, us), n_known=int(len(ks)), n_unknown=int(len(us)), unknown_min_distance_mm=float(-us.max()))
    for kind in ('rgb', 'rgbe'):
        kc = [r['confidence'] for r in cov.cnn_rows(kind, 'f0')]
        uc = [r['confidence'] for r in read_jsonl(E23 / f'{kind}_f0_s42_predictions.jsonl')]
        out[kind] = dict(auroc=auroc(kc, uc), n_known=len(kc), n_unknown=len(uc), unknown_max_conf=float(max(uc)), known_min_conf=float(min(kc)))
    dump(ROOT / 'reports/extension_v5/e23_auroc.json', out)
    print(out)


if __name__ == '__main__':
    main()
