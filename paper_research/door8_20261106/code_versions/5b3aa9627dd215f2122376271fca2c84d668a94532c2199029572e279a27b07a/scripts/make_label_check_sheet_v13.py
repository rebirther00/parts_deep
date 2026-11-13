"""Contact sheet for the author's visual check of the two doors the CNNs classified differently from the operator input.

Writes reports/label_visual_check_20261111/: sheet.png (two target doors, three E25 LH FRT doors, three E30 LH FRT doors,
same crop) and one full-resolution image per door. The measured D in the labels is the stored five-fold out-of-fold value
of the field-adapted hole detector; nothing is recomputed here and no decision is made by this script.
"""
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from door8.core import ROOT, snapshot

OUT = ROOT / 'reports/label_visual_check_20261111'
CROP = (200, 1780, 60, 1190)                       # same crop as figures/v12c_fig_classes
TARGETS = ['20261022/E25_door_LH_FRT/s_084658', '20261022/E25_door_LH_FRT/s_123253']
E25 = ['20261001/E25_door_LH_FRT/s_142956', '20261026/E25_door_LH_FRT/s_100727', '20261029/E25_door_LH_FRT/s_123610']
E30 = ['20261021/E30_door_LH_FRT/s_125151', '20261021/E30_door_LH_FRT/s_131627', '20261027/E30_door_LH_FRT/s_074619']
PRED = [ROOT / f'reports/extension_v3/holes_field8_f{k}_s42_v3_pixel_predictions.jsonl' for k in '1234'] \
    + [ROOT / 'reports/fold0_test_v3/holes_field8_f0_s42_v2_pixel_test_predictions.jsonl']


def measured():
    d = {}
    for f in PRED:
        for line in open(f):
            r = json.loads(line)
            if r.get('measurement') is not None:
                d.setdefault(r['group_id'], []).append(r['measurement'])
    return {g: statistics.median(v) for g, v in d.items()}


def panel(row, title, colour, scale=0.5):
    x0, x1, y0, y1 = CROP; im = cv2.imread(row['rgb'])[y0:y1, x0:x1]
    im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    bar = np.full((64, im.shape[1], 3), 255, np.uint8); cv2.rectangle(bar, (0, 0), (10, 64), colour, -1)
    for k, text in enumerate(title):
        cv2.putText(bar, text, (20, 24 + 28 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (30, 30, 30), 1, cv2.LINE_AA)
    return np.vstack([bar, im])


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True); _, _, frames, _ = snapshot(); d = measured(); by = {}
    for r in frames:
        by.setdefault(r['group_id'], []).append(r)
    pick = lambda g: sorted(by[g], key=lambda r: r['frame_id'])[len(by[g]) // 2]
    name = lambda g: g.split('/')[0][:4] + '-' + g.split('/')[0][4:6] + '-' + g.split('/')[0][6:] + ' ' + g.split('/')[2]
    rows = []
    for groups, kind, colour in ((TARGETS, 'CHECK: operator E25 LH FRT, CNN said E30 LH FRT', (40, 40, 220)),
                                 (E25, 'reference E25 LH FRT (CAD D 724 mm)', (60, 160, 40)),
                                 (E30, 'reference E30 LH FRT (CAD D 765 mm)', (200, 120, 30))):
        cells = []
        for g in groups:
            r = pick(g); cells.append(panel(r, [kind, f'{name(g)}   hole D = {d[g]:.1f} mm'], colour))
            cv2.imwrite(str(OUT / f"{g.replace('/', '_')}.png"), cv2.imread(r['rgb'])); print(g, r['frame_id'], round(d[g], 1))
        while len(cells) < 3:
            cells.append(np.full_like(cells[0], 255))
        rows.append(np.hstack([np.hstack([c, np.full((c.shape[0], 12, 3), 255, np.uint8)]) for c in cells]))
    sheet = np.vstack([np.vstack([r, np.full((14, r.shape[1], 3), 255, np.uint8)]) for r in rows])
    cv2.imwrite(str(OUT / 'sheet.png'), sheet); print(OUT / 'sheet.png', sheet.shape)
