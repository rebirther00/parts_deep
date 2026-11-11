"""Column-width figure of the eight door shape classes; writes figures/v12c_fig_classes.{png,pdf}.

One fold-0 TEST image per class, chosen by a fixed rule (first session in sorted order, middle frame), same crop for all
classes so that the relative size is preserved. Rows are grouped by position (LH FRT / LH RR / RH) to show that doors of
the same position differ mainly in width. No detection or decision output is drawn.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_figures_v3 as base   # noqa: E402  (sets runtime, style, helpers)
import cv2
import matplotlib.pyplot as plt

from door8.core import partition, read_json, snapshot

plt.rcParams.update({'font.size': 7})
W = 3.3
ROWS = [['E25_door_LH_FRT', 'E30_door_LH_FRT', 'E38_door_LH_FRT'],
        ['E25_door_LH_RR', 'E30_door_LH_RR', 'E38_door_LH_RR'],
        ['E25_door_RH', 'E30_E38_door_RH', None]]
NAME = {'E25_door_LH_FRT': 'E25 LH FRT', 'E30_door_LH_FRT': 'E30 LH FRT', 'E38_door_LH_FRT': 'E38 LH FRT', 'E25_door_LH_RR': 'E25 LH RR',
        'E30_door_LH_RR': 'E30 LH RR', 'E38_door_LH_RR': 'E38 LH RR', 'E25_door_RH': 'E25 RH', 'E30_E38_door_RH': 'E30/E38 RH'}
CROP = (200, 1780, 60, 1190)          # x0, x1, y0, y1 in the 1920x1200 image, identical for every class


if __name__ == '__main__':
    sp, _, frames, splits = snapshot(); cfg = read_json(sp / 'protocol.json'); parts = partition(frames, splits, '0')
    cad = dict(zip(cfg['classes'], cfg['cad_hole_distance_mm'])); x0, x1, y0, y1 = CROP
    fig, axes = plt.subplots(3, 3, figsize=(W, W * (y1 - y0) / (x1 - x0) + .75), gridspec_kw=dict(wspace=.04, hspace=.34))
    for i, row in enumerate(ROWS):
        for j, c in enumerate(row):
            ax = axes[i][j]; ax.axis('off')
            if c is None:
                continue
            rows = sorted([r for r in parts['test'] if r['class_name'] == c], key=lambda r: (r['group_id'], r['frame_id']))
            rr = [r for r in rows if r['group_id'] == rows[0]['group_id']]; r = rr[len(rr) // 2]
            im = cv2.cvtColor(cv2.imread(r['rgb']), cv2.COLOR_BGR2RGB)
            ax.imshow(im[y0:y1, x0:x1]); ax.set_title(f'{NAME[c]}\nD = {cad[c]:.0f} mm', fontsize=6, color=base.INK, pad=2, linespacing=1.05)
            print(c, r['frame_id'], r['group_id'])
    base.save(fig, 'v12c_fig_classes'); print('ok')
