"""Images around the camera bump and re-adjustment of September 2026, for the author's visual check.

Writes reports/camera_readjustment_202609/: sheet.png (four E38 LH RR sessions: normal, bumped x2, re-adjusted), two
red/cyan overlays against the normal view, and the four full-resolution frames. Session states follow the operational
records (pixel scale s_session, depth-plane tilt); this script only reads images and draws them.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from door8.core import ROOT, snapshot

OUT = ROOT / 'reports/camera_readjustment_202609'
SESSIONS = [('20260917/E38_door_LH_RR/s_173807', 'normal            2026-09-17 17:38   scale 1478.2 mm   tilt 4.9 deg', (60, 160, 40)),
            ('20260921/E38_door_LH_RR/s_155719', 'after the bump    2026-09-21 15:57   scale 1485.8 mm   tilt 4.4 deg', (40, 40, 220)),
            ('20260922/E38_door_LH_RR/s_080559', 'after the bump    2026-09-22 08:05   scale 1488.9 mm   tilt 5.0 deg', (40, 40, 220)),
            ('20260922/E38_door_LH_RR/s_081005', 're-adjusted       2026-09-22 08:10   scale 1471.2 mm   tilt 7.4 deg', (200, 120, 30))]


def label(im, text, colour):
    bar = np.full((44, im.shape[1], 3), 255, np.uint8); cv2.rectangle(bar, (0, 0), (10, 44), colour, -1)
    cv2.putText(bar, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2, cv2.LINE_AA)
    return np.vstack([bar, im])


def overlay(a, b, text):
    """a in cyan, b in red: parts of the scene that did not move look gray, parts that moved show coloured fringes."""
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    return label(cv2.merge([ga, ga, gb]), text, (90, 90, 90))


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True); _, _, frames, _ = snapshot(); by = {}
    for r in frames:
        by.setdefault(r['group_id'], []).append(r)
    ims = []
    for g, text, colour in SESSIONS:
        rr = sorted(by[g], key=lambda r: r['frame_id']); r = rr[len(rr) // 2]; im = cv2.imread(r['rgb'])
        cv2.imwrite(str(OUT / f"{g.replace('/', '_')}.png"), im); ims.append(im); print(g, r['frame_id'], len(rr))
    half = [label(cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), text, colour) for im, (_, text, colour) in zip(ims, SESSIONS)]
    gap = np.full((half[0].shape[0], 12, 3), 255, np.uint8)
    sheet = np.vstack([np.hstack([half[0], gap, half[1]]), np.full((12, 2 * half[0].shape[1] + 12, 3), 255, np.uint8), np.hstack([half[2], gap, half[3]])])
    cv2.imwrite(str(OUT / 'sheet.png'), sheet)
    cv2.imwrite(str(OUT / 'overlay_normal_vs_bumped.png'), overlay(ims[0], ims[1], 'cyan: normal 2026-09-17   red: after the bump 2026-09-21'))
    cv2.imwrite(str(OUT / 'overlay_normal_vs_readjusted.png'), overlay(ims[0], ims[3], 'cyan: normal 2026-09-17   red: re-adjusted 2026-09-22 08:10'))
    print(OUT, sheet.shape)
