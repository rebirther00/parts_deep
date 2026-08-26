#!/usr/bin/env python
"""단일 이미지(rgb/depth) 판정 CLI.  사용: python scripts/hole_classify_image.py rgb_0003.png [depth_0003.png] [--out result.jpg]"""
import argparse, os, sys, json
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hole_classifier import load_model, classify, CAD_D
ap = argparse.ArgumentParser(); ap.add_argument('rgb'); ap.add_argument('depth', nargs='?'); ap.add_argument('--out'); a = ap.parse_args()
rgb = cv2.imread(a.rgb); dp = a.depth or a.rgb.replace('rgb_', 'depth_')
depth = cv2.imread(dp, cv2.IMREAD_UNCHANGED) if os.path.exists(dp) else None
net, dev = load_model(); r = classify(net, dev, rgb, depth)
print(json.dumps({k: (v if k != 'points' else {c: [[round(x, 1) for x in p] for p in pts] for c, pts in v.items()}) for k, v in r.items()}, ensure_ascii=False, indent=1, default=float))
if a.out:
    for c, col in (('bolt', (255, 0, 0)), ('corner_hinge', (0, 0, 255)), ('corner_latch', (0, 140, 255))):
        for p in r['points'][c]: cv2.circle(rgb, (int(p[0]), int(p[1])), 9, col, 2)
    cv2.putText(rgb, f"{r['pred']} D={r['D_mm'] and round(r['D_mm'])}mm gate={r['gate']}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    cv2.imwrite(a.out, rgb); print('saved', a.out)
