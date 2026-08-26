"""보류(판정 안 함) 이미지 검토용 갤러리 생성.

eval_classifier.json의 보류 행을 사유별 폴더에 검출점·사유를 그려 저장하고 index.html을 만든다.
  python tools/make_abstain_review.py            # datasets_all 세트
  python -m http.server 8091 -d report/hole_analysis/abstain_review   → http://<IP>:8091
"""
import collections, json, os, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hole_classifier import load_model, classify, CAD_D

DOOR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(DOOR, 'report', 'hole_analysis', 'abstain_review')
SET = sys.argv[1] if len(sys.argv) > 1 else 'datasets_all'
E = json.load(open(os.path.join(DOOR, 'attribute_models', 'hole_landmarks', 'eval_classifier.json')))
rows = [r for r in E[SET]['rows'] if not r['pred'] and r['cls'] in CAD_D]
COL = {'bolt': (255, 0, 0), 'corner_hinge': (0, 0, 255), 'corner_latch': (0, 140, 255)}
DESC = {'no_corner': '모서리 홀 미검출 (힌지측 또는 래치측 홀이 없음/안 보임)',
        'near_border': '모서리 홀이 이미지 경계 20px 이내',
        'same_side': '두 모서리 홀이 볼트 사각형 기준 같은 쪽 (한쪽이 잘못 잡힘)',
        'not_collinear': '두 홀이 같은 선상이 아님',
        'latch_offset': '래치측 홀이 볼트 사각형 기준 예상 위치(±160, ±190mm)에서 벗어남',
        'no_frame': '볼트홀 4개가 157×96 직사각형을 이루지 못함',
        'D_range': 'D가 600~1500mm 범위 밖'}

net, dev = load_model()
by = collections.defaultdict(list)
for r in rows:
    f = os.path.join(DOOR, r['image']); rgb = cv2.imread(f)
    dp = f.replace('rgb_', 'depth_'); depth = cv2.imread(dp, cv2.IMREAD_UNCHANGED) if os.path.exists(dp) else None
    d = classify(net, dev, rgb, depth)
    v = rgb.copy()
    for c, pts in d['points'].items():
        for p in pts:
            cv2.circle(v, (int(p[0]), int(p[1])), 10, COL[c], 2)
            cv2.putText(v, f'{p[2]:.2f}', (int(p[0]) + 11, int(p[1]) - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL[c], 1)
    cv2.rectangle(v, (0, 0), (1100, 40), (0, 0, 0), -1)
    cv2.putText(v, f"[{d['gate']}] {r['cls']} {os.path.basename(f)}  D={d['D_mm'] and round(d['D_mm'])}  bolts={len(d['points']['bolt'])} hinge={len(d['points']['corner_hinge'])} latch={len(d['points']['corner_latch'])}",
                (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    sub = os.path.join(OUT, d['gate']); os.makedirs(sub, exist_ok=True)
    name = f"{r['cls']}_{os.path.basename(f)[4:8]}.jpg"
    cv2.imwrite(os.path.join(sub, name), cv2.resize(v, (1280, int(v.shape[0] * 1280 / v.shape[1]))), [cv2.IMWRITE_JPEG_QUALITY, 82])
    by[d['gate']].append((name, r['cls'], d['D_mm']))

html = ['<!doctype html><meta charset="utf-8"><title>보류 이미지 검토</title><style>body{font-family:sans-serif;background:#111;color:#eee;margin:16px}img{width:420px;margin:4px;border:1px solid #444}h2{margin-top:28px}a{color:#8ab4f8}.g{display:flex;flex-wrap:wrap}figure{margin:4px;text-align:center;font-size:12px}</style>',
        f'<h1>보류 {len(rows)}장 — {SET}</h1><p>파란 원 = 볼트홀, 빨간 원 = 힌지측 모서리 홀, 주황 원 = 래치측 모서리 홀 (숫자는 히트맵 점수). 클릭하면 원본 크기.</p>',
        '<p>' + ' · '.join(f'<a href="#{g}">{g} ({len(v)})</a>' for g, v in sorted(by.items(), key=lambda t: -len(t[1]))) + '</p>']
for g, v in sorted(by.items(), key=lambda t: -len(t[1])):
    html.append(f'<h2 id="{g}">{g} — {len(v)}장</h2><p>{DESC.get(g, "")}</p><div class="g">')
    for name, cls, D in sorted(v):
        html.append(f'<figure><a href="{g}/{name}" target="_blank"><img src="{g}/{name}" loading="lazy"></a><figcaption>{cls} {name[-8:-4]}{" D=%d" % D if D else ""}</figcaption></figure>')
    html.append('</div>')
open(os.path.join(OUT, 'index.html'), 'w').write('\n'.join(html))
print(f'{len(rows)}장 → {OUT}/index.html', {g: len(v) for g, v in by.items()})
