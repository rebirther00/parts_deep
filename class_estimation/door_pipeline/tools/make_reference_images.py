"""클래스×레이더 대표 이미지 생성 → reference_images/<class>__radar{0,1}.jpg (FRT·E23 은 <class>.jpg).

추론 서비스(07) 와 webapp 이 "판정 대표 이미지 옆에 작업자 라벨 데이터의 대표 이미지"를 나란히 보여 주기 위한 정적 세트.
출처: db/decisions.csv(세션별 사람 판단: 정정 클래스·유효·레이더 확정) + 로컬 미러 datasets_factory_collect/.
선정: 유효 세션 중 레이더가 user 확정인 세션 우선 → 세션 중간 프레임들 중 홀 판별기 게이트 'ok' 이고 예측이 라벨과 일치하며
마진이 가장 큰 프레임. 960px 폭 jpg(점 오버레이 없음) + 같은 이름 *_pts.jpg(6점 오버레이) 두 벌.

    python tools/make_reference_images.py            # 전체 (없는 것만)
    python tools/make_reference_images.py --force    # 다시 생성
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

DOOR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DOOR))
import hole_classifier as hc                               # noqa: E402

MIRROR = DOOR / 'datasets_factory_collect'
OUT = DOOR / 'reference_images'
COLORS = (('bolt', (255, 0, 0)), ('corner_hinge', (0, 0, 255)), ('corner_latch', (0, 140, 255)), ('bracket', (255, 0, 255)))


def targets():
    for cls in hc.CAD_D:
        if hc.GROUP[cls] == 'FRT':
            yield cls, None, f'{cls}.jpg'
        else:
            for r in (0, 1):
                yield cls, r, f'{cls}__radar{r}.jpg'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--per-session', type=int, default=6, help='세션당 검사 프레임 수')
    ap.add_argument('--max-sessions', type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(DOOR / 'db' / 'decisions.csv', encoding='utf-8')))
    net, dev = hc.load_model(hc.MODEL_PATH)
    manifest = {}
    for cls, radar, name in targets():
        if (OUT / name).exists() and not args.force:
            print(f'skip {name} (exists)')
            continue
        cands = [r for r in rows if r['valid'] == '1' and r['class_db'] == cls and (MIRROR / r['session_dir']).is_dir()
                 and (radar is None or r['option_radar'] == str(radar))]
        cands.sort(key=lambda r: (r['option_source'] != 'user', r['session_dir']), reverse=False)
        cands = sorted(cands, key=lambda r: (0 if r['option_source'] == 'user' else 1, -int(r['n_local'] or 0)))[:args.max_sessions]
        best = None
        for r in cands:
            sd = MIRROR / r['session_dir']
            meta = json.loads((sd / 'meta.json').read_text()) if (sd / 'meta.json').exists() else {}
            intr = meta.get('intrinsics')
            rgbs = sorted(sd.glob('rgb_*.png'))
            if not rgbs:
                continue
            idx = np.linspace(len(rgbs) * 0.2, len(rgbs) * 0.8, args.per_session).astype(int)
            for i in sorted(set(idx)):
                rp = rgbs[i]
                dp = sd / rp.name.replace('rgb_', 'depth_')
                rgb = cv2.imread(str(rp))
                depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED).astype(np.float32) if dp.exists() else None
                hr = hc.classify(net, dev, rgb, depth, intrinsics=intr)
                if hr['gate'] != 'ok' or hr['pred'] != cls:
                    continue
                m = hr.get('margin_mm') or 0
                if best is None or m > best[0]:
                    best = (m, r['session_dir'], rp.name, rgb, hr)
        if best is None:
            print(f'!! {name}: 적합한 프레임 없음 (후보 세션 {len(cands)})')
            continue
        m, sdir, fname, rgb, hr = best
        h, w = rgb.shape[:2]
        sc = 960 / w
        img = cv2.resize(rgb, (960, int(h * sc)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(OUT / name), img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        ov = img.copy()
        for c, col in COLORS:
            for p in hr['points'].get(c, []):
                cv2.circle(ov, (int(p[0] * sc), int(p[1] * sc)), 5, col, 2)
        cv2.imwrite(str(OUT / name.replace('.jpg', '_pts.jpg')), ov, [cv2.IMWRITE_JPEG_QUALITY, 82])
        manifest[name] = {'class': cls, 'radar': radar, 'session_dir': sdir, 'file': fname,
                          'D_mm': round(hr['D_mm'], 1), 'margin_mm': round(m, 1), 'part': None}
        print(f'{name}: {sdir}/{fname} D={hr["D_mm"]:.0f} margin={m:.0f}')
    mp = OUT / 'manifest.json'
    old = json.loads(mp.read_text(encoding='utf-8')) if mp.exists() else {}
    old.update(manifest)
    mp.write_text(json.dumps(old, indent=1, ensure_ascii=False), encoding='utf-8')
    print('manifest:', mp)


if __name__ == '__main__':
    main()
