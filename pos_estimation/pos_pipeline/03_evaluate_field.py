"""현장/대차 세션 포즈 평가 — GT 없는 프록시 지표 (정의 2026-08-31).

  정합 정확도(우선): 홀별 CAD 정합 잔차 RMS(mm)   — 면내 지표
  반복 정밀도: 세션(정지 연속 프레임) 내 x/y/z/θ/tilt 표준편차
  z·틸트 신뢰도: depth 평면 inlier RMS·픽셀 수
절대 정확도는 여기서 주장하지 않음(합성/레이저 트래커 검증 몫).

  python 03_evaluate_field.py                                  # datasets_factory_v2/all
  python 03_evaluate_field.py --base datasets_field
  python 03_evaluate_field.py --base datasets_factory_collect  # <date>/<cls>/s_*/ 구조

출력: artifacts/eval_field_<base>.json + artifacts/overlays/<세션>_<idx>.jpg (CAD 재투영)
"""
import argparse
import collections
import glob
import json
import os
import time

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
import pose_solver as ps
import pose_utils as pu

DOOR = ps.DOOR
ap = argparse.ArgumentParser()
ap.add_argument('--base', default='datasets_factory_v2/all')
ap.add_argument('--per_session', type=int, default=20)
ap.add_argument('--overlays', type=int, default=2, help='세션당 오버레이 저장 수(첫/최악)')
args = ap.parse_args()
OUT_DIR = os.path.join(BASE, 'artifacts')
OVL_DIR = os.path.join(OUT_DIR, 'overlays')


def sessions_for(base):
    """{(클래스, 세션키): [rgb 파일]} — v2 링크 뷰 / 수집 / 현장 구조 지원."""
    b = os.path.join(DOOR, base)
    out = {}
    for f in glob.glob(os.path.join(b, '*', 'rgb_*_s_*_*.png')):          # v2: cls/rgb_<date>_s_<sess>_<idx>
        cls = os.path.basename(os.path.dirname(f))
        sess = os.path.basename(f)[4:-4].rsplit('_', 1)[0]
        out.setdefault((cls, f'{cls}__{sess}'), set()).add(f)
    for f in glob.glob(os.path.join(b, '*', '*', 's_*', 'rgb_*.png')):    # 수집: date/cls/s_*/rgb_<idx>
        sd = os.path.dirname(f)
        cls = os.path.basename(os.path.dirname(sd))
        out.setdefault((cls, os.path.relpath(sd, b).replace(os.sep, '__')), set()).add(f)
    for f in glob.glob(os.path.join(b, '*', 'rgb_*.png')):                # 현장: <cls>_s_*/rgb_<idx>
        d = os.path.basename(os.path.dirname(f))
        if '_s_' in d:
            out.setdefault((d.split('_s_')[0], d), set()).add(f)
    return {k: sorted(v) for k, v in sorted(out.items()) if k[0] in ps.hc.CAD_D}


def reproject(p_door, R, t, K):
    q = R @ p_door + t
    return (K['fx'] * q[0] / q[2] + K['cx'], K['fy'] * q[1] / q[2] + K['cy'])


def overlay(rgb, f, cad_pts, K, title):
    img = rgb.copy()
    for b in f['det']['bolt']:
        cv2.drawMarker(img, (int(b[0]), int(b[1])), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)
    for k, col in (('corner_hinge', (60, 60, 239)), ('corner_latch', (8, 151, 249))):
        if f['det'][k]:
            p = f['det'][k][0]
            cv2.drawMarker(img, (int(p[0]), int(p[1])), col, cv2.MARKER_CROSS, 22, 2)
    R, t = f['R'], f['t']
    for k, p in cad_pts.items():
        u, v = reproject(p, R, t, K)
        cv2.circle(img, (int(u), int(v)), 10, (80, 220, 80), 2)
    o = np.zeros(3)
    for ax, col in ((np.array([150, 0, 0]), (0, 0, 255)), (np.array([0, 150, 0]), (0, 255, 0)),
                    (np.array([0, 0, 150]), (255, 0, 0))):
        p0, p1 = reproject(o, R, t, K), reproject(ax, R, t, K)
        cv2.arrowedLine(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), col, 2, tipLength=0.15)
    cv2.putText(img, title, (16, 34), 0, 0.9, (0, 0, 0), 4)
    cv2.putText(img, title, (16, 34), 0, 0.9, (255, 255, 255), 2)
    return img


def main():
    os.makedirs(OVL_DIR, exist_ok=True)
    cad = ps.load_cad()
    net, dev = ps.hc.load_model()
    sessions = sessions_for(args.base)
    assert sessions, f'세션 없음: {args.base}'
    t0 = time.time()
    rows = []
    print(f"{'세션':42s} {'n':>3s} {'판정':>4s} {'잔차med':>7s} | 반복 std: {'x':>5s} {'y':>5s} {'z':>5s} {'θ°':>5s} {'tilt°':>5s}")
    for (cls, key), files in sessions.items():
        files = files[::max(1, len(files) // args.per_session)][:args.per_session]
        frames, imgs = [], []
        for fp in files:
            rgb = cv2.imread(fp)
            depth = cv2.imread(fp.replace('rgb_', 'depth_'), cv2.IMREAD_UNCHANGED)
            if rgb is None or depth is None:
                continue
            f = ps.solve(net, dev, rgb, depth, cls, cad)
            frames.append(f); imgs.append((fp, rgb))
        agg = ps.aggregate_poses(frames)
        ok_idx = [i for i, f in enumerate(frames) if f.get('ok') and f.get('gate')]
        if ok_idx and args.overlays:
            K = pu.intrinsics_for(imgs[0][1].shape)
            picks = {ok_idx[0], max(ok_idx, key=lambda i: frames[i]['rms'])}
            for i in list(picks)[:args.overlays]:
                f = frames[i]
                ttl = (f"{cls}  rms {f['rms']:.1f}mm  th {f['theta_deg']:+.1f}  tilt {f['tilt_deg']:.1f}  "
                       f"z {f['t'][2]:.0f}mm  holes {f['n_holes']}")
                img = overlay(imgs[i][1], f, cad[cls]['pts'], K, ttl)
                cv2.imwrite(os.path.join(OVL_DIR, f"{key}_{os.path.basename(imgs[i][0])[:-4]}.jpg"),
                            img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        reasons = collections.Counter(f.get('reason', 'ok') for f in frames if not (f.get('ok') and f.get('gate')))
        row = dict(cls=cls, session=key, n=len(frames), n_used=agg.get('n_used', 0),
                   agg={k: v for k, v in agg.items() if k not in ('T',)}, fail=dict(reasons),
                   rms_all=[round(f['rms'], 2) for f in frames if f.get('ok')])
        rows.append(row)
        if agg.get('ok'):
            s = agg['std']
            print(f"{key:42s} {row['n']:3d} {row['n_used']:4d} {agg['rms_med']:7.2f} | "
                  f"{s['x']:10.2f} {s['y']:5.2f} {s['z']:5.2f} {s['theta']:5.2f} {s['tilt']:5.2f}")
        else:
            print(f"{key:42s} {row['n']:3d}    0       - | 실패 {dict(reasons)}")

    # 클래스/전체 요약
    print('\n[클래스 요약]  잔차=정합 정확도(우선지표), std=반복 정밀도  (목표 ±5mm/±1°)')
    print(f"{'클래스':17s} {'세션':>4s} {'프레임':>5s} {'판정':>5s} {'잔차med':>7s} {'잔차max':>7s} | "
          f"med std: {'x':>5s} {'y':>5s} {'z':>5s} {'θ°':>5s} {'tilt°':>5s}")
    summary = {}
    for cls in sorted({r['cls'] for r in rows}):
        rs = [r for r in rows if r['cls'] == cls]
        ok = [r for r in rows if r['cls'] == cls and r['agg'].get('ok')]
        rmss = [x for r in rs for x in r['rms_all']]
        if ok:
            meds = {a: float(np.median([r['agg']['std'][a] for r in ok])) for a in ('x', 'y', 'z', 'theta', 'tilt')}
            print(f"{cls:17s} {len(rs):4d} {sum(r['n'] for r in rs):5d} {sum(r['n_used'] for r in rs):5d} "
                  f"{np.median(rmss):7.2f} {max(rmss):7.2f} | "
                  f"{meds['x']:10.2f} {meds['y']:5.2f} {meds['z']:5.2f} {meds['theta']:5.2f} {meds['tilt']:5.2f}")
        else:
            meds = None
            print(f"{cls:17s} {len(rs):4d} {sum(r['n'] for r in rs):5d}     0       -       - |")
        summary[cls] = dict(sessions=len(rs), frames=sum(r['n'] for r in rs),
                            used=sum(r['n_used'] for r in rs),
                            rms_med=float(np.median(rmss)) if rmss else None,
                            rms_max=float(max(rmss)) if rmss else None, std_med=meds)
    out = dict(base=args.base, per_session=args.per_session, rms_gate=ps.RMS_GATE,
               elapsed_s=round(time.time() - t0, 1), sessions=rows, summary=summary)
    name = 'eval_field_' + args.base.strip('/').replace('/', '_') + '.json'
    json.dump(out, open(os.path.join(OUT_DIR, name), 'w'), ensure_ascii=False, indent=1, default=float)
    print(f"\n→ {os.path.join(OUT_DIR, name)}\n→ {OVL_DIR}/")


if __name__ == '__main__':
    main()
