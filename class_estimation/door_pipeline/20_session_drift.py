"""세션별 홀 판별기 드리프트 지표 기록 → DB session_hole_metrics (webapp /drift 에서 시계열 열람).

  python 20_session_drift.py                 # DB 의 유효 현장 세션 중 미기록분만 계산
  python 20_session_drift.py --recompute     # 전부 다시 (K_CAMERA 갱신·모델 교체 후)
  python 20_session_drift.py --print         # 계산 없이 현재 표만 출력

지표(세션당, 로컬 동기화된 유효 프레임 ≤20장):
  k_session = CAD_D / median(D_raw)  — 카메라 depth 스케일 드리프트. 클래스 무관이라 전 세션을 한 축에 놓을 수 있다.
                                       기준 k_applied(hole_classifier.K_CAMERA) 대비 ±1% 밖이면 경보(2026-09-16 분석: 세션 std 0.63%).
  dev_mm    = median(D_mm) − CAD_D   — 판정 마진 소모. |dev| > 15mm 경보(FRT 최소 간격 41mm 의 1/3 이상).
  z_med, tilt_med                    — 도어 평면 깊이·기울기 (depth 편향과 −0.95 상관, 원인 추적용).
  s_session = CAD_D·fx/median(span_px) — 픽셀 폭 일관성(2026-09-24). 판정이 픽셀 폭 기준(hole_classifier.S_PIXEL)으로 바뀐 뒤의 주 지표:
                                       S_PIXEL 대비 ±0.5% 밖이면 경보(기준 세션 std 0.16%). dev_pix_mm = 픽셀 D 중앙값 − CAD_D 가 현행 판정의 마진 소모.
                                       K_session·dev_mm 은 depth 경로 참고 지표로 유지.
  pred_major / n_pred_major          — 프레임별 판정(현행 픽셀 폭 D) 다수결 클래스와 표 수(2026-09-23). build_dataset.py auto-relabel 이
                                       라벨과 다수결이 크게 어긋난 세션(CAD D 차 ≥100mm·표 10장·일치 90%)을 자동 재배정하는 근거.
D 계산 경로는 17/18 과 동일(세션 meta intrinsics + K_CAMERA). 라벨은 DB 정정 라벨(classes.name), Unknown·is_valid=0 세션 제외.
"""
import argparse, collections, os, sqlite3, sys, time
import cv2, numpy as np

DOOR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DOOR); sys.path.insert(0, os.path.join(DOOR, 'db'))
import hole_classifier as hc
from hole_classifier import CAD_D
from camera_utils import intrinsics_for_image
import build_dataset as bd
from migrate_session_metrics import ensure

ap = argparse.ArgumentParser()
ap.add_argument('--db', default=os.path.join(DOOR, 'db', 'door_pipeline.db'))
ap.add_argument('--model', default=hc.MODEL_PATH)
ap.add_argument('--recompute', action='store_true')
ap.add_argument('--print', action='store_true', help='계산 없이 표만')
ap.add_argument('--limit', type=int, default=0, help='세션 수 제한(디버그)')
args = ap.parse_args()
DEV_WARN, K_WARN, S_WARN = 15.0, 0.01, 0.005


def session_rows(con):
    ds = con.execute("SELECT id FROM datasets WHERE name=?", (bd.DATASET_NAME,)).fetchone()[0]
    return con.execute(
        """SELECT s.id sid, s.session_dir, s.started_at, c.name cls,
                  GROUP_CONCAT(i.rgb_path) rgbs, GROUP_CONCAT(i.depth_path) depths
           FROM images i JOIN classes c ON c.id=i.class_id JOIN capture_sessions s ON s.id=i.session_id
           WHERE c.dataset_id=? AND i.synced_local AND i.is_valid AND c.name!='Unknown'
           GROUP BY s.id ORDER BY s.started_at""", (ds,)).fetchall()


def compute(net, dev, r):
    rows = []
    for rp, dp in zip(r['rgbs'].split(','), r['depths'].split(',')):
        rgb = cv2.imread(str(bd.MIRROR / rp)); depth = cv2.imread(str(bd.MIRROR / dp), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None: continue
        intr = intrinsics_for_image(str(bd.MIRROR / rp), rgb.shape)
        det = hc.detect(net, dev, rgb)
        hinge = det['corner_hinge'][0] if det['corner_hinge'] else None
        latch = det['corner_latch'][0] if det['corner_latch'] else None
        gate = hc.geometry_gate(hc.bolt_frame(det['bolt']), hinge, latch, rgb.shape)
        m = hc.measure_D(depth, det['bolt'] + [hinge, latch], hinge, latch, intr) if gate == 'ok' else None
        if m and not (hc.D_RANGE[0] <= m['D_mm'] <= hc.D_RANGE[1]): m = None
        pred = nearest = margin = None
        if m: pred, nearest, margin = hc.judge(m['D_mm'])
        span = D_pix = margin_pix = None
        sn = (intr or {}).get('serial'); fx = (intr or {}).get('fx')
        if gate == 'ok' and hinge and latch:
            span = float(np.hypot(hinge[0] - latch[0], hinge[1] - latch[1]))
            if sn in hc.S_PIXEL and fx:
                D_pix = span * hc.S_PIXEL[sn] / fx; margin_pix = hc.judge(D_pix)[2]
        # 현행 판정(SCALE_DEFAULT='pixel')과 같은 D 로 프레임 클래스 — 세션 다수결(pred_major, 자동 재배정 근거)
        pred_use = hc.judge(D_pix)[0] if (D_pix is not None and hc.D_RANGE[0] <= D_pix <= hc.D_RANGE[1]) else pred
        rows.append(dict(gate=gate, m=m, pred=pred, margin=margin, k_src=hc.active_k(intr)[1],
                         serial=sn, fx=fx, span=span, D_pix=D_pix, margin_pix=margin_pix, pred_use=pred_use))
    ok = [x for x in rows if x['m']]
    out = dict(n_frames=len(rows), n_judged=len(ok), serial=rows[0]['serial'] if rows else None,
               k_src=rows[0]['k_src'] if rows else None, class_name=r['cls'], cad_d=CAD_D.get(r['cls']))
    votes = collections.Counter(x['pred_use'] for x in rows if x['pred_use'])
    if votes:
        out['pred_major'], out['n_pred_major'] = votes.most_common(1)[0]
    if ok:
        med = lambda key: float(np.median([x['m'][key] for x in ok]))
        out.update(k_applied=ok[0]['m']['k'], d_raw_med=med('D_raw'), d_med=med('D_mm'), z_med=med('z'), tilt_med=med('tilt'),
                   margin_min=float(min(x['margin'] for x in ok)),
                   n_wrong=sum(x['pred'] != r['cls'] for x in ok), n_unknown=sum(x['pred'] == hc.UNKNOWN for x in ok))
        if out['cad_d']:
            out['dev_mm'] = out['d_med'] - out['cad_d']; out['k_session'] = out['cad_d'] / out['d_raw_med']
    sp = [x for x in rows if x['span']]
    if sp:
        out['span_med'] = float(np.median([x['span'] for x in sp])); fx = sp[0]['fx']; sn = sp[0]['serial']
        if out['cad_d'] and fx:
            out['s_session'] = out['cad_d'] * fx / out['span_med']
        if sn in hc.S_PIXEL and fx:
            out['s_ref'] = hc.S_PIXEL[sn]; out['d_pix_med'] = out['span_med'] * out['s_ref'] / fx
            out['margin_pix_min'] = float(min(x['margin_pix'] for x in sp if x['margin_pix'] is not None))
            if out['cad_d']:
                out['dev_pix_mm'] = out['d_pix_med'] - out['cad_d']
    return out


def print_table(con):
    rows = con.execute(
        """SELECT m.*, s.session_dir, s.started_at FROM session_hole_metrics m JOIN capture_sessions s ON s.id=m.session_id
           ORDER BY s.started_at""").fetchall()
    print(f"\n{'세션':34s} {'클래스':16s} {'n':>3s} {'판정':>3s} {'D med':>7s} {'CAD':>5s} {'dev':>6s} {'K_sess':>7s} {'z':>6s} {'tilt':>4s} {'마진':>5s} 경보")
    for m in rows:
        flag = []
        if m['dev_mm'] is not None and abs(m['dev_mm']) > DEV_WARN: flag.append(f'dev{m["dev_mm"]:+.0f}')
        if m['k_session'] and m['k_applied'] and abs(m['k_session'] / m['k_applied'] - 1) > K_WARN: flag.append(f'K{100*(m["k_session"]/m["k_applied"]-1):+.1f}%')
        if m['dev_pix_mm'] is not None and abs(m['dev_pix_mm']) > DEV_WARN: flag.append(f'devP{m["dev_pix_mm"]:+.0f}')
        if m['s_session'] and m['s_ref'] and abs(m['s_session'] / m['s_ref'] - 1) > S_WARN: flag.append(f'S{100*(m["s_session"]/m["s_ref"]-1):+.2f}%')
        if m['n_wrong']: flag.append(f'오판{m["n_wrong"]}')
        if m['n_judged'] < 3: flag.append('판정<3')
        f = lambda v, w: (f'{v:{w}.1f}' if v is not None else ' ' * (w - 1) + '-')
        print(f"{m['session_dir']:34s} {m['class_name']:16s} {m['n_frames']:3d} {m['n_judged']:3d} {f(m['d_med'],7)} {m['cad_d'] or 0:5.0f} "
              f"{f(m['dev_mm'],6)} {f(m['k_session'],7) if m['k_session'] is None else m['k_session']:>7.4f} {f(m['z_med'],6)} {f(m['tilt_med'],4)} {f(m['margin_min'],5)} {' '.join(flag)}"
              if m['k_session'] is not None else
              f"{m['session_dir']:34s} {m['class_name']:16s} {m['n_frames']:3d} {m['n_judged']:3d} {'-':>7s} {m['cad_d'] or 0:5.0f} {'-':>6s} {'-':>7s} {'-':>6s} {'-':>4s} {'-':>5s} {' '.join(flag)}")
    ks = [m['k_session'] for m in rows if m['k_session']]
    if ks: print(f"\n세션 {len(rows)}  K_session med {np.median(ks):.4f} std {np.std(ks):.4f}  (기준 K_applied {rows[-1]['k_applied']})")
    ss = [m['s_session'] for m in rows if m['s_session']]; dp = [abs(m['dev_pix_mm']) for m in rows if m['dev_pix_mm'] is not None]
    if ss: print(f"S_session med {np.median(ss):.1f} std {100*np.std(ss)/np.median(ss):.2f}% (기준 S_PIXEL {rows[-1]['s_ref']})   |dev_pix| med {np.median(dp):.1f} p95 {np.percentile(dp,95):.1f} max {max(dp):.1f}")


if __name__ == '__main__':
    con = sqlite3.connect(args.db); con.row_factory = sqlite3.Row; ensure(con)
    if args.print:
        print_table(con); sys.exit()
    sys.path.insert(0, os.path.join(DOOR, 'db'))
    from db_log import DBLog
    mid = DBLog(args.db).find_model(weights_path=os.path.relpath(args.model, DOOR), name='hole_landmarks_resnet18')
    if mid is None: raise SystemExit('models 테이블에 홀 판별기 모델이 없습니다')
    done = {r[0] for r in con.execute("SELECT session_id FROM session_hole_metrics WHERE model_id=?", (mid,))}
    todo = [r for r in session_rows(con) if args.recompute or r['sid'] not in done]
    if args.limit: todo = todo[:args.limit]
    print(f"모델 id {mid}, 계산 대상 {len(todo)}세션 (기록됨 {len(done)})")
    if todo:
        net, dev = hc.load_model(args.model); t0 = time.time()
        for i, r in enumerate(todo):
            o = compute(net, dev, r)
            cols = ['session_id', 'model_id'] + list(o)
            con.execute(f"INSERT OR REPLACE INTO session_hole_metrics ({','.join(cols)}, evaluated_at) VALUES ({','.join('?' * len(cols))}, datetime('now','localtime'))",
                        [r['sid'], mid] + list(o.values()))
            con.commit()
            print(f"  [{i + 1}/{len(todo)}] {r['session_dir']} {r['cls']} n={o['n_frames']} judged={o['n_judged']} "
                  f"dev={o.get('dev_mm', float('nan')):+.1f} K={o.get('k_session', float('nan')):.4f} "
                  f"devP={o.get('dev_pix_mm', float('nan')):+.1f} S={o.get('s_session', float('nan')):.1f}  {time.time() - t0:.0f}s")
    print_table(con)
