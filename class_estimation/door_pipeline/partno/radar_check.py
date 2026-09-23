"""RADAR 옵션 규칙 검사 — 홀 랜드마크 6점으로 만든 도어 좌표계에 CAD 브래킷 홀 2개를 투영해 '어두운 홀' 유무를 판정 (학습 없음).

  python partno/radar_check.py                       # DB 유효 RR/RH 세션 중 미검사분 → session_radar_check + capture_sessions.option_*(auto) + 몽타주
  python partno/radar_check.py --recompute           # 전부 다시 (임계값·브래킷 좌표 변경 후)
  python partno/radar_check.py --session 20260914/E25_door_RH/s_124221 [--session ...]   # 특정 세션
  python partno/radar_check.py --dry-run --limit 20  # DB 기록 없이 점수·몽타주만
  python partno/radar_check.py --print               # 현재 표만
  python partno/radar_check.py --accept-auto         # 육안 확인 후 자동 판정을 사용자 확정으로 일괄 승인 (미정 제외)
  python partno/radar_check.py --set <session_dir> 1|0|clear   # 세션 하나 확정(웹 접근 불가 시 CLI)
  python partno/radar_check.py --recompute --cls E30_door_LH_RR   # 특정 클래스만 재검사 (브래킷 좌표 갱신 후)

프레임 판정:
  ① hole_classifier.detect 로 볼트 4 + 코너 2 검출, geometry_gate 통과 프레임만
  ② 6점 도어 XY(pos_pipeline/cad_holes.json holes_door) ↔ 픽셀 최소자승 아핀 (볼트 4개 이름은 3점 초기 아핀으로 매칭)
  ③ radar_bracket.json 의 홀 2개(도어 XY, 6홀 평면보다 z_plate≈70mm 카메라 쪽) 투영 — depth 평면 깊이로 시차 보정
  ④ 밝기 점수 = (주변 고리 밝기 − 홀 안 밝기)/고리 밝기 (회색조). 두 홀에 같은 오프셋(±12px 탐색)을 주어 min(홀1,홀2) 최대화
     판 높이 h = 홀 주변 고리의 depth 가 6홀 평면보다 카메라 쪽인 거리(mm): 브래킷 판 ≈70, 브래킷 없는 보강재 ≈29 (조명 무관 단서)
  ⑤ 점수 ≥ 0.30 → 레이더. h 가 있으면: 점수 ≥ 0.08 이고 h ≥ 37(판 높이) → 레이더, 점수 ≤ 0.12 이고 h < 49 → 없음.
     점수 ≤ 0.04(홀 흔적 없음) → 없음. h 가 없으면 점수 ≤ 0.12 → 없음. 나머지 미정. 판이 프레임 밖이거나 아핀 잔차 > 12px 면 미판정
세션 판정: 판정 프레임 ≥ MIN_JUDGED(3) 이고 다수 표 비율 ≥ MIN_AGREE(80%) 일 때만 auto_flag(1/0), 아니면 NULL(미정).
확정값(capture_sessions.option_radar)은 option_source 가 NULL/auto 일 때만 갱신 — 사용자 확정('user')은 덮어쓰지 않는다.
FRT 클래스는 옵션이 없으므로 검사하지 않는다.
"""
import argparse, collections, itertools, json, math, os, sqlite3, sys, time
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOOR = os.path.dirname(HERE)
ROOT = os.path.normpath(os.path.join(DOOR, '..', '..'))
sys.path.insert(0, DOOR); sys.path.insert(0, os.path.join(DOOR, 'db'))
import hole_classifier as hc
from camera_utils import intrinsics_for_image
import build_dataset as bd

CAD_HOLES = os.path.join(ROOT, 'pos_estimation', 'pos_pipeline', 'cad_holes.json')
BRACKET_JSON = os.path.join(HERE, 'radar_bracket.json')
CROP_DIR = os.path.join(HERE, 'artifacts', 'radar_crops')
BOLT_NAMES = ['bolt_tl', 'bolt_tr', 'bolt_bl', 'bolt_br']
LM6 = BOLT_NAMES + ['corner_hinge', 'corner_latch']
THR_ON, THR_OFF = 0.30, 0.12        # 밝기 점수 임계 (depth 높이 단서가 없을 때 단독 판정)
THR_DARK_MIN = 0.08                 # 판 높이가 브래킷 쪽일 때 요구하는 최소 어두움(조명이 밝아 홀이 회색으로 보이는 세션 대응)
THR_FLAT = 0.04                     # 이 이하는 홀 흔적이 전혀 없음 → 높이와 무관하게 없음
MIN_JUDGED, MIN_AGREE = 3, 0.8      # 짧은 세션(3장)도 만장일치면 판정
H_SPLIT_FRAC = 0.2                  # 판 높이 분리점 = z_strip + 0.2·(z_plate − z_strip) ≈ 37mm. depth 가 판 가장자리를 뭉개 실측 판 높이는 50~56mm(CAD 70), 보강재 13~19mm(CAD 29)
SEARCH_PX, SEARCH_STEP = 12, 4
MAX_RESID_PX = 12.0
Z_NOMINAL = 1450.0                  # depth 없을 때 도어 평면 깊이(mm) — 시차 보정용


def load_geometry():
    cadh = json.load(open(CAD_HOLES))['classes']
    br = json.load(open(BRACKET_JSON))['classes']
    return {cls: dict(lm={k: np.array(v[:2], float) for k, v in cadh[cls]['holes_door'].items()}, **br[cls]) for cls in br}


def door_affine(det, lm):
    """검출 6점 ↔ 도어 XY 아핀 A(2×3). 반환 (A, resid_px, named) 또는 None."""
    if len(det['bolt']) < 4 or not det['corner_hinge'] or not det['corner_latch']:
        return None
    hinge = np.array(det['corner_hinge'][0][:2], float); latch = np.array(det['corner_latch'][0][:2], float)
    bolts = [np.array(b[:2], float) for b in det['bolt'][:4]]
    src3 = np.float32([lm['corner_hinge'], lm['corner_latch'], np.mean([lm[k] for k in BOLT_NAMES], 0)])
    dst3 = np.float32([hinge, latch, np.mean(bolts, 0)])
    A0 = cv2.getAffineTransform(src3, dst3)
    proj = {k: A0 @ np.array([*lm[k], 1.0]) for k in BOLT_NAMES}
    perm = min(itertools.permutations(range(4)), key=lambda p: sum(np.linalg.norm(proj[k] - bolts[p[i]]) for i, k in enumerate(BOLT_NAMES)))
    named = {k: bolts[perm[i]] for i, k in enumerate(BOLT_NAMES)}
    named['corner_hinge'], named['corner_latch'] = hinge, latch
    X = np.hstack([np.float64([lm[k] for k in LM6]), np.ones((6, 1))]); Y = np.float64([named[k] for k in LM6])
    sol = np.linalg.lstsq(X, Y, rcond=None)[0]
    resid = float(np.linalg.norm(X @ sol - Y, axis=1).mean())
    return sol.T, resid, named


def project(A, xy, z_up, K, Z0):
    """도어 XY(6홀 평면) → 픽셀, 평면보다 z_up(mm) 카메라 쪽 점은 주점 기준 Z0/(Z0−z_up) 배 시차 보정."""
    p = A @ np.array([xy[0], xy[1], 1.0])
    if z_up and Z0:
        pp = np.array([K['cx'], K['cy']]); p = pp + (p - pp) * Z0 / max(Z0 - z_up, 1.0)
    return p


def hole_score(gray, u, v, r):
    """홀 안(0.65r) 대비 고리(1.35~1.9r) 밝기 대비. 범위 밖이면 None."""
    h, w = gray.shape; R = int(r * 1.9) + 2
    if u - R < 0 or v - R < 0 or u + R >= w or v + R >= h:
        return None
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1]; d = np.hypot(xx, yy)
    patch = gray[int(v) - R:int(v) + R + 1, int(u) - R:int(u) + R + 1].astype(np.float32)
    inner = patch[d <= 0.65 * r]; ring = patch[(d >= 1.35 * r) & (d <= 1.9 * r)]
    if inner.size < 5 or ring.size < 5:
        return None
    mr = float(ring.mean()); return (mr - float(inner.mean())) / max(mr, 1.0)


def check_frame(net, dev, rgb, depth, cls, intr, geom, det=None):
    """단일 프레임 검사. 반환 dict(status, score, s1, s2, dxy, holes_px, r_px, plate_px, resid, z0, depth_open)."""
    g = geom.get(cls)
    if g is None:
        return dict(status='no_geom')
    det = det or hc.detect(net, dev, rgb)
    hinge = det['corner_hinge'][0] if det['corner_hinge'] else None; latch = det['corner_latch'][0] if det['corner_latch'] else None
    gate = hc.geometry_gate(hc.bolt_frame(det['bolt']), hinge, latch, rgb.shape)
    if gate != 'ok':
        return dict(status='gate:' + gate)
    fit = door_affine(det, g['lm'])
    if fit is None or fit[1] > MAX_RESID_PX:
        return dict(status='affine', resid=fit[1] if fit else None)
    A, resid, _ = fit
    h, w = rgb.shape[:2]
    K = intr or dict(cx=w / 2.0, cy=h / 2.0, fx=hc.FX_APPROX, fy=hc.FX_APPROX)
    z0, to3d = None, None
    if depth is not None:
        pf = hc.plane_from_depth(depth, det['bolt'] + [hinge, latch], intr)
        if pf:
            to3d = pf[0]; z0 = float(to3d((A @ np.array([*np.mean(g['holes'], 0), 1.0])))[2])
    Z0 = z0 or Z_NOMINAL
    scale = math.sqrt(abs(np.linalg.det(A[:, :2]))) * Z0 / max(Z0 - g['z_plate'], 1.0)   # px/mm (판 높이 시차 배율 포함)
    holes = [project(A, xy, g['z_plate'], K, Z0) for xy in g['holes']]
    r_px = g['r_mm'] * scale
    x0, y0, x1, y1 = g['plate']
    plate = np.array([project(A, xy, g['z_plate'], K, Z0) for xy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])
    if plate[:, 0].min() < 0 or plate[:, 1].min() < 0 or plate[:, 0].max() >= w or plate[:, 1].max() >= h:
        return dict(status='plate_outside', holes_px=holes, r_px=r_px, plate_px=plate, resid=resid)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    best = None
    for dx in range(-SEARCH_PX, SEARCH_PX + 1, SEARCH_STEP):
        for dy in range(-SEARCH_PX, SEARCH_PX + 1, SEARCH_STEP):
            s = [hole_score(gray, p[0] + dx, p[1] + dy, r_px) for p in holes]
            if None in s:
                continue
            if best is None or min(s) > best[0]:
                best = (min(s), s, (dx, dy))
    if best is None:
        return dict(status='hole_outside', holes_px=holes, r_px=r_px, plate_px=plate, resid=resid)
    score, (s1, s2), dxy = best
    # 판 높이 단서(조명 무관): 홀 주변 고리(1.3~2.0r, 판 위)의 depth 가 6홀 평면보다 얼마나 카메라 쪽인가(mm).
    # 브래킷 판 ≈ z_plate(70mm), 브래킷 없는 보강재 ≈ z_strip(29mm). ZED depth 는 홀 안을 메워 버리므로 홀 안 대신 고리를 잰다.
    h_mm = None
    if depth is not None and to3d is not None:
        vals = []
        for p in holes:
            u, v = int(p[0] + dxy[0]), int(p[1] + dxy[1]); R = int(r_px * 2.0) + 1
            y0, y1, x0, x1 = max(0, v - R), v + R + 1, max(0, u - R), u + R + 1
            patch = depth[y0:y1, x0:x1].astype(np.float32)
            yy, xx = np.mgrid[y0:y1, x0:x1]; d = np.hypot(xx - u, yy - v)
            m = (d >= 1.3 * r_px) & (d <= 2.0 * r_px) & (patch > 0)
            if m.sum() >= 20:
                zp = float(to3d((u, v))[2]); vals.append(zp - float(np.median(patch[m])))
        if vals:
            h_mm = float(np.mean(vals))
    status = decide(score, h_mm, g)
    return dict(status=status, score=float(score), s1=float(s1), s2=float(s2), dxy=dxy, holes_px=holes, r_px=float(r_px),
                plate_px=plate, resid=resid, z0=z0, h_mm=h_mm)


def decide(score, h_mm, g):
    """프레임 판정. 밝기 점수(어두운 홀)와 판 높이(depth)를 함께 본다.
    높이가 있으면: 판 높이 쪽(≥ H_SPLIT)이고 홀이 조금이라도 어두우면 radar, 보강재 높이 쪽(< H_SPLIT)이고 홀 어둡지 않으면 none.
    높이가 없으면(depth 없음/평면 실패): 밝기만으로 THR_ON/THR_OFF."""
    if score >= THR_ON:
        return 'radar'                                   # 홀 2개가 뚜렷이 어두움
    if h_mm is not None:
        lo = g['z_strip'] + H_SPLIT_FRAC * (g['z_plate'] - g['z_strip'])     # ≈ 35~37mm: 이 위면 판 높이로 본다
        hi = g['z_strip'] + 0.5 * (g['z_plate'] - g['z_strip'])              # ≈ 48~50mm: 이 아래면 판이 아니다
        if score >= THR_DARK_MIN and h_mm >= lo: return 'radar'             # 약하게 어두운 홀 + 판 높이
        if score <= THR_OFF and h_mm < hi: return 'none'                     # 홀 흔적 없음 + 판 높이 아님
    elif score <= THR_OFF:
        return 'none'
    if score <= THR_FLAT:
        return 'none'                                    # 홀 흔적이 전혀 없으면(균일한 밝기) 높이 측정이 어긋나도 없음
    return 'unsure'


def aggregate(frames):
    judged = [f for f in frames if f['status'] in ('radar', 'none', 'unsure')]
    n_radar = sum(f['status'] == 'radar' for f in judged); n_none = sum(f['status'] == 'none' for f in judged)
    flag, agree = None, None
    if len(judged) >= MIN_JUDGED:
        top = max(n_radar, n_none); agree = top / len(judged)
        if agree >= MIN_AGREE:
            flag = 1 if n_radar > n_none else 0
    scores = [f['score'] for f in judged]; hs = [f['h_mm'] for f in judged if f.get('h_mm') is not None]
    return dict(n_frames=len(frames), n_judged=len(judged), n_radar=n_radar, n_none=n_none,
                score_med=float(np.median(scores)) if scores else None, h_med=float(np.median(hs)) if hs else None, auto_flag=flag, agree=agree,
                reasons=dict(collections.Counter(f['status'] for f in frames if f['status'] not in ('radar', 'none', 'unsure'))))


def montage(frames_rgb, title, path, n=6, tile=(320, 130)):
    """판정 프레임 n장의 판 영역 크롭(홀 원 표시) 타일 + 제목. frames_rgb: [(rgb, frame_result)]"""
    picks = [x for x in frames_rgb if x[1].get('plate_px') is not None]
    picks = picks[::max(1, len(picks) // n)][:n]
    tiles = []
    for rgb, f in picks:
        P = f['plate_px']; cx, cy = P.mean(0); wdt = max(float(np.ptp(P[:, 0])), 60) * 1.4; hgt = max(float(np.ptp(P[:, 1])), 40) * 1.6
        x0, y0 = int(max(0, cx - wdt / 2)), int(max(0, cy - hgt / 2)); x1, y1 = int(min(rgb.shape[1], cx + wdt / 2)), int(min(rgb.shape[0], cy + hgt / 2))
        crop = rgb[y0:y1, x0:x1].copy()
        if crop.size == 0:
            continue
        dx, dy = f.get('dxy', (0, 0))
        col = {'radar': (0, 200, 0), 'none': (0, 0, 230), 'unsure': (0, 200, 230)}.get(f['status'], (200, 200, 200))
        for p in f['holes_px']:
            cv2.circle(crop, (int(p[0] + dx - x0), int(p[1] + dy - y0)), int(f['r_px']), col, 2)
        crop = cv2.resize(crop, tile)
        cv2.putText(crop, f"{f['status']} {f.get('score', 0):.2f} h{f['h_mm']:.0f}" if f.get('h_mm') is not None else f"{f['status']} {f.get('score', 0):.2f}", (4, 16), 0, 0.5, col, 1)
        tiles.append(crop)
    if not tiles:
        tiles = [np.full((tile[1], tile[0], 3), 40, np.uint8)]
    while len(tiles) % 3:
        tiles.append(np.full((tile[1], tile[0], 3), 40, np.uint8))
    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    img = np.vstack(rows); hdr = np.full((26, img.shape[1], 3), 30, np.uint8)
    cv2.putText(hdr, title, (6, 18), 0, 0.55, (255, 255, 255), 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.vstack([hdr, img]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def session_rows(con, only=None):
    ds = con.execute("SELECT id FROM datasets WHERE name=?", (bd.DATASET_NAME,)).fetchone()[0]
    q = """SELECT s.id sid, s.session_dir, s.option_source, c.name cls,
                  GROUP_CONCAT(i.rgb_path) rgbs, GROUP_CONCAT(i.depth_path) depths
           FROM images i JOIN classes c ON c.id=i.class_id JOIN capture_sessions s ON s.id=i.session_id
           WHERE c.dataset_id=? AND i.synced_local AND i.is_valid AND c.name!='Unknown' """
    args = [ds]
    if only:
        q += " AND s.session_dir IN (%s)" % ','.join('?' * len(only)); args += list(only)
    return con.execute(q + " GROUP BY s.id ORDER BY s.session_dir", args).fetchall()


def run_session(net, dev, geom, r, crop_path):
    frames, keep = [], []
    for rp, dp in zip(r['rgbs'].split(','), r['depths'].split(',')):
        rgb = cv2.imread(str(bd.MIRROR / rp)); depth = cv2.imread(str(bd.MIRROR / dp), cv2.IMREAD_UNCHANGED) if dp else None
        if rgb is None:
            continue
        intr = intrinsics_for_image(str(bd.MIRROR / rp), rgb.shape)
        f = check_frame(net, dev, rgb, depth, r['cls'], intr, geom)
        frames.append(f); keep.append((rgb, f))
    agg = aggregate(frames)
    title = f"{r['session_dir']} {r['cls']}  radar {agg['n_radar']} / none {agg['n_none']} / judged {agg['n_judged']} of {agg['n_frames']}  -> {agg['auto_flag']}"
    montage(keep, title, crop_path)
    return agg


def print_table(con):
    rows = con.execute("""SELECT r.*, s.option_radar, s.option_source FROM session_radar_check r JOIN capture_sessions s ON s.id=r.session_id
                          ORDER BY r.class_name, s.session_dir""").fetchall()
    print(f"\n{'세션':36s} {'클래스':16s} {'n':>3s} {'판정':>3s} {'레이더':>4s} {'없음':>4s} {'점수':>5s} 자동 확정")
    for m in rows:
        print(f"{m['session_dir'] if 'session_dir' in m.keys() else m['session_id']!s:36s} {m['class_name']:16s} {m['n_frames']:3d} {m['n_judged']:3d} {m['n_radar']:4d} {m['n_none']:4d} "
              f"{(m['score_med'] if m['score_med'] is not None else float('nan')):5.2f} {str(m['auto_flag']):>4s} {str(m['option_radar']) + '/' + str(m['option_source'])}")
    c = collections.Counter((m['class_name'], m['auto_flag']) for m in rows)
    print("\n클래스별 자동 판정:", {f"{k[0]}:{k[1]}": v for k, v in sorted(c.items(), key=lambda t: (t[0][0], str(t[0][1])))})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=os.path.join(DOOR, 'db', 'door_pipeline.db'))
    ap.add_argument('--session', action='append', default=[])
    ap.add_argument('--recompute', action='store_true'); ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--print', action='store_true')
    ap.add_argument('--thr-on', type=float, default=THR_ON); ap.add_argument('--thr-off', type=float, default=THR_OFF)
    ap.add_argument('--cls', action='append', default=[], help='이 클래스 세션만 (예: 브래킷 좌표 갱신 후 --recompute --cls E30_door_LH_RR)')
    ap.add_argument('--accept-auto', action='store_true',
                    help='자동 판정(auto 1/0)을 사용자 확정(user)으로 일괄 승인 — 웹 /options 육안 확인을 마쳤을 때. 미정(NULL)은 그대로')
    ap.add_argument('--set', nargs=2, metavar=('SESSION_DIR', 'VALUE'), action='append', default=[],
                    help="세션 하나를 사용자 확정으로 설정 (웹 /options 버튼과 동일, 외부망 등 웹 접근 불가 시): --set 20260922/E38_door_LH_RR/s_080559 1  (VALUE 1|0|clear)")
    a = ap.parse_args()
    THR_ON, THR_OFF = a.thr_on, a.thr_off
    con = sqlite3.connect(a.db); con.row_factory = sqlite3.Row
    from migrate_options import ensure; ensure(con)
    if a.print:
        print_table(con); sys.exit()
    if a.set:
        now = time.strftime('%Y-%m-%d %H:%M')
        for sd, val in a.set:
            r = con.execute("SELECT id FROM capture_sessions WHERE session_dir=?", (sd,)).fetchone()
            if not r: print(f"세션 없음: {sd}"); continue
            auto = con.execute("SELECT auto_flag, score_med FROM session_radar_check WHERE session_id=?", (r['id'],)).fetchone()
            if val == 'clear':
                con.execute("UPDATE capture_sessions SET option_radar=NULL, option_source=NULL, option_note=?, option_at=datetime('now','localtime'), notes=COALESCE(notes || '\n', '') || ? WHERE id=?",
                            (f"user cleared {now} (cli)", f"[{now}] 옵션 확정 취소 (cli)", r['id'])); print(f"{sd}: 확정 취소")
            else:
                v = int(val)
                con.execute("UPDATE capture_sessions SET option_radar=?, option_source='user', option_note=?, option_at=datetime('now','localtime'), notes=COALESCE(notes || '\n', '') || ? WHERE id=?",
                            (v, f"user confirm cli (auto {auto['auto_flag'] if auto else None}, score {auto['score_med'] if auto and auto['score_med'] is not None else float('nan'):.2f})",
                             f"[{now}] 옵션 확정(cli): 레이더 {'O' if v else 'X'} (자동 판정 {auto['auto_flag'] if auto else '미검사'})", r['id']))
                print(f"{sd}: 레이더 {'O' if v else 'X'} 확정")
        con.commit(); sys.exit()
    if a.accept_auto:
        now = time.strftime('%Y-%m-%d %H:%M')
        rows = con.execute("SELECT id, session_dir, option_radar FROM capture_sessions WHERE option_source='auto' AND option_radar IS NOT NULL").fetchall()
        for r in rows:
            con.execute("UPDATE capture_sessions SET option_source='user', option_note=COALESCE(option_note,'') || ?, option_at=datetime('now','localtime'), "
                        "notes=COALESCE(notes || '\n', '') || ? WHERE id=?",
                        (f" | bulk accept {now}", f"[{now}] 옵션 확정(자동 판정 일괄 승인): 레이더 {'O' if r['option_radar'] else 'X'}", r['id']))
        con.commit(); print(f"자동 판정 {len(rows)}세션을 사용자 확정(user)으로 승인했습니다. 이후 partno/evaluate_partno.py --db-log 로 공식 수치 기록."); sys.exit()
    geom = load_geometry()
    done = {r[0] for r in con.execute("SELECT session_id FROM session_radar_check")}
    rows = [r for r in session_rows(con, a.session or None) if r['cls'] in geom and (a.recompute or a.session or r['sid'] not in done)
            and (not a.cls or r['cls'] in a.cls)]
    if a.limit: rows = rows[:a.limit]
    print(f"검사 대상 {len(rows)}세션 (기록됨 {len(done)}, 브래킷 등록 클래스 {sorted(geom)})  임계 on {THR_ON} off {THR_OFF}")
    net, dev = hc.load_model(); t0 = time.time()
    for i, r in enumerate(rows):
        date, _, sess = r['session_dir'].split('/')
        crop = os.path.join(CROP_DIR, f"{date}_{sess}.jpg")
        agg = run_session(net, dev, geom, r, crop)
        print(f"  [{i + 1}/{len(rows)}] {r['session_dir']:36s} {r['cls']:16s} radar {agg['n_radar']:2d} none {agg['n_none']:2d} judged {agg['n_judged']:2d}/{agg['n_frames']:2d} "
              f"score {agg['score_med'] if agg['score_med'] is not None else float('nan'):5.2f} h {agg['h_med'] if agg['h_med'] is not None else float('nan'):4.0f} → {agg['auto_flag']}  {agg['reasons'] or ''}  {time.time() - t0:.0f}s")
        if a.dry_run:
            continue
        con.execute("""INSERT OR REPLACE INTO session_radar_check (session_id, class_name, n_frames, n_judged, n_radar, n_none, score_med, auto_flag, agree, crop_path, params, checked_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?, datetime('now','localtime'))""",
                    (r['sid'], r['cls'], agg['n_frames'], agg['n_judged'], agg['n_radar'], agg['n_none'], agg['score_med'], agg['auto_flag'], agg['agree'],
                     os.path.relpath(crop, DOOR), json.dumps(dict(thr_on=THR_ON, thr_off=THR_OFF, thr_dark_min=THR_DARK_MIN, thr_flat=THR_FLAT, min_judged=MIN_JUDGED, min_agree=MIN_AGREE,
                                                                   h_med=agg['h_med'], reasons=agg['reasons']))))
        if r['option_source'] != 'user':
            con.execute("UPDATE capture_sessions SET option_radar=?, option_source=?, option_note=?, option_at=datetime('now','localtime') WHERE id=?",
                        (agg['auto_flag'], 'auto' if agg['auto_flag'] is not None else None,
                         f"auto radar {agg['n_radar']}/none {agg['n_none']}/judged {agg['n_judged']} score {agg['score_med'] if agg['score_med'] is not None else float('nan'):.2f}", r['sid']))
        con.commit()
    if not a.dry_run:
        print_table(con)
