"""레이저 트래커 GT 대비 절대 정확도 평가 (GT_TRACKER_PROTOCOL.md 분석 단계).

입력(--input): tracker_template.json 형식 — 포즈별 트래커 6홀 좌표(mm)+카메라 세션 경로.
  방법 1: 포즈 쌍 변화량 비교 (등록 불필요 — 스케일·회전 오차 직접 검증)
  방법 2: 전 포즈로 X=T_cam←trk 최적 추정 → 포즈별 잔차 (포즈 의존 편향)
  방법 3: reference_targets 있으면 독립 등록 X 로 완전 절대 오차

  python 05_evaluate_tracker_gt.py --input tracker_20260905.json
  python 05_evaluate_tracker_gt.py --selftest    # 현장 전 분석 파이프라인 검증(카메라 불필요)

트래커 홀 좌표는 6홀↔CAD 강체 정합 잔차(정상 ≤1mm)로 품질을 게이트한다.
합격: 방법1·2 병진 med ≤5mm, 회전 med ≤1° (프로토콜 8절).
"""
import argparse
import glob
import json
import os
import time

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
import pose_solver as ps
import pose_utils as pu

from scipy.spatial.transform import Rotation

ap = argparse.ArgumentParser()
ap.add_argument('--input', help='측정 기록 json (tracker_template.json 형식)')
ap.add_argument('--selftest', action='store_true')
ap.add_argument('--per_session', type=int, default=20)
args = ap.parse_args()


def rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def T_of(R, t):
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def pose_from_tracker(holes_mm, cad_pts, pin_offset=0.0):
    """트래커 6홀 좌표 → T_trk→door + 정합 잔차. pin_offset: 핀 네스트 오프셋(표면 법선 위 mm)."""
    names = [k for k in ps.LM6 if k in holes_mm]
    assert len(names) >= 4, f'홀 {len(names)}개 — 최소 4개 필요'
    A = np.stack([np.asarray(cad_pts[k], float) for k in names])
    B = np.stack([np.asarray(holes_mm[k], float) for k in names])
    R, t, rms, res = pu.umeyama_rigid(A, B)
    if pin_offset:                                  # 도어 +Z(법선) 방향 오프셋 제거 후 재정합
        B = B - pin_offset * R[:, 2]
        R, t, rms, res = pu.umeyama_rigid(A, B)
    return T_of(R, t), rms, dict(zip(names, np.round(res, 3)))


def pose_from_session(sess_dir, cls, cad, net, dev):
    """카메라 세션 → 집계 T_cam→door."""
    import cv2
    files = sorted(glob.glob(os.path.join(sess_dir, 'rgb_*.png')))[:args.per_session]
    assert files, f'세션 프레임 없음: {sess_dir}'
    frames = []
    for f in files:
        rgb = cv2.imread(f)
        depth = cv2.imread(f.replace('rgb_', 'depth_'), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            continue
        frames.append(ps.solve(net, dev, rgb, depth, cls, cad))
    agg = ps.aggregate_poses(frames)
    assert agg.get('ok'), f'세션 포즈 실패: {sess_dir}'
    return agg['T'], agg


def method1(ids, Tc, Tt):
    """포즈 쌍 변화량 비교 (등록 불필요). 반환: 쌍별 (병진 mm, 회전 °)."""
    out = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            A = np.linalg.inv(Tc[i]) @ Tc[j]
            B = np.linalg.inv(Tt[i]) @ Tt[j]
            E = np.linalg.inv(A) @ B
            out.append((f'{ids[i]}-{ids[j]}', float(np.linalg.norm(E[:3, 3])), rot_angle(E[:3, :3]),
                        float(np.linalg.norm(B[:3, 3]))))
    return out


def fit_X(Tc, Tt):
    """X = T_cam←trk 최적 추정 (X_i = Tc_i·Tt_i⁻¹ 의 회전 쿼터니언 평균 + 병진 평균)."""
    Xs = [Tc[i] @ np.linalg.inv(Tt[i]) for i in range(len(Tc))]
    R = Rotation.from_matrix(np.stack([X[:3, :3] for X in Xs])).mean().as_matrix()
    t = np.mean([X[:3, 3] for X in Xs], 0)
    return T_of(R, t)


def residuals_vs_X(ids, Tc, Tt, X):
    """포즈별: X·T_trk 예측 vs 카메라 측정 — (|Δt| mm, 도어 프레임 Δt, 회전 °)."""
    out = []
    for i, pid in enumerate(ids):
        P = X @ Tt[i]
        dt = Tc[i][:3, 3] - P[:3, 3]
        dt_door = P[:3, :3].T @ dt
        dR = P[:3, :3].T @ Tc[i][:3, :3]
        out.append((pid, float(np.linalg.norm(dt)), [round(float(v), 2) for v in dt_door],
                    rot_angle(dR)))
    return out


def report(m1, m2, probe_rms, tag, extra=''):
    t1 = np.array([r[1] for r in m1]); r1 = np.array([r[2] for r in m1])
    t2 = np.array([r[1] for r in m2]); r2 = np.array([r[3] for r in m2])
    ok = (np.median(t1) <= 5 and np.median(r1) <= 1 and np.median(t2) <= 5 and np.median(r2) <= 1)
    print(f'\n[{tag}] 트래커 프로브 정합 잔차: ' + ' '.join(f'{v:.2f}' for v in probe_rms) + ' mm')
    print(f'방법1 변화량({len(m1)}쌍): 병진 med {np.median(t1):.2f} p95 {np.percentile(t1, 95):.2f} mm'
          f' | 회전 med {np.median(r1):.3f} p95 {np.percentile(r1, 95):.3f} °')
    print(f'방법2 잔차({len(m2)}포즈): 병진 med {np.median(t2):.2f} max {t2.max():.2f} mm'
          f' | 회전 med {np.median(r2):.3f} max {r2.max():.3f} °{extra}')
    print(f'판정(±5mm/±1°): {"합격" if ok else "불합격"}')
    return dict(ok=bool(ok), probe_rms=[float(v) for v in probe_rms],
                m1=dict(n=len(m1), t_med=float(np.median(t1)), t_p95=float(np.percentile(t1, 95)),
                        r_med=float(np.median(r1)), r_p95=float(np.percentile(r1, 95)),
                        pairs=[dict(pair=p, dt=round(t, 2), dr=round(r, 3), move=round(mv, 1))
                               for p, t, r, mv in m1]),
                m2=dict(n=len(m2), t_med=float(np.median(t2)), t_max=float(t2.max()),
                        r_med=float(np.median(r2)), r_max=float(r2.max()),
                        poses=[dict(pose=p, dt=round(t, 2), dt_door=d, dr=round(r, 3))
                               for p, t, d, r in m2]))


def selftest():
    """가짜 트래커 데이터 + 오차 주입 카메라 포즈로 분석 경로 검증 (주입: 2mm/0.3°)."""
    rng = np.random.default_rng(7)
    cad = ps.load_cad(); cls = 'E25_door_RH'
    pts = cad[cls]['pts']
    rvec = rng.normal(0, 0.5, 3)
    X_true = T_of(Rotation.from_rotvec(rvec).as_matrix(), rng.uniform(-800, 800, 3))
    ids, Tt, Tc, probe = [], [], [], []
    for i in range(8):
        Rt = Rotation.from_rotvec(rng.normal(0, 0.3, 3)).as_matrix()
        tt = rng.uniform(-500, 500, 3) + [0, 0, 1500]
        T_gt = T_of(Rt, tt)
        holes = {k: (T_gt[:3, :3] @ np.asarray(p) + T_gt[:3, 3] + rng.normal(0, 0.1, 3)).tolist()
                 for k, p in pts.items()}
        Ti, rms, _ = pose_from_tracker(holes, pts)
        probe.append(rms)
        dR = Rotation.from_rotvec(rng.normal(0, np.radians(0.3) / np.sqrt(3), 3)).as_matrix()
        dt = rng.normal(0, 2.0 / np.sqrt(3), 3)
        Tc.append(X_true @ T_of(T_gt[:3, :3] @ dR, T_gt[:3, 3] + dt))
        Tt.append(Ti); ids.append(f'P{i + 1}')
    m1 = method1(ids, Tc, Tt)
    X = fit_X(Tc, Tt)
    m2 = residuals_vs_X(ids, Tc, Tt, X)
    res = report(m1, m2, probe, 'SELFTEST', '  (주입 오차: 병진 ~2mm, 회전 ~0.3°)')
    t_ok = 0.8 <= res['m1']['t_med'] <= 5.0 and 0.5 <= res['m2']['t_med'] <= 4.0
    r_ok = 0.1 <= res['m1']['r_med'] <= 0.8 and 0.05 <= res['m2']['r_med'] <= 0.6
    print('SELFTEST', 'PASS' if (t_ok and r_ok and max(probe) < 0.5) else 'FAIL')


def main():
    d = json.load(open(args.input))
    cls = d['class']
    cad = ps.load_cad()
    pin = float(d.get('pin_offset_mm', 0.0))
    net, dev = ps.hc.load_model()
    ids, Tt, Tc, probe, cam_stats = [], [], [], [], {}
    for pid, e in d['poses'].items():
        Ti, rms, res = pose_from_tracker(e['tracker_holes_mm'], cad[cls]['pts'], pin)
        if rms > 2.0:
            print(f'경고: {pid} 트래커 정합 잔차 {rms:.2f}mm > 2mm — 측정 재검 권장 {res}')
        sess = e['camera_session']
        Tci, agg = pose_from_session(os.path.join(ps.DOOR, sess), cls, cad, net, dev)
        ids.append(pid); Tt.append(Ti); Tc.append(Tci); probe.append(rms)
        cam_stats[pid] = dict(n_used=agg['n_used'], rms_med=agg['rms_med'], std=agg['std'])
    assert len(ids) >= 3, '포즈 3개 이상 필요 (권장 8)'
    m1 = method1(ids, Tc, Tt)
    X = fit_X(Tc, Tt)
    m2 = residuals_vs_X(ids, Tc, Tt, X)
    out = report(m1, m2, probe, f'{cls} {len(ids)}포즈')
    if 'reference_targets' in d:                     # 방법 3: 독립 등록
        rt = d['reference_targets']
        names = sorted(set(rt['tracker_mm']) & set(rt['camera_mm']))
        A = np.stack([rt['tracker_mm'][k] for k in names])
        B = np.stack([rt['camera_mm'][k] for k in names])
        R3, t3, rms3, _ = pu.umeyama_rigid(A, B)
        m3 = residuals_vs_X(ids, Tc, Tt, T_of(R3, t3))
        t3v = np.array([r[1] for r in m3]); r3v = np.array([r[3] for r in m3])
        print(f'방법3 절대({len(names)}기준점, 등록잔차 {rms3:.2f}mm): '
              f'병진 med {np.median(t3v):.2f} max {t3v.max():.2f} mm | '
              f'회전 med {np.median(r3v):.3f} °')
        out['m3'] = dict(reg_rms=float(rms3), t_med=float(np.median(t3v)),
                         poses=[dict(pose=p, dt=round(t, 2), dt_door=dd, dr=round(r, 3))
                                for p, t, dd, r in m3])
    out['camera'] = cam_stats
    out['input'] = args.input
    path = os.path.join(BASE, 'artifacts',
                        'eval_tracker_gt_' + os.path.basename(args.input).replace('.json', '') + '.json')
    json.dump(out, open(path, 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'→ {path}')


if __name__ == '__main__':
    t0 = time.time()
    selftest() if args.selftest else main()
    print(f'({time.time() - t0:.0f}s)')
