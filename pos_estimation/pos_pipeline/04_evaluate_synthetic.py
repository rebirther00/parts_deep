"""합성 GT 검증 — 절대 정확도 (지표 체계의 3축, 정의 2026-08-31).

STL 메시를 GT 포즈로 z-버퍼 렌더링(depth, 배경 평면 포함)하고 CAD 랜드마크를
투영(+노이즈)해 검출기를 우회한 채 기하 체인(평면 피팅 → 리세스 오프셋 역투영 →
볼트 대응 → Umeyama)의 출력을 GT와 직접 비교한다. 검출기는 현장 학습 유지 —
여기서 검증하는 것은 솔버뿐이므로 '합성 학습 금지' 원칙과 무충돌.

조건: clean(노이즈 0 — 체인 정합성) / field(랜드마크 1px + depth 3mm — 현장 유사)
     / stress(2px + 6mm). depth_scale=1.0(합성은 무편향), intrinsics는 현장 실측값.

  python 04_evaluate_synthetic.py [--n_poses 6]
출력: artifacts/eval_synthetic.json + 표 (오차는 도어 프레임 분해: 면내 dxy / 법선 dz)
"""
import argparse
import json
import os
import time

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
import pose_solver as ps
import pose_utils as pu
from stl import mesh as stl_mesh

W, H = 1920, 1200
CALIB = dict(fx=1269.746, fy=1269.746, cx=961.669, cy=598.594, depth_scale=1.0)
COND = dict(clean=(0.0, 0.0), field=(1.0, 3.0), stress=(2.0, 6.0))   # (랜드마크 px, depth mm)
BG_OFFSET = 400.0    # 배경 평면: 도어 뒤 mm (홀·벤트 관통 픽셀이 배경 depth를 갖게)

ap = argparse.ArgumentParser()
ap.add_argument('--n_poses', type=int, default=6)
ap.add_argument('--seed', type=int, default=0)
args = ap.parse_args()


def load_door_mesh(cls, cad):
    """STL → 도어 로컬 프레임 삼각형 (n,3,3)."""
    m = stl_mesh.Mesh.from_file(os.path.join(ps.BASE, '..', '..', 'cad', 'door_stl', cls + '.stl'))
    T = np.array(json.load(open(os.path.join(BASE, 'cad_holes.json')))['classes'][cls]['T_door_to_cad'])
    R, o = T[:3, :3], T[:3, 3]
    v = m.vectors.astype(np.float64).reshape(-1, 3)
    return ((v - o) @ R).reshape(-1, 3, 3)          # cad→door: Rᵀ(v-o)


def sample_pose(rng, tris_door):
    """도어가 화면 안에 들어오는 무작위 GT 포즈 T_cam→door."""
    F = np.diag([1.0, -1.0, -1.0])                   # 정대면(도어 +Z → 카메라)
    for _ in range(50):
        th = np.radians(rng.uniform(-15, 15))
        tilt = np.radians(rng.uniform(0, 15)); ax = rng.uniform(0, 2 * np.pi)
        ct, st = np.cos(th), np.sin(th)
        Rz = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
        k = np.array([np.cos(ax), np.sin(ax), 0.0])
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Rt = np.eye(3) + np.sin(tilt) * Kx + (1 - np.cos(tilt)) * (Kx @ Kx)
        R = Rz @ Rt @ F
        # 도어 무게중심이 화면 중앙 근처로 오도록 t 선택
        c_door = tris_door.reshape(-1, 3).mean(0)
        z = rng.uniform(1300, 1750)
        t = np.array([rng.uniform(-80, 80), rng.uniform(-80, 80), z]) - R @ c_door
        V = tris_door.reshape(-1, 3) @ R.T + t
        u = CALIB['fx'] * V[:, 0] / V[:, 2] + CALIB['cx']
        v = CALIB['fy'] * V[:, 1] / V[:, 2] + CALIB['cy']
        if u.min() > 30 and u.max() < W - 30 and v.min() > 30 and v.max() < H - 30 and V[:, 2].min() > 300:
            return R, t
    raise RuntimeError('포즈 샘플 실패')


def render_depth(tris_cam):
    """z-버퍼 래스터라이저 (1/z 보간, 원근 보정). 배경 평면 포함 uint16 mm."""
    zmax = float(tris_cam[:, :, 2].max())
    depth = np.full((H, W), zmax + BG_OFFSET)        # 배경 평면
    P = tris_cam[(tris_cam[:, :, 2] > 100).all(1)]
    u = CALIB['fx'] * P[:, :, 0] / P[:, :, 2] + CALIB['cx']
    v = CALIB['fy'] * P[:, :, 1] / P[:, :, 2] + CALIB['cy']
    iz = 1.0 / P[:, :, 2]
    x0 = np.clip(np.floor(u.min(1)).astype(int), 0, W - 1)
    x1 = np.clip(np.ceil(u.max(1)).astype(int) + 1, 0, W)
    y0 = np.clip(np.floor(v.min(1)).astype(int), 0, H - 1)
    y1 = np.clip(np.ceil(v.max(1)).astype(int) + 1, 0, H)
    area = (u[:, 1] - u[:, 0]) * (v[:, 2] - v[:, 0]) - (u[:, 2] - u[:, 0]) * (v[:, 1] - v[:, 0])
    for i in np.where((np.abs(area) > 1e-9) & (x1 > x0) & (y1 > y0))[0]:
        xs = np.arange(x0[i], x1[i]); ys = np.arange(y0[i], y1[i])
        gx, gy = np.meshgrid(xs + 0.5, ys + 0.5)
        w0 = ((u[i, 1] - gx) * (v[i, 2] - gy) - (u[i, 2] - gx) * (v[i, 1] - gy)) / area[i]
        w1 = ((u[i, 2] - gx) * (v[i, 0] - gy) - (u[i, 0] - gx) * (v[i, 2] - gy)) / area[i]
        w2 = 1.0 - w0 - w1
        ins = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not ins.any():
            continue
        zpx = 1.0 / (w0 * iz[i, 0] + w1 * iz[i, 1] + w2 * iz[i, 2])
        sub = depth[y0[i]:y1[i], x0[i]:x1[i]]
        upd = ins & (zpx < sub)
        sub[upd] = zpx[upd]
    return depth


def synth_det(cad_pts, R, t, rng, sigma):
    """GT 투영 + 노이즈 → hole_classifier.detect 형태 (볼트 순서 셔플)."""
    px = {}
    for k, p in cad_pts.items():
        q = R @ p + t
        px[k] = (CALIB['fx'] * q[0] / q[2] + CALIB['cx'] + rng.normal(0, sigma),
                 CALIB['fy'] * q[1] / q[2] + CALIB['cy'] + rng.normal(0, sigma), 1.0)
    bolts = [px[k] for k in ps.BOLT_NAMES]
    rng.shuffle(bolts)
    return dict(bolt=bolts, corner_hinge=[px['corner_hinge']], corner_latch=[px['corner_latch']])


def main():
    cad = ps.load_cad()
    K = pu.intrinsics_for((H, W), calib=CALIB)
    rng = np.random.default_rng(args.seed)
    rows = []
    t0 = time.time()
    for cls in sorted(cad):
        tris_door = load_door_mesh(cls, cad)
        for ip in range(args.n_poses):
            R, t = sample_pose(rng, tris_door)
            depth_clean = render_depth((tris_door.reshape(-1, 3) @ R.T + t).reshape(-1, 3, 3))
            for cond, (s_lm, s_d) in COND.items():
                d = depth_clean + (rng.normal(0, s_d, depth_clean.shape) if s_d else 0)
                d16 = np.clip(d, 0, 65535).astype(np.uint16)
                det = synth_det(cad[cls]['pts'], R, t, rng, s_lm)
                f = ps.solve_frame(det, d16, cls, cad, K)
                if not f.get('ok'):
                    rows.append(dict(cls=cls, pose=ip, cond=cond, ok=False, reason=f.get('reason')))
                    continue
                dt_cam = f['t'] - t
                dt_door = R.T @ dt_cam                       # 면내(x,y) vs 법선(z) 분해
                dR = R.T @ f['R']
                dang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
                rows.append(dict(cls=cls, pose=ip, cond=cond, ok=True,
                                 dx=float(dt_door[0]), dy=float(dt_door[1]), dz=float(dt_door[2]),
                                 dxy=float(np.hypot(dt_door[0], dt_door[1])),
                                 dz_cam=float(dt_cam[2]), dang=dang, rms=f['rms'],
                                 gate=f['gate'], n_holes=f['n_holes']))
    # 요약
    print(f"합성 검증: {len(set(r['cls'] for r in rows))}클래스 × {args.n_poses}포즈 × {len(COND)}조건  "
          f"({time.time() - t0:.0f}s)\n")
    print(f"{'조건':8s} {'n':>3s} {'ok':>3s} | {'면내 dxy':>12s} {'법선 dz':>12s} {'회전각':>11s} | {'잔차rms':>7s}")
    print(f"{'':8s} {'':>3s} {'':>3s} | {'med':>5s} {'p95':>6s} {'med':>5s} {'p95':>6s} {'med':>5s} {'p95':>5s} | {'med':>7s}")
    summary = {}
    for cond in COND:
        rs = [r for r in rows if r['cond'] == cond]
        ok = [r for r in rs if r['ok']]
        if not ok:
            print(f'{cond:8s} {len(rs):3d}   0 | 전부 실패'); continue
        g = lambda k: np.array([r[k] for r in ok])
        s = dict(n=len(rs), ok=len(ok),
                 dxy_med=float(np.median(g('dxy'))), dxy_p95=float(np.percentile(g('dxy'), 95)),
                 dz_med=float(np.median(np.abs(g('dz')))), dz_p95=float(np.percentile(np.abs(g('dz')), 95)),
                 dang_med=float(np.median(g('dang'))), dang_p95=float(np.percentile(g('dang'), 95)),
                 rms_med=float(np.median(g('rms'))))
        summary[cond] = s
        print(f"{cond:8s} {s['n']:3d} {s['ok']:3d} | {s['dxy_med']:5.2f} {s['dxy_p95']:6.2f} "
              f"{s['dz_med']:5.2f} {s['dz_p95']:6.2f} {s['dang_med']:5.2f} {s['dang_p95']:5.2f} | "
              f"{s['rms_med']:7.2f}")
    out = os.path.join(BASE, 'artifacts', 'eval_synthetic.json')
    json.dump(dict(calib=CALIB, cond={k: list(v) for k, v in COND.items()},
                   n_poses=args.n_poses, seed=args.seed, summary=summary, rows=rows),
              open(out, 'w'), ensure_ascii=False, indent=1, default=float)
    print(f'\n(오차 = GT 대비 절대 오차, 도어 프레임 분해. 목표 ±5mm/±1°)\n→ {out}')


if __name__ == '__main__':
    main()
