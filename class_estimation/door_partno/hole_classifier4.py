"""4채널 홀 랜드마크 판별기 — bolt(4) / corner_hinge / corner_latch / bracket(2) 를 한 모델로 검출해
형상군(코너 홀 거리 D, door_pipeline 과 동일 규칙)과 레이더 옵션(브래킷 홀 피크 + CAD 기하 검증)을 한 번에 판정한다.

door_pipeline/hole_classifier.py(3채널, 운영 정본)를 임포트해 CAD_D·게이트·스케일 규칙을 그대로 쓰고, 검출망만 4채널로 확장한다.
  classify(net, dev, rgb, depth, intrinsics) → dict(pred, D_mm, gate, radar, radar_info, points)
레이더 판정(브래킷 채널):
  피크 2개 점수 ≥ PEAK_ON 이고, 6점 아핀으로 도어 프레임에 놓았을 때 CAD 브래킷 홀(partno/radar_bracket.json)과 각각 ≤ TOL_MM,
  두 피크 간격이 91.6±PITCH_TOL mm 이면 'radar'. 최대 피크 점수 < PEAK_OFF 이면 'none'. 그 사이는 'unsure'.
  (판 높이 70mm 시차로 투영이 ~10mm 어긋날 수 있어 TOL_MM 40)
"""
import math, os, sys
import cv2, numpy as np, torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
DOOR = os.path.normpath(os.path.join(HERE, '..', 'door_pipeline'))
sys.path.insert(0, DOOR); sys.path.insert(0, os.path.join(DOOR, 'partno'))
import hole_classifier as base   # door_pipeline 정본(3채널) — 이 파일은 hole_classifier4 라 이름 충돌 없음
import radar_check as rc

MODEL_PATH = os.path.join(HERE, 'attribute_models', 'hole_landmarks_bracket', 'model.pth')
CH = base.CH + ['bracket']                 # ['bolt', 'corner_hinge', 'corner_latch', 'bracket']
KS = [4, 1, 1, 2]
PEAK_ON, PEAK_OFF = 0.30, 0.12
TOL_MM, PITCH_TOL = 40.0, 20.0
CAD_D, GROUP, UNKNOWN = base.CAD_D, base.GROUP, base.UNKNOWN
_geom = None


def geometry():
    global _geom
    if _geom is None:
        _geom = rc.load_geometry()
    return _geom


class Net(base.Net):
    """door_pipeline.Net 과 동일 구조, 출력 채널만 4."""
    def __init__(self):
        super().__init__()
        self.head[-1] = nn.Conv2d(64, len(CH), 1)


def init_from_base(net, base_path=base.MODEL_PATH):
    """3채널 정본 가중치로 초기화 — 마지막 1×1 conv 는 앞 3채널만 복사, 브래킷 채널은 작은 난수."""
    sd = torch.load(base_path, map_location='cpu')
    own = net.state_dict()
    for k, v in sd.items():
        if k in own and own[k].shape == v.shape:
            own[k] = v
        elif k in own and k.startswith('head.') and own[k].shape[0] == len(CH):
            own[k][:v.shape[0]] = v
    net.load_state_dict(own)
    return net


def load_model(path=MODEL_PATH, device=None):
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    net = Net().to(dev)
    net.load_state_dict(torch.load(path, map_location=dev)); net.eval()
    return net, dev


@torch.no_grad()
def detect(net, dev, rgb):
    """원본 BGR → {'bolt': [(x,y,score)×≤4], 'corner_hinge': [≤1], 'corner_latch': [≤1], 'bracket': [≤2]} (원본 픽셀)."""
    h, w = rgb.shape[:2]
    M = base._letterbox(h, w)
    x = cv2.warpAffine(rgb, M, (base.IN_W, base.IN_H), borderValue=(114, 114, 114))
    t = torch.from_numpy(x[:, :, ::-1].copy()).permute(2, 0, 1).float()[None] / 255.
    t = ((t - base.MEAN) / base.STD).to(dev)
    hm = net(t)[0].float().cpu().numpy()
    Mi = cv2.invertAffineTransform(M)
    return {c: [(*base._apply(Mi, (px * base.STRIDE, py * base.STRIDE)), sc) for px, py, sc in base._peaks(hm[ci], KS[ci])]
            for ci, c in enumerate(CH)}


def radar_from_bracket(det, cls, geom=None):
    """브래킷 채널 피크 → 'radar' | 'none' | 'unsure' + 근거."""
    geom = geom or geometry(); g = geom.get(cls)
    pk = det.get('bracket', [])
    info = dict(n_peaks=len(pk), scores=[round(p[2], 3) for p in pk], dist_mm=None, pitch_mm=None)
    if g is None:
        return 'n/a', info
    top = max((p[2] for p in pk), default=0.0)
    if len(pk) < 2 or min(p[2] for p in pk[:2]) < PEAK_ON:
        return ('none' if top < PEAK_OFF else 'unsure'), info
    fit = rc.door_affine(det, g['lm'])
    if fit is None or fit[1] > rc.MAX_RESID_PX:
        return 'unsure', info
    A = fit[0]; A3 = np.vstack([A, [0, 0, 1]]); Ai = np.linalg.inv(A3)
    P = [Ai @ np.array([p[0], p[1], 1.0]) for p in pk[:2]]
    P = [q[:2] / q[2] for q in P]
    exp = [np.array(h) for h in g['holes']]
    d = [min(np.linalg.norm(p - e) for e in exp) for p in P]
    pitch = float(np.linalg.norm(P[0] - P[1]))
    info.update(dist_mm=[round(float(x), 1) for x in d], pitch_mm=round(pitch, 1))
    ok = max(d) <= TOL_MM and abs(pitch - g['pitch_mm']) <= PITCH_TOL
    return ('radar' if ok else 'unsure'), info


def classify(net, dev, rgb, depth=None, intrinsics=None, geom=None, unknown_mm=base.UNKNOWN_MM):
    """형상군(door_pipeline 규칙: 픽셀 폭 D) + 레이더(브래킷 피크) 프레임 판정."""
    det = detect(net, dev, rgb)
    hinge = det['corner_hinge'][0] if det['corner_hinge'] else None
    latch = det['corner_latch'][0] if det['corner_latch'] else None
    fr = base.bolt_frame(det['bolt'])
    gate = base.geometry_gate(fr, hinge, latch, rgb.shape)
    out = dict(points=det, gate=gate, pred=None, D_mm=None, D_src=None, margin_mm=None, radar=None, radar_info=None)
    if hinge is None or latch is None:
        return out
    corners = [hinge, latch]; pf = None
    if depth is not None and len(det['bolt']) + 2 >= 3:
        pf = base.plane_from_depth(depth, det['bolt'] + corners, intrinsics)
    span = math.hypot(hinge[0] - latch[0], hinge[1] - latch[1])
    sn = (intrinsics or {}).get('serial'); D = None
    if sn in base.S_PIXEL and intrinsics.get('fx'):
        D = span * base.S_PIXEL[sn] / intrinsics['fx']; out['D_src'] = 'pixel'
    elif pf is not None:
        D = float(np.linalg.norm(pf[0](hinge) - pf[0](latch))) * pf[1]; out['D_src'] = 'depth'
    elif fr is not None:
        D = span / fr['s']; out['D_src'] = 'bolt'
    out['D_mm'] = D
    if D is not None and gate == 'ok' and not (base.D_RANGE[0] <= D <= base.D_RANGE[1]):
        gate = out['gate'] = 'D_range'
    if D is not None and gate == 'ok':
        out['pred'], _, out['margin_mm'] = base.judge(D, None, unknown_mm)
    cls = out['pred'] if out['pred'] in CAD_D else None
    if cls is not None and GROUP[cls] != 'FRT' and gate == 'ok':
        out['radar'], out['radar_info'] = radar_from_bracket(det, cls, geom)
    elif cls is not None:
        out['radar'] = 'n/a'
    return out


def aggregate_radar(results, min_judged=3, min_agree=0.8):
    """프레임 radar 판정 → 세션 플래그(1/0/None), 표."""
    votes = [r['radar'] for r in results if r.get('radar') in ('radar', 'none')]
    n1, n0 = votes.count('radar'), votes.count('none')
    flag = None
    if len(votes) >= min_judged and max(n1, n0) / len(votes) >= min_agree:
        flag = 1 if n1 > n0 else 0
    return dict(flag=flag, n_radar=n1, n_none=n0, n_judged=len(votes), n=len(results))
