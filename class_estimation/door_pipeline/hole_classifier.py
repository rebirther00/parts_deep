"""홀 랜드마크 기반 도어 판별기 (추론 모듈).

classify(rgb, depth) → dict(pred, D_mm, gate, points, scores, ...)
  1) ResNet18-FPN 히트맵(16_train_hole_landmarks.py 학습)으로 볼트홀 4 + 상단 모서리 홀 2 검출
  2) depth 평면상 두 모서리 홀 거리(mm) × K_DEPTH[카메라 모드] → D
  3) 볼트 프레임 기하 게이트 통과 시 CAD D 최근접 클래스, 아니면 pred=None(보류)

스케일 상수 K_DEPTH는 근사 intrinsics(fx=1065) 편향 보정값 — 이미지 세로 해상도(카메라 모드)별로 GT 캘리브레이션:
  1080p(사무실 datasets) 0.8235, 1200p(현장 ZED X Mini HD1200) 0.8505

렌즈 독립 추론 (광각/협각 카메라 공용):
  K_DEPTH ≈ FX_APPROX / fx_true 이므로(0.8505 → fx≈1252, 0.8235 → fx≈1293) 수집 카메라는
  협각 4mm 렌즈(공장 캘리브 fx≈1274 @1200p)로 판단. 카메라의 실제 intrinsics 가 주어지면
  그것으로 역투영하고 잔차 K_METRIC = K_DEPTH × FX_REF / FX_APPROX 만 곱한다
  (수집 카메라에서는 기존 결과와 동일, 다른 렌즈에서는 fx 오차가 사라짐).
  검출망 입력 스케일까지 맞추려면 emulate_fx()로 FX_REF 화각을 에뮬레이션한다.
"""
import math
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

DOOR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DOOR, 'attribute_models', 'hole_landmarks', 'model.pth')
IN_W, IN_H, STRIDE = 1280, 768, 4
CH = ['bolt', 'corner_hinge', 'corner_latch']
K_DEPTH = {1080: 0.8235, 1200: 0.8505}
FX_APPROX = 1065.0                        # intrinsics 없을 때 쓰는 근사 fx (K_DEPTH 의 기준)
from camera_utils import FX_REF, emulate_fx   # noqa: E402  수집(학습) 카메라 fx 추정·화각 정합 (공용)
K_METRIC = {h: k * FX_REF / FX_APPROX for h, k in K_DEPTH.items()}   # 실제 intrinsics 사용 시 잔차
# 카메라(시리얼)별 실측 잔차 K — 19_checker_scale_calib.py 로 25mm 체커 측정 (depth 절대 스케일 편향).
# intrinsics 에 serial 이 있고 여기 등록돼 있으면 K_METRIC 대신 사용. 없으면 K_METRIC[세로해상도].
K_CAMERA = {
    57497357: 0.9797,   # 사무실 ZED X Mini 광각(fx 723), 2026-09-11 체커 1.28m 측정 (가로 +1.58%/세로 +2.57%) — 검증 중
}


def active_k(intrinsics):
    """intrinsics 에 대해 실제 적용되는 잔차 K 와 출처 ('camera' | 'metric' | 'depth')."""
    if intrinsics:
        sn = intrinsics.get('serial')
        if sn in K_CAMERA:
            return K_CAMERA[sn], 'camera'
        h = intrinsics.get('height', 1080)
        return K_METRIC.get(h, K_METRIC[1080]), 'metric'
    return None, 'depth'
CAD_D = {'E23_door_LH_FRT': 456, 'E25_door_LH_FRT': 724, 'E30_door_LH_FRT': 765, 'E38_door_LH_FRT': 812,
         'E25_door_LH_RR': 1037, 'E30_door_LH_RR': 1158, 'E38_door_LH_RR': 1352, 'E25_door_RH': 886, 'E30_E38_door_RH': 1087}
# D = 도어 폭 − 106mm. E23은 2026-09-07 추가(STP 폭 562 → 456, 현장 실측 중앙값 460)
D_RANGE = (400, 1500)   # 유효 코너 홀 거리(mm) — 게이트(볼트 스케일)와 최종 depth D 공통
GROUP = {c: ('FRT' if 'FRT' in c else 'RH' if c.endswith('RH') else 'RR') for c in CAD_D}
MEAN = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        r = models.resnet18(weights=None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.l1 = nn.Sequential(r.maxpool, r.layer1)
        self.l2, self.l3, self.l4 = r.layer2, r.layer3, r.layer4
        self.lat4 = nn.Conv2d(512, 128, 1); self.lat3 = nn.Conv2d(256, 128, 1)
        self.lat2 = nn.Conv2d(128, 128, 1); self.lat1 = nn.Conv2d(64, 128, 1)
        self.head = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                                  nn.Conv2d(64, len(CH), 1))

    def forward(self, x):
        c1 = self.l1(self.stem(x)); c2 = self.l2(c1); c3 = self.l3(c2); c4 = self.l4(c3)
        p = self.lat4(c4)
        p = F.interpolate(p, size=c3.shape[2:], mode='bilinear', align_corners=False) + self.lat3(c3)
        p = F.interpolate(p, size=c2.shape[2:], mode='bilinear', align_corners=False) + self.lat2(c2)
        p = F.interpolate(p, size=c1.shape[2:], mode='bilinear', align_corners=False) + self.lat1(c1)
        return self.head(p)


def load_model(path=MODEL_PATH, device=None):
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    net = Net().to(dev); net.load_state_dict(torch.load(path, map_location=dev)); net.eval()
    return net, dev


def _letterbox(h, w):
    s = min(IN_W / w, IN_H / h)
    return np.array([[s, 0, (IN_W - w * s) / 2], [0, s, (IN_H - h * s) / 2]], np.float32)


def _apply(M, p):
    return np.array([M[0, 0] * p[0] + M[0, 1] * p[1] + M[0, 2], M[1, 0] * p[0] + M[1, 1] * p[1] + M[1, 2]])


_dev_cache = {}


def _dev_buffers(dev):
    """디바이스별 정규화 상수 + 고정(pinned) 업로드 버퍼 (1회 생성, 재사용)."""
    key = str(dev)
    if key not in _dev_cache:
        pin = (torch.empty((IN_H, IN_W, 3), dtype=torch.uint8).pin_memory()
               if dev.type == 'cuda' else torch.empty((IN_H, IN_W, 3), dtype=torch.uint8))
        _dev_cache[key] = dict(mean=MEAN.to(dev), std=STD.to(dev), pin=pin)
    return _dev_cache[key]


@torch.no_grad()
def _peaks_gpu(hm, ks, thr=0.15, nms=5):
    """GPU 피크 탐색. hm: (C,H,W) float 텐서, ks: 채널별 최대 개수.
    → 채널별 [(x, y, score)] (히트맵 픽셀 좌표, 서브픽셀 보정). 결과는 이전 CPU 버전(_peaks)과 동일.
    D2H 전송은 채널 전체를 합쳐 1회만 한다."""
    C, H, W = hm.shape
    mx = F.max_pool2d(hm[None], nms * 2 + 1, stride=1, padding=nms)[0]
    cand = (hm == mx) & (hm > thr)
    rows, counts = [], []
    for c, k in enumerate(ks):
        idx = cand[c].nonzero()                      # (n, 2) = (y, x)
        n = idx.shape[0]
        if n == 0:
            counts.append(0)
            continue
        y, x = idx[:, 0], idx[:, 1]
        sc = hm[c, y, x]
        if n > k:
            sc, o = torch.topk(sc, k)
        else:
            sc, o = torch.sort(sc, descending=True)
        y, x = y[o], x[o]
        v = hm[c, y, x]
        l = hm[c, y, (x - 1).clamp(0, W - 1)]; r = hm[c, y, (x + 1).clamp(0, W - 1)]
        u = hm[c, (y - 1).clamp(0, H - 1), x]; d = hm[c, (y + 1).clamp(0, H - 1), x]
        dx = (0.5 * (r - l) / torch.clamp(2 * v - r - l, min=1e-6)).clamp(-1, 1)
        dy = (0.5 * (d - u) / torch.clamp(2 * v - d - u, min=1e-6)).clamp(-1, 1)
        dx = torch.where((x > 0) & (x < W - 1), dx, torch.zeros_like(dx))
        dy = torch.where((y > 0) & (y < H - 1), dy, torch.zeros_like(dy))
        rows.append(torch.stack([x.float() + dx, y.float() + dy, v], 1))
        counts.append(len(o))
    flat = torch.cat(rows, 0).cpu().numpy() if rows else np.zeros((0, 3), np.float32)
    out, i = [], 0
    for n in counts:
        out.append([(float(a), float(b), float(c_)) for a, b, c_ in flat[i:i + n]])
        i += n
    return out


def _peaks(hm, k, thr=0.15, nms=5):
    """(호환용) 단일 채널 numpy 히트맵 CPU 피크 탐색. 실시간 경로는 _peaks_gpu 사용."""
    return _peaks_gpu(torch.from_numpy(np.ascontiguousarray(hm))[None], [k], thr, nms)[0]


@torch.no_grad()
def detect(net, dev, rgb):
    """원본 BGR → {'bolt': [(x,y,score)×≤4], 'corner_hinge': [...≤1], 'corner_latch': [...≤1]} (원본 픽셀).

    letterbox(warpAffine)만 CPU, 이후 BGR→RGB·정규화·forward·피크 탐색은 모두 GPU에서 수행."""
    h, w = rgb.shape[:2]
    M = _letterbox(h, w)
    x = cv2.warpAffine(rgb, M, (IN_W, IN_H), borderValue=(114, 114, 114))
    buf = _dev_buffers(dev)
    buf['pin'].numpy()[:] = x                                   # pinned 버퍼로 memcpy
    t = buf['pin'].to(dev, non_blocking=True)                   # uint8 H2D (3MB)
    t = t.permute(2, 0, 1)[[2, 1, 0]].float()[None] / 255.      # BGR→RGB, GPU
    t = (t - buf['mean']) / buf['std']
    hm = net(t)[0].float()
    ks = [4 if c == 'bolt' else 1 for c in CH]
    pk = _peaks_gpu(hm, ks)
    Mi = cv2.invertAffineTransform(M)
    return {c: [(*_apply(Mi, (px * STRIDE, py * STRIDE)), sc) for px, py, sc in pk[ci]]
            for ci, c in enumerate(CH)}


def bolt_frame(bolts):
    """볼트 4점 → 중심·157축·96축·px/mm. 직사각형 불일치면 None."""
    if len(bolts) < 4:
        return None
    P = np.array([(b[0], b[1]) for b in bolts[:4]], float)
    c = P.mean(0); Q = P - c
    pairs = sorted(((np.linalg.norm(P[a] - P[b]), a, b) for a in range(4) for b in range(a + 1, 4)), key=lambda t: t[0])
    _, i, j = pairs[2]
    ex = P[j] - P[i]; ex /= np.linalg.norm(ex); ey = np.array([-ex[1], ex[0]])
    u = Q @ ex; w = Q @ ey
    s1, s2 = (u.max() - u.min()) / 157.0, (w.max() - w.min()) / 96.0
    if abs(s1 - s2) / max(s1, s2) > 0.18:
        return None
    return dict(c=c, ex=ex, ey=ey, s=(s1 + s2) / 2)


def geometry_gate(fr, hinge, latch, shape, margin=20):
    if hinge is None or latch is None:
        return 'no_corner'
    if fr is None:
        return 'no_frame'
    h, w = shape[:2]
    for p in (hinge, latch):
        if p[0] < margin or p[1] < margin or p[0] > w - margin or p[1] > h - margin:
            return 'near_border'
    s = fr['s']
    vl = np.array(latch[:2]) - fr['c']; ul, wl = vl @ fr['ex'] / s, vl @ fr['ey'] / s
    vh = np.array(hinge[:2]) - fr['c']; uh, wh = vh @ fr['ex'] / s, vh @ fr['ey'] / s
    if not (100 <= abs(ul) <= 230 and 120 <= abs(wl) <= 270):
        return 'latch_offset'
    if abs(wh - wl) > 70:
        return 'not_collinear'
    if np.sign(uh) == np.sign(ul):
        return 'same_side'
    if not (D_RANGE[0] <= abs(uh - ul) <= D_RANGE[1]):
        return 'D_range'
    return 'ok'


BOLT_PITCH = (157.0, 96.0)   # 볼트홀 4개 직사각형 CAD 피치(mm): 장변, 단변
# 볼트 정규화: D 를 (CAD 장변 / depth 로 잰 장변) 배 보정. 볼트 직사각형은 코너 홀과 같은 평면 위의
# 기지 치수이므로 depth 스케일 편향·잔차 K·렌즈 차이가 모두 상쇄된다 (K_DEPTH 자체가 볼트 피치로
# 1회 캘리브레이션한 값 → 이를 프레임마다 수행하는 셈). 장변만 쓰는 이유: D 와 같은 축(ex, 157 축)이라
# 평면 기울기 오차 방향이 같고, 픽셀 국소화 오차의 상대 비율이 단변보다 작다.
BOLT_NORM_RANGE = (0.85, 1.15)   # 이 범위 밖의 보정 배율은 볼트 오검출로 보고 정규화 생략


def plane_from_depth(depth, pts_all, intrinsics=None):
    """검출점 볼록껍질 내부 depth 로 평면 피팅.

    반환 (to3d, k): to3d(p)=픽셀→평면상 3D(mm), k=스케일 잔차. 실패 시 None.
    intrinsics 가 있으면 실제 fx/fy/cx/cy 로 역투영 후 K_METRIC 잔차, 없으면
    근사 fx=FX_APPROX 로 역투영 후 K_DEPTH(수집 카메라 전용 보정)."""
    h, w = depth.shape
    if intrinsics:
        K = intrinsics
        KTAB = ({h: K_CAMERA[intrinsics['serial']] for h in K_METRIC}
                if intrinsics.get('serial') in K_CAMERA else K_METRIC)
    else:
        K = dict(fx=FX_APPROX, fy=FX_APPROX, cx=w / 2.0, cy=h / 2.0); KTAB = K_DEPTH
    P = np.array([(p[0], p[1]) for p in pts_all], np.float32)
    if len(P) < 3:
        return None
    m = np.zeros(depth.shape, np.uint8); cv2.fillConvexPoly(m, cv2.convexHull(P).astype(np.int32), 255)
    m = cv2.erode(m, np.ones((15, 15), np.uint8))
    rows, cols = np.where((m > 0) & (depth > 0))
    if len(rows) < 500:
        return None
    if len(rows) > 40000:
        sel = np.random.default_rng(0).choice(len(rows), 40000, replace=False); rows, cols = rows[sel], cols[sel]
    z = depth[rows, cols].astype(np.float64)
    Q = np.stack([(cols - K['cx']) * z / K['fx'], (rows - K['cy']) * z / K['fy'], z], 1)
    c = Q.mean(0); _, _, Vt = np.linalg.svd(Q - c, full_matrices=False); n = Vt[2]
    keep = np.abs((Q - c) @ n) < 10
    if keep.sum() > 200:
        c = Q[keep].mean(0); _, _, Vt = np.linalg.svd(Q[keep] - c, full_matrices=False); n = Vt[2]

    def to3d(p):
        r = np.array([(p[0] - K['cx']) / K['fx'], (p[1] - K['cy']) / K['fy'], 1.0])
        return r * (np.dot(c, n) / np.dot(r, n))
    k = KTAB.get(h, KTAB[1080])
    return to3d, k


def depth_distance_mm(depth, pts_all, pa, pb, intrinsics=None):
    """두 점의 depth 평면상 거리(mm). 평면 피팅 실패 시 None."""
    pf = plane_from_depth(depth, pts_all, intrinsics)
    if pf is None:
        return None
    to3d, k = pf
    return float(np.linalg.norm(to3d(pa) - to3d(pb))) * k


def bolt_pitch_mm(pf, bolts):
    """볼트 4점의 depth 평면상 직사각형 피치(mm): dict(long, short) — CAD BOLT_PITCH 와 비교용.

    6개 쌍거리 정렬 → 최소 2개 평균=단변, 다음 2개 평균=장변 (나머지 2개는 대각선)."""
    if pf is None or len(bolts) < 4:
        return None
    to3d, k = pf
    P = [to3d(b) for b in bolts[:4]]
    d = sorted(float(np.linalg.norm(P[a] - P[b])) * k for a in range(4) for b in range(a + 1, 4))
    return dict(long=(d[2] + d[3]) / 2, short=(d[0] + d[1]) / 2)


def nearest_class(D, group=None):
    """CAD D 최근접 클래스. group(FRT/RR/RH)이 주어지면 그 그룹 안에서만 탐색 (속성 파이프라인 그룹 판별과 결합)."""
    cands = [k for k in CAD_D if group is None or GROUP[k] == group]
    return min(cands, key=lambda k: abs(CAD_D[k] - D))


def classify(net, dev, rgb, depth=None, group=None, intrinsics=None, bolt_norm=False):
    """단일 프레임 판정. depth 없으면 볼트 피치 스케일 사용. group 지정 시 그룹 내 최근접.

    intrinsics: dict(fx, fy, cx, cy) — 카메라 실제 값(렌즈 무관 D). None 이면 수집 카메라 가정.
    bolt_norm: depth D 를 볼트 장변 CAD/실측 비율로 정규화 (D_src='depth+bolt', D_raw_mm 보존).
        기본 꺼짐 — 볼트 국소화 오차(-7% 관측)가 D 에 그대로 증폭되므로 실험용으로만."""
    det = detect(net, dev, rgb)
    hinge = det['corner_hinge'][0] if det['corner_hinge'] else None
    latch = det['corner_latch'][0] if det['corner_latch'] else None
    fr = bolt_frame(det['bolt'])
    gate = geometry_gate(fr, hinge, latch, rgb.shape)
    out = dict(points=det, gate=gate, pred=None, D_mm=None, D_src=None, group=None, bolt_mm=None,
               D_raw_mm=None, k_bolt=None)
    corners = [p for p in (hinge, latch) if p is not None]
    pf = None
    if depth is not None and len(det['bolt']) + len(corners) >= 3:
        pf = plane_from_depth(depth, det['bolt'] + corners, intrinsics)
        out['bolt_mm'] = bolt_pitch_mm(pf, det['bolt'])   # 스케일 체인(depth·intrinsics) 진단용
    if hinge is None or latch is None:
        return out
    D = None
    if pf is not None:
        D = float(np.linalg.norm(pf[0](hinge) - pf[0](latch))) * pf[1]; out['D_src'] = 'depth'
        if D is not None and bolt_norm and out['bolt_mm'] and out['bolt_mm']['long'] > 0:
            kb = BOLT_PITCH[0] / out['bolt_mm']['long']
            if BOLT_NORM_RANGE[0] <= kb <= BOLT_NORM_RANGE[1]:
                out['D_raw_mm'], out['k_bolt'] = D, kb
                D *= kb; out['D_src'] = 'depth+bolt'
    if D is None and fr is not None:
        D = math.hypot(hinge[0] - latch[0], hinge[1] - latch[1]) / fr['s']; out['D_src'] = 'bolt'
    out['D_mm'] = D
    if D is not None and gate == 'ok' and not (D_RANGE[0] <= D <= D_RANGE[1]):
        gate = out['gate'] = 'D_range'   # depth 평면 피팅 붕괴(예: 작업자 가림) 시 D가 비현실적 → 보류
    if D is not None and gate == 'ok':
        out['pred'] = nearest_class(D, group)
        out['group'] = GROUP[out['pred']]
        cands = [k for k in CAD_D if group is None or GROUP[k] == group]
        ds = sorted(abs(CAD_D[k] - D) for k in cands)
        out['margin_mm'] = float((ds[1] if len(ds) > 1 else 1e9) - ds[0])
    return out


def aggregate(results, group=None):
    """N프레임 집계: 게이트 통과 프레임의 D 중앙값 → 클래스(group 지정 시 그룹 내). 판정 프레임이 없으면 None."""
    Ds = [r['D_mm'] for r in results if r.get('gate') == 'ok' and r.get('D_mm')]
    if not Ds:
        return dict(pred=None, D_mm=None, n_judged=0, n=len(results))
    D = float(np.median(Ds)); pred = nearest_class(D, group)
    cands = [k for k in CAD_D if group is None or GROUP[k] == group]
    ds = sorted(abs(CAD_D[k] - D) for k in cands)
    return dict(pred=pred, group=GROUP[pred], D_mm=D, n_judged=len(Ds), n=len(results),
                margin_mm=float((ds[1] if len(ds) > 1 else 1e9) - ds[0]))
