"""홀 랜드마크 기반 도어 판별기 (추론 모듈).

classify(rgb, depth) → dict(pred, D_mm, gate, points, scores, ...)
  1) ResNet18-FPN 히트맵(15_train_hole_landmarks.py 학습)으로 볼트홀 4 + 상단 모서리 홀 2 검출
  2) depth 평면상 두 모서리 홀 거리(mm) × K_DEPTH[카메라 모드] → D
  3) 볼트 프레임 기하 게이트 통과 시 CAD D 최근접 클래스, 아니면 pred=None(보류)

스케일 상수 K_DEPTH는 근사 intrinsics(fx=1065) 편향 보정값 — 이미지 세로 해상도(카메라 모드)별로 GT 캘리브레이션:
  1080p(사무실 datasets) 0.8235, 1200p(현장 ZED X Mini HD1200) 0.8505
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
CAD_D = {'E25_door_LH_FRT': 724, 'E30_door_LH_FRT': 765, 'E38_door_LH_FRT': 812, 'E25_door_LH_RR': 1037,
         'E30_door_LH_RR': 1158, 'E38_door_LH_RR': 1352, 'E25_door_RH': 886, 'E30_E38_door_RH': 1087}
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


def _peaks(hm, k, thr=0.15, nms=5):
    h = torch.from_numpy(hm)[None, None]
    mx = F.max_pool2d(h, nms * 2 + 1, stride=1, padding=nms)
    cand = ((h == mx) & (h > thr))[0, 0].nonzero().numpy()
    out = []
    for y, x in cand:
        dx = dy = 0.0
        if 0 < x < hm.shape[1] - 1:
            dx = 0.5 * (hm[y, x + 1] - hm[y, x - 1]) / max(1e-6, (2 * hm[y, x] - hm[y, x + 1] - hm[y, x - 1]))
        if 0 < y < hm.shape[0] - 1:
            dy = 0.5 * (hm[y + 1, x] - hm[y - 1, x]) / max(1e-6, (2 * hm[y, x] - hm[y + 1, x] - hm[y - 1, x]))
        out.append((float(x + np.clip(dx, -1, 1)), float(y + np.clip(dy, -1, 1)), float(hm[y, x])))
    out.sort(key=lambda t: -t[2])
    return out[:k]


@torch.no_grad()
def detect(net, dev, rgb):
    """원본 BGR → {'bolt': [(x,y,score)×≤4], 'corner_hinge': [...≤1], 'corner_latch': [...≤1]} (원본 픽셀)."""
    h, w = rgb.shape[:2]
    M = _letterbox(h, w)
    x = cv2.warpAffine(rgb, M, (IN_W, IN_H), borderValue=(114, 114, 114))
    t = torch.from_numpy(x[:, :, ::-1].copy()).permute(2, 0, 1).float()[None] / 255.
    t = ((t - MEAN) / STD).to(dev)
    hm = net(t)[0].float().cpu().numpy()
    Mi = cv2.invertAffineTransform(M)
    out = {}
    for ci, c in enumerate(CH):
        pk = _peaks(hm[ci], 4 if c == 'bolt' else 1)
        out[c] = [(*_apply(Mi, (px * STRIDE, py * STRIDE)), sc) for px, py, sc in pk]
    return out


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
    if not (600 <= abs(uh - ul) <= 1500):
        return 'D_range'
    return 'ok'


def depth_distance_mm(depth, pts_all, pa, pb, intrinsics=None):
    """검출점 볼록껍질 내부 depth로 평면 피팅 → 두 점의 평면상 거리(mm, K_DEPTH 보정)."""
    h, w = depth.shape
    K = intrinsics or dict(fx=1065.0, fy=1065.0, cx=w / 2.0, cy=h / 2.0)
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
    k = K_DEPTH.get(h, K_DEPTH[1080])
    return float(np.linalg.norm(to3d(pa) - to3d(pb))) * k


def classify(net, dev, rgb, depth=None):
    """단일 프레임 판정. depth 없으면 볼트 피치 스케일 사용."""
    det = detect(net, dev, rgb)
    hinge = det['corner_hinge'][0] if det['corner_hinge'] else None
    latch = det['corner_latch'][0] if det['corner_latch'] else None
    fr = bolt_frame(det['bolt'])
    gate = geometry_gate(fr, hinge, latch, rgb.shape)
    out = dict(points=det, gate=gate, pred=None, D_mm=None, D_src=None, group=None)
    if hinge is None or latch is None:
        return out
    D = None
    if depth is not None:
        D = depth_distance_mm(depth, det['bolt'] + [hinge, latch], hinge, latch); out['D_src'] = 'depth'
    if D is None and fr is not None:
        D = math.hypot(hinge[0] - latch[0], hinge[1] - latch[1]) / fr['s']; out['D_src'] = 'bolt'
    out['D_mm'] = D
    if D is not None and gate == 'ok':
        out['pred'] = min(CAD_D, key=lambda k: abs(CAD_D[k] - D))
        out['group'] = GROUP[out['pred']]
        out['margin_mm'] = float(sorted(abs(CAD_D[k] - D) for k in CAD_D)[1] - abs(CAD_D[out['pred']] - D))
    return out


def aggregate(results):
    """N프레임 집계: 판정된 프레임의 D 중앙값 → 클래스. 판정 프레임이 없으면 None."""
    Ds = [r['D_mm'] for r in results if r.get('pred')]
    if not Ds:
        return dict(pred=None, D_mm=None, n_judged=0, n=len(results))
    D = float(np.median(Ds)); pred = min(CAD_D, key=lambda k: abs(CAD_D[k] - D))
    return dict(pred=pred, group=GROUP[pred], D_mm=D, n_judged=len(Ds), n=len(results))
