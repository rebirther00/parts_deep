"""도어 속성 기반 분류 파이프라인 공용 모듈.

구조 (2단계):
  1) 그룹 판별: 렉티파이 → U-Net 통풍구 세그멘테이션 → CAD 템플릿 매칭
     (FRT: 통풍구 없음 / RH: 가장자리 루버 1밴드 / RR: 대형 밴드 2개)
  2) 차종 판별: 슬롯 패턴 점수 × 종횡비(PCA 실측) 결합
     score(c) = (템플릿IoU + eps) * exp(-|log(aspect/center_c)| / sigma)
     - FRT는 템플릿 점수가 전 클래스 ~0이라 자동으로 종횡비가 지배

실사용은 N프레임(기본 10) 집계: 그룹 다수결 + 종횡비 중앙값.
검증 결과 (N=10, 2026-07-07): datasets 100/97.2, datasets_aug2 99.9/97.5,
현장 그룹 100% (group%/class%). 상세: DOC_attribute_pipeline.md

신규 부품/리비전 추가: STL 1개 → 10_generate_cad_templates.py 재실행.
"""
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from dimension_utils import (
    preprocess_depth, refine_mask, measure_pca, ZED_INTRINSICS)

DOOR_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(DOOR_DIR, 'attribute_models')
SPEC_PATH = os.path.join(MODELS_DIR, 'class_spec.json')
TEMPLATES_PATH = os.path.join(MODELS_DIR, 'cad_templates.npz')
UNET_PATH = os.path.join(MODELS_DIR, 'vent_unet2.pth')

ORTHO_RES = 2.0          # 렉티파이 해상도 (mm/px)
MIN_VENT_COMP_CM2 = 3.0  # 통풍구 소형 성분 제거 임계


# ── 스펙/템플릿 로드 ─────────────────────────────────────

def load_spec(path=SPEC_PATH):
    return json.load(open(path, encoding='utf-8'))


def load_templates(path=TEMPLATES_PATH):
    return dict(np.load(path).items())


_SPEC = load_spec()
CLASSES = sorted(_SPEC['classes'])
GROUP = {c: _SPEC['classes'][c]['group'] for c in CLASSES}
ASPECT_CENTER = {c: _SPEC['classes'][c]['aspect_center'] for c in CLASSES}
EPS = _SPEC['decision']['eps']
SIGMA = _SPEC['decision']['sigma']


# ── U-Net (통풍구 세그멘테이션) ──────────────────────────

def _conv_block(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True),
        nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(True))


class VentUNet(nn.Module):
    """소형 U-Net. 입력 RGB [3,H,W] → 통풍구 로짓 [1,H,W]."""

    def __init__(self, ch=(32, 64, 128, 256)):
        super().__init__()
        self.enc = nn.ModuleList()
        ci = 3
        for co in ch:
            self.enc.append(_conv_block(ci, co))
            ci = co
        self.pool = nn.MaxPool2d(2)
        self.dec = nn.ModuleList()
        self.up = nn.ModuleList()
        for i in range(len(ch) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(ch[i], ch[i - 1], 2, 2))
            self.dec.append(_conv_block(ch[i - 1] * 2, ch[i - 1]))
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        feats = []
        for i, e in enumerate(self.enc):
            x = e(x)
            if i < len(self.enc) - 1:
                feats.append(x)
                x = self.pool(x)
        for up, dec, f in zip(self.up, self.dec, feats[::-1]):
            x = up(x)
            x = dec(torch.cat([x, f], 1))
        return self.head(x)


def load_vent_unet(path=UNET_PATH, device='cuda'):
    net = VentUNet().to(device).eval()
    net.load_state_dict(torch.load(path, map_location=device,
                                   weights_only=True))
    return net


# ── 기하 처리 ────────────────────────────────────────────

def rectify(rgb, depth_mm, mask, intrinsics=None, res=ORTHO_RES):
    """마스크 영역을 도어 평면 직교뷰(mm 고정 스케일)로 변환."""
    K = intrinsics or ZED_INTRINSICS
    m = (mask > 127) & (depth_mm > 0)
    rows, cols = np.where(m)
    z = depth_mm[m].astype(np.float64)
    X = (cols - K['cx']) * z / K['fx']
    Y = (rows - K['cy']) * z / K['fy']
    pts = np.stack([X, Y, z], axis=1)
    c = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - c, full_matrices=False)
    u, v = (pts - c) @ Vt[0], (pts - c) @ Vt[1]
    lo_u, hi_u = np.percentile(u, [0.5, 99.5])
    lo_v, hi_v = np.percentile(v, [0.5, 99.5])
    keep = (u >= lo_u) & (u <= hi_u) & (v >= lo_v) & (v <= hi_v)
    u, v, rows, cols = u[keep], v[keep], rows[keep], cols[keep]
    W = int((hi_u - lo_u) / res) + 1
    H = int((hi_v - lo_v) / res) + 1
    ui = ((u - lo_u) / res).astype(int)
    vi = ((v - lo_v) / res).astype(int)
    ortho = np.zeros((H, W, 3), np.uint8)
    omask = np.zeros((H, W), np.uint8)
    ortho[vi, ui] = rgb[rows, cols]
    omask[vi, ui] = 255
    kernel = np.ones((3, 3), np.uint8)
    omask = cv2.dilate(omask, kernel)
    ortho = cv2.morphologyEx(ortho, cv2.MORPH_CLOSE, kernel)
    return ortho, omask


def axis_align(ortho, omask):
    """도어를 minAreaRect 기준 축 정렬(가로≥세로)하고 bbox로 크롭."""
    cnts, _ = cv2.findContours(omask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
    if rw < rh:
        ang += 90
    M = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    H, W = omask.shape
    diag = int(np.hypot(H, W)) + 4
    M[0, 2] += diag / 2 - cx
    M[1, 2] += diag / 2 - cy
    ro = cv2.warpAffine(ortho, M, (diag, diag))
    rm = cv2.warpAffine(omask, M, (diag, diag), flags=cv2.INTER_NEAREST)
    ys, xs = np.where(rm > 0)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    return ro[y0:y1 + 1, x0:x1 + 1], rm[y0:y1 + 1, x0:x1 + 1]


def measure_aspect(mask, depth_mm, intrinsics=None):
    """PCA 평면 투영 실측 종횡비(최장변/차장변). 스케일 오차에 불변."""
    K = intrinsics or ZED_INTRINSICS
    m = measure_pca(refine_mask(mask),
                    preprocess_depth(depth_mm.astype(np.float32)), K)
    if m['width_mm'] <= 0 or m['height_mm'] <= 0:
        return None
    return m['width_mm'] / m['height_mm']


# ── 통풍구 예측 + 템플릿 매칭 ────────────────────────────

def _pad8(img):
    H, W = img.shape[:2]
    H2, W2 = (H + 7) // 8 * 8, (W + 7) // 8 * 8
    out = np.zeros((H2, W2) + img.shape[2:], img.dtype)
    out[:H, :W] = img
    return out, H, W


@torch.no_grad()
def predict_vent(net, img_bgr, device='cuda'):
    x, H, W = _pad8(img_bgr)
    x = torch.from_numpy(x.transpose(2, 0, 1))[None].float().to(device) / 255.
    p = torch.sigmoid(net(x))[0, 0].cpu().numpy()[:H, :W]
    return p


def clean_vent(prob, res=ORTHO_RES):
    """이진화 + 소형 성분(<MIN_VENT_COMP_CM2) 제거 — 스캐터 노이즈 억제."""
    pb = (prob > 0.5).astype(np.uint8)
    min_px = int(MIN_VENT_COMP_CM2 * 100 / (res * res))
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(pb, 8)
    out = np.zeros_like(pb)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_px:
            out[lbl == i] = 1
    return out


def match_templates(vent_bin, templates):
    """클래스별 슬롯 템플릿 IoU. 방향(가로/세로)을 맞춰 회전 후 리사이즈."""
    H, W = vent_bin.shape
    scores = {}
    for cls in CLASSES:
        slot = (templates[f'{cls}_slot'] > 0).astype(np.uint8)
        th, tw = slot.shape
        pr = vent_bin
        if (W > H) != (tw > th):
            pr = cv2.rotate(pr, cv2.ROTATE_90_CLOCKWISE)
        pr = cv2.resize(pr, (tw, th), interpolation=cv2.INTER_NEAREST)
        best = 0.0
        for fl in range(4):
            s = slot[:, ::-1] if fl & 1 else slot
            s = s[::-1] if fl & 2 else s
            inter = (pr & s).sum()
            union = (pr | s).sum()
            iou = inter / union if union else (1.0 if pr.sum() == 0 else 0.0)
            best = max(best, float(iou))
        scores[cls] = best
    return scores


# ── 프레임 처리 + N프레임 판정 ───────────────────────────

def frame_scores(rgb, depth_mm, mask, net, templates,
                 intrinsics=None, device='cuda'):
    """한 프레임 → {'scores': 클래스별 템플릿 IoU, 'asp': 실측 종횡비}"""
    ortho, omask = rectify(rgb, depth_mm, mask, intrinsics)
    ao, am = axis_align(ortho, omask)
    ao = cv2.medianBlur(ao, 3)
    p = predict_vent(net, ao, device)
    p[am == 0] = 0
    vent = clean_vent(p)
    return {'scores': match_templates(vent, templates),
            'asp': measure_aspect(mask, depth_mm, intrinsics)}


def decide(frames):
    """N프레임 집계 판정 → (클래스, 그룹, 클래스별 결합 점수).

    그룹: 프레임별 최고 템플릿 그룹의 다수결.
    차종: (평균 템플릿IoU + eps) * 종횡비 근접도, 그룹 내 최대.
    """
    mean_scores = {c: float(np.mean([f['scores'][c] for f in frames]))
                   for c in CLASSES}
    grps = [max(['FRT', 'RH', 'RR'],
                key=lambda g: max(f['scores'][c] for c in CLASSES
                                  if GROUP[c] == g)) for f in frames]
    grp = max(set(grps), key=grps.count)
    cand = [c for c in CLASSES if GROUP[c] == grp]
    asps = [f['asp'] for f in frames if f['asp']]
    if not asps:
        pred = max(cand, key=lambda c: mean_scores[c])
        return pred, grp, mean_scores
    med = float(np.median(asps))
    combined = {}
    for c in cand:
        a = float(np.exp(-abs(np.log(med) - np.log(ASPECT_CENTER[c]))
                         / SIGMA))
        combined[c] = (mean_scores[c] + EPS) * a
    pred = max(combined, key=combined.get)
    return pred, grp, combined
