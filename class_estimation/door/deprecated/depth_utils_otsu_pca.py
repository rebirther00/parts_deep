"""RGBD 공통 유틸리티 모듈

Depth 로드/저장, RGBD ResNet18 모델 생성, RGBD Dataset/Transform 등
모든 학습·평가·추론 스크립트에서 공유하는 기능을 모은다.
"""

import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models
import torchvision.transforms.functional as TF

# ── 상수 ──────────────────────────────────────────────────
MAX_DEPTH_MM = 5000       # 정규화 시 클리핑 거리 (5 m)
DEPTH_MEAN = 0.5
DEPTH_STD = 0.25
IN_CHANNELS = 4           # R, G, B, D
NUM_AUX_FEATURES = 4      # 보조 피처 수

# Isaac Sim 기본 USD 카메라 (focal=50mm, aperture=36mm, 1920×1080)
ISAAC_SIM_INTRINSICS = {
    "fx": 2666.67,
    "fy": 2666.67,
    "cx": 960.0,
    "cy": 540.0,
    "width": 1920,
    "height": 1080,
}


# ── Depth 로드 / 저장 ────────────────────────────────────

def load_depth_png(path, max_depth_mm=MAX_DEPTH_MM):
    """16-bit PNG depth → 정규화된 float32 [0, 1].

    무효값(0)은 0.0, max_depth_mm 이상은 1.0으로 클리핑.
    """
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    return np.clip(raw.astype(np.float32) / max_depth_mm, 0.0, 1.0)


def save_depth_png(depth_meters, path):
    """float32 depth (미터) → 16-bit PNG (밀리미터).

    NaN/Inf는 0으로 치환, [0, 65535] 범위로 클리핑.
    """
    depth_mm = np.nan_to_num(depth_meters * 1000.0, nan=0.0,
                             posinf=0.0, neginf=0.0)
    depth_uint16 = np.clip(depth_mm, 0, 65535).astype(np.uint16)
    cv2.imwrite(path, depth_uint16)


# ── 보조 피처 (물리 치수) ─────────────────────────────────

def _segment_foreground(depth_raw_mm):
    """depth 이미지에서 전경(부품) 영역을 분리.

    Otsu 이진화 → 형태학적 정리 → 최대 연결 영역 추출.
    Otsu가 실패하면 가장 가까운 60% 깊이값 기반으로 분리한다.
    """
    valid = depth_raw_mm > 0
    if valid.sum() < 100:
        return valid

    d_valid = depth_raw_mm[valid]
    d_min, d_max = float(d_valid.min()), float(d_valid.max())

    if d_max - d_min < 1.0:
        return valid

    d8 = np.zeros_like(depth_raw_mm, dtype=np.uint8)
    d8[valid] = np.clip(
        ((depth_raw_mm[valid] - d_min) / (d_max - d_min) * 255), 0, 255
    ).astype(np.uint8)

    otsu_t, _ = cv2.threshold(d8[valid].ravel(), 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    depth_thresh = d_min + (d_max - d_min) * otsu_t / 255.0
    fg = valid & (depth_raw_mm <= depth_thresh)

    fg_ratio = fg.sum() / valid.sum()
    if fg_ratio < 0.15 or fg_ratio > 0.85:
        pct60 = np.percentile(d_valid, 60)
        fg = valid & (depth_raw_mm <= pct60)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_u8 = fg.astype(np.uint8) * 255
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_CLOSE, kernel)
    fg_u8 = cv2.morphologyEx(fg_u8, cv2.MORPH_OPEN, kernel)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg_u8, connectivity=8)
    if n_labels <= 1:
        return fg_u8 > 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    return labels == largest


def compute_aux_features(depth_raw_mm, intrinsics=None):
    """원본 depth(mm)에서 부품의 물리 치수(mm)를 계산.

    전경 분리(Otsu + 형태학) 후 3D 점으로 변환, PCA로 주성분
    2축의 범위를 구하여 시점에 무관한 가로·세로 치수를 반환한다.

    Returns:
        [physical_width_mm, physical_height_mm, aspect_ratio, mean_depth_mm]
    """
    if intrinsics is None:
        intrinsics = ISAAC_SIM_INTRINSICS

    valid = depth_raw_mm > 0
    if valid.sum() < 10:
        return [0.0, 0.0, 1.0, 0.0]

    fg_mask = _segment_foreground(depth_raw_mm)

    if fg_mask.sum() < 10:
        return [0.0, 0.0, 1.0, 0.0]

    rows, cols = np.where(fg_mask)
    depths = depth_raw_mm[fg_mask].astype(np.float64)
    mean_depth = float(depths.mean())

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    X = (cols.astype(np.float64) - cx) * depths / fx
    Y = (rows.astype(np.float64) - cy) * depths / fy
    Z = depths

    pts = np.stack([X, Y, Z], axis=1)
    centered = pts - pts.mean(axis=0)

    try:
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return [0.0, 0.0, 1.0, mean_depth]

    proj = centered @ Vt[:2].T
    extent_0 = float(proj[:, 0].max() - proj[:, 0].min())
    extent_1 = float(proj[:, 1].max() - proj[:, 1].min())

    phys_w = max(extent_0, extent_1)
    phys_h = min(extent_0, extent_1)
    aspect = phys_w / max(phys_h, 1e-6)

    return [phys_w, phys_h, aspect, mean_depth]


# ── RGBD ResNet18 ────────────────────────────────────────

def create_rgbd_resnet18(num_classes, pretrained=True):
    """4-채널(RGBD) 입력 ResNet18.

    - RGB 3채널 ImageNet 가중치를 보존하고
    - D 채널은 RGB 평균으로 초기화하여 안정적인 학습 시작을 보장한다.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    old_conv = model.conv1
    new_conv = nn.Conv2d(IN_CHANNELS, 64,
                         kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
    model.conv1 = new_conv

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )
    return model


# ── RGBD + Aux ResNet18 ──────────────────────────────────

class RGBDAuxResNet18(nn.Module):
    """RGBD 이미지 + 물리 치수 보조 피처를 결합하는 분류 모델.

    backbone(ResNet18, 4ch→512dim) + Aux MLP(4→32dim) → 544dim → FC → num_classes
    """

    def __init__(self, num_classes, num_aux=NUM_AUX_FEATURES, pretrained=True):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        old_conv = backbone.conv1
        new_conv = nn.Conv2d(IN_CHANNELS, 64,
                             kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv

        self.backbone_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.aux_fc = nn.Sequential(
            nn.Linear(num_aux, 32),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone_features + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, images, aux_features):
        img_feat = self.backbone(images)
        aux_feat = self.aux_fc(aux_features)
        combined = torch.cat([img_feat, aux_feat], dim=1)
        return self.classifier(combined)


# ── RGBD Transform ───────────────────────────────────────

class RGBDTransform:
    """RGB PIL Image + Depth ndarray를 동기화 변환하여 [4, H, W] 텐서로 반환.

    Letterbox 방식: 장변 기준 Resize + 패딩으로 종횡비를 보존한다.
    - 공간 변환(Resize, Pad, Flip, Rotation, Scale)은 RGB·Depth에 동일 적용
    - 색상 변환(Brightness, Contrast, Saturation, Hue)은 RGB에만 적용
    - Gaussian Noise는 RGB에만 적용
    - Depth는 별도 정규화: (x - 0.5) / 0.25
    """

    def __init__(self, image_size, is_train=False):
        self.size = image_size
        self.is_train = is_train

    def __call__(self, rgb_pil, depth_np):
        """
        Args:
            rgb_pil: PIL.Image (RGB)
            depth_np: np.float32 [H, W], 값 범위 [0, 1]
        Returns:
            torch.Tensor [4, size, size]
        """
        w, h = rgb_pil.size

        # 1) Letterbox Resize — 장변 기준으로 축소하여 종횡비 보존
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        rgb_pil = rgb_pil.resize((new_w, new_h), Image.BILINEAR)
        depth_np = cv2.resize(depth_np, (new_w, new_h),
                              interpolation=cv2.INTER_NEAREST)

        # 2) 중앙 배치 패딩 — 검정(0) 패딩으로 종횡비 보존
        pad_left = (self.size - new_w) // 2
        pad_top = (self.size - new_h) // 2

        rgb_canvas = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        rgb_canvas.paste(rgb_pil, (pad_left, pad_top))
        rgb_pil = rgb_canvas

        depth_canvas = np.zeros((self.size, self.size), dtype=np.float32)
        depth_canvas[pad_top:pad_top + new_h,
                     pad_left:pad_left + new_w] = depth_np
        depth_np = depth_canvas

        # 3) 학습 시 Augmentation
        if self.is_train:
            # 수평 반전 (50% 확률)
            if random.random() < 0.5:
                rgb_pil = TF.hflip(rgb_pil)
                depth_np = np.flip(depth_np, axis=1).copy()

            # 회전 (±15°)
            angle = random.uniform(-15, 15)
            rgb_pil = TF.rotate(rgb_pil, angle)
            center = (self.size / 2, self.size / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            depth_np = cv2.warpAffine(depth_np, M, (self.size, self.size),
                                      flags=cv2.INTER_NEAREST, borderValue=0)

            # 스케일 변환 (90~110%, 카메라 거리 변화 시뮬레이션)
            if random.random() < 0.5:
                s = random.uniform(0.9, 1.1)
                Ms = cv2.getRotationMatrix2D(center, 0, s)
                rgb_np = np.array(rgb_pil)
                rgb_np = cv2.warpAffine(rgb_np, Ms, (self.size, self.size),
                                        flags=cv2.INTER_LINEAR, borderValue=0)
                rgb_pil = Image.fromarray(rgb_np)
                depth_np = cv2.warpAffine(depth_np, Ms, (self.size, self.size),
                                          flags=cv2.INTER_NEAREST,
                                          borderValue=0)

            # 색상 변환 (RGB에만 적용)
            rgb_pil = TF.adjust_brightness(rgb_pil, random.uniform(0.6, 1.4))
            rgb_pil = TF.adjust_contrast(rgb_pil, random.uniform(0.6, 1.4))
            rgb_pil = TF.adjust_saturation(rgb_pil, random.uniform(0.7, 1.3))
            rgb_pil = TF.adjust_hue(rgb_pil, random.uniform(-0.05, 0.05))

            # 가우시안 노이즈 (RGB에만, 30% 확률)
            if random.random() < 0.3:
                rgb_np = np.array(rgb_pil).astype(np.float32)
                noise = np.random.normal(0, random.uniform(3, 10),
                                         rgb_np.shape).astype(np.float32)
                rgb_np = np.clip(rgb_np + noise, 0, 255).astype(np.uint8)
                rgb_pil = Image.fromarray(rgb_np)

            # 가우시안 블러 (RGB에만, 20% 확률, 촬영 포커스 변화)
            if random.random() < 0.2:
                rgb_pil = TF.gaussian_blur(rgb_pil, kernel_size=3)

        # 4) To tensor + normalize
        rgb_t = TF.to_tensor(rgb_pil)
        rgb_t = TF.normalize(rgb_t,
                             [0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])

        depth_t = torch.from_numpy(depth_np.copy()).unsqueeze(0).float()
        depth_t = (depth_t - DEPTH_MEAN) / DEPTH_STD

        return torch.cat([rgb_t, depth_t], dim=0)


# ── RGBD Dataset ─────────────────────────────────────────

class RGBDDataset(Dataset):
    """RGB + Depth 쌍을 로딩하는 공통 Dataset.

    label 결정 방식:
      - labels 리스트를 직접 전달하거나
      - class_names를 전달하면 폴더명으로 자동 결정
    Depth 파일이 없으면 0으로 채움 (graceful degradation).

    반환: (rgbd_tensor, aux_tensor, label)
      - rgbd_tensor: [4, H, W]
      - aux_tensor:  [NUM_AUX_FEATURES] 물리 치수 보조 피처
      - label:       scalar
    """

    def __init__(self, image_paths, labels=None, transform=None,
                 class_names=None, intrinsics=None):
        self.image_paths = image_paths
        self.transform = transform
        self.intrinsics = intrinsics or ISAAC_SIM_INTRINSICS

        if labels is not None:
            self.labels = list(labels)
        elif class_names is not None:
            self.labels = []
            for p in image_paths:
                cls = os.path.basename(os.path.dirname(p))
                if cls not in class_names:
                    raise ValueError(f"알 수 없는 클래스: {cls}")
                self.labels.append(class_names.index(cls))
        else:
            raise ValueError("labels 또는 class_names 중 하나를 지정해야 합니다")

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def _depth_path(rgb_path):
        return rgb_path.replace("rgb_", "depth_")

    def __getitem__(self, idx):
        rgb_path = self.image_paths[idx]
        depth_path = self._depth_path(rgb_path)

        rgb = Image.open(rgb_path).convert("RGB")

        depth_raw_mm = None
        if os.path.exists(depth_path):
            raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if raw is not None:
                depth_raw_mm = raw.astype(np.float32)
                depth_norm = np.clip(depth_raw_mm / MAX_DEPTH_MM, 0.0, 1.0)
            else:
                depth_norm = np.zeros(
                    (rgb.height, rgb.width), dtype=np.float32)
        else:
            depth_norm = np.zeros(
                (rgb.height, rgb.width), dtype=np.float32)

        aux = compute_aux_features(
            depth_raw_mm if depth_raw_mm is not None
            else np.zeros((rgb.height, rgb.width), dtype=np.float32),
            self.intrinsics,
        )
        aux_tensor = torch.tensor(aux, dtype=torch.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            rgbd = self.transform(rgb, depth_norm)
            return rgbd, aux_tensor, label

        rgb_t = TF.to_tensor(rgb)
        depth_t = torch.from_numpy(depth_norm).unsqueeze(0).float()
        return torch.cat([rgb_t, depth_t], dim=0), aux_tensor, label
