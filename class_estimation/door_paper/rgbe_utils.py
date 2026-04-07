"""RGBE 공통 유틸리티 모듈

RGB + Canny Edge 4채널 입력 분류 모델을 위한 공통 기능:
- RGBETransform: RGB에 Canny Edge 채널을 추가하여 [4, H, W] 텐서 생성
- RGBEDataset: RGB 로드 → RGBE 변환, Depth에서 Aux 피처 추출

모델은 depth_utils.RGBDAuxResNet18을 그대로 재사용한다
(4채널 입력 구조가 동일, D→E만 다름).
"""

import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from depth_utils import (
    compute_aux_features, DEFAULT_INTRINSICS, MAX_DEPTH_MM, NUM_AUX_FEATURES,
    DEPTH_MEAN, DEPTH_STD,
)

# ── 상수 ──────────────────────────────────────────────────
RGBE_IN_CHANNELS = 4    # R, G, B, E
CANNY_LOW = 80
CANNY_HIGH = 200
EDGE_MEAN = 0.5
EDGE_STD = 0.25


# ── RGBE Transform ───────────────────────────────────────

class RGBETransform:
    """RGB PIL Image를 RGBE [4, H, W] 텐서로 변환.

    Letterbox 방식: 장변 기준 Resize + 패딩으로 종횡비를 보존한다.
    - 공간/색상 증강을 RGB에 적용한 후 Canny Edge를 계산
    - Edge는 증강된 RGB에서 생성되므로 별도 동기화 불필요
    - RGB: ImageNet normalize, Edge: (x - 0.5) / 0.25
    """

    def __init__(self, image_size, is_train=False):
        self.size = image_size
        self.is_train = is_train

    def __call__(self, rgb_pil):
        w, h = rgb_pil.size

        # 1) Letterbox Resize
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        rgb_pil = rgb_pil.resize((new_w, new_h), Image.BILINEAR)

        # 2) 중앙 배치 패딩
        pad_left = (self.size - new_w) // 2
        pad_top = (self.size - new_h) // 2

        rgb_canvas = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        rgb_canvas.paste(rgb_pil, (pad_left, pad_top))
        rgb_pil = rgb_canvas

        # 3) 학습 시 Augmentation (RGB에만, Edge는 이후 계산)
        if self.is_train:
            if random.random() < 0.5:
                rgb_pil = TF.hflip(rgb_pil)

            angle = random.uniform(-15, 15)
            rgb_pil = TF.rotate(rgb_pil, angle)

            if random.random() < 0.5:
                s = random.uniform(0.9, 1.1)
                center = (self.size / 2, self.size / 2)
                Ms = cv2.getRotationMatrix2D(center, 0, s)
                rgb_np = np.array(rgb_pil)
                rgb_np = cv2.warpAffine(rgb_np, Ms, (self.size, self.size),
                                        flags=cv2.INTER_LINEAR, borderValue=0)
                rgb_pil = Image.fromarray(rgb_np)

            rgb_pil = TF.adjust_brightness(rgb_pil, random.uniform(0.6, 1.4))
            rgb_pil = TF.adjust_contrast(rgb_pil, random.uniform(0.6, 1.4))
            rgb_pil = TF.adjust_saturation(rgb_pil, random.uniform(0.7, 1.3))
            rgb_pil = TF.adjust_hue(rgb_pil, random.uniform(-0.05, 0.05))

            if random.random() < 0.3:
                rgb_np = np.array(rgb_pil).astype(np.float32)
                noise = np.random.normal(0, random.uniform(3, 10),
                                         rgb_np.shape).astype(np.float32)
                rgb_np = np.clip(rgb_np + noise, 0, 255).astype(np.uint8)
                rgb_pil = Image.fromarray(rgb_np)

            if random.random() < 0.2:
                rgb_pil = TF.gaussian_blur(rgb_pil, kernel_size=3)

        # 4) Canny Edge 계산 (증강 후 RGB에서)
        rgb_np = np.array(rgb_pil)
        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)

        if self.is_train:
            low = random.randint(60, 100)
            high = random.randint(160, 240)
        else:
            low, high = CANNY_LOW, CANNY_HIGH

        edges = cv2.Canny(gray, low, high)
        edge_float = edges.astype(np.float32) / 255.0

        # 5) To tensor + normalize
        rgb_t = TF.to_tensor(rgb_pil)
        rgb_t = TF.normalize(rgb_t,
                             [0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])

        edge_t = torch.from_numpy(edge_float).unsqueeze(0).float()
        edge_t = (edge_t - EDGE_MEAN) / EDGE_STD

        return torch.cat([rgb_t, edge_t], dim=0)


# ── RGBE Dataset ─────────────────────────────────────────

class RGBEDataset(Dataset):
    """RGB를 RGBE(4채널)로 변환하여 로딩하는 Dataset.

    반환: (rgbe_tensor, aux_tensor, label)
      - rgbe_tensor: [4, H, W] (R, G, B, E)
      - aux_tensor:  [NUM_AUX_FEATURES] 물리 치수 보조 피처 (Depth에서 계산)
      - label:       scalar
    """

    def __init__(self, image_paths, labels=None, transform=None,
                 class_names=None, intrinsics=None):
        self.image_paths = image_paths
        self.transform = transform
        self.intrinsics = intrinsics or DEFAULT_INTRINSICS

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

    def __getitem__(self, idx):
        rgb_path = self.image_paths[idx]
        depth_path = rgb_path.replace("rgb_", "depth_")
        mask_path = rgb_path.replace("rgb_", "mask_")

        rgb = Image.open(rgb_path).convert("RGB")

        # Depth는 aux 피처 계산에만 사용 (모델 입력은 RGBE)
        depth_raw_mm = None
        if os.path.exists(depth_path):
            raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if raw is not None:
                depth_raw_mm = raw.astype(np.float32)

        fg_mask = None
        if os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                fg_mask = m

        aux = compute_aux_features(
            depth_raw_mm if depth_raw_mm is not None
            else np.zeros((rgb.height, rgb.width), dtype=np.float32),
            self.intrinsics,
            fg_mask=fg_mask,
        )
        aux_tensor = torch.tensor(aux, dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            rgbe = self.transform(rgb)
            return rgbe, aux_tensor, label

        # transform 없이 기본 변환
        rgb_np = np.array(rgb)
        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
        edge_float = edges.astype(np.float32) / 255.0

        rgb_t = TF.to_tensor(rgb)
        edge_t = torch.from_numpy(edge_float).unsqueeze(0).float()
        return torch.cat([rgb_t, edge_t], dim=0), aux_tensor, label
