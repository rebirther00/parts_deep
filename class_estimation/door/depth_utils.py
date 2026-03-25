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


# ── RGBD Transform ───────────────────────────────────────

class RGBDTransform:
    """RGB PIL Image + Depth ndarray를 동기화 변환하여 [4, H, W] 텐서로 반환.

    - 공간 변환(Resize, CenterCrop, Rotation)은 RGB·Depth에 동일 적용
    - 색상 변환(Brightness, Contrast, Saturation)은 RGB에만 적용
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

        # 1) Resize — shorter edge → self.size
        if w <= h:
            new_w, new_h = self.size, int(h * self.size / w)
        else:
            new_w, new_h = int(w * self.size / h), self.size

        rgb_pil = rgb_pil.resize((new_w, new_h), Image.BILINEAR)
        depth_np = cv2.resize(depth_np, (new_w, new_h),
                              interpolation=cv2.INTER_NEAREST)

        # 2) CenterCrop
        left = (new_w - self.size) // 2
        top = (new_h - self.size) // 2
        rgb_pil = rgb_pil.crop((left, top,
                                left + self.size, top + self.size))
        depth_np = depth_np[top:top + self.size, left:left + self.size]

        # 3) 학습 시 Augmentation
        if self.is_train:
            angle = random.uniform(-5, 5)
            rgb_pil = TF.rotate(rgb_pil, angle)
            center = (self.size / 2, self.size / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            depth_np = cv2.warpAffine(depth_np, M, (self.size, self.size),
                                      flags=cv2.INTER_NEAREST, borderValue=0)

            rgb_pil = TF.adjust_brightness(rgb_pil, random.uniform(0.7, 1.3))
            rgb_pil = TF.adjust_contrast(rgb_pil, random.uniform(0.7, 1.3))
            rgb_pil = TF.adjust_saturation(rgb_pil, random.uniform(0.8, 1.2))

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
    """

    def __init__(self, image_paths, labels=None, transform=None,
                 class_names=None):
        self.image_paths = image_paths
        self.transform = transform

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

        if os.path.exists(depth_path):
            raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if raw is not None:
                depth_norm = np.clip(
                    raw.astype(np.float32) / MAX_DEPTH_MM, 0.0, 1.0)
            else:
                depth_norm = np.zeros(
                    (rgb.height, rgb.width), dtype=np.float32)
        else:
            depth_norm = np.zeros(
                (rgb.height, rgb.width), dtype=np.float32)

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            rgbd = self.transform(rgb, depth_norm)
            return rgbd, label

        rgb_t = TF.to_tensor(rgb)
        depth_t = torch.from_numpy(depth_norm).unsqueeze(0).float()
        return torch.cat([rgb_t, depth_t], dim=0), label
