"""Edge 전용 유틸리티 모듈

Canny Edge 검출 기반 분류 모델을 위한 공통 기능:
- EdgeAuxResNet18: Edge 3채널 + 보조 피처(물리 치수) 분류 모델
- EdgeTransform: RGB→Canny Edge 변환 + Letterbox Resize + 증강
- EdgeDataset: RGB 로드 → Edge 변환, Depth에서 Aux 피처 추출

Texture 의존을 완전히 제거하여 shape/구조 기반 분류를 수행한다.
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

from depth_utils import (
    compute_aux_features, DEFAULT_INTRINSICS, MAX_DEPTH_MM, NUM_AUX_FEATURES,
)

# ── 상수 ──────────────────────────────────────────────────
EDGE_IN_CHANNELS = 3    # Canny Edge를 3채널로 복제 (ImageNet pretrained 활용)
CANNY_LOW = 80
CANNY_HIGH = 200


# ── Edge + Aux ResNet18 ──────────────────────────────────

class EdgeAuxResNet18(nn.Module):
    """Canny Edge 3채널 이미지 + 물리 치수 보조 피처를 결합하는 분류 모델.

    backbone(ResNet18, 3ch→512dim) + Aux MLP(3→32dim) → 544dim → FC → num_classes
    Edge를 (E,E,E) 3채널로 복제하여 ImageNet pretrained 가중치를 그대로 활용한다.
    """

    def __init__(self, num_classes, num_aux=NUM_AUX_FEATURES, pretrained=True):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        self.backbone_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.aux_fc = nn.Sequential(
            nn.Linear(num_aux, 32),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone_features + 32, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, images, aux_features):
        img_feat = self.backbone(images)
        aux_feat = self.aux_fc(aux_features)
        combined = torch.cat([img_feat, aux_feat], dim=1)
        return self.classifier(combined)


# ── Edge Transform ───────────────────────────────────────

class EdgeTransform:
    """RGB PIL Image를 Canny Edge로 변환하여 [3, H, W] 텐서로 반환.

    Letterbox 방식: 장변 기준 Resize + 패딩으로 종횡비를 보존한다.
    - 공간 변환(Rotation, Scale)은 Edge 검출 전 RGB에 적용
    - 색상 변환은 없음 (Edge는 texture-free)
    - 학습 시 Canny threshold에 약간의 변동을 주어 일반화 향상
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

        # 3) 학습 시 공간 증강 (색상 증강은 Edge에 무의미하므로 제외)
        if self.is_train:
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

        # 4) Canny Edge 검출
        rgb_np = np.array(rgb_pil)
        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)

        if self.is_train:
            low = random.randint(60, 100)
            high = random.randint(160, 240)
        else:
            low, high = CANNY_LOW, CANNY_HIGH

        edges = cv2.Canny(gray, low, high)

        # 5) [0,1] 정규화 → 3채널 복제 → ImageNet normalize
        edges_float = edges.astype(np.float32) / 255.0
        edge_t = torch.from_numpy(np.stack([edges_float] * 3, axis=0))
        edge_t = TF.normalize(edge_t,
                              [0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225])

        return edge_t


# ── Edge Dataset ─────────────────────────────────────────

class EdgeDataset(Dataset):
    """RGB 이미지를 Canny Edge로 변환하여 로딩하는 Dataset.

    반환: (edge_tensor, aux_tensor, label)
      - edge_tensor: [3, H, W] Canny Edge (E,E,E)
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

        # Depth는 aux 피처 계산에만 사용 (모델 입력은 Edge)
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
            edge = self.transform(rgb)
            return edge, aux_tensor, label

        # transform 없이 기본 변환
        rgb_np = np.array(rgb)
        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
        edges_float = edges.astype(np.float32) / 255.0
        edge_t = torch.from_numpy(np.stack([edges_float] * 3, axis=0))
        return edge_t, aux_tensor, label
