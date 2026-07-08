"""RGB 전용 유틸리티 모듈

순수 RGB 3채널 입력 분류 모델을 위한 공통 기능:
- RGBTransform: RGB Letterbox Resize + 증강 + ImageNet normalize → [3, H, W]
- RGBDataset: RGB 로드, Depth에서 Aux 피처 추출 (API 호환)
- RGB_IN_CHANNELS = 3 (표준 ImageNet pretrained 가중치 그대로 사용)
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

RGB_IN_CHANNELS = 3


class RGBTransform:
    """RGB PIL Image를 [3, H, W] 텐서로 변환.

    Letterbox 방식: 장변 기준 Resize + 패딩으로 종횡비를 보존한다.
    RGBD/RGBE와 동일한 증강 파이프라인을 사용하되 depth/edge 채널 없음.
    """

    def __init__(self, image_size, is_train=False):
        self.size = image_size
        self.is_train = is_train

    def __call__(self, rgb_pil):
        w, h = rgb_pil.size

        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        rgb_pil = rgb_pil.resize((new_w, new_h), Image.BILINEAR)

        pad_left = (self.size - new_w) // 2
        pad_top = (self.size - new_h) // 2

        rgb_canvas = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        rgb_canvas.paste(rgb_pil, (pad_left, pad_top))
        rgb_pil = rgb_canvas

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

        rgb_t = TF.to_tensor(rgb_pil)
        rgb_t = TF.normalize(rgb_t,
                             [0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        return rgb_t


class RGBDataset(Dataset):
    """순수 RGB 3채널 이미지를 로딩하는 Dataset.

    반환: (rgb_tensor, aux_tensor, label)
      - rgb_tensor: [3, H, W]
      - aux_tensor: [NUM_AUX_FEATURES] (API 호환용, no_aux 모드에서 무시됨)
      - label: scalar
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
            rgb_t = self.transform(rgb)
            return rgb_t, aux_tensor, label

        rgb_t = TF.to_tensor(rgb)
        return rgb_t, aux_tensor, label
