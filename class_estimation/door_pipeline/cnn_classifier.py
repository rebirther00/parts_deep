"""RGBE NoAux ResNet18 도어 분류 CNN — 임포트용 추론 모듈 (홀 판별기 보류 시 폴백).

02_train.py 로 학습한 artifacts/<run>/model.pth + split_info.json(class_names) 을 읽어 프레임 1장을 분류한다.
모델 구조(NoAuxResNet18)는 02_train.py·05_realtime_inference.py 와 동일해야 한다 — 두 스크립트는 모듈 수준에서
argparse 를 실행해 임포트할 수 없으므로 여기 복제해 둔다(구조를 바꾸면 세 곳을 같이 고칠 것).

정식 폴백 모델(2026-09-07 채택, DB run #10): artifacts/rgbe_noaux_448_seed42_datasets_factory_v2/model.pth
  — 8/27 이후 현장 데이터 세션 단위 분할 학습, test 378장 97.4%, 5-fold 세션 CV 97.4%.
입력은 전체 프레임 레터박스(크롭 없음) 448×448 RGB + Canny 엣지 4채널.

사용:
    from cnn_classifier import CNNClassifier
    cnn = CNNClassifier()                       # 기본 경로, cuda 있으면 cuda
    pred, prob, probs = cnn.predict(bgr_frame)  # probs: [{'class', 'prob'}] 내림차순
"""
import json
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image as PILImage
from torchvision import models

from rgbe_utils import RGBETransform, RGBE_IN_CHANNELS

DOOR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUN = os.path.join(DOOR, 'artifacts', 'rgbe_noaux_448_seed42_datasets_factory_v2')
MODEL_PATH = os.path.join(DEFAULT_RUN, 'model.pth')
IMAGE_SIZE = 448


class NoAuxResNet18(nn.Module):
    """Aux MLP 없이 이미지만 사용하는 분류 모델 (02_train.py 와 동일 구조)."""

    def __init__(self, num_classes, in_channels=RGBE_IN_CHANNELS, pretrained=False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                for c in range(3, in_channels):
                    new_conv.weight[:, c:c + 1] = old_conv.weight.mean(dim=1, keepdim=True)
            backbone.conv1 = new_conv
        self.backbone_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(self.backbone_features, 256), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(256, num_classes))

    def forward(self, images, aux_features=None):
        return self.classifier(self.backbone(images))


def load_class_names(model_path, explicit_path=None):
    """--class_names JSON 또는 모델 폴더의 split_info.json['class_names']."""
    if explicit_path and os.path.exists(explicit_path):
        return json.load(open(explicit_path, encoding='utf-8'))
    sp = os.path.join(os.path.dirname(model_path), 'split_info.json')
    if os.path.exists(sp):
        info = json.load(open(sp, encoding='utf-8'))
        if 'class_names' in info:
            return info['class_names']
    raise FileNotFoundError(f'클래스명을 찾을 수 없습니다: {sp} (class_names 지정 필요)')


class CNNClassifier:
    def __init__(self, model_path=MODEL_PATH, class_names=None, device=None, image_size=IMAGE_SIZE):
        self.model_path = model_path
        self.class_names = class_names or load_class_names(model_path)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = NoAuxResNet18(len(self.class_names), in_channels=RGBE_IN_CHANNELS, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device).eval()
        self.transform = RGBETransform(image_size, is_train=False)
        if self.device.type == 'cuda':   # 워밍업 (첫 프레임 지연 제거)
            with torch.no_grad():
                self.model(torch.randn(1, RGBE_IN_CHANNELS, image_size, image_size, device=self.device))
            torch.cuda.synchronize()

    @torch.no_grad()
    def predict(self, bgr):
        """BGR 프레임 → (pred_class, prob, [{'class','prob'}] 내림차순)."""
        pil = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        inp = self.transform(pil).unsqueeze(0).to(self.device)
        probs = torch.softmax(self.model(inp).float(), dim=1)[0].cpu().numpy()
        order = np.argsort(-probs)
        ranked = [{'class': self.class_names[i], 'prob': float(probs[i])} for i in order]
        return ranked[0]['class'], ranked[0]['prob'], ranked
