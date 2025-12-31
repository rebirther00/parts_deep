#!/usr/bin/env python3
"""
DeepIM Refiner Network
- 실제 이미지와 렌더링된 이미지를 비교하여 포즈 잔차 예측
- 8채널 입력: Real RGB(3) + Rendered RGB(3) + Real Depth(1) + Rendered Depth(1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights


class DeepIMRefiner(nn.Module):
    """DeepIM Pose Refiner Network
    
    입력: 8채널 (Real RGB 3 + Rendered RGB 3 + Real Depth 1 + Rendered Depth 1)
    출력: Pose Residual (ΔPosition 3 + ΔRotation 6D)
    """
    
    def __init__(self, in_channels=8, position_scale=0.1, rotation_scale=0.1):
        """
        Args:
            in_channels: 입력 채널 수 (기본 8)
            position_scale: 위치 residual 스케일 (학습 안정화)
            rotation_scale: 회전 residual 스케일
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        
        # 첫 Conv 레이어 (8채널 → 64채널)
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        # ResNet18 백본 (첫 conv 제외)
        resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.layer1 = resnet.layer1  # 64 → 64
        self.layer2 = resnet.layer2  # 64 → 128
        self.layer3 = resnet.layer3  # 128 → 256
        self.layer4 = resnet.layer4  # 256 → 512
        self.avgpool = resnet.avgpool
        
        # Feature dimension
        feature_dim = 512
        
        # Position Residual Head (ΔT: dx, dy, dz)
        self.position_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )
        
        # Rotation Residual Head (Δ6D rotation)
        self.rotation_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6),
        )
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화 - residual이 작게 시작하도록"""
        # Position head 마지막 레이어를 작은 값으로 초기화
        nn.init.zeros_(self.position_head[-1].bias)
        nn.init.normal_(self.position_head[-1].weight, std=0.01)
        
        # Rotation head도 마찬가지
        nn.init.zeros_(self.rotation_head[-1].bias)
        nn.init.normal_(self.rotation_head[-1].weight, std=0.01)
    
    def forward(self, real_rgb, rendered_rgb, real_depth, rendered_depth):
        """
        Args:
            real_rgb: (N, 3, H, W) 실제 RGB 이미지
            rendered_rgb: (N, 3, H, W) 렌더링된 RGB 이미지
            real_depth: (N, 1, H, W) 실제 Depth 이미지
            rendered_depth: (N, 1, H, W) 렌더링된 Depth 이미지
        
        Returns:
            delta_position: (N, 3) 위치 잔차 [dx, dy, dz]
            delta_rotation: (N, 6) 회전 잔차 (6D representation)
        """
        # 입력 concat (8채널)
        x = torch.cat([real_rgb, rendered_rgb, real_depth, rendered_depth], dim=1)
        
        # Feature extraction
        x = self.input_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Residual 예측
        delta_position = self.position_head(x) * self.position_scale
        delta_rotation = self.rotation_head(x) * self.rotation_scale
        
        return delta_position, delta_rotation


class DeepIMRefinerLight(nn.Module):
    """경량화된 DeepIM Refiner (더 빠른 추론)"""
    
    def __init__(self, in_channels=8, position_scale=0.1, rotation_scale=0.1):
        super().__init__()
        
        self.position_scale = position_scale
        self.rotation_scale = rotation_scale
        
        # 경량 CNN 백본
        self.features = nn.Sequential(
            # Block 1: 8 → 32
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 2: 32 → 64
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Block 3: 64 → 128
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Block 4: 128 → 256
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # Block 5: 256 → 256
            nn.Conv2d(256, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d(1),
        )
        
        feature_dim = 256
        
        # Position Head
        self.position_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3),
        )
        
        # Rotation Head
        self.rotation_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 6),
        )
    
    def forward(self, real_rgb, rendered_rgb, real_depth, rendered_depth):
        x = torch.cat([real_rgb, rendered_rgb, real_depth, rendered_depth], dim=1)
        x = self.features(x)
        x = torch.flatten(x, 1)
        
        delta_position = self.position_head(x) * self.position_scale
        delta_rotation = self.rotation_head(x) * self.rotation_scale
        
        return delta_position, delta_rotation


# 테스트 코드
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 모델 생성
    refiner = DeepIMRefiner().to(device)
    print(f"DeepIMRefiner 파라미터: {sum(p.numel() for p in refiner.parameters()):,}")
    
    # 테스트 입력
    batch_size = 4
    real_rgb = torch.randn(batch_size, 3, 224, 224).to(device)
    rendered_rgb = torch.randn(batch_size, 3, 224, 224).to(device)
    real_depth = torch.randn(batch_size, 1, 224, 224).to(device)
    rendered_depth = torch.randn(batch_size, 1, 224, 224).to(device)
    
    # Forward
    delta_pos, delta_rot = refiner(real_rgb, rendered_rgb, real_depth, rendered_depth)
    print(f"Delta Position: {delta_pos.shape}")
    print(f"Delta Rotation: {delta_rot.shape}")
    
    # 경량 버전 테스트
    refiner_light = DeepIMRefinerLight().to(device)
    print(f"DeepIMRefinerLight 파라미터: {sum(p.numel() for p in refiner_light.parameters()):,}")

