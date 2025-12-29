#!/usr/bin/env python3
# ==========================================
# 딥러닝 기반 6DoF Pose Estimation (DenseFusion 방식)
# RGB + Depth → 6DoF Pose (위치 + 자세)
# ==========================================
#
# 아키텍처:
# 1. RGB → CNN Encoder → RGB Feature
# 2. Depth → PointNet-like Encoder → Depth Feature
# 3. Feature Fusion → 6DoF Pose Regression
#
# 출력:
# - 위치: (x, y, z) in meters
# - 회전: 6D rotation representation → quaternion
#
# 필요 라이브러리:
# pip install torch torchvision numpy pillow tqdm
# ==========================================

import os
import sys
import json
import glob
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ==========================================
# 설정
# ==========================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset_pos_depth")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")

# 학습 설정
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
TRAIN_RATIO = 0.8

# 카메라 내재 파라미터
CAMERA_INTRINSICS = {
    "fx": 768.0, "fy": 768.0,
    "cx": 512.0, "cy": 512.0,
    "width": 1024, "height": 1024
}

# ==========================================
# 6D Rotation Representation
# ==========================================
def rotation_6d_to_matrix(r6d):
    """6D rotation representation → 3x3 rotation matrix
    
    Reference: "On the Continuity of Rotation Representations in Neural Networks"
    https://arxiv.org/abs/1812.07035
    """
    # r6d: (batch, 6) → (batch, 3, 3)
    a1 = r6d[:, :3]  # (batch, 3)
    a2 = r6d[:, 3:6]  # (batch, 3)
    
    # Gram-Schmidt orthogonalization
    b1 = F.normalize(a1, dim=1)
    b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    
    # Stack to form rotation matrix
    return torch.stack([b1, b2, b3], dim=2)  # (batch, 3, 3)


def matrix_to_rotation_6d(matrix):
    """3x3 rotation matrix → 6D rotation representation"""
    # matrix: (batch, 3, 3) → (batch, 6)
    return matrix[:, :, :2].reshape(-1, 6)


def euler_to_rotation_matrix(roll, pitch, yaw):
    """Euler angles (degrees) → 3x3 rotation matrix"""
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    
    # Roll (X-axis rotation)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Pitch (Y-axis rotation)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Yaw (Z-axis rotation)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Ry @ Rx


def rotation_matrix_to_euler(R):
    """3x3 rotation matrix → Euler angles (degrees)"""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


# ==========================================
# 데이터셋
# ==========================================
class RGBDPoseDataset(Dataset):
    """RGB + Depth → 6DoF Pose 데이터셋"""
    
    def __init__(self, dataset_dir, split='train', train_ratio=TRAIN_RATIO):
        self.dataset_dir = dataset_dir
        self.split = split
        self.samples = []
        
        # 클래스별 폴더 스캔
        class_dirs = sorted(glob.glob(os.path.join(dataset_dir, "*")))
        class_dirs = [d for d in class_dirs if os.path.isdir(d) and not d.endswith('__pycache__')]
        
        self.class_names = [os.path.basename(d) for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # 모든 샘플 수집
        all_samples = []
        for class_dir in class_dirs:
            class_name = os.path.basename(class_dir)
            pose_files = sorted(glob.glob(os.path.join(class_dir, "pose_*.json")))
            
            for pose_file in pose_files:
                frame_idx = int(os.path.basename(pose_file).split('_')[-1].split('.')[0])
                rgb_file = os.path.join(class_dir, f"rgb_{frame_idx:04d}.png")
                depth_file = os.path.join(class_dir, f"distance_to_camera_{frame_idx:04d}.npy")
                
                if os.path.exists(rgb_file) and os.path.exists(depth_file):
                    all_samples.append({
                        'rgb_path': rgb_file,
                        'depth_path': depth_file,
                        'pose_path': pose_file,
                        'class_name': class_name,
                        'class_idx': self.class_to_idx[class_name]
                    })
        
        # Train/Test 분할
        random.seed(42)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        
        if split == 'train':
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
        
        # RGB 변환
        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"[{split}] 샘플 수: {len(self.samples)}, 클래스 수: {len(self.class_names)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # RGB 로드
        rgb = Image.open(sample['rgb_path']).convert('RGB')
        rgb = self.rgb_transform(rgb)
        
        # Depth 로드 및 전처리
        depth = np.load(sample['depth_path'])
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        
        # Depth 정규화 및 리사이즈
        depth = np.clip(depth, 0.1, 10.0)  # 유효 범위 제한
        depth = (depth - 0.1) / (10.0 - 0.1)  # 0~1 정규화
        
        # 리사이즈 (PIL 사용)
        depth_pil = Image.fromarray((depth * 255).astype(np.uint8))
        depth_pil = depth_pil.resize((224, 224), Image.BILINEAR)
        depth = np.array(depth_pil).astype(np.float32) / 255.0
        depth = torch.from_numpy(depth).unsqueeze(0)  # (1, 224, 224)
        
        # Pose 로드
        with open(sample['pose_path'], 'r') as f:
            pose_data = json.load(f)
        
        # 위치 (카메라 기준)
        t = pose_data['camTobj']['t_xyz_m']
        position = torch.tensor(t, dtype=torch.float32)
        
        # 회전 (카메라 기준) → 6D representation
        r = pose_data['camTobj']['r_xyz_deg']
        R_matrix = euler_to_rotation_matrix(r[0], r[1], r[2])
        r6d = R_matrix[:, :2].flatten()  # 6D representation
        rotation = torch.tensor(r6d, dtype=torch.float32)
        
        # 클래스 인덱스
        class_idx = sample['class_idx']
        
        return {
            'rgb': rgb,
            'depth': depth,
            'position': position,  # (3,)
            'rotation': rotation,  # (6,)
            'class_idx': class_idx
        }


# ==========================================
# 모델 아키텍처
# ==========================================
class DepthEncoder(nn.Module):
    """Depth 인코더 (간단한 CNN)"""
    
    def __init__(self, out_features=256):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fc = nn.Linear(256, out_features)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Deep6DoFModel(nn.Module):
    """RGB + Depth → 6DoF Pose 모델
    
    DenseFusion 스타일의 Feature Fusion
    """
    
    def __init__(self, num_classes=4, rgb_features=512, depth_features=256):
        super().__init__()
        
        # RGB Encoder (ResNet18)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.rgb_encoder = nn.Sequential(*list(resnet.children())[:-1])  # 마지막 FC 제외
        self.rgb_fc = nn.Linear(512, rgb_features)
        
        # Depth Encoder
        self.depth_encoder = DepthEncoder(out_features=depth_features)
        
        # Fusion + Pose Head
        fusion_features = rgb_features + depth_features
        
        self.pose_head = nn.Sequential(
            nn.Linear(fusion_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # 위치 출력 (x, y, z)
        self.position_head = nn.Linear(256, 3)
        
        # 회전 출력 (6D representation)
        self.rotation_head = nn.Linear(256, 6)
        
        # 분류 출력 (optional)
        self.class_head = nn.Linear(256, num_classes)
    
    def forward(self, rgb, depth):
        # RGB Encoding
        rgb_feat = self.rgb_encoder(rgb)
        rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)
        rgb_feat = self.rgb_fc(rgb_feat)
        
        # Depth Encoding
        depth_feat = self.depth_encoder(depth)
        
        # Feature Fusion (concatenation)
        fused = torch.cat([rgb_feat, depth_feat], dim=1)
        
        # Pose prediction
        pose_feat = self.pose_head(fused)
        
        position = self.position_head(pose_feat)
        rotation = self.rotation_head(pose_feat)
        class_logits = self.class_head(pose_feat)
        
        return {
            'position': position,  # (batch, 3)
            'rotation': rotation,  # (batch, 6)
            'class_logits': class_logits  # (batch, num_classes)
        }


# ==========================================
# 손실 함수
# ==========================================
class PoseLoss(nn.Module):
    """6DoF Pose 손실 함수"""
    
    def __init__(self, position_weight=1.0, rotation_weight=1.0, class_weight=0.1):
        super().__init__()
        self.position_weight = position_weight
        self.rotation_weight = rotation_weight
        self.class_weight = class_weight
        
        self.position_loss = nn.SmoothL1Loss()
        self.rotation_loss = nn.SmoothL1Loss()
        self.class_loss = nn.CrossEntropyLoss()
    
    def forward(self, pred, target_position, target_rotation, target_class):
        # 위치 손실
        pos_loss = self.position_loss(pred['position'], target_position)
        
        # 회전 손실 (6D representation)
        rot_loss = self.rotation_loss(pred['rotation'], target_rotation)
        
        # 분류 손실
        cls_loss = self.class_loss(pred['class_logits'], target_class)
        
        total_loss = (
            self.position_weight * pos_loss +
            self.rotation_weight * rot_loss +
            self.class_weight * cls_loss
        )
        
        return {
            'total': total_loss,
            'position': pos_loss,
            'rotation': rot_loss,
            'class': cls_loss
        }


# ==========================================
# 평가 메트릭
# ==========================================
def compute_metrics(pred_pos, target_pos, pred_rot, target_rot, pred_class, target_class):
    """평가 메트릭 계산"""
    # 위치 오차 (mm)
    pos_error = torch.sqrt(((pred_pos - target_pos) ** 2).sum(dim=1)).mean() * 1000
    
    # 회전 오차 (6D → matrix → 각도 차이)
    # 간단히: 6D representation의 L2 거리
    rot_error = torch.sqrt(((pred_rot - target_rot) ** 2).sum(dim=1)).mean()
    
    # 분류 정확도
    _, pred_labels = torch.max(pred_class, 1)
    class_acc = (pred_labels == target_class).float().mean() * 100
    
    return {
        'position_error_mm': pos_error.item(),
        'rotation_error': rot_error.item(),
        'class_accuracy': class_acc.item()
    }


# ==========================================
# 학습 루프
# ==========================================
def train_model(dataset_dir=DATASET_DIR):
    """모델 학습"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 데이터셋 로드
    train_dataset = RGBDPoseDataset(dataset_dir, split='train')
    test_dataset = RGBDPoseDataset(dataset_dir, split='test')
    
    if len(train_dataset) == 0:
        print(f"⚠️  데이터셋이 없습니다: {dataset_dir}")
        print("먼저 01_generate_mult_class_dataset_with_pos.py를 실행하여 데이터셋을 생성하세요.")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 모델 생성
    num_classes = len(train_dataset.class_names)
    model = Deep6DoFModel(num_classes=num_classes).to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = PoseLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # 결과 저장
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    best_pos_error = float('inf')
    
    print(f"\n{'='*70}")
    print("딥러닝 6DoF Pose Estimation 학습")
    print(f"{'='*70}")
    print(f"Train 샘플: {len(train_dataset)}, Test 샘플: {len(test_dataset)}")
    print(f"클래스: {train_dataset.class_names}")
    print(f"Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print()
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_losses = {'total': 0, 'position': 0, 'rotation': 0, 'class': 0}
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1:02d} Train", leave=False):
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            position = batch['position'].to(device)
            rotation = batch['rotation'].to(device)
            class_idx = batch['class_idx'].to(device)
            
            optimizer.zero_grad()
            pred = model(rgb, depth)
            losses = criterion(pred, position, rotation, class_idx)
            losses['total'].backward()
            optimizer.step()
            
            for k in train_losses:
                train_losses[k] += losses[k].item()
        
        for k in train_losses:
            train_losses[k] /= len(train_loader)
        
        # Evaluation
        model.eval()
        test_metrics = {'position_error_mm': 0, 'rotation_error': 0, 'class_accuracy': 0}
        
        with torch.no_grad():
            for batch in test_loader:
                rgb = batch['rgb'].to(device)
                depth = batch['depth'].to(device)
                position = batch['position'].to(device)
                rotation = batch['rotation'].to(device)
                class_idx = batch['class_idx'].to(device)
                
                pred = model(rgb, depth)
                metrics = compute_metrics(
                    pred['position'], position,
                    pred['rotation'], rotation,
                    pred['class_logits'], class_idx
                )
                
                for k in test_metrics:
                    test_metrics[k] += metrics[k]
        
        for k in test_metrics:
            test_metrics[k] /= len(test_loader)
        
        scheduler.step()
        
        # Best 모델 저장
        if test_metrics['position_error_mm'] < best_pos_error:
            best_pos_error = test_metrics['position_error_mm']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'class_names': train_dataset.class_names,
                'best_pos_error': best_pos_error
            }, os.path.join(ARTIFACTS_DIR, 'deep_6dof_best.pt'))
        
        # 로그 출력 (5 에폭마다)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
                  f"Loss={train_losses['total']:.4f} | "
                  f"PosErr={test_metrics['position_error_mm']:.1f}mm | "
                  f"RotErr={test_metrics['rotation_error']:.4f} | "
                  f"ClassAcc={test_metrics['class_accuracy']:.1f}% | "
                  f"best={best_pos_error:.1f}mm")
    
    print(f"\n{'='*70}")
    print(f"학습 완료! 최고 위치 오차: {best_pos_error:.2f}mm")
    print(f"모델 저장: {os.path.join(ARTIFACTS_DIR, 'deep_6dof_best.pt')}")


def evaluate_model(dataset_dir=DATASET_DIR):
    """모델 평가"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 데이터셋 로드
    test_dataset = RGBDPoseDataset(dataset_dir, split='test')
    if len(test_dataset) == 0:
        print("테스트 데이터셋이 없습니다.")
        return
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # 모델 로드
    model_path = os.path.join(ARTIFACTS_DIR, 'deep_6dof_best.pt')
    if not os.path.exists(model_path):
        print(f"모델 파일 없음: {model_path}")
        return
    
    num_classes = len(test_dataset.class_names)
    model = Deep6DoFModel(num_classes=num_classes).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\n{'='*70}")
    print("딥러닝 6DoF Pose Estimation 평가")
    print(f"{'='*70}")
    print(f"모델: {model_path}")
    print(f"테스트 샘플: {len(test_dataset)}")
    
    all_pos_errors = []
    all_rot_errors = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            position = batch['position'].to(device)
            rotation = batch['rotation'].to(device)
            class_idx = batch['class_idx'].to(device)
            
            pred = model(rgb, depth)
            
            # 위치 오차 (mm)
            pos_error = torch.sqrt(((pred['position'] - position) ** 2).sum(dim=1)) * 1000
            all_pos_errors.extend(pos_error.cpu().numpy())
            
            # 회전 오차
            rot_error = torch.sqrt(((pred['rotation'] - rotation) ** 2).sum(dim=1))
            all_rot_errors.extend(rot_error.cpu().numpy())
            
            # 분류 정확도
            _, pred_labels = torch.max(pred['class_logits'], 1)
            correct += (pred_labels == class_idx).sum().item()
            total += class_idx.size(0)
    
    # 결과 출력
    print(f"\n결과:")
    print(f"  평균 위치 오차: {np.mean(all_pos_errors):.2f}mm (std={np.std(all_pos_errors):.2f})")
    print(f"  평균 회전 오차: {np.mean(all_rot_errors):.4f}")
    print(f"  분류 정확도: {100 * correct / total:.2f}%")
    print(f"\n  위치 오차 분포:")
    print(f"    < 5mm: {100 * sum(1 for e in all_pos_errors if e < 5) / len(all_pos_errors):.1f}%")
    print(f"    < 10mm: {100 * sum(1 for e in all_pos_errors if e < 10) / len(all_pos_errors):.1f}%")
    print(f"    < 20mm: {100 * sum(1 for e in all_pos_errors if e < 20) / len(all_pos_errors):.1f}%")


# ==========================================
# 메인 실행
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="딥러닝 6DoF Pose Estimation")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'],
                        help='실행 모드: train(학습), eval(평가)')
    parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR,
                        help='데이터셋 경로')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='배치 크기')
    
    args = parser.parse_args()
    
    # 전역 설정 업데이트
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    
    if args.mode == 'train':
        train_model(args.dataset_dir)
    elif args.mode == 'eval':
        evaluate_model(args.dataset_dir)

