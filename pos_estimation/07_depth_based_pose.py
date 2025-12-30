#!/usr/bin/env python3
# ==========================================
# Depth 기반 6DoF Pose Estimation
# Ground Truth를 Depth에서 직접 계산
# ==========================================
#
# 핵심 아이디어:
# 1. Depth 이미지 → Point Cloud 변환
# 2. 전경(물체) 영역 분리 (배경보다 가까운 영역)
# 3. 물체 Point Cloud의 중심(centroid) = Ground Truth 위치
# 4. 모델: RGB → 3D 위치 예측
#
# 이 방식의 장점:
# - 좌표계 변환 오류 없음 (Depth에서 직접 계산)
# - Ground Truth가 실제 측정값
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
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.8

# 카메라 내재 파라미터
CAMERA_INTRINSICS = {
    "fx": 768.0, "fy": 768.0,
    "cx": 512.0, "cy": 512.0,
    "width": 1024, "height": 1024
}

# Depth 기반 GT 설정
DEPTH_MIN = 0.01  # 최소 유효 Depth (m) - 작은 스케일 허용
DEPTH_MAX = 100.0 # 최대 유효 Depth (m) - 큰 범위
FOREGROUND_PERCENTILE = 10  # 전경 결정 백분위수 (가장 가까운 10%)

# ==========================================
# Depth → Point Cloud → 객체 중심 계산
# ==========================================
def depth_to_pointcloud(depth, fx, fy, cx, cy):
    """Depth 이미지 → Point Cloud 변환
    
    Args:
        depth: (H, W) depth 이미지 (meters)
        fx, fy: focal lengths
        cx, cy: principal point
    
    Returns:
        points: (N, 3) point cloud [X, Y, Z]
        valid_mask: (H, W) 유효한 픽셀 마스크
    """
    h, w = depth.shape
    
    # 유효한 Depth 마스크
    valid_mask = (depth > DEPTH_MIN) & (depth < DEPTH_MAX) & np.isfinite(depth)
    
    # 픽셀 좌표 생성
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)
    
    # 3D 좌표 계산 (카메라 좌표계)
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # 유효한 점만 추출
    points = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=1)
    
    return points, valid_mask


def get_meters_per_unit(class_dir):
    """클래스 디렉토리에서 meters_per_unit 읽기"""
    metadata_path = os.path.join(class_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        return metadata.get("meters_per_unit", 1.0)
    
    # pose 파일에서 읽기 시도
    pose_files = glob.glob(os.path.join(class_dir, "pose_*.json"))
    if pose_files:
        with open(pose_files[0]) as f:
            pose = json.load(f)
        return pose.get("stage_info", {}).get("meters_per_unit", 1.0)
    
    return 1.0


def compute_object_centroid_from_depth(depth, fx, fy, cx, cy, scale_factor=1.0):
    """Depth에서 객체 중심 좌표 계산
    
    전경(물체) 분리 전략:
    - 가장 가까운 영역 = 물체
    - 백분위수 기반 임계값으로 전경/배경 분리
    
    Args:
        depth: (H, W) depth 이미지 (meters 또는 스케일된 값)
        scale_factor: Depth 값에 곱할 스케일 (metersPerUnit 보정용)
    
    Returns:
        centroid: (3,) 객체 중심 좌표 [X, Y, Z] in camera frame (meters)
        valid: bool 유효한 결과인지
    """
    # 스케일 적용
    depth_scaled = depth * scale_factor
    
    # 전체 Point Cloud 생성
    points, valid_mask = depth_to_pointcloud(depth_scaled, fx, fy, cx, cy)
    
    if len(points) < 100:
        return np.array([0, 0, 0]), False
    
    # 전경 분리: 가장 가까운 영역 (낮은 Z값)
    z_values = points[:, 2]
    foreground_threshold = np.percentile(z_values, FOREGROUND_PERCENTILE)
    
    # 전경 마스크 (물체 영역)
    foreground_mask = z_values < foreground_threshold
    foreground_points = points[foreground_mask]
    
    if len(foreground_points) < 50:
        return np.array([0, 0, 0]), False
    
    # 중심(centroid) 계산
    centroid = foreground_points.mean(axis=0)
    
    return centroid, True


# ==========================================
# 데이터셋: Depth에서 GT 계산
# ==========================================
class DepthGTDataset(Dataset):
    """Depth에서 Ground Truth를 계산하는 데이터셋"""
    
    def __init__(self, dataset_dir, split='train', train_ratio=TRAIN_RATIO, 
                 use_augmentation=True, position_stats=None):
        self.dataset_dir = dataset_dir
        self.split = split
        self.samples = []
        self.use_augmentation = use_augmentation and (split == 'train')
        self.position_stats = position_stats
        
        # 카메라 파라미터
        self.fx = CAMERA_INTRINSICS["fx"]
        self.fy = CAMERA_INTRINSICS["fy"]
        self.cx = CAMERA_INTRINSICS["cx"]
        self.cy = CAMERA_INTRINSICS["cy"]
        
        # 클래스별 폴더 스캔
        class_dirs = sorted(glob.glob(os.path.join(dataset_dir, "*")))
        class_dirs = [d for d in class_dirs if os.path.isdir(d) and not d.endswith('__pycache__')]
        
        self.class_names = [os.path.basename(d) for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # 모든 샘플 수집 및 GT 사전 계산
        print("Depth에서 Ground Truth 계산 중...")
        all_samples = []
        valid_count = 0
        invalid_count = 0
        
        for class_dir in class_dirs:
            class_name = os.path.basename(class_dir)
            
            # 클래스별 스케일 팩터 (Depth 범위로 자동 감지)
            # 첫 번째 depth 파일로 스케일 감지
            depth_files = sorted(glob.glob(os.path.join(class_dir, "distance_to_camera_*.npy")))
            if depth_files:
                sample_depth = np.load(depth_files[0])
                if len(sample_depth.shape) == 3:
                    sample_depth = sample_depth[:, :, 0]
                valid_depth = sample_depth[(sample_depth > 0.001) & np.isfinite(sample_depth)]
                if len(valid_depth) > 0:
                    depth_mean = valid_depth.mean()
                    # Depth mean이 1m 미만이면 스케일 보정 필요
                    if depth_mean < 0.5:  # 50cm 미만이면 100배 스케일업
                        scale_factor = 100.0
                    elif depth_mean < 1.0:  # 1m 미만이면 10배 스케일업
                        scale_factor = 10.0
                    else:
                        scale_factor = 1.0
                else:
                    scale_factor = 1.0
            else:
                scale_factor = 1.0
            
            print(f"  {class_name}: depth_mean={depth_mean:.3f}m → scale={scale_factor}")
            
            depth_files = sorted(glob.glob(os.path.join(class_dir, "distance_to_camera_*.npy")))
            
            for depth_file in tqdm(depth_files, desc=f"  {class_name}", leave=False):
                frame_idx = int(os.path.basename(depth_file).split('_')[-1].split('.')[0])
                rgb_file = os.path.join(class_dir, f"rgb_{frame_idx:04d}.png")
                
                if not os.path.exists(rgb_file):
                    continue
                
                # Depth에서 GT 계산 (스케일 적용)
                depth = np.load(depth_file)
                if len(depth.shape) == 3:
                    depth = depth[:, :, 0]
                
                centroid, valid = compute_object_centroid_from_depth(
                    depth, self.fx, self.fy, self.cx, self.cy, scale_factor
                )
                
                if valid:
                    all_samples.append({
                        'rgb_path': rgb_file,
                        'depth_path': depth_file,
                        'gt_position': centroid.tolist(),  # Depth 기반 GT
                        'class_name': class_name,
                        'class_idx': self.class_to_idx[class_name]
                    })
                    valid_count += 1
                else:
                    invalid_count += 1
        
        print(f"  유효 샘플: {valid_count}, 무효 샘플: {invalid_count}")
        
        # Train/Test 분할
        random.seed(42)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        
        if split == 'train':
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
        
        # 위치 통계 계산 (train에서만)
        if split == 'train' and position_stats is None:
            self._compute_position_stats()
        
        # RGB 변환
        if self.use_augmentation:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        print(f"[{split}] 샘플 수: {len(self.samples)}, 클래스 수: {len(self.class_names)}")
    
    def _compute_position_stats(self):
        """위치 정규화를 위한 통계 계산"""
        positions = np.array([s['gt_position'] for s in self.samples])
        self.position_stats = {
            'mean': positions.mean(axis=0).tolist(),
            'std': positions.std(axis=0).tolist()
        }
        print(f"  위치 통계 (Depth GT): mean={self.position_stats['mean']}, std={self.position_stats['std']}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # RGB 로드
        rgb = Image.open(sample['rgb_path']).convert('RGB')
        rgb = self.rgb_transform(rgb)
        
        # Ground Truth (Depth에서 계산된 것)
        gt_pos = np.array(sample['gt_position'], dtype=np.float32)
        
        # 정규화
        if self.position_stats is not None:
            mean = np.array(self.position_stats['mean'], dtype=np.float32)
            std = np.array(self.position_stats['std'], dtype=np.float32) + 1e-6
            position_normalized = (gt_pos - mean) / std
        else:
            position_normalized = gt_pos
        
        return {
            'rgb': rgb,
            'position': torch.tensor(position_normalized),
            'position_raw': torch.tensor(gt_pos),
            'class_idx': sample['class_idx']
        }


# ==========================================
# 모델: RGB → 3D 위치
# ==========================================
class RGBTo3DModel(nn.Module):
    """RGB 이미지 → 3D 위치 예측 모델
    
    Depth를 입력으로 사용하지 않음!
    RGB만 보고 3D 위치를 예측하는 것이 목표
    """
    
    def __init__(self, num_classes=4, use_resnet50=False):
        super().__init__()
        
        # RGB Encoder
        if use_resnet50:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            resnet_out = 2048
        else:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            resnet_out = 512
        
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        
        # Position Head (3D 좌표 예측)
        self.position_head = nn.Sequential(
            nn.Linear(resnet_out, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)  # X, Y, Z
        )
        
        # Classification Head
        self.class_head = nn.Sequential(
            nn.Linear(resnet_out, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        print(f"  모델: {'ResNet50' if use_resnet50 else 'ResNet18'} → 3D Position")
    
    def forward(self, rgb):
        # Encoding
        feat = self.encoder(rgb)
        feat = feat.view(feat.size(0), -1)
        
        # Predictions
        position = self.position_head(feat)
        class_logits = self.class_head(feat)
        
        return {
            'position': position,
            'class_logits': class_logits
        }


# ==========================================
# 학습
# ==========================================
def train_model(dataset_dir=DATASET_DIR):
    """모델 학습"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 데이터셋 로드
    train_dataset = DepthGTDataset(dataset_dir, split='train', use_augmentation=True)
    test_dataset = DepthGTDataset(
        dataset_dir, split='test', 
        use_augmentation=False,
        position_stats=train_dataset.position_stats
    )
    
    if len(train_dataset) == 0:
        print("데이터셋이 비어있습니다.")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    position_stats = train_dataset.position_stats
    
    # 모델 생성
    num_classes = len(train_dataset.class_names)
    model = RGBTo3DModel(num_classes=num_classes, use_resnet50=False).to(device)
    
    # 손실 함수 및 옵티마이저
    position_criterion = nn.SmoothL1Loss()
    class_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    best_pos_error = float('inf')
    patience_counter = 0
    early_stop_patience = 20
    
    print(f"\n{'='*70}")
    print("Depth 기반 GT로 6DoF Pose Estimation 학습")
    print(f"{'='*70}")
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    print(f"클래스: {train_dataset.class_names}")
    print(f"Epochs: {NUM_EPOCHS}, Batch: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print(f"위치 정규화: mean={position_stats['mean']}")
    print()
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1:03d} Train", leave=False):
            rgb = batch['rgb'].to(device)
            position = batch['position'].to(device)
            class_idx = batch['class_idx'].to(device)
            
            optimizer.zero_grad()
            pred = model(rgb)
            
            pos_loss = position_criterion(pred['position'], position)
            cls_loss = class_criterion(pred['class_logits'], class_idx)
            loss = pos_loss + 0.1 * cls_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluation
        model.eval()
        all_pos_errors = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                rgb = batch['rgb'].to(device)
                position_raw = batch['position_raw'].to(device)
                class_idx = batch['class_idx'].to(device)
                
                pred = model(rgb)
                
                # 역정규화
                mean = torch.tensor(position_stats['mean'], device=device)
                std = torch.tensor(position_stats['std'], device=device) + 1e-6
                pred_pos_raw = pred['position'] * std + mean
                
                # 위치 오차 (mm)
                pos_error = torch.sqrt(((pred_pos_raw - position_raw) ** 2).sum(dim=1)) * 1000
                all_pos_errors.extend(pos_error.cpu().numpy())
                
                # 분류 정확도
                _, pred_labels = torch.max(pred['class_logits'], 1)
                correct += (pred_labels == class_idx).sum().item()
                total += class_idx.size(0)
        
        avg_pos_error = np.mean(all_pos_errors)
        class_acc = 100 * correct / total
        
        scheduler.step()
        
        # Best 모델 저장
        if avg_pos_error < best_pos_error:
            best_pos_error = avg_pos_error
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'class_names': train_dataset.class_names,
                'position_stats': position_stats,
                'best_pos_error': best_pos_error
            }, os.path.join(ARTIFACTS_DIR, 'depth_gt_pose_best.pt'))
        else:
            patience_counter += 1
        
        # 로그
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{NUM_EPOCHS}] "
                  f"Loss={train_loss:.4f} | "
                  f"PosErr={avg_pos_error:.1f}mm | "
                  f"ClassAcc={class_acc:.1f}% | "
                  f"best={best_pos_error:.1f}mm")
        
        # Early Stopping
        if patience_counter >= early_stop_patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*70}")
    print(f"학습 완료! 최고 위치 오차: {best_pos_error:.2f}mm")
    print(f"모델 저장: {os.path.join(ARTIFACTS_DIR, 'depth_gt_pose_best.pt')}")
    
    # 오차 분포
    print(f"\n위치 오차 분포:")
    print(f"  < 10mm: {100 * sum(1 for e in all_pos_errors if e < 10) / len(all_pos_errors):.1f}%")
    print(f"  < 50mm: {100 * sum(1 for e in all_pos_errors if e < 50) / len(all_pos_errors):.1f}%")
    print(f"  < 100mm: {100 * sum(1 for e in all_pos_errors if e < 100) / len(all_pos_errors):.1f}%")
    print(f"  < 200mm: {100 * sum(1 for e in all_pos_errors if e < 200) / len(all_pos_errors):.1f}%")


def verify_depth_gt(dataset_dir=DATASET_DIR, num_samples=5):
    """Depth에서 계산된 GT 검증"""
    print(f"\n{'='*70}")
    print("Depth 기반 GT 검증")
    print(f"{'='*70}")
    
    fx = CAMERA_INTRINSICS["fx"]
    fy = CAMERA_INTRINSICS["fy"]
    cx = CAMERA_INTRINSICS["cx"]
    cy = CAMERA_INTRINSICS["cy"]
    
    class_dirs = sorted(glob.glob(os.path.join(dataset_dir, "*")))
    class_dirs = [d for d in class_dirs if os.path.isdir(d) and not d.endswith('__pycache__')]
    
    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        
        # 스케일 팩터 (Depth 범위로 자동 감지)
        depth_files_sample = sorted(glob.glob(os.path.join(class_dir, "distance_to_camera_*.npy")))
        if depth_files_sample:
            sample_depth = np.load(depth_files_sample[0])
            if len(sample_depth.shape) == 3:
                sample_depth = sample_depth[:, :, 0]
            valid_depth = sample_depth[(sample_depth > 0.001) & np.isfinite(sample_depth)]
            if len(valid_depth) > 0:
                depth_mean = valid_depth.mean()
                if depth_mean < 0.5:
                    scale_factor = 100.0
                elif depth_mean < 1.0:
                    scale_factor = 10.0
                else:
                    scale_factor = 1.0
            else:
                depth_mean = 0
                scale_factor = 1.0
        else:
            depth_mean = 0
            scale_factor = 1.0
        
        print(f"\n{class_name} (depth_mean={depth_mean:.3f}m, scale={scale_factor}):")
        
        depth_files = sorted(glob.glob(os.path.join(class_dir, "distance_to_camera_*.npy")))[:num_samples]
        
        for depth_file in depth_files:
            frame_idx = int(os.path.basename(depth_file).split('_')[-1].split('.')[0])
            pose_file = os.path.join(class_dir, f"pose_{frame_idx:04d}.json")
            
            # Depth에서 GT 계산 (스케일 적용)
            depth = np.load(depth_file)
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]
            
            # Depth 통계 출력
            valid_depth = depth[(depth > 0.001) & np.isfinite(depth)]
            if len(valid_depth) > 0:
                depth_min = valid_depth.min()
                depth_max = valid_depth.max()
            else:
                depth_min = depth_max = 0
            
            centroid, valid = compute_object_centroid_from_depth(
                depth, fx, fy, cx, cy, scale_factor
            )
            
            # 기존 pose 파일의 camTobj와 비교
            if os.path.exists(pose_file):
                with open(pose_file) as f:
                    pose = json.load(f)
                old_gt = pose["camTobj"]["t_xyz_m"]
                
                # Z축 부호 보정해서 비교 (기존 GT는 -Z, Depth GT는 +Z)
                old_gt_corrected = [old_gt[0], old_gt[1], abs(old_gt[2])]
                
                diff = np.sqrt(sum((centroid[i] - old_gt_corrected[i])**2 for i in range(3)))
                
                print(f"  Frame {frame_idx}: (depth range: {depth_min:.3f}~{depth_max:.3f})")
                print(f"    Depth GT:     [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")
                print(f"    기존 GT:      [{old_gt[0]:.2f}, {old_gt[1]:.2f}, {old_gt[2]:.2f}]")
                print(f"    기존 GT(|Z|): [{old_gt_corrected[0]:.2f}, {old_gt_corrected[1]:.2f}, {old_gt_corrected[2]:.2f}]")
                print(f"    차이 (|Z| 보정): {diff*1000:.1f}mm {'✅' if diff < 0.5 else '❌'}")


# ==========================================
# 메인
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Depth 기반 6DoF Pose Estimation")
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'verify'],
                        help='실행 모드')
    parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    
    args = parser.parse_args()
    
    NUM_EPOCHS = args.epochs
    
    if args.mode == 'train':
        train_model(args.dataset_dir)
    elif args.mode == 'verify':
        verify_depth_gt(args.dataset_dir)

