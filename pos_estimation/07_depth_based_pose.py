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
# 로깅 설정
# ==========================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, REPO_DIR)
from utils.logger import setup_logging, finish_logging

# ==========================================
# 설정
# ==========================================
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
                 use_augmentation=True, position_stats=None, use_bbox_crop=False):
        self.dataset_dir = dataset_dir
        self.split = split
        self.samples = []
        self.use_augmentation = use_augmentation and (split == 'train')
        self.position_stats = position_stats
        self.use_bbox_crop = use_bbox_crop
        
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
                    # pose 파일 읽기 (bbox와 rotation 정보)
                    bbox = None
                    rotation_6d = None
                    pose_file = os.path.join(class_dir, f"pose_{frame_idx:04d}.json")
                    
                    if os.path.exists(pose_file):
                        with open(pose_file) as f:
                            pose_data = json.load(f)
                        
                        # bbox_2d 읽기
                        if self.use_bbox_crop:
                            bbox_data = pose_data.get("bbox_2d", {})
                            if bbox_data.get("x_max", 0) > 0:
                                bbox = {
                                    'x_min': int(bbox_data.get("x_min", 0)),
                                    'y_min': int(bbox_data.get("y_min", 0)),
                                    'x_max': int(bbox_data.get("x_max", 1024)),
                                    'y_max': int(bbox_data.get("y_max", 1024))
                                }
                        
                        # rotation 읽기 (camTobj.r_xyz_deg → 6D)
                        camTobj = pose_data.get("camTobj", {})
                        euler_deg = camTobj.get("r_xyz_deg", [0, 0, 0])
                        R = euler_to_rotation_matrix(euler_deg)
                        rotation_6d = rotation_matrix_to_6d(R).tolist()
                    else:
                        # pose 파일이 없으면 단위 회전
                        rotation_6d = [1, 0, 0, 0, 1, 0]  # Identity rotation
                    
                    all_samples.append({
                        'rgb_path': rgb_file,
                        'depth_path': depth_file,
                        'gt_position': centroid.tolist(),  # Depth 기반 GT
                        'gt_rotation_6d': rotation_6d,  # 6D Rotation GT
                        'class_name': class_name,
                        'class_idx': self.class_to_idx[class_name],
                        'scale_factor': scale_factor,  # Depth 스케일 보정용
                        'bbox': bbox  # bbox_2d 정보
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
        
        # Depth 로드 및 전처리
        depth = np.load(sample['depth_path'])
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        
        # bbox crop 적용
        bbox = sample.get('bbox')
        if self.use_bbox_crop and bbox is not None:
            x_min, y_min = bbox['x_min'], bbox['y_min']
            x_max, y_max = bbox['x_max'], bbox['y_max']
            
            # 10% 마진 추가
            w, h = x_max - x_min, y_max - y_min
            margin_x, margin_y = int(w * 0.1), int(h * 0.1)
            x_min = max(0, x_min - margin_x)
            y_min = max(0, y_min - margin_y)
            x_max = min(rgb.width, x_max + margin_x)
            y_max = min(rgb.height, y_max + margin_y)
            
            # RGB crop
            rgb = rgb.crop((x_min, y_min, x_max, y_max))
            
            # Depth crop
            depth = depth[y_min:y_max, x_min:x_max]
        
        # RGB transform
        rgb = self.rgb_transform(rgb)
        
        # Depth 스케일 적용 (클래스별로 저장된 scale_factor 사용)
        depth = depth * sample.get('scale_factor', 1.0)
        
        # Depth 정규화 (0~1 범위로)
        depth_valid = depth[(depth > DEPTH_MIN) & (depth < DEPTH_MAX)]
        if len(depth_valid) > 0:
            depth_min = depth_valid.min()
            depth_max = depth_valid.max()
            depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-6)
        else:
            depth_normalized = depth / (DEPTH_MAX + 1e-6)
        
        depth_normalized = np.clip(depth_normalized, 0, 1).astype(np.float32)
        
        # Depth 리사이즈 (224x224)
        depth_pil = Image.fromarray((depth_normalized * 255).astype(np.uint8))
        depth_pil = depth_pil.resize((224, 224), Image.BILINEAR)
        depth_tensor = torch.tensor(np.array(depth_pil) / 255.0, dtype=torch.float32).unsqueeze(0)
        
        # Ground Truth (Depth에서 계산된 것)
        gt_pos = np.array(sample['gt_position'], dtype=np.float32)
        gt_rot_6d = np.array(sample['gt_rotation_6d'], dtype=np.float32)
        
        # 위치 정규화
        if self.position_stats is not None:
            mean = np.array(self.position_stats['mean'], dtype=np.float32)
            std = np.array(self.position_stats['std'], dtype=np.float32) + 1e-6
            position_normalized = (gt_pos - mean) / std
        else:
            position_normalized = gt_pos
        
        return {
            'rgb': rgb,
            'depth': depth_tensor,
            'position': torch.tensor(position_normalized),
            'position_raw': torch.tensor(gt_pos),
            'rotation_6d': torch.tensor(gt_rot_6d),
            'class_idx': sample['class_idx']
        }


# ==========================================
# 회전 표현 변환 함수
# ==========================================
def euler_to_rotation_matrix(euler_deg):
    """Euler angles (degrees) → Rotation Matrix (3x3)"""
    roll, pitch, yaw = np.radians(euler_deg)
    
    # Roll (X축 회전)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Pitch (Y축 회전)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Yaw (Z축 회전)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # ZYX 순서로 결합
    R = Rz @ Ry @ Rx
    return R


def rotation_matrix_to_6d(R):
    """Rotation Matrix (3x3) → 6D Representation
    
    6D = 회전 행렬의 처음 두 열 (6개 값)
    """
    return np.concatenate([R[:, 0], R[:, 1]])


def rotation_6d_to_matrix(rot_6d):
    """6D Representation → Rotation Matrix (3x3)
    
    Gram-Schmidt 정규화로 직교 행렬 복원
    """
    a1 = rot_6d[:3]
    a2 = rot_6d[3:6]
    
    # 첫 번째 열 정규화
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    
    # 두 번째 열: a2에서 b1 방향 성분 제거 후 정규화
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    
    # 세 번째 열: 외적
    b3 = np.cross(b1, b2)
    
    R = np.stack([b1, b2, b3], axis=1)
    return R


def rotation_6d_to_matrix_batch(rot_6d):
    """6D Representation → Rotation Matrix (배치 처리, PyTorch)"""
    a1 = rot_6d[:, :3]
    a2 = rot_6d[:, 3:6]
    
    b1 = F.normalize(a1, dim=1)
    b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    
    R = torch.stack([b1, b2, b3], dim=2)
    return R


def compute_rotation_error(pred_6d, gt_6d):
    """두 6D 회전 표현 간의 각도 오차 (degrees)"""
    R_pred = rotation_6d_to_matrix_batch(pred_6d)
    R_gt = rotation_6d_to_matrix_batch(gt_6d)
    
    # R_pred^T @ R_gt의 trace로 각도 계산
    R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
    trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
    
    # trace = 1 + 2*cos(theta) → theta = arccos((trace-1)/2)
    cos_angle = (trace - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1, 1)
    angle_rad = torch.acos(cos_angle)
    angle_deg = torch.rad2deg(angle_rad)
    
    return angle_deg


# ==========================================
# Depth Encoder
# ==========================================
class DepthEncoder(nn.Module):
    """Depth 이미지를 인코딩하는 CNN"""
    
    def __init__(self, out_features=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, out_features)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ==========================================
# 모델: RGB + Depth → 3D 위치
# ==========================================
class RGBDepthTo3DModel(nn.Module):
    """RGB + Depth 융합 → 3D 위치 예측 모델
    
    RGB: 시각적 특징 (물체 인식, 형태)
    Depth: 거리 정보 (Z 좌표에 직접적 기여)
    """
    
    def __init__(self, num_classes=4, use_resnet50=False, depth_features=256):
        super().__init__()
        
        # RGB Encoder (ResNet)
        if use_resnet50:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            rgb_out = 2048
        else:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            rgb_out = 512
        
        self.rgb_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.rgb_fc = nn.Linear(rgb_out, 512)
        
        # Depth Encoder (Custom CNN)
        self.depth_encoder = DepthEncoder(out_features=depth_features)
        
        # Fusion dimension
        fusion_dim = 512 + depth_features  # RGB(512) + Depth(256) = 768
        
        # Position Head (3D 좌표 예측)
        self.position_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)  # X, Y, Z
        )
        
        # Rotation Head (6D 회전 표현 예측)
        self.rotation_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6)  # 6D Rotation Representation
        )
        
        # Classification Head
        self.class_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        print(f"  모델: {'ResNet50' if use_resnet50 else 'ResNet18'} + DepthEncoder → 6DoF Pose")
        print(f"  RGB features: 512, Depth features: {depth_features}, Fusion: {fusion_dim}")
        print(f"  출력: Position(3) + Rotation(6D) + Class({num_classes})")
    
    def forward(self, rgb, depth):
        # RGB Encoding
        rgb_feat = self.rgb_encoder(rgb)
        rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)
        rgb_feat = self.rgb_fc(rgb_feat)
        
        # Depth Encoding
        depth_feat = self.depth_encoder(depth)
        
        # Fusion
        fused = torch.cat([rgb_feat, depth_feat], dim=1)
        
        # Predictions
        position = self.position_head(fused)
        rotation_6d = self.rotation_head(fused)
        class_logits = self.class_head(fused)
        
        return {
            'position': position,
            'rotation_6d': rotation_6d,
            'class_logits': class_logits
        }


# ==========================================
# 학습
# ==========================================
def train_model(dataset_dir=DATASET_DIR, use_resnet50=False, use_bbox_crop=False):
    """모델 학습"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 데이터셋 로드
    train_dataset = DepthGTDataset(dataset_dir, split='train', use_augmentation=True, 
                                    use_bbox_crop=use_bbox_crop)
    test_dataset = DepthGTDataset(
        dataset_dir, split='test', 
        use_augmentation=False,
        position_stats=train_dataset.position_stats,
        use_bbox_crop=use_bbox_crop
    )
    
    if use_bbox_crop:
        print(f"  BBox Crop: ✅ 활성화")
    
    if len(train_dataset) == 0:
        print("데이터셋이 비어있습니다.")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    position_stats = train_dataset.position_stats
    
    # 모델 생성 (RGB + Depth 융합)
    num_classes = len(train_dataset.class_names)
    model = RGBDepthTo3DModel(num_classes=num_classes, use_resnet50=use_resnet50, depth_features=256).to(device)
    
    # 손실 함수 및 옵티마이저
    position_criterion = nn.SmoothL1Loss()
    rotation_criterion = nn.SmoothL1Loss()  # 6D rotation에 대한 L1 loss
    class_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    best_pos_error = float('inf')
    best_rot_error = float('inf')
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
            depth = batch['depth'].to(device)
            position = batch['position'].to(device)
            rotation_6d = batch['rotation_6d'].to(device)
            class_idx = batch['class_idx'].to(device)
            
            optimizer.zero_grad()
            pred = model(rgb, depth)  # RGB + Depth 입력
            
            pos_loss = position_criterion(pred['position'], position)
            rot_loss = rotation_criterion(pred['rotation_6d'], rotation_6d)
            cls_loss = class_criterion(pred['class_logits'], class_idx)
            
            # 가중치 조합: 위치(1.0) + 회전(0.5) + 분류(0.1)
            loss = pos_loss + 0.5 * rot_loss + 0.1 * cls_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Evaluation
        model.eval()
        all_pos_errors = []
        all_rot_errors = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                rgb = batch['rgb'].to(device)
                depth = batch['depth'].to(device)
                position_raw = batch['position_raw'].to(device)
                rotation_6d_gt = batch['rotation_6d'].to(device)
                class_idx = batch['class_idx'].to(device)
                
                pred = model(rgb, depth)  # RGB + Depth 입력
                
                # 역정규화
                mean = torch.tensor(position_stats['mean'], device=device)
                std = torch.tensor(position_stats['std'], device=device) + 1e-6
                pred_pos_raw = pred['position'] * std + mean
                
                # 위치 오차 (mm)
                pos_error = torch.sqrt(((pred_pos_raw - position_raw) ** 2).sum(dim=1)) * 1000
                all_pos_errors.extend(pos_error.cpu().numpy())
                
                # 회전 오차 (degrees)
                rot_error = compute_rotation_error(pred['rotation_6d'], rotation_6d_gt)
                all_rot_errors.extend(rot_error.cpu().numpy())
                
                # 분류 정확도
                _, pred_labels = torch.max(pred['class_logits'], 1)
                correct += (pred_labels == class_idx).sum().item()
                total += class_idx.size(0)
        
        avg_pos_error = np.mean(all_pos_errors)
        avg_rot_error = np.mean(all_rot_errors)
        class_acc = 100 * correct / total
        
        scheduler.step()
        
        # Best 모델 저장
        if avg_pos_error < best_pos_error:
            best_pos_error = avg_pos_error
            best_rot_error = avg_rot_error
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'class_names': train_dataset.class_names,
                'position_stats': position_stats,
                'best_pos_error': best_pos_error,
                'best_rot_error': best_rot_error
            }, os.path.join(ARTIFACTS_DIR, 'depth_gt_6dof_best.pt'))
        else:
            patience_counter += 1
        
        # 로그
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{NUM_EPOCHS}] "
                  f"Loss={train_loss:.4f} | "
                  f"PosErr={avg_pos_error:.1f}mm | "
                  f"RotErr={avg_rot_error:.1f}° | "
                  f"ClassAcc={class_acc:.1f}% | "
                  f"best={best_pos_error:.1f}mm")
        
        # Early Stopping
        if patience_counter >= early_stop_patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*70}")
    print(f"🎯 6DoF 학습 완료!")
    print(f"  최고 위치 오차: {best_pos_error:.2f}mm")
    print(f"  최고 회전 오차: {best_rot_error:.2f}°")
    print(f"모델 저장: {os.path.join(ARTIFACTS_DIR, 'depth_gt_6dof_best.pt')}")
    
    # 위치 오차 분포
    print(f"\n위치 오차 분포:")
    print(f"  < 10mm: {100 * sum(1 for e in all_pos_errors if e < 10) / len(all_pos_errors):.1f}%")
    print(f"  < 50mm: {100 * sum(1 for e in all_pos_errors if e < 50) / len(all_pos_errors):.1f}%")
    print(f"  < 100mm: {100 * sum(1 for e in all_pos_errors if e < 100) / len(all_pos_errors):.1f}%")
    print(f"  < 200mm: {100 * sum(1 for e in all_pos_errors if e < 200) / len(all_pos_errors):.1f}%")
    
    # 회전 오차 분포
    print(f"\n회전 오차 분포:")
    print(f"  < 5°: {100 * sum(1 for e in all_rot_errors if e < 5) / len(all_rot_errors):.1f}%")
    print(f"  < 10°: {100 * sum(1 for e in all_rot_errors if e < 10) / len(all_rot_errors):.1f}%")
    print(f"  < 20°: {100 * sum(1 for e in all_rot_errors if e < 20) / len(all_rot_errors):.1f}%")
    print(f"  < 45°: {100 * sum(1 for e in all_rot_errors if e < 45) / len(all_rot_errors):.1f}%")


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
# 평가 (Train/Test 오차 통계)
# ==========================================
def evaluate_model(dataset_dir=DATASET_DIR, use_resnet50=False, use_bbox_crop=False):
    """저장된 모델로 Train/Test 데이터 평가"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 모델 로드
    model_path = os.path.join(ARTIFACTS_DIR, 'depth_gt_6dof_best.pt')
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    position_stats = checkpoint['position_stats']
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    
    print(f"모델 로드: {model_path}")
    print(f"  클래스: {class_names}")
    print(f"  위치 정규화: mean={position_stats['mean']}")
    
    # 모델 생성 및 가중치 로드
    model = RGBDepthTo3DModel(num_classes=num_classes, use_resnet50=use_resnet50, depth_features=256).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 데이터셋 로드
    print("\n데이터셋 로드 중...")
    train_dataset = DepthGTDataset(dataset_dir, split='train', use_augmentation=False, 
                                    use_bbox_crop=use_bbox_crop)
    test_dataset = DepthGTDataset(
        dataset_dir, split='test', 
        use_augmentation=False,
        position_stats=train_dataset.position_stats,
        use_bbox_crop=use_bbox_crop
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    def compute_errors(data_loader, split_name):
        """데이터셋에 대한 오차 계산"""
        all_pos_errors = []
        all_rot_errors = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc=f"{split_name} 평가"):
                rgb = batch['rgb'].to(device)
                depth = batch['depth'].to(device)
                position_raw = batch['position_raw'].to(device)
                rotation_6d_gt = batch['rotation_6d'].to(device)
                
                pred = model(rgb, depth)
                
                # 역정규화
                mean = torch.tensor(position_stats['mean'], device=device)
                std = torch.tensor(position_stats['std'], device=device) + 1e-6
                pred_pos_raw = pred['position'] * std + mean
                
                # 위치 오차 (mm)
                pos_error = torch.sqrt(((pred_pos_raw - position_raw) ** 2).sum(dim=1)) * 1000
                all_pos_errors.extend(pos_error.cpu().numpy())
                
                # 회전 오차 (degrees)
                rot_error = compute_rotation_error(pred['rotation_6d'], rotation_6d_gt)
                all_rot_errors.extend(rot_error.cpu().numpy())
        
        return np.array(all_pos_errors), np.array(all_rot_errors)
    
    # Train 데이터 평가
    train_pos_errors, train_rot_errors = compute_errors(train_loader, "Train")
    
    # Test 데이터 평가
    test_pos_errors, test_rot_errors = compute_errors(test_loader, "Test")
    
    # 결과 출력
    print(f"\n{'='*70}")
    print("📊 Train/Test 오차 통계")
    print(f"{'='*70}")
    
    print(f"\n=== Train 데이터 ({len(train_pos_errors):,}장) ===")
    print(f"  위치 오차: 평균 {train_pos_errors.mean():.2f} mm, 표준편차 {train_pos_errors.std():.2f} mm")
    print(f"  회전 오차: 평균 {train_rot_errors.mean():.2f}°, 표준편차 {train_rot_errors.std():.2f}°")
    print(f"  위치 오차 범위: {train_pos_errors.min():.2f} ~ {train_pos_errors.max():.2f} mm")
    print(f"  회전 오차 범위: {train_rot_errors.min():.2f} ~ {train_rot_errors.max():.2f}°")
    
    print(f"\n=== Test 데이터 ({len(test_pos_errors):,}장) ===")
    print(f"  위치 오차: 평균 {test_pos_errors.mean():.2f} mm, 표준편차 {test_pos_errors.std():.2f} mm")
    print(f"  회전 오차: 평균 {test_rot_errors.mean():.2f}°, 표준편차 {test_rot_errors.std():.2f}°")
    print(f"  위치 오차 범위: {test_pos_errors.min():.2f} ~ {test_pos_errors.max():.2f} mm")
    print(f"  회전 오차 범위: {test_rot_errors.min():.2f} ~ {test_rot_errors.max():.2f}°")
    
    # 위치 오차 분포
    print(f"\n=== 위치 오차 분포 ===")
    print(f"{'기준':<12} {'Train':>12} {'Test':>12}")
    print(f"{'-'*36}")
    for threshold in [10, 20, 50, 100, 200]:
        train_pct = 100 * (train_pos_errors < threshold).sum() / len(train_pos_errors)
        test_pct = 100 * (test_pos_errors < threshold).sum() / len(test_pos_errors)
        print(f"< {threshold}mm{'':<6} {train_pct:>10.1f}% {test_pct:>10.1f}%")
    
    # 회전 오차 분포
    print(f"\n=== 회전 오차 분포 ===")
    print(f"{'기준':<12} {'Train':>12} {'Test':>12}")
    print(f"{'-'*36}")
    for threshold in [1, 2, 5, 10, 20]:
        train_pct = 100 * (train_rot_errors < threshold).sum() / len(train_rot_errors)
        test_pct = 100 * (test_rot_errors < threshold).sum() / len(test_rot_errors)
        print(f"< {threshold}°{'':<8} {train_pct:>10.1f}% {test_pct:>10.1f}%")
    
    print(f"\n{'='*70}")


# ==========================================
# 메인
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Depth 기반 6DoF Pose Estimation")
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'verify', 'eval'],
                        help='실행 모드')
    parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--resnet50', action='store_true', 
                        help='ResNet50 사용 (기본: ResNet18)')
    parser.add_argument('--bbox_crop', action='store_true',
                        help='bbox_2d로 이미지 크롭 후 학습')
    
    args = parser.parse_args()
    
    # 로그 파일 생성
    LOG_PATH = setup_logging(f"07_pose_{args.mode}")
    
    NUM_EPOCHS = args.epochs
    
    if args.mode == 'train':
        train_model(args.dataset_dir, use_resnet50=args.resnet50, use_bbox_crop=args.bbox_crop)
    elif args.mode == 'verify':
        verify_depth_gt(args.dataset_dir)
    elif args.mode == 'eval':
        evaluate_model(args.dataset_dir, use_resnet50=args.resnet50, use_bbox_crop=args.bbox_crop)
    
    # 로깅 종료
    finish_logging()

