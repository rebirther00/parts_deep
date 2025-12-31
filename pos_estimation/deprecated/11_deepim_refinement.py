#!/usr/bin/env python3
# ==========================================
# DeepIM (Deep Iterative Matching) 학습 스크립트
# ==========================================
#
# 기존 포즈 추정 모델의 예측을 초기값으로 사용하여
# 반복적 정제를 통해 10mm 이하의 고정밀 포즈 추정을 목표로 합니다.
#
# 핵심 아이디어:
# 1. 기존 모델로 초기 포즈 예측
# 2. 해당 포즈로 3D 모델 렌더링
# 3. 실제 이미지와 렌더링 이미지를 비교하여 포즈 잔차 예측
# 4. 포즈 업데이트 및 반복
#
# 사용법:
#   python 11_deepim_refinement.py --mode train
#   python 11_deepim_refinement.py --mode train --initial_model depth_gt_pose_v2_best.pt
#
# ==========================================

import os
import sys
import json
import glob
import random
import argparse
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image

# ==========================================
# 경로 설정
# ==========================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, REPO_DIR)

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset_pos_depth")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
ASSETS_DIR = os.path.join(REPO_DIR, "assets")

# 로깅 설정
from utils.logger import setup_logging, finish_logging

# DeepIM 모듈 임포트
from deepim.refiner import DeepIMRefiner, DeepIMRefinerLight
from deepim.loss import DeepIMLoss, compute_pose_error, rotation_6d_to_matrix, compose_rotation_6d

# 렌더러는 PyTorch3D가 있을 때만 임포트
try:
    from deepim.renderer import MeshRenderer, PYTORCH3D_AVAILABLE
except ImportError:
    PYTORCH3D_AVAILABLE = False

# ==========================================
# 설정
# ==========================================
BATCH_SIZE = None
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.8
NUM_ITERATIONS = 4  # 반복 정제 횟수
WARMUP_EPOCHS = 5

# Depth 설정
DEPTH_MIN = 0.01
DEPTH_MAX = 100.0
FOREGROUND_PERCENTILE = 10

CAMERA_INTRINSICS = {
    "fx": 768.0, "fy": 768.0,
    "cx": 512.0, "cy": 512.0,
    "width": 1024, "height": 1024
}


# ==========================================
# 6D Rotation 함수들
# ==========================================
def euler_to_rotation_matrix(roll, pitch, yaw, degrees=True):
    """Euler angles → 3x3 회전 행렬"""
    if degrees:
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
    
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    R = Rz @ Ry @ Rx
    return R.astype(np.float32)


def rotation_matrix_to_6d(R):
    """3x3 회전 행렬 → 6D 표현"""
    return np.concatenate([R[:, 0], R[:, 1]], axis=0).astype(np.float32)


def euler_to_6d(roll, pitch, yaw, degrees=True):
    """Euler angles → 6D rotation"""
    R = euler_to_rotation_matrix(roll, pitch, yaw, degrees)
    return rotation_matrix_to_6d(R)


# ==========================================
# Depth → 객체 중심 계산
# ==========================================
def depth_to_pointcloud(depth, fx, fy, cx, cy):
    """Depth → Point Cloud"""
    h, w = depth.shape
    valid_mask = (depth > DEPTH_MIN) & (depth < DEPTH_MAX) & np.isfinite(depth)
    
    u = np.arange(w)
    v = np.arange(h)
    u, v = np.meshgrid(u, v)
    
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=1)
    return points, valid_mask


def compute_object_centroid_from_depth(depth, fx, fy, cx, cy, scale_factor=1.0):
    """Depth에서 객체 중심 계산"""
    depth_scaled = depth * scale_factor
    points, valid_mask = depth_to_pointcloud(depth_scaled, fx, fy, cx, cy)
    
    if len(points) < 100:
        return np.array([0, 0, 0]), False
    
    z_values = points[:, 2]
    foreground_threshold = np.percentile(z_values, FOREGROUND_PERCENTILE)
    
    foreground_mask = z_values < foreground_threshold
    foreground_points = points[foreground_mask]
    
    if len(foreground_points) < 50:
        return np.array([0, 0, 0]), False
    
    centroid = foreground_points.mean(axis=0)
    return centroid, True


# ==========================================
# GPU 최적화
# ==========================================
def get_optimal_batch_size(data_size, force_cpu=False):
    """최적 배치 사이즈 결정"""
    if data_size < 100:
        batch_size = 8
    elif data_size < 500:
        batch_size = 16
    elif data_size < 2000:
        batch_size = 32
    else:
        batch_size = 64
    
    if torch.cuda.is_available() and not force_cpu:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb < 8:
            batch_size = min(batch_size, 16)
        elif gpu_memory_gb < 16:
            batch_size = min(batch_size, 32)
        elif gpu_memory_gb < 32:
            batch_size = max(batch_size, 48)
        else:
            batch_size = max(batch_size, 64)
    
    return batch_size


# ==========================================
# 기존 포즈 모델 (초기값 생성용)
# ==========================================
class DepthEncoder(nn.Module):
    """Depth 인코더"""
    
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


class RGBDepthTo3DModel(nn.Module):
    """기존 RGB+Depth → 6DoF 모델"""
    
    def __init__(self, num_classes=4, depth_features=256, use_rotation=True):
        super().__init__()
        self.use_rotation = use_rotation
        
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.rgb_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.rgb_fc = nn.Linear(2048, 512)
        
        self.depth_encoder = DepthEncoder(out_features=depth_features)
        
        fusion_dim = 512 + depth_features
        
        self.position_head = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 3)
        )
        
        if use_rotation:
            self.rotation_head = nn.Sequential(
                nn.Linear(fusion_dim, 512), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 6)
            )
        
        self.class_head = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, rgb, depth):
        rgb_feat = self.rgb_encoder(rgb)
        rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)
        rgb_feat = self.rgb_fc(rgb_feat)
        
        depth_feat = self.depth_encoder(depth)
        fused = torch.cat([rgb_feat, depth_feat], dim=1)
        
        position = self.position_head(fused)
        class_logits = self.class_head(fused)
        
        result = {'position': position, 'class_logits': class_logits}
        
        if self.use_rotation:
            result['rotation'] = self.rotation_head(fused)
        
        return result


# ==========================================
# DeepIM 데이터셋
# ==========================================
class DeepIMDataset(Dataset):
    """DeepIM 학습용 데이터셋
    
    학습 시에는 GT 포즈에 노이즈를 추가하여 초기 포즈를 시뮬레이션
    """
    
    def __init__(self, dataset_dir, split='train', train_ratio=0.8,
                 position_stats=None, noise_position=0.05, noise_rotation=3.0):
        """
        Args:
            noise_position: 위치 노이즈 표준편차 (meters)
            noise_rotation: 회전 노이즈 표준편차 (degrees)
        """
        self.dataset_dir = dataset_dir
        self.split = split
        self.samples = []
        self.position_stats = position_stats
        self.noise_position = noise_position
        self.noise_rotation = noise_rotation
        
        self.fx = CAMERA_INTRINSICS["fx"]
        self.fy = CAMERA_INTRINSICS["fy"]
        self.cx = CAMERA_INTRINSICS["cx"]
        self.cy = CAMERA_INTRINSICS["cy"]
        
        # 클래스 폴더 스캔
        class_dirs = sorted(glob.glob(os.path.join(dataset_dir, "*")))
        class_dirs = [d for d in class_dirs if os.path.isdir(d) and not d.endswith('__pycache__')]
        
        self.class_names = [os.path.basename(d) for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # 스케일 팩터 결정
        self.class_scale_factors = {}
        for class_dir in class_dirs:
            class_name = os.path.basename(class_dir)
            depth_files = sorted(glob.glob(os.path.join(class_dir, "distance_to_camera_*.npy")))
            if depth_files:
                sample_depth = np.load(depth_files[0])
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
                    scale_factor = 1.0
            else:
                scale_factor = 1.0
            self.class_scale_factors[class_name] = scale_factor
        
        # 샘플 수집
        self._collect_samples(class_dirs, train_ratio)
        
        # Transform
        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"[{split}] 샘플 수: {len(self.samples)}, 클래스: {self.class_names}")
    
    def _collect_samples(self, class_dirs, train_ratio):
        """샘플 수집 및 GT 계산"""
        print("데이터 로드 중...")
        all_samples = []
        
        for class_dir in tqdm(class_dirs, desc="클래스 스캔"):
            class_name = os.path.basename(class_dir)
            scale_factor = self.class_scale_factors[class_name]
            
            depth_files = sorted(glob.glob(os.path.join(class_dir, "distance_to_camera_*.npy")))
            
            for depth_file in depth_files:
                frame_idx = int(os.path.basename(depth_file).split('_')[-1].split('.')[0])
                rgb_file = os.path.join(class_dir, f"rgb_{frame_idx:04d}.png")
                pose_file = os.path.join(class_dir, f"pose_{frame_idx:04d}.json")
                
                if not os.path.exists(rgb_file) or not os.path.exists(pose_file):
                    continue
                
                # Depth에서 GT 위치 계산
                depth = np.load(depth_file)
                if len(depth.shape) == 3:
                    depth = depth[:, :, 0]
                
                centroid, valid = compute_object_centroid_from_depth(
                    depth, self.fx, self.fy, self.cx, self.cy, scale_factor
                )
                
                if not valid:
                    continue
                
                # 자세 정보 로드
                with open(pose_file, 'r') as f:
                    pose_data = json.load(f)
                r_xyz_deg = pose_data.get('camTobj', {}).get('r_xyz_deg', [0, 0, 0])
                rot_6d = euler_to_6d(r_xyz_deg[0], r_xyz_deg[1], r_xyz_deg[2], degrees=True)
                
                all_samples.append({
                    'rgb_path': rgb_file,
                    'depth_path': depth_file,
                    'gt_position': centroid.tolist(),
                    'gt_rotation_6d': rot_6d.tolist(),
                    'gt_euler_deg': r_xyz_deg,
                    'class_name': class_name,
                    'class_idx': self.class_to_idx[class_name],
                    'scale_factor': scale_factor
                })
        
        # Train/Test 분할
        random.seed(42)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        
        if self.split == 'train':
            self.samples = all_samples[:split_idx]
            if self.position_stats is None:
                self._compute_position_stats()
        else:
            self.samples = all_samples[split_idx:]
    
    def _compute_position_stats(self):
        """위치 정규화 통계"""
        positions = np.array([s['gt_position'] for s in self.samples])
        self.position_stats = {
            'mean': positions.mean(axis=0).tolist(),
            'std': positions.std(axis=0).tolist()
        }
        print(f"위치 통계: mean={self.position_stats['mean']}, std={self.position_stats['std']}")
    
    def _add_pose_noise(self, position, rotation_6d):
        """초기 포즈에 노이즈 추가 (학습 시 사용)"""
        # 위치 노이즈
        noisy_pos = position + np.random.normal(0, self.noise_position, 3)
        
        # 회전 노이즈 (Euler 각도에 추가)
        euler_noise = np.random.normal(0, self.noise_rotation, 3)
        R_gt = euler_to_rotation_matrix(0, 0, 0)  # 단위 행렬에서 시작
        R_noise = euler_to_rotation_matrix(euler_noise[0], euler_noise[1], euler_noise[2])
        
        # GT 회전 행렬 복원
        R_gt_full = np.zeros((3, 3), dtype=np.float32)
        R_gt_full[:, 0] = rotation_6d[:3]
        b2 = rotation_6d[3:6] - np.dot(rotation_6d[3:6], rotation_6d[:3]) * rotation_6d[:3]
        R_gt_full[:, 1] = b2 / (np.linalg.norm(b2) + 1e-8)
        R_gt_full[:, 2] = np.cross(R_gt_full[:, 0], R_gt_full[:, 1])
        
        # 노이즈 적용
        R_noisy = R_noise @ R_gt_full
        noisy_rot_6d = rotation_matrix_to_6d(R_noisy)
        
        return noisy_pos.astype(np.float32), noisy_rot_6d
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # RGB 로드
        rgb = Image.open(sample['rgb_path']).convert('RGB')
        rgb = self.rgb_transform(rgb)
        
        # Depth 로드
        depth_raw = np.load(sample['depth_path'])
        if len(depth_raw.shape) == 3:
            depth_raw = depth_raw[:, :, 0]
        
        # Depth 전처리
        depth = depth_raw * sample['scale_factor']
        depth_valid = depth[(depth > DEPTH_MIN) & (depth < DEPTH_MAX)]
        if len(depth_valid) > 0:
            depth_min = depth_valid.min()
            depth_max = depth_valid.max()
            depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-6)
        else:
            depth_normalized = depth / (DEPTH_MAX + 1e-6)
        
        depth_normalized = np.clip(depth_normalized, 0, 1).astype(np.float32)
        depth_pil = Image.fromarray((depth_normalized * 255).astype(np.uint8))
        depth_pil = depth_pil.resize((224, 224), Image.BILINEAR)
        depth_tensor = torch.tensor(np.array(depth_pil) / 255.0, dtype=torch.float32).unsqueeze(0)
        
        # GT 포즈
        gt_pos = np.array(sample['gt_position'], dtype=np.float32)
        gt_rot_6d = np.array(sample['gt_rotation_6d'], dtype=np.float32)
        
        # 학습 시: 노이즈가 추가된 초기 포즈 생성
        if self.split == 'train':
            init_pos, init_rot_6d = self._add_pose_noise(gt_pos, gt_rot_6d)
        else:
            # 평가 시: GT에 고정 노이즈 (더 작은 노이즈)
            init_pos = gt_pos + np.random.RandomState(idx).normal(0, 0.03, 3).astype(np.float32)
            init_rot_6d = gt_rot_6d.copy()
        
        return {
            'rgb': rgb,
            'depth': depth_tensor,
            'gt_position': torch.tensor(gt_pos),
            'gt_rotation_6d': torch.tensor(gt_rot_6d),
            'init_position': torch.tensor(init_pos),
            'init_rotation_6d': torch.tensor(init_rot_6d),
            'class_name': sample['class_name'],
            'class_idx': sample['class_idx'],
        }


# ==========================================
# DeepIM 학습기
# ==========================================
class DeepIMTrainer:
    """DeepIM 학습 및 추론 클래스"""
    
    def __init__(self, renderer, refiner, device, num_iterations=4):
        """
        Args:
            renderer: MeshRenderer 인스턴스
            refiner: DeepIMRefiner 인스턴스
            device: torch device
            num_iterations: 반복 정제 횟수
        """
        self.renderer = renderer
        self.refiner = refiner
        self.device = device
        self.num_iterations = num_iterations
    
    def refine_pose(self, real_rgb, real_depth, class_names, init_positions, init_rotations):
        """포즈 반복 정제
        
        Args:
            real_rgb: (N, 3, H, W) 실제 RGB 이미지
            real_depth: (N, 1, H, W) 실제 Depth 이미지
            class_names: 클래스 이름 리스트
            init_positions: (N, 3) 초기 위치
            init_rotations: (N, 6) 초기 회전 (6D)
        
        Returns:
            final_positions: (N, 3) 최종 위치
            final_rotations: (N, 6) 최종 회전
        """
        positions = init_positions.clone()
        rotations = init_rotations.clone()
        
        for i in range(self.num_iterations):
            # 현재 포즈로 렌더링
            rotation_matrices = rotation_6d_to_matrix(rotations)
            rendered_rgbs, rendered_depths = self.renderer.render_batch(
                class_names, positions, rotation_matrices
            )
            
            # 렌더링된 이미지 정규화
            rendered_rgbs = self._normalize_rendered_rgb(rendered_rgbs)
            rendered_depths = self._normalize_depth(rendered_depths)
            
            # Refiner로 잔차 예측
            delta_pos, delta_rot = self.refiner(
                real_rgb, rendered_rgbs, real_depth, rendered_depths
            )
            
            # 포즈 업데이트
            positions = positions + delta_pos
            rotations = compose_rotation_6d(rotations, delta_rot)
        
        return positions, rotations
    
    def _normalize_rendered_rgb(self, rgb):
        """렌더링된 RGB 정규화 (ImageNet 통계)"""
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (rgb - mean) / std
    
    def _normalize_depth(self, depth):
        """Depth 정규화 (0-1)"""
        # 배치별로 정규화
        batch_size = depth.size(0)
        for i in range(batch_size):
            d = depth[i]
            valid = d > 0
            if valid.sum() > 0:
                d_min = d[valid].min()
                d_max = d[valid].max()
                depth[i] = torch.where(valid, (d - d_min) / (d_max - d_min + 1e-6), d)
        return depth


# ==========================================
# 학습 함수
# ==========================================
def train_deepim(args):
    """DeepIM 모델 학습"""
    
    # 로깅 설정
    log_path = setup_logging("11_deepim_refinement")
    
    total_start_time = time.time()
    
    print("=" * 80)
    print("🎯 DeepIM (Deep Iterative Matching) 학습")
    print("=" * 80)
    
    # 디바이스 설정
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("⚠️ CPU 모드")
    else:
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🚀 GPU 사용: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # 렌더러 초기화
    if not PYTORCH3D_AVAILABLE:
        print("❌ PyTorch3D가 설치되어 있지 않습니다.")
        print("   pip install 'git+https://github.com/facebookresearch/pytorch3d.git@stable'")
        finish_logging()
        sys.exit(1)
    
    # 클래스 이름 (데이터셋에서 자동 감지)
    class_dirs = sorted(glob.glob(os.path.join(args.dataset_dir, "*")))
    class_dirs = [d for d in class_dirs if os.path.isdir(d) and not d.endswith('__pycache__')]
    class_names = [os.path.basename(d) for d in class_dirs]
    
    print(f"\n📦 3D 모델 렌더러 초기화...")
    renderer = MeshRenderer(args.assets_dir, class_names, image_size=224, device=device)
    
    # 데이터셋 로드
    print(f"\n📂 데이터셋 로드: {args.dataset_dir}")
    train_dataset = DeepIMDataset(
        args.dataset_dir, split='train',
        noise_position=args.noise_position,
        noise_rotation=args.noise_rotation
    )
    test_dataset = DeepIMDataset(
        args.dataset_dir, split='test',
        position_stats=train_dataset.position_stats,
        noise_position=args.noise_position * 0.5,  # 테스트는 더 작은 노이즈
        noise_rotation=args.noise_rotation * 0.5
    )
    
    if len(train_dataset) == 0:
        print("❌ 데이터셋이 비어있습니다.")
        finish_logging()
        sys.exit(1)
    
    # 배치 사이즈
    if args.batch_size is None:
        batch_size = get_optimal_batch_size(len(train_dataset), args.cpu)
    else:
        batch_size = args.batch_size
    print(f"📦 배치 사이즈: {batch_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    
    # Refiner 모델
    print(f"\n🏗️ DeepIM Refiner 모델 생성...")
    refiner = DeepIMRefiner(in_channels=8).to(device)
    print(f"   파라미터 수: {sum(p.numel() for p in refiner.parameters()):,}")
    
    # 손실 함수 및 옵티마이저
    criterion = DeepIMLoss(lambda_pos=1.0, lambda_rot=0.5)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # AMP
    use_amp = device.type == 'cuda'
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        print("✅ Mixed Precision Training 활성화")
    
    # DeepIM Trainer
    trainer = DeepIMTrainer(renderer, refiner, device, num_iterations=args.num_iterations)
    
    # 학습
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    best_pos_error = float('inf')
    best_rot_error = float('inf')
    patience_counter = 0
    
    print(f"\n{'='*80}")
    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    print(f"Epochs: {args.epochs}, Iterations: {args.num_iterations}, LR: {args.lr}")
    print(f"Position Noise: {args.noise_position}m, Rotation Noise: {args.noise_rotation}°")
    print(f"{'='*80}\n")
    
    for epoch in range(args.epochs):
        # Training
        refiner.train()
        train_losses = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1:03d} Train", leave=False):
            real_rgb = batch['rgb'].to(device)
            real_depth = batch['depth'].to(device)
            gt_pos = batch['gt_position'].to(device)
            gt_rot = batch['gt_rotation_6d'].to(device)
            init_pos = batch['init_position'].to(device)
            init_rot = batch['init_rotation_6d'].to(device)
            class_names_batch = batch['class_name']
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    # 포즈 정제
                    pred_pos, pred_rot = trainer.refine_pose(
                        real_rgb, real_depth, class_names_batch, init_pos, init_rot
                    )
                    
                    # 손실 계산
                    loss, loss_dict = criterion(pred_pos, pred_rot, gt_pos, gt_rot)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_pos, pred_rot = trainer.refine_pose(
                    real_rgb, real_depth, class_names_batch, init_pos, init_rot
                )
                loss, loss_dict = criterion(pred_pos, pred_rot, gt_pos, gt_rot)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
                optimizer.step()
            
            train_losses.append(loss_dict)
        
        # 평균 학습 손실
        avg_train_loss = np.mean([l['total'] for l in train_losses])
        
        # Evaluation
        refiner.eval()
        all_pos_errors = []
        all_rot_errors = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1:03d} Eval", leave=False):
                real_rgb = batch['rgb'].to(device)
                real_depth = batch['depth'].to(device)
                gt_pos = batch['gt_position'].to(device)
                gt_rot = batch['gt_rotation_6d'].to(device)
                init_pos = batch['init_position'].to(device)
                init_rot = batch['init_rotation_6d'].to(device)
                class_names_batch = batch['class_name']
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        pred_pos, pred_rot = trainer.refine_pose(
                            real_rgb, real_depth, class_names_batch, init_pos, init_rot
                        )
                else:
                    pred_pos, pred_rot = trainer.refine_pose(
                        real_rgb, real_depth, class_names_batch, init_pos, init_rot
                    )
                
                # 오차 계산
                pos_err, rot_err = compute_pose_error(pred_pos, pred_rot, gt_pos, gt_rot)
                all_pos_errors.extend(pos_err.cpu().numpy())
                all_rot_errors.extend(rot_err.cpu().numpy())
        
        avg_pos_error = np.mean(all_pos_errors)
        avg_rot_error = np.mean(all_rot_errors)
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Best 모델 저장
        if avg_pos_error < best_pos_error:
            best_pos_error = avg_pos_error
            best_rot_error = avg_rot_error
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': refiner.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_pos_error': best_pos_error,
                'best_rot_error': best_rot_error,
                'class_names': class_names,
                'position_stats': train_dataset.position_stats,
                'num_iterations': args.num_iterations,
            }, os.path.join(ARTIFACTS_DIR, 'deepim_refiner_best.pt'))
        else:
            patience_counter += 1
        
        # 로그
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:03d}/{args.epochs}] "
                  f"Loss={avg_train_loss:.4f} | "
                  f"PosErr={avg_pos_error:.1f}mm | "
                  f"RotErr={avg_rot_error:.2f}° | "
                  f"LR={current_lr:.2e} | "
                  f"best={best_pos_error:.1f}mm")
        
        # Early stopping
        if patience_counter >= 30:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break
    
    # 결과 요약
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print(f"🎉 DeepIM 학습 완료!")
    print(f"   최고 위치 오차: {best_pos_error:.2f}mm")
    print(f"   최고 자세 오차: {best_rot_error:.2f}°")
    print(f"   총 학습 시간: {total_time/60:.1f}분")
    print(f"   모델 저장: {os.path.join(ARTIFACTS_DIR, 'deepim_refiner_best.pt')}")
    print(f"{'='*80}")
    
    # 오차 분포
    print(f"\n위치 오차 분포:")
    print(f"  < 5mm:  {100 * sum(1 for e in all_pos_errors if e < 5) / len(all_pos_errors):.1f}%")
    print(f"  < 10mm: {100 * sum(1 for e in all_pos_errors if e < 10) / len(all_pos_errors):.1f}%")
    print(f"  < 25mm: {100 * sum(1 for e in all_pos_errors if e < 25) / len(all_pos_errors):.1f}%")
    print(f"  < 50mm: {100 * sum(1 for e in all_pos_errors if e < 50) / len(all_pos_errors):.1f}%")
    
    finish_logging()


# ==========================================
# 메인
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepIM (Deep Iterative Matching) 학습")
    parser.add_argument('--mode', type=str, default='train', choices=['train'])
    parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR)
    parser.add_argument('--assets_dir', type=str, default=ASSETS_DIR)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--num_iterations', type=int, default=NUM_ITERATIONS)
    parser.add_argument('--noise_position', type=float, default=0.05,
                        help='초기 포즈 위치 노이즈 (meters)')
    parser.add_argument('--noise_rotation', type=float, default=3.0,
                        help='초기 포즈 회전 노이즈 (degrees)')
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_deepim(args)

