#!/usr/bin/env python3
# ==========================================
# DeepIM 평가 스크립트
# ==========================================
#
# 학습된 DeepIM 모델을 사용하여 포즈 추정 정확도를 평가합니다.
#
# 사용법:
#   python 12_deepim_evaluation.py
#   python 12_deepim_evaluation.py --num_samples 100
#   python 12_deepim_evaluation.py --save_results
#   python 12_deepim_evaluation.py --verbose
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

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "deepim_refiner_best.pt")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "deepim_evaluation_results.json")

# 로깅 설정
from utils.logger import setup_logging, finish_logging

# DeepIM 모듈 임포트
from deepim.refiner import DeepIMRefiner
from deepim.loss import rotation_6d_to_matrix, compute_pose_error

try:
    from deepim.renderer import MeshRenderer, PYTORCH3D_AVAILABLE
except ImportError:
    PYTORCH3D_AVAILABLE = False

# ==========================================
# 설정
# ==========================================
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


def rotation_matrix_to_euler(R, degrees=True):
    """3x3 회전 행렬 → Euler angles"""
    if isinstance(R, torch.Tensor):
        R = R.cpu().numpy()
    
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
    
    if degrees:
        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
    return roll, pitch, yaw


def compose_rotation_6d(base_6d, delta_6d):
    """두 6D 회전 표현을 합성"""
    R_base = rotation_6d_to_matrix(base_6d)
    R_delta = rotation_6d_to_matrix(delta_6d)
    
    R_composed = torch.bmm(R_delta, R_base)
    composed_6d = torch.cat([R_composed[:, :, 0], R_composed[:, :, 1]], dim=1)
    
    return composed_6d


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
# 평가용 데이터셋
# ==========================================
class DeepIMEvalDataset(Dataset):
    """DeepIM 평가용 데이터셋"""
    
    def __init__(self, dataset_dir, position_stats, class_names, train_ratio=0.8):
        self.dataset_dir = dataset_dir
        self.samples = []
        self.position_stats = position_stats
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        
        self.fx = CAMERA_INTRINSICS["fx"]
        self.fy = CAMERA_INTRINSICS["fy"]
        self.cx = CAMERA_INTRINSICS["cx"]
        self.cy = CAMERA_INTRINSICS["cy"]
        
        # 스케일 팩터
        self.class_scale_factors = {}
        class_dirs = sorted(glob.glob(os.path.join(dataset_dir, "*")))
        class_dirs = [d for d in class_dirs if os.path.isdir(d) and not d.endswith('__pycache__')]
        
        for class_dir in class_dirs:
            class_name = os.path.basename(class_dir)
            if class_name not in self.class_to_idx:
                continue
            
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
        
        # 테스트 샘플 수집
        self._collect_test_samples(class_dirs, train_ratio)
        
        # Transform
        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _collect_test_samples(self, class_dirs, train_ratio):
        """테스트 샘플만 수집"""
        all_samples = []
        
        for class_dir in class_dirs:
            class_name = os.path.basename(class_dir)
            if class_name not in self.class_to_idx:
                continue
            
            scale_factor = self.class_scale_factors.get(class_name, 1.0)
            depth_files = sorted(glob.glob(os.path.join(class_dir, "distance_to_camera_*.npy")))
            
            for depth_file in depth_files:
                frame_idx = int(os.path.basename(depth_file).split('_')[-1].split('.')[0])
                rgb_file = os.path.join(class_dir, f"rgb_{frame_idx:04d}.png")
                pose_file = os.path.join(class_dir, f"pose_{frame_idx:04d}.json")
                
                if not os.path.exists(rgb_file) or not os.path.exists(pose_file):
                    continue
                
                all_samples.append({
                    'rgb_path': rgb_file,
                    'depth_path': depth_file,
                    'pose_path': pose_file,
                    'class_name': class_name,
                    'class_idx': self.class_to_idx[class_name],
                    'scale_factor': scale_factor
                })
        
        # Train/Test 분할
        random.seed(42)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        
        self.samples = all_samples[split_idx:]
        print(f"  테스트 샘플: {len(self.samples)} / 전체 {len(all_samples)}")
    
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
        
        # Depth GT 계산
        scale_factor = sample['scale_factor']
        centroid, valid = compute_object_centroid_from_depth(
            depth_raw, self.fx, self.fy, self.cx, self.cy, scale_factor
        )
        gt_pos = centroid if valid else np.array([0, 0, 0])
        
        # 자세 정보 로드
        with open(sample['pose_path'], 'r') as f:
            pose_data = json.load(f)
        r_xyz_deg = pose_data.get('camTobj', {}).get('r_xyz_deg', [0, 0, 0])
        gt_rot_6d = euler_to_6d(r_xyz_deg[0], r_xyz_deg[1], r_xyz_deg[2], degrees=True)
        gt_euler_deg = np.array(r_xyz_deg, dtype=np.float32)
        
        # Depth 전처리
        depth = depth_raw * scale_factor
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
        
        return {
            'rgb': rgb,
            'depth': depth_tensor,
            'gt_position': torch.tensor(gt_pos.astype(np.float32)),
            'gt_rotation_6d': torch.tensor(gt_rot_6d),
            'gt_euler_deg': torch.tensor(gt_euler_deg),
            'class_idx': sample['class_idx'],
            'class_name': sample['class_name'],
            'rgb_path': sample['rgb_path']
        }


# ==========================================
# DeepIM 추론기
# ==========================================
class DeepIMInference:
    """DeepIM 추론 클래스"""
    
    def __init__(self, renderer, refiner, initial_model, position_stats, device, num_iterations=4):
        """
        Args:
            renderer: MeshRenderer
            refiner: DeepIMRefiner
            initial_model: 초기 포즈 예측 모델
            position_stats: 위치 정규화 통계
            device: torch device
            num_iterations: 반복 정제 횟수
        """
        self.renderer = renderer
        self.refiner = refiner
        self.initial_model = initial_model
        self.position_stats = position_stats
        self.device = device
        self.num_iterations = num_iterations
    
    def predict(self, rgb, depth, class_names):
        """전체 파이프라인 추론
        
        Args:
            rgb: (N, 3, H, W) RGB 이미지
            depth: (N, 1, H, W) Depth 이미지
            class_names: 클래스 이름 리스트
        
        Returns:
            final_positions: (N, 3) 최종 위치
            final_rotations: (N, 6) 최종 회전
            initial_positions: (N, 3) 초기 위치 (비교용)
            initial_rotations: (N, 6) 초기 회전 (비교용)
        """
        # 1. 초기 포즈 예측
        with torch.no_grad():
            initial_pred = self.initial_model(rgb, depth)
        
        # 위치 역정규화
        mean = torch.tensor(self.position_stats['mean'], device=self.device)
        std = torch.tensor(self.position_stats['std'], device=self.device) + 1e-6
        initial_positions = initial_pred['position'] * std + mean
        initial_rotations = initial_pred['rotation']
        
        # 2. 반복 정제
        positions = initial_positions.clone()
        rotations = initial_rotations.clone()
        
        for i in range(self.num_iterations):
            # 현재 포즈로 렌더링
            rotation_matrices = rotation_6d_to_matrix(rotations)
            rendered_rgbs, rendered_depths = self.renderer.render_batch(
                class_names, positions, rotation_matrices
            )
            
            # 정규화
            rendered_rgbs = self._normalize_rendered_rgb(rendered_rgbs)
            rendered_depths = self._normalize_depth(rendered_depths)
            
            # Refiner
            delta_pos, delta_rot = self.refiner(rgb, rendered_rgbs, depth, rendered_depths)
            
            # 업데이트
            positions = positions + delta_pos
            rotations = compose_rotation_6d(rotations, delta_rot)
        
        return positions, rotations, initial_positions, initial_rotations
    
    def _normalize_rendered_rgb(self, rgb):
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (rgb - mean) / std
    
    def _normalize_depth(self, depth):
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
# 평가 함수
# ==========================================
def evaluate(args):
    """DeepIM 모델 평가"""
    
    # 로깅 설정
    log_path = setup_logging("12_deepim_evaluation")
    
    print("=" * 80)
    print("📊 DeepIM Pose Estimation 평가")
    print("=" * 80)
    
    # DeepIM 모델 로드
    if not os.path.exists(args.model):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {args.model}")
        print("   먼저 학습을 실행하세요: python 11_deepim_refinement.py --mode train")
        finish_logging()
        sys.exit(1)
    
    print(f"\n✅ DeepIM 모델 로드: {args.model}")
    checkpoint = torch.load(args.model, map_location='cpu', weights_only=False)
    
    class_names = checkpoint['class_names']
    position_stats = checkpoint['position_stats']
    num_iterations = checkpoint.get('num_iterations', 4)
    best_pos_error = checkpoint.get('best_pos_error', 0)
    best_rot_error = checkpoint.get('best_rot_error', 0)
    
    print(f"   학습 시 최고 위치 오차: {best_pos_error:.2f}mm")
    print(f"   학습 시 최고 자세 오차: {best_rot_error:.2f}°")
    print(f"   반복 정제 횟수: {num_iterations}")
    print(f"   클래스: {class_names}")
    
    # 디바이스 설정
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        print(f"   디바이스: CPU")
    else:
        device = torch.device('cuda')
        print(f"   디바이스: {torch.cuda.get_device_name(0)}")
    
    # PyTorch3D 확인
    if not PYTORCH3D_AVAILABLE:
        print("❌ PyTorch3D가 설치되어 있지 않습니다.")
        finish_logging()
        sys.exit(1)
    
    # 렌더러 초기화
    print(f"\n📦 렌더러 초기화...")
    renderer = MeshRenderer(args.assets_dir, class_names, image_size=224, device=device)
    
    # Refiner 모델 로드
    refiner = DeepIMRefiner(in_channels=8).to(device)
    refiner.load_state_dict(checkpoint['model_state_dict'])
    refiner.eval()
    
    # 초기 포즈 모델 로드
    print(f"\n📦 초기 포즈 모델 로드...")
    initial_model_path = os.path.join(ARTIFACTS_DIR, 'depth_gt_pose_best.pt')
    if not os.path.exists(initial_model_path):
        print(f"⚠️ 초기 모델을 찾을 수 없습니다: {initial_model_path}")
        print("   먼저 07_depth_based_pose.py로 학습하세요.")
        finish_logging()
        sys.exit(1)
    
    initial_checkpoint = torch.load(initial_model_path, map_location='cpu', weights_only=False)
    initial_model = RGBDepthTo3DModel(
        num_classes=len(class_names),
        depth_features=256,
        use_rotation=True
    ).to(device)
    initial_model.load_state_dict(initial_checkpoint['model_state_dict'])
    initial_model.eval()
    
    # 데이터셋 로드
    print(f"\n📂 데이터셋 로드: {args.dataset_dir}")
    test_dataset = DeepIMEvalDataset(
        args.dataset_dir,
        position_stats=position_stats,
        class_names=class_names
    )
    
    if len(test_dataset) == 0:
        print("❌ 테스트 데이터셋이 비어있습니다.")
        finish_logging()
        sys.exit(1)
    
    # 샘플 수 제한
    if args.num_samples is not None:
        num_samples = min(args.num_samples, len(test_dataset))
    else:
        num_samples = len(test_dataset)
    
    print(f"\n평가 샘플 수: {num_samples} / {len(test_dataset)}")
    print("=" * 80)
    
    # 추론기 생성
    inferencer = DeepIMInference(
        renderer, refiner, initial_model, 
        initial_checkpoint['position_stats'],
        device, num_iterations
    )
    
    # 평가 수행
    all_pos_errors = []  # DeepIM 후 오차
    all_rot_errors = []
    all_init_pos_errors = []  # 초기 모델 오차 (비교용)
    all_init_rot_errors = []
    all_x_errors = []
    all_y_errors = []
    all_z_errors = []
    
    class_pos_errors = {name: [] for name in class_names}
    class_rot_errors = {name: [] for name in class_names}
    
    detailed_results = []
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="평가 중"):
            sample = test_dataset[i]
            
            rgb = sample['rgb'].unsqueeze(0).to(device)
            depth = sample['depth'].unsqueeze(0).to(device)
            gt_pos = sample['gt_position'].to(device)
            gt_rot = sample['gt_rotation_6d'].to(device)
            gt_euler = sample['gt_euler_deg'].numpy()
            class_name = sample['class_name']
            rgb_path = sample['rgb_path']
            
            # 예측
            pred_pos, pred_rot, init_pos, init_rot = inferencer.predict(
                rgb, depth, [class_name]
            )
            
            pred_pos = pred_pos[0]
            pred_rot = pred_rot[0]
            init_pos = init_pos[0]
            init_rot = init_rot[0]
            
            # 오차 계산
            pos_error = torch.sqrt(((pred_pos - gt_pos) ** 2).sum()).item() * 1000
            init_pos_error = torch.sqrt(((init_pos - gt_pos) ** 2).sum()).item() * 1000
            
            x_error = abs(pred_pos[0].item() - gt_pos[0].item()) * 1000
            y_error = abs(pred_pos[1].item() - gt_pos[1].item()) * 1000
            z_error = abs(pred_pos[2].item() - gt_pos[2].item()) * 1000
            
            # 회전 오차
            pos_err_tensor, rot_err_tensor = compute_pose_error(
                pred_pos.unsqueeze(0), pred_rot.unsqueeze(0),
                gt_pos.unsqueeze(0), gt_rot.unsqueeze(0)
            )
            rot_error = rot_err_tensor[0].item()
            
            init_pos_err_tensor, init_rot_err_tensor = compute_pose_error(
                init_pos.unsqueeze(0), init_rot.unsqueeze(0),
                gt_pos.unsqueeze(0), gt_rot.unsqueeze(0)
            )
            init_rot_error = init_rot_err_tensor[0].item()
            
            # 저장
            all_pos_errors.append(pos_error)
            all_rot_errors.append(rot_error)
            all_init_pos_errors.append(init_pos_error)
            all_init_rot_errors.append(init_rot_error)
            all_x_errors.append(x_error)
            all_y_errors.append(y_error)
            all_z_errors.append(z_error)
            
            class_pos_errors[class_name].append(pos_error)
            class_rot_errors[class_name].append(rot_error)
            
            # 상세 결과
            pred_R = rotation_6d_to_matrix(pred_rot.unsqueeze(0))[0].cpu().numpy()
            pred_euler = rotation_matrix_to_euler(pred_R, degrees=True)
            
            result_entry = {
                'sample_idx': i,
                'class_name': class_name,
                'rgb_path': rgb_path,
                'gt_position': [float(v) for v in gt_pos.cpu().numpy()],
                'pred_position': [float(v) for v in pred_pos.cpu().numpy()],
                'init_position': [float(v) for v in init_pos.cpu().numpy()],
                'pos_error_mm': float(pos_error),
                'init_pos_error_mm': float(init_pos_error),
                'improvement_mm': float(init_pos_error - pos_error),
                'gt_euler_deg': [float(v) for v in gt_euler],
                'pred_euler_deg': [float(v) for v in pred_euler],
                'rot_error_deg': float(rot_error),
            }
            detailed_results.append(result_entry)
            
            # 상세 출력
            if args.verbose and i < 5:
                print(f"\n샘플 {i+1} ({class_name}):")
                print(f"  GT 위치:    [{gt_pos[0]:.3f}, {gt_pos[1]:.3f}, {gt_pos[2]:.3f}]")
                print(f"  초기 위치:  [{init_pos[0]:.3f}, {init_pos[1]:.3f}, {init_pos[2]:.3f}] → 오차: {init_pos_error:.1f}mm")
                print(f"  최종 위치:  [{pred_pos[0]:.3f}, {pred_pos[1]:.3f}, {pred_pos[2]:.3f}] → 오차: {pos_error:.1f}mm")
                print(f"  개선:       {init_pos_error - pos_error:.1f}mm ({100*(init_pos_error-pos_error)/init_pos_error:.1f}%)")
                print(f"  자세 오차:  {rot_error:.2f}°")
    
    # 결과 요약
    print(f"\n{'='*80}")
    print("📈 평가 결과 요약")
    print(f"{'='*80}")
    
    avg_pos_error = np.mean(all_pos_errors)
    std_pos_error = np.std(all_pos_errors)
    avg_init_pos_error = np.mean(all_init_pos_errors)
    improvement = avg_init_pos_error - avg_pos_error
    
    print(f"\n📍 위치 오차 (Position Error):")
    print(f"   초기 모델:  {avg_init_pos_error:.2f}mm (±{np.std(all_init_pos_errors):.2f}mm)")
    print(f"   DeepIM 후: {avg_pos_error:.2f}mm (±{std_pos_error:.2f}mm)")
    print(f"   개선:      {improvement:.2f}mm ({100*improvement/avg_init_pos_error:.1f}%)")
    print(f"\n   중앙값: {np.median(all_pos_errors):.2f}mm")
    print(f"   최소: {np.min(all_pos_errors):.2f}mm")
    print(f"   최대: {np.max(all_pos_errors):.2f}mm")
    
    print(f"\n   축별 오차:")
    print(f"   X: {np.mean(all_x_errors):.2f}mm (±{np.std(all_x_errors):.2f}mm)")
    print(f"   Y: {np.mean(all_y_errors):.2f}mm (±{np.std(all_y_errors):.2f}mm)")
    print(f"   Z: {np.mean(all_z_errors):.2f}mm (±{np.std(all_z_errors):.2f}mm)")
    
    print(f"\n   분포:")
    print(f"   < 5mm:  {100 * sum(1 for e in all_pos_errors if e < 5) / len(all_pos_errors):.1f}%")
    print(f"   < 10mm: {100 * sum(1 for e in all_pos_errors if e < 10) / len(all_pos_errors):.1f}%")
    print(f"   < 25mm: {100 * sum(1 for e in all_pos_errors if e < 25) / len(all_pos_errors):.1f}%")
    print(f"   < 50mm: {100 * sum(1 for e in all_pos_errors if e < 50) / len(all_pos_errors):.1f}%")
    
    avg_rot_error = np.mean(all_rot_errors)
    avg_init_rot_error = np.mean(all_init_rot_errors)
    
    print(f"\n🔄 자세 오차 (Rotation Error):")
    print(f"   초기 모델:  {avg_init_rot_error:.2f}° (±{np.std(all_init_rot_errors):.2f}°)")
    print(f"   DeepIM 후: {avg_rot_error:.2f}° (±{np.std(all_rot_errors):.2f}°)")
    
    print(f"\n   분포:")
    print(f"   < 2°:  {100 * sum(1 for e in all_rot_errors if e < 2) / len(all_rot_errors):.1f}%")
    print(f"   < 5°:  {100 * sum(1 for e in all_rot_errors if e < 5) / len(all_rot_errors):.1f}%")
    print(f"   < 10°: {100 * sum(1 for e in all_rot_errors if e < 10) / len(all_rot_errors):.1f}%")
    
    # 클래스별 결과
    print(f"\n{'='*80}")
    print("📊 클래스별 결과")
    print(f"{'='*80}")
    for name in class_names:
        if class_pos_errors[name]:
            pos_mean = np.mean(class_pos_errors[name])
            pos_std = np.std(class_pos_errors[name])
            rot_mean = np.mean(class_rot_errors[name])
            print(f"  {name}: 위치 {pos_mean:.1f}mm (±{pos_std:.1f}), 자세 {rot_mean:.2f}°")
    
    # 결과 저장
    if args.save_results:
        summary_results = {
            'num_samples': num_samples,
            'num_iterations': num_iterations,
            'position_error': {
                'mean_mm': float(avg_pos_error),
                'std_mm': float(std_pos_error),
                'median_mm': float(np.median(all_pos_errors)),
                'min_mm': float(np.min(all_pos_errors)),
                'max_mm': float(np.max(all_pos_errors)),
            },
            'initial_position_error': {
                'mean_mm': float(avg_init_pos_error),
            },
            'improvement_mm': float(improvement),
            'improvement_percent': float(100*improvement/avg_init_pos_error),
            'rotation_error': {
                'mean_deg': float(avg_rot_error),
                'std_deg': float(np.std(all_rot_errors)),
            },
            'class_names': class_names,
        }
        
        with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False)
        print(f"\n💾 결과 저장: {RESULTS_PATH}")
        
        if args.save_detailed:
            detailed_path = os.path.join(ARTIFACTS_DIR, "deepim_evaluation_detailed.json")
            with open(detailed_path, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            print(f"💾 상세 결과 저장: {detailed_path}")
    
    print(f"\n{'='*80}")
    print("✅ 평가 완료")
    print(f"{'='*80}")
    
    finish_logging()


# ==========================================
# 메인
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepIM 모델 평가")
    parser.add_argument('--model', type=str, default=MODEL_PATH,
                        help='DeepIM 모델 경로')
    parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR)
    parser.add_argument('--assets_dir', type=str, default=ASSETS_DIR)
    parser.add_argument('--num_samples', type=int, default=None,
                        help='평가할 샘플 수 (None: 전체)')
    parser.add_argument('--save_results', action='store_true',
                        help='평가 결과를 JSON으로 저장')
    parser.add_argument('--save_detailed', action='store_true',
                        help='샘플별 상세 결과 저장')
    parser.add_argument('--verbose', action='store_true',
                        help='상세 출력 (처음 5개 샘플)')
    parser.add_argument('--cpu', action='store_true')
    
    args = parser.parse_args()
    
    evaluate(args)

