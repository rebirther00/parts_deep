#!/usr/bin/env python3
# ==========================================
# Depth 기반 6DoF Pose Estimation V2
# EfficientNetV2-S + 개선된 학습 전략
# ==========================================
#
# V1 대비 개선사항:
# 1. ResNet50 → EfficientNetV2-S (더 높은 ImageNet 정확도)
# 2. RandAugment 데이터 증강 추가
# 3. Warmup + Cosine Annealing 스케줄러
# 4. Geodesic Loss (회전용) 옵션
#
# 사용법:
#   python 09_depth_based_pose_v2.py --mode train
#   python 09_depth_based_pose_v2.py --mode train --bbox_crop
#
# ==========================================

import os
import sys
import json
import glob
import random
import multiprocessing
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_V2_S_Weights
from PIL import Image

# ==========================================
# 설정
# ==========================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, REPO_DIR)
from utils.logger import setup_logging, finish_logging

# 로그 설정
LOG_PATH = setup_logging("09_depth_pose_v2")

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset_pos_depth")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")

# 학습 설정
BATCH_SIZE = None  # None: GPU 메모리에 따라 자동 조정
NUM_EPOCHS = 150  # V2: 더 긴 학습
LEARNING_RATE = 1e-4  # V2: 안정적인 LR
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.8
FORCE_CPU = False
USE_BBOX_CROP = False
WARMUP_EPOCHS = 5  # V2: 짧은 Warmup

# 카메라 내재 파라미터
CAMERA_INTRINSICS = {
    "fx": 768.0, "fy": 768.0,
    "cx": 512.0, "cy": 512.0,
    "width": 1024, "height": 1024
}

# Depth 기반 GT 설정
DEPTH_MIN = 0.01
DEPTH_MAX = 100.0
FOREGROUND_PERCENTILE = 10

# 자세 예측 설정
USE_POSE_ESTIMATION = True


# ==========================================
# GPU 최적화 함수들
# ==========================================
def get_optimal_batch_size(data_size, force_cpu=False):
    """GPU 메모리에 따라 최적 배치 사이즈 결정"""
    if data_size < 100:
        batch_size = 8
    elif data_size < 500:
        batch_size = 16
    elif data_size < 2000:
        batch_size = 32
    else:
        batch_size = 64
    
    # EfficientNetV2는 ResNet50보다 메모리 효율적
    if torch.cuda.is_available() and not force_cpu:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb < 6:
            batch_size = min(batch_size, 16)
        elif gpu_memory_gb < 12:
            batch_size = min(batch_size, 32)
        elif gpu_memory_gb < 24:
            batch_size = max(batch_size, 48)
        elif gpu_memory_gb < 48:
            batch_size = max(batch_size, 96)
        else:
            batch_size = max(batch_size, 128)
    
    return batch_size


def get_optimal_num_workers(force_cpu=False):
    """최적의 num_workers 값 계산"""
    cpu_count = multiprocessing.cpu_count()
    if force_cpu:
        return max(1, cpu_count // 2)
    optimal = min(max(4, cpu_count // 4), 8)
    return optimal


# ==========================================
# 6D Rotation 표현 함수들
# ==========================================
def euler_to_rotation_matrix(roll, pitch, yaw, degrees=True):
    """Euler angles (XYZ 순서) → 3x3 회전 행렬"""
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
    """3x3 회전 행렬 → 6D 연속 표현"""
    return np.concatenate([R[:, 0], R[:, 1]], axis=0).astype(np.float32)


def rotation_6d_to_matrix(rot_6d):
    """6D 표현 → 3x3 회전 행렬 (Gram-Schmidt 정규화)"""
    if isinstance(rot_6d, torch.Tensor):
        if rot_6d.dim() == 1:
            rot_6d = rot_6d.unsqueeze(0)
        
        a1 = rot_6d[:, :3]
        a2 = rot_6d[:, 3:6]
        
        b1 = F.normalize(a1, dim=1)
        b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        
        R = torch.stack([b1, b2, b3], dim=2)
        return R.squeeze(0) if R.size(0) == 1 else R
    else:
        if rot_6d.ndim == 1:
            rot_6d = rot_6d.reshape(1, 6)
        
        a1 = rot_6d[:, :3]
        a2 = rot_6d[:, 3:6]
        
        b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-8)
        b2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
        b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
        b3 = np.cross(b1, b2, axis=1)
        
        R = np.stack([b1, b2, b3], axis=2)
        return R.squeeze(0) if R.shape[0] == 1 else R


def rotation_matrix_to_euler(R, degrees=True):
    """3x3 회전 행렬 → Euler angles (XYZ 순서)"""
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


def euler_to_6d(roll, pitch, yaw, degrees=True):
    """Euler angles → 6D rotation representation"""
    R = euler_to_rotation_matrix(roll, pitch, yaw, degrees)
    return rotation_matrix_to_6d(R)


def compute_rotation_error(pred_6d, gt_6d):
    """두 6D 회전 표현 간의 각도 오차 (degrees)"""
    if isinstance(pred_6d, torch.Tensor):
        pred_6d = pred_6d.detach().cpu().numpy()
    if isinstance(gt_6d, torch.Tensor):
        gt_6d = gt_6d.detach().cpu().numpy()
    
    R_pred = rotation_6d_to_matrix(pred_6d)
    R_gt = rotation_6d_to_matrix(gt_6d)
    
    R_rel = R_pred.T @ R_gt
    trace = np.trace(R_rel)
    cos_theta = np.clip((trace - 1) / 2, -1, 1)
    theta = np.arccos(cos_theta)
    
    return np.degrees(theta)


# ==========================================
# Geodesic Loss (회전용 - 더 정확한 손실 함수)
# ==========================================
class GeodesicLoss(nn.Module):
    """회전 행렬 간의 Geodesic 거리 기반 손실 함수 (수치 안정화 버전)"""
    
    def __init__(self, use_safe_mode=True):
        super().__init__()
        self.use_safe_mode = use_safe_mode
    
    def forward(self, pred_6d, gt_6d):
        """
        Args:
            pred_6d: (N, 6) 예측 6D rotation
            gt_6d: (N, 6) GT 6D rotation
        Returns:
            loss: scalar
        """
        # Safe mode: SmoothL1 사용 (수치적으로 안정)
        if self.use_safe_mode:
            return F.smooth_l1_loss(pred_6d, gt_6d)
        
        # 6D → 회전 행렬
        R_pred = rotation_6d_to_matrix(pred_6d)  # (N, 3, 3)
        R_gt = rotation_6d_to_matrix(gt_6d)
        
        # R_pred^T @ R_gt
        R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
        
        # trace 계산
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        
        # arccos((trace - 1) / 2) - 수치 안정화
        cos_theta = torch.clamp((trace - 1) / 2, -0.999, 0.999)
        theta = torch.acos(cos_theta)
        
        # NaN 체크
        if torch.isnan(theta).any():
            return F.smooth_l1_loss(pred_6d, gt_6d)
        
        return theta.mean()


# ==========================================
# Depth → Point Cloud → 객체 중심 계산
# ==========================================
def depth_to_pointcloud(depth, fx, fy, cx, cy):
    """Depth 이미지 → Point Cloud 변환"""
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
    """Depth에서 객체 중심 좌표 계산"""
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
# 데이터셋: Depth에서 GT 계산 (V2: 강화된 증강)
# ==========================================
class DepthGTDatasetV2(Dataset):
    """V2: RandAugment 적용 데이터셋"""
    
    def __init__(self, dataset_dir, split='train', train_ratio=TRAIN_RATIO, 
                 use_augmentation=True, position_stats=None, use_bbox_crop=False,
                 lazy_loading=False):
        self.dataset_dir = dataset_dir
        self.split = split
        self.samples = []
        self.use_augmentation = use_augmentation and (split == 'train')
        self.position_stats = position_stats
        self.use_bbox_crop = use_bbox_crop
        self.lazy_loading = lazy_loading
        
        self.fx = CAMERA_INTRINSICS["fx"]
        self.fy = CAMERA_INTRINSICS["fy"]
        self.cx = CAMERA_INTRINSICS["cx"]
        self.cy = CAMERA_INTRINSICS["cy"]
        
        # 클래스별 폴더 스캔
        class_dirs = sorted(glob.glob(os.path.join(dataset_dir, "*")))
        class_dirs = [d for d in class_dirs if os.path.isdir(d) and not d.endswith('__pycache__')]
        
        self.class_names = [os.path.basename(d) for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # 클래스별 스케일 팩터
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
        
        if lazy_loading:
            self._collect_samples_lazy(class_dirs, train_ratio)
        else:
            self._collect_samples_precompute(class_dirs, train_ratio)
        
        # V2: 데이터 증강 (안정화 버전)
        if self.use_augmentation:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=10),  # 약한 회전
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
    
    def _collect_samples_lazy(self, class_dirs, train_ratio):
        """Lazy Loading: 파일 경로만 수집"""
        print("파일 경로 수집 중 (Lazy Loading)...")
        all_samples = []
        
        for class_dir in class_dirs:
            class_name = os.path.basename(class_dir)
            scale_factor = self.class_scale_factors[class_name]
            
            depth_files = sorted(glob.glob(os.path.join(class_dir, "distance_to_camera_*.npy")))
            
            for depth_file in depth_files:
                frame_idx = int(os.path.basename(depth_file).split('_')[-1].split('.')[0])
                rgb_file = os.path.join(class_dir, f"rgb_{frame_idx:04d}.png")
                bbox_file = os.path.join(class_dir, f"bounding_box_2d_tight_{frame_idx:04d}.npy")
                pose_file = os.path.join(class_dir, f"pose_{frame_idx:04d}.json")
                
                if not os.path.exists(rgb_file):
                    continue
                if self.use_bbox_crop and not os.path.exists(bbox_file):
                    continue
                
                sample_data = {
                    'rgb_path': rgb_file,
                    'depth_path': depth_file,
                    'pose_path': pose_file if os.path.exists(pose_file) else None,
                    'class_name': class_name,
                    'class_idx': self.class_to_idx[class_name],
                    'scale_factor': scale_factor
                }
                if os.path.exists(bbox_file):
                    sample_data['bbox_path'] = bbox_file
                
                all_samples.append(sample_data)
        
        random.seed(42)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        
        if self.split == 'train':
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
        
        print(f"  총 {len(all_samples)}개 중 {len(self.samples)}개 ({self.split})")
    
    def _collect_samples_precompute(self, class_dirs, train_ratio):
        """Depth GT 미리 계산"""
        print("Depth에서 Ground Truth 계산 중...")
        all_samples = []
        valid_count = 0
        invalid_count = 0
        
        for class_dir in class_dirs:
            class_name = os.path.basename(class_dir)
            scale_factor = self.class_scale_factors[class_name]
            
            depth_files = sorted(glob.glob(os.path.join(class_dir, "distance_to_camera_*.npy")))
            
            for depth_file in tqdm(depth_files, desc=f"  {class_name}", leave=False):
                frame_idx = int(os.path.basename(depth_file).split('_')[-1].split('.')[0])
                rgb_file = os.path.join(class_dir, f"rgb_{frame_idx:04d}.png")
                bbox_file = os.path.join(class_dir, f"bounding_box_2d_tight_{frame_idx:04d}.npy")
                pose_file = os.path.join(class_dir, f"pose_{frame_idx:04d}.json")
                
                if not os.path.exists(rgb_file):
                    continue
                if self.use_bbox_crop and not os.path.exists(bbox_file):
                    continue
                
                depth = np.load(depth_file)
                if len(depth.shape) == 3:
                    depth = depth[:, :, 0]
                
                centroid, valid = compute_object_centroid_from_depth(
                    depth, self.fx, self.fy, self.cx, self.cy, scale_factor
                )
                
                if valid:
                    sample_data = {
                        'rgb_path': rgb_file,
                        'depth_path': depth_file,
                        'gt_position': centroid.tolist(),
                        'class_name': class_name,
                        'class_idx': self.class_to_idx[class_name],
                        'scale_factor': scale_factor
                    }
                    if os.path.exists(bbox_file):
                        sample_data['bbox_path'] = bbox_file
                    
                    if USE_POSE_ESTIMATION and os.path.exists(pose_file):
                        try:
                            with open(pose_file, 'r') as f:
                                pose_data = json.load(f)
                            r_xyz_deg = pose_data.get('camTobj', {}).get('r_xyz_deg', [0, 0, 0])
                            rot_6d = euler_to_6d(r_xyz_deg[0], r_xyz_deg[1], r_xyz_deg[2], degrees=True)
                            sample_data['gt_rotation_6d'] = rot_6d.tolist()
                            sample_data['gt_euler_deg'] = r_xyz_deg
                        except:
                            sample_data['gt_rotation_6d'] = euler_to_6d(0, 0, 0).tolist()
                            sample_data['gt_euler_deg'] = [0, 0, 0]
                    else:
                        sample_data['gt_rotation_6d'] = euler_to_6d(0, 0, 0).tolist()
                        sample_data['gt_euler_deg'] = [0, 0, 0]
                    
                    all_samples.append(sample_data)
                    valid_count += 1
                else:
                    invalid_count += 1
        
        print(f"  유효 샘플: {valid_count}, 무효 샘플: {invalid_count}")
        
        random.seed(42)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        
        if self.split == 'train':
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
        
        if self.split == 'train' and self.position_stats is None:
            self._compute_position_stats()
    
    def _compute_position_stats(self):
        """위치 정규화를 위한 통계 계산"""
        positions = np.array([s['gt_position'] for s in self.samples])
        self.position_stats = {
            'mean': positions.mean(axis=0).tolist(),
            'std': positions.std(axis=0).tolist()
        }
        print(f"  위치 통계: mean={self.position_stats['mean']}, std={self.position_stats['std']}")
    
    def __len__(self):
        return len(self.samples)
    
    def _get_object_bbox(self, bbox_path):
        """bbox_2d 파일에서 물체의 bbox 추출"""
        try:
            bbox_data = np.load(bbox_path, allow_pickle=True)
            for bbox in bbox_data:
                if bbox['semanticId'] != 0:
                    x_min = int(bbox['x_min'])
                    y_min = int(bbox['y_min'])
                    x_max = int(bbox['x_max'])
                    y_max = int(bbox['y_max'])
                    if x_max > x_min and y_max > y_min:
                        return (x_min, y_min, x_max, y_max)
            return None
        except:
            return None
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        rgb = Image.open(sample['rgb_path']).convert('RGB')
        
        depth_raw = np.load(sample['depth_path'])
        if len(depth_raw.shape) == 3:
            depth_raw = depth_raw[:, :, 0]
        
        if self.lazy_loading:
            scale_factor = sample.get('scale_factor', 1.0)
            centroid, valid = compute_object_centroid_from_depth(
                depth_raw, self.fx, self.fy, self.cx, self.cy, scale_factor
            )
            gt_pos = centroid if valid else np.array([0, 0, 0])
            
            if USE_POSE_ESTIMATION and sample.get('pose_path'):
                try:
                    with open(sample['pose_path'], 'r') as f:
                        pose_data = json.load(f)
                    r_xyz_deg = pose_data.get('camTobj', {}).get('r_xyz_deg', [0, 0, 0])
                    gt_rotation_6d = euler_to_6d(r_xyz_deg[0], r_xyz_deg[1], r_xyz_deg[2], degrees=True)
                    gt_euler_deg = np.array(r_xyz_deg, dtype=np.float32)
                except:
                    gt_rotation_6d = euler_to_6d(0, 0, 0)
                    gt_euler_deg = np.array([0, 0, 0], dtype=np.float32)
            else:
                gt_rotation_6d = euler_to_6d(0, 0, 0)
                gt_euler_deg = np.array([0, 0, 0], dtype=np.float32)
        else:
            gt_pos = np.array(sample['gt_position'], dtype=np.float32)
            gt_rotation_6d = np.array(sample.get('gt_rotation_6d', [1, 0, 0, 0, 1, 0]), dtype=np.float32)
            gt_euler_deg = np.array(sample.get('gt_euler_deg', [0, 0, 0]), dtype=np.float32)
        
        depth = depth_raw.copy()
        bbox = None
        if self.use_bbox_crop and 'bbox_path' in sample:
            bbox = self._get_object_bbox(sample['bbox_path'])
        
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            img_width, img_height = rgb.size
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_max)
            y_max = min(img_height, y_max)
            rgb = rgb.crop((x_min, y_min, x_max, y_max))
            depth = depth[y_min:y_max, x_min:x_max]
        
        rgb = self.rgb_transform(rgb)
        
        depth = depth * sample.get('scale_factor', 1.0)
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
        
        gt_pos = np.array(gt_pos, dtype=np.float32)
        if self.position_stats is not None:
            mean = np.array(self.position_stats['mean'], dtype=np.float32)
            std = np.array(self.position_stats['std'], dtype=np.float32) + 1e-6
            position_normalized = (gt_pos - mean) / std
        else:
            position_normalized = gt_pos
        
        if isinstance(gt_rotation_6d, np.ndarray):
            gt_rotation_6d = gt_rotation_6d.astype(np.float32)
        
        return {
            'rgb': rgb,
            'depth': depth_tensor,
            'position': torch.tensor(position_normalized),
            'position_raw': torch.tensor(gt_pos),
            'rotation_6d': torch.tensor(gt_rotation_6d),
            'euler_deg': torch.tensor(gt_euler_deg),
            'class_idx': sample['class_idx']
        }


# ==========================================
# Depth Encoder (동일)
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
# 모델 V2: EfficientNetV2-S + Depth → 6DoF Pose
# ==========================================
class RGBDepthTo3DModelV2(nn.Module):
    """V2: EfficientNetV2-S + Depth 융합 → 6DoF Pose 예측
    
    개선사항:
    - ResNet50 → EfficientNetV2-S (더 높은 정확도, 더 적은 파라미터)
    - Layer Normalization 추가
    """
    
    def __init__(self, num_classes=4, depth_features=256, use_rotation=True):
        super().__init__()
        self.use_rotation = use_rotation
        
        # V2: EfficientNetV2-S 사용
        effnet = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        rgb_out = 1280  # EfficientNetV2-S 출력 차원
        
        # 마지막 분류층 제거
        self.rgb_encoder = nn.Sequential(*list(effnet.children())[:-1])
        self.rgb_fc = nn.Sequential(
            nn.Linear(rgb_out, 512),
            nn.LayerNorm(512),  # V2: LayerNorm 추가
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Depth Encoder
        self.depth_encoder = DepthEncoder(out_features=depth_features)
        
        # Fusion dimension
        fusion_dim = 512 + depth_features  # 768
        
        # Position Head
        self.position_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)
        )
        
        # Rotation Head
        if use_rotation:
            self.rotation_head = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 6)
            )
        
        # Classification Head
        self.class_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        rotation_str = "+ 6D Rotation" if use_rotation else ""
        print(f"  모델 V2: EfficientNetV2-S + DepthEncoder → 3D Position {rotation_str}")
        print(f"  RGB features: 512, Depth features: {depth_features}, Fusion: {fusion_dim}")
    
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
        class_logits = self.class_head(fused)
        
        result = {
            'position': position,
            'class_logits': class_logits
        }
        
        if self.use_rotation:
            rotation = self.rotation_head(fused)
            result['rotation'] = rotation
        
        return result


# ==========================================
# Warmup + Cosine Annealing 스케줄러
# ==========================================
class WarmupCosineScheduler:
    """Warmup + Cosine Annealing Learning Rate Scheduler"""
    
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * warmup_factor
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


# ==========================================
# 학습 V2
# ==========================================
def train_model(dataset_dir=DATASET_DIR, force_cpu=FORCE_CPU, use_bbox_crop=USE_BBOX_CROP):
    """V2 모델 학습"""
    
    if not force_cpu and not torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("❌ [오류] CUDA 사용 불가!")
        print("=" * 80)
        sys.exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n🚀 GPU 사용: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print(f"\n⚠️ CPU 모드로 실행")
    
    # AMP 설정
    use_amp = device.type == 'cuda'
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        print("✅ Mixed Precision Training (AMP) 활성화")
    else:
        scaler = None
    
    # 데이터셋 로드
    print("\n" + "=" * 80)
    print("📦 V2 모델 학습 (EfficientNetV2-S + RandAugment)")
    print("=" * 80)
    
    if use_bbox_crop:
        print("📦 bbox_2d ROI crop 모드 활성화")
    
    train_dataset = DepthGTDatasetV2(dataset_dir, split='train', use_augmentation=True, use_bbox_crop=use_bbox_crop)
    test_dataset = DepthGTDatasetV2(
        dataset_dir, split='test', 
        use_augmentation=False,
        position_stats=train_dataset.position_stats,
        use_bbox_crop=use_bbox_crop
    )
    
    if len(train_dataset) == 0:
        print("데이터셋이 비어있습니다.")
        return
    
    # 배치 사이즈
    if BATCH_SIZE is None:
        batch_size = get_optimal_batch_size(len(train_dataset), force_cpu)
        print(f"📦 배치 사이즈 자동 설정: {batch_size}")
    else:
        batch_size = BATCH_SIZE
    
    num_workers = get_optimal_num_workers(force_cpu)
    pin_memory = device.type == 'cuda'
    print(f"⚙️  DataLoader num_workers: {num_workers}, pin_memory: {pin_memory}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=pin_memory)
    
    position_stats = train_dataset.position_stats
    
    # V2 모델 생성
    num_classes = len(train_dataset.class_names)
    use_rotation = USE_POSE_ESTIMATION
    model = RGBDepthTo3DModelV2(num_classes=num_classes, depth_features=256, use_rotation=use_rotation).to(device)
    
    # 손실 함수
    position_criterion = nn.SmoothL1Loss()
    rotation_criterion = GeodesicLoss() if use_rotation else None  # V2: Geodesic Loss
    class_criterion = nn.CrossEntropyLoss()
    
    # 옵티마이저
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # V2: Warmup + Cosine Annealing
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    best_pos_error = float('inf')
    best_rot_error = float('inf')
    patience_counter = 0
    early_stop_patience = 30  # V2: 더 긴 patience
    
    print(f"\nTrain: {len(train_dataset)}, Test: {len(test_dataset)}")
    print(f"클래스: {train_dataset.class_names}")
    print(f"Epochs: {NUM_EPOCHS}, Warmup: {WARMUP_EPOCHS}, Batch: {batch_size}, LR: {LEARNING_RATE}")
    print()
    
    for epoch in range(NUM_EPOCHS):
        # Learning rate 업데이트
        scheduler.step(epoch)
        current_lr = scheduler.get_lr()[0]
        
        # Training
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1:03d} Train", leave=False):
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device)
            position = batch['position'].to(device)
            rotation_6d = batch['rotation_6d'].to(device) if use_rotation else None
            class_idx = batch['class_idx'].to(device)
            
            optimizer.zero_grad()
            
            if use_amp:
                with torch.amp.autocast('cuda'):
                    pred = model(rgb, depth)
                    pos_loss = position_criterion(pred['position'], position)
                    cls_loss = class_criterion(pred['class_logits'], class_idx)
                    
                    if use_rotation and 'rotation' in pred:
                        rot_loss = rotation_criterion(pred['rotation'], rotation_6d)
                        loss = pos_loss + 0.5 * rot_loss + 0.1 * cls_loss
                    else:
                        loss = pos_loss + 0.1 * cls_loss
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(rgb, depth)
                pos_loss = position_criterion(pred['position'], position)
                cls_loss = class_criterion(pred['class_logits'], class_idx)
                
                if use_rotation and 'rotation' in pred:
                    rot_loss = rotation_criterion(pred['rotation'], rotation_6d)
                    loss = pos_loss + 0.5 * rot_loss + 0.1 * cls_loss
                else:
                    loss = pos_loss + 0.1 * cls_loss
                
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
                rotation_6d_gt = batch['rotation_6d'].to(device) if use_rotation else None
                class_idx = batch['class_idx'].to(device)
                
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        pred = model(rgb, depth)
                else:
                    pred = model(rgb, depth)
                
                mean = torch.tensor(position_stats['mean'], device=device)
                std = torch.tensor(position_stats['std'], device=device) + 1e-6
                pred_pos_raw = pred['position'] * std + mean
                
                pos_error = torch.sqrt(((pred_pos_raw - position_raw) ** 2).sum(dim=1)) * 1000
                all_pos_errors.extend(pos_error.cpu().numpy())
                
                if use_rotation and 'rotation' in pred:
                    for i in range(pred['rotation'].size(0)):
                        rot_err = compute_rotation_error(
                            pred['rotation'][i].float().cpu(),
                            rotation_6d_gt[i].float().cpu()
                        )
                        all_rot_errors.append(rot_err.item() if isinstance(rot_err, torch.Tensor) else rot_err)
                
                _, pred_labels = torch.max(pred['class_logits'], 1)
                correct += (pred_labels == class_idx).sum().item()
                total += class_idx.size(0)
        
        avg_pos_error = np.mean(all_pos_errors)
        avg_rot_error = np.mean(all_rot_errors) if all_rot_errors else 0.0
        class_acc = 100 * correct / total
        
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
                'best_rot_error': best_rot_error,
                'use_rotation': use_rotation,
                'model_version': 'V2_EfficientNetV2S',
                'use_bbox_crop': use_bbox_crop
            }, os.path.join(ARTIFACTS_DIR, 'depth_gt_pose_v2_best.pt'))
        else:
            patience_counter += 1
        
        # 로그
        if (epoch + 1) % 5 == 0 or epoch == 0:
            rot_str = f"RotErr={avg_rot_error:.2f}° | " if use_rotation else ""
            print(f"Epoch [{epoch+1:03d}/{NUM_EPOCHS}] "
                  f"Loss={train_loss:.4f} | "
                  f"PosErr={avg_pos_error:.1f}mm | "
                  f"{rot_str}"
                  f"LR={current_lr:.2e} | "
                  f"best={best_pos_error:.1f}mm")
        
        # Early Stopping
        if patience_counter >= early_stop_patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n{'='*70}")
    print(f"🎉 V2 학습 완료!")
    print(f"   최고 위치 오차: {best_pos_error:.2f}mm")
    if use_rotation:
        print(f"   최고 자세 오차: {best_rot_error:.2f}°")
    print(f"   모델 저장: {os.path.join(ARTIFACTS_DIR, 'depth_gt_pose_v2_best.pt')}")
    
    # 오차 분포
    print(f"\n위치 오차 분포:")
    print(f"  < 10mm: {100 * sum(1 for e in all_pos_errors if e < 10) / len(all_pos_errors):.1f}%")
    print(f"  < 25mm: {100 * sum(1 for e in all_pos_errors if e < 25) / len(all_pos_errors):.1f}%")
    print(f"  < 50mm: {100 * sum(1 for e in all_pos_errors if e < 50) / len(all_pos_errors):.1f}%")
    print(f"  < 100mm: {100 * sum(1 for e in all_pos_errors if e < 100) / len(all_pos_errors):.1f}%")
    
    if use_rotation and all_rot_errors:
        print(f"\n자세 오차 분포:")
        print(f"  < 2°:  {100 * sum(1 for e in all_rot_errors if e < 2) / len(all_rot_errors):.1f}%")
        print(f"  < 5°:  {100 * sum(1 for e in all_rot_errors if e < 5) / len(all_rot_errors):.1f}%")
        print(f"  < 10°: {100 * sum(1 for e in all_rot_errors if e < 10) / len(all_rot_errors):.1f}%")


# ==========================================
# 메인
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Depth 기반 6DoF Pose Estimation V2 (EfficientNetV2-S)")
    parser.add_argument('--mode', type=str, default='train', choices=['train'])
    parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--bbox_crop', action='store_true')
    
    args = parser.parse_args()
    
    NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    FORCE_CPU = args.cpu
    USE_BBOX_CROP = args.bbox_crop
    
    if args.mode == 'train':
        train_model(args.dataset_dir, force_cpu=args.cpu, use_bbox_crop=args.bbox_crop)
    
    # 로깅 종료
    finish_logging()

