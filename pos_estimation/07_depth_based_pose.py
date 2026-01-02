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
import time
import random
import multiprocessing
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights
from PIL import Image

# ==========================================
# 설정
# ==========================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, REPO_DIR)
from utils.logger import setup_logging, finish_logging

# 로그 설정
LOG_PATH = setup_logging("07_depth_pose_v1")

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset_pos_depth")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")

# 학습 설정
BATCH_SIZE = None  # None: GPU 메모리에 따라 자동 조정
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.8
FORCE_CPU = False  # CPU 강제 실행 여부
USE_BBOX_CROP = False  # bbox_2d로 ROI crop 사용 여부


# ==========================================
# GPU 최적화 함수들
# ==========================================
def get_optimal_batch_size(data_size, force_cpu=False):
    """GPU 메모리에 따라 최적 배치 사이즈 결정"""
    # 기본 배치 사이즈 결정 (ResNet50은 메모리 사용량이 더 큼)
    if data_size < 100:
        batch_size = 8
    elif data_size < 500:
        batch_size = 16
    elif data_size < 2000:
        batch_size = 32
    else:
        batch_size = 64
    
    # GPU 메모리 고려 (대용량 GPU 지원) - ResNet50 기준
    if torch.cuda.is_available() and not force_cpu:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb < 6:
            batch_size = min(batch_size, 8)
        elif gpu_memory_gb < 12:
            batch_size = min(batch_size, 16)
        elif gpu_memory_gb < 24:
            batch_size = max(batch_size, 32)
        elif gpu_memory_gb < 48:
            batch_size = max(batch_size, 64)
        else:
            # 48GB 이상 대용량 GPU (GB10 128GB 등)
            batch_size = max(batch_size, 128)
    
    return batch_size


def get_optimal_num_workers(force_cpu=False):
    """최적의 num_workers 값 계산"""
    cpu_count = multiprocessing.cpu_count()
    
    if force_cpu:
        return max(1, cpu_count // 2)
    
    # GPU 모드: CPU 코어의 1/4 ~ 8개 사이
    optimal = min(max(4, cpu_count // 4), 8)
    return optimal

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

# 자세 예측 설정
USE_POSE_ESTIMATION = True  # 자세(회전) 예측 사용 여부


# ==========================================
# 6D Rotation 표현 함수들 (Zhou et al. 2019)
# ==========================================
def euler_to_rotation_matrix(roll, pitch, yaw, degrees=True):
    """Euler angles (XYZ 순서) → 3x3 회전 행렬
    
    Args:
        roll, pitch, yaw: Euler angles
        degrees: True면 degree, False면 radian
    """
    if degrees:
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
    
    # 개별 회전 행렬
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
    
    # XYZ 순서: R = Rz @ Ry @ Rx
    R = Rz @ Ry @ Rx
    return R.astype(np.float32)


def rotation_matrix_to_6d(R):
    """3x3 회전 행렬 → 6D 연속 표현
    
    6D 표현: 회전 행렬의 첫 두 열 (column vectors)
    [r1, r2] where R = [r1, r2, r3]
    """
    return np.concatenate([R[:, 0], R[:, 1]], axis=0).astype(np.float32)


def rotation_6d_to_matrix(rot_6d):
    """6D 표현 → 3x3 회전 행렬 (Gram-Schmidt 정규화)
    
    Args:
        rot_6d: (6,) or (N, 6) 6D rotation representation
    """
    if isinstance(rot_6d, torch.Tensor):
        # PyTorch 버전
        if rot_6d.dim() == 1:
            rot_6d = rot_6d.unsqueeze(0)
        
        a1 = rot_6d[:, :3]
        a2 = rot_6d[:, 3:6]
        
        # Gram-Schmidt 정규화
        b1 = F.normalize(a1, dim=1)
        b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        
        # 회전 행렬 조립
        R = torch.stack([b1, b2, b3], dim=2)  # (N, 3, 3)
        return R.squeeze(0) if R.size(0) == 1 else R
    else:
        # NumPy 버전
        if rot_6d.ndim == 1:
            rot_6d = rot_6d.reshape(1, 6)
        
        a1 = rot_6d[:, :3]
        a2 = rot_6d[:, 3:6]
        
        # Gram-Schmidt 정규화
        b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-8)
        b2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
        b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
        b3 = np.cross(b1, b2, axis=1)
        
        R = np.stack([b1, b2, b3], axis=2)
        return R.squeeze(0) if R.shape[0] == 1 else R


def rotation_matrix_to_euler(R, degrees=True):
    """3x3 회전 행렬 → Euler angles (XYZ 순서)
    
    Returns:
        roll, pitch, yaw
    """
    if isinstance(R, torch.Tensor):
        R = R.cpu().numpy()
    
    # XYZ 순서 Euler 추출
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
    """두 6D 회전 표현 간의 각도 오차 (degrees)
    
    Args:
        pred_6d: (6,) 예측된 6D rotation
        gt_6d: (6,) Ground Truth 6D rotation
    
    Returns:
        float: 각도 오차 (degrees)
    """
    # NumPy로 변환하여 계산 (안정성을 위해)
    if isinstance(pred_6d, torch.Tensor):
        pred_6d = pred_6d.detach().cpu().numpy()
    if isinstance(gt_6d, torch.Tensor):
        gt_6d = gt_6d.detach().cpu().numpy()
    
    # 6D → 회전 행렬
    R_pred = rotation_6d_to_matrix(pred_6d)
    R_gt = rotation_6d_to_matrix(gt_6d)
    
    # 상대 회전: R_rel = R_pred^T @ R_gt
    R_rel = R_pred.T @ R_gt
    
    # trace로 각도 계산: theta = arccos((trace(R) - 1) / 2)
    trace = np.trace(R_rel)
    cos_theta = np.clip((trace - 1) / 2, -1, 1)
    theta = np.arccos(cos_theta)
    
    return np.degrees(theta)

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
    """Depth에서 Ground Truth를 계산하는 데이터셋
    
    Args:
        lazy_loading: True면 파일 경로만 저장하고 __getitem__에서 Depth GT 계산 (평가용)
                      False면 미리 모든 Depth GT 계산 (학습용)
    """
    
    def __init__(self, dataset_dir, split='train', train_ratio=TRAIN_RATIO, 
                 use_augmentation=True, position_stats=None, use_bbox_crop=False,
                 lazy_loading=False):
        self.dataset_dir = dataset_dir
        self.split = split
        self.samples = []
        self.use_augmentation = use_augmentation and (split == 'train')
        self.position_stats = position_stats
        self.use_bbox_crop = use_bbox_crop  # bbox_2d ROI crop 사용 여부
        self.lazy_loading = lazy_loading  # Lazy loading 모드 (평가용)
        
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
        
        # 클래스별 스케일 팩터 저장
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
            # Lazy Loading: 파일 경로만 수집 (빠름)
            self._collect_samples_lazy(class_dirs, train_ratio)
        else:
            # 기존 방식: Depth GT 미리 계산 (학습용)
            self._collect_samples_precompute(class_dirs, train_ratio)
        
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
    
    def _collect_samples_lazy(self, class_dirs, train_ratio):
        """Lazy Loading: 파일 경로만 수집 (평가용, 빠름)"""
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
        
        # Train/Test 분할
        random.seed(42)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        
        if self.split == 'train':
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
        
        print(f"  총 {len(all_samples)}개 중 {len(self.samples)}개 ({self.split})")
    
    def _collect_samples_precompute(self, class_dirs, train_ratio):
        """기존 방식: Depth GT 미리 계산 (학습용)"""
        print("Depth에서 Ground Truth 계산 중...")
        all_samples = []
        valid_count = 0
        invalid_count = 0
        
        for class_dir in class_dirs:
            class_name = os.path.basename(class_dir)
            scale_factor = self.class_scale_factors[class_name]
            
            # 스케일 정보 출력
            depth_files_sample = sorted(glob.glob(os.path.join(class_dir, "distance_to_camera_*.npy")))
            if depth_files_sample:
                sample_depth = np.load(depth_files_sample[0])
                if len(sample_depth.shape) == 3:
                    sample_depth = sample_depth[:, :, 0]
                valid_depth = sample_depth[(sample_depth > 0.001) & np.isfinite(sample_depth)]
                depth_mean = valid_depth.mean() if len(valid_depth) > 0 else 0
                print(f"  {class_name}: depth_mean={depth_mean:.3f}m → scale={scale_factor}")
            
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
                
                # Depth에서 GT 계산 (스케일 적용)
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
                    
                    # 자세 정보 로드
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
        
        # Train/Test 분할
        random.seed(42)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        
        if self.split == 'train':
            self.samples = all_samples[:split_idx]
        else:
            self.samples = all_samples[split_idx:]
        
        # 위치 통계 계산 (train에서만)
        if self.split == 'train' and self.position_stats is None:
            self._compute_position_stats()
    
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
    
    def _get_object_bbox(self, bbox_path):
        """bbox_2d 파일에서 물체(semanticId != 0)의 bbox 추출"""
        try:
            bbox_data = np.load(bbox_path, allow_pickle=True)
            # semanticId != 0 인 첫 번째 물체 찾기
            for bbox in bbox_data:
                if bbox['semanticId'] != 0:  # 배경이 아닌 물체
                    x_min = int(bbox['x_min'])
                    y_min = int(bbox['y_min'])
                    x_max = int(bbox['x_max'])
                    y_max = int(bbox['y_max'])
                    # 유효한 bbox인지 확인
                    if x_max > x_min and y_max > y_min:
                        return (x_min, y_min, x_max, y_max)
            return None
        except Exception as e:
            print(f"bbox 로드 실패: {bbox_path}, {e}")
            return None
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # RGB 로드
        rgb = Image.open(sample['rgb_path']).convert('RGB')
        
        # Depth 로드 및 전처리
        depth_raw = np.load(sample['depth_path'])
        if len(depth_raw.shape) == 3:
            depth_raw = depth_raw[:, :, 0]
        
        # Lazy Loading 모드: Depth GT 계산
        if self.lazy_loading:
            scale_factor = sample.get('scale_factor', 1.0)
            centroid, valid = compute_object_centroid_from_depth(
                depth_raw, self.fx, self.fy, self.cx, self.cy, scale_factor
            )
            gt_pos = centroid if valid else np.array([0, 0, 0])
            
            # 자세 정보 로드
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
            # Precomputed 모드: 저장된 GT 사용
            gt_pos = np.array(sample['gt_position'], dtype=np.float32)
            gt_rotation_6d = np.array(sample.get('gt_rotation_6d', [1, 0, 0, 0, 1, 0]), dtype=np.float32)
            gt_euler_deg = np.array(sample.get('gt_euler_deg', [0, 0, 0]), dtype=np.float32)
        
        # bbox crop 적용 (옵션)
        depth = depth_raw.copy()
        bbox = None
        if self.use_bbox_crop and 'bbox_path' in sample:
            bbox = self._get_object_bbox(sample['bbox_path'])
        
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            # 이미지 크기 내로 클리핑
            img_width, img_height = rgb.size
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_max)
            y_max = min(img_height, y_max)
            
            # RGB crop
            rgb = rgb.crop((x_min, y_min, x_max, y_max))
            # Depth crop
            depth = depth[y_min:y_max, x_min:x_max]
        
        # RGB transform 적용 (crop 후)
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
        
        # 정규화
        gt_pos = np.array(gt_pos, dtype=np.float32)
        if self.position_stats is not None:
            mean = np.array(self.position_stats['mean'], dtype=np.float32)
            std = np.array(self.position_stats['std'], dtype=np.float32) + 1e-6
            position_normalized = (gt_pos - mean) / std
        else:
            position_normalized = gt_pos
        
        # numpy → float32 변환 보장
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
    """RGB + Depth 융합 → 6DoF Pose 예측 모델
    
    RGB: 시각적 특징 (물체 인식, 형태) - ResNet50 사용
    Depth: 거리 정보 (Z 좌표에 직접적 기여)
    
    출력:
    - position: 3D 위치 (X, Y, Z)
    - rotation: 6D 회전 표현 (연속적 표현, Euler로 변환 가능)
    - class_logits: 클래스 분류
    """
    
    def __init__(self, num_classes=4, depth_features=256, use_rotation=True):
        super().__init__()
        self.use_rotation = use_rotation
        
        # RGB Encoder (ResNet50 고정) - 최신 PyTorch API 사용
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        rgb_out = 2048
        
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
        if use_rotation:
            self.rotation_head = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 6)  # 6D rotation representation
            )
        
        # Classification Head
        self.class_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        rotation_str = "+ 6D Rotation" if use_rotation else ""
        print(f"  모델: ResNet50 + DepthEncoder → 3D Position {rotation_str}")
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
        
        # Rotation 예측 (옵션)
        if self.use_rotation:
            rotation = self.rotation_head(fused)
            result['rotation'] = rotation
        
        return result


# ==========================================
# 학습
# ==========================================
def train_model(dataset_dir=DATASET_DIR, force_cpu=FORCE_CPU, use_bbox_crop=USE_BBOX_CROP):
    """모델 학습 (GPU 최적화 버전)"""
    
    # 전체 학습 시간 측정 시작
    total_start_time = time.time()
    
    # GPU 사용 불가 시 에러와 함께 종료
    if not force_cpu and not torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("❌ [오류] CUDA 사용 불가!")
        print("=" * 80)
        print("\nGPU를 사용할 수 없습니다. 가능한 원인:")
        print("  1. PyTorch가 CPU 버전으로 설치됨")
        print("  2. NVIDIA 드라이버가 설치되지 않음")
        print("  3. CUDA 버전 불일치")
        print("\n해결 방법:")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        print("\n  또는 CPU로 실행하려면:")
        print("  python 07_depth_based_pose.py --cpu")
        print("=" * 80)
        sys.exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')
    
    # GPU 정보 출력
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n🚀 GPU 사용: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print(f"\n⚠️ CPU 모드로 실행 (느림)")
    
    # Mixed Precision Training 설정
    use_amp = device.type == 'cuda'
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        print("✅ Mixed Precision Training (AMP) 활성화")
    else:
        scaler = None
        print("❌ Mixed Precision Training 비활성화 (CPU 모드)")
    
    # 데이터셋 로드
    if use_bbox_crop:
        print("📦 bbox_2d ROI crop 모드 활성화")
    train_dataset = DepthGTDataset(dataset_dir, split='train', use_augmentation=True, use_bbox_crop=use_bbox_crop)
    test_dataset = DepthGTDataset(
        dataset_dir, split='test', 
        use_augmentation=False,
        position_stats=train_dataset.position_stats,
        use_bbox_crop=use_bbox_crop
    )
    
    if len(train_dataset) == 0:
        print("데이터셋이 비어있습니다.")
        return
    
    # 배치 사이즈 자동 조정
    if BATCH_SIZE is None:
        batch_size = get_optimal_batch_size(len(train_dataset), force_cpu)
        print(f"📦 배치 사이즈 자동 설정: {batch_size}")
    else:
        batch_size = BATCH_SIZE
        print(f"📦 배치 사이즈 고정: {batch_size}")
    
    # num_workers 자동 조정
    num_workers = get_optimal_num_workers(force_cpu)
    pin_memory = device.type == 'cuda'
    print(f"⚙️  DataLoader num_workers: {num_workers}, pin_memory: {pin_memory}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    position_stats = train_dataset.position_stats
    
    # 모델 생성 (RGB + Depth 융합, ResNet50 사용)
    num_classes = len(train_dataset.class_names)
    use_rotation = USE_POSE_ESTIMATION
    model = RGBDepthTo3DModel(num_classes=num_classes, depth_features=256, use_rotation=use_rotation).to(device)
    
    # 손실 함수 및 옵티마이저
    position_criterion = nn.SmoothL1Loss()
    rotation_criterion = nn.SmoothL1Loss()  # 6D rotation에 대한 MSE/SmoothL1
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
    print(f"Epochs: {NUM_EPOCHS}, Batch: {batch_size}, LR: {LEARNING_RATE}")
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
            rotation_6d = batch['rotation_6d'].to(device) if use_rotation else None
            class_idx = batch['class_idx'].to(device)
            
            optimizer.zero_grad()
            
            # Mixed Precision Training
            if use_amp:
                with torch.amp.autocast('cuda'):
                    pred = model(rgb, depth)  # RGB + Depth 입력
                    pos_loss = position_criterion(pred['position'], position)
                    cls_loss = class_criterion(pred['class_logits'], class_idx)
                    
                    # Rotation loss (6D 표현)
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
                pred = model(rgb, depth)  # RGB + Depth 입력
                pos_loss = position_criterion(pred['position'], position)
                cls_loss = class_criterion(pred['class_logits'], class_idx)
                
                # Rotation loss (6D 표현)
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
        all_rot_errors = []  # 자세 오류 (degrees)
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                rgb = batch['rgb'].to(device)
                depth = batch['depth'].to(device)
                position_raw = batch['position_raw'].to(device)
                rotation_6d_gt = batch['rotation_6d'].to(device) if use_rotation else None
                class_idx = batch['class_idx'].to(device)
                
                # Mixed Precision 평가
                if use_amp:
                    with torch.amp.autocast('cuda'):
                        pred = model(rgb, depth)  # RGB + Depth 입력
                else:
                    pred = model(rgb, depth)  # RGB + Depth 입력
                
                # 역정규화
                mean = torch.tensor(position_stats['mean'], device=device)
                std = torch.tensor(position_stats['std'], device=device) + 1e-6
                pred_pos_raw = pred['position'] * std + mean
                
                # 위치 오차 (mm)
                pos_error = torch.sqrt(((pred_pos_raw - position_raw) ** 2).sum(dim=1)) * 1000
                all_pos_errors.extend(pos_error.cpu().numpy())
                
                # 자세 오차 (degrees)
                if use_rotation and 'rotation' in pred:
                    for i in range(pred['rotation'].size(0)):
                        rot_err = compute_rotation_error(
                            pred['rotation'][i].float().cpu(),  # float()로 변환 (AMP Half → Float)
                            rotation_6d_gt[i].float().cpu()
                        )
                        all_rot_errors.append(rot_err.item() if isinstance(rot_err, torch.Tensor) else rot_err)
                
                # 분류 정확도
                _, pred_labels = torch.max(pred['class_logits'], 1)
                correct += (pred_labels == class_idx).sum().item()
                total += class_idx.size(0)
        
        avg_pos_error = np.mean(all_pos_errors)
        avg_rot_error = np.mean(all_rot_errors) if all_rot_errors else 0.0
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
                'best_rot_error': best_rot_error,
                'use_rotation': use_rotation
            }, os.path.join(ARTIFACTS_DIR, 'depth_gt_pose_best.pt'))
        else:
            patience_counter += 1
        
        # 로그
        if (epoch + 1) % 5 == 0 or epoch == 0:
            rot_str = f"RotErr={avg_rot_error:.1f}° | " if use_rotation else ""
            print(f"Epoch [{epoch+1:03d}/{NUM_EPOCHS}] "
                  f"Loss={train_loss:.4f} | "
                  f"PosErr={avg_pos_error:.1f}mm | "
                  f"{rot_str}"
                  f"ClassAcc={class_acc:.1f}% | "
                  f"best={best_pos_error:.1f}mm")
        
        # Early Stopping
        if patience_counter >= early_stop_patience:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
    
    # 전체 학습 시간 계산
    total_time = time.time() - total_start_time
    
    print(f"\n{'='*70}")
    print(f"🎉 학습 완료!")
    print(f"   최고 위치 오차: {best_pos_error:.2f}mm")
    if use_rotation:
        print(f"   최고 자세 오차: {best_rot_error:.2f}°")
    print(f"   총 학습 시간: {total_time/60:.1f}분 ({total_time/3600:.2f}시간)")
    print(f"   모델 저장: {os.path.join(ARTIFACTS_DIR, 'depth_gt_pose_best.pt')}")
    
    # 오차 분포
    print(f"\n위치 오차 분포:")
    print(f"  < 10mm: {100 * sum(1 for e in all_pos_errors if e < 10) / len(all_pos_errors):.1f}%")
    print(f"  < 50mm: {100 * sum(1 for e in all_pos_errors if e < 50) / len(all_pos_errors):.1f}%")
    print(f"  < 100mm: {100 * sum(1 for e in all_pos_errors if e < 100) / len(all_pos_errors):.1f}%")
    print(f"  < 200mm: {100 * sum(1 for e in all_pos_errors if e < 200) / len(all_pos_errors):.1f}%")
    
    if use_rotation and all_rot_errors:
        print(f"\n자세 오차 분포:")
        print(f"  < 5°:  {100 * sum(1 for e in all_rot_errors if e < 5) / len(all_rot_errors):.1f}%")
        print(f"  < 10°: {100 * sum(1 for e in all_rot_errors if e < 10) / len(all_rot_errors):.1f}%")
        print(f"  < 15°: {100 * sum(1 for e in all_rot_errors if e < 15) / len(all_rot_errors):.1f}%")
        print(f"  < 30°: {100 * sum(1 for e in all_rot_errors if e < 30) / len(all_rot_errors):.1f}%")


def evaluate_model(dataset_dir=DATASET_DIR, use_bbox_crop=False, num_samples=None):
    """학습된 모델의 위치 및 자세 추정 정확도 평가
    
    Args:
        dataset_dir: 데이터셋 경로
        use_bbox_crop: bbox crop 사용 여부
        num_samples: 평가할 샘플 수 (None: 전체)
    """
    print(f"\n{'='*70}")
    print("📊 학습된 모델 평가 (위치 + 자세)")
    print(f"{'='*70}")
    
    # 모델 로드
    model_path = os.path.join(ARTIFACTS_DIR, 'depth_gt_pose_best.pt')
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        print("   먼저 학습을 실행하세요: python 07_depth_based_pose.py --mode train")
        return
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    position_stats = checkpoint['position_stats']
    class_names = checkpoint['class_names']
    use_rotation = checkpoint.get('use_rotation', True)
    
    print(f"✅ 모델 로드: {model_path}")
    print(f"   클래스: {class_names}")
    print(f"   자세 예측: {'활성화' if use_rotation else '비활성화'}")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   디바이스: {device}")
    
    # 모델 생성 및 가중치 로드
    num_classes = len(class_names)
    model = RGBDepthTo3DModel(num_classes=num_classes, depth_features=256, use_rotation=use_rotation)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 테스트 데이터셋 로드 (Lazy Loading으로 빠르게)
    if use_bbox_crop:
        print("📦 bbox_2d ROI crop 모드")
    
    # Lazy Loading: 파일 경로만 수집하고, 실제 접근 시 Depth GT 계산
    test_dataset = DepthGTDataset(
        dataset_dir, split='test', 
        use_augmentation=False,
        position_stats=position_stats,
        use_bbox_crop=use_bbox_crop,
        lazy_loading=True  # 평가용 Lazy Loading (빠름)
    )
    
    if len(test_dataset) == 0:
        print("❌ 테스트 데이터셋이 비어있습니다.")
        return
    
    # 샘플 수 제한
    if num_samples is not None:
        num_samples = min(num_samples, len(test_dataset))
    else:
        num_samples = len(test_dataset)
    
    print(f"\n평가 샘플 수: {num_samples} / {len(test_dataset)}")
    print(f"{'='*70}")
    
    # 평가
    all_pos_errors = []
    all_rot_errors = []
    all_x_errors = []
    all_y_errors = []
    all_z_errors = []
    class_correct = 0
    class_total = 0
    
    # 클래스별 오차 저장
    class_pos_errors = {name: [] for name in class_names}
    class_rot_errors = {name: [] for name in class_names}
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="평가 중"):
            sample = test_dataset[i]
            
            rgb = sample['rgb'].unsqueeze(0).to(device)
            depth = sample['depth'].unsqueeze(0).to(device)
            gt_pos_raw = sample['position_raw'].numpy()
            gt_rot_6d = sample['rotation_6d'].numpy() if use_rotation else None
            gt_euler = sample['euler_deg'].numpy() if use_rotation else None
            class_idx = sample['class_idx']
            class_name = class_names[class_idx]
            
            # 예측
            pred = model(rgb, depth)
            
            # 위치 역정규화
            mean = np.array(position_stats['mean'])
            std = np.array(position_stats['std']) + 1e-6
            pred_pos_raw = pred['position'].cpu().numpy()[0] * std + mean
            
            # 위치 오차 (mm)
            pos_error = np.sqrt(np.sum((pred_pos_raw - gt_pos_raw) ** 2)) * 1000
            x_error = abs(pred_pos_raw[0] - gt_pos_raw[0]) * 1000
            y_error = abs(pred_pos_raw[1] - gt_pos_raw[1]) * 1000
            z_error = abs(pred_pos_raw[2] - gt_pos_raw[2]) * 1000
            
            all_pos_errors.append(pos_error)
            all_x_errors.append(x_error)
            all_y_errors.append(y_error)
            all_z_errors.append(z_error)
            class_pos_errors[class_name].append(pos_error)
            
            # 자세 오차 (degrees)
            if use_rotation and 'rotation' in pred:
                pred_rot_6d = pred['rotation'].float().cpu().numpy()[0]
                rot_error = compute_rotation_error(pred_rot_6d, gt_rot_6d)
                all_rot_errors.append(rot_error)
                class_rot_errors[class_name].append(rot_error)
                
                # Euler로 변환하여 출력 (처음 5개 샘플만)
                if i < 5:
                    pred_R = rotation_6d_to_matrix(pred_rot_6d)
                    pred_euler = rotation_matrix_to_euler(pred_R, degrees=True)
                    print(f"\n샘플 {i+1} ({class_name}):")
                    print(f"  위치 GT:    [{gt_pos_raw[0]:.3f}, {gt_pos_raw[1]:.3f}, {gt_pos_raw[2]:.3f}]")
                    print(f"  위치 예측:  [{pred_pos_raw[0]:.3f}, {pred_pos_raw[1]:.3f}, {pred_pos_raw[2]:.3f}]")
                    print(f"  위치 오차:  {pos_error:.1f}mm (X:{x_error:.1f}, Y:{y_error:.1f}, Z:{z_error:.1f})")
                    print(f"  자세 GT:    [R:{gt_euler[0]:.1f}°, P:{gt_euler[1]:.1f}°, Y:{gt_euler[2]:.1f}°]")
                    print(f"  자세 예측:  [R:{pred_euler[0]:.1f}°, P:{pred_euler[1]:.1f}°, Y:{pred_euler[2]:.1f}°]")
                    print(f"  자세 오차:  {rot_error:.2f}°")
            
            # 분류 정확도
            pred_class = torch.argmax(pred['class_logits'], dim=1).item()
            if pred_class == class_idx:
                class_correct += 1
            class_total += 1
    
    # 결과 요약
    print(f"\n{'='*70}")
    print("📈 평가 결과 요약")
    print(f"{'='*70}")
    
    avg_pos_error = np.mean(all_pos_errors)
    std_pos_error = np.std(all_pos_errors)
    median_pos_error = np.median(all_pos_errors)
    
    print(f"\n📍 위치 오차:")
    print(f"   평균: {avg_pos_error:.2f}mm (±{std_pos_error:.2f}mm)")
    print(f"   중앙값: {median_pos_error:.2f}mm")
    print(f"   최소: {np.min(all_pos_errors):.2f}mm")
    print(f"   최대: {np.max(all_pos_errors):.2f}mm")
    print(f"\n   축별 오차:")
    print(f"   X: {np.mean(all_x_errors):.2f}mm (±{np.std(all_x_errors):.2f}mm)")
    print(f"   Y: {np.mean(all_y_errors):.2f}mm (±{np.std(all_y_errors):.2f}mm)")
    print(f"   Z: {np.mean(all_z_errors):.2f}mm (±{np.std(all_z_errors):.2f}mm)")
    
    print(f"\n   분포:")
    print(f"   < 10mm:  {100 * sum(1 for e in all_pos_errors if e < 10) / len(all_pos_errors):.1f}%")
    print(f"   < 25mm:  {100 * sum(1 for e in all_pos_errors if e < 25) / len(all_pos_errors):.1f}%")
    print(f"   < 50mm:  {100 * sum(1 for e in all_pos_errors if e < 50) / len(all_pos_errors):.1f}%")
    print(f"   < 100mm: {100 * sum(1 for e in all_pos_errors if e < 100) / len(all_pos_errors):.1f}%")
    
    if use_rotation and all_rot_errors:
        avg_rot_error = np.mean(all_rot_errors)
        std_rot_error = np.std(all_rot_errors)
        median_rot_error = np.median(all_rot_errors)
        
        print(f"\n🔄 자세 오차:")
        print(f"   평균: {avg_rot_error:.2f}° (±{std_rot_error:.2f}°)")
        print(f"   중앙값: {median_rot_error:.2f}°")
        print(f"   최소: {np.min(all_rot_errors):.2f}°")
        print(f"   최대: {np.max(all_rot_errors):.2f}°")
        
        print(f"\n   분포:")
        print(f"   < 2°:  {100 * sum(1 for e in all_rot_errors if e < 2) / len(all_rot_errors):.1f}%")
        print(f"   < 5°:  {100 * sum(1 for e in all_rot_errors if e < 5) / len(all_rot_errors):.1f}%")
        print(f"   < 10°: {100 * sum(1 for e in all_rot_errors if e < 10) / len(all_rot_errors):.1f}%")
        print(f"   < 15°: {100 * sum(1 for e in all_rot_errors if e < 15) / len(all_rot_errors):.1f}%")
    
    print(f"\n🏷️ 분류 정확도: {100 * class_correct / class_total:.1f}%")
    
    # 클래스별 결과
    print(f"\n{'='*70}")
    print("📊 클래스별 결과")
    print(f"{'='*70}")
    for name in class_names:
        if class_pos_errors[name]:
            pos_mean = np.mean(class_pos_errors[name])
            pos_std = np.std(class_pos_errors[name])
            if use_rotation and class_rot_errors[name]:
                rot_mean = np.mean(class_rot_errors[name])
                print(f"  {name}: 위치 {pos_mean:.1f}mm (±{pos_std:.1f}), 자세 {rot_mean:.2f}°")
            else:
                print(f"  {name}: 위치 {pos_mean:.1f}mm (±{pos_std:.1f})")
    
    print(f"\n{'='*70}")
    print("✅ 평가 완료")
    print(f"{'='*70}")


# ==========================================
# 메인
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Depth 기반 6DoF Pose Estimation (ResNet50)")
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'verify'],
                        help='실행 모드')
    parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=None,
                        help='배치 크기 (None: GPU 메모리에 따라 자동 조정)')
    parser.add_argument('--cpu', action='store_true',
                        help='CPU 강제 실행 (GPU 사용 불가 시)')
    parser.add_argument('--bbox_crop', action='store_true',
                        help='bbox_2d로 ROI crop하여 학습')
    
    args = parser.parse_args()
    
    # 전역 설정 업데이트
    NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    FORCE_CPU = args.cpu
    USE_BBOX_CROP = args.bbox_crop
    
    if args.mode == 'train':
        train_model(args.dataset_dir, force_cpu=args.cpu, use_bbox_crop=args.bbox_crop)
    elif args.mode in ['evaluate', 'eval']:
        evaluate_model(args.dataset_dir, use_bbox_crop=args.bbox_crop, num_samples=args.num_samples)
    
    # 로깅 종료
    finish_logging()

