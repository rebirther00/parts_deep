#!/usr/bin/env python3
# ==========================================
# Depth 기반 6DoF Pose Estimation V3 (RTX 5090 최적화)
# ConvNeXt-Small + GELU + Stochastic Depth
# ==========================================
#
# RTX 5090 최적화:
# - 배치 사이즈 64
# - NUM_WORKERS = 8
# - persistent_workers, non_blocking
#
# V2 대비 개선사항:
# 1. EfficientNetV2-S → ConvNeXt-Small (더 최신, 강력한 아키텍처)
# 2. ReLU → GELU (더 부드러운 활성화)
# 3. Stochastic Depth (Drop Path) 적용
#
# 사용법:
#   python 11_depth_based_pose_v3_5090.py --mode train
#   python 11_depth_based_pose_v3_5090.py --mode train --bbox_crop
#
# ==========================================

import os
import sys
import json
import glob
import random
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ConvNeXt_Small_Weights
from PIL import Image

# ==========================================
# 설정
# ==========================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, REPO_DIR)
from utils.logger import setup_logging, finish_logging

LOG_PATH = setup_logging("11_depth_pose_v3_5090")

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset_pos_depth")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")

# RTX 5090 최적화 학습 설정
BATCH_SIZE = 64
NUM_EPOCHS = 150
LEARNING_RATE = 5e-5  # ConvNeXt은 더 낮은 LR 권장
WEIGHT_DECAY = 1e-4
TRAIN_RATIO = 0.8
WARMUP_EPOCHS = 5

# RTX 5090 DataLoader 최적화
NUM_WORKERS = 8
PREFETCH_FACTOR = 2
EARLY_STOP_PATIENCE = 40

FORCE_CPU = False
USE_BBOX_CROP = False

# 카메라/Depth 설정
CAMERA_INTRINSICS = {"fx": 768.0, "fy": 768.0, "cx": 512.0, "cy": 512.0, "width": 1024, "height": 1024}
DEPTH_MIN, DEPTH_MAX = 0.01, 100.0
FOREGROUND_PERCENTILE = 10
USE_POSE_ESTIMATION = True


# ==========================================
# 6D Rotation 함수들
# ==========================================
def euler_to_rotation_matrix(roll, pitch, yaw, degrees=True):
    if degrees:
        roll, pitch, yaw = np.radians(roll), np.radians(pitch), np.radians(yaw)
    Rx = np.array([[1,0,0],[0,np.cos(roll),-np.sin(roll)],[0,np.sin(roll),np.cos(roll)]])
    Ry = np.array([[np.cos(pitch),0,np.sin(pitch)],[0,1,0],[-np.sin(pitch),0,np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    return (Rz @ Ry @ Rx).astype(np.float32)


def rotation_matrix_to_6d(R):
    return np.concatenate([R[:, 0], R[:, 1]], axis=0).astype(np.float32)


def rotation_6d_to_matrix(rot_6d):
    if isinstance(rot_6d, torch.Tensor):
        if rot_6d.dim() == 1:
            rot_6d = rot_6d.unsqueeze(0)
        a1, a2 = rot_6d[:, :3], rot_6d[:, 3:6]
        b1 = F.normalize(a1, dim=1)
        b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        R = torch.stack([b1, b2, b3], dim=2)
        return R.squeeze(0) if R.size(0) == 1 else R
    else:
        if rot_6d.ndim == 1:
            rot_6d = rot_6d.reshape(1, 6)
        a1, a2 = rot_6d[:, :3], rot_6d[:, 3:6]
        b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-8)
        b2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
        b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8)
        b3 = np.cross(b1, b2, axis=1)
        R = np.stack([b1, b2, b3], axis=2)
        return R.squeeze(0) if R.shape[0] == 1 else R


def euler_to_6d(roll, pitch, yaw, degrees=True):
    R = euler_to_rotation_matrix(roll, pitch, yaw, degrees)
    return rotation_matrix_to_6d(R)


def compute_rotation_error(pred_6d, gt_6d):
    if isinstance(pred_6d, torch.Tensor):
        pred_6d = pred_6d.detach().cpu().numpy()
    if isinstance(gt_6d, torch.Tensor):
        gt_6d = gt_6d.detach().cpu().numpy()
    R_pred, R_gt = rotation_6d_to_matrix(pred_6d), rotation_6d_to_matrix(gt_6d)
    R_rel = R_pred.T @ R_gt
    cos_theta = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos_theta))


# ==========================================
# Depth → 객체 중심 계산
# ==========================================
def depth_to_pointcloud(depth, fx, fy, cx, cy):
    h, w = depth.shape
    valid_mask = (depth > DEPTH_MIN) & (depth < DEPTH_MAX) & np.isfinite(depth)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x, y = (u - cx) * z / fx, (v - cy) * z / fy
    points = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=1)
    return points, valid_mask


def compute_object_centroid_from_depth(depth, fx, fy, cx, cy, scale_factor=1.0):
    depth_scaled = depth * scale_factor
    points, _ = depth_to_pointcloud(depth_scaled, fx, fy, cx, cy)
    if len(points) < 100:
        return np.array([0, 0, 0]), False
    z_values = points[:, 2]
    foreground_mask = z_values < np.percentile(z_values, FOREGROUND_PERCENTILE)
    foreground_points = points[foreground_mask]
    if len(foreground_points) < 50:
        return np.array([0, 0, 0]), False
    return foreground_points.mean(axis=0), True


# ==========================================
# Stochastic Depth (Drop Path)
# ==========================================
class DropPath(nn.Module):
    """Stochastic Depth 구현"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# ==========================================
# V3용 Geodesic Loss
# ==========================================
class GeodesicLoss(nn.Module):
    def __init__(self, use_safe_mode=True):
        super().__init__()
        self.use_safe_mode = use_safe_mode
    
    def forward(self, pred_6d, gt_6d):
        if self.use_safe_mode:
            return F.smooth_l1_loss(pred_6d, gt_6d)
        R_pred = rotation_6d_to_matrix(pred_6d)
        R_gt = rotation_6d_to_matrix(gt_6d)
        R_diff = torch.bmm(R_pred.transpose(1, 2), R_gt)
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        cos_theta = torch.clamp((trace - 1) / 2, -0.999, 0.999)
        theta = torch.acos(cos_theta)
        if torch.isnan(theta).any():
            return F.smooth_l1_loss(pred_6d, gt_6d)
        return theta.mean()


# ==========================================
# 데이터셋 V3 (GELU 데이터 증강)
# ==========================================
class DepthGTDatasetV3(Dataset):
    def __init__(self, dataset_dir, split='train', train_ratio=TRAIN_RATIO, 
                 use_augmentation=True, position_stats=None, use_bbox_crop=False, lazy_loading=False):
        self.dataset_dir = dataset_dir
        self.split = split
        self.samples = []
        self.use_augmentation = use_augmentation and (split == 'train')
        self.position_stats = position_stats
        self.use_bbox_crop = use_bbox_crop
        self.lazy_loading = lazy_loading
        
        self.fx, self.fy = CAMERA_INTRINSICS["fx"], CAMERA_INTRINSICS["fy"]
        self.cx, self.cy = CAMERA_INTRINSICS["cx"], CAMERA_INTRINSICS["cy"]
        
        class_dirs = sorted([d for d in glob.glob(os.path.join(dataset_dir, "*")) 
                            if os.path.isdir(d) and not d.endswith('__pycache__')])
        self.class_names = [os.path.basename(d) for d in class_dirs]
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
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
                    scale_factor = 100.0 if depth_mean < 0.5 else (10.0 if depth_mean < 1.0 else 1.0)
                else:
                    scale_factor = 1.0
            else:
                scale_factor = 1.0
            self.class_scale_factors[class_name] = scale_factor
        
        self._collect_samples(class_dirs, train_ratio)
        
        if self.use_augmentation:
            self.rgb_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=10),
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
    
    def _collect_samples(self, class_dirs, train_ratio):
        print("Depth에서 Ground Truth 계산 중...")
        all_samples = []
        valid_count, invalid_count = 0, 0
        
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
                        'rgb_path': rgb_file, 'depth_path': depth_file,
                        'gt_position': centroid.tolist(),
                        'class_name': class_name, 'class_idx': self.class_to_idx[class_name],
                        'scale_factor': scale_factor
                    }
                    if os.path.exists(bbox_file):
                        sample_data['bbox_path'] = bbox_file
                    
                    if USE_POSE_ESTIMATION and os.path.exists(pose_file):
                        try:
                            with open(pose_file, 'r') as f:
                                pose_data = json.load(f)
                            r_xyz_deg = pose_data.get('camTobj', {}).get('r_xyz_deg', [0, 0, 0])
                            sample_data['gt_rotation_6d'] = euler_to_6d(r_xyz_deg[0], r_xyz_deg[1], r_xyz_deg[2]).tolist()
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
            self._compute_position_stats()
        else:
            self.samples = all_samples[split_idx:]
    
    def _compute_position_stats(self):
        positions = np.array([s['gt_position'] for s in self.samples])
        self.position_stats = {'mean': positions.mean(axis=0).tolist(), 'std': positions.std(axis=0).tolist()}
        print(f"  위치 통계: mean={self.position_stats['mean']}, std={self.position_stats['std']}")
    
    def __len__(self):
        return len(self.samples)
    
    def _get_object_bbox(self, bbox_path):
        try:
            bbox_data = np.load(bbox_path, allow_pickle=True)
            for bbox in bbox_data:
                if bbox['semanticId'] != 0:
                    x_min, y_min = int(bbox['x_min']), int(bbox['y_min'])
                    x_max, y_max = int(bbox['x_max']), int(bbox['y_max'])
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
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(img_width, x_max), min(img_height, y_max)
            rgb = rgb.crop((x_min, y_min, x_max, y_max))
            depth = depth[y_min:y_max, x_min:x_max]
        
        rgb = self.rgb_transform(rgb)
        
        depth = depth * sample.get('scale_factor', 1.0)
        depth_valid = depth[(depth > DEPTH_MIN) & (depth < DEPTH_MAX)]
        if len(depth_valid) > 0:
            depth_min, depth_max = depth_valid.min(), depth_valid.max()
            depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-6)
        else:
            depth_normalized = depth / (DEPTH_MAX + 1e-6)
        depth_normalized = np.clip(depth_normalized, 0, 1).astype(np.float32)
        depth_pil = Image.fromarray((depth_normalized * 255).astype(np.uint8))
        depth_pil = depth_pil.resize((224, 224), Image.BILINEAR)
        depth_tensor = torch.tensor(np.array(depth_pil) / 255.0, dtype=torch.float32).unsqueeze(0)
        
        if self.position_stats is not None:
            mean = np.array(self.position_stats['mean'], dtype=np.float32)
            std = np.array(self.position_stats['std'], dtype=np.float32) + 1e-6
            position_normalized = (gt_pos - mean) / std
        else:
            position_normalized = gt_pos
        
        return {
            'rgb': rgb, 'depth': depth_tensor,
            'position': torch.tensor(position_normalized), 'position_raw': torch.tensor(gt_pos),
            'rotation_6d': torch.tensor(gt_rotation_6d), 'euler_deg': torch.tensor(gt_euler_deg),
            'class_idx': sample['class_idx']
        }


# ==========================================
# Depth Encoder V3 (GELU)
# ==========================================
class DepthEncoderV3(nn.Module):
    def __init__(self, out_features=256, drop_path_rate=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            DropPath(drop_path_rate),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            DropPath(drop_path_rate),
            nn.Conv2d(256, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, out_features)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ==========================================
# 모델 V3: ConvNeXt-Small + Depth → 6DoF Pose
# ==========================================
class RGBDepthTo3DModelV3(nn.Module):
    def __init__(self, num_classes=4, depth_features=256, use_rotation=True, drop_path_rate=0.1):
        super().__init__()
        self.use_rotation = use_rotation
        
        convnext = models.convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
        rgb_out = 768
        
        self.rgb_encoder = nn.Sequential(*list(convnext.children())[:-1])
        self.rgb_fc = nn.Sequential(
            nn.Linear(rgb_out, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2)
        )
        
        self.depth_encoder = DepthEncoderV3(out_features=depth_features, drop_path_rate=drop_path_rate)
        fusion_dim = 512 + depth_features
        
        self.position_head = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 3)
        )
        
        if use_rotation:
            self.rotation_head = nn.Sequential(
                nn.Linear(fusion_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
                nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
                nn.Linear(256, 6)
            )
        
        self.class_head = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        rotation_str = "+ 6D Rotation" if use_rotation else ""
        print(f"  모델 V3: ConvNeXt-Small + DepthEncoderV3 → 3D Position {rotation_str}")
        print(f"  RGB features: 512, Depth features: {depth_features}, Fusion: {fusion_dim}")
    
    def forward(self, rgb, depth):
        rgb_feat = self.rgb_encoder(rgb)
        rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)
        rgb_feat = self.rgb_fc(rgb_feat)
        depth_feat = self.depth_encoder(depth)
        fused = torch.cat([rgb_feat, depth_feat], dim=1)
        
        result = {'position': self.position_head(fused), 'class_logits': self.class_head(fused)}
        if self.use_rotation:
            result['rotation'] = self.rotation_head(fused)
        return result


# ==========================================
# Warmup + Cosine Annealing 스케줄러
# ==========================================
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer, self.warmup_epochs, self.total_epochs, self.min_lr = optimizer, warmup_epochs, total_epochs, min_lr
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / self.warmup_epochs
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr * warmup_factor
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
    
    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


# ==========================================
# 학습 (RTX 5090 최적화)
# ==========================================
def train_model(dataset_dir=DATASET_DIR, force_cpu=FORCE_CPU, use_bbox_crop=USE_BBOX_CROP):
    if not force_cpu and not torch.cuda.is_available():
        print("❌ CUDA 사용 불가!")
        sys.exit(1)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n🚀 GPU 사용: {gpu_name} ({gpu_memory:.1f}GB)")
    
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    print("\n" + "=" * 80)
    print("📦 V3 모델 학습 (ConvNeXt-Small + GELU) - RTX 5090 최적화")
    print("=" * 80)
    
    train_dataset = DepthGTDatasetV3(dataset_dir, split='train', use_augmentation=True, use_bbox_crop=use_bbox_crop)
    test_dataset = DepthGTDatasetV3(
        dataset_dir, split='test', use_augmentation=False, 
        position_stats=train_dataset.position_stats, use_bbox_crop=use_bbox_crop
    )
    
    if len(train_dataset) == 0:
        print("데이터셋이 비어있습니다.")
        return
    
    print(f"📦 배치 사이즈: {BATCH_SIZE}")
    
    pin_memory = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin_memory,
                              prefetch_factor=PREFETCH_FACTOR, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=pin_memory,
                             prefetch_factor=PREFETCH_FACTOR, persistent_workers=True)
    
    position_stats = train_dataset.position_stats
    num_classes = len(train_dataset.class_names)
    use_rotation = USE_POSE_ESTIMATION
    
    model = RGBDepthTo3DModelV3(num_classes=num_classes, depth_features=256, use_rotation=use_rotation).to(device)
    
    position_criterion = nn.SmoothL1Loss()
    rotation_criterion = GeodesicLoss() if use_rotation else None
    class_criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    best_pos_error = float('inf')
    best_rot_error = float('inf')
    patience_counter = 0
    
    print(f"\nTrain: {len(train_dataset)}, Test: {len(test_dataset)}")
    print(f"클래스: {train_dataset.class_names}")
    print(f"Epochs: {NUM_EPOCHS}, LR: {LEARNING_RATE}")
    print()
    
    for epoch in range(NUM_EPOCHS):
        scheduler.step(epoch)
        current_lr = scheduler.get_lr()[0]
        
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1:03d} Train", leave=False):
            rgb = batch['rgb'].to(device, non_blocking=True)
            depth = batch['depth'].to(device, non_blocking=True)
            position = batch['position'].to(device, non_blocking=True)
            rotation_6d = batch['rotation_6d'].to(device, non_blocking=True) if use_rotation else None
            class_idx = batch['class_idx'].to(device, non_blocking=True)
            
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
        
        model.eval()
        all_pos_errors, all_rot_errors = [], []
        correct, total = 0, 0
        
        with torch.no_grad():
            for batch in test_loader:
                rgb = batch['rgb'].to(device, non_blocking=True)
                depth = batch['depth'].to(device, non_blocking=True)
                position_raw = batch['position_raw'].to(device, non_blocking=True)
                rotation_6d_gt = batch['rotation_6d'].to(device, non_blocking=True) if use_rotation else None
                class_idx = batch['class_idx'].to(device, non_blocking=True)
                
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
                        rot_err = compute_rotation_error(pred['rotation'][i].float().cpu(), rotation_6d_gt[i].float().cpu())
                        all_rot_errors.append(rot_err.item() if isinstance(rot_err, torch.Tensor) else rot_err)
                
                _, pred_labels = torch.max(pred['class_logits'], 1)
                correct += (pred_labels == class_idx).sum().item()
                total += class_idx.size(0)
        
        avg_pos_error = np.mean(all_pos_errors)
        avg_rot_error = np.mean(all_rot_errors) if all_rot_errors else 0.0
        
        if avg_pos_error < best_pos_error:
            best_pos_error = avg_pos_error
            best_rot_error = avg_rot_error
            patience_counter = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'class_names': train_dataset.class_names, 'position_stats': position_stats,
                'best_pos_error': best_pos_error, 'best_rot_error': best_rot_error,
                'use_rotation': use_rotation, 'model_version': 'V3_ConvNeXtSmall_5090',
                'use_bbox_crop': use_bbox_crop
            }, os.path.join(ARTIFACTS_DIR, 'depth_gt_pose_v3_5090_best.pt'))
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            rot_str = f"RotErr={avg_rot_error:.2f}° | " if use_rotation else ""
            print(f"Epoch [{epoch+1:03d}/{NUM_EPOCHS}] Loss={train_loss:.4f} | "
                  f"PosErr={avg_pos_error:.1f}mm | {rot_str}LR={current_lr:.2e} | best={best_pos_error:.1f}mm")
        
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️  Early stopping at epoch {epoch+1}")
            break
    
    print(f"\n🎉 V3 학습 완료!")
    print(f"   최고 위치 오차: {best_pos_error:.2f}mm")
    if use_rotation:
        print(f"   최고 자세 오차: {best_rot_error:.2f}°")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Depth 기반 6DoF Pose V3 (RTX 5090)")
    parser.add_argument('--mode', type=str, default='train', choices=['train'])
    parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--bbox_crop', action='store_true')
    args = parser.parse_args()
    
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    FORCE_CPU = args.cpu
    USE_BBOX_CROP = args.bbox_crop
    
    if args.mode == 'train':
        train_model(args.dataset_dir, force_cpu=args.cpu, use_bbox_crop=args.bbox_crop)
    
    finish_logging()
