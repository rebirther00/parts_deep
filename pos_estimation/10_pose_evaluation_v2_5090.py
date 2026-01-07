#!/usr/bin/env python3
# ==========================================
# Depth 기반 6DoF Pose Estimation V2 평가 스크립트 (RTX 5090 최적화)
# EfficientNetV2-S + 224×224
# ==========================================
#
# V2 모델(EfficientNetV2-S)의 위치 및 자세 추정 정확도를 평가합니다.
# RTX 5090 최적화: 더 큰 배치 사이즈, non_blocking 데이터 전송
#
# 사용법:
#   python 10_pose_evaluation_v2_5090.py
#   python 10_pose_evaluation_v2_5090.py --num_samples 100
#   python 10_pose_evaluation_v2_5090.py --save_results --verbose
#
# ==========================================

import os
import sys
import json
import glob
import random
import argparse
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
# 경로 설정
# ==========================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, REPO_DIR)
from utils.logger import setup_logging, finish_logging

LOG_PATH = setup_logging("10_pose_eval_v2_5090")

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset_pos_depth")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "depth_gt_pose_v2_5090_best.pt")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "evaluation_results_v2_5090.json")

# RTX 5090 최적화
BATCH_SIZE = 64
NUM_WORKERS = 8
PREFETCH_FACTOR = 2

# ==========================================
# Depth 설정
# ==========================================
DEPTH_MIN = 0.01
DEPTH_MAX = 100.0
FOREGROUND_PERCENTILE = 10

CAMERA_INTRINSICS = {
    "fx": 768.0, "fy": 768.0,
    "cx": 512.0, "cy": 512.0,
    "width": 1024, "height": 1024
}

INPUT_SIZE = 224


# ==========================================
# 6D Rotation 함수들
# ==========================================
def euler_to_rotation_matrix(roll, pitch, yaw, degrees=True):
    if degrees:
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)
    
    Rx = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    
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


def rotation_matrix_to_euler(R, degrees=True):
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
    R = euler_to_rotation_matrix(roll, pitch, yaw, degrees)
    return rotation_matrix_to_6d(R)


def compute_rotation_error(pred_6d, gt_6d):
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
# Depth → 객체 중심 계산
# ==========================================
def depth_to_pointcloud(depth, fx, fy, cx, cy):
    h, w = depth.shape
    valid_mask = (depth > DEPTH_MIN) & (depth < DEPTH_MAX) & np.isfinite(depth)
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=1)
    return points, valid_mask


def compute_object_centroid_from_depth(depth, fx, fy, cx, cy, scale_factor=1.0):
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
# V2 평가용 데이터셋
# ==========================================
class PoseEvalDatasetV2(Dataset):
    def __init__(self, dataset_dir, position_stats, class_names, use_bbox_crop=False, train_ratio=0.8):
        self.dataset_dir = dataset_dir
        self.samples = []
        self.position_stats = position_stats
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.use_bbox_crop = use_bbox_crop
        
        self.fx, self.fy = CAMERA_INTRINSICS["fx"], CAMERA_INTRINSICS["fy"]
        self.cx, self.cy = CAMERA_INTRINSICS["cx"], CAMERA_INTRINSICS["cy"]
        
        self.class_scale_factors = {}
        class_dirs = sorted([d for d in glob.glob(os.path.join(dataset_dir, "*")) 
                            if os.path.isdir(d) and not d.endswith('__pycache__')])
        
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
                    scale_factor = 100.0 if depth_mean < 0.5 else (10.0 if depth_mean < 1.0 else 1.0)
                else:
                    scale_factor = 1.0
            else:
                scale_factor = 1.0
            self.class_scale_factors[class_name] = scale_factor
        
        self._collect_test_samples(class_dirs, train_ratio)
        
        self.rgb_transform = transforms.Compose([
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print(f"        입력 해상도: {INPUT_SIZE}×{INPUT_SIZE}")
    
    def _collect_test_samples(self, class_dirs, train_ratio):
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
                bbox_file = os.path.join(class_dir, f"bounding_box_2d_tight_{frame_idx:04d}.npy")
                pose_file = os.path.join(class_dir, f"pose_{frame_idx:04d}.json")
                if not os.path.exists(rgb_file):
                    continue
                if self.use_bbox_crop and not os.path.exists(bbox_file):
                    continue
                sample_data = {
                    'rgb_path': rgb_file, 'depth_path': depth_file,
                    'pose_path': pose_file if os.path.exists(pose_file) else None,
                    'class_name': class_name, 'class_idx': self.class_to_idx[class_name],
                    'scale_factor': scale_factor
                }
                if os.path.exists(bbox_file):
                    sample_data['bbox_path'] = bbox_file
                all_samples.append(sample_data)
        random.seed(42)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        self.samples = all_samples[split_idx:]
        print(f"  테스트 샘플: {len(self.samples)} / 전체 {len(all_samples)}")
    
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
        
        scale_factor = sample.get('scale_factor', 1.0)
        centroid, valid = compute_object_centroid_from_depth(
            depth_raw, self.fx, self.fy, self.cx, self.cy, scale_factor
        )
        gt_pos = centroid if valid else np.array([0, 0, 0])
        
        if sample.get('pose_path'):
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
        depth_pil = depth_pil.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
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
            'rgb': rgb, 'depth': depth_tensor,
            'position': torch.tensor(position_normalized), 'position_raw': torch.tensor(gt_pos),
            'rotation_6d': torch.tensor(gt_rotation_6d), 'euler_deg': torch.tensor(gt_euler_deg),
            'class_idx': sample['class_idx'], 'class_name': sample['class_name'],
            'rgb_path': sample['rgb_path']
        }


# ==========================================
# Depth Encoder
# ==========================================
class DepthEncoder(nn.Module):
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
        return self.fc(x)


# ==========================================
# V2 모델
# ==========================================
class RGBDepthTo3DModelV2(nn.Module):
    def __init__(self, num_classes=4, depth_features=256, use_rotation=True):
        super().__init__()
        self.use_rotation = use_rotation
        effnet = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        self.rgb_encoder = nn.Sequential(*list(effnet.children())[:-1])
        self.rgb_fc = nn.Sequential(
            nn.Linear(1280, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.2)
        )
        self.depth_encoder = DepthEncoder(out_features=depth_features)
        fusion_dim = 512 + depth_features
        self.position_head = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 3)
        )
        if use_rotation:
            self.rotation_head = nn.Sequential(
                nn.Linear(fusion_dim, 512), nn.LayerNorm(512), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(512, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(256, 6)
            )
        self.class_head = nn.Sequential(
            nn.Linear(fusion_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
    
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
# 메인 평가 함수
# ==========================================
def evaluate(args, model_path=None):
    if model_path is None:
        model_path = MODEL_PATH
    
    print("=" * 80)
    print("📊 V2 6DoF Pose Estimation 모델 평가 (RTX 5090 최적화)")
    print("   (EfficientNetV2-S + 224×224)")
    print("=" * 80)
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        print("   먼저 학습을 실행하세요: python 09_depth_based_pose_v2_5090.py --mode train")
        sys.exit(1)
    
    print(f"\n✅ 모델 로드: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    position_stats = checkpoint['position_stats']
    class_names = checkpoint['class_names']
    use_rotation = checkpoint.get('use_rotation', True)
    best_pos_error = checkpoint.get('best_pos_error', 0)
    best_rot_error = checkpoint.get('best_rot_error', 0)
    model_version = checkpoint.get('model_version', 'V2_EfficientNetV2S_5090')
    
    print(f"   모델 버전: {model_version}")
    print(f"   학습 시 최고 위치 오차: {best_pos_error:.2f}mm")
    if use_rotation:
        print(f"   학습 시 최고 자세 오차: {best_rot_error:.2f}°")
    print(f"   클래스: {class_names}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   디바이스: {device}")
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"   GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    num_classes = len(class_names)
    model = RGBDepthTo3DModelV2(num_classes=num_classes, depth_features=256, use_rotation=use_rotation)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"\n📂 데이터셋 로드: {args.dataset_dir}")
    test_dataset = PoseEvalDatasetV2(
        args.dataset_dir, position_stats=position_stats,
        class_names=class_names, use_bbox_crop=args.bbox_crop
    )
    
    if len(test_dataset) == 0:
        print("❌ 테스트 데이터셋이 비어있습니다.")
        sys.exit(1)
    
    num_samples = min(args.num_samples, len(test_dataset)) if args.num_samples else len(test_dataset)
    print(f"\n평가 샘플 수: {num_samples} / {len(test_dataset)}")
    print("=" * 80)
    
    all_pos_errors, all_rot_errors = [], []
    all_x_errors, all_y_errors, all_z_errors = [], [], []
    class_correct, class_total = 0, 0
    class_pos_errors = {name: [] for name in class_names}
    class_rot_errors = {name: [] for name in class_names}
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="평가 중"):
            sample = test_dataset[i]
            rgb = sample['rgb'].unsqueeze(0).to(device, non_blocking=True)
            depth = sample['depth'].unsqueeze(0).to(device, non_blocking=True)
            gt_pos_raw = sample['position_raw'].numpy()
            gt_rot_6d = sample['rotation_6d'].numpy() if use_rotation else None
            gt_euler = sample['euler_deg'].numpy() if use_rotation else None
            class_idx = sample['class_idx']
            class_name = sample['class_name']
            
            pred = model(rgb, depth)
            mean = np.array(position_stats['mean'])
            std = np.array(position_stats['std']) + 1e-6
            pred_pos_raw = pred['position'].cpu().numpy()[0] * std + mean
            
            pos_error = np.sqrt(np.sum((pred_pos_raw - gt_pos_raw) ** 2)) * 1000
            x_error = abs(pred_pos_raw[0] - gt_pos_raw[0]) * 1000
            y_error = abs(pred_pos_raw[1] - gt_pos_raw[1]) * 1000
            z_error = abs(pred_pos_raw[2] - gt_pos_raw[2]) * 1000
            
            all_pos_errors.append(pos_error)
            all_x_errors.append(x_error)
            all_y_errors.append(y_error)
            all_z_errors.append(z_error)
            class_pos_errors[class_name].append(pos_error)
            
            rot_error = 0.0
            pred_euler = [0, 0, 0]
            if use_rotation and 'rotation' in pred:
                pred_rot_6d = pred['rotation'].float().cpu().numpy()[0]
                rot_error = compute_rotation_error(pred_rot_6d, gt_rot_6d)
                all_rot_errors.append(rot_error)
                class_rot_errors[class_name].append(rot_error)
                pred_R = rotation_6d_to_matrix(pred_rot_6d)
                pred_euler = rotation_matrix_to_euler(pred_R, degrees=True)
            
            pred_class = torch.argmax(pred['class_logits'], dim=1).item()
            if pred_class == class_idx:
                class_correct += 1
            class_total += 1
            
            if args.verbose and i < 5:
                print(f"\n샘플 {i+1} ({class_name}):")
                print(f"  위치 GT:    [{gt_pos_raw[0]:.3f}, {gt_pos_raw[1]:.3f}, {gt_pos_raw[2]:.3f}]")
                print(f"  위치 예측:  [{pred_pos_raw[0]:.3f}, {pred_pos_raw[1]:.3f}, {pred_pos_raw[2]:.3f}]")
                print(f"  위치 오차:  {pos_error:.1f}mm")
                if use_rotation:
                    print(f"  자세 오차:  {rot_error:.2f}°")
    
    print(f"\n{'='*80}")
    print("📈 V2 평가 결과 요약 (RTX 5090)")
    print(f"{'='*80}")
    
    avg_pos_error = np.mean(all_pos_errors)
    print(f"\n📍 위치 오차: {avg_pos_error:.2f}mm (±{np.std(all_pos_errors):.2f}mm)")
    print(f"   X: {np.mean(all_x_errors):.2f}mm, Y: {np.mean(all_y_errors):.2f}mm, Z: {np.mean(all_z_errors):.2f}mm")
    print(f"\n   분포: <10mm: {100*sum(1 for e in all_pos_errors if e<10)/len(all_pos_errors):.1f}%, "
          f"<50mm: {100*sum(1 for e in all_pos_errors if e<50)/len(all_pos_errors):.1f}%")
    
    if use_rotation and all_rot_errors:
        print(f"\n🔄 자세 오차: {np.mean(all_rot_errors):.2f}° (±{np.std(all_rot_errors):.2f}°)")
    
    print(f"\n🏷️ 분류 정확도: {100*class_correct/class_total:.1f}%")
    
    if args.save_results:
        summary = {
            'model_version': model_version, 'num_samples': num_samples,
            'position_error_mm': float(avg_pos_error),
            'classification_accuracy': float(100*class_correct/class_total)
        }
        if use_rotation:
            summary['rotation_error_deg'] = float(np.mean(all_rot_errors))
        with open(RESULTS_PATH, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n💾 결과 저장: {RESULTS_PATH}")
    
    print(f"\n{'='*80}")
    print("✅ V2 평가 완료")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V2 6DoF Pose 평가 (RTX 5090)")
    parser.add_argument('--model', type=str, default='depth_gt_pose_v2_5090_best.pt')
    parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--bbox_crop', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    model_path = args.model if os.path.isabs(args.model) else os.path.join(ARTIFACTS_DIR, args.model)
    evaluate(args, model_path=model_path)
    finish_logging()
