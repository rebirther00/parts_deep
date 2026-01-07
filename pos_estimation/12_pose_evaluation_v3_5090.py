#!/usr/bin/env python3
# ==========================================
# Depth 기반 6DoF Pose Estimation V3 평가 스크립트 (RTX 5090 최적화)
# ConvNeXt-Small + GELU + 224×224
# ==========================================
#
# V3 모델(ConvNeXt-Small)의 위치 및 자세 추정 정확도를 평가합니다.
#
# 사용법:
#   python 12_pose_evaluation_v3_5090.py
#   python 12_pose_evaluation_v3_5090.py --num_samples 100 --verbose
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
from torchvision.models import ConvNeXt_Small_Weights
from PIL import Image

# 경로 설정
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, REPO_DIR)
from utils.logger import setup_logging, finish_logging

LOG_PATH = setup_logging("12_pose_eval_v3_5090")

DATASET_DIR = os.path.join(PROJECT_DIR, "dataset_pos_depth")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "depth_gt_pose_v3_5090_best.pt")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "evaluation_results_v3_5090.json")

BATCH_SIZE = 64
NUM_WORKERS = 8
INPUT_SIZE = 224

DEPTH_MIN, DEPTH_MAX = 0.01, 100.0
FOREGROUND_PERCENTILE = 10
CAMERA_INTRINSICS = {"fx": 768.0, "fy": 768.0, "cx": 512.0, "cy": 512.0}


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
    return rotation_matrix_to_6d(euler_to_rotation_matrix(roll, pitch, yaw, degrees))

def compute_rotation_error(pred_6d, gt_6d):
    if isinstance(pred_6d, torch.Tensor):
        pred_6d = pred_6d.detach().cpu().numpy()
    if isinstance(gt_6d, torch.Tensor):
        gt_6d = gt_6d.detach().cpu().numpy()
    R_pred, R_gt = rotation_6d_to_matrix(pred_6d), rotation_6d_to_matrix(gt_6d)
    cos_theta = np.clip((np.trace(R_pred.T @ R_gt) - 1) / 2, -1, 1)
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
    foreground_mask = points[:, 2] < np.percentile(points[:, 2], FOREGROUND_PERCENTILE)
    foreground_points = points[foreground_mask]
    if len(foreground_points) < 50:
        return np.array([0, 0, 0]), False
    return foreground_points.mean(axis=0), True


# ==========================================
# V3 평가용 데이터셋
# ==========================================
class PoseEvalDatasetV3(Dataset):
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
        centroid, valid = compute_object_centroid_from_depth(depth_raw, self.fx, self.fy, self.cx, self.cy, scale_factor)
        gt_pos = centroid if valid else np.array([0, 0, 0])
        
        if sample.get('pose_path'):
            try:
                with open(sample['pose_path'], 'r') as f:
                    pose_data = json.load(f)
                r_xyz_deg = pose_data.get('camTobj', {}).get('r_xyz_deg', [0, 0, 0])
                gt_rotation_6d = euler_to_6d(r_xyz_deg[0], r_xyz_deg[1], r_xyz_deg[2])
                gt_euler_deg = np.array(r_xyz_deg, dtype=np.float32)
            except:
                gt_rotation_6d, gt_euler_deg = euler_to_6d(0, 0, 0), np.array([0, 0, 0], dtype=np.float32)
        else:
            gt_rotation_6d, gt_euler_deg = euler_to_6d(0, 0, 0), np.array([0, 0, 0], dtype=np.float32)
        
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
        
        depth = depth * scale_factor
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
        
        return {
            'rgb': rgb, 'depth': depth_tensor,
            'position': torch.tensor(position_normalized), 'position_raw': torch.tensor(gt_pos),
            'rotation_6d': torch.tensor(gt_rotation_6d.astype(np.float32)),
            'euler_deg': torch.tensor(gt_euler_deg),
            'class_idx': sample['class_idx'], 'class_name': sample['class_name']
        }


# ==========================================
# DropPath
# ==========================================
class DropPath(nn.Module):
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
# Depth Encoder V3
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
        return self.fc(x.view(x.size(0), -1))


# ==========================================
# V3 모델
# ==========================================
class RGBDepthTo3DModelV3(nn.Module):
    def __init__(self, num_classes=4, depth_features=256, use_rotation=True, drop_path_rate=0.1):
        super().__init__()
        self.use_rotation = use_rotation
        convnext = models.convnext_small(weights=ConvNeXt_Small_Weights.IMAGENET1K_V1)
        self.rgb_encoder = nn.Sequential(*list(convnext.children())[:-1])
        self.rgb_fc = nn.Sequential(nn.Linear(768, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.2))
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
    
    def forward(self, rgb, depth):
        rgb_feat = self.rgb_encoder(rgb)
        rgb_feat = self.rgb_fc(rgb_feat.view(rgb_feat.size(0), -1))
        depth_feat = self.depth_encoder(depth)
        fused = torch.cat([rgb_feat, depth_feat], dim=1)
        result = {'position': self.position_head(fused), 'class_logits': self.class_head(fused)}
        if self.use_rotation:
            result['rotation'] = self.rotation_head(fused)
        return result


# ==========================================
# 평가 함수
# ==========================================
def evaluate(args, model_path=None):
    if model_path is None:
        model_path = MODEL_PATH
    
    print("=" * 80)
    print("📊 V3 6DoF Pose Estimation 모델 평가 (RTX 5090 최적화)")
    print("   (ConvNeXt-Small + GELU + 224×224)")
    print("=" * 80)
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        sys.exit(1)
    
    print(f"\n✅ 모델 로드: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    position_stats = checkpoint['position_stats']
    class_names = checkpoint['class_names']
    use_rotation = checkpoint.get('use_rotation', True)
    model_version = checkpoint.get('model_version', 'V3_ConvNeXtSmall_5090')
    
    print(f"   모델 버전: {model_version}")
    print(f"   클래스: {class_names}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   디바이스: {device}")
    
    num_classes = len(class_names)
    model = RGBDepthTo3DModelV3(num_classes=num_classes, depth_features=256, use_rotation=use_rotation)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"\n📂 데이터셋 로드: {args.dataset_dir}")
    test_dataset = PoseEvalDatasetV3(args.dataset_dir, position_stats=position_stats, class_names=class_names, use_bbox_crop=args.bbox_crop)
    
    if len(test_dataset) == 0:
        print("❌ 테스트 데이터셋이 비어있습니다.")
        sys.exit(1)
    
    num_samples = min(args.num_samples, len(test_dataset)) if args.num_samples else len(test_dataset)
    print(f"\n평가 샘플 수: {num_samples}")
    print("=" * 80)
    
    all_pos_errors, all_rot_errors = [], []
    class_correct, class_total = 0, 0
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="평가 중"):
            sample = test_dataset[i]
            rgb = sample['rgb'].unsqueeze(0).to(device, non_blocking=True)
            depth = sample['depth'].unsqueeze(0).to(device, non_blocking=True)
            gt_pos_raw = sample['position_raw'].numpy()
            gt_rot_6d = sample['rotation_6d'].numpy() if use_rotation else None
            class_idx = sample['class_idx']
            
            pred = model(rgb, depth)
            mean = np.array(position_stats['mean'])
            std = np.array(position_stats['std']) + 1e-6
            pred_pos_raw = pred['position'].cpu().numpy()[0] * std + mean
            
            pos_error = np.sqrt(np.sum((pred_pos_raw - gt_pos_raw) ** 2)) * 1000
            all_pos_errors.append(pos_error)
            
            if use_rotation and 'rotation' in pred:
                rot_error = compute_rotation_error(pred['rotation'].cpu().numpy()[0], gt_rot_6d)
                all_rot_errors.append(rot_error)
            
            pred_class = torch.argmax(pred['class_logits'], dim=1).item()
            if pred_class == class_idx:
                class_correct += 1
            class_total += 1
    
    print(f"\n{'='*80}")
    print("📈 V3 평가 결과 요약")
    print(f"{'='*80}")
    
    avg_pos_error = np.mean(all_pos_errors)
    print(f"\n📍 위치 오차: {avg_pos_error:.2f}mm (±{np.std(all_pos_errors):.2f}mm)")
    print(f"   Median: {np.median(all_pos_errors):.2f}mm, Min: {np.min(all_pos_errors):.2f}mm, Max: {np.max(all_pos_errors):.2f}mm")
    
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
    print("✅ V3 평가 완료")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="V3 6DoF Pose 평가 (RTX 5090)")
    parser.add_argument('--model', type=str, default='depth_gt_pose_v3_5090_best.pt')
    parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR)
    parser.add_argument('--num_samples', type=int, default=None)
    parser.add_argument('--bbox_crop', action='store_true')
    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    
    model_path = args.model if os.path.isabs(args.model) else os.path.join(ARTIFACTS_DIR, args.model)
    evaluate(args, model_path=model_path)
    finish_logging()
