#!/usr/bin/env python3
# ==========================================
# 분류(Classification) 전용 평가 스크립트
# ==========================================
#
# 07_depth_based_pose.py로 학습된 모델의 분류 성능만 평가합니다.
# 위치/자세 평가는 제외하고 분류 정확도, 혼동 행렬, 시각화 등을 제공합니다.
#
# 사용법:
#   python 09_class_only_evaluation.py
#   python 09_class_only_evaluation.py --dataset_dir ../class_estimation/datasets
#   python 09_class_only_evaluation.py --num_samples 100
#   python 09_class_only_evaluation.py --bbox_crop
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
from torchvision.models import ResNet50_Weights
from PIL import Image, ImageDraw, ImageFont

# ==========================================
# 로깅 설정
# ==========================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, REPO_DIR)
from utils.logger import setup_logging, finish_logging

# ==========================================
# 경로 설정
# ==========================================
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset_pos_depth")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "depth_gt_6dof_best.pt")
RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "evaluation_class_results.json")
OUTPUT_IMAGE_PATH = os.path.join(ARTIFACTS_DIR, "evaluation_class_grid.png")
OUTPUT_WRONG_IMAGE_PATH = os.path.join(ARTIFACTS_DIR, "evaluation_class_wrong.png")

# ==========================================
# Depth 설정
# ==========================================
DEPTH_MIN = 0.01
DEPTH_MAX = 100.0

CAMERA_INTRINSICS = {
    "fx": 768.0, "fy": 768.0,
    "cx": 512.0, "cy": 512.0,
    "width": 1024, "height": 1024
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
# RGB+Depth → 6DoF Pose 모델
# ==========================================
class RGBDepthTo3DModel(nn.Module):
    """RGB + Depth 융합 → 6DoF Pose 예측 모델"""
    
    def __init__(self, num_classes=4, depth_features=256, use_rotation=True):
        super().__init__()
        self.use_rotation = use_rotation
        
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        rgb_out = 2048
        
        self.rgb_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.rgb_fc = nn.Linear(rgb_out, 512)
        
        self.depth_encoder = DepthEncoder(out_features=depth_features)
        
        fusion_dim = 512 + depth_features
        
        self.position_head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3)
        )
        
        if use_rotation:
            self.rotation_head = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 6)
            )
        
        self.class_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, rgb, depth):
        rgb_feat = self.rgb_encoder(rgb)
        rgb_feat = rgb_feat.view(rgb_feat.size(0), -1)
        rgb_feat = self.rgb_fc(rgb_feat)
        
        depth_feat = self.depth_encoder(depth)
        
        fused = torch.cat([rgb_feat, depth_feat], dim=1)
        
        class_logits = self.class_head(fused)
        
        return {'class_logits': class_logits}


# ==========================================
# 평가용 데이터셋
# ==========================================
class ClassEvalDataset(Dataset):
    """분류 평가 전용 데이터셋"""
    
    def __init__(self, dataset_dir, class_names, use_bbox_crop=False, 
                 train_ratio=0.8, has_depth=True):
        self.dataset_dir = dataset_dir
        self.samples = []
        self.class_names = class_names
        self.class_to_idx = {name: idx for idx, name in enumerate(class_names)}
        self.use_bbox_crop = use_bbox_crop
        self.has_depth = has_depth
        
        # 클래스별 스케일 팩터 (depth가 있는 경우)
        self.class_scale_factors = {}
        
        class_dirs = sorted(glob.glob(os.path.join(dataset_dir, "*")))
        class_dirs = [d for d in class_dirs if os.path.isdir(d) and not d.endswith('__pycache__')]
        
        if has_depth:
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
        
        # 샘플 수집
        self._collect_test_samples(class_dirs, train_ratio)
        
        # RGB Transform
        self.rgb_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def _collect_test_samples(self, class_dirs, train_ratio):
        """테스트 샘플 수집"""
        all_samples = []
        
        for class_dir in class_dirs:
            class_name = os.path.basename(class_dir)
            if class_name not in self.class_to_idx:
                continue
            
            scale_factor = self.class_scale_factors.get(class_name, 1.0)
            
            # RGB 파일 기준으로 스캔
            rgb_files = sorted(glob.glob(os.path.join(class_dir, "rgb_*.png")))
            
            for rgb_file in rgb_files:
                frame_idx = int(os.path.basename(rgb_file).split('_')[-1].split('.')[0])
                
                sample_data = {
                    'rgb_path': rgb_file,
                    'class_name': class_name,
                    'class_idx': self.class_to_idx[class_name],
                    'scale_factor': scale_factor
                }
                
                # Depth 파일 확인
                if self.has_depth:
                    depth_file = os.path.join(class_dir, f"distance_to_camera_{frame_idx:04d}.npy")
                    if os.path.exists(depth_file):
                        sample_data['depth_path'] = depth_file
                    else:
                        sample_data['depth_path'] = None
                else:
                    sample_data['depth_path'] = None
                
                # bbox 파일 확인
                bbox_file = os.path.join(class_dir, f"bounding_box_2d_tight_{frame_idx:04d}.npy")
                if os.path.exists(bbox_file):
                    sample_data['bbox_path'] = bbox_file
                
                all_samples.append(sample_data)
        
        # Train/Test 분할 (학습 시와 동일한 seed 사용)
        random.seed(42)
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * train_ratio)
        
        # 테스트 샘플만 사용
        self.samples = all_samples[split_idx:]
        print(f"  테스트 샘플: {len(self.samples)} / 전체 {len(all_samples)}")
    
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # RGB 로드
        rgb = Image.open(sample['rgb_path']).convert('RGB')
        
        # Depth 로드 (있는 경우)
        depth_tensor = None
        if sample.get('depth_path') and os.path.exists(sample['depth_path']):
            depth_raw = np.load(sample['depth_path'])
            if len(depth_raw.shape) == 3:
                depth_raw = depth_raw[:, :, 0]
            depth = depth_raw.copy()
        else:
            # Depth가 없으면 더미 생성
            depth = np.ones((224, 224), dtype=np.float32) * 0.5
        
        # bbox crop 적용
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
            if sample.get('depth_path'):
                depth = depth[y_min:y_max, x_min:x_max]
        
        # RGB transform
        rgb_tensor = self.rgb_transform(rgb)
        
        # Depth 정규화
        scale_factor = sample.get('scale_factor', 1.0)
        depth = depth * scale_factor
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
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'class_idx': sample['class_idx'],
            'class_name': sample['class_name'],
            'rgb_path': sample['rgb_path']
        }


# ==========================================
# 시각화 함수
# ==========================================
def get_original_image(path, bbox_crop=False, bbox_path=None):
    """원본 이미지 로드 (bbox_crop 옵션 적용)"""
    image = Image.open(path).convert('RGB')
    
    if bbox_crop and bbox_path and os.path.exists(bbox_path):
        try:
            bbox_data = np.load(bbox_path, allow_pickle=True)
            for bbox in bbox_data:
                if bbox['semanticId'] != 0:
                    x_min = int(bbox['x_min'])
                    y_min = int(bbox['y_min'])
                    x_max = int(bbox['x_max'])
                    y_max = int(bbox['y_max'])
                    if x_max > x_min and y_max > y_min:
                        w, h = image.size
                        x_min = max(0, min(w - 1, x_min))
                        y_min = max(0, min(h - 1, y_min))
                        x_max = max(0, min(w - 1, x_max))
                        y_max = max(0, min(h - 1, y_max))
                        image = image.crop((x_min, y_min, x_max + 1, y_max + 1))
                    break
        except:
            pass
    
    return image


def create_result_grid(results, class_names, output_path, num_cols=5, img_size=200, max_images=50):
    """예측 결과를 그리드 형태로 시각화"""
    num_images = min(len(results), max_images)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    text_height = 100
    cell_width = img_size
    cell_height = img_size + text_height
    
    grid_width = num_cols * cell_width
    grid_height = num_rows * cell_height
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # 폰트 로드
    try:
        font_paths = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        ]
        font = None
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, 12)
                break
            except:
                continue
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    for idx in range(num_images):
        r = results[idx]
        row = idx // num_cols
        col = idx % num_cols
        
        x = col * cell_width
        y = row * cell_height
        
        # 이미지 로드 및 리사이즈
        img = get_original_image(r['rgb_path'], r.get('bbox_crop', False), r.get('bbox_path'))
        img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        grid_image.paste(img, (x, y))
        
        # 텍스트 정보
        draw = ImageDraw.Draw(grid_image)
        text_y = y + img_size + 5
        
        actual = class_names[r['label']]
        predicted = class_names[r['pred']]
        conf = r['confidence']
        is_correct = r['correct']
        
        symbol = "O" if is_correct else "X"
        text_color = (0, 150, 0) if is_correct else (200, 0, 0)
        
        # 배경색
        bg_color = (230, 255, 230) if is_correct else (255, 230, 230)
        draw.rectangle([x, y + img_size, x + cell_width, y + cell_height], fill=bg_color)
        
        # 텍스트
        text_line1 = f"{symbol} Actual: {actual}"
        text_line2 = f"Pred: {predicted}"
        text_line3 = f"Conf: {conf:.1f}%"
        text_line4 = f"src: {os.path.basename(r['rgb_path'])}"
        
        draw.text((x + 5, text_y), text_line1, fill=text_color, font=font)
        draw.text((x + 5, text_y + 20), text_line2, fill=(0, 0, 0), font=font)
        draw.text((x + 5, text_y + 40), text_line3, fill=(100, 100, 100), font=font)
        draw.text((x + 5, text_y + 60), text_line4, fill=(60, 60, 60), font=font)
    
    grid_image.save(output_path, 'PNG', quality=95)
    print(f"\n결과 이미지 저장: {output_path}")
    return grid_image


def create_wrong_predictions_grid(results, class_names, output_path, num_cols=5, img_size=200):
    """틀린 예측 결과만 그리드 형태로 시각화"""
    wrong_results = [r for r in results if not r['correct']]
    
    if len(wrong_results) == 0:
        print("\n✓ 모든 예측이 정확합니다! 틀린 예측 이미지가 없습니다.")
        return None
    
    num_images = len(wrong_results)
    num_rows = (num_images + num_cols - 1) // num_cols
    
    text_height = 100
    cell_width = img_size
    cell_height = img_size + text_height
    
    grid_width = num_cols * cell_width
    grid_height = num_rows * cell_height
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # 폰트 로드
    try:
        font_paths = [
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        ]
        font = None
        for path in font_paths:
            try:
                font = ImageFont.truetype(path, 12)
                break
            except:
                continue
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    for idx, r in enumerate(wrong_results):
        row = idx // num_cols
        col = idx % num_cols
        
        x = col * cell_width
        y = row * cell_height
        
        # 이미지 로드 및 리사이즈
        img = get_original_image(r['rgb_path'], r.get('bbox_crop', False), r.get('bbox_path'))
        img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        grid_image.paste(img, (x, y))
        
        # 텍스트 정보
        draw = ImageDraw.Draw(grid_image)
        text_y = y + img_size + 5
        
        actual = class_names[r['label']]
        predicted = class_names[r['pred']]
        conf = r['confidence']
        
        # 배경색 (틀린 예측)
        bg_color = (255, 220, 220)
        draw.rectangle([x, y + img_size, x + cell_width, y + cell_height], fill=bg_color)
        
        # 텍스트
        text_line1 = f"X Actual: {actual}"
        text_line2 = f"Pred: {predicted}"
        text_line3 = f"Conf: {conf:.1f}%"
        text_line4 = f"src: {os.path.basename(r['rgb_path'])}"
        
        draw.text((x + 5, text_y), text_line1, fill=(200, 0, 0), font=font)
        draw.text((x + 5, text_y + 20), text_line2, fill=(0, 0, 0), font=font)
        draw.text((x + 5, text_y + 40), text_line3, fill=(100, 100, 100), font=font)
        draw.text((x + 5, text_y + 60), text_line4, fill=(60, 60, 60), font=font)
    
    grid_image.save(output_path, 'PNG', quality=95)
    print(f"\n틀린 예측 이미지 저장: {output_path} ({num_images}개)")
    return grid_image


# ==========================================
# 메인 평가 함수
# ==========================================
def evaluate(args):
    """분류 성능만 평가"""
    
    print("=" * 80)
    print("📊 분류(Classification) 전용 평가")
    print("=" * 80)
    
    # 모델 로드
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {MODEL_PATH}")
        print("   먼저 학습을 실행하세요: python 07_depth_based_pose.py --mode train")
        sys.exit(1)
    
    print(f"\n✅ 모델 로드: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    
    class_names = checkpoint['class_names']
    use_rotation = checkpoint.get('use_rotation', True)
    
    print(f"   클래스: {class_names}")
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   디바이스: {device}")
    
    # 데이터셋 경로 확인
    dataset_dir = args.dataset_dir
    
    # Depth 파일 존재 여부 확인
    has_depth = True
    test_depth_file = glob.glob(os.path.join(dataset_dir, "*", "distance_to_camera_*.npy"))
    if len(test_depth_file) == 0:
        print(f"   ⚠️  Depth 파일이 없습니다. 더미 Depth를 사용합니다.")
        has_depth = False
    
    # 모델 생성 및 가중치 로드
    num_classes = len(class_names)
    model = RGBDepthTo3DModel(num_classes=num_classes, depth_features=256, use_rotation=use_rotation)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # 테스트 데이터셋 로드
    print(f"\n📂 데이터셋 로드: {dataset_dir}")
    if args.bbox_crop:
        print("   📦 bbox_2d ROI crop 모드")
    
    test_dataset = ClassEvalDataset(
        dataset_dir,
        class_names=class_names,
        use_bbox_crop=args.bbox_crop,
        has_depth=has_depth
    )
    
    if len(test_dataset) == 0:
        print("❌ 테스트 데이터셋이 비어있습니다.")
        sys.exit(1)
    
    # 샘플 수 제한
    if args.num_samples is not None:
        num_samples = min(args.num_samples, len(test_dataset))
    else:
        num_samples = len(test_dataset)
    
    print(f"\n평가 샘플 수: {num_samples} / {len(test_dataset)}")
    print("=" * 80)
    
    # 평가 수행
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    all_results = []
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for i in tqdm(range(num_samples), desc="평가 중"):
            sample = test_dataset[i]
            
            rgb = sample['rgb'].unsqueeze(0).to(device)
            depth = sample['depth'].unsqueeze(0).to(device)
            class_idx = sample['class_idx']
            class_name = sample['class_name']
            rgb_path = sample['rgb_path']
            
            # 예측
            pred = model(rgb, depth)
            
            # 분류 결과
            probs = torch.softmax(pred['class_logits'], dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item() * 100
            
            is_correct = pred_class == class_idx
            
            if is_correct:
                correct += 1
                class_correct[class_idx] += 1
            
            total += 1
            class_total[class_idx] += 1
            
            all_predictions.append(pred_class)
            all_labels.append(class_idx)
            
            # 결과 저장
            result_entry = {
                'rgb_path': rgb_path,
                'label': class_idx,
                'pred': pred_class,
                'confidence': confidence,
                'correct': is_correct,
                'bbox_crop': args.bbox_crop
            }
            
            # bbox_path 추가 (시각화용)
            frame_idx = int(os.path.basename(rgb_path).split('_')[-1].split('.')[0])
            bbox_path = os.path.join(os.path.dirname(rgb_path), f"bounding_box_2d_tight_{frame_idx:04d}.npy")
            if os.path.exists(bbox_path):
                result_entry['bbox_path'] = bbox_path
            
            all_results.append(result_entry)
            
            # 샘플 출력 (처음 20개)
            if i < 20:
                symbol = "✓" if is_correct else "✗"
                print(f"{symbol} [{class_names[class_idx]:20s}] → [{class_names[pred_class]:20s}] | "
                      f"신뢰도: {confidence:.1f}%")
    
    if num_samples > 20:
        print(f"... ({num_samples - 20}개 결과 생략)")
    
    # 결과 요약
    print(f"\n{'='*80}")
    print("📈 평가 결과 요약")
    print(f"{'='*80}")
    
    overall_accuracy = 100.0 * correct / total
    print(f"\n🎯 전체 정확도: {overall_accuracy:.2f}% ({correct}/{total})")
    
    # 클래스별 정확도
    print(f"\n클래스별 정확도:")
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            print(f"  - {name}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"  - {name}: N/A (샘플 없음)")
    
    # 혼동 행렬
    print(f"\n혼동 행렬:")
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
    for true_label, pred_label in zip(all_labels, all_predictions):
        confusion_matrix[true_label][pred_label] += 1
    
    print("\n" + " " * 20 + "예측")
    print(" " * 20 + "".join([f"{name[:12]:>12s}" for name in class_names]))
    print("-" * (20 + 12 * num_classes))
    for i, name in enumerate(class_names):
        row = "".join([f"{confusion_matrix[i][j]:>12d}" for j in range(num_classes)])
        print(f"실제 {name[:14]:>14s} |{row}")
    
    # Precision, Recall, F1-Score
    print(f"\n성능 지표 (클래스별):")
    print(f"{'클래스':<20s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s}")
    print("-" * 52)
    
    all_precision = []
    all_recall = []
    all_f1 = []
    
    for i, name in enumerate(class_names):
        tp = confusion_matrix[i][i]
        fp = sum(confusion_matrix[j][i] for j in range(num_classes)) - tp
        fn = sum(confusion_matrix[i][j] for j in range(num_classes)) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        all_precision.append(precision)
        all_recall.append(recall)
        all_f1.append(f1)
        
        print(f"{name:<20s} {precision*100:>9.2f}% {recall*100:>9.2f}% {f1*100:>9.2f}%")
    
    print("-" * 52)
    avg_precision = np.mean(all_precision) * 100
    avg_recall = np.mean(all_recall) * 100
    avg_f1 = np.mean(all_f1) * 100
    print(f"{'평균 (Macro)':<20s} {avg_precision:>9.2f}% {avg_recall:>9.2f}% {avg_f1:>9.2f}%")
    
    # 오류 분석
    errors = total - correct
    print(f"\n오류 개수: {errors}개")
    if errors > 0:
        print("\n오류 분석 (신뢰도 높은 상위 5개):")
        error_results = [r for r in all_results if not r['correct']]
        error_results.sort(key=lambda x: -x['confidence'])
        for r in error_results[:5]:
            actual = class_names[r['label']]
            predicted = class_names[r['pred']]
            print(f"  - {actual} → {predicted} (신뢰도: {r['confidence']:.1f}%)")
    
    # 시각화 생성
    print(f"\n{'='*80}")
    print("📸 시각화 생성")
    print(f"{'='*80}")
    
    create_result_grid(all_results, class_names, OUTPUT_IMAGE_PATH)
    create_wrong_predictions_grid(all_results, class_names, OUTPUT_WRONG_IMAGE_PATH)
    
    # 결과 저장
    summary_results = {
        'accuracy': overall_accuracy,
        'total_samples': total,
        'correct_samples': correct,
        'class_names': class_names,
        'class_accuracies': {name: 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0 
                            for i, name in enumerate(class_names)},
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'dataset_dir': dataset_dir
    }
    
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 결과 저장: {RESULTS_PATH}")
    
    print(f"\n{'='*80}")
    print("✅ 평가 완료!")
    print(f"{'='*80}")


# ==========================================
# 메인
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="분류(Classification) 전용 평가")
    parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR,
                        help='평가 데이터셋 경로 (기본: dataset_pos_depth)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='평가할 샘플 수 (None: 전체)')
    parser.add_argument('--bbox_crop', action='store_true',
                        help='bbox_2d로 ROI crop 사용')
    
    args = parser.parse_args()
    
    # 로그 파일 생성
    LOG_PATH = setup_logging("09_class_evaluation")
    
    evaluate(args)
    
    # 로깅 종료
    finish_logging()

