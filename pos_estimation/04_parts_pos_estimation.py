"""
굴착기 부품 위치 추정(작업대 XY) 학습 스크립트

목표(스몰 프로젝트/패스트 스터디):
- 입력: bbox crop된 부품 이미지(Isaac Sim 합성)
- 출력: 작업대(world) 기준 XY 위치 (meter)
- 평가: 평균 위치 오차(mean position error, mm) 별도 관리

데이터셋:
- /home/rebirther/isaac_data_output/dataset_pos
  - rgb_####.png + bounding_box_2d_tight_####.npy + bounding_box_2d_tight_labels_####.json + pose_####.json

주의:
- 회전/자세는 이번 스텝에서는 학습하지 않습니다(요구사항: 작업대 XY).
- bbox crop은 데이터셋에 "미리" 적용되어 있지 않으므로, 로더에서 bbox로 crop합니다.
"""

import argparse
import glob
import json
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ================================================================================
# 명령줄 인자
# ================================================================================
parser = argparse.ArgumentParser(description="굴착기 부품 작업대 XY 위치 추정 학습")
DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_pos")
DEFAULT_ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR, help="dataset_pos 경로 (기본: pos_estimation/dataset_pos)")
parser.add_argument("--image_size", type=int, default=224, help="입력 이미지 크기(ResNet 기본 224)")
parser.add_argument("--test_size", type=float, default=0.2, help="테스트셋 비율")
parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
parser.add_argument("--epochs", type=int, default=40, help="에포크")
parser.add_argument("--batch_size", type=int, default=32, help="배치 크기")
parser.add_argument("--lr", type=float, default=1e-3, help="학습률")
parser.add_argument("--cpu", action="store_true", help="CPU 강제 실행")
parser.add_argument("--model_out", type=str, default=os.path.join(DEFAULT_ARTIFACTS_DIR, "best_parts_xy_regressor.pth"), help="모델 저장 경로")
parser.add_argument("--stats_out", type=str, default=os.path.join(DEFAULT_ARTIFACTS_DIR, "xy_normalization_stats.json"), help="정규화 통계 저장 경로")
args = parser.parse_args()


# ================================================================================
# 재현성
# ================================================================================
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.cpu:
    torch.cuda.manual_seed_all(args.seed)

device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

# artifacts 디렉토리 보장
os.makedirs(os.path.dirname(os.path.abspath(args.model_out)), exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(args.stats_out)), exist_ok=True)


# ================================================================================
# 데이터 스캔
# ================================================================================
@dataclass
class Sample:
    rgb_path: str
    bbox_path: str
    labels_path: str
    pose_path: str
    class_name: str


def _frame_id_from_rgb(rgb_path: str) -> str:
    base = os.path.basename(rgb_path)
    return base.replace("rgb_", "").replace(".png", "")


def scan_dataset_pos(dataset_dir: str):
    """dataset_pos 폴더를 스캔하여 샘플 목록 생성"""
    class_folders = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])
    samples: list[Sample] = []
    for class_name in class_folders:
        class_dir = os.path.join(dataset_dir, class_name)
        rgb_files = sorted(glob.glob(os.path.join(class_dir, "rgb_*.png")))
        for rgb_path in rgb_files:
            fid = _frame_id_from_rgb(rgb_path)
            bbox_path = os.path.join(class_dir, f"bounding_box_2d_tight_{fid}.npy")
            labels_path = os.path.join(class_dir, f"bounding_box_2d_tight_labels_{fid}.json")
            pose_path = os.path.join(class_dir, f"pose_{fid}.json")
            if not (os.path.exists(bbox_path) and os.path.exists(labels_path) and os.path.exists(pose_path)):
                continue
            samples.append(Sample(rgb_path, bbox_path, labels_path, pose_path, class_name))
    return class_folders, samples


def _select_object_bbox(bboxes: np.ndarray, labels_map: dict):
    """
    bbox npy에서 'background가 아닌' bbox 중 가장 큰 bbox를 선택.
    반환: (x_min, y_min, x_max, y_max) (int)
    """
    best = None
    best_area = -1
    for bb in bboxes:
        sid = str(int(bb["semanticId"]))
        cls = labels_map.get(sid, {}).get("class", "unknown")
        if cls == "background":
            continue
        x_min = int(bb["x_min"])
        y_min = int(bb["y_min"])
        x_max = int(bb["x_max"])
        y_max = int(bb["y_max"])
        area = max(0, x_max - x_min) * max(0, y_max - y_min)
        if area > best_area:
            best_area = area
            best = (x_min, y_min, x_max, y_max)
    return best


def _crop_with_bbox(img: Image.Image, bbox_xyxy):
    """PIL 이미지에서 bbox로 crop. bbox는 inclusive로 들어올 수 있어 right/bottom에 +1 적용."""
    w, h = img.size
    x_min, y_min, x_max, y_max = bbox_xyxy
    x_min = max(0, min(w - 1, x_min))
    y_min = max(0, min(h - 1, y_min))
    x_max = max(0, min(w - 1, x_max))
    y_max = max(0, min(h - 1, y_max))
    # PIL crop은 right/bottom exclusive
    return img.crop((x_min, y_min, x_max + 1, y_max + 1))


class PartsXYDataset(Dataset):
    """bbox crop + 작업대(world) XY 회귀 Dataset"""

    def __init__(self, samples: list[Sample], transform, xy_mean, xy_std):
        self.samples = samples
        self.transform = transform
        self.xy_mean = np.asarray(xy_mean, dtype=np.float32)
        self.xy_std = np.asarray(xy_std, dtype=np.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        img = Image.open(s.rgb_path)
        if img.mode != "RGB":
            img = img.convert("RGB")

        bboxes = np.load(s.bbox_path, allow_pickle=True)
        with open(s.labels_path, "r") as f:
            labels_map = json.load(f)
        bbox = _select_object_bbox(bboxes, labels_map)
        if bbox is None:
            # 예외 케이스: background만 있는 프레임이면 전체 이미지를 사용
            cropped = img
        else:
            cropped = _crop_with_bbox(img, bbox)

        with open(s.pose_path, "r") as f:
            pose = json.load(f)
        # 카메라 기준 오브젝트 상대 위치 사용 (camTobj)
        t_cam = pose["camTobj"]["t_xyz_m"]
        xy = np.asarray([float(t_cam[0]), float(t_cam[1])], dtype=np.float32)
        xy_norm = (xy - self.xy_mean) / (self.xy_std + 1e-8)

        if self.transform:
            cropped = self.transform(cropped)

        return cropped, torch.tensor(xy_norm, dtype=torch.float32)


def _compute_xy_stats(train_samples: list[Sample]):
    """train split에서 카메라 기준 오브젝트 XY(camTobj)의 mean/std 계산"""
    xs = []
    for s in train_samples:
        with open(s.pose_path, "r") as f:
            pose = json.load(f)
        t_cam = pose["camTobj"]["t_xyz_m"]
        xs.append([float(t_cam[0]), float(t_cam[1])])
    arr = np.asarray(xs, dtype=np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.tolist(), std.tolist()


# ================================================================================
# 실행
# ================================================================================
print("=" * 80)
print("굴착기 부품 작업대 XY 위치 추정 학습")
print("=" * 80)
print(f"dataset_dir: {args.dataset_dir}")
print(f"device: {device}")
print(f"image_size: {args.image_size}")

class_names, samples = scan_dataset_pos(args.dataset_dir)
print(f"\n클래스 수: {len(class_names)}개")
print(f"총 샘플 수: {len(samples)}개")

if len(samples) == 0:
    raise RuntimeError("샘플이 없습니다. dataset_pos 생성 여부/경로를 확인하세요.")

# train/test split (클래스 비율 유지)
labels_for_stratify = [class_names.index(s.class_name) for s in samples]
indices = list(range(len(samples)))
train_idx, test_idx = train_test_split(
    indices, test_size=args.test_size, random_state=args.seed, stratify=labels_for_stratify
)
train_samples = [samples[i] for i in train_idx]
test_samples = [samples[i] for i in test_idx]
print(f"Train: {len(train_samples)} / Test: {len(test_samples)}")

# 정규화 통계(train 기준)
xy_mean, xy_std = _compute_xy_stats(train_samples)
with open(args.stats_out, "w", encoding="utf-8") as f:
    json.dump({"xy_mean": xy_mean, "xy_std": xy_std, "unit": "m"}, f, ensure_ascii=False, indent=2)
print(f"정규화 통계 저장: {args.stats_out}")

# 변환(회귀이므로 flip/rotation은 라벨 보정이 필요 → 최소 전처리만)
train_tf = transforms.Compose(
    [
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_tf = transforms.Compose(
    [
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_ds = PartsXYDataset(train_samples, train_tf, xy_mean, xy_std)
test_ds = PartsXYDataset(test_samples, test_tf, xy_mean, xy_std)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

# 모델(ResNet18 + 회귀 헤드)
model = models.resnet18(pretrained=True)
in_features = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, 128), nn.ReLU(), nn.Linear(128, 2))
model = model.to(device)

criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)


def evaluate_mean_pos_error_mm(model, loader):
    model.eval()
    errs_mm = []
    x_abs_mm = []
    y_abs_mm = []
    with torch.no_grad():
        for images, xy_norm in loader:
            images = images.to(device)
            xy_norm = xy_norm.to(device)
            pred_norm = model(images)
            # 역정규화
            mean = torch.tensor(xy_mean, dtype=torch.float32, device=device).view(1, 2)
            std = torch.tensor(xy_std, dtype=torch.float32, device=device).view(1, 2)
            pred = pred_norm * std + mean
            gt = xy_norm * std + mean
            diff = pred - gt
            dist = torch.linalg.norm(diff, dim=1)  # meters
            errs_mm.extend((dist * 1000.0).detach().cpu().numpy().tolist())
            x_abs_mm.extend((diff[:, 0].abs() * 1000.0).detach().cpu().numpy().tolist())
            y_abs_mm.extend((diff[:, 1].abs() * 1000.0).detach().cpu().numpy().tolist())
    return float(np.mean(errs_mm)), float(np.mean(x_abs_mm)), float(np.mean(y_abs_mm))


best_mpe = float("inf")
start_time = time.time()

print("\n학습 시작...")
for epoch in range(args.epochs):
    model.train()
    running = 0.0
    for images, xy_norm in train_loader:
        images = images.to(device)
        xy_norm = xy_norm.to(device)

        optimizer.zero_grad()
        pred = model(images)
        loss = criterion(pred, xy_norm)
        loss.backward()
        optimizer.step()

        running += loss.item() * images.size(0)

    train_loss = running / len(train_loader.dataset)
    mpe_mm, x_mae_mm, y_mae_mm = evaluate_mean_pos_error_mm(model, test_loader)

    if mpe_mm < best_mpe:
        best_mpe = mpe_mm
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "xy_mean": xy_mean,
                "xy_std": xy_std,
                "image_size": args.image_size,
                "class_names": class_names,
                "note": "bbox crop + world xy regression (meters), metrics in mm",
            },
            args.model_out,
        )

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(
            f"Epoch [{epoch+1:02d}/{args.epochs}] "
            f"train_loss={train_loss:.4f} | "
            f"MeanPosErr={mpe_mm:.2f}mm (x={x_mae_mm:.2f}mm, y={y_mae_mm:.2f}mm) | "
            f"best={best_mpe:.2f}mm"
        )

total_time = time.time() - start_time
print("\n학습 완료!")
print(f"Best Mean Position Error: {best_mpe:.2f} mm")
print(f"모델 저장: {args.model_out}")
print(f"총 소요 시간: {total_time/60:.2f} 분")


