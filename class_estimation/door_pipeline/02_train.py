"""
논문용 통합 학습 스크립트
- 6종 모델(rgbd, texture_aug, edge, rgbe, rgbe_texture_aug, + no_aux 변형) 지원
- Train/Val/Test 70/15/15 stratified split
- Val accuracy 기반 early stopping (data leakage 제거)
- Epoch별 메트릭 JSON 로깅
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import random
import os
import sys
import argparse
import time
import json
import glob

# ── 프로젝트 경로 설정 ──────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(PROJECT_DIR))
sys.path.insert(0, REPO_DIR)

from torchvision import models

from depth_utils import (
    RGBDAuxResNet18, RGBDTransform, RGBDDataset, IN_CHANNELS,
    NUM_AUX_FEATURES,
)
from rgb_utils import RGBTransform, RGBDataset, RGB_IN_CHANNELS
from rgbe_utils import RGBETransform, RGBEDataset, RGBE_IN_CHANNELS
from edge_utils import EdgeAuxResNet18, EdgeTransform, EdgeDataset, EDGE_IN_CHANNELS

from PIL import ImageFilter
from depth_utils import RGBDTransform as _BaseRGBDTransform


# ── Aux 없는 모델 (ablation용) ──────────────────────────
class NoAuxResNet18(nn.Module):
    """Aux MLP 없이 이미지만 사용하는 분류 모델 (ablation용)

    RGBDAuxResNet18과 동일한 backbone이지만 aux branch를 완전 제거.
    forward()는 aux_features를 인자로 받되 무시하여 API 호환성 유지.
    """

    def __init__(self, num_classes, in_channels=IN_CHANNELS, pretrained=True):
        super().__init__()

        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(in_channels, 64,
                                 kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                for c in range(3, in_channels):
                    new_conv.weight[:, c:c+1] = old_conv.weight.mean(
                        dim=1, keepdim=True)
            backbone.conv1 = new_conv

        self.backbone_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, images, aux_features):
        img_feat = self.backbone(images)
        return self.classifier(img_feat)


class TextureInvariantRGBDTransform:
    """RGBD + texture 불변 증강 (학습 전용)

    RandomGrayscale(p=0.5) + GaussianBlur(p=0.3)로 색상/질감 의존 차단.
    """

    def __init__(self, image_size):
        self.base = _BaseRGBDTransform(image_size, is_train=True)

    def __call__(self, rgb_pil, depth_np):
        if random.random() < 0.5:
            rgb_pil = rgb_pil.convert('L').convert('RGB')
        if random.random() < 0.3:
            sigma = random.uniform(0.5, 2.0)
            rgb_pil = rgb_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
        return self.base(rgb_pil, depth_np)


class TextureInvariantRGBETransform:
    """RGBE + texture 불변 증강 (학습 전용)

    RGB에 RandomGrayscale(p=0.5) + GaussianBlur(p=0.3)를 적용한 후
    RGBETransform에 전달. Edge는 변환된 RGB에서 계산됨.
    """

    def __init__(self, image_size):
        self.base = RGBETransform(image_size, is_train=True)

    def __call__(self, rgb_pil):
        if random.random() < 0.5:
            rgb_pil = rgb_pil.convert('L').convert('RGB')
        if random.random() < 0.3:
            sigma = random.uniform(0.5, 2.0)
            rgb_pil = rgb_pil.filter(ImageFilter.GaussianBlur(radius=sigma))
        return self.base(rgb_pil)


# ── 모델 타입별 설정 ────────────────────────────────────
MODEL_CONFIGS = {
    "rgb": {
        "dataset_cls": RGBDataset,
        "train_transform_fn": lambda sz: RGBTransform(sz, is_train=True),
        "val_transform_fn": lambda sz: RGBTransform(sz, is_train=False),
        "model_fn": lambda nc: NoAuxResNet18(nc, in_channels=RGB_IN_CHANNELS,
                                              pretrained=True),
        "model_load_fn": lambda nc: NoAuxResNet18(nc, in_channels=RGB_IN_CHANNELS,
                                                   pretrained=False),
        "in_channels": RGB_IN_CHANNELS,
    },
    "rgbd": {
        "dataset_cls": RGBDDataset,
        "train_transform_fn": lambda sz: RGBDTransform(sz, is_train=True),
        "val_transform_fn": lambda sz: RGBDTransform(sz, is_train=False),
        "model_fn": lambda nc: RGBDAuxResNet18(nc, pretrained=True),
        "model_load_fn": lambda nc: RGBDAuxResNet18(nc, pretrained=False),
        "in_channels": IN_CHANNELS,
    },
    "texture_aug": {
        "dataset_cls": RGBDDataset,
        "train_transform_fn": lambda sz: TextureInvariantRGBDTransform(sz),
        "val_transform_fn": lambda sz: RGBDTransform(sz, is_train=False),
        "model_fn": lambda nc: RGBDAuxResNet18(nc, pretrained=True),
        "model_load_fn": lambda nc: RGBDAuxResNet18(nc, pretrained=False),
        "in_channels": IN_CHANNELS,
    },
    "edge": {
        "dataset_cls": EdgeDataset,
        "train_transform_fn": lambda sz: EdgeTransform(sz, is_train=True),
        "val_transform_fn": lambda sz: EdgeTransform(sz, is_train=False),
        "model_fn": lambda nc: EdgeAuxResNet18(nc, pretrained=True),
        "model_load_fn": lambda nc: EdgeAuxResNet18(nc, pretrained=False),
        "in_channels": EDGE_IN_CHANNELS,
    },
    "rgbe": {
        "dataset_cls": RGBEDataset,
        "train_transform_fn": lambda sz: RGBETransform(sz, is_train=True),
        "val_transform_fn": lambda sz: RGBETransform(sz, is_train=False),
        "model_fn": lambda nc: RGBDAuxResNet18(nc, pretrained=True),
        "model_load_fn": lambda nc: RGBDAuxResNet18(nc, pretrained=False),
        "in_channels": IN_CHANNELS,
    },
    "rgbe_texture_aug": {
        "dataset_cls": RGBEDataset,
        "train_transform_fn": lambda sz: TextureInvariantRGBETransform(sz),
        "val_transform_fn": lambda sz: RGBETransform(sz, is_train=False),
        "model_fn": lambda nc: RGBDAuxResNet18(nc, pretrained=True),
        "model_load_fn": lambda nc: RGBDAuxResNet18(nc, pretrained=False),
        "in_channels": IN_CHANNELS,
    },
}

# ── CLI ──────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='논문용 통합 학습')
parser.add_argument('--model_type', type=str, required=True,
                    choices=list(MODEL_CONFIGS.keys()))
parser.add_argument('--seed', type=int, default=None,
                    help='미지정 시 랜덤 시드 자동 생성 '
                         '(run 이름·split_info.json에 기록되어 재현 가능)')
parser.add_argument('--image_size', type=int, default=448)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--no_aux', action='store_true',
                    help='Aux MLP 제거 ablation 실험')
parser.add_argument('-cpu', '--cpu', action='store_true')
args = parser.parse_args()
if args.seed is None:
    args.seed = random.randint(0, 99999)
    print(f"시드 미지정 → 랜덤 시드 사용: {args.seed}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def scan_dataset(dataset_dir):
    """데이터셋 폴더를 스캔하여 클래스별 이미지 경로 수집"""
    classes, image_paths, labels = [], [], []
    class_folders = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])
    class_idx = 0
    for class_name in class_folders:
        class_path = os.path.join(dataset_dir, class_name)
        png_files = sorted(glob.glob(os.path.join(class_path, "rgb_*.png")))
        if not png_files:
            continue
        classes.append(class_name)
        for img_path in png_files:
            image_paths.append(img_path)
            labels.append(class_idx)
        class_idx += 1
    return classes, image_paths, labels


def split_data(image_paths, labels, seed):
    """Stratified 2단계 분할: 70% train / 15% val / 15% test"""
    indices = list(range(len(image_paths)))
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=seed, stratify=labels
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=seed, stratify=temp_labels
    )
    return train_idx, val_idx, test_idx


def auto_batch_size(n_train, image_size, force_cpu=False):
    if n_train < 100:
        base = 16
    elif n_train < 500:
        base = 32
    else:
        base = 64
    scale = (448 / image_size) ** 2
    bs = min(int(base * scale), 128)
    if torch.cuda.is_available() and not force_cpu:
        mem_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3) \
            if hasattr(torch.cuda.get_device_properties(0), 'total_mem') else 32
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if mem_gb < 16:
            bs = min(bs, 32)
    return bs


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, aux, labels in loader:
        images = images.to(device, non_blocking=True)
        aux = aux.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images, aux)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    return total_loss / len(loader.dataset), 100.0 * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, aux, labels in loader:
            images = images.to(device, non_blocking=True)
            aux = aux.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images, aux)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    return total_loss / len(loader.dataset), 100.0 * correct / total


def main():
    cfg = MODEL_CONFIGS[args.model_type]
    suffix = "_noaux" if args.no_aux else ""
    run_name = f"{args.model_type}{suffix}_{args.image_size}_seed{args.seed}"
    run_dir = os.path.join(PROJECT_DIR, "artifacts", run_name)
    os.makedirs(run_dir, exist_ok=True)

    model_path = os.path.join(run_dir, "model.pth")
    if os.path.exists(model_path):
        print(f"[건너뜀] 이미 학습 완료: {run_name}")
        return

    # 로깅 설정
    from utils.logger import setup_logging, finish_logging
    log_path = setup_logging(f"train_{run_name}",
                             log_dir=os.path.join(PROJECT_DIR, "logs"))

    print(f"실험: {run_name}")
    print(f"  모델: {args.model_type}, 해상도: {args.image_size}, seed: {args.seed}")
    start_time = time.time()

    set_seed(args.seed)

    # 데이터 로드 및 분할
    dataset_dir = os.path.join(PROJECT_DIR, "datasets")
    class_names, image_paths, labels = scan_dataset(dataset_dir)
    num_classes = len(class_names)
    print(f"  클래스: {num_classes}개, 전체 이미지: {len(image_paths)}장")

    train_idx, val_idx, test_idx = split_data(image_paths, labels, args.seed)
    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_paths = [image_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"  Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    # 분할 정보 저장
    split_info = {
        "seed": args.seed,
        "model_type": args.model_type,
        "image_size": args.image_size,
        "class_names": class_names,
        "train_paths": train_paths,
        "val_paths": val_paths,
        "test_paths": test_paths,
    }
    with open(os.path.join(run_dir, "split_info.json"), 'w', encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    # Dataset & DataLoader
    train_transform = cfg["train_transform_fn"](args.image_size)
    val_transform = cfg["val_transform_fn"](args.image_size)

    train_dataset = cfg["dataset_cls"](train_paths, train_labels,
                                       transform=train_transform)
    val_dataset = cfg["dataset_cls"](val_paths, val_labels,
                                     transform=val_transform)

    batch_size = auto_batch_size(len(train_paths), args.image_size, args.cpu)
    num_workers = 4 if torch.cuda.is_available() and not args.cpu else 0
    pin_memory = torch.cuda.is_available() and not args.cpu

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory)

    print(f"  배치 크기: {batch_size}, workers: {num_workers}")

    # 모델
    device = torch.device('cpu') if args.cpu else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.no_aux:
        in_ch = cfg.get("in_channels", IN_CHANNELS)
        model = NoAuxResNet18(num_classes, in_channels=in_ch,
                              pretrained=True).to(device)
    else:
        model = cfg["model_fn"](num_classes).to(device)
    print(f"  디바이스: {device}, 파라미터: {sum(p.numel() for p in model.parameters()):,}")

    # 손실 함수 (역빈도 클래스 가중치)
    class_counts = [sum(1 for l in train_labels if l == i)
                    for i in range(num_classes)]
    total_count = len(train_labels)
    weights = torch.tensor(
        [total_count / (num_classes * c) for c in class_counts],
        dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5)

    # 학습 루프
    best_val_acc = 0.0
    patience_counter = 0
    train_log = {"epochs": []}

    print(f"\n학습 시작 (max {args.epochs} epochs, patience {args.patience})")
    print("-" * 70)

    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device)
        scheduler.step(val_loss)

        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 4),
            "lr": optimizer.param_groups[0]['lr'],
        }
        train_log["epochs"].append(epoch_data)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or improved:
            mark = "*" if improved else ""
            print(f"  Epoch {epoch+1:3d} | "
                  f"Train {train_acc:6.2f}% L={train_loss:.4f} | "
                  f"Val {val_acc:6.2f}% L={val_loss:.4f} | "
                  f"Best {best_val_acc:.2f}% "
                  f"ES={patience_counter}/{args.patience} {mark}")

        if torch.cuda.is_available() and (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

    elapsed = time.time() - start_time
    train_log["best_val_acc"] = best_val_acc
    train_log["total_epochs"] = len(train_log["epochs"])
    train_log["elapsed_seconds"] = round(elapsed, 2)
    train_log["run_name"] = run_name

    with open(os.path.join(run_dir, "train_log.json"), 'w') as f:
        json.dump(train_log, f, indent=2)

    print(f"\n학습 완료: {run_name}")
    print(f"  Best Val Acc: {best_val_acc:.2f}%")
    print(f"  소요 시간: {elapsed/60:.1f}분")
    print(f"  모델 저장: {model_path}")

    finish_logging()


if __name__ == "__main__":
    main()
