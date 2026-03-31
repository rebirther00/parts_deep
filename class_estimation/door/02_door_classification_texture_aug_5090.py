"""
굴착기 도어 분류 모델 학습 스크립트 (Texture 불변 증강 버전)
- 실제 카메라(ZED X Mini)로 촬영한 도어 데이터셋 (RGB + Depth) 사용
- RGBD 4채널 입력 ResNet18 Transfer Learning
- Texture 불변 증강: RandomGrayscale, 강한 ColorJitter, GaussianBlur
  → 같은 종류의 다른 개체(door A→B) 추론 시 texture 의존 감소
- RTX 5090 GPU에 최적화된 설정
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

import cv2
from PIL import Image, ImageFilter

from depth_utils import (
    RGBDAuxResNet18, RGBDTransform, RGBDDataset, IN_CHANNELS,
    NUM_AUX_FEATURES, DEFAULT_INTRINSICS,
)


class TextureInvariantRGBDTransform:
    """RGBDTransform + texture 불변 증강 (학습 전용)

    RGB에만 적용되는 texture 파괴 증강:
    - RandomGrayscale(p=0.5): 50% 확률 흑백 → 색상 의존 차단
    - GaussianBlur(p=0.3): 미세 texture 흐리기
    - 강한 ColorJitter는 기존 RGBDTransform 내부에서 처리
    Depth 채널은 증강하지 않음 (물리적 의미 보존)
    """

    def __init__(self, image_size):
        self.base = RGBDTransform(image_size, is_train=True)

    def __call__(self, rgb_pil, depth_np):
        if random.random() < 0.5:
            rgb_pil = rgb_pil.convert('L').convert('RGB')

        if random.random() < 0.3:
            sigma = random.uniform(0.5, 2.0)
            rgb_pil = rgb_pil.filter(ImageFilter.GaussianBlur(radius=sigma))

        return self.base(rgb_pil, depth_np)

# ================================================================================
# 명령줄 인자 파싱
# ================================================================================
parser = argparse.ArgumentParser(description='굴착기 도어 분류 모델 학습 (RTX 5090 최적화)')
parser.add_argument('-cpu', '--cpu', action='store_true',
                    help='CPU로 강제 실행 (기본값: GPU 사용 가능 시 GPU 사용)')
DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
parser.add_argument('--dataset_dir', type=str, default=DEFAULT_DATASET_DIR,
                    help='데이터셋 경로 (기본: door/datasets)')
parser.add_argument('--full_train', action='store_true',
                    help='전체 데이터를 학습에 사용 (배포용, Train/Test 분할 없음)')
args = parser.parse_args()

# ================================================================================
# 로깅 설정
# ================================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(PROJECT_DIR))
sys.path.insert(0, REPO_DIR)
from utils.logger import setup_logging, finish_logging

LOG_PATH = setup_logging("02_door_classification_texture_aug_5090")

# ================================================================================
# 설정 변수
# ================================================================================
DATASET_DIR = args.dataset_dir
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "best_door_texture_aug_model_5090.pth")
ONNX_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "best_door_texture_aug_model_5090.onnx")
TRAIN_INDICES_PATH = os.path.join(ARTIFACTS_DIR, "training_indices_door_texture_aug_5090.json")

BATCH_SIZE = None   # None: 자동 조정
RANDOM_SEED = 42
TEST_SIZE = 0.3
NUM_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 10

IMAGE_SIZE = 448

# RTX 5090 최적화 설정
NUM_WORKERS = 8
PREFETCH_FACTOR = 2

total_start_time = time.time()

# ================================================================================
# 1. 데이터 로드 및 전처리
# ================================================================================
print("=" * 80)
print("1단계: 데이터셋 로드 및 전처리")
print("=" * 80)
step1_start_time = time.time()

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available() and not args.cpu:
    torch.cuda.manual_seed_all(RANDOM_SEED)

print(f"\n데이터셋 경로: {DATASET_DIR}")


def scan_dataset(dataset_dir):
    """데이터셋 폴더를 스캔하여 클래스별 이미지 경로 수집 (이미지가 없는 폴더 자동 제외)"""
    classes = []
    image_paths = []
    labels = []

    class_folders = sorted([d for d in os.listdir(dataset_dir)
                           if os.path.isdir(os.path.join(dataset_dir, d))])

    print(f"\n스캔된 폴더: {len(class_folders)}개")

    class_idx = 0
    for class_name in class_folders:
        class_path = os.path.join(dataset_dir, class_name)
        png_files = sorted(glob.glob(os.path.join(class_path, "rgb_*.png")))

        if len(png_files) == 0:
            print(f"  [건너뜀] {class_name}: 이미지 없음")
            continue

        print(f"  [{class_idx}] {class_name}: {len(png_files)}장")

        classes.append(class_name)
        for img_path in png_files:
            image_paths.append(img_path)
            labels.append(class_idx)
        class_idx += 1

    return classes, image_paths, labels


class_names, image_paths, labels = scan_dataset(DATASET_DIR)
num_classes = len(class_names)

print(f"\n총 이미지 수: {len(image_paths)}장")
print(f"유효 클래스 수: {num_classes}개")

class_names_path = os.path.join(ARTIFACTS_DIR, "class_names_door_texture_aug_5090.json")
with open(class_names_path, 'w', encoding='utf-8') as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)
print(f"클래스 이름 저장: {class_names_path}")

step1_time = time.time() - step1_start_time
print(f"\n[1단계 완료] 소요 시간: {step1_time:.2f}초")

# ================================================================================
# 2. PyTorch Dataset 클래스 정의
# ================================================================================
print("\n" + "=" * 80)
print("2단계: PyTorch Dataset 클래스 정의")
print("=" * 80)
step2_start_time = time.time()


train_transform = TextureInvariantRGBDTransform(IMAGE_SIZE)
val_transform = RGBDTransform(IMAGE_SIZE, is_train=False)

print(f"RGBD Texture 불변 증강 전처리 설정 완료:")
print(f"  입력 크기: {IMAGE_SIZE}x{IMAGE_SIZE}, 채널: {IN_CHANNELS} (R,G,B,D)")
print(f"  Train: Letterbox + Pad + Rotation(5°) + ColorJitter(RGB)")
print(f"         + RandomGrayscale(p=0.5) + GaussianBlur(p=0.3)")
print(f"  Val: Letterbox Resize + Pad + Normalize (증강 없음)")
print(f"  보조 피처: {NUM_AUX_FEATURES}개 (물리 치수 mm)")

step2_time = time.time() - step2_start_time
print(f"\n[2단계 완료] 소요 시간: {step2_time:.2f}초")

# ================================================================================
# 3. Train/Test 데이터 분할
# ================================================================================
print("\n" + "=" * 80)
print("3단계: Train/Test 데이터 분할")
print("=" * 80)
step3_start_time = time.time()

FULL_TRAIN = args.full_train

if FULL_TRAIN:
    print("\n⚡ 전체 학습 모드 (--full_train): 모든 데이터를 학습에 사용합니다.")
    train_paths = list(image_paths)
    train_labels = list(labels)
    test_paths = []
    test_labels = []

    train_data_info = {
        "mode": "full_train",
        "train_paths": train_paths,
        "class_names": class_names,
        "random_seed": RANDOM_SEED,
    }
else:
    indices = list(range(len(image_paths)))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_paths = [image_paths[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    train_data_info = {
        "mode": "split",
        "train_indices": train_indices,
        "test_indices": test_indices,
        "train_paths": train_paths,
        "test_paths": test_paths,
        "class_names": class_names,
        "random_seed": RANDOM_SEED,
    }

with open(TRAIN_INDICES_PATH, 'w', encoding='utf-8') as f:
    json.dump(train_data_info, f, ensure_ascii=False, indent=2)
print(f"학습 데이터 정보 저장: {TRAIN_INDICES_PATH}")

print(f"\nTrain 데이터: {len(train_paths)}장")
if not FULL_TRAIN:
    print(f"Test 데이터: {len(test_paths)}장")

print("\n클래스별 분포:")
for class_idx, class_name in enumerate(class_names):
    train_count = sum(1 for l in train_labels if l == class_idx)
    if FULL_TRAIN:
        print(f"  {class_name}: {train_count}장 (전체)")
    else:
        test_count = sum(1 for l in test_labels if l == class_idx)
        print(f"  {class_name}: Train {train_count}장, Test {test_count}장")


def adjust_batch_size_5090(data_size, force_cpu=False):
    """RTX 5090 GPU에 최적화된 배치 사이즈 조정 (448x448 기준)"""
    if data_size < 100:
        batch_size = 16
    elif data_size < 500:
        batch_size = 32
    elif data_size < 2000:
        batch_size = 64
    else:
        batch_size = 64

    if torch.cuda.is_available() and not force_cpu:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb >= 24:
            if data_size >= 2000:
                batch_size = min(batch_size, 128)
            elif data_size >= 500:
                batch_size = min(batch_size, 64)
        elif gpu_memory_gb >= 16:
            if data_size >= 2000:
                batch_size = min(batch_size, 64)
            elif data_size >= 500:
                batch_size = min(batch_size, 32)
        elif gpu_memory_gb >= 8:
            batch_size = min(batch_size, 16)

    return batch_size


if BATCH_SIZE is None:
    batch_size = adjust_batch_size_5090(len(train_paths), force_cpu=args.cpu)
    print(f"\n배치 사이즈 자동 설정 (RTX 5090 최적화): {batch_size}")
else:
    batch_size = BATCH_SIZE
    print(f"\n배치 사이즈 고정: {batch_size}")

train_dataset = RGBDDataset(train_paths, train_labels, transform=train_transform)

num_workers = NUM_WORKERS if torch.cuda.is_available() and not args.cpu else 0
pin_memory = torch.cuda.is_available() and not args.cpu

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    prefetch_factor=PREFETCH_FACTOR if num_workers > 0 else None,
    persistent_workers=True if num_workers > 0 else False
)

test_loader = None
if not FULL_TRAIN:
    test_dataset = RGBDDataset(test_paths, test_labels, transform=val_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=PREFETCH_FACTOR if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )

print(f"Train Batch 개수: {len(train_loader)}")
if test_loader:
    print(f"Test Batch 개수: {len(test_loader)}")
print(f"DataLoader 설정: num_workers={num_workers}, pin_memory={pin_memory}, "
      f"prefetch_factor={PREFETCH_FACTOR if num_workers > 0 else 'N/A'}")

step3_time = time.time() - step3_start_time
print(f"\n[3단계 완료] 소요 시간: {step3_time:.2f}초")

# ================================================================================
# 4. Transfer Learning 모델 정의 (ResNet18)
# ================================================================================
print("\n" + "=" * 80)
print("4단계: Transfer Learning 모델 정의 (ResNet18)")
print("=" * 80)
step4_start_time = time.time()


def create_resnet_model(num_classes, pretrained=True):
    """RGBD 4채널 + 보조 피처 입력 ResNet18 모델 생성"""
    return RGBDAuxResNet18(num_classes, pretrained=pretrained)


if args.cpu:
    device = torch.device('cpu')
    print("\n[디바이스 설정] CPU로 강제 실행 (--cpu 플래그 사용)")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n[디바이스 설정] GPU 사용: {gpu_name}")
        print(f"  GPU 메모리: {gpu_memory_gb:.2f} GB")
        if '5090' in gpu_name or gpu_memory_gb >= 24:
            print("  RTX 5090 또는 고성능 GPU 감지 - 최적화 설정 적용")
    else:
        print("\n[디바이스 설정] CUDA 사용 불가, CPU로 실행")

model = create_resnet_model(num_classes=num_classes, pretrained=True).to(device)

print(f"RGBDAuxResNet18 모델 로드 완료 (RGB: ImageNet 가중치, D: 평균 초기화, Aux: {NUM_AUX_FEATURES}피처)")
print(f"출력 클래스 수: {num_classes}")
print(f"모델 파라미터 개수: {sum(p.numel() for p in model.parameters()):,}")

step4_time = time.time() - step4_start_time
print(f"\n[4단계 완료] 소요 시간: {step4_time:.2f}초")

# ================================================================================
# 5. 손실 함수 및 옵티마이저 설정
# ================================================================================
print("\n" + "=" * 80)
print("5단계: 손실 함수 및 옵티마이저 설정")
print("=" * 80)
step5_start_time = time.time()

class_counts = [sum(1 for l in train_labels if l == i) for i in range(num_classes)]
total_count = len(train_labels)
class_weights = torch.tensor([total_count / (num_classes * c) for c in class_counts],
                             dtype=torch.float32).to(device)

print(f"클래스별 분포: {dict(zip(class_names, class_counts))}")
print(f"클래스 가중치: {dict(zip(class_names, [f'{w:.4f}' for w in class_weights.cpu().numpy()]))}")

criterion = nn.CrossEntropyLoss(weight=class_weights)

LEARNING_RATE = 0.001
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

print("\n손실 함수: Cross Entropy Loss (클래스 가중치 적용)")
print(f"옵티마이저: Adam (lr={LEARNING_RATE})")
print("스케줄러: ReduceLROnPlateau (patience=3, factor=0.5)")

step5_time = time.time() - step5_start_time
print(f"\n[5단계 완료] 소요 시간: {step5_time:.2f}초")

# ================================================================================
# 6. 학습 및 평가 함수 정의
# ================================================================================
print("\n" + "=" * 80)
print("6단계: 학습 및 평가 함수 정의")
print("=" * 80)
step6_start_time = time.time()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """한 에포크 학습"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, aux, labels in dataloader:
        images = images.to(device, non_blocking=True)
        aux = aux.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images, aux)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = 100.0 * correct / total

    return epoch_loss, epoch_accuracy


def evaluate(model, dataloader, criterion, device, class_names):
    """모델 평가 (클래스별 정확도 포함)"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)

    with torch.no_grad():
        for images, aux, labels in dataloader:
            images = images.to(device, non_blocking=True)
            aux = aux.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images, aux)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = 100.0 * correct / total

    class_accuracies = {}
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            class_accuracies[name] = 100.0 * class_correct[i] / class_total[i]
        else:
            class_accuracies[name] = 0.0

    return epoch_loss, epoch_accuracy, class_accuracies


print("학습 및 평가 함수 정의 완료 (RTX 5090 최적화: non_blocking 전송)")

step6_time = time.time() - step6_start_time
print(f"\n[6단계 완료] 소요 시간: {step6_time:.2f}초")

# ================================================================================
# 7. 모델 학습 실행
# ================================================================================
print("\n" + "=" * 80)
print("7단계: 모델 학습 시작")
print("=" * 80)
step7_start_time = time.time()

best_train_loss = float('inf')
best_val_accuracy = 0.0
best_val_loss = float('inf')
patience_counter = 0

print(f"총 에포크: {NUM_EPOCHS}")
print(f"배치 크기: {batch_size} (RTX 5090 최적화)")
if FULL_TRAIN:
    print(f"모드: 전체 학습 (Train Loss 기반 Early Stopping)")
else:
    print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE} 에포크")
print("\n학습 시작...\n")

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

    if FULL_TRAIN:
        scheduler.step(train_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
    else:
        val_loss, val_acc, class_accs = evaluate(model, test_loader, criterion, device, class_names)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1

    if torch.cuda.is_available() and not args.cpu:
        if (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        if FULL_TRAIN:
            print(f"  Best Train Loss: {best_train_loss:.4f} | LR: {current_lr:.6f}")
        else:
            print(f"  Val Loss:   {val_loss:.4f} | Val Accuracy:   {val_acc:.2f}%")
            print(f"  클래스별 정확도:")
            for name, acc in class_accs.items():
                print(f"    - {name}: {acc:.2f}%")
            print(f"  Best Val Accuracy: {best_val_accuracy:.2f}% | LR: {current_lr:.6f}")
        if patience_counter > 0:
            print(f"  Early Stopping: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        print("-" * 60)

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly Stopping: {EARLY_STOPPING_PATIENCE} 에포크 동안 개선이 없어 학습을 중단합니다.")
        break

print("\n학습 완료!")
if FULL_TRAIN:
    print(f"최저 Train Loss: {best_train_loss:.4f}")
else:
    print(f"최고 Validation Accuracy: {best_val_accuracy:.2f}%")
print(f"모델 저장 위치: {MODEL_SAVE_PATH}")

step7_time = time.time() - step7_start_time
print(f"\n[7단계 완료] 소요 시간: {step7_time:.2f}초 ({step7_time/60:.2f}분)")

# ================================================================================
# 8. 최종 평가
# ================================================================================
print("\n" + "=" * 80)
print("8단계: 최종 평가")
print("=" * 80)
step8_start_time = time.time()

if FULL_TRAIN:
    print("\n⚡ 전체 학습 모드: 별도 테스트 셋이 없으므로 학습 데이터에 대한 최종 정확도를 확인합니다.")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    final_loss, final_acc, final_class_accs = evaluate(model, train_loader, criterion, device, class_names)

    print(f"\n최종 학습 데이터 결과 (참고용):")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Accuracy: {final_acc:.2f}%")
    print(f"\n클래스별 정확도:")
    for name, acc in final_class_accs.items():
        print(f"  - {name}: {acc:.2f}%")
else:
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    final_loss, final_acc, final_class_accs = evaluate(model, test_loader, criterion, device, class_names)

    print(f"\n최종 테스트 결과:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Accuracy: {final_acc:.2f}%")
    print(f"\n클래스별 정확도:")
    for name, acc in final_class_accs.items():
        print(f"  - {name}: {acc:.2f}%")

step8_time = time.time() - step8_start_time
print(f"\n[8단계 완료] 소요 시간: {step8_time:.2f}초")

# ================================================================================
# 9. ONNX 변환 (ZED Box Mini 추론용)
# ================================================================================
print("\n" + "=" * 80)
print("9단계: ONNX 변환 (ZED Box Mini 추론용)")
print("=" * 80)
step9_start_time = time.time()

try:
    model_cpu = create_resnet_model(num_classes=num_classes, pretrained=False)
    model_cpu.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location='cpu'))
    model_cpu.eval()
    dummy_images = torch.randn(1, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    dummy_aux = torch.randn(1, NUM_AUX_FEATURES)

    torch.onnx.export(
        model_cpu,
        (dummy_images, dummy_aux),
        ONNX_SAVE_PATH,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['images', 'aux_features'],
        output_names=['output'],
        dynamic_axes={
            'images': {0: 'batch_size'},
            'aux_features': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False
    )

    # 외부 데이터 파일이 남아있으면 삭제
    onnx_data_path = ONNX_SAVE_PATH + ".data"
    if os.path.exists(onnx_data_path):
        os.remove(onnx_data_path)
        print(f"외부 데이터 파일 삭제: {onnx_data_path}")

    onnx_size_mb = os.path.getsize(ONNX_SAVE_PATH) / (1024 * 1024)
    print(f"ONNX 모델 저장: {ONNX_SAVE_PATH}")
    print(f"ONNX 모델 크기: {onnx_size_mb:.2f} MB (가중치 내장)")
except Exception as e:
    print(f"\n[경고] ONNX 변환 실패: {e}")
    print("  → pip install onnxscript onnx 설치 후 다시 실행하세요.")
    print("  → 학습된 모델(.pth)은 정상 저장되어 있습니다.")

step9_time = time.time() - step9_start_time
print(f"\n[9단계 완료] 소요 시간: {step9_time:.2f}초")

# ================================================================================
# 전체 실행 시간 요약
# ================================================================================
total_time = time.time() - total_start_time

print("\n" + "=" * 80)
print("모든 작업 완료!")
print("=" * 80)
print(f"\n[전체 실행 시간 요약]")
print(f"  모드: {'전체 학습 (--full_train)' if FULL_TRAIN else f'분할 학습 ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)})'}")
print(f"  디바이스: {device}")
print(f"  배치 사이즈: {batch_size} (RTX 5090 최적화)")
print(f"  워커 수: {num_workers}")
print(f"  1단계 (데이터셋 로드): {step1_time:.2f}초")
print(f"  2단계 (Dataset 클래스 정의): {step2_time:.2f}초")
print(f"  3단계 (Train/Test 분할): {step3_time:.2f}초")
print(f"  4단계 (모델 정의): {step4_time:.2f}초")
print(f"  5단계 (손실 함수 및 옵티마이저): {step5_time:.2f}초")
print(f"  6단계 (학습/평가 함수 정의): {step6_time:.2f}초")
print(f"  7단계 (모델 학습): {step7_time:.2f}초 ({step7_time/60:.2f}분)")
print(f"  8단계 (최종 평가): {step8_time:.2f}초")
print(f"  9단계 (ONNX 변환): {step9_time:.2f}초")
print(f"  ─────────────────────────────────────────────")
print(f"  총 실행 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
print("=" * 80)

finish_logging()
