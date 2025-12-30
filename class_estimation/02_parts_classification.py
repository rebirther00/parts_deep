"""
굴착기 부품 분류 모델 학습 스크립트
- Isaac Sim에서 생성한 데이터셋 사용
- ResNet18 Transfer Learning
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import random
import psutil
import os
import multiprocessing
import sys
import argparse
import time
import json
import glob

# ================================================================================
# 명령줄 인자 파싱
# ================================================================================
parser = argparse.ArgumentParser(description='굴착기 부품 분류 모델 학습')
parser.add_argument('-cpu', '--cpu', action='store_true', 
                    help='CPU로 강제 실행 (기본값: GPU 사용 가능 시 GPU 사용)')
DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
parser.add_argument('--dataset_dir', type=str, default=DEFAULT_DATASET_DIR,
                    help='데이터셋 경로 (기본: class_estimation/datasets)')
parser.add_argument('--bbox_crop', action='store_true',
                    help='bounding_box_2d_tight로 부품 bbox crop 후 학습 (dataset_pos 학습 시 권장)')
args = parser.parse_args()

# ================================================================================
# 로깅 설정
# ================================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PROJECT_DIR)
sys.path.insert(0, REPO_DIR)
from utils.logger import setup_logging, finish_logging

LOG_PATH = setup_logging("02_classification")

# ================================================================================
# 설정 변수
# ================================================================================
DATASET_DIR = args.dataset_dir  # 데이터셋 경로
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(ARTIFACTS_DIR, "best_parts_model.pth")  # 모델 저장 경로
TRAIN_INDICES_PATH = os.path.join(ARTIFACTS_DIR, "training_indices_parts.json")  # 학습 인덱스 저장 경로

BATCH_SIZE = None   # None: 자동 조정, 숫자: 고정 배치 사이즈
RANDOM_SEED = 42    # 재현성을 위한 랜덤 시드
TEST_SIZE = 0.2     # 테스트셋 비율 (20%)
NUM_EPOCHS = 30     # 학습 에포크 수
EARLY_STOPPING_PATIENCE = 5  # Early Stopping patience

# 이미지 크기 (데이터셋 이미지 크기에 따라 자동 조정)
# Isaac Sim에서 생성한 이미지는 1024x1024이지만, ResNet은 224x224 사용
IMAGE_SIZE = 224

# 전체 실행 시간 측정 시작
total_start_time = time.time()

# ================================================================================
# 1. 데이터 로드 및 전처리
# ================================================================================
print("=" * 80)
print("1단계: 데이터셋 로드 및 전처리")
print("=" * 80)
step1_start_time = time.time()

# 랜덤 시드 설정 (재현성)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available() and not args.cpu:
    torch.cuda.manual_seed_all(RANDOM_SEED)

# 데이터셋 폴더 스캔
print(f"\n데이터셋 경로: {DATASET_DIR}")

def scan_dataset(dataset_dir):
    """데이터셋 폴더를 스캔하여 클래스별 이미지 경로 수집"""
    classes = []
    image_paths = []
    labels = []
    
    # 하위 폴더 스캔 (각 폴더가 하나의 클래스)
    class_folders = sorted([d for d in os.listdir(dataset_dir) 
                           if os.path.isdir(os.path.join(dataset_dir, d))])
    
    print(f"\n발견된 클래스: {len(class_folders)}개")
    
    for class_idx, class_name in enumerate(class_folders):
        class_path = os.path.join(dataset_dir, class_name)
        
        # RGB 이미지 파일 검색
        png_files = sorted(glob.glob(os.path.join(class_path, "rgb_*.png")))
        
        print(f"  [{class_idx}] {class_name}: {len(png_files)}장")
        
        classes.append(class_name)
        for img_path in png_files:
            image_paths.append(img_path)
            labels.append(class_idx)
    
    return classes, image_paths, labels

# 데이터셋 스캔
class_names, image_paths, labels = scan_dataset(DATASET_DIR)
num_classes = len(class_names)

print(f"\n총 이미지 수: {len(image_paths)}장")
print(f"클래스 수: {num_classes}개")

# 클래스 이름 저장 (평가 시 사용)
class_names_path = os.path.join(ARTIFACTS_DIR, "class_names.json")
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


class ExcavatorPartsDataset(Dataset):
    """굴착기 부품 이미지 Dataset 클래스"""
    
    def __init__(self, image_paths, labels, transform=None, bbox_crop=False):
        """
        Args:
            image_paths: 이미지 파일 경로 리스트
            labels: 레이블 리스트 (클래스 인덱스)
            transform: 이미지 변환 (torchvision.transforms)
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.bbox_crop = bbox_crop
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        
        # RGB로 변환 (RGBA인 경우 대비)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # bbox crop (옵션): dataset_pos / datasets 모두 bounding_box_2d_tight를 가지고 있음
        if self.bbox_crop:
            frame_num = os.path.basename(img_path).replace('rgb_', '').replace('.png', '')
            class_dir = os.path.dirname(img_path)
            bbox_file = os.path.join(class_dir, f'bounding_box_2d_tight_{frame_num}.npy')
            label_file = os.path.join(class_dir, f'bounding_box_2d_tight_labels_{frame_num}.json')
            try:
                if os.path.exists(bbox_file) and os.path.exists(label_file):
                    bboxes = np.load(bbox_file, allow_pickle=True)
                    with open(label_file, 'r') as f:
                        labels_map = json.load(f)
                    # background가 아닌 bbox 중 가장 큰 bbox 선택
                    best = None
                    best_area = -1
                    for bb in bboxes:
                        sid = str(int(bb['semanticId']))
                        cls = labels_map.get(sid, {}).get('class', 'unknown')
                        if cls == 'background':
                            continue
                        x_min = int(bb['x_min']); y_min = int(bb['y_min'])
                        x_max = int(bb['x_max']); y_max = int(bb['y_max'])
                        area = max(0, x_max - x_min) * max(0, y_max - y_min)
                        if area > best_area:
                            best_area = area
                            best = (x_min, y_min, x_max, y_max)
                    if best is not None:
                        w, h = image.size
                        x_min, y_min, x_max, y_max = best
                        x_min = max(0, min(w - 1, x_min))
                        y_min = max(0, min(h - 1, y_min))
                        x_max = max(0, min(w - 1, x_max))
                        y_max = max(0, min(h - 1, y_max))
                        # PIL crop right/bottom exclusive
                        image = image.crop((x_min, y_min, x_max + 1, y_max + 1))
            except Exception:
                # crop 실패해도 학습은 계속 진행
                pass

        # 레이블
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # 이미지 전처리 적용
        if self.transform:
            image = self.transform(image)
        
        return image, label


# 이미지 전처리 정의
# Train용: Data Augmentation 포함
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Validation/Test용: Augmentation 없이 기본 전처리만
val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

print(f"이미지 전처리 설정 완료:")
print(f"  입력 이미지 크기: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"  Train: Resize + Augmentation (Flip, Rotation, ColorJitter) + Normalize")
print(f"  Validation: Resize + Normalize")

step2_time = time.time() - step2_start_time
print(f"\n[2단계 완료] 소요 시간: {step2_time:.2f}초")

# ================================================================================
# 3. Train/Test 데이터 분할
# ================================================================================
print("\n" + "=" * 80)
print("3단계: Train/Test 데이터 분할")
print("=" * 80)
step3_start_time = time.time()

# 인덱스 기반 분할 (클래스 비율 유지)
indices = list(range(len(image_paths)))
train_indices, test_indices = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=labels  # 클래스 비율 유지
)

# Train/Test 데이터 분리
train_paths = [image_paths[i] for i in train_indices]
train_labels = [labels[i] for i in train_indices]
test_paths = [image_paths[i] for i in test_indices]
test_labels = [labels[i] for i in test_indices]

# 학습에 사용한 인덱스 저장 (평가 시 참조)
train_data_info = {
    "train_indices": train_indices,
    "test_indices": test_indices,
    "train_paths": train_paths,
    "test_paths": test_paths,
    "class_names": class_names,
    "random_seed": RANDOM_SEED
}
with open(TRAIN_INDICES_PATH, 'w', encoding='utf-8') as f:
    json.dump(train_data_info, f, ensure_ascii=False, indent=2)
print(f"학습 데이터 정보 저장: {TRAIN_INDICES_PATH}")

print(f"\nTrain 데이터: {len(train_paths)}장")
print(f"Test 데이터: {len(test_paths)}장")

# 클래스별 분포 확인
print("\n클래스별 분포:")
for class_idx, class_name in enumerate(class_names):
    train_count = sum(1 for l in train_labels if l == class_idx)
    test_count = sum(1 for l in test_labels if l == class_idx)
    print(f"  {class_name}: Train {train_count}장, Test {test_count}장")

# 배치 사이즈 자동 조정
def adjust_batch_size(data_size, force_cpu=False):
    """데이터 크기와 GPU 메모리를 고려하여 배치 사이즈 조정"""
    # 기본 배치 사이즈 결정
    if data_size < 100:
        batch_size = 8
    elif data_size < 500:
        batch_size = 16
    elif data_size < 2000:
        batch_size = 32
    else:
        batch_size = 64
    
    # GPU 메모리 고려 (대용량 GPU 지원 추가)
    if torch.cuda.is_available() and not force_cpu:
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if gpu_memory_gb < 4:
            batch_size = min(batch_size, 16)
        elif gpu_memory_gb < 8:
            batch_size = min(batch_size, 32)
        elif gpu_memory_gb < 16:
            batch_size = max(batch_size, 64)
        elif gpu_memory_gb < 32:
            batch_size = max(batch_size, 128)
        else:
            # 32GB 이상 대용량 GPU (GB10 등)
            batch_size = max(batch_size, 256)
    
    return batch_size


def get_optimal_num_workers(force_cpu=False):
    """최적의 num_workers 값 계산"""
    cpu_count = multiprocessing.cpu_count()
    
    if force_cpu:
        # CPU 모드에서는 절반만 사용 (학습에도 CPU 필요)
        return max(1, cpu_count // 2)
    
    # GPU 모드에서는 CPU 코어의 1/4 ~ 8개 사이로 설정
    # 너무 많으면 오히려 오버헤드 발생
    optimal = min(max(4, cpu_count // 4), 8)
    return optimal

if BATCH_SIZE is None:
    batch_size = adjust_batch_size(len(train_paths), force_cpu=args.cpu)
    print(f"\n배치 사이즈 자동 설정: {batch_size}")
else:
    batch_size = BATCH_SIZE
    print(f"\n배치 사이즈 고정: {batch_size}")

# Dataset 및 DataLoader 생성
train_dataset = ExcavatorPartsDataset(train_paths, train_labels, transform=train_transform, bbox_crop=args.bbox_crop)
test_dataset = ExcavatorPartsDataset(test_paths, test_labels, transform=val_transform, bbox_crop=args.bbox_crop)

# num_workers 자동 조정 (데이터 로딩 병목 해소)
num_workers = get_optimal_num_workers(force_cpu=args.cpu)
pin_memory = torch.cuda.is_available() and not args.cpu
print(f"DataLoader num_workers: {num_workers}")

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

print(f"Train Batch 개수: {len(train_loader)}")
print(f"Test Batch 개수: {len(test_loader)}")

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
    """ResNet18 기반 Transfer Learning 모델 생성"""
    # PyTorch 최신 버전 권장 방식: weights 파라미터 사용
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    
    # 마지막 FC layer를 우리의 분류 작업에 맞게 수정
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    
    return model


# 디바이스 설정
if args.cpu:
    device = torch.device('cpu')
    print("\n[디바이스 설정] CPU로 강제 실행 (--cpu 플래그 사용)")
else:
    # GPU 사용 불가 시 에러와 함께 종료
    if not torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("❌ [오류] CUDA 사용 불가!")
        print("=" * 80)
        print("\nGPU를 사용할 수 없습니다. 가능한 원인:")
        print("  1. PyTorch가 CPU 버전으로 설치됨 (현재: torch+cpu)")
        print("  2. NVIDIA 드라이버가 설치되지 않음")
        print("  3. CUDA 버전 불일치")
        print("\n해결 방법:")
        print("  # CUDA 지원 PyTorch 설치")
        print("  pip uninstall torch torchvision -y")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
        print("\n  또는 CPU로 실행하려면:")
        print("  python 02_parts_classification.py --cpu")
        print("=" * 80)
        finish_logging()
        sys.exit(1)
    
    device = torch.device('cuda')
    print(f"\n[디바이스 설정] GPU 사용: {torch.cuda.get_device_name(0)}")

# 모델 초기화
model = create_resnet_model(num_classes=num_classes, pretrained=True).to(device)

print(f"사전학습된 ResNet18 모델 로드 완료 (ImageNet 가중치 사용)")
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

# 클래스 가중치 계산 (불균형 데이터 보정)
class_counts = [sum(1 for l in train_labels if l == i) for i in range(num_classes)]
total_count = len(train_labels)
class_weights = torch.tensor([total_count / (num_classes * c) for c in class_counts], 
                             dtype=torch.float32).to(device)

print(f"클래스별 분포: {dict(zip(class_names, class_counts))}")
print(f"클래스 가중치: {dict(zip(class_names, [f'{w:.4f}' for w in class_weights.cpu().numpy()]))}")

# Cross Entropy Loss (클래스 가중치 적용)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning Rate Scheduler (옵션)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# Mixed Precision Training 설정 (GPU에서만 사용)
use_amp = torch.cuda.is_available() and not args.cpu
if use_amp:
    scaler = torch.amp.GradScaler('cuda')
    print("\nMixed Precision Training (AMP) 활성화")
else:
    scaler = None
    print("\nMixed Precision Training 비활성화 (CPU 모드)")

print("\n손실 함수: Cross Entropy Loss (클래스 가중치 적용)")
print("옵티마이저: Adam (lr=0.001)")
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


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """한 에포크 학습 (Mixed Precision 지원)"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed Precision Training
        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
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


def evaluate(model, dataloader, criterion, device, class_names, use_amp=False):
    """모델 평가 (클래스별 정확도 포함, Mixed Precision 지원)"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Mixed Precision 평가
            if use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 클래스별 정확도 계산
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = 100.0 * correct / total
    
    # 클래스별 정확도
    class_accuracies = {}
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            class_accuracies[name] = 100.0 * class_correct[i] / class_total[i]
        else:
            class_accuracies[name] = 0.0
    
    return epoch_loss, epoch_accuracy, class_accuracies


print("학습 및 평가 함수 정의 완료")

step6_time = time.time() - step6_start_time
print(f"\n[6단계 완료] 소요 시간: {step6_time:.2f}초")

# ================================================================================
# 7. 모델 학습 실행
# ================================================================================
print("\n" + "=" * 80)
print("7단계: 모델 학습 시작")
print("=" * 80)
step7_start_time = time.time()

best_val_accuracy = 0.0
best_val_loss = float('inf')
patience_counter = 0

print(f"총 에포크: {NUM_EPOCHS}")
print(f"배치 크기: {batch_size}")
print(f"Early Stopping Patience: {EARLY_STOPPING_PATIENCE} 에포크")
print("\n학습 시작...\n")

for epoch in range(NUM_EPOCHS):
    # 학습 (Mixed Precision 적용)
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
    
    # 검증 (Test 데이터 사용, Mixed Precision 적용)
    val_loss, val_acc, class_accs = evaluate(model, test_loader, criterion, device, class_names, use_amp)
    
    # Learning Rate 스케줄러 업데이트
    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    # 최고 모델 저장
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    else:
        patience_counter += 1
    
    # GPU 메모리 정리 (Early Stopping 직전에만 수행하여 오버헤드 최소화)
    # 매 에포크마다 호출하면 메모리 재할당 비용으로 오히려 성능 저하 발생
    
    # 진행 상황 출력 (5 에포크마다 또는 첫 에포크)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Accuracy:   {val_acc:.2f}%")
        print(f"  클래스별 정확도:")
        for name, acc in class_accs.items():
            print(f"    - {name}: {acc:.2f}%")
        print(f"  Best Val Accuracy: {best_val_accuracy:.2f}% | LR: {current_lr:.6f}")
        if patience_counter > 0:
            print(f"  Early Stopping: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        print("-" * 60)
    
    # Early Stopping 체크
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly Stopping: {EARLY_STOPPING_PATIENCE} 에포크 동안 개선이 없어 학습을 중단합니다.")
        break

print("\n학습 완료!")
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

# 최고 모델 로드
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()

# 최종 테스트 평가 (Mixed Precision 적용)
final_loss, final_acc, final_class_accs = evaluate(model, test_loader, criterion, device, class_names, use_amp)

print(f"\n최종 테스트 결과:")
print(f"  Loss: {final_loss:.4f}")
print(f"  Accuracy: {final_acc:.2f}%")
print(f"\n클래스별 정확도:")
for name, acc in final_class_accs.items():
    print(f"  - {name}: {acc:.2f}%")

step8_time = time.time() - step8_start_time
print(f"\n[8단계 완료] 소요 시간: {step8_time:.2f}초")

# 전체 실행 시간 계산
total_time = time.time() - total_start_time

print("\n" + "=" * 80)
print("모든 작업 완료!")
print("=" * 80)
print(f"\n[전체 실행 시간 요약]")
print(f"  디바이스: {device}")
print(f"  1단계 (데이터셋 로드): {step1_time:.2f}초")
print(f"  2단계 (Dataset 클래스 정의): {step2_time:.2f}초")
print(f"  3단계 (Train/Test 분할): {step3_time:.2f}초")
print(f"  4단계 (모델 정의): {step4_time:.2f}초")
print(f"  5단계 (손실 함수 및 옵티마이저): {step5_time:.2f}초")
print(f"  6단계 (학습/평가 함수 정의): {step6_time:.2f}초")
print(f"  7단계 (모델 학습): {step7_time:.2f}초 ({step7_time/60:.2f}분)")
print(f"  8단계 (최종 평가): {step8_time:.2f}초")
print(f"  ─────────────────────────────────────────────")
print(f"  총 실행 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
print("=" * 80)

# 로깅 종료
finish_logging()
