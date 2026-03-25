"""
CAD 사전학습 RGBD 모델 → 실물 데이터 Fine-tuning 검증 스크립트
- CAD 합성 데이터로 학습한 RGBD 모델 가중치를 초기값으로 사용
- 실물 이미지(RGB+Depth)로 도메인 적응
- 빠른 검증 목적: 합성→실물 전이학습이 유효한지 확인
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

from depth_utils import (
    create_rgbd_resnet18, RGBDTransform, RGBDDataset, IN_CHANNELS,
)

# ================================================================================
# 명령줄 인자
# ================================================================================
parser = argparse.ArgumentParser(description='CAD 사전학습 → 실물 데이터 Fine-tuning')
parser.add_argument('-cpu', '--cpu', action='store_true', help='CPU 강제 실행')
args = parser.parse_args()

# ================================================================================
# 경로 설정
# ================================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(PROJECT_DIR))
sys.path.insert(0, REPO_DIR)
from utils.logger import setup_logging, finish_logging

LOG_PATH = setup_logging("04_door_cad_finetune_real_5090")

ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
REAL_DATASET_DIR = os.path.join(PROJECT_DIR, "datasets")
CAD_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_door_cad_model_5090.pth")
CAD_CLASS_NAMES_PATH = os.path.join(ARTIFACTS_DIR, "class_names_door_cad_5090.json")

FINETUNE_MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_door_finetune_model_5090.pth")
FINETUNE_ONNX_PATH = os.path.join(ARTIFACTS_DIR, "best_door_finetune_model_5090.onnx")
FINETUNE_RESULTS_PATH = os.path.join(ARTIFACTS_DIR, "evaluation_results_finetune_5090.json")

# Fine-tuning 하이퍼파라미터 (사전학습 모델이므로 보수적 설정)
IMAGE_SIZE = 224
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 8
TEST_SIZE = 0.2
RANDOM_SEED = 42
NUM_WORKERS = 8

total_start_time = time.time()

# ================================================================================
# 1. CAD 모델 클래스 정보 및 실물 데이터 스캔
# ================================================================================
print("=" * 80)
print("CAD 사전학습 → 실물 데이터 Fine-tuning 검증")
print("=" * 80)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

with open(CAD_CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
    class_names = json.load(f)
num_classes = len(class_names)

print(f"\nCAD 모델 클래스 ({num_classes}개): {class_names}")
print(f"실물 데이터셋: {REAL_DATASET_DIR}")

image_paths = []
labels = []
available_classes = []

for idx, class_name in enumerate(class_names):
    class_dir = os.path.join(REAL_DATASET_DIR, class_name)
    if not os.path.isdir(class_dir):
        print(f"  [스킵] {class_name}: 폴더 없음")
        continue
    pngs = sorted(glob.glob(os.path.join(class_dir, "rgb_*.png")))
    if len(pngs) == 0:
        print(f"  [스킵] {class_name}: 이미지 없음")
        continue
    print(f"  [로드] {class_name} (idx={idx}): {len(pngs)}장")
    available_classes.append(class_name)
    for p in pngs:
        image_paths.append(p)
        labels.append(idx)

print(f"\n사용 가능한 클래스: {len(available_classes)}개 / {num_classes}개")
print(f"총 실물 이미지: {len(image_paths)}장")

# ================================================================================
# 2. Train/Test 분할 및 DataLoader
# ================================================================================
print("\n" + "=" * 80)
print("2단계: 데이터 분할 및 DataLoader")
print("=" * 80)

indices = list(range(len(image_paths)))
train_indices, test_indices = train_test_split(
    indices, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=labels
)

train_paths = [image_paths[i] for i in train_indices]
train_labels = [labels[i] for i in train_indices]
test_paths = [image_paths[i] for i in test_indices]
test_labels = [labels[i] for i in test_indices]

print(f"Train: {len(train_paths)}장, Test: {len(test_paths)}장")
for cls in available_classes:
    idx = class_names.index(cls)
    tr = sum(1 for l in train_labels if l == idx)
    te = sum(1 for l in test_labels if l == idx)
    print(f"  {cls}: Train {tr}장, Test {te}장")


train_transform = RGBDTransform(IMAGE_SIZE, is_train=True)
val_transform = RGBDTransform(IMAGE_SIZE, is_train=False)

train_dataset = RGBDDataset(train_paths, train_labels, transform=train_transform)
test_dataset = RGBDDataset(test_paths, test_labels, transform=val_transform)

pin_memory = torch.cuda.is_available() and not args.cpu
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=pin_memory)

# ================================================================================
# 3. CAD 사전학습 모델 로드 → Fine-tuning 설정
# ================================================================================
print("\n" + "=" * 80)
print("3단계: CAD 사전학습 모델 로드")
print("=" * 80)

def create_resnet_model(num_classes):
    """RGBD 4채널 입력 ResNet18 모델 (CAD 학습과 동일한 구조)"""
    return create_rgbd_resnet18(num_classes, pretrained=False)


if args.cpu:
    device = torch.device('cpu')
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = create_resnet_model(num_classes).to(device)
model.load_state_dict(torch.load(CAD_MODEL_PATH, map_location=device))
print(f"CAD 사전학습 가중치 로드 완료: {CAD_MODEL_PATH}")
print(f"디바이스: {device}")
print(f"모델 파라미터: {sum(p.numel() for p in model.parameters()):,}")

# Feature extractor는 낮은 lr, classifier는 높은 lr (차별적 학습률)
feature_params = []
classifier_params = []
for name, param in model.named_parameters():
    if 'fc' in name:
        classifier_params.append(param)
    else:
        feature_params.append(param)

optimizer = optim.Adam([
    {'params': feature_params, 'lr': LEARNING_RATE * 0.1},
    {'params': classifier_params, 'lr': LEARNING_RATE},
], lr=LEARNING_RATE)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
criterion = nn.CrossEntropyLoss()

print(f"\n[Fine-tuning 설정]")
print(f"  Feature Extractor LR: {LEARNING_RATE * 0.1}")
print(f"  Classifier LR: {LEARNING_RATE}")
print(f"  에포크: {NUM_EPOCHS}, Early Stopping: {EARLY_STOPPING_PATIENCE}")

# ================================================================================
# 4. Fine-tuning 학습
# ================================================================================
print("\n" + "=" * 80)
print("4단계: Fine-tuning 시작")
print("=" * 80)
step4_start = time.time()

best_val_acc = 0.0
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    # Train
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for images, labels_batch in train_loader:
        images = images.to(device, non_blocking=True)
        labels_batch = labels_batch.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        train_total += labels_batch.size(0)
        train_correct += (preds == labels_batch).sum().item()

    train_acc = 100.0 * train_correct / train_total

    # Evaluate
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    all_preds_list = []
    all_labels_list = []

    with torch.no_grad():
        for images, labels_batch in test_loader:
            images = images.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            val_total += labels_batch.size(0)
            val_correct += (preds == labels_batch).sum().item()
            all_preds_list.extend(preds.cpu().numpy())
            all_labels_list.extend(labels_batch.cpu().numpy())
            for i in range(labels_batch.size(0)):
                lbl = labels_batch[i].item()
                class_total[lbl] += 1
                if preds[i] == labels_batch[i]:
                    class_correct[lbl] += 1

    val_acc = 100.0 * val_correct / val_total
    scheduler.step(val_loss / val_total)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), FINETUNE_MODEL_PATH)
    else:
        patience_counter += 1

    if (epoch + 1) % 5 == 0 or epoch == 0 or patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Train Loss: {train_loss/train_total:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/val_total:.4f} | Acc: {val_acc:.2f}%")
        for cls in available_classes:
            ci = class_names.index(cls)
            if class_total[ci] > 0:
                print(f"    {cls}: {100.0*class_correct[ci]/class_total[ci]:.1f}%")
        print(f"  Best: {best_val_acc:.2f}% | Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly Stopping: {EARLY_STOPPING_PATIENCE} 에포크 동안 개선 없음")
        break

step4_time = time.time() - step4_start
print(f"\n학습 완료! Best Val Accuracy: {best_val_acc:.2f}%")
print(f"소요 시간: {step4_time:.1f}초 ({step4_time/60:.1f}분)")

# ================================================================================
# 5. 최종 평가 + 혼동 행렬
# ================================================================================
print("\n" + "=" * 80)
print("5단계: 최종 평가")
print("=" * 80)

model.load_state_dict(torch.load(FINETUNE_MODEL_PATH, map_location=device))
model.eval()

final_correct, final_total = 0, 0
class_correct = [0] * num_classes
class_total = [0] * num_classes
all_preds_list = []
all_labels_list = []

with torch.no_grad():
    for images, labels_batch in test_loader:
        images = images.to(device, non_blocking=True)
        labels_batch = labels_batch.to(device, non_blocking=True)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        final_total += labels_batch.size(0)
        final_correct += (preds == labels_batch).sum().item()
        all_preds_list.extend(preds.cpu().numpy())
        all_labels_list.extend(labels_batch.cpu().numpy())
        for i in range(labels_batch.size(0)):
            lbl = labels_batch[i].item()
            class_total[lbl] += 1
            if preds[i] == labels_batch[i]:
                class_correct[lbl] += 1

final_acc = 100.0 * final_correct / final_total
print(f"\n최종 정확도: {final_acc:.2f}% ({final_correct}/{final_total})")

print(f"\n클래스별 정확도:")
for cls in available_classes:
    ci = class_names.index(cls)
    if class_total[ci] > 0:
        acc = 100.0 * class_correct[ci] / class_total[ci]
        print(f"  {cls}: {acc:.2f}% ({class_correct[ci]}/{class_total[ci]})")

# 혼동 행렬 (사용 가능한 클래스만)
avail_indices = [class_names.index(c) for c in available_classes]
n = len(available_classes)
conf = [[0] * n for _ in range(n)]
for true_l, pred_l in zip(all_labels_list, all_preds_list):
    if true_l in avail_indices and pred_l in avail_indices:
        ri = avail_indices.index(true_l)
        ci = avail_indices.index(pred_l)
        conf[ri][ci] += 1

short = [c.replace("door_", "") for c in available_classes]
max_len = max(len(s) for s in short)
col_w = max(max_len, 5)

print(f"\n혼동 행렬 (행: 실제, 열: 예측):")
print(" " * (max_len + 2) + " ".join(f"{s:>{col_w}}" for s in short))
print(" " * (max_len + 2) + "-" * (n * (col_w + 1)))
for i, row in enumerate(conf):
    print(f"{short[i]:>{max_len}} | " + " ".join(f"{v:>{col_w}}" for v in row))

# 8클래스 외 예측 (다른 클래스로 오분류된 경우)
other_preds = sum(1 for p in all_preds_list if p not in avail_indices)
if other_preds > 0:
    print(f"\n⚠ 데이터 없는 클래스로 오분류: {other_preds}건")
    for ci in range(num_classes):
        if ci not in avail_indices:
            cnt = sum(1 for p in all_preds_list if p == ci)
            if cnt > 0:
                print(f"  → {class_names[ci]}: {cnt}건")

# ================================================================================
# 6. 비교 요약 (CAD only vs Fine-tuning)
# ================================================================================
print("\n" + "=" * 80)
print("6단계: CAD 모델 vs Fine-tuning 비교")
print("=" * 80)

cad_results_path = os.path.join(ARTIFACTS_DIR, "evaluation_results_door_cad_5090.json")
if os.path.exists(cad_results_path):
    with open(cad_results_path, 'r') as f:
        cad_results = json.load(f)
    print(f"\n{'':>20s} {'CAD only':>12s} {'Fine-tuned':>12s}")
    print("-" * 46)
    for cls in available_classes:
        cad_acc = cad_results.get("class_accuracies", {}).get(cls, 0)
        ft_acc = 100.0 * class_correct[class_names.index(cls)] / max(class_total[class_names.index(cls)], 1)
        print(f"{cls:>20s} {cad_acc:>11.1f}% {ft_acc:>11.1f}%")
    cad_overall = cad_results.get("accuracy", 0)
    print("-" * 46)
    print(f"{'전체':>20s} {cad_overall:>11.1f}% {final_acc:>11.1f}%")
else:
    print("\n(CAD 평가 결과 없음 - 03 스크립트 실행 필요)")

# 결과 저장
results = {
    "accuracy": final_acc,
    "total_samples": final_total,
    "available_classes": available_classes,
    "class_accuracies": {
        cls: 100.0 * class_correct[class_names.index(cls)] / max(class_total[class_names.index(cls)], 1)
        for cls in available_classes
    },
}
with open(FINETUNE_RESULTS_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# ================================================================================
# 7. ONNX 변환
# ================================================================================
print("\n" + "=" * 80)
print("7단계: ONNX 변환")
print("=" * 80)

try:
    model_cpu = create_resnet_model(num_classes)
    model_cpu.load_state_dict(torch.load(FINETUNE_MODEL_PATH, map_location='cpu'))
    model_cpu.eval()
    dummy = torch.randn(1, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(model_cpu, dummy, FINETUNE_ONNX_PATH,
                      export_params=True, opset_version=13,
                      do_constant_folding=True,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
                      dynamo=False)
    onnx_data = FINETUNE_ONNX_PATH + ".data"
    if os.path.exists(onnx_data):
        os.remove(onnx_data)
    print(f"ONNX 저장: {FINETUNE_ONNX_PATH}")
    print(f"크기: {os.path.getsize(FINETUNE_ONNX_PATH) / (1024*1024):.2f} MB")
except Exception as e:
    print(f"ONNX 변환 실패: {e}")

total_time = time.time() - total_start_time
print(f"\n총 실행 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")

finish_logging()
