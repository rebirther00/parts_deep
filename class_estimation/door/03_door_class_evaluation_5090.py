"""
굴착기 도어 분류 모델 평가 스크립트 (RTX 5090 버전)
- 02_door_classification_5090.py에서 학습한 모델 평가
- Test 셋에 대한 성능 평가 및 시각화
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import os
import sys
import json
import argparse
import glob
import time
from sklearn.model_selection import train_test_split

# ================================================================================
# 명령줄 인자 파싱
# ================================================================================
parser = argparse.ArgumentParser(description='굴착기 도어 분류 모델 평가 (RTX 5090)')
parser.add_argument('-cpu', '--cpu', action='store_true',
                    help='CPU로 강제 실행')
parser.add_argument('--num_samples', type=int, default=None,
                    help='평가할 샘플 수 (기본값: 전체 테스트셋)')
parser.add_argument('--dataset_dir', type=str, default=None,
                    help='평가에 사용할 데이터셋 경로. 미지정 시 학습 시 저장된 test_paths 사용')
parser.add_argument('--test_size', type=float, default=0.2,
                    help='dataset_dir 스캔 시 test 비율 (기본 0.2)')
parser.add_argument('--seed', type=int, default=42,
                    help='dataset_dir 스캔 시 랜덤 시드 (기본 42)')
args = parser.parse_args()

# ================================================================================
# 로깅 설정
# ================================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(PROJECT_DIR))
sys.path.insert(0, REPO_DIR)
from utils.logger import setup_logging, finish_logging

LOG_PATH = setup_logging("03_door_evaluation_5090")

# ================================================================================
# 설정 변수
# ================================================================================
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_door_model_5090.pth")
TRAIN_INDICES_PATH = os.path.join(ARTIFACTS_DIR, "training_indices_door_5090.json")
CLASS_NAMES_PATH = os.path.join(ARTIFACTS_DIR, "class_names_door_5090.json")
IMAGE_SIZE = 224
BATCH_SIZE = 32
OUTPUT_IMAGE_PATH = os.path.join(ARTIFACTS_DIR, "evaluation_results_door_5090.png")
OUTPUT_WRONG_IMAGE_PATH = os.path.join(ARTIFACTS_DIR, "evaluation_wrong_predictions_door_5090.png")
RESULTS_JSON_PATH = os.path.join(ARTIFACTS_DIR, "evaluation_results_door_5090.json")

total_start_time = time.time()

# ================================================================================
# 1. 데이터 및 모델 정보 로드
# ================================================================================
print("=" * 80)
print("굴착기 도어 분류 모델 평가 (RTX 5090 버전)")
print("=" * 80)
step1_start_time = time.time()


def scan_dataset_for_eval(dataset_dir, class_names):
    """dataset_dir에서 rgb_*.png를 스캔하여 평가 경로 리스트 생성"""
    all_paths = []
    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        pngs = sorted(glob.glob(os.path.join(class_dir, "rgb_*.png")))
        all_paths.extend(pngs)
    return all_paths


def split_paths_train_test(image_paths, class_names, test_size=0.2, seed=42):
    """이미지 경로 목록을 클래스 비율 유지(stratify)로 train/test split"""
    labels = []
    for p in image_paths:
        cls = os.path.basename(os.path.dirname(p))
        labels.append(class_names.index(cls))

    indices = list(range(len(image_paths)))
    _, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels
    )
    test_paths = [image_paths[i] for i in test_idx]
    return test_paths


if not os.path.exists(TRAIN_INDICES_PATH):
    raise FileNotFoundError(
        f"학습 데이터 정보 파일을 찾을 수 없습니다: {TRAIN_INDICES_PATH}\n"
        f"먼저 02_door_classification_5090.py를 실행하세요."
    )

with open(TRAIN_INDICES_PATH, 'r', encoding='utf-8') as f:
    train_data_info = json.load(f)

class_names = train_data_info['class_names']
num_classes = len(class_names)

if args.dataset_dir:
    saved_test_paths = train_data_info.get("test_paths", [])
    test_paths = [p for p in saved_test_paths if p.startswith(args.dataset_dir)]

    if len(test_paths) == 0:
        all_paths = scan_dataset_for_eval(args.dataset_dir, class_names)
        if len(all_paths) == 0:
            raise FileNotFoundError(f"dataset_dir에서 rgb_*.png를 찾지 못했습니다: {args.dataset_dir}")
        test_paths = split_paths_train_test(all_paths, class_names,
                                            test_size=args.test_size, seed=args.seed)
else:
    test_paths = train_data_info['test_paths']

print(f"\n모델 파일: {MODEL_PATH}")
print(f"테스트 데이터: {len(test_paths)}장")
print(f"클래스 수: {num_classes}개")
print(f"클래스 이름: {class_names}")

step1_time = time.time() - step1_start_time
print(f"\n[1단계 완료] 소요 시간: {step1_time:.2f}초")

if args.num_samples and args.num_samples < len(test_paths):
    random.seed(42)
    test_paths = random.sample(test_paths, args.num_samples)
    print(f"평가 샘플 수: {len(test_paths)}장 (랜덤 샘플링)")

# ================================================================================
# 2. Dataset 클래스 정의 및 DataLoader 생성
# ================================================================================
print("\n" + "=" * 80)
print("2단계: Dataset 클래스 정의 및 DataLoader 생성")
print("=" * 80)
step2_start_time = time.time()


class DoorDataset(Dataset):
    """굴착기 도어 이미지 Dataset 클래스 (평가용)"""

    def __init__(self, image_paths, class_names, transform=None):
        self.image_paths = image_paths
        self.class_names = class_names
        self.transform = transform

        self.labels = []
        for path in image_paths:
            class_folder = os.path.basename(os.path.dirname(path))
            if class_folder in class_names:
                self.labels.append(class_names.index(class_folder))
            else:
                raise ValueError(f"알 수 없는 클래스: {class_folder}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        if image.mode != 'RGB':
            image = image.convert('RGB')

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


# 학습과 동일한 종횡비 유지 리사이즈
eval_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

test_dataset = DoorDataset(test_paths, class_names, transform=eval_transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"\n테스트 배치 개수: {len(test_loader)}")

step2_time = time.time() - step2_start_time
print(f"\n[2단계 완료] 소요 시간: {step2_time:.2f}초")

# ================================================================================
# 3. 모델 정의 및 로드
# ================================================================================
print("\n" + "=" * 80)
print("3단계: 모델 정의 및 로드")
print("=" * 80)
step3_start_time = time.time()


def create_resnet_model(num_classes, pretrained=False):
    """ResNet18 기반 모델 생성"""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model


if args.cpu:
    device = torch.device('cpu')
    print("디바이스: CPU (강제)")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}\n"
                           f"먼저 02_door_classification_5090.py를 실행하세요.")

model = create_resnet_model(num_classes=num_classes, pretrained=False).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("모델 로드 완료!")

step3_time = time.time() - step3_start_time
print(f"\n[3단계 완료] 소요 시간: {step3_time:.2f}초")

# ================================================================================
# 4. 평가 실행
# ================================================================================
print("\n" + "=" * 80)
print("4단계: 평가 실행")
print("=" * 80)
step4_start_time = time.time()

correct = 0
total = 0
class_correct = [0] * num_classes
class_total = [0] * num_classes

all_predictions = []
all_labels = []
all_confidences = []
all_image_paths = []

print("\n예측 결과:")
print("-" * 80)

with torch.no_grad():
    sample_idx = 0
    for batch_idx, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

        for i in range(len(labels)):
            label = labels[i].item()
            pred = predictions[i].item()
            confidence = torch.softmax(outputs[i], dim=0)[pred].item() * 100

            is_correct = (label == pred)
            symbol = "O" if is_correct else "X"

            if is_correct:
                correct += 1
                class_correct[label] += 1

            total += 1
            class_total[label] += 1

            all_predictions.append(pred)
            all_labels.append(label)
            all_confidences.append(confidence)
            all_image_paths.append(test_paths[sample_idx])

            if sample_idx < 50:
                print(f"{symbol} [{class_names[label]:20s}] -> [{class_names[pred]:20s}] | "
                      f"신뢰도: {confidence:.1f}%")

            sample_idx += 1

if sample_idx > 50:
    print(f"... ({sample_idx - 50}개 결과 생략)")

step4_time = time.time() - step4_start_time
print(f"\n[4단계 완료] 소요 시간: {step4_time:.2f}초")

# ================================================================================
# 5. 결과 시각화
# ================================================================================
print("\n" + "=" * 80)
print("5단계: 결과 시각화")
print("=" * 80)
step5_start_time = time.time()


def load_font():
    """시각화용 폰트 로드"""
    font_paths = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, 12)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def create_result_grid(image_paths, labels, predictions, confidences, class_names,
                       num_cols=5, img_size=200, max_images=50):
    """예측 결과를 그리드 형태로 시각화"""
    num_images = min(len(image_paths), max_images)
    num_rows = (num_images + num_cols - 1) // num_cols

    text_height = 100
    cell_width = img_size
    cell_height = img_size + text_height

    grid_width = num_cols * cell_width
    grid_height = num_rows * cell_height
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')

    font = load_font()

    for idx in range(num_images):
        row = idx // num_cols
        col = idx % num_cols

        x = col * cell_width
        y = row * cell_height

        img = Image.open(image_paths[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        grid_image.paste(img, (x, y))

        draw = ImageDraw.Draw(grid_image)
        text_y = y + img_size + 5

        actual = class_names[labels[idx]]
        predicted = class_names[predictions[idx]]
        conf = confidences[idx]
        is_correct = labels[idx] == predictions[idx]

        symbol = "O" if is_correct else "X"
        text_color = (0, 150, 0) if is_correct else (200, 0, 0)
        bg_color = (230, 255, 230) if is_correct else (255, 230, 230)
        draw.rectangle([x, y + img_size, x + cell_width, y + cell_height], fill=bg_color)

        draw.text((x + 5, text_y), f"{symbol} Actual: {actual}", fill=text_color, font=font)
        draw.text((x + 5, text_y + 20), f"Pred: {predicted}", fill=(0, 0, 0), font=font)
        draw.text((x + 5, text_y + 40), f"Conf: {conf:.1f}%", fill=(100, 100, 100), font=font)
        draw.text((x + 5, text_y + 60), f"src: {os.path.basename(image_paths[idx])}", fill=(60, 60, 60), font=font)

    grid_image.save(OUTPUT_IMAGE_PATH, 'PNG', quality=95)
    print(f"\n결과 이미지 저장: {OUTPUT_IMAGE_PATH}")

    return grid_image


def create_wrong_predictions_grid(image_paths, labels, predictions, confidences, class_names,
                                   num_cols=5, img_size=200):
    """틀린 예측 결과만 그리드 형태로 시각화"""
    wrong_indices = [i for i in range(len(labels)) if labels[i] != predictions[i]]

    if len(wrong_indices) == 0:
        print("\n모든 예측이 정확합니다! 틀린 예측 이미지가 없습니다.")
        return None

    num_images = len(wrong_indices)
    num_rows = (num_images + num_cols - 1) // num_cols

    text_height = 100
    cell_width = img_size
    cell_height = img_size + text_height

    grid_width = num_cols * cell_width
    grid_height = num_rows * cell_height
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')

    font = load_font()

    for grid_idx, data_idx in enumerate(wrong_indices):
        row = grid_idx // num_cols
        col = grid_idx % num_cols

        x = col * cell_width
        y = row * cell_height

        img = Image.open(image_paths[data_idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        grid_image.paste(img, (x, y))

        draw = ImageDraw.Draw(grid_image)
        text_y = y + img_size + 5

        actual = class_names[labels[data_idx]]
        predicted = class_names[predictions[data_idx]]
        conf = confidences[data_idx]

        bg_color = (255, 220, 220)
        draw.rectangle([x, y + img_size, x + cell_width, y + cell_height], fill=bg_color)

        draw.text((x + 5, text_y), f"X Actual: {actual}", fill=(200, 0, 0), font=font)
        draw.text((x + 5, text_y + 20), f"Pred: {predicted}", fill=(0, 0, 0), font=font)
        draw.text((x + 5, text_y + 40), f"Conf: {conf:.1f}%", fill=(100, 100, 100), font=font)
        draw.text((x + 5, text_y + 60), f"src: {os.path.basename(image_paths[data_idx])}", fill=(60, 60, 60), font=font)

    grid_image.save(OUTPUT_WRONG_IMAGE_PATH, 'PNG', quality=95)
    print(f"\n틀린 예측 이미지 저장: {OUTPUT_WRONG_IMAGE_PATH} ({num_images}개)")

    return grid_image


print("\n결과 이미지 생성 중...")
create_result_grid(all_image_paths, all_labels, all_predictions, all_confidences, class_names)

print("\n틀린 예측 이미지 생성 중...")
create_wrong_predictions_grid(all_image_paths, all_labels, all_predictions, all_confidences, class_names)

step5_time = time.time() - step5_start_time
print(f"\n[5단계 완료] 소요 시간: {step5_time:.2f}초")

# ================================================================================
# 6. 최종 결과 요약
# ================================================================================
print("\n" + "=" * 80)
print("6단계: 평가 결과 요약")
print("=" * 80)
step6_start_time = time.time()

overall_accuracy = 100.0 * correct / total

print(f"\n전체 정확도: {overall_accuracy:.2f}% ({correct}/{total})")
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

# 오류 분석
errors = total - correct
print(f"\n오류 개수: {errors}개")
if errors > 0:
    print("\n오류 분석 (상위 5개):")
    error_pairs = []
    for i in range(len(all_labels)):
        if all_labels[i] != all_predictions[i]:
            error_pairs.append((
                class_names[all_labels[i]],
                class_names[all_predictions[i]],
                all_confidences[i]
            ))

    error_pairs.sort(key=lambda x: -x[2])
    for actual, predicted, conf in error_pairs[:5]:
        print(f"  - {actual} -> {predicted} (신뢰도: {conf:.1f}%)")

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

step6_time = time.time() - step6_start_time
print(f"\n[6단계 완료] 소요 시간: {step6_time:.2f}초")

# 전체 실행 시간 요약
total_time = time.time() - total_start_time

print("\n" + "=" * 80)
print("평가 완료!")
print("=" * 80)
print(f"\n[전체 실행 시간 요약]")
print(f"  1단계 (데이터 및 모델 정보 로드): {step1_time:.2f}초")
print(f"  2단계 (Dataset/DataLoader 생성): {step2_time:.2f}초")
print(f"  3단계 (모델 정의 및 로드): {step3_time:.2f}초")
print(f"  4단계 (평가 실행): {step4_time:.2f}초")
print(f"  5단계 (결과 시각화): {step5_time:.2f}초")
print(f"  6단계 (결과 요약): {step6_time:.2f}초")
print(f"  ─────────────────────────────────────────────")
print(f"  총 실행 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
print("=" * 80)

# 결과 JSON 저장
results = {
    "accuracy": overall_accuracy,
    "total_samples": total,
    "correct_samples": correct,
    "class_accuracies": {name: 100.0 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
                        for i, name in enumerate(class_names)},
    "avg_precision": avg_precision,
    "avg_recall": avg_recall,
    "avg_f1": avg_f1
}

with open(RESULTS_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n결과 저장: {RESULTS_JSON_PATH}")

finish_logging()
