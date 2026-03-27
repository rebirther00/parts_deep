"""PyTorch 기반 도어 분류 모델 평가 (추론 파이프라인 검증)

04_door_realtime_inference.py와 동일한 전처리 + 모델로
테스트 데이터셋을 평가하여, 실시간 추론 전 모델 정확도를 검증한다.

실행:
    python 03_5_trt_evaluation.py
    python 03_5_trt_evaluation.py --dataset_dir /path/to/datasets
"""

import argparse
import glob
import json
import os
import random
import sys
import time

import numpy as np
import torch
from PIL import Image
import cv2

from depth_utils import (
    RGBDAuxResNet18, RGBDTransform, IN_CHANNELS, MAX_DEPTH_MM,
    compute_aux_features, ISAAC_SIM_INTRINSICS,
)

# ================================================================================
# 명령줄 인자
# ================================================================================
parser = argparse.ArgumentParser(description="도어 분류 모델 추론 파이프라인 평가")
parser.add_argument("--dataset_dir", type=str, default=None,
                    help="평가할 데이터셋 경로 (미지정 시 학습 시 저장된 test_paths 사용)")
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ================================================================================
# 로깅 설정
# ================================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(PROJECT_DIR))
sys.path.insert(0, REPO_DIR)
from utils.logger import setup_logging, finish_logging

LOG_PATH = setup_logging("03_5_inference_evaluation")

# ================================================================================
# 경로 설정
# ================================================================================
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_door_model_5090.pth")
CLASS_NAMES_PATH = os.path.join(ARTIFACTS_DIR, "class_names_door_5090.json")
TRAIN_INDICES_PATH = os.path.join(ARTIFACTS_DIR, "training_indices_door_5090.json")
DATASETS_DIR = os.path.join(PROJECT_DIR, "datasets")
RESULTS_JSON_PATH = os.path.join(ARTIFACTS_DIR, "inference_evaluation_results.json")
IMAGE_SIZE = 448

total_start_time = time.time()

# ================================================================================
# 1. 테스트 데이터 로드
# ================================================================================
print("=" * 80)
print("도어 분류 모델 추론 파이프라인 평가")
print("=" * 80)

if not os.path.exists(TRAIN_INDICES_PATH):
    raise FileNotFoundError(
        f"학습 데이터 정보 파일을 찾을 수 없습니다: {TRAIN_INDICES_PATH}\n"
        f"먼저 02_door_classification_5090.py를 실행하세요."
    )

with open(TRAIN_INDICES_PATH, "r", encoding="utf-8") as f:
    train_data_info = json.load(f)

class_names = train_data_info["class_names"]
num_classes = len(class_names)


def remap_path(p):
    """학습 시 저장된 절대 경로를 현재 머신 경로로 변환"""
    basename = os.path.basename(p)
    cls_name = os.path.basename(os.path.dirname(p))
    return os.path.join(DATASETS_DIR, cls_name, basename)


if args.dataset_dir:
    saved_test_paths = train_data_info.get("test_paths", [])
    test_paths = [p for p in saved_test_paths if p.startswith(args.dataset_dir)]
    if not test_paths:
        all_paths = []
        for cls in class_names:
            cls_dir = os.path.join(args.dataset_dir, cls)
            if os.path.isdir(cls_dir):
                all_paths.extend(sorted(glob.glob(os.path.join(cls_dir, "rgb_*.png"))))
        if not all_paths:
            raise FileNotFoundError(f"데이터셋을 찾지 못했습니다: {args.dataset_dir}")
        random.seed(args.seed)
        random.shuffle(all_paths)
        split_idx = int(len(all_paths) * (1 - args.test_size))
        test_paths = all_paths[split_idx:]
else:
    test_paths = [remap_path(p) for p in train_data_info["test_paths"]]

print(f"\n테스트 데이터: {len(test_paths)}장")
print(f"클래스: {class_names}")


# ================================================================================
# 2. 모델 로드
# ================================================================================
print("\n" + "=" * 80)
print("2단계: PyTorch 모델 로드")
print("=" * 80)

device = torch.device("cpu")

model = RGBDAuxResNet18(num_classes, pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

val_transform = RGBDTransform(IMAGE_SIZE, is_train=False)

print(f"디바이스: {device}")
print("RGBD 모델 로드 완료")


# ================================================================================
# 3. 테스트셋 평가
# ================================================================================
print("\n" + "=" * 80)
print("3단계: 테스트셋 평가")
print("=" * 80)

correct = 0
total = 0
class_correct = [0] * num_classes
class_total = [0] * num_classes
wrong_samples = []
inference_times = []

with torch.no_grad():
    for img_path in test_paths:
        if not os.path.exists(img_path):
            continue

        gt_class = os.path.basename(os.path.dirname(img_path))
        if gt_class not in class_names:
            continue
        gt_idx = class_names.index(gt_class)

        pil_img = Image.open(img_path).convert("RGB")
        depth_path = img_path.replace("rgb_", "depth_")
        depth_raw_mm = None
        if os.path.exists(depth_path):
            raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if raw is not None:
                depth_raw_mm = raw.astype(np.float32)
                depth_norm = np.clip(depth_raw_mm / MAX_DEPTH_MM, 0.0, 1.0)
            else:
                depth_norm = np.zeros(
                    (pil_img.height, pil_img.width), dtype=np.float32)
        else:
            depth_norm = np.zeros(
                (pil_img.height, pil_img.width), dtype=np.float32)

        inp = val_transform(pil_img, depth_norm).unsqueeze(0).to(device)
        aux = compute_aux_features(
            depth_raw_mm if depth_raw_mm is not None
            else np.zeros((pil_img.height, pil_img.width), dtype=np.float32),
            ISAAC_SIM_INTRINSICS,
        )
        aux_t = torch.tensor([aux], dtype=torch.float32).to(device)

        t0 = time.time()
        output = model(inp, aux_t)
        elapsed_ms = (time.time() - t0) * 1000
        inference_times.append(elapsed_ms)

        probs = torch.softmax(output, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))

        total += 1
        class_total[gt_idx] += 1
        if pred_idx == gt_idx:
            correct += 1
            class_correct[gt_idx] += 1
        else:
            wrong_samples.append({
                "path": img_path,
                "gt": class_names[gt_idx],
                "pred": class_names[pred_idx],
                "confidence": float(probs[pred_idx]),
            })

# ================================================================================
# 4. 결과 출력
# ================================================================================
print("\n" + "=" * 80)
print("평가 결과")
print("=" * 80)

accuracy = 100.0 * correct / total if total > 0 else 0
avg_ms = np.mean(inference_times) if inference_times else 0

print(f"\n전체 정확도: {accuracy:.2f}% ({correct}/{total})")
print(f"평균 추론 시간: {avg_ms:.2f} ms/장")

print("\n클래스별 정확도:")
class_accuracies = {}
for i, name in enumerate(class_names):
    if class_total[i] > 0:
        acc = 100.0 * class_correct[i] / class_total[i]
        class_accuracies[name] = acc
        print(f"  {name}: {acc:.1f}% ({class_correct[i]}/{class_total[i]})")
    else:
        class_accuracies[name] = 0.0
        print(f"  {name}: 데이터 없음")

if wrong_samples:
    print(f"\n오분류 샘플 ({len(wrong_samples)}건):")
    for w in wrong_samples:
        print(f"  {os.path.basename(w['path'])}: "
              f"정답={w['gt']}, 예측={w['pred']} ({w['confidence']*100:.1f}%)")

results = {
    "engine": f"PyTorch {torch.__version__} ({device})",
    "accuracy": accuracy,
    "total_samples": total,
    "correct_samples": correct,
    "class_accuracies": class_accuracies,
    "avg_inference_ms": round(avg_ms, 2),
    "wrong_count": len(wrong_samples),
    "wrong_samples": wrong_samples,
}
with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\n결과 저장: {RESULTS_JSON_PATH}")

total_time = time.time() - total_start_time
print(f"\n총 소요 시간: {total_time:.2f}초")

finish_logging()
