"""2단계 굴착기 도어 분류 - CAD 합성 데이터 (RTX 5090)

Step 1: 모양 분류 (LH_FRT / LH_RR / RH) - 저해상도(448)
Step 2: 기종 분류 (E25 / E30 / E38) - 고해상도(960+)로 종횡비 차이 학습

실행:
    python 02_door_cad_classification_5090_2step.py
    python 02_door_cad_classification_5090_2step.py --step2_size 1920
    python 02_door_cad_classification_5090_2step.py --full_train
"""
import argparse
import glob as glob_mod
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from depth_utils import (
    RGBDAuxResNet18, RGBDTransform, RGBDDataset, IN_CHANNELS,
    NUM_AUX_FEATURES, MAX_DEPTH_MM, compute_aux_features,
    ISAAC_SIM_INTRINSICS,
)

# ── 모양 그룹 정의 ─────────────────────────────────────────
SHAPE_GROUPS = {
    "LH_FRT": ["E25_door_LH_FRT", "E30_door_LH_FRT", "E38_door_LH_FRT"],
    "LH_RR":  ["E25_door_LH_RR",  "E30_door_LH_RR",  "E38_door_LH_RR"],
    "RH":     ["E25_door_RH",      "E30_E38_door_RH"],
}

# ── 명령줄 인자 ────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DS = os.path.join(PROJECT_DIR, "datasets_cad")

parser = argparse.ArgumentParser(
    description="2단계 도어 분류 학습 (CAD, RTX 5090)")
parser.add_argument("-cpu", "--cpu", action="store_true",
                    help="CPU로 강제 실행")
parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DS,
                    help="데이터셋 경로 (기본: datasets_cad)")
parser.add_argument("--full_train", action="store_true",
                    help="전체 데이터를 학습에 사용 (배포용)")
parser.add_argument("--step1_size", type=int, default=448,
                    help="Step 1 입력 크기 (기본: 448)")
parser.add_argument("--step2_size", type=int, default=960,
                    help="Step 2 입력 크기 (기본: 960, 풀해상도: 1920)")
args = parser.parse_args()

# ── 로깅 / 경로 ────────────────────────────────────────────
REPO_DIR = os.path.dirname(os.path.dirname(PROJECT_DIR))
sys.path.insert(0, REPO_DIR)
from utils.logger import setup_logging, finish_logging

LOG_PATH = setup_logging("02_door_cad_classification_5090_2step")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── 설정 ───────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.2
NUM_EPOCHS = 60
PATIENCE = 10
NUM_WORKERS = 8
PREFETCH = 2

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

total_start = time.time()
print("=" * 70)
print("2단계 도어 분류 학습 (CAD 데이터)")
print("=" * 70)
print(f"디바이스: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Step 1 해상도: {args.step1_size}, Step 2 해상도: {args.step2_size}")
print(f"모드: {'전체 학습' if args.full_train else '분할 학습 (80/20)'}")


# ═══════════════════════════════════════════════════════════
# 공통 유틸리티
# ═══════════════════════════════════════════════════════════

def scan_dataset(dataset_dir):
    """데이터셋 폴더 스캔 → (클래스명 리스트, 이미지 경로, 레이블)"""
    classes, paths, labels = [], [], []
    class_idx = 0
    for name in sorted(os.listdir(dataset_dir)):
        d = os.path.join(dataset_dir, name)
        if not os.path.isdir(d):
            continue
        files = sorted(glob_mod.glob(os.path.join(d, "rgb_*.png")))
        if not files:
            continue
        print(f"  [{class_idx}] {name}: {len(files)}장")
        classes.append(name)
        paths.extend(files)
        labels.extend([class_idx] * len(files))
        class_idx += 1
    return classes, paths, labels


def to_shape_labels(class_names, labels):
    """원본 8클래스 레이블 → 3클래스 모양 레이블"""
    shape_names = list(SHAPE_GROUPS.keys())
    cls_to_shape = {}
    for shape, members in SHAPE_GROUPS.items():
        for m in members:
            cls_to_shape[m] = shape_names.index(shape)
    return shape_names, [cls_to_shape[class_names[l]] for l in labels]


def subgroup_data(class_names, paths, labels, shape):
    """특정 모양 그룹의 데이터만 추출, 서브클래스 레이블 생성"""
    members = SHAPE_GROUPS[shape]
    sub_names = [m.split("_door_")[0] for m in members]
    sp, sl = [], []
    for p, l in zip(paths, labels):
        if class_names[l] in members:
            sp.append(p)
            sl.append(members.index(class_names[l]))
    return sub_names, sp, sl


def make_loader(paths, labels, img_size, is_train, batch_size):
    """DataLoader 생성"""
    tf = RGBDTransform(img_size, is_train=is_train)
    ds = RGBDDataset(paths, labels, transform=tf)
    nw = NUM_WORKERS if device.type == "cuda" else 0
    return DataLoader(
        ds, batch_size=batch_size, shuffle=is_train,
        num_workers=nw, pin_memory=device.type == "cuda",
        prefetch_factor=PREFETCH if nw > 0 else None,
        persistent_workers=nw > 0,
    )


def class_weights(labels, num_classes):
    """클래스 불균형 가중치 계산"""
    counts = [max(1, sum(1 for l in labels if l == i))
              for i in range(num_classes)]
    total = len(labels)
    return torch.tensor(
        [total / (num_classes * c) for c in counts],
        dtype=torch.float32,
    ).to(device)


def evaluate(model, loader, criterion):
    """모델 평가 → (loss, accuracy)"""
    model.eval()
    loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, aux, lbl in loader:
            imgs = imgs.to(device, non_blocking=True)
            aux = aux.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            out = model(imgs, aux)
            loss += criterion(out, lbl).item() * imgs.size(0)
            _, pred = torch.max(out, 1)
            total += lbl.size(0)
            correct += (pred == lbl).sum().item()
    return loss / len(loader.dataset), 100.0 * correct / total


def train_model(tag, train_loader, test_loader, num_classes, weights):
    """모델 학습 → (best_state_dict, best_metric)"""
    model = RGBDAuxResNet18(num_classes, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=3, factor=0.5)

    full_mode = test_loader is None
    best_state = None
    best_metric = float("inf") if full_mode else 0.0
    patience_cnt = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        t_loss, correct, total = 0.0, 0, 0
        for imgs, aux, lbl in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            aux = aux.to(device, non_blocking=True)
            lbl = lbl.to(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(imgs, aux)
            loss = criterion(out, lbl)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * imgs.size(0)
            _, pred = torch.max(out, 1)
            total += lbl.size(0)
            correct += (pred == lbl).sum().item()

        t_loss /= len(train_loader.dataset)
        t_acc = 100.0 * correct / total

        if full_mode:
            scheduler.step(t_loss)
            improved = t_loss < best_metric
        else:
            v_loss, v_acc = evaluate(model, test_loader, criterion)
            scheduler.step(v_loss)
            improved = v_acc > best_metric

        if improved:
            best_metric = v_acc if not full_mode else t_loss
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            if full_mode:
                print(f"  [{tag}] E{epoch+1:>3d}/{NUM_EPOCHS}: "
                      f"Loss={t_loss:.4f}, Acc={t_acc:.1f}%, LR={lr:.6f}")
            else:
                print(f"  [{tag}] E{epoch+1:>3d}/{NUM_EPOCHS}: "
                      f"Train={t_acc:.1f}%, Val={v_acc:.1f}%, LR={lr:.6f}")

        if device.type == "cuda" and (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()

        if patience_cnt >= PATIENCE:
            print(f"  [{tag}] Early stopping (epoch {epoch+1})")
            break

    return best_state, best_metric


def auto_batch_size(img_size, force_cpu=False):
    """해상도에 따른 배치 사이즈 자동 조정"""
    if force_cpu or not torch.cuda.is_available():
        return 8
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if img_size >= 1920:
        return 4 if gpu_gb >= 24 else 2
    if img_size >= 960:
        return 16 if gpu_gb >= 24 else 8
    return 64 if gpu_gb >= 24 else 32


# ═══════════════════════════════════════════════════════════
# 데이터 로드 및 분할
# ═══════════════════════════════════════════════════════════

print(f"\n데이터셋: {args.dataset_dir}")
class_names, all_paths, all_labels = scan_dataset(args.dataset_dir)
print(f"총 {len(all_paths)}장, {len(class_names)}클래스")

if args.full_train:
    train_paths, train_labels = all_paths, all_labels
    test_paths, test_labels = [], []
else:
    indices = list(range(len(all_paths)))
    tr_idx, te_idx = train_test_split(
        indices, test_size=TEST_SIZE,
        random_state=RANDOM_SEED, stratify=all_labels)
    train_paths = [all_paths[i] for i in tr_idx]
    train_labels = [all_labels[i] for i in tr_idx]
    test_paths = [all_paths[i] for i in te_idx]
    test_labels = [all_labels[i] for i in te_idx]
    print(f"Train: {len(train_paths)}장, Test: {len(test_paths)}장")


# ═══════════════════════════════════════════════════════════
# Step 1: 모양 분류 (LH_FRT / LH_RR / RH)
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("Step 1: 모양 분류 (LH_FRT / LH_RR / RH)")
print(f"  해상도: {args.step1_size}x{args.step1_size}")
print("=" * 70)
step1_start = time.time()

shape_names, train_shape_lbl = to_shape_labels(class_names, train_labels)
print(f"클래스: {shape_names}")

bs1 = auto_batch_size(args.step1_size, args.cpu)
step1_train = make_loader(train_paths, train_shape_lbl, args.step1_size,
                          True, bs1)
step1_test = None
if not args.full_train:
    _, test_shape_lbl = to_shape_labels(class_names, test_labels)
    step1_test = make_loader(test_paths, test_shape_lbl, args.step1_size,
                             False, bs1)

w1 = class_weights(train_shape_lbl, len(shape_names))
step1_state, step1_metric = train_model(
    "Shape", step1_train, step1_test, len(shape_names), w1)

step1_path = os.path.join(ARTIFACTS_DIR, "best_door_cad_shape_5090.pth")
torch.save(step1_state, step1_path)
with open(os.path.join(ARTIFACTS_DIR,
          "class_names_door_cad_shape_5090.json"), "w",
          encoding="utf-8") as f:
    json.dump(shape_names, f, ensure_ascii=False, indent=2)

step1_time = time.time() - step1_start
metric_label = "최저 Train Loss" if args.full_train else "Val Accuracy"
metric_fmt = f"{step1_metric:.4f}" if args.full_train else f"{step1_metric:.1f}%"
print(f"\nStep 1 완료: {metric_label}={metric_fmt} ({step1_time:.1f}초)")


# ═══════════════════════════════════════════════════════════
# Step 2: 기종 분류 (그룹별, 고해상도)
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print(f"Step 2: 기종 분류 (해상도 {args.step2_size}x{args.step2_size})")
print("=" * 70)
step2_start = time.time()

bs2 = auto_batch_size(args.step2_size, args.cpu)
print(f"배치 사이즈: {bs2}")

step2_info = {}
for shape in shape_names:
    print(f"\n── {shape} 그룹 ──")

    sub_names, sub_tr_p, sub_tr_l = subgroup_data(
        class_names, train_paths, train_labels, shape)
    print(f"  클래스: {sub_names}, 학습: {len(sub_tr_p)}장")

    sub_train = make_loader(sub_tr_p, sub_tr_l, args.step2_size, True, bs2)

    sub_test = None
    if not args.full_train:
        _, sub_te_p, sub_te_l = subgroup_data(
            class_names, test_paths, test_labels, shape)
        if sub_te_p:
            sub_test = make_loader(sub_te_p, sub_te_l,
                                   args.step2_size, False, bs2)
            print(f"  테스트: {len(sub_te_p)}장")

    w2 = class_weights(sub_tr_l, len(sub_names))
    state, metric = train_model(shape, sub_train, sub_test,
                                len(sub_names), w2)

    model_path = os.path.join(ARTIFACTS_DIR,
                              f"best_door_cad_{shape}_5090.pth")
    torch.save(state, model_path)

    cn_path = os.path.join(ARTIFACTS_DIR,
                           f"class_names_door_cad_{shape}_5090.json")
    with open(cn_path, "w", encoding="utf-8") as f:
        json.dump(sub_names, f, ensure_ascii=False, indent=2)

    step2_info[shape] = {"sub_names": sub_names, "metric": metric}
    m_fmt = f"{metric:.4f}" if args.full_train else f"{metric:.1f}%"
    print(f"  → {metric_label}: {m_fmt}")

step2_time = time.time() - step2_start
print(f"\nStep 2 완료 ({step2_time:.1f}초)")


# ═══════════════════════════════════════════════════════════
# Cascade 평가 (Step1 → Step2)
# ═══════════════════════════════════════════════════════════

if not args.full_train and test_paths:
    print("\n" + "=" * 70)
    print("Cascade 평가 (Step1 → Step2)")
    print("=" * 70)
    cascade_start = time.time()

    # 모델 로드
    s1_model = RGBDAuxResNet18(len(shape_names), pretrained=False).to(device)
    s1_model.load_state_dict(step1_state)
    s1_model.eval()
    s1_tf = RGBDTransform(args.step1_size, is_train=False)

    s2_models, s2_tfs = {}, {}
    for shape in shape_names:
        n_cls = len(step2_info[shape]["sub_names"])
        m = RGBDAuxResNet18(n_cls, pretrained=False).to(device)
        p = os.path.join(ARTIFACTS_DIR, f"best_door_cad_{shape}_5090.pth")
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        s2_models[shape] = m
        s2_tfs[shape] = RGBDTransform(args.step2_size, is_train=False)

    # 테스트 셋 cascade 추론
    correct, total = 0, 0
    class_correct = [0] * len(class_names)
    class_total = [0] * len(class_names)
    step1_errors = 0

    for path, gt_label in zip(test_paths, test_labels):
        gt_class = class_names[gt_label]

        rgb = Image.open(path).convert("RGB")
        depth_path = path.replace("rgb_", "depth_")
        if os.path.exists(depth_path):
            raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth_mm = (raw.astype(np.float32)
                        if raw is not None
                        else np.zeros((rgb.height, rgb.width), np.float32))
            depth_norm = np.clip(depth_mm / MAX_DEPTH_MM, 0, 1)
        else:
            depth_mm = np.zeros((rgb.height, rgb.width), np.float32)
            depth_norm = depth_mm

        aux = compute_aux_features(depth_mm, ISAAC_SIM_INTRINSICS)
        aux_t = torch.tensor([aux], dtype=torch.float32).to(device)

        # Step 1: 모양 예측
        inp1 = s1_tf(rgb, depth_norm).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_shape_idx = s1_model(inp1, aux_t).argmax(1).item()
        pred_shape = shape_names[pred_shape_idx]

        # Step 1 정답 확인
        gt_shape = None
        for s, members in SHAPE_GROUPS.items():
            if gt_class in members:
                gt_shape = s
                break
        if pred_shape != gt_shape:
            step1_errors += 1

        # Step 2: 기종 예측
        inp2 = s2_tfs[pred_shape](rgb, depth_norm).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_sub_idx = s2_models[pred_shape](inp2, aux_t).argmax(1).item()
        pred_sub = step2_info[pred_shape]["sub_names"][pred_sub_idx]

        pred_full = f"{pred_sub}_door_{pred_shape}"

        total += 1
        class_total[gt_label] += 1
        if pred_full == gt_class:
            correct += 1
            class_correct[gt_label] += 1

    cascade_acc = 100.0 * correct / total
    step1_acc = 100.0 * (total - step1_errors) / total

    print(f"\nStep 1 정확도: {step1_acc:.1f}% ({total - step1_errors}/{total})")
    print(f"Cascade 최종 정확도: {cascade_acc:.1f}% ({correct}/{total})")
    print(f"\n클래스별 정확도:")
    for i, cn in enumerate(class_names):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            print(f"  {cn}: {acc:.1f}% ({class_correct[i]}/{class_total[i]})")

    cascade_time = time.time() - cascade_start
    print(f"\nCascade 평가 완료 ({cascade_time:.1f}초)")


# ═══════════════════════════════════════════════════════════
# 학습 정보 저장 및 요약
# ═══════════════════════════════════════════════════════════

train_info = {
    "mode": "full_train" if args.full_train else "split",
    "step1_size": args.step1_size,
    "step2_size": args.step2_size,
    "shape_groups": SHAPE_GROUPS,
    "class_names": class_names,
    "step1_metric": step1_metric,
    "step2_metrics": {s: v["metric"] for s, v in step2_info.items()},
}
info_path = os.path.join(ARTIFACTS_DIR,
                         "training_info_door_cad_2step_5090.json")
with open(info_path, "w", encoding="utf-8") as f:
    json.dump(train_info, f, ensure_ascii=False, indent=2)

total_time = time.time() - total_start

print("\n" + "=" * 70)
print("2단계 학습 완료!")
print("=" * 70)
print(f"  모드: {'전체 학습' if args.full_train else '분할 학습 (80/20)'}")
print(f"  Step 1: {args.step1_size}px, {metric_label}={metric_fmt}")
for shape, info in step2_info.items():
    m = info["metric"]
    m2 = f"{m:.4f}" if args.full_train else f"{m:.1f}%"
    print(f"  Step 2 [{shape}]: {args.step2_size}px, "
          f"{metric_label}={m2}, 클래스={info['sub_names']}")
print(f"  총 실행 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
print(f"\n저장된 모델:")
print(f"  Step 1: {step1_path}")
for shape in shape_names:
    p = os.path.join(ARTIFACTS_DIR, f"best_door_cad_{shape}_5090.pth")
    print(f"  Step 2 [{shape}]: {p}")

finish_logging()
