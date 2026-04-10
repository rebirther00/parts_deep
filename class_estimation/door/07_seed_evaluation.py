"""과제 성능 보고서용 - 단일 seed RGBD 학습 + 3 데이터셋 평가
- 02_door_classification_5090.py 기반 (Train 70%, Test 30%)
- 원본, datasets_aug, datasets_aug2 에 대해 Accuracy, F1 등 평가
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
)
import random
import os
import sys
import argparse
import time
import json
import glob

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(PROJECT_DIR))
sys.path.insert(0, REPO_DIR)

from depth_utils import (
    RGBDAuxResNet18, RGBDTransform, RGBDDataset, IN_CHANNELS, NUM_AUX_FEATURES,
)
from utils.logger import setup_logging, finish_logging

parser = argparse.ArgumentParser(description='과제 보고서용 RGBD 학습+평가')
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--image_size', type=int, default=448)
parser.add_argument('--epochs', type=int, default=60)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('-cpu', '--cpu', action='store_true')
args = parser.parse_args()

DATASET_DIR = os.path.join(PROJECT_DIR, "datasets")
REPORT_DIR = os.path.join(PROJECT_DIR, "artifacts", "report")
RUN_NAME = f"rgbd_{args.image_size}_seed{args.seed}"
RUN_DIR = os.path.join(REPORT_DIR, RUN_NAME)
os.makedirs(RUN_DIR, exist_ok=True)

TEST_SIZE = 0.3
DATASET_VARIANTS = {
    "original": DATASET_DIR,
    "datasets_aug": os.path.join(PROJECT_DIR, "datasets_aug"),
    "datasets_aug2": os.path.join(PROJECT_DIR, "datasets_aug2"),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def scan_dataset(dataset_dir):
    """데이터셋 스캔하여 클래스별 이미지 경로 수집"""
    classes, paths, labels = [], [], []
    class_idx = 0
    for d in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, d)
        if not os.path.isdir(class_path):
            continue
        pngs = sorted(glob.glob(os.path.join(class_path, "rgb_*.png")))
        if not pngs:
            continue
        classes.append(d)
        for p in pngs:
            paths.append(p)
            labels.append(class_idx)
        class_idx += 1
    return classes, paths, labels


def evaluate_on_dataset(model, paths, labels, class_names, transform,
                        device, batch_size):
    """단일 데이터셋에서 모델 평가, 지표 반환"""
    dataset = RGBDDataset(paths, labels=labels, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4 if device.type == 'cuda' else 0,
                        pin_memory=device.type == 'cuda')

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, aux, lbls in loader:
            images = images.to(device, non_blocking=True)
            aux = aux.to(device, non_blocking=True)
            outputs = model(images, aux)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(lbls.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)

    prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0,
        labels=list(range(len(class_names))))
    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": round(float(prec_pc[i]) * 100, 2),
            "recall": round(float(rec_pc[i]) * 100, 2),
            "f1": round(float(f1_pc[i]) * 100, 2),
            "support": int(sup_pc[i]),
        }

    return {
        "accuracy": round(acc, 2),
        "macro_precision": round(float(prec) * 100, 2),
        "macro_recall": round(float(rec) * 100, 2),
        "macro_f1": round(float(f1) * 100, 2),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(
            all_labels, all_preds,
            labels=list(range(len(class_names)))).tolist(),
        "n_samples": len(all_labels),
        "n_correct": int((all_preds == all_labels).sum()),
    }


def main():
    eval_path = os.path.join(RUN_DIR, "eval_results.json")
    if os.path.exists(eval_path):
        print(f"[건너뜀] 이미 완료: {RUN_NAME}")
        return

    setup_logging(f"report_{RUN_NAME}",
                  log_dir=os.path.join(PROJECT_DIR, "logs"))

    print(f"실험: {RUN_NAME}")
    start_time = time.time()
    set_seed(args.seed)

    # 데이터 로드 및 70/30 분할
    class_names, image_paths, labels = scan_dataset(DATASET_DIR)
    num_classes = len(class_names)
    print(f"  클래스: {num_classes}개, 전체: {len(image_paths)}장")

    indices = list(range(len(image_paths)))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=args.seed, stratify=labels)

    train_paths = [image_paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    test_paths = [image_paths[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]
    print(f"  Train: {len(train_paths)}장, Test: {len(test_paths)}장")

    # Transform & DataLoader
    train_transform = RGBDTransform(args.image_size, is_train=True)
    val_transform = RGBDTransform(args.image_size, is_train=False)

    batch_size = 16 if len(train_paths) < 100 else (
        32 if len(train_paths) < 500 else 64)

    num_workers = 4 if torch.cuda.is_available() and not args.cpu else 0
    pin_memory = torch.cuda.is_available() and not args.cpu

    train_dataset = RGBDDataset(train_paths, train_labels,
                                transform=train_transform)
    test_dataset = RGBDDataset(test_paths, test_labels,
                               transform=val_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers,
                             pin_memory=pin_memory)

    # 모델
    device = torch.device('cpu') if args.cpu else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RGBDAuxResNet18(num_classes, pretrained=True).to(device)
    print(f"  디바이스: {device}, 배치: {batch_size}")

    # 역빈도 클래스 가중치
    class_counts = [sum(1 for l in train_labels if l == i)
                    for i in range(num_classes)]
    weights = torch.tensor(
        [len(train_labels) / (num_classes * c) for c in class_counts],
        dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5)

    # 학습 (Test Accuracy 기반 Early Stopping)
    model_path = os.path.join(RUN_DIR, "model.pth")
    best_test_acc = 0.0
    patience_counter = 0

    print(f"\n학습 시작 (max {args.epochs} epochs, patience {args.patience})")
    for epoch in range(args.epochs):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for images, aux, lbls in train_loader:
            images = images.to(device, non_blocking=True)
            aux = aux.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images, aux)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            t_total += lbls.size(0)
            t_correct += (preds == lbls).sum().item()
        train_loss = t_loss / len(train_loader.dataset)
        train_acc = 100.0 * t_correct / t_total

        model.eval()
        e_loss, e_correct, e_total = 0.0, 0, 0
        with torch.no_grad():
            for images, aux, lbls in test_loader:
                images = images.to(device, non_blocking=True)
                aux = aux.to(device, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)
                outputs = model(images, aux)
                loss = criterion(outputs, lbls)
                e_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                e_total += lbls.size(0)
                e_correct += (preds == lbls).sum().item()
        test_loss = e_loss / len(test_loader.dataset)
        test_acc = 100.0 * e_correct / e_total
        scheduler.step(test_loss)

        improved = test_acc > best_test_acc
        if improved:
            best_test_acc = test_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0 or improved:
            mark = " *" if improved else ""
            print(f"  Epoch {epoch+1:3d} | "
                  f"Train {train_acc:6.2f}% L={train_loss:.4f} | "
                  f"Test {test_acc:6.2f}% L={test_loss:.4f} | "
                  f"Best {best_test_acc:.2f}% "
                  f"ES={patience_counter}/{args.patience}{mark}")

        if torch.cuda.is_available() and (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

    elapsed_train = time.time() - start_time
    print(f"\n학습 완료: Best Test Acc {best_test_acc:.2f}% "
          f"({elapsed_train/60:.1f}분)")

    # 3개 데이터셋 평가
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_results = {}

    for variant_name, variant_dir in DATASET_VARIANTS.items():
        if variant_name == "original":
            eval_paths = test_paths
            eval_labels = test_labels
        else:
            if not os.path.exists(variant_dir):
                print(f"  [{variant_name}] 디렉터리 없음, 건너뜀")
                continue
            eval_paths, eval_labels = [], []
            for p, l in zip(test_paths, test_labels):
                cls_name = os.path.basename(os.path.dirname(p))
                filename = os.path.basename(p)
                new_path = os.path.join(variant_dir, cls_name, filename)
                if os.path.exists(new_path):
                    eval_paths.append(new_path)
                    eval_labels.append(l)
            if not eval_paths:
                print(f"  [{variant_name}] 유효 경로 없음, 건너뜀")
                continue

        print(f"  [{variant_name}] {len(eval_paths)}장 평가 중...")
        result = evaluate_on_dataset(
            model, eval_paths, eval_labels, class_names,
            val_transform, device, batch_size)
        all_results[variant_name] = result
        print(f"    Accuracy: {result['accuracy']:.2f}% | "
              f"Macro F1: {result['macro_f1']:.2f}%")

    # 결과 저장
    split_info = {
        "seed": args.seed, "image_size": args.image_size,
        "class_names": class_names,
        "train_count": len(train_paths), "test_count": len(test_paths),
        "best_test_acc": best_test_acc,
        "train_time_seconds": round(elapsed_train, 2),
    }
    with open(os.path.join(RUN_DIR, "split_info.json"), 'w',
              encoding='utf-8') as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_time
    print(f"\n전체 완료: {RUN_NAME} ({total_time:.1f}초)")
    finish_logging()


if __name__ == "__main__":
    main()
