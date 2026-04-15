"""
논문용 통합 평가 스크립트
- train_paper.py로 학습된 모델을 3개 데이터셋에서 평가
- sklearn 기반 Precision/Recall/F1 산출
- Confusion matrix PNG 생성 (matplotlib)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import json
import argparse
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix as sk_confusion_matrix, classification_report,
)
from torchvision import models

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(PROJECT_DIR))
sys.path.insert(0, REPO_DIR)

from depth_utils import (
    RGBDAuxResNet18, RGBDTransform, RGBDDataset, IN_CHANNELS, NUM_AUX_FEATURES,
)
from rgb_utils import RGBTransform, RGBDataset, RGB_IN_CHANNELS
from rgbe_utils import RGBETransform, RGBEDataset
from edge_utils import EdgeAuxResNet18, EdgeTransform, EdgeDataset, EDGE_IN_CHANNELS


class NoAuxResNet18(nn.Module):
    """Aux MLP 없이 이미지만 사용하는 분류 모델 (ablation용)"""

    def __init__(self, num_classes, in_channels=IN_CHANNELS, pretrained=False):
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


MODEL_CONFIGS = {
    "rgb": {
        "dataset_cls": RGBDataset,
        "transform_fn": lambda sz: RGBTransform(sz, is_train=False),
        "model_fn": lambda nc: NoAuxResNet18(nc, in_channels=RGB_IN_CHANNELS,
                                              pretrained=False),
        "in_channels": RGB_IN_CHANNELS,
    },
    "rgbd": {
        "dataset_cls": RGBDDataset,
        "transform_fn": lambda sz: RGBDTransform(sz, is_train=False),
        "model_fn": lambda nc: RGBDAuxResNet18(nc, pretrained=False),
        "in_channels": IN_CHANNELS,
    },
    "texture_aug": {
        "dataset_cls": RGBDDataset,
        "transform_fn": lambda sz: RGBDTransform(sz, is_train=False),
        "model_fn": lambda nc: RGBDAuxResNet18(nc, pretrained=False),
        "in_channels": IN_CHANNELS,
    },
    "edge": {
        "dataset_cls": EdgeDataset,
        "transform_fn": lambda sz: EdgeTransform(sz, is_train=False),
        "model_fn": lambda nc: EdgeAuxResNet18(nc, pretrained=False),
        "in_channels": EDGE_IN_CHANNELS,
    },
    "rgbe": {
        "dataset_cls": RGBEDataset,
        "transform_fn": lambda sz: RGBETransform(sz, is_train=False),
        "model_fn": lambda nc: RGBDAuxResNet18(nc, pretrained=False),
        "in_channels": IN_CHANNELS,
    },
    "rgbe_texture_aug": {
        "dataset_cls": RGBEDataset,
        "transform_fn": lambda sz: RGBETransform(sz, is_train=False),
        "model_fn": lambda nc: RGBDAuxResNet18(nc, pretrained=False),
        "in_channels": IN_CHANNELS,
    },
}

DATASET_VARIANTS = ["original", "datasets_aug", "datasets_aug2"]

parser = argparse.ArgumentParser(description='논문용 통합 평가')
parser.add_argument('--model_type', type=str, required=True,
                    choices=list(MODEL_CONFIGS.keys()))
parser.add_argument('--seed', type=int, required=True)
parser.add_argument('--image_size', type=int, default=448)
parser.add_argument('--no_aux', action='store_true',
                    help='Aux MLP 제거 ablation 실험')
parser.add_argument('-cpu', '--cpu', action='store_true')
args = parser.parse_args()


def remap_paths(test_paths, target_dir):
    """test_paths의 데이터셋 루트를 target_dir로 치환"""
    remapped = []
    for p in test_paths:
        cls_name = os.path.basename(os.path.dirname(p))
        filename = os.path.basename(p)
        new_path = os.path.join(target_dir, cls_name, filename)
        if os.path.exists(new_path):
            remapped.append(new_path)
    return remapped


def evaluate_on_dataset(model, dataset_cls, transform, paths, class_names,
                        device, batch_size=32):
    """단일 데이터셋에서 모델 평가, 지표 반환"""
    dataset = dataset_cls(paths, transform=transform, class_names=class_names)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4 if device.type == 'cuda' else 0,
                        pin_memory=device.type == 'cuda')

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, aux, labels in loader:
            images = images.to(device, non_blocking=True)
            aux = aux.to(device, non_blocking=True)
            outputs = model(images, aux)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds) * 100
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    cm = sk_confusion_matrix(all_labels, all_preds,
                             labels=list(range(len(class_names))))

    per_class = {}
    prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0,
        labels=list(range(len(class_names))))
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
        "confusion_matrix": cm.tolist(),
        "n_samples": len(all_labels),
        "n_correct": int((all_preds == all_labels).sum()),
    }


def plot_confusion_matrix(cm, class_names, title, save_path):
    """Confusion matrix 히트맵 저장"""
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.2), max(7, n * 1.1)))

    cm_pct = cm.astype(float)
    row_sums = cm_pct.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm_pct / row_sums * 100, 0)

    im = ax.imshow(cm_pct, cmap='Blues', aspect='auto',
                   vmin=0, vmax=100)

    short_names = [n.replace("_door_", "\n").replace("E30_E38", "E3x")
                   for n in class_names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title, fontsize=11)

    for i in range(n):
        for j in range(n):
            count = cm[i][j]
            pct = cm_pct[i][j]
            color = 'white' if pct > 60 else 'black'
            ax.text(j, i, f"{count}\n({pct:.0f}%)",
                    ha='center', va='center', fontsize=7, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label='%')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Confusion matrix 저장: {save_path}")


def main():
    cfg = MODEL_CONFIGS[args.model_type]
    suffix = "_noaux" if args.no_aux else ""
    run_name = f"{args.model_type}{suffix}_{args.image_size}_seed{args.seed}"
    run_dir = os.path.join(PROJECT_DIR, "artifacts", run_name)

    eval_path = os.path.join(run_dir, "eval_results.json")
    if os.path.exists(eval_path):
        print(f"[건너뜀] 이미 평가 완료: {run_name}")
        return

    model_path = os.path.join(run_dir, "model.pth")
    split_path = os.path.join(run_dir, "split_info.json")

    if not os.path.exists(model_path):
        print(f"[오류] 모델 파일 없음: {model_path}")
        return
    if not os.path.exists(split_path):
        print(f"[오류] 분할 정보 없음: {split_path}")
        return

    from utils.logger import setup_logging, finish_logging
    setup_logging(f"eval_{run_name}",
                  log_dir=os.path.join(PROJECT_DIR, "logs"))

    print(f"평가: {run_name}")
    start_time = time.time()

    with open(split_path, 'r', encoding='utf-8') as f:
        split_info = json.load(f)

    class_names = split_info["class_names"]
    test_paths = split_info["test_paths"]
    num_classes = len(class_names)

    device = torch.device('cpu') if args.cpu else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.no_aux:
        in_ch = cfg.get("in_channels", IN_CHANNELS)
        model = NoAuxResNet18(num_classes, in_channels=in_ch,
                              pretrained=False).to(device)
    else:
        model = cfg["model_fn"](num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"  모델 로드 완료 ({device})")

    transform = cfg["transform_fn"](args.image_size)
    all_results = {}

    for variant in DATASET_VARIANTS:
        if variant == "original":
            paths = test_paths
            ds_label = "Test Original"
        else:
            target_dir = os.path.join(PROJECT_DIR, variant)
            paths = remap_paths(test_paths, target_dir)
            ds_label = variant

        if not paths:
            print(f"  [{ds_label}] 경로 없음, 건너뜀")
            continue

        print(f"  [{ds_label}] {len(paths)}장 평가 중...")
        result = evaluate_on_dataset(
            model, cfg["dataset_cls"], transform, paths,
            class_names, device)
        all_results[variant] = result

        print(f"    Accuracy: {result['accuracy']:.2f}% | "
              f"Macro F1: {result['macro_f1']:.2f}%")

        cm = np.array(result["confusion_matrix"])
        cm_path = os.path.join(run_dir, f"confusion_matrix_{variant}.png")
        plot_confusion_matrix(
            cm, class_names,
            f"{args.model_type.upper()} {args.image_size} seed{args.seed}\n{ds_label}",
            cm_path)

    # 하락폭 계산
    if "original" in all_results:
        orig_acc = all_results["original"]["accuracy"]
        orig_f1 = all_results["original"]["macro_f1"]
        for variant in ["datasets_aug", "datasets_aug2"]:
            if variant in all_results:
                all_results[variant]["delta_acc"] = round(
                    all_results[variant]["accuracy"] - orig_acc, 2)
                all_results[variant]["delta_f1"] = round(
                    all_results[variant]["macro_f1"] - orig_f1, 2)

    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print(f"\n평가 완료: {run_name} ({elapsed:.1f}초)")
    print(f"  결과 저장: {eval_path}")

    finish_logging()


if __name__ == "__main__":
    main()
