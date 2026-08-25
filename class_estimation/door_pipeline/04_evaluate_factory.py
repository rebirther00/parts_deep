"""RGBE NoAux 모델 현장 데이터(datasets_factory) 평가 스크립트

ZED 2i로 촬영한 현장 데이터에 대해 RGBE NoAux 모델을 평가한다.
- 모델: NoAuxResNet18 (RGBE 4ch, Aux 미사용)
- 데이터: datasets_factory (일부 클래스만 존재할 수 있음)

실행:
    python evaluate_factory.py
    python evaluate_factory.py --model artifacts/rgbe_noaux_448_seed42/model.pth
    python evaluate_factory.py --dataset_dir ../door/datasets_factory
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import sys
import json
import argparse
import glob
import time

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(os.path.dirname(PROJECT_DIR))
sys.path.insert(0, REPO_DIR)

from rgbe_utils import RGBETransform, RGBEDataset, RGBE_IN_CHANNELS

ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
DEFAULT_MODEL_DIR = os.path.join(ARTIFACTS_DIR, "rgbe_noaux_448_seed42")
DEFAULT_FACTORY_DIR = os.path.join(
    os.path.dirname(PROJECT_DIR), "door", "datasets_factory")


# ── NoAuxResNet18 ────────────────────────────────────────

class NoAuxResNet18(nn.Module):
    """Aux MLP 없이 이미지만 사용하는 분류 모델.

    forward()는 aux_features를 인자로 받되 무시하여 API 호환성 유지.
    """

    def __init__(self, num_classes, in_channels=RGBE_IN_CHANNELS,
                 pretrained=False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(in_channels, 64,
                                 kernel_size=7, stride=2, padding=3,
                                 bias=False)
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                for c in range(3, in_channels):
                    new_conv.weight[:, c:c + 1] = old_conv.weight.mean(
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

    def forward(self, images, aux_features=None):
        img_feat = self.backbone(images)
        return self.classifier(img_feat)


# ── CLI ──────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description='RGBE NoAux 모델 현장 데이터 평가',
    formatter_class=argparse.RawTextHelpFormatter,
)
parser.add_argument(
    '--model', type=str,
    default=os.path.join(DEFAULT_MODEL_DIR, "model.pth"),
    help='모델 파일 경로',
)
parser.add_argument(
    '--dataset_dir', type=str,
    default=DEFAULT_FACTORY_DIR,
    help='평가 데이터셋 경로 (기본: ../door/datasets_factory)',
)
parser.add_argument(
    '--image_size', type=int, default=448,
    help='입력 이미지 크기 (기본: 448)',
)
parser.add_argument('-cpu', '--cpu', action='store_true',
                    help='CPU로 강제 실행')
args = parser.parse_args()


# ── 유틸리티 ─────────────────────────────────────────────

def load_class_names(model_dir):
    """split_info.json에서 클래스명을 로드한다."""
    path = os.path.join(model_dir, "split_info.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"split_info.json이 없습니다: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)["class_names"]


def scan_factory_dataset(dataset_dir, class_names):
    """datasets_factory에서 클래스별 rgb_*.png를 스캔한다.

    존재하는 클래스만 수집하고, 빈 클래스는 건너뛴다.
    """
    all_paths = []
    found_classes = {}
    for cls_name in class_names:
        cls_dir = os.path.join(dataset_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        pngs = sorted(glob.glob(os.path.join(cls_dir, "rgb_*.png")))
        if pngs:
            found_classes[cls_name] = len(pngs)
            all_paths.extend(pngs)
    return all_paths, found_classes


def load_font():
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


# ── 시각화 ───────────────────────────────────────────────

def create_result_grid(image_paths, labels, predictions, confidences,
                       class_names, save_path, num_cols=5, img_size=200,
                       max_images=50):
    """예측 결과를 그리드 형태로 시각화"""
    num_images = min(len(image_paths), max_images)
    num_rows = (num_images + num_cols - 1) // num_cols

    text_height = 100
    cell_height = img_size + text_height

    grid_image = Image.new(
        'RGB', (num_cols * img_size, num_rows * cell_height), 'white')
    font = load_font()

    for idx in range(num_images):
        row, col = divmod(idx, num_cols)
        x, y = col * img_size, row * cell_height

        img = Image.open(image_paths[idx]).convert('RGB')
        img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        grid_image.paste(img, (x, y))

        draw = ImageDraw.Draw(grid_image)
        text_y = y + img_size + 5

        is_correct = labels[idx] == predictions[idx]
        text_color = (0, 150, 0) if is_correct else (200, 0, 0)
        bg_color = (230, 255, 230) if is_correct else (255, 230, 230)
        draw.rectangle(
            [x, y + img_size, x + img_size, y + cell_height], fill=bg_color)

        symbol = "O" if is_correct else "X"
        draw.text((x + 5, text_y),
                  f"{symbol} Actual: {class_names[labels[idx]]}",
                  fill=text_color, font=font)
        draw.text((x + 5, text_y + 20),
                  f"Pred: {class_names[predictions[idx]]}",
                  fill=(0, 0, 0), font=font)
        draw.text((x + 5, text_y + 40),
                  f"Conf: {confidences[idx]:.1f}%",
                  fill=(100, 100, 100), font=font)
        draw.text((x + 5, text_y + 60),
                  f"src: {os.path.basename(image_paths[idx])}",
                  fill=(60, 60, 60), font=font)

    grid_image.save(save_path, 'PNG', quality=95)
    print(f"결과 이미지 저장: {save_path}")


def create_wrong_predictions_grid(image_paths, labels, predictions,
                                  confidences, class_names, save_path,
                                  num_cols=5, img_size=200):
    """틀린 예측만 그리드로 시각화"""
    wrong = [i for i in range(len(labels)) if labels[i] != predictions[i]]
    if not wrong:
        print("모든 예측이 정확합니다! 틀린 예측 이미지가 없습니다.")
        return

    num_rows = (len(wrong) + num_cols - 1) // num_cols
    text_height = 100
    cell_height = img_size + text_height

    grid_image = Image.new(
        'RGB', (num_cols * img_size, num_rows * cell_height), 'white')
    font = load_font()

    for grid_idx, data_idx in enumerate(wrong):
        row, col = divmod(grid_idx, num_cols)
        x, y = col * img_size, row * cell_height

        img = Image.open(image_paths[data_idx]).convert('RGB')
        img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        grid_image.paste(img, (x, y))

        draw = ImageDraw.Draw(grid_image)
        text_y = y + img_size + 5
        draw.rectangle(
            [x, y + img_size, x + img_size, y + cell_height],
            fill=(255, 220, 220))
        draw.text((x + 5, text_y),
                  f"X Actual: {class_names[labels[data_idx]]}",
                  fill=(200, 0, 0), font=font)
        draw.text((x + 5, text_y + 20),
                  f"Pred: {class_names[predictions[data_idx]]}",
                  fill=(0, 0, 0), font=font)
        draw.text((x + 5, text_y + 40),
                  f"Conf: {confidences[data_idx]:.1f}%",
                  fill=(100, 100, 100), font=font)
        draw.text((x + 5, text_y + 60),
                  f"src: {os.path.basename(image_paths[data_idx])}",
                  fill=(60, 60, 60), font=font)

    grid_image.save(save_path, 'PNG', quality=95)
    print(f"틀린 예측 이미지 저장: {save_path} ({len(wrong)}개)")


def create_confusion_matrix_heatmap(cm, class_names, save_path):
    """혼동 행렬 히트맵 이미지 저장"""
    n = len(class_names)
    cm_arr = np.array(cm, dtype=np.float32)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    with np.errstate(invalid='ignore'):
        cm_pct = np.where(row_sums > 0, cm_arr / row_sums * 100, 0)

    cell_size = 80
    label_margin = 180
    title_height = 50
    colorbar_width = 60
    w = label_margin + n * cell_size + colorbar_width + 20
    h = title_height + label_margin + n * cell_size + 10

    img = Image.new('RGB', (w, h), 'white')
    draw = ImageDraw.Draw(img)
    font = load_font()

    draw.text((w // 2 - 80, 10), "Confusion Matrix (Factory)",
              fill='black', font=font)

    ox, oy = label_margin, title_height + label_margin // 2

    for i in range(n):
        for j in range(n):
            x0, y0 = ox + j * cell_size, oy + i * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            pct = cm_pct[i][j]

            if i == j:
                intensity = int(min(pct / 100, 1.0) * 200)
                color = (220 - intensity, 255 - intensity // 4,
                         220 - intensity)
            else:
                intensity = int(min(pct / 50, 1.0) * 200)
                color = (255 - intensity // 4, 220 - intensity,
                         220 - intensity)

            draw.rectangle([x0, y0, x1, y1], fill=color,
                           outline=(180, 180, 180))

            count = cm[i][j]
            text = f"{count}\n({pct:.0f}%)" if count > 0 else "0"
            txt_color = (0, 0, 0) if pct < 80 else (255, 255, 255)
            for li, line in enumerate(text.split('\n')):
                tw = (font.getlength(line) if hasattr(font, 'getlength')
                      else len(line) * 7)
                tx = x0 + (cell_size - tw) / 2
                ty = y0 + cell_size // 2 - 12 + li * 14
                draw.text((tx, ty), line, fill=txt_color, font=font)

    for i, name in enumerate(class_names):
        short = name.replace("_door_", "_").replace("E30_E38", "E3x")
        ty = oy + i * cell_size + cell_size // 2 - 6
        draw.text((5, ty), short, fill='black', font=font)
        tx = ox + i * cell_size + cell_size // 2 - len(short) * 3
        draw.text((tx, oy - 18), short, fill='black', font=font)

    draw.text((ox + n * cell_size // 2 - 20, oy - 35), "Predicted",
              fill='black', font=font)

    img.save(save_path, 'PNG', quality=95)
    print(f"혼동 행렬 히트맵 저장: {save_path}")


# ── 메인 ─────────────────────────────────────────────────

def main():
    total_start = time.time()
    model_dir = os.path.dirname(args.model)
    output_prefix = os.path.join(model_dir, "factory")

    # 1. 클래스명·데이터셋 로드
    print("=" * 70)
    print("RGBE NoAux 모델 현장 데이터(datasets_factory) 평가")
    print("=" * 70)

    class_names = load_class_names(model_dir)
    num_classes = len(class_names)

    dataset_dir = os.path.abspath(args.dataset_dir)
    test_paths, found_classes = scan_factory_dataset(dataset_dir, class_names)

    if not test_paths:
        print(f"평가할 이미지가 없습니다: {dataset_dir}")
        raise SystemExit(1)

    print(f"\n모델: {args.model}")
    print(f"데이터셋: {dataset_dir}")
    print(f"전체 클래스: {num_classes}종 {class_names}")
    print(f"\n현장 데이터 현황:")
    for cls_name in class_names:
        count = found_classes.get(cls_name, 0)
        mark = f"{count}장" if count > 0 else "없음"
        print(f"  {cls_name}: {mark}")
    print(f"\n평가 대상: {len(test_paths)}장 "
          f"({len(found_classes)}종: {list(found_classes.keys())})")

    # 2. DataLoader 생성
    transform = RGBETransform(args.image_size, is_train=False)
    dataset = RGBEDataset(test_paths, transform=transform,
                          class_names=class_names)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # 3. 모델 로드
    device = (torch.device('cpu') if args.cpu
              else torch.device('cuda' if torch.cuda.is_available()
                                else 'cpu'))
    model = NoAuxResNet18(num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"\n모델 로드 완료 (디바이스: {device})")

    # 4. 평가 실행
    print("\n" + "-" * 70)
    print("평가 실행")
    print("-" * 70)

    correct, total = 0, 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    all_preds, all_labels, all_confs, all_paths = [], [], [], []

    sample_idx = 0
    with torch.no_grad():
        for images, aux, labels in loader:
            images = images.to(device)
            aux = aux.to(device)
            labels = labels.to(device)

            outputs = model(images, aux)
            _, predictions = torch.max(outputs, 1)

            for i in range(len(labels)):
                label = labels[i].item()
                pred = predictions[i].item()
                conf = torch.softmax(outputs[i], dim=0)[pred].item() * 100

                is_correct = (label == pred)
                if is_correct:
                    correct += 1
                    class_correct[label] += 1
                total += 1
                class_total[label] += 1

                all_preds.append(pred)
                all_labels.append(label)
                all_confs.append(conf)
                all_paths.append(test_paths[sample_idx])

                symbol = "O" if is_correct else "X"
                print(f"{symbol} [{class_names[label]:20s}] -> "
                      f"[{class_names[pred]:20s}] | "
                      f"신뢰도: {conf:.1f}%")
                sample_idx += 1

    # 5. 결과 시각화
    print("\n" + "-" * 70)
    print("결과 시각화")
    print("-" * 70)

    create_result_grid(
        all_paths, all_labels, all_preds, all_confs, class_names,
        f"{output_prefix}_results.png")
    create_wrong_predictions_grid(
        all_paths, all_labels, all_preds, all_confs, class_names,
        f"{output_prefix}_wrong.png")

    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
    for true_l, pred_l in zip(all_labels, all_preds):
        confusion_matrix[true_l][pred_l] += 1

    create_confusion_matrix_heatmap(
        confusion_matrix, class_names,
        f"{output_prefix}_confusion_matrix.png")

    # 6. 결과 요약
    print("\n" + "=" * 70)
    print("평가 결과 요약")
    print("=" * 70)

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"\n전체 정확도: {accuracy:.2f}% ({correct}/{total})")

    print(f"\n클래스별 정확도:")
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            print(f"  {name}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")
        else:
            print(f"  {name}: -- (현장 데이터 없음)")

    # Precision / Recall / F1
    print(f"\n{'클래스':<22s} {'Precision':>10s} {'Recall':>10s} "
          f"{'F1-Score':>10s} {'Support':>8s}")
    print("-" * 62)

    all_precision, all_recall, all_f1 = [], [], []
    for i, name in enumerate(class_names):
        if class_total[i] == 0:
            continue
        tp = confusion_matrix[i][i]
        fp = sum(confusion_matrix[j][i] for j in range(num_classes)) - tp
        fn = sum(confusion_matrix[i][j] for j in range(num_classes)) - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)

        all_precision.append(prec)
        all_recall.append(rec)
        all_f1.append(f1)

        print(f"{name:<22s} {prec * 100:>9.2f}% {rec * 100:>9.2f}% "
              f"{f1 * 100:>9.2f}% {class_total[i]:>7d}")

    avg_prec = np.mean(all_precision) * 100 if all_precision else 0.0
    avg_rec = np.mean(all_recall) * 100 if all_recall else 0.0
    avg_f1 = np.mean(all_f1) * 100 if all_f1 else 0.0
    print("-" * 62)
    print(f"{'평균 (Macro)':<22s} {avg_prec:>9.2f}% {avg_rec:>9.2f}% "
          f"{avg_f1:>9.2f}% {total:>7d}")

    # 오류 분석
    errors = total - correct
    if errors > 0:
        print(f"\n오류 분석 (상위 5개):")
        error_list = [
            (class_names[all_labels[i]], class_names[all_preds[i]],
             all_confs[i])
            for i in range(len(all_labels))
            if all_labels[i] != all_preds[i]
        ]
        error_list.sort(key=lambda x: -x[2])
        for actual, predicted, conf in error_list[:5]:
            print(f"  {actual} -> {predicted} (신뢰도: {conf:.1f}%)")

    # JSON 저장
    results = {
        "model": os.path.basename(args.model),
        "model_dir": os.path.basename(model_dir),
        "dataset_dir": dataset_dir,
        "accuracy": round(accuracy, 2),
        "total_samples": total,
        "correct_samples": correct,
        "found_classes": found_classes,
        "class_accuracies": {
            name: round(100.0 * class_correct[i] / class_total[i], 2)
            if class_total[i] > 0 else None
            for i, name in enumerate(class_names)
        },
        "avg_precision": round(avg_prec, 2),
        "avg_recall": round(avg_rec, 2),
        "avg_f1": round(avg_f1, 2),
        "confusion_matrix": confusion_matrix,
    }
    json_path = f"{output_prefix}_eval_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n결과 JSON 저장: {json_path}")

    # DB 기록 (cross_domain: 현장 데이터)
    from db.db_log import DBLog
    db = DBLog()
    db_model_id = db.find_model(weights_path=args.model,
                                name=os.path.basename(model_dir))
    if db_model_id is None:
        db_model_id = db.register_model(
            name=os.path.basename(model_dir), architecture="NoAuxResNet18",
            in_channels=RGBE_IN_CHANNELS, num_classes=num_classes,
            pretrained_base="ImageNet", weights_path=args.model,
            input_size=f"{args.image_size}x{args.image_size}",
            description="04_evaluate_factory.py에서 소급 등록")
    db.log_evaluation(
        model_id=db_model_id, dataset_name=dataset_dir,
        eval_type="cross_domain", total_samples=total, correct=correct,
        accuracy=results["accuracy"], precision_macro=results["avg_precision"],
        recall_macro=results["avg_recall"], f1_macro=results["avg_f1"],
        confusion_matrix=confusion_matrix,
        per_class_results={"class_accuracies": results["class_accuracies"],
                           "found_classes": found_classes},
        inference_device=str(device), report_path=json_path)
    db.close()

    elapsed = time.time() - total_start
    print(f"\n총 실행 시간: {elapsed:.2f}초")
    print("=" * 70)


if __name__ == "__main__":
    main()
