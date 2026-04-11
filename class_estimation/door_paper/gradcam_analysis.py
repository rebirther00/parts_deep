"""Grad-CAM 시각화 분석 스크립트

4종 모델(RGBD, Texture Aug, Edge-only, RGBE)의 attention 영역을
Grad-CAM으로 시각화하여 논문 Fig. 6, 7을 생성한다.
"""

import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import torch.nn as nn
from torchvision import models

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'door'))
from depth_utils import RGBDTransform, DEPTH_MEAN, DEPTH_STD, MAX_DEPTH_MM
from edge_utils import EdgeTransform
from rgbe_utils import RGBETransform

IN_CHANNELS = 4


class NoAuxResNet18(nn.Module):
    """train_paper.py와 동일한 Aux-free ResNet18 (import 부작용 회피용 복제)."""

    def __init__(self, num_classes, in_channels=IN_CHANNELS, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)
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
        return self.classifier(self.backbone(images))

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
DATASETS_DIR = os.path.join(PROJECT_DIR, "datasets")
DATASETS_AUG_DIR = os.path.join(PROJECT_DIR, "datasets_aug")
OUTPUT_DIR = os.path.join(ARTIFACTS_DIR, "summary_noaux")
IMAGE_SIZE = 448
SEED = 42


# ── Grad-CAM 핵심 구현 ──────────────────────────────────

class GradCAM:
    """ResNet18 backbone.layer4 기반 Grad-CAM."""

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    @torch.no_grad()
    def predict(self, img_tensor, device):
        """추론만 수행 (예측 클래스 반환)."""
        self.model.eval()
        x = img_tensor.unsqueeze(0).to(device)
        dummy_aux = torch.zeros(1, 3, device=device)
        logits = self.model(x, dummy_aux)
        return logits.argmax(dim=1).item()

    def generate(self, img_tensor, target_class, device):
        """Grad-CAM heatmap 생성 (H, W) 범위 [0, 1]."""
        self.model.eval()
        x = img_tensor.unsqueeze(0).to(device).requires_grad_(True)
        dummy_aux = torch.zeros(1, 3, device=device)

        logits = self.model(x, dummy_aux)
        self.model.zero_grad()
        logits[0, target_class].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE),
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


# ── 유틸리티 ─────────────────────────────────────────────

def load_model(model_type, device):
    """모델 로드 및 GradCAM 객체 생성."""
    in_ch = 3 if model_type == "edge" else 4
    model = NoAuxResNet18(num_classes=8, in_channels=in_ch, pretrained=False)

    run_name = f"{model_type}_noaux_{IMAGE_SIZE}_seed{SEED}"
    pth_path = os.path.join(ARTIFACTS_DIR, run_name, "model.pth")
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)

    cam = GradCAM(model, model.backbone.layer4)
    return cam


def preprocess_image(rgb_path, model_type):
    """이미지를 모델별 전처리하여 텐서 반환."""
    rgb_pil = Image.open(rgb_path).convert("RGB")

    if model_type in ("rgbd", "texture_aug"):
        depth_path = rgb_path.replace("rgb_", "depth_")
        if os.path.exists(depth_path):
            depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth_np = depth_raw.astype(np.float32) / MAX_DEPTH_MM
            depth_np = np.clip(depth_np, 0, 1)
        else:
            depth_np = np.zeros((rgb_pil.size[1], rgb_pil.size[0]),
                                dtype=np.float32)
        transform = RGBDTransform(IMAGE_SIZE, is_train=False)
        return transform(rgb_pil, depth_np)

    elif model_type == "edge":
        transform = EdgeTransform(IMAGE_SIZE, is_train=False)
        return transform(rgb_pil)

    elif model_type == "rgbe":
        transform = RGBETransform(IMAGE_SIZE, is_train=False)
        return transform(rgb_pil)


def get_display_rgb(rgb_path):
    """시각화용 RGB (Letterbox 적용)."""
    rgb_pil = Image.open(rgb_path).convert("RGB")
    w, h = rgb_pil.size
    scale = IMAGE_SIZE / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    rgb_pil = rgb_pil.resize((new_w, new_h), Image.BILINEAR)
    pad_left = (IMAGE_SIZE - new_w) // 2
    pad_top = (IMAGE_SIZE - new_h) // 2
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (0, 0, 0))
    canvas.paste(rgb_pil, (pad_left, pad_top))
    return np.array(canvas)


def overlay_heatmap(rgb_np, heatmap, alpha=0.5):
    """RGB 이미지 위에 heatmap을 오버레이."""
    heatmap_color = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return (rgb_np * (1 - alpha) + heatmap_color * alpha).astype(np.uint8)


def select_test_images(class_names, target_classes):
    """seed42 test split에서 지정 클래스의 첫 이미지 선택."""
    split_path = os.path.join(
        ARTIFACTS_DIR, f"rgbd_noaux_{IMAGE_SIZE}_seed{SEED}", "split_info.json")
    with open(split_path) as f:
        split = json.load(f)

    result = {}
    for tc in target_classes:
        for p in split["test_paths"]:
            cls = os.path.basename(os.path.dirname(p))
            if cls == tc:
                result[tc] = p
                break
    return result


# ── Fig. 6: 4모델 Grad-CAM 비교 ─────────────────────────

def generate_fig6(device):
    """3 클래스 x (원본RGB + 4모델 heatmap) = 3x5 그리드."""
    target_classes = ["E25_door_LH_FRT", "E30_E38_door_RH", "E38_door_LH_RR"]
    model_types = ["rgbd", "texture_aug", "edge", "rgbe"]
    model_labels = ["Baseline\nRGBD", "Texture Aug\nRGBD",
                    "Edge-only", "RGBE\nHybrid"]

    with open(os.path.join(ARTIFACTS_DIR,
              f"rgbd_noaux_{IMAGE_SIZE}_seed{SEED}", "split_info.json")) as f:
        class_names = json.load(f)["class_names"]

    images = select_test_images(class_names, target_classes)

    cams = {}
    for mt in model_types:
        print(f"  Loading {mt} model...")
        cams[mt] = load_model(mt, device)

    fig, axes = plt.subplots(3, 5, figsize=(18, 11))

    for row, tc in enumerate(target_classes):
        rgb_path = images[tc]
        rgb_display = get_display_rgb(rgb_path)
        class_idx = class_names.index(tc)

        axes[row, 0].imshow(rgb_display)
        axes[row, 0].set_ylabel(tc.replace("_door_", "\n"),
                                fontsize=10, fontweight='bold')

        for col, mt in enumerate(model_types):
            img_t = preprocess_image(rgb_path, mt)
            pred = cams[mt].predict(img_t, device)
            heatmap = cams[mt].generate(img_t, class_idx, device)
            overlay = overlay_heatmap(rgb_display, heatmap)

            pred_label = class_names[pred]
            color = 'green' if pred == class_idx else 'red'

            axes[row, col + 1].imshow(overlay)
            axes[row, col + 1].set_title(
                f"pred: {pred_label.split('_')[-1]}",
                fontsize=8, color=color)

    for col, label in enumerate(["Original\nRGB"] + model_labels):
        axes[0, col].set_title(label, fontsize=11, fontweight='bold', pad=10)

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle("Fig. 6. Grad-CAM Comparison of Four Models (448×448, seed=42)",
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(OUTPUT_DIR, "gradcam_fig6.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ── Fig. 7: RGBD vs RGBE 강건성 비교 ────────────────────

def generate_fig7(device):
    """2 클래스 x (원본RGB, RGBD원본, RGBD증강, RGBE원본, RGBE증강) = 2x5."""
    target_classes = ["E25_door_LH_FRT", "E38_door_LH_RR"]
    model_pairs = [("rgbd", "Baseline RGBD"), ("rgbe", "RGBE Hybrid")]

    with open(os.path.join(ARTIFACTS_DIR,
              f"rgbd_noaux_{IMAGE_SIZE}_seed{SEED}", "split_info.json")) as f:
        class_names = json.load(f)["class_names"]

    images = select_test_images(class_names, target_classes)

    cam_models = {}
    for mt, _ in model_pairs:
        print(f"  Loading {mt} model...")
        cam_models[mt] = load_model(mt, device)

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))

    col_labels = ["Original\nRGB", "RGBD\n(Original)", "RGBD\n(Augmented)",
                  "RGBE\n(Original)", "RGBE\n(Augmented)"]

    for row, tc in enumerate(target_classes):
        rgb_path = images[tc]
        rgb_display = get_display_rgb(rgb_path)
        class_idx = class_names.index(tc)

        fname = os.path.basename(rgb_path)
        aug_path = os.path.join(DATASETS_AUG_DIR, tc, fname)
        if not os.path.exists(aug_path):
            print(f"  WARNING: {aug_path} not found, using original")
            aug_path = rgb_path
        aug_display = get_display_rgb(aug_path)

        axes[row, 0].imshow(rgb_display)
        axes[row, 0].set_ylabel(tc.replace("_door_", "\n"),
                                fontsize=10, fontweight='bold')

        for mi, (mt, label) in enumerate(model_pairs):
            orig_t = preprocess_image(rgb_path, mt)
            hm_orig = cam_models[mt].generate(orig_t, class_idx, device)
            axes[row, 1 + mi * 2].imshow(
                overlay_heatmap(rgb_display, hm_orig))

            aug_t = preprocess_image(aug_path, mt)
            hm_aug = cam_models[mt].generate(aug_t, class_idx, device)
            axes[row, 2 + mi * 2].imshow(
                overlay_heatmap(aug_display, hm_aug))

    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=11, fontweight='bold', pad=10)

    for ax in axes.flat:
        ax.axis('off')

    plt.suptitle(
        "Fig. 7. Grad-CAM Robustness: RGBD vs RGBE (Original vs Augmented)",
        fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(OUTPUT_DIR, "gradcam_fig7.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return out_path


# ── Main ─────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n[1/2] Generating Fig. 6 (4-model comparison)...")
    generate_fig6(device)

    print("\n[2/2] Generating Fig. 7 (RGBD vs RGBE robustness)...")
    generate_fig7(device)

    print("\nDone!")
