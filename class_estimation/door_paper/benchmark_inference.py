"""RGBE Hybrid vs RGBD + Aux MLP 추론 속도 벤치마크

두 모델의 추론 속도를 정밀하게 비교한다.
- 모델 가중치 없이 (랜덤 초기화) 실행 가능 → 속도는 가중치 영향 없음
- CUDA / CPU 모두 지원
- 배치 크기별 throughput 측정
- 모델 forward만 / 전처리 포함 두 가지 모드
- Edge 디바이스에 그대로 옮겨서 동일한 명령으로 실행 가능

사용 예:
    # GPU에서 모델 forward만 측정
    python benchmark_inference.py --device cuda --num_runs 1000

    # CPU에서 전처리 포함 측정
    python benchmark_inference.py --device cpu --include_preprocess

    # 배치 크기별 throughput
    python benchmark_inference.py --batch_sizes 1,4,8,16
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from depth_utils import RGBDAuxResNet18, RGBDTransform, NUM_AUX_FEATURES, IN_CHANNELS
from rgbe_utils import RGBETransform, RGBE_IN_CHANNELS


# ── RGBE용 NoAuxResNet18 (train_paper.py와 동일 구조) ────
# train_paper.py 모듈 레벨 argparse 회피를 위해 여기에 직접 정의.
# 구조가 변경되면 양쪽을 함께 갱신해야 한다.
class NoAuxResNet18(nn.Module):
    """Aux MLP 없이 이미지만 사용하는 분류 모델 (RGBE Hybrid 베이스)."""

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

    def forward(self, images, aux_features=None):
        img_feat = self.backbone(images)
        return self.classifier(img_feat)

# ── 기본 설정 ─────────────────────────────────────────────
DEFAULT_NUM_CLASSES = 8
DEFAULT_IMAGE_SIZE = 448
DEFAULT_WARMUP = 50
DEFAULT_NUM_RUNS = 500


def parse_args():
    parser = argparse.ArgumentParser(
        description='RGBE Hybrid vs RGBD + Aux MLP 추론 속도 벤치마크')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='추론 디바이스 (default: auto)')
    parser.add_argument('--image_size', type=int, default=DEFAULT_IMAGE_SIZE,
                        help='입력 이미지 크기 (default: 448)')
    parser.add_argument('--num_classes', type=int, default=DEFAULT_NUM_CLASSES,
                        help='클래스 수 (default: 8)')
    parser.add_argument('--num_runs', type=int, default=DEFAULT_NUM_RUNS,
                        help='측정 반복 횟수 (default: 500)')
    parser.add_argument('--warmup', type=int, default=DEFAULT_WARMUP,
                        help='워밍업 횟수 (default: 50)')
    parser.add_argument('--batch_sizes', type=str, default='1',
                        help='쉼표 구분 배치 크기 목록 (default: "1")')
    parser.add_argument('--include_preprocess', action='store_true',
                        help='전처리 시간 포함하여 측정')
    parser.add_argument('--rgbe_model_path', type=str, default=None,
                        help='RGBE 모델 가중치 경로 (선택, 속도 측정에 영향 없음)')
    parser.add_argument('--rgbd_model_path', type=str, default=None,
                        help='RGBD 모델 가중치 경로 (선택, 속도 측정에 영향 없음)')
    return parser.parse_args()


# ── 디바이스 결정 ────────────────────────────────────────
def resolve_device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


# ── 모델 로드 ────────────────────────────────────────────
def build_models(num_classes, device, rgbe_ckpt=None, rgbd_ckpt=None):
    """두 모델을 생성하고 eval 모드로 디바이스에 올린다.

    가중치 파일은 선택적이다. 없으면 랜덤 초기화로 동작한다
    (추론 속도는 weight 값에 영향받지 않으므로 측정에 문제 없음).
    """
    rgbe_model = NoAuxResNet18(
        num_classes, in_channels=RGBE_IN_CHANNELS, pretrained=False)
    rgbd_model = RGBDAuxResNet18(num_classes, pretrained=False)

    if rgbe_ckpt and os.path.exists(rgbe_ckpt):
        rgbe_model.load_state_dict(torch.load(rgbe_ckpt, map_location='cpu'))
        print(f"  RGBE 가중치 로드: {rgbe_ckpt}")
    else:
        print("  RGBE 가중치: 랜덤 초기화 (속도 측정에는 영향 없음)")

    if rgbd_ckpt and os.path.exists(rgbd_ckpt):
        rgbd_model.load_state_dict(torch.load(rgbd_ckpt, map_location='cpu'))
        print(f"  RGBD 가중치 로드: {rgbd_ckpt}")
    else:
        print("  RGBD 가중치: 랜덤 초기화 (속도 측정에는 영향 없음)")

    rgbe_model.to(device).eval()
    rgbd_model.to(device).eval()
    return rgbe_model, rgbd_model


# ── 파라미터 수 측정 ─────────────────────────────────────
def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ── 시간 측정 유틸 ───────────────────────────────────────
def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def now():
    return time.perf_counter()


def percentile_stats(times_ms):
    arr = np.array(times_ms)
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        'p50': float(np.percentile(arr, 50)),
        'p95': float(np.percentile(arr, 95)),
        'p99': float(np.percentile(arr, 99)),
        'min': float(arr.min()),
        'max': float(arr.max()),
    }


# ── 모델 forward 벤치마크 ────────────────────────────────
def benchmark_model_forward(model, image_input, aux_input, device,
                            warmup, num_runs):
    """모델 forward만 측정 (전처리/후처리 제외)."""
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(image_input, aux_input)
        sync(device)

        times_ms = []
        for _ in range(num_runs):
            sync(device)
            t0 = now()
            _ = model(image_input, aux_input)
            sync(device)
            times_ms.append((now() - t0) * 1000)

    return percentile_stats(times_ms)


# ── 전처리 벤치마크 ──────────────────────────────────────
def benchmark_rgbe_preprocess(image_size, warmup, num_runs):
    """RGBE 전처리: RGB ndarray → Canny → 4ch tensor."""
    from PIL import Image
    transform = RGBETransform(image_size, is_train=False)

    rng = np.random.default_rng(0)
    rgb_np = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    pil_img = Image.fromarray(rgb_np)

    for _ in range(warmup):
        _ = transform(pil_img)

    times_ms = []
    for _ in range(num_runs):
        t0 = now()
        _ = transform(pil_img)
        times_ms.append((now() - t0) * 1000)

    return percentile_stats(times_ms)


def benchmark_rgbd_preprocess(image_size, warmup, num_runs):
    """RGBD 전처리: RGB + Depth → letterbox 4ch tensor (Aux 제외).

    Aux 피처 계산(SVD)은 별도 측정.
    """
    from PIL import Image
    transform = RGBDTransform(image_size, is_train=False)

    rng = np.random.default_rng(0)
    rgb_np = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    depth_np = rng.random((1080, 1920)).astype(np.float32)
    pil_img = Image.fromarray(rgb_np)

    for _ in range(warmup):
        _ = transform(pil_img, depth_np)

    times_ms = []
    for _ in range(num_runs):
        t0 = now()
        _ = transform(pil_img, depth_np)
        times_ms.append((now() - t0) * 1000)

    return percentile_stats(times_ms)


def benchmark_aux_compute(warmup, num_runs):
    """RGBD의 Aux 피처 계산 (depth → width/height/aspect, SVD 포함)."""
    from depth_utils import compute_aux_features

    rng = np.random.default_rng(0)
    depth_mm = (rng.random((1080, 1920)) * 3000 + 500).astype(np.float32)

    for _ in range(warmup):
        _ = compute_aux_features(depth_mm)

    times_ms = []
    for _ in range(num_runs):
        t0 = now()
        _ = compute_aux_features(depth_mm)
        times_ms.append((now() - t0) * 1000)

    return percentile_stats(times_ms)


# ── 결과 출력 ────────────────────────────────────────────
def print_stats(label, stats, batch_size=1):
    fps = (1000.0 * batch_size) / stats['mean'] if stats['mean'] > 0 else 0
    print(f"  {label:<25s} "
          f"mean={stats['mean']:7.3f}ms ± {stats['std']:5.3f}  "
          f"p50={stats['p50']:7.3f}  p95={stats['p95']:7.3f}  "
          f"p99={stats['p99']:7.3f}  → {fps:7.1f} FPS")


def print_section(title):
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print('=' * 78)


# ── 메인 ─────────────────────────────────────────────────
def main():
    args = parse_args()
    device = resolve_device(args.device)
    batch_sizes = [int(b) for b in args.batch_sizes.split(',')]

    print_section("환경 정보")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  Device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")
        torch.backends.cudnn.benchmark = True
    print(f"  이미지 크기: {args.image_size}x{args.image_size}")
    print(f"  반복: warmup={args.warmup}, runs={args.num_runs}")

    print_section("모델 로드")
    rgbe_model, rgbd_model = build_models(
        args.num_classes, device,
        args.rgbe_model_path, args.rgbd_model_path)

    rgbe_params = count_params(rgbe_model)
    rgbd_params = count_params(rgbd_model)
    print(f"\n  RGBE Hybrid 파라미터:    {rgbe_params:>12,}")
    print(f"  RGBD + Aux MLP 파라미터: {rgbd_params:>12,}")
    print(f"  차이: {rgbd_params - rgbe_params:+,} "
          f"({100*(rgbd_params - rgbe_params)/rgbe_params:+.3f}%)")

    print_section("모델 Forward 추론 속도 (배치별)")

    for bs in batch_sizes:
        print(f"\n[배치 크기 = {bs}]")
        image_input = torch.randn(
            bs, 4, args.image_size, args.image_size, device=device)
        aux_input = torch.randn(bs, NUM_AUX_FEATURES, device=device)

        rgbe_stats = benchmark_model_forward(
            rgbe_model, image_input, aux_input, device,
            args.warmup, args.num_runs)
        rgbd_stats = benchmark_model_forward(
            rgbd_model, image_input, aux_input, device,
            args.warmup, args.num_runs)

        print_stats("RGBE Hybrid", rgbe_stats, bs)
        print_stats("RGBD + Aux MLP", rgbd_stats, bs)

        diff_ms = rgbd_stats['mean'] - rgbe_stats['mean']
        diff_pct = 100 * diff_ms / rgbe_stats['mean'] \
            if rgbe_stats['mean'] > 0 else 0
        print(f"  → 차이: {diff_ms:+.3f}ms ({diff_pct:+.2f}%)")

    if args.include_preprocess:
        print_section("CPU 전처리 속도 (단일 이미지, 1920×1080 입력 가정)")

        rgbe_pre = benchmark_rgbe_preprocess(
            args.image_size, args.warmup, args.num_runs)
        rgbd_pre = benchmark_rgbd_preprocess(
            args.image_size, args.warmup, args.num_runs)
        aux_pre = benchmark_aux_compute(args.warmup, args.num_runs // 2)

        print_stats("RGBE 전처리 (Canny 포함)", rgbe_pre)
        print_stats("RGBD 전처리 (Resize만)", rgbd_pre)
        print_stats("RGBD Aux 계산 (SVD)", aux_pre)

        rgbe_total = rgbe_pre['mean']
        rgbd_total = rgbd_pre['mean'] + aux_pre['mean']
        print(f"\n  RGBE 전처리 합계:  {rgbe_total:7.3f}ms")
        print(f"  RGBD 전처리 합계:  {rgbd_total:7.3f}ms "
              f"(Resize + Aux)")

    print_section("측정 완료")
    print()


if __name__ == '__main__':
    main()
