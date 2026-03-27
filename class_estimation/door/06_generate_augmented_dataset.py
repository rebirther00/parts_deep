"""
극단적 RGB 증강 데이터셋 생성 스크립트
- datasets/의 RGB 이미지에 극단적 증강 적용 → datasets_aug/ 생성
- Depth, Mask는 원본 그대로 복사
- 목적: 모델이 형상(depth/shape) vs 표면 패턴(RGB texture) 중
  무엇을 학습했는지 검증 (일반화 vs 암기)
"""
import argparse
import glob
import os
import random
import shutil

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# 증강 강도 프리셋
PRESETS = {
    "mild": {
        "brightness": (0.5, 1.5),
        "contrast": (0.5, 1.5),
        "saturation": (0.3, 1.7),
        "hue_shift": 30,
        "noise_sigma": (10, 25),
        "blur_kernel": (3, 7),
    },
    "moderate": {
        "brightness": (0.5, 1.8),
        "contrast": (0.5, 1.8),
        "saturation": (0.2, 2.0),
        "hue_shift": 40,
        "noise_sigma": (15, 30),
        "blur_kernel": (3, 9),
    },
    "extreme": {
        "brightness": (0.2, 3.0),
        "contrast": (0.2, 3.0),
        "saturation": (0.0, 3.0),
        "hue_shift": 90,
        "noise_sigma": (30, 60),
        "blur_kernel": (7, 15),
    },
}


def augment_rgb(img: Image.Image, preset: dict, rng: random.Random) -> Image.Image:
    """RGB 이미지에 극단적 증강을 순차 적용"""
    # 1) Brightness
    lo, hi = preset["brightness"]
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(lo, hi))

    # 2) Contrast
    lo, hi = preset["contrast"]
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(lo, hi))

    # 3) Saturation
    lo, hi = preset["saturation"]
    img = ImageEnhance.Color(img).enhance(rng.uniform(lo, hi))

    # 4) Hue shift (OpenCV HSV 공간에서)
    arr = np.array(img)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.int16)
    shift = rng.randint(-preset["hue_shift"], preset["hue_shift"])
    hsv[:, :, 0] = (hsv[:, :, 0] + shift) % 180
    hsv = hsv.astype(np.uint8)
    arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    img = Image.fromarray(arr)

    # 5) Gaussian Noise
    lo, hi = preset["noise_sigma"]
    sigma = rng.uniform(lo, hi)
    arr = np.array(img).astype(np.float32)
    noise = np.random.RandomState(rng.randint(0, 2**31)).normal(0, sigma, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # 6) Gaussian Blur
    lo, hi = preset["blur_kernel"]
    k = rng.choice(range(lo, hi + 1, 2))  # 홀수만
    img = img.filter(ImageFilter.GaussianBlur(radius=k // 2))

    return img


def apply_mask_blend(original: Image.Image, augmented: Image.Image,
                     mask_path: str) -> Image.Image:
    """전경(mask=255)은 증강 이미지, 배경은 원본 이미지로 합성"""
    if not os.path.exists(mask_path):
        return augmented
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return augmented
    mask_f = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
    orig_arr = np.array(original).astype(np.float32)
    aug_arr = np.array(augmented).astype(np.float32)
    blended = orig_arr * (1.0 - mask_f) + aug_arr * mask_f
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def process_class(src_dir, dst_dir, preset, seed, mask_only=False):
    """한 클래스 폴더의 이미지들을 증강 처리"""
    os.makedirs(dst_dir, exist_ok=True)
    rgb_files = sorted(glob.glob(os.path.join(src_dir, "rgb_*.png")))
    count = 0

    for rgb_path in rgb_files:
        basename = os.path.basename(rgb_path)
        stem = basename.replace("rgb_", "").replace(".png", "")
        per_image_seed = seed + hash(basename) % (2**31)
        rng = random.Random(per_image_seed)

        img = Image.open(rgb_path).convert("RGB")
        aug = augment_rgb(img, preset, rng)

        if mask_only:
            mask_path = os.path.join(src_dir, f"mask_{stem}.png")
            aug = apply_mask_blend(img, aug, mask_path)

        aug.save(os.path.join(dst_dir, basename))

        # Depth: 원본 그대로 복사
        depth_name = f"depth_{stem}.png"
        depth_src = os.path.join(src_dir, depth_name)
        if os.path.exists(depth_src):
            shutil.copy2(depth_src, os.path.join(dst_dir, depth_name))

        # Mask: 원본 그대로 복사
        mask_name = f"mask_{stem}.png"
        mask_src = os.path.join(src_dir, mask_name)
        if os.path.exists(mask_src):
            shutil.copy2(mask_src, os.path.join(dst_dir, mask_name))

        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="극단적 RGB 증강 데이터셋 생성")
    parser.add_argument("--src", type=str, default="datasets",
                        help="원본 데이터셋 경로 (기본: datasets)")
    parser.add_argument("--dst", type=str, default="datasets_aug",
                        help="출력 데이터셋 경로 (기본: datasets_aug)")
    parser.add_argument("--level", type=str, default="extreme",
                        choices=["mild", "moderate", "extreme"],
                        help="증강 강도 (기본: extreme)")
    parser.add_argument("--seed", type=int, default=42,
                        help="재현성을 위한 랜덤 시드 (기본: 42)")
    parser.add_argument("--mask_only", action="store_true",
                        help="마스크 영역(전경)만 증강, 배경은 원본 유지")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_root = os.path.join(script_dir, args.src)
    dst_root = os.path.join(script_dir, args.dst)

    if not os.path.isdir(src_root):
        raise FileNotFoundError(f"원본 데이터셋을 찾을 수 없습니다: {src_root}")

    preset = PRESETS[args.level]
    print(f"증강 레벨: {args.level}")
    print(f"마스크 전용 증강: {args.mask_only}")
    print(f"입력: {src_root}")
    print(f"출력: {dst_root}")
    print(f"시드: {args.seed}")
    print(f"증강 파라미터: {preset}")
    print("-" * 60)

    # 클래스 폴더 스캔 (rgb_*.png가 있는 하위 폴더만)
    class_dirs = sorted([
        d for d in os.listdir(src_root)
        if os.path.isdir(os.path.join(src_root, d))
        and glob.glob(os.path.join(src_root, d, "rgb_*.png"))
    ])

    if not class_dirs:
        raise FileNotFoundError(f"rgb_*.png가 있는 클래스 폴더를 찾지 못했습니다: {src_root}")

    total = 0
    for cls_name in class_dirs:
        src_cls = os.path.join(src_root, cls_name)
        dst_cls = os.path.join(dst_root, cls_name)
        n = process_class(src_cls, dst_cls, preset, args.seed, args.mask_only)
        total += n
        print(f"  {cls_name}: {n}장 증강 완료")

    print("-" * 60)
    print(f"총 {total}장 증강 데이터셋 생성 완료 → {dst_root}")


if __name__ == "__main__":
    main()
