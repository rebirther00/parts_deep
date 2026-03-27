"""SAM2 기반 전경 마스크 일괄 생성 스크립트

RGB 이미지에서 SAM2를 사용하여 도어(부품) 영역을 자동 세그멘테이션하고,
각 이미지에 대응하는 이진 마스크(mask_NNNN.png)를 저장한다.

사용법:
    python 05_generate_masks.py                          # 전체 (CAD + 실제)
    python 05_generate_masks.py --dataset cad            # CAD 데이터만
    python 05_generate_masks.py --dataset real           # 실제 데이터만
    python 05_generate_masks.py --preview 5              # 처음 5장 미리보기
"""

import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SAM2_MODEL_ID = "facebook/sam2.1-hiera-small"


def build_predictor(device="cuda"):
    """SAM2 이미지 예측기를 로드한다."""
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    predictor = SAM2ImagePredictor.from_pretrained(
        SAM2_MODEL_ID, device=device)
    return predictor


def generate_mask_for_image(predictor, rgb_path, device="cuda"):
    """단일 RGB 이미지에서 최대 면적 전경 마스크를 생성한다.

    이미지 중심점을 positive 프롬프트로 사용하여 도어 영역을 추출한다.
    SAM2가 반환하는 여러 마스크 중 최대 면적의 것을 선택한다.
    """
    img = cv2.imread(rgb_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    predictor.set_image(img_rgb)

    h, w = img_rgb.shape[:2]
    center_point = np.array([[w // 2, h // 2]], dtype=np.float32)
    center_label = np.array([1], dtype=np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=center_point,
        point_labels=center_label,
        multimask_output=True,
    )

    # 가장 큰 면적의 마스크 선택
    best_idx = max(range(len(masks)), key=lambda i: masks[i].sum())
    mask = masks[best_idx].astype(np.uint8) * 255

    return mask


def get_image_list(dataset_type):
    """지정된 데이터셋에서 RGB 이미지 경로 목록을 반환한다."""
    paths = []

    if dataset_type in ("cad", "all"):
        cad_dir = os.path.join(BASE_DIR, "datasets_cad")
        if os.path.isdir(cad_dir):
            paths.extend(sorted(glob.glob(
                os.path.join(cad_dir, "*", "rgb_*.png"))))

    if dataset_type in ("real", "all"):
        real_dir = os.path.join(BASE_DIR, "datasets")
        if os.path.isdir(real_dir):
            paths.extend(sorted(glob.glob(
                os.path.join(real_dir, "*", "rgb_*.png"))))

    return paths


def rgb_to_mask_path(rgb_path):
    """rgb_NNNN.png → mask_NNNN.png 경로 변환"""
    directory = os.path.dirname(rgb_path)
    filename = os.path.basename(rgb_path)
    mask_filename = filename.replace("rgb_", "mask_")
    return os.path.join(directory, mask_filename)


def main():
    parser = argparse.ArgumentParser(
        description="SAM2 기반 전경 마스크 일괄 생성")
    parser.add_argument(
        "--dataset", choices=["cad", "real", "all"], default="all",
        help="대상 데이터셋 (기본: all)")
    parser.add_argument(
        "--preview", type=int, default=0,
        help="미리보기 이미지 수 (0이면 미리보기 없이 전체 처리)")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="기존 마스크가 있어도 덮어쓰기")
    parser.add_argument(
        "--device", default="cuda",
        help="디바이스 (cuda/cpu)")
    args = parser.parse_args()

    image_paths = get_image_list(args.dataset)
    if not image_paths:
        print("[오류] 처리할 이미지가 없습니다.")
        sys.exit(1)

    if not args.overwrite:
        to_process = [p for p in image_paths
                      if not os.path.exists(rgb_to_mask_path(p))]
    else:
        to_process = image_paths

    total = len(to_process)
    if total == 0:
        print("모든 마스크가 이미 생성되어 있습니다. (--overwrite로 재생성 가능)")
        return

    if args.preview > 0:
        to_process = to_process[:args.preview]
        total = len(to_process)

    print(f"SAM2 모델 로딩 중... ({args.device})")
    predictor = build_predictor(args.device)
    print(f"모델 로드 완료. 처리 대상: {total}장\n")

    start_time = time.time()
    success = 0
    fail = 0

    for i, rgb_path in enumerate(to_process):
        mask_path = rgb_to_mask_path(rgb_path)
        rel_path = os.path.relpath(rgb_path, BASE_DIR)

        try:
            mask = generate_mask_for_image(predictor, rgb_path, args.device)
            if mask is not None:
                cv2.imwrite(mask_path, mask)
                success += 1
                fg_pct = mask.sum() / 255 / mask.size * 100
                if (i + 1) % 100 == 0 or i < 3 or (i + 1) == total:
                    elapsed = time.time() - start_time
                    eta = elapsed / (i + 1) * (total - i - 1)
                    print(f"  [{i+1:>5}/{total}] {rel_path} "
                          f"전경: {fg_pct:.1f}% "
                          f"({elapsed:.0f}s / ETA {eta:.0f}s)")
            else:
                fail += 1
                print(f"  [{i+1:>5}/{total}] {rel_path} [실패: 이미지 로드 불가]")
        except Exception as e:
            fail += 1
            print(f"  [{i+1:>5}/{total}] {rel_path} [오류: {e}]")

    elapsed = time.time() - start_time
    print(f"\n완료: 성공 {success}장, 실패 {fail}장, "
          f"소요 {elapsed:.1f}초 ({elapsed/max(success,1):.2f}초/장)")


if __name__ == "__main__":
    main()
