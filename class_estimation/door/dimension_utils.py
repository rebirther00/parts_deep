"""부품 치수 측정 유틸리티

minAreaRect / PCA 기반 물리 치수 측정, 노이즈 저감 함수 모음.
99_dimension_measurement.py에서 사용.
"""

import collections
import os

import cv2
import numpy as np
import torch

# ZED X Mini 내부 파라미터 (HD1080 기본값)
ZED_INTRINSICS = {
    "fx": 1065.0, "fy": 1065.0,
    "cx": 960.0, "cy": 540.0,
    "width": 1920, "height": 1080,
}


def preprocess_depth(depth_mm: np.ndarray) -> np.ndarray:
    """Depth bilateral filter로 노이즈 저감."""
    valid = depth_mm > 0
    if valid.sum() < 100:
        return depth_mm
    depth_u16 = np.clip(depth_mm, 0, 65535).astype(np.uint16)
    filtered = cv2.bilateralFilter(depth_u16, d=5, sigmaColor=50, sigmaSpace=50)
    result = filtered.astype(np.float32)
    result[~valid] = 0
    return result


def refine_mask(mask: np.ndarray) -> np.ndarray:
    """Morphology 연산으로 마스크 경계 정리."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)
    return refined


def _pixel_to_3d(px, py, depth_mm, intrinsics, radius=5):
    """단일 픽셀을 3D 좌표로 변환. 주변 depth 중앙값 사용."""
    h, w = depth_mm.shape
    py_c = np.clip(py, 0, h - 1)
    px_c = np.clip(px, 0, w - 1)
    y0, y1 = max(0, py_c - radius), min(h, py_c + radius + 1)
    x0, x1 = max(0, px_c - radius), min(w, px_c + radius + 1)
    patch = depth_mm[y0:y1, x0:x1]
    valid = patch[patch > 0]
    if len(valid) == 0:
        return None
    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]
    z = float(np.median(valid))
    x3d = (px_c - cx) * z / fx
    y3d = (py_c - cy) * z / fy
    return np.array([x3d, y3d, z])


def measure_min_area_rect(mask: np.ndarray, depth_mm: np.ndarray,
                          intrinsics: dict) -> dict:
    """minAreaRect 기반 물리 치수 측정.

    마스크 외곽선 → 최소 면적 회전 사각형 → 꼭짓점 3D 변환 → 변 길이(mm).
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"width_mm": 0, "height_mm": 0, "ar": 0, "angle": 0}

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect).astype(np.int32)

    pts_3d = [_pixel_to_3d(int(p[0]), int(p[1]), depth_mm, intrinsics)
              for p in box]
    if any(p is None for p in pts_3d):
        return {"width_mm": 0, "height_mm": 0, "ar": 0, "angle": 0}

    side_a = float(np.linalg.norm(pts_3d[1] - pts_3d[0]))
    side_b = float(np.linalg.norm(pts_3d[2] - pts_3d[1]))

    phys_w = max(side_a, side_b)
    phys_h = min(side_a, side_b)
    ar = phys_w / max(phys_h, 1e-6)

    return {
        "width_mm": round(phys_w, 1),
        "height_mm": round(phys_h, 1),
        "ar": round(ar, 4),
        "angle": round(rect[2], 1),
        "box": box,
    }


def measure_pca(mask: np.ndarray, depth_mm: np.ndarray,
                intrinsics: dict) -> dict:
    """PCA 기반 물리 치수 측정 (아웃라이어 제거 포함)."""
    valid = (depth_mm > 0) & (mask > 0)
    if valid.sum() < 50:
        return {"width_mm": 0, "height_mm": 0, "ar": 0}

    rows, cols = np.where(valid)
    depths = depth_mm[valid].astype(np.float64)

    fx, fy = intrinsics["fx"], intrinsics["fy"]
    cx, cy = intrinsics["cx"], intrinsics["cy"]

    X = (cols.astype(np.float64) - cx) * depths / fx
    Y = (rows.astype(np.float64) - cy) * depths / fy
    Z = depths
    pts = np.stack([X, Y, Z], axis=1)
    centered = pts - pts.mean(axis=0)

    try:
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return {"width_mm": 0, "height_mm": 0, "ar": 0}

    proj = centered @ Vt[:2].T

    for axis in range(2):
        lo = np.percentile(proj[:, axis], 2)
        hi = np.percentile(proj[:, axis], 98)
        keep = (proj[:, axis] >= lo) & (proj[:, axis] <= hi)
        proj = proj[keep]

    if len(proj) < 10:
        return {"width_mm": 0, "height_mm": 0, "ar": 0}

    extent_0 = float(proj[:, 0].max() - proj[:, 0].min())
    extent_1 = float(proj[:, 1].max() - proj[:, 1].min())

    phys_w = max(extent_0, extent_1)
    phys_h = min(extent_0, extent_1)
    ar = phys_w / max(phys_h, 1e-6)

    return {
        "width_mm": round(phys_w, 1),
        "height_mm": round(phys_h, 1),
        "ar": round(ar, 4),
    }


class DimensionEngine:
    """SAM + 이중 측정 (minAreaRect/PCA) + 이동 평균."""

    MOBILE_SAM_PATH = None  # 외부에서 설정

    def __init__(self, sam_path: str, use_sam=True, avg_window=10):
        self.device = torch.device("cpu")
        self.sam_predictor = None
        if use_sam and os.path.exists(sam_path):
            try:
                self.sam_predictor = _load_mobile_sam(sam_path, str(self.device))
                print(f"MobileSAM 로드 완료 ({self.device})")
            except Exception as e:
                print(f"MobileSAM 로드 실패: {e}")

        self.history_rect = collections.deque(maxlen=avg_window)
        self.history_pca = collections.deque(maxlen=avg_window)
        self.intrinsics = ZED_INTRINSICS.copy()

    def update_intrinsics(self, fx, fy, cx, cy):
        self.intrinsics.update({"fx": fx, "fy": fy, "cx": cx, "cy": cy})

    def _generate_mask(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        if self.sam_predictor is None:
            return None
        self.sam_predictor.set_image(frame_rgb)
        h, w = frame_rgb.shape[:2]
        center = np.array([[w // 2, h // 2]], dtype=np.float32)
        label = np.array([1], dtype=np.int32)
        masks, scores, _ = self.sam_predictor.predict(
            point_coords=center, point_labels=label, multimask_output=True)
        best_idx = max(range(len(masks)), key=lambda i: masks[i].sum())
        return masks[best_idx].astype(np.uint8) * 255

    def measure(self, frame: np.ndarray,
                depth: np.ndarray | None) -> dict:
        """한 프레임에서 치수 측정."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        raw_mask = self._generate_mask(frame_rgb)
        if raw_mask is None:
            if depth is not None and depth.max() > 0:
                d8 = np.clip(depth / depth[depth > 0].max() * 255,
                             0, 255).astype(np.uint8)
                _, raw_mask = cv2.threshold(
                    d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            else:
                return self._empty_result()

        mask = refine_mask(raw_mask)
        if depth is None:
            return self._empty_result()

        depth_filtered = preprocess_depth(depth.astype(np.float32))

        rect_result = measure_min_area_rect(mask, depth_filtered, self.intrinsics)
        pca_result = measure_pca(mask, depth_filtered, self.intrinsics)

        if rect_result["width_mm"] > 0:
            self.history_rect.append(rect_result)
        if pca_result["width_mm"] > 0:
            self.history_pca.append(pca_result)

        return {
            "rect": rect_result,
            "pca": pca_result,
            "rect_avg": self._compute_avg(self.history_rect),
            "pca_avg": self._compute_avg(self.history_pca),
            "rect_stats": self._compute_stats(self.history_rect),
            "pca_stats": self._compute_stats(self.history_pca),
            "mask": mask,
            "sample_count": len(self.history_rect),
        }

    def reset_history(self):
        self.history_rect.clear()
        self.history_pca.clear()

    def _compute_avg(self, history):
        if not history:
            return {"width_mm": 0, "height_mm": 0, "ar": 0}
        ws = [h["width_mm"] for h in history]
        hs = [h["height_mm"] for h in history]
        ars = [h["ar"] for h in history]
        return {
            "width_mm": round(float(np.mean(ws)), 1),
            "height_mm": round(float(np.mean(hs)), 1),
            "ar": round(float(np.mean(ars)), 4),
        }

    def _compute_stats(self, history):
        if len(history) < 2:
            return {"w_std": 0, "h_std": 0, "ar_std": 0,
                    "w_min": 0, "w_max": 0, "h_min": 0, "h_max": 0}
        ws = [h["width_mm"] for h in history]
        hs = [h["height_mm"] for h in history]
        ars = [h["ar"] for h in history]
        return {
            "w_std": round(float(np.std(ws)), 1),
            "h_std": round(float(np.std(hs)), 1),
            "ar_std": round(float(np.std(ars)), 4),
            "w_min": round(min(ws), 1), "w_max": round(max(ws), 1),
            "h_min": round(min(hs), 1), "h_max": round(max(hs), 1),
        }

    def _empty_result(self):
        empty = {"width_mm": 0, "height_mm": 0, "ar": 0}
        return {
            "rect": empty.copy(), "pca": empty.copy(),
            "rect_avg": empty.copy(), "pca_avg": empty.copy(),
            "rect_stats": self._compute_stats([]),
            "pca_stats": self._compute_stats([]),
            "mask": None, "sample_count": 0,
        }


# ── 내부 헬퍼 ────────────────────────────────────────────


def _load_mobile_sam(checkpoint_path, device="cpu"):
    from mobile_sam import sam_model_registry, SamPredictor
    model = sam_model_registry["vit_t"](checkpoint=checkpoint_path)
    model.to(device).eval()
    return SamPredictor(model)
