"""depth(mm) 맵을 눈으로 볼 수 있는 컬러 그림으로 — 서비스 웹 [배경 보기](/bg.jpg) 와 tools/depth_colorize.py 공용.

16비트 depth PNG(세션 depth_NNNN.png, 시드 reference_images/empty_jig_depth_small.png) 와 background.npz 는 값이 거리(mm) 라
일반 뷰어에서 검게 보인다. 색: 가까움=빨강 … 멀리=파랑, 0/무효=검정.
차이 그림: 빨강=기준보다 가까움(도어·돌출), 파랑=멀어짐, 회색=같음(±bg_mm), 검정=측정 불가.
"""
import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


def auto_range(d, pad=50.0, roi=None):
    """유효 픽셀 2~98 퍼센타일 ± pad (mm). roi 가 있으면 그 안에서만 — 지그 바깥 바닥을 빼야 지그 판 위 요철이 보인다."""
    if roi is not None:
        x0, y0, x1, y1 = roi_rect(d.shape, roi)
        d = d[y0:y1, x0:x1]
    v = d[d > 0]
    if v.size == 0:
        return 1000.0, 2000.0
    lo, hi = (float(x) for x in np.percentile(v, [2, 98]))
    return lo - pad, hi + pad


def depth_to_color(d, vmin, vmax):
    d = d.astype(np.float32)
    v = np.clip((d - vmin) / max(1.0, vmax - vmin), 0, 1)
    img = cv2.applyColorMap((255 - v * 255).astype(np.uint8), cv2.COLORMAP_JET)   # 가까움=빨강, 멀리=파랑
    img[d <= 0] = 0
    return img


def roi_rect(shape, roi):
    """roi=(x0,y0,x1,y1) 비율 → 픽셀 (x0,y0,x1,y1). 서비스 SceneGates._roi 와 같은 절단."""
    h, w = shape[:2]
    x0, y0, x1, y1 = roi
    return int(w * x0), int(h * y0), int(w * x1), int(h * y1)


def diff_image(ref, now, bg_mm):
    img = np.zeros(ref.shape + (3,), np.uint8)
    both = (ref > 0) & (now > 0)
    delta = now.astype(np.float32) - ref.astype(np.float32)
    img[both] = (90, 90, 90)
    img[both & (delta < -bg_mm)] = (40, 40, 220)      # 기준보다 가까움 → 빨강
    img[both & (delta > bg_mm)] = (220, 120, 40)      # 멀어짐 → 파랑
    return img


def diff_frac(ref, now, bg_mm, roi=None):
    """ROI 안에서 |now-ref| > bg_mm 인 유효 픽셀 비율 — 서비스 diff 지표와 같은 정의."""
    both = (ref > 0) & (now > 0)
    delta = np.abs(now.astype(np.float32) - ref.astype(np.float32))
    if roi is not None:
        x0, y0, x1, y1 = roi_rect(ref.shape, roi)
        both, delta = both[y0:y1, x0:x1], delta[y0:y1, x0:x1]
    n = int(both.sum())
    return float((delta > bg_mm)[both].sum() / n) if n else None


def _label(img, text, y=16):
    cv2.putText(img, text, (6, y), FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, y), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def _roi_box(img, roi, color=(0, 255, 255), thick=1):
    if roi is not None:
        x0, y0, x1, y1 = roi_rect(img.shape, roi)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, thick)


def colorbar(width, vmin, vmax, legend=None):
    bar = np.linspace(vmin, vmax, width, dtype=np.float32)[None, :].repeat(14, 0)
    strip = np.zeros((34, width, 3), np.uint8)
    strip[4:18] = depth_to_color(bar, vmin, vmax)
    cv2.putText(strip, f"{vmin:.0f}mm near", (4, 31), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(strip, f"{vmax:.0f}mm far", (width - 112, 31), FONT, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    if legend:
        cv2.putText(strip, legend, (max(130, width // 2 - 4 * len(legend)), 31), FONT, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    return strip


def render_single(d, vmin=None, vmax=None, roi=None, title=None, occ_roi=None):
    """depth 한 장 → 컬러 + 색 눈금. 노랑 = 도어 ROI, 초록 = 점유 판정 영역(OCC_ROI)."""
    if vmin is None:
        vmin, vmax = auto_range(d, roi=roi)
    img = depth_to_color(d, vmin, vmax)
    _roi_box(img, roi)
    _roi_box(img, occ_roi, (60, 220, 60), 2)
    if title:
        _label(img, title)
    return np.vstack([img, colorbar(img.shape[1], vmin, vmax)])


def render_compare(ref, now, bg_mm=25.0, roi=None, titles=('REF', 'NOW', 'DIFF'), vmin=None, vmax=None, occ_roi=None):
    """[기준 | 현재 | 차이] 3장 + 색 눈금. 한쪽이 없으면 빈 패널. 노랑 = 도어 ROI, 초록 = 점유 판정 영역(OCC_ROI).
    반환 (BGR 이미지, diff 비율 또는 None) — diff 는 occ_roi 가 있으면 그 안에서(서비스 점유 판정과 같음), 없으면 roi 안에서."""
    base = ref if ref is not None else now
    h, w = base.shape[:2]
    if vmin is None:
        vmin, vmax = auto_range(base, roi=roi)
    blank = np.zeros((h, w, 3), np.uint8)
    ok_now = now is not None and now.shape[:2] == (h, w)
    p_ref = depth_to_color(ref, vmin, vmax) if ref is not None else blank.copy()
    p_now = depth_to_color(now, vmin, vmax) if ok_now else blank.copy()
    frac = None
    if ref is not None and ok_now:
        p_diff = diff_image(ref, now, bg_mm)
        frac = diff_frac(ref, now, bg_mm, occ_roi or roi)
    else:
        p_diff = blank.copy()
    t = list(titles)
    t[2] = (f"{t[2]} {frac:.3f}" if frac is not None else t[2]) + f" (|d|>{bg_mm:.0f}mm)"
    panels = []
    for p, title in zip((p_ref, p_now, p_diff), t):
        _roi_box(p, roi)
        _roi_box(p, occ_roi, (60, 220, 60), 2)
        _label(p, title)
        panels.append(p)
    gap = np.full((h, 6, 3), 30, np.uint8)
    top = np.hstack([panels[0], gap, panels[1], gap, panels[2]])
    legend = 'DIFF: red = closer than ref   blue = farther   gray = same   black = no depth'
    return np.vstack([top, colorbar(top.shape[1], vmin, vmax, legend)]), frac
