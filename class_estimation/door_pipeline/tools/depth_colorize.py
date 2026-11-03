"""16비트 depth PNG(mm) / background.npz 를 눈으로 볼 수 있는 컬러 PNG 로 (가까움=빨강 … 멀리=파랑, 무효=검정) + 색 눈금.

추론 서비스 세션의 depth_NNNN.png, 시드 reference_images/empty_jig_depth_small.png, 학습된 datasets_factory_infer/background.npz 는
값이 거리(mm) 라 일반 뷰어에서 검게 보인다. 서비스가 떠 있으면 웹 [배경 보기](/bg.jpg) 가 같은 색으로 배경·현재·차이를 보여 준다.

    python tools/depth_colorize.py reference_images/empty_jig_depth_small.png             # 옆에 *_color.png
    python tools/depth_colorize.py datasets_factory_infer/background.npz -o /tmp/bg.png
    python tools/depth_colorize.py a.png b.png --range 1300 1700                            # 색 범위(mm) 고정
    python tools/depth_colorize.py now.png --ref reference_images/empty_jig_depth_small.png # [기준|현재|차이] 3장 + ROI diff
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import depth_view  # noqa: E402


def load(p):
    p = Path(p)
    if p.suffix == '.npz':
        return np.load(p)['bg'].astype(np.float32)
    d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise SystemExit(f'읽기 실패: {p}')
    if d.ndim == 3:
        d = d[..., 0]
    return d.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('inputs', nargs='+', help='depth png 또는 background.npz')
    ap.add_argument('-o', '--out', help='출력 파일(입력 1개) 또는 폴더. 기본: 입력 옆 <이름>_color.png')
    ap.add_argument('--ref', help='기준 depth(png/npz). 주면 [기준|현재|차이] 3장으로 그리고 ROI diff 비율을 출력')
    ap.add_argument('--range', nargs=2, type=float, metavar=('VMIN', 'VMAX'), help='색 범위(mm) 고정. 기본 2~98 퍼센타일 ±50')
    ap.add_argument('--roi', default='0.12,0.05,0.88,0.95', help="노란 상자 ROI 비율 x0,y0,x1,y1 (서비스 기본값, '' 이면 없음)")
    ap.add_argument('--occ-roi', default='0.42,0.10,0.66,0.90', help="초록 상자 = 점유 판정 영역 OCC_ROI (diff 비율은 이 안에서, '' 이면 노란 ROI 안)")
    ap.add_argument('--bg-mm', type=float, default=25.0, help='차이 임계(mm) = 서비스 BG_MM')
    ap.add_argument('--scale', type=float, default=0, help='출력 배율. 0 이면 폭 800px 미만일 때 2배')
    a = ap.parse_args()
    roi = tuple(float(v) for v in a.roi.split(',')) if a.roi else None
    occ = tuple(float(v) for v in a.occ_roi.split(',')) if a.occ_roi else None
    vmin, vmax = (a.range if a.range else (None, None))
    ref = load(a.ref) if a.ref else None
    out_dir = Path(a.out) if a.out and (len(a.inputs) > 1 or Path(a.out).is_dir()) else None
    for inp in a.inputs:
        d = load(inp)
        if ref is not None:
            if d.shape != ref.shape:
                d = cv2.resize(d, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)
            img, frac = depth_view.render_compare(ref, d, a.bg_mm, roi, (f'REF {Path(a.ref).name}', f'NOW {Path(inp).name}', 'DIFF'), vmin, vmax, occ)
        else:
            img, frac = depth_view.render_single(d, vmin, vmax, roi, Path(inp).name, occ), None
        s = a.scale or (2.0 if img.shape[1] < 800 else 1.0)
        if s != 1.0:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f'{Path(inp).stem}_color.png'
        elif a.out:
            out = Path(a.out)
        else:
            out = Path(inp).with_name(f'{Path(inp).stem}_color.png')
        cv2.imwrite(str(out), img)
        print(f'{inp} -> {out}' + (f'   diff({"OCC_ROI" if occ else "ROI"})={frac:.3f}' if frac is not None else ''))


if __name__ == '__main__':
    main()
