# door_pipeline — 도어 분류 정리본 파이프라인

`door/`(실험 이력)와 `door_paper/`(논문 실험)에서 **가장 결과가 좋은 방식만
추출·정리**한 폴더 (2026-07-08). 기존 폴더는 이력 보존용으로 그대로 두며,
신규 작업은 여기서 진행한다.

두 축으로 구성된다:

| 축 | 스크립트 | 강점 |
|---|---|---|
| A. CNN 분류 (RGBE NoAux 448) | `01`~`05` | 실험실 조건 최강 (test 100%) |
| B. 속성 파이프라인 (CAD+U-Net) | `10`~`13` + `attribute_utils.py` | 현장 강건 (현장 그룹 100%), 리비전 대응 |

검증 수치·원리·촬영 프로토콜은 `DOC_attribute_pipeline.md` 참조.
데이터셋은 심볼릭 링크로 `../door/` 원본을 참조한다 (복사본 아님).

## A. CNN 파이프라인 (RGBE NoAux 448)

```bash
cd ~/parts_deep/class_estimation/door_pipeline
source ~/parts_deep/venv/parts_deep/bin/activate
```

```bash
# ① 데이터 취득 (ZED 카메라, 웹 UI)
python 01_capture_dataset.py

# ② 학습 — train/val/test 70/15/15 자동 분할
python 02_train.py --model_type rgbe --no_aux --image_size 448 --seed 42
# → artifacts/rgbe_noaux_448_seed42/{model.pth, split_info.json, train_log.json}

# ③ 평가 — 기본 test셋만 (--eval_sets로 강건성 확인 추가 가능)
python 03_evaluate.py --model_type rgbe --no_aux --image_size 448 --seed 42
python 03_evaluate.py --model_type rgbe --no_aux --image_size 448 --seed 42 \
    --eval_sets aug aug2        # 강건성 확인 (기존 결과에 병합 저장)
# → artifacts/rgbe_noaux_448_seed42/eval_results.json

# ④ 현장 데이터 평가
python 04_evaluate_factory.py
# 옵션: --model <경로> --dataset_dir datasets_factory --image_size 448

# ⑤ 실시간 추론 서버 (ZED 2i 호환, 웹 UI :5001)
python 05_realtime_inference.py --port 5001
```

## B. 속성 파이프라인 (CAD 정합 + 통풍구 U-Net + 종횡비)

```bash
# 템플릿/스펙 생성 (신규 STL·리비전 추가 시에만)
python 10_generate_cad_templates.py

# 학습 라벨 자동 생성 (수작업 라벨링 불필요)
python 11_generate_vent_labels.py                                        # 원본
python 11_generate_vent_labels.py --base datasets_aug  --out vent_labels/datasets_aug   # 평가용
python 11_generate_vent_labels.py --base datasets_aug2 --out vent_labels/datasets_aug2  # 평가용

# U-Net 학습 (train/val/test 70/15/15; aug는 학습 금지 — 강건성 평가 전용)
python 12_train_vent_unet.py

# 평가 (datasets는 test 분할, aug/현장은 전체)
python 13_evaluate_attribute_pipeline.py
python 13_evaluate_attribute_pipeline.py --base datasets_factory
```

추론 API:

```python
from attribute_utils import frame_scores, decide, load_vent_unet, load_templates
net, templates = load_vent_unet(), load_templates()
frames = [frame_scores(rgb, depth_mm, mask, net, templates) for ...]  # 도어당 10프레임
pred_class, group, scores = decide(frames)
```

## 산출물/데이터 위치

- `artifacts/` — CNN 모델 (rgbe_noaux_448_seed42 이식됨; *.pth는 git 미추적)
- `attribute_models/` — 속성 파이프라인 (class_spec.json, cad_templates.npz, vent_unet.pth)
- `vent_labels/` — 자동 생성 라벨 (이미지 미추적, meta/split.json만 추적)
- `datasets*`, `sam_models` — `../door/` 심볼릭 링크
- `factory_masks/` — 현장 MobileSAM 마스크 캐시 (재생성 가능)

## 유의사항

- 평가 방법론: train/val/test 3분할, test 완전 격리, **datasets_aug는 학습
  금지**(강건성 확인 전용), 현장은 별도 평가.
- 향후 과제: 05 실시간 서버에 속성 파이프라인 통합(CNN과 하이브리드 —
  불일치 시 판정 보류), 현장 프로토콜 준수 재취득.
