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
# → attribute_models/runs/vent_unet_seed<시드>/{model.pth, train_info.json}
#   + 배포 포인터 attribute_models/vent_unet.pth 자동 갱신 (--no_promote로 억제)
#   시드 미지정 시 랜덤 (train_info.json에 기록)

# 평가 (datasets는 test 분할, aug/현장은 전체)
python 13_evaluate_attribute_pipeline.py
python 13_evaluate_attribute_pipeline.py --base datasets_factory

# 실시간 추론 서버 (ZED 필수 — depth 사용, 웹 UI :5003)
python 14_realtime_inference_attribute.py
python 14_realtime_inference_attribute.py --replay datasets_factory/E30_E38_door_RH  # 카메라 없이 검증
```

추론 API:

```python
from attribute_utils import frame_scores, decide, load_vent_unet, load_templates
net, templates = load_vent_unet(), load_templates()
frames = [frame_scores(rgb, depth_mm, mask, net, templates) for ...]  # 도어당 10프레임
pred_class, group, scores = decide(frames)
```


## C. 홀 랜드마크 판별기 (2026-08-26, 1순위 판정)

모서리 프레임 홀 2개(래치 각공 쪽 변, 실제 장착 기준 바닥 쪽)의 거리 D(=도어 폭−106mm)로 8종을 판정한다. CNN이 홀 6점(래치 볼트홀 4 + 모서리 홀 2)을
찾고, depth 평면상 거리 × K_DEPTH(카메라 모드별 캘리브레이션) → CAD D 최근접. 속성 파이프라인(통풍구 U-Net)의
그룹 판별을 제약으로 써서 그룹 내 최근접을 취하고, 볼트 프레임 기하 게이트에 걸리면 보류한다.

실행 순서 (A/B와 같은 번호 순, 래퍼는 `scripts/hole_*.sh`):

```bash
# ⑮ 라벨링 (브라우저 :8090) → labels/holes/*.json          [scripts/hole_label.sh]
python 15_label_holes.py --per-class 15 --extra datasets_field
# ⑯ 학습 → attribute_models/hole_landmarks/model.pth (+DB)  [scripts/hole_train.sh]
python 16_train_hole_landmarks.py
# ⑰ 평가 → eval_classifier.json, report/hole_analysis/samples/ (+DB)  [scripts/hole_evaluate.sh]
python 17_evaluate_hole_classifier.py                 # test 분할 · datasets 전체 · datasets_field
python 17_evaluate_hole_classifier.py --base datasets_field
python tools/make_hole_samples.py                     # 성공/오판/보류 샘플 + 학습 곡선
# ⑭ 실시간 추론 (홀 1순위 + 속성 폴백, 웹 UI :5003)        [scripts/hole_inference.sh]
python 14_realtime_inference_attribute.py --replay datasets_field/E25_door_RH_s_091317
python scripts/hole_classify_image.py rgb_0003.png    # 단일 이미지
```

- 평가(라벨 학습분 제외): test 분할 134/189 판정·**100%**, datasets 전체 814/1162 판정·**99.0%**(그룹 제약 시 99.5%),
  현장 16/16·**100%**. 보류(~30%)는 힌지측 홀이 프레임 밖/경계 근접 — 사무실 촬영 프레이밍 문제, 현장은 0%.
- 오판은 검출이 아니라 **거리 측정 오차**(depth 평면, 20~58mm)에서 나온다. 그룹 제약이 RH↔RR 혼동을 막고, FRT 내(41mm 간격)는 N프레임 집계·intrinsics 정밀화로 대응.
- 중앙 보강대 패드 홀은 옵션/리비전에 따라 달라 사용 금지. 분석 기록: `report/hole_analysis/`.
- API `/api/inference_result`에 `source`(hole/attr/conflict)와 `hole{pred, D_mm, n_judged, gate, group_constraint, points}`.

## 산출물/데이터 위치

- `artifacts/` — CNN 모델 (rgbe_noaux_448_seed42 이식됨; *.pth는 git 미추적)
- `attribute_models/` — 속성 파이프라인 스펙·모델
  - `runs/vent_unet_seed<시드>/` — 학습 run별 보관 (레거시 artifacts 방식)
  - `vent_unet.pth` — 배포 포인터 (attribute_utils·13번 기본 참조, 최신 run 복사본)
- `vent_labels/` — 자동 생성 라벨 (이미지 미추적, meta/split.json만 추적)
- `datasets*`, `sam_models` — `../door/` 심볼릭 링크
- `factory_masks/` — 현장 MobileSAM 마스크 캐시 (재생성 가능)
- `datasets_factory_collect/` — NAS 미러(세션당 20장 샘플, `db/pull_nas.py`), `datasets_factory_v2/` — 학습·평가용 링크 뷰(`db/build_dataset.py build`). 흐름은 `db/README.md` 참조

## 유의사항

- 평가 방법론: train/val/test 3분할, test 완전 격리, **datasets_aug는 학습
  금지**(강건성 확인 전용), 현장은 별도 평가.
- 향후 과제: 05 실시간 서버에 속성 파이프라인 통합(CNN과 하이브리드 —
  불일치 시 판정 보류), 현장 프로토콜 준수 재취득.
