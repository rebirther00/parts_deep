# archive_attribute_unet_20260911 — 속성 파이프라인(CAD 정합 + 통풍구 U-Net + 종횡비) 아카이브

**2026-09-11 폐기·아카이빙** (사용자 결정). door_pipeline B축 전체를 이곳으로 이동했다. 원리·검증 수치·촬영 프로토콜은
`DOC_attribute_pipeline.md` (이동 당시 그대로) 참조.

## 폐기 사유

- 1순위 판정기인 홀 랜드마크 판별기(`../hole_classifier.py`, 15~19)가 단독으로 충분: datasets 전체 99.0%, 현장 100%.
  U-Net 그룹 판별을 제약으로 붙였을 때 이득은 datasets +0.5%p(99.5%)뿐이고 현장은 동일(100%).
- 홀 보류 시 폴백 역할은 2026-09-07 현장 재학습 CNN(정식 run DB #8, 세션 CV 97.4%)으로 결정되어 U-Net의 남은 역할이 없음.
- 실시간은 홀 전용 서버 18(모델 1개, 로드 0.5 s)이 통합 서버 14(MobileSAM+U-Net+홀)를 대체. 발표 자료에서도 이미 제외.

## 최종 수치 (U-Net v3, `attribute_models/vent_unet.pth`, 2026-07-09)

| | class% / group% |
|---|---|
| datasets test | 99.2 / 100 |
| datasets_aug | 95.5 / 97.7 |
| datasets_aug2 | 93.6 / 95.5 |
| 현장(datasets_factory, 2클래스) | 21.6 / **100** |

## 내용물

| 항목 | 설명 |
|---|---|
| `10_generate_cad_templates.py` | STL → `attribute_models/cad_templates.npz`·`class_spec.json` (E23 포함 9종, 2026-09-07 재생성) |
| `11_generate_vent_labels.py` | CAD 슬롯 자동 라벨 → `vent_labels/` (이미지는 삭제, `meta.json`·`split.json`만 보존 — 재실행으로 복원) |
| `12_train_vent_unet.py` | U-Net 학습 (DB models/training_sessions 기록 — DB 기록은 유지) |
| `13_evaluate_attribute_pipeline.py` | 평가 → `attribute_models/eval_attr_*.{json,png}` |
| `14_realtime_inference_attribute.py` + `scripts/hole_inference.sh` + `templates/inference_attr.html` | 통합 실시간 서버(:5003, 홀 1순위 + 속성 폴백) |
| `attribute_utils.py` | 렉티파이/축정렬/VentUNet/템플릿 매칭/N프레임 판정 |
| `attribute_models/` | `vent_unet.pth`(Gitea LFS), `runs/vent_unet_seed{42,29186}/`, 템플릿, 평가 산출물 |
| `factory_masks/` | 현장 MobileSAM 마스크 캐시 |
| `datasets*`, `sam_models` | 원본과 같은 대상의 심볼릭 링크 (아카이브 스크립트 실행용) |

## 남은 사용처

- `../report/hole_analysis/scripts/hole_measure2.py`, `hole_measure3.py` (8월 분석용)가 `attribute_utils.axis_align/load_templates`와
  `vent_labels/datasets/meta.json`을 참조 → 이 폴더 경로로 수정해 둠.
- `../hole_classifier.py`의 `group=` 인자(그룹 제약)는 남아 있으나 호출부 없음.
- `../dimension_utils.py`(MobileSAM 로더)는 door_pipeline에 남겨 둠 (14 외 사용처 없음).

## 다시 실행하려면

```bash
cd ~/parts_deep/class_estimation/door_pipeline/archive_attribute_unet_20260911
PYTHONPATH=.. python 14_realtime_inference_attribute.py --replay datasets_field/E25_door_RH_s_091317   # dimension_utils·hole_classifier·camera_utils 는 상위 폴더
```
`vent_labels/` 이미지는 `python 11_generate_vent_labels.py` 로 재생성한다.
