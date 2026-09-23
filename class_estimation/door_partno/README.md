# door_partno/ — 학습형 품번(14)·레이더 판정 연구 (2026-09-23 착수)

`door_pipeline/partno/`(규칙 기반 레이더 검사, 운영 정본)의 **확정 라벨**로 학습형 판정기를 만들어 규칙과 비교한다.
door_pipeline 과 같은 번호 체계를 쓰고, 데이터·DB·공용 유틸은 심볼릭 링크로 공유한다(`datasets_factory_v2`, `datasets_factory_collect`, `datasets`, `db`, `*_utils.py`).
결정(사용자 2026-09-23): 폴더 분리, 검출기는 브래킷 채널 추가 + 자동(의사) 라벨, CNN 은 9클래스+레이더 헤드 주안·14클래스 비교·448 유지(640 변형 1회),
평가는 고정 test + 세션 5-fold CV, 기준선 = 규칙 검사, 이득은 규칙 미판정(가림·판 밖) 프레임에서 본다.

## 구성

| 파일 | 역할 |
|---|---|
| `partno_labels.py` | `datasets_factory_v2` 프레임 → (클래스, 레이더 GT(user), 품번, split) 매니페스트 `labels/partno_manifest.json` |
| `15_make_pseudo_labels.py` | 검출기 학습용 의사 라벨: 6점은 정본 3채널 검출기(게이트 통과), 브래킷 2점은 규칙 판정(레이더 확정 세션) → `labels/holes_pseudo/` (+사람 라벨 134장 보강) |
| `hole_classifier4.py` | 4채널 판별기 모듈(bolt/corner_hinge/corner_latch/bracket): 형상군 = door_pipeline 규칙(픽셀 폭 D), 레이더 = 브래킷 피크 2개 + CAD 기하(간격 91.6·위치 ≤40mm) |
| `16_train_hole_landmarks.py` | 정본 3채널 가중치에서 브래킷 채널 미세조정(채널 마스크 손실) → `attribute_models/hole_landmarks_bracket/` |
| `17_evaluate_hole_classifier.py` | 고정 test 에서 검출기 vs 규칙: 형상군·레이더·품번 판정률/정확도, 규칙 미판정 프레임 구간, 세션 다수결 → `report/hole4_*.md` |
| `02_train_multitask.py` | CNN: `--mode multi`(9클래스+레이더 헤드) / `--mode part14`(품번 14), `--image_size 448|640`, seed916 초기화 → `artifacts/partno_*` |
| `03_evaluate_multitask.py` | CNN 고정 test/전량 평가(프레임·세션, 품번별) → `report/cnn_*.md` |
| `scripts/session_cv.py` | CNN 세션 5-fold CV → `report/cv_*.json` |
| `19_blind_eval.py` | **블라인드 평가**: 학습에 쓰지 않은 세션(기본 DB split 미지정 35세션, `--since YYYYMMDD` 로 신규 유입)에서 규칙·4채널 검출기·CNN 을 같은 프레임으로 채점 → `report/blind_<tag>.md` |
| `scripts/run_chain.sh` | 학습 종료 순서대로 17·03 평가·CV·리포트 자동 실행 |

## 실행 순서

```bash
cd class_estimation/door_partno && PY=../../venv/parts_deep/bin/python
$PY partno_labels.py                                   # ① 매니페스트 (DB 확정 라벨 반영)
$PY 15_make_pseudo_labels.py                           # ② 의사 라벨 (GPU, 약 5분)
$PY 16_train_hole_landmarks.py --epochs 10 --lr 5e-5   # ③ 4채널 검출기 미세조정 (split 미지정 세션은 기본 제외 — CNN 과 동일 기준)
$PY 17_evaluate_hole_classifier.py                     # ④ 검출기 vs 규칙 (고정 test 596장)
$PY 02_train_multitask.py --mode multi --image_size 448 --seed 42     # ⑤ CNN 주안
$PY 02_train_multitask.py --mode part14 --image_size 448 --seed 42    #    비교 기준선
$PY 02_train_multitask.py --mode multi --image_size 640 --seed 42     #    해상도 변형
$PY 03_evaluate_multitask.py --run partno_multi_448_seed42 [--db-log] # ⑥ CNN 평가
$PY scripts/session_cv.py --mode multi --image_size 448 --tag cvA     # ⑦ 5-fold CV (fold당 수십 분)
$PY 19_blind_eval.py --tag unsplit                                     # ⑧ 블라인드(split 미지정 35세션; 검출기는 --exclude-unsplit 로 따로 학습한 모델 지정)
$PY report/build_report.py                                             # ⑨ report/README.md 취합
```

신규 세션 블라인드 루틴(운영): `db/ingest_nas.py` → `db/pull_nas.py` → `19_blind_eval.py --since <날짜> --tag <태그>`(확정 전 예측 저장) → 웹 `/options` 확정 → 같은 명령 재실행(채점).
검출기 블라인드 모델: `16_train_hole_landmarks.py --exclude-unsplit --out hole_landmarks_bracket_blind`.

## 결과 (2026-09-23, report/README.md)

- 블라인드 35세션 662장(학습 미사용): 규칙 100%(판정률 99.8%) · 35세션을 뺀 4채널 검출기 100%(판정률 97.1%) · CNN multi448 90.9% / part14 90.3% / multi640 95.2% — CNN 오류는 전부 형상군 혼동(레이더 오류 0).
- 고정 test 596장: 규칙·검출기·CNN 3종 모두 품번 100%. 전량 3,126장: 검출기 레이더 2,344/2,350 판정·100%, 규칙 2,346/2,350·100%, 규칙 미판정 4장 중 검출기 3장 정답.
- CNN multi448 세션 5-fold CV 99.35% (fold별 97.3/100/100/100/100).
- 결론: 형상군은 홀 거리 규칙, 레이더는 검출기 브래킷 채널 → 검출기 한 모델이 실시간 통합 후보. git 미추적: `artifacts/`(349MB), `attribute_models/*/model.pth`, `labels/holes_pseudo/`(재생성), `logs/`.

## 주의

- 라벨은 규칙 판정을 사람이 검토·승인한 것(약지도). 검출기 6점 의사 라벨은 정본 검출기 출력이라 6점 정밀도는 사람 홀드아웃(27장)으로만 확인한다.
- 레이더 GT 가 없는 세션(미정 3)과 FRT 는 레이더 학습·평가에서 제외(FRT 는 브래킷 음성 예로만 사용).
- 18번 실시간 서버의 옵션·품번 출력은 이 연구 결과가 나오면 함께 통합한다(사용자 결정).
