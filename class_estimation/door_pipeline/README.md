# door_pipeline — 도어 분류 정리본 파이프라인

`door/`(실험 이력)와 `door_paper/`(논문 실험)에서 **가장 결과가 좋은 방식만
추출·정리**한 폴더 (2026-07-08). 기존 폴더는 이력 보존용으로 그대로 두며,
신규 작업은 여기서 진행한다.

두 축으로 구성된다 (2026-09-11부터):

| 축 | 스크립트 | 역할 |
|---|---|---|
| C. 홀 랜드마크 판별기 | `15`~`19` + `hole_classifier.py` | **1순위 판정** (현장 100%), 실시간 `18` |
| A. CNN 분류 (RGBE NoAux 448) | `01`~`05` | 현장 재학습 정식 run(DB #8, 세션 CV 97.4%) — 홀 보류 시 **폴백** |

- ~~B. 속성 파이프라인 (CAD 정합 + 통풍구 U-Net + 종횡비, `10`~`14` + `attribute_utils.py`)~~ →
  **2026-09-11 아카이빙** (`archive_attribute_unet_20260911/`, 사유·최종 수치는 그 안의 README). 홀 판별기 대비 이점이 없어 폐기.
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

## C. 홀 랜드마크 판별기 (2026-08-26, 1순위 판정)

모서리 프레임 홀 2개(래치 각공 쪽 변, 실제 장착 기준 바닥 쪽)의 거리 D(=도어 폭−106mm)로 8종을 판정한다. CNN이 홀 6점(래치 볼트홀 4 + 모서리 홀 2)을
찾고, depth 평면상 거리 × K_DEPTH(카메라 모드별 캘리브레이션) → CAD D 최근접 (2026-09-07부터 E23 포함 9종).
볼트 프레임 기하 게이트에 걸리면 보류한다. (`hole_classifier.classify(group=)` 그룹 제약 인자는 남아 있으나
속성 파이프라인 아카이빙 후 호출부 없음 — 제약 유무 차이는 datasets 99.0→99.5%, 현장 100→100%였다.)

실행 순서 (A와 같은 번호 순, 래퍼는 `scripts/hole_*.sh`):

```bash
# ⑮ 라벨링 (브라우저 :8090) → labels/holes/*.json          [scripts/hole_label.sh]
python 15_label_holes.py --per-class 15 --extra datasets_field
# ⑯ 학습 → attribute_models/hole_landmarks/model.pth (+DB)  [scripts/hole_train.sh]
python 16_train_hole_landmarks.py
# ⑰ 평가 → eval_classifier.json, report/hole_analysis/samples/ (+DB)  [scripts/hole_evaluate.sh]
python 17_evaluate_hole_classifier.py                 # test 분할 · datasets 전체 · datasets_field
python 17_evaluate_hole_classifier.py --base datasets_field
python tools/make_hole_samples.py                     # 성공/오판/보류 샘플 + 학습 곡선
# ⑱ 실시간 추론 — 홀 판별기 (모델 1개, 로드 약 0.5 s, 웹 UI :5004)  [scripts/hole_only_inference.sh]
python 18_realtime_inference_hole.py                                   # ZED 카메라
python 18_realtime_inference_hole.py --replay datasets_field/E25_door_RH_s_091317 --fp16
python scripts/hole_classify_image.py rgb_0003.png    # 단일 이미지
# ⑲ 체커보드 스케일 캘리브 (K_DEPTH)
python 19_checker_scale_calib.py
```

- ⑱은 보류 시 판정 없음("보류")으로 남긴다. 홀 보류 시 CNN 정식 run 폴백 통합은 **미구현 과제**(아래 유의사항).
  종전 통합 서버 ⑭(홀 1순위 + U-Net 속성 폴백, :5003)는 아카이브에 있다.
  ⑱ API `/api/inference_result`: `class, group, confidence(=판정 프레임 비율), D_mm(윈도 중앙값), margin_mm, n_judged, gate, gate_counts(윈도 게이트 분포),
  candidates[{class, cad_D_mm, diff_mm}], frame{gate, D_mm, D_src, points}`. `/api/reset` 으로 도어 교체 시 윈도 초기화. depth 없는 리플레이 폴더는 볼트 스케일 D(검증용).

- 평가(라벨 학습분 제외): test 분할 134/189 판정·**100%**, datasets 전체 814/1162 판정·**99.0%**(그룹 제약 시 99.5%),
  현장 16/16·**100%**. 보류(~30%)는 힌지측 홀이 프레임 밖/경계 근접 — 사무실 촬영 프레이밍 문제, 현장은 0%.
- 오판은 검출이 아니라 **거리 측정 오차**(depth 평면, 20~58mm)에서 나온다. 그룹 제약이 RH↔RR 혼동을 막고, FRT 내(41mm 간격)는 N프레임 집계·intrinsics 정밀화로 대응.
- 중앙 보강대 패드 홀은 옵션/리비전에 따라 달라 사용 금지. 분석 기록: `report/hole_analysis/`.
- **E23_door_LH_FRT 추가 (2026-09-07)**: 현장에서 E25_LH_FRT로 입력된 9/3·9/5 4세션이 D≈460mm로 일관 보류 → 육안·CAD 확인 결과 E23(폭 562, D=456). `CAD_D`에 9번째 클래스 등록, 유효 D 범위 `D_RANGE`=400~1500mm(게이트·최종 depth D 공통; 종전 게이트 하한 600, depth D 무검사). CAD `cad/door_stp/E23) 110982-02723 LH FRT.stp`는 외판 단일 솔리드(보강재·힌지 없음)라 코너 프레임 홀이 없음 → 속성 템플릿(10) 생성. 자세 추정 `cad_holes.json`은 **잠정 등록**(`01_extract_cad_holes.PROVISIONAL`: 볼트홀·래치 코너는 E25와 동일 좌표, 힌지 코너는 CAD_D 이동 합성; 현장 4세션 80장 정합 잔차 med 3.3mm·max 5.1mm로 8종과 동급). 보강재 포함 어셈블리 STEP 확보 시 정식 추출로 교체. STL 변환은 `gmsh <stp> -2 -format stl`(numpy-stl은 STEP 불가).

## 산출물/데이터 위치

- `artifacts/` — CNN 모델 run별 산출물. *.pth는 기본 git 미추적이며, **정식 배포 run**(`rgbe_noaux_448_seed42_datasets_factory_v2`, DB models #8)의 `model.pth`만 Gitea LFS로 추적
- `attribute_models/hole_landmarks/` — 홀 판별기 모델(`model.pth`, Gitea LFS)·split·평가 json
- `archive_attribute_unet_20260911/` — 속성 파이프라인(U-Net) 아카이브: 스크립트 10~14, attribute_utils, DOC, 모델·템플릿·평가 산출물, vent_labels meta, factory_masks
- `datasets*`, `sam_models` — `../door/` 심볼릭 링크 (`sam_models`는 아카이브 스크립트만 사용)
- `datasets_factory_collect/` — NAS 미러(세션당 20장 샘플, `db/pull_nas.py`), `datasets_factory_v2/` — 학습·평가용 링크 뷰(`db/build_dataset.py build`). 흐름은 `db/README.md` 참조

## 모델 배포 동기화 (GitHub + Gitea LFS, 2026-09-10)

학습 PC ↔ 추론 PC 사이의 모델(pth) 교환은 **Gitea LFS**로 한다 (ros2_ws_ms/dx300_rl과 같은 방식).

- 원격: `origin` fetch = GitHub, push = GitHub **+** Gitea(`http://git.kocetismart.kr:3000/rebirther00/part_deep.git`) 동시(pushurl 2개).
- LFS 저장소는 `.lfsconfig`로 **Gitea 전용**. GitHub에는 포인터 파일만 올라가고 실제 바이너리는 Gitea에만 저장된다(GitHub LFS 용량 미사용).
- `.gitattributes`: `*.pth *.pt *.onnx *.engine` → LFS. 단 `.gitignore`가 pth를 기본 제외하므로 **배포 모델만** `.gitignore` 하단에 예외 등록:
  `attribute_models/hole_landmarks/model.pth`(홀 판별기) · `artifacts/rgbe_noaux_448_seed42_datasets_factory_v2/model.pth`(CNN 정식 run).
  (U-Net `vent_unet.pth`·MobileSAM `mobile_sam.pt`는 2026-09-11 속성 파이프라인 아카이빙으로 배포 목록에서 제외. git/LFS 이력은 유지.)
  새 run을 배포본으로 승격하면 `.gitignore` 예외와 이 목록을 같이 갱신한다.

```bash
# 학습 PC: 모델 갱신 후 (같은 경로에 덮어쓰면 LFS가 새 객체로 인식)
scripts/model_sync.sh push          # git add 배포 모델 → commit → GitHub+Gitea 동시 push (LFS는 Gitea)

# 추론 PC: 최초 1회
git lfs install
git clone http://git.kocetismart.kr:3000/rebirther00/part_deep.git parts_deep   # Gitea에서 clone (LFS 자동 수신)
#   또는 GitHub clone 후: .lfsconfig 가 Gitea를 가리키므로  git lfs pull  만 하면 됨
# 추론 PC: 이후 갱신
scripts/model_sync.sh pull          # git pull + git lfs pull + 배포 모델이 포인터가 아닌 실체인지 검사
```

- 추론 PC가 GitHub에 push 권한이 없으면 `git config remote.origin.pushurl http://git.kocetismart.kr:3000/rebirther00/part_deep.git` 로 Gitea 단일 push로 바꿔 쓴다.
- Gitea 인증은 `credential.helper=store`(`~/.git-credentials`)에 `git.kocetismart.kr:3000` 토큰이 있어야 한다.

## 유의사항

- 평가 방법론: train/val/test 3분할, test 완전 격리, **datasets_aug는 학습
  금지**(강건성 확인 전용), 현장은 별도 평가.
- 향후 과제: **18 홀 서버에 CNN 정식 run 폴백 통합**(홀 보류 시 CNN 판정, 불일치 시 보류 — 종전 14의 U-Net 폴백 자리를 CNN이 대체),
  현장 프로토콜 준수 재취득.
