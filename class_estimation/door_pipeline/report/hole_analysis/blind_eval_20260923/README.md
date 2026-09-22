# CNN 폴백 정식 run 교체 판단 — 블라인드 평가 (2026-09-23)

## 평가 집합 (두 CNN run 모두 학습·검증·test 에 쓰지 않은 세션)
- NAS 9/16 오후~9/23 오전 유입 **35세션**(DB capture_sessions #596~#629 + 9/23 1세션). `db/pull_nas.py` 로 세션당 20장 pull(합계 3,835장 미러),
  auto-split·build·`20_session_drift.py` 는 **미적용**(이 평가는 분할 전 상태에서 수행).
- **663장** = E25_LH_RR 8세션 158 · E25_RH 6/120 · E30_E38_RH 7/139 · E38_LH_RR 5/66(1장·5장짜리 짧은 세션 2개 포함) · E38_LH_FRT 3/60 · E30_LH_RR 3/60 · E30_LH_FRT 2/40 · **E25_LH_FRT 1/20**(9/16 s_135409).
- 뷰: `datasets_blind_20260923/collect/<날짜>/<클래스>/s_*`(17용) · `datasets_blind_20260923/cnn/<클래스>/rgb_*.png`(04용), 심볼릭 링크·git 미추적.
- GPU 는 9/23 새벽 드라이버 자동 업데이트(580.173→580.178, 커널 모듈 불일치)로 CUDA 불가 → 세 평가 모두 CPU(04 각 ~3.5분, 17 4.5분).

## 결과

| 판별기 | 663장 | 비고 |
|---|---|---|
| **홀 판별기**(17, K_CAMERA 1.0064) | **663/663 = 100%**, 판정률 100%, unknown 0, 게이트 ok 663 | 9/16 `E25_door_LH_FRT/s_135409` D med **718mm**(CAD 724) → 진짜 E25_LH_FRT(E23=456 오기입 아님) → **E25_LH_FRT 유효 세션 2개**(9/4, 9/16) |
| **CNN seed916**(9/16 학습, train 103세션) | **659/663 = 99.4%**, macro F1 99.0 | 오답 4: E30_LH_FRT→E38_LH_FRT 1(p 0.70), E38_LH_FRT→E30_LH_FRT 3(p 0.55~0.59). E30_LH_RR 60/60, E25_LH_FRT 20/20 |
| CNN run #10 seed42(9/7 학습) | 559/663 = 84.3%, macro F1 65.4 | **E30_LH_RR 12/60**(→E38_LH_RR p 0.93~0.95), E38_LH_FRT 24/60(→E30_LH_FRT), E25_LH_FRT 0/20 |

클래스별 정확도(CNN):

| 클래스 | n | seed916 | run #10 |
|---|---|---|---|
| E25_door_LH_FRT | 20 | 100 | 0 |
| E25_door_LH_RR | 158 | 100 | 100 |
| E25_door_RH | 120 | 100 | 100 |
| E30_E38_door_RH | 139 | 100 | 100 |
| E30_door_LH_FRT | 40 | 97.5 | 100 |
| E30_door_LH_RR | 60 | 100 | 20.0 |
| E38_door_LH_FRT | 60 | 95.0 | 40.0 |
| E38_door_LH_RR | 66 | 100 | 100 |

## 판단 → **seed916 을 정식 폴백 run 으로 교체 (2026-09-23 반영)**
- run #10 은 폴백의 존재 이유인 RR 3종에서 결함(자기 학습 세션 E30_LH_RR 0/14, test 19/59, 블라인드 12/60)이 세 집합에서 일관되게 재현됨.
  val loss 폭발(8~9에폭 26.7) 뒤 3에폭짜리 덜 학습된 체크포인트가 저장된 run.
- seed916 은 9/16 단일 run(seed 선택 없음), 체크포인트는 val 기준(5에폭). test 596장 99.8% + 블라인드 663장 99.4%, 두 집합 합쳐 오답 5장 모두 E30↔E38 LH_FRT 인접 쌍·저신뢰.
- 9/9 5-fold 세션 CV 도 같은 결론: E30_LH_RR 은 4/5 fold 100%, 무너진 fold1(69%)·fold0(E38_LH_FRT 2.5%)은 val loss 스파이크가 있던 run → **데이터 한계가 아니라 학습 run 의 운에 따라 특정 클래스가 붕괴하는 레시피 문제**.
- 홀 판별기 주·CNN 폴백(보류 프레임만, 홀 우선) 구조라 교체 운영 위험은 작음. 두 run 모두 9클래스·같은 구조, 클래스명은 run 폴더 split_info 에서 읽으므로 드롭인.

### seed42 와 seed916 의 차이 (질문 답)
seed 값 자체는 난수 시드(random·numpy·torch)로, 신규 4채널 첫 conv·분류 head 초기화, 미니배치 순서, 증강 난수만 정한다. `datasets_factory_v2` 에 val/ 이 있어 **분할은 seed 와 무관**(DB 세션 분할). 916 은 9/16 학습이라 붙인 임의 값.
두 run 은 seed 외에 **학습 데이터 시점이 다르다**: run #10 = 9/7 스냅샷(8/27~9/7, train 1,139장) / seed916 = 9/16 스냅샷(8/27~9/16, train 2,094장·103세션). 레시피(Adam 1e-3·batch 64·역빈도 가중 CE·val_acc 선택·patience 10)는 동일.
성능 차이는 세션 증가(E30_LH_RR 6→11)와 학습 run 의 운(스파이크 후 회복 여부)이 섞인 것이며, 9/16 데이터로 seed42 를 다시 돌리지 않는 한 분해되지 않는다. 5-fold 재실행이 그 역할을 한다.

## 교체 반영 (코드)
- `cnn_classifier.py` `DEFAULT_RUN` → seed916, `18_realtime_inference_hole.py` 도움말, `scripts/model_sync.sh` 배포 목록, 루트 `.gitignore` LFS 예외 블록(model.pth·split_info.json), `README.md` 산출물/배포 절, `db/webapp.py` CNN 카드 이름.
- `04_evaluate_factory.py --out_dir` 추가(결과를 모델 폴더 밖에 저장 — 9/16 test 결과 보존).
- 학습 레시피: `02_train.py` 클래스 가중치 기본 **sqrt 역빈도**(`--class_weight inv` 로 이전 재현, `none` 가능) + `--lr`(기본 0.001). run #10·seed916·CV fold0/1 의 val loss 폭발은
  역빈도 가중치(E25_LH_FRT 20장 → 11.6 vs E30_E38_RH 0.51, **23배**)와 lr 1e-3 조합이 유력 원인. sqrt 면 3.4 vs 0.72(4.8배).
- `scripts/session_cv.py`: 9/7 `report/hole_analysis/field_20260907/session_cv.py` 의 범용판(`--tag`·`--seed`·02_train 인자 전달, 9/9 fold 산출물 덮어쓰지 않음).

## 홀 판별기 세션별 D (K_CAMERA 1.0064 적용, mm) — 드리프트 참고
날짜별 dev 중앙값: 9/16 −7.1(3세션) · 9/17 −6.8(10) · 9/18 +0.9(7) · 9/21 −6.6(1) · 9/22 −0.9(13) · 9/23 −7.8(1).
9/9 이후의 −7mm 편향이 날마다 있다 없다 하며, 세션 간 산포가 ±13mm 로 커졌다. **최소 마진 14mm**(9/22 E38_LH_FRT s_130857, dev +13.8) — FRT 간격 41mm 의 1/3 소모. K 운영 규칙 결정은 여전히 미결(`20_session_drift.py` 미실행).

| 세션 | 클래스 | n | 오판 | D med | dev | margin min |
|---|---|---|---|---|---|---|
| 20260916/E25_door_LH_FRT/s_135409 | E25_door_LH_FRT | 20 | 0 | 718.2 | -5.8 | 41 |
| 20260916/E25_door_LH_RR/s_125051 | E25_door_LH_RR | 19 | 0 | 1029.9 | -7.1 | 50 |
| 20260916/E25_door_RH/s_132345 | E25_door_RH | 20 | 0 | 877.9 | -8.1 | 51 |
| 20260917/E25_door_LH_RR/s_092119 | E25_door_LH_RR | 20 | 0 | 1032.6 | -4.4 | 50 |
| 20260917/E25_door_LH_RR/s_095747 | E25_door_LH_RR | 20 | 0 | 1028.0 | -9.0 | 50 |
| 20260917/E25_door_LH_RR/s_122607 | E25_door_LH_RR | 19 | 0 | 1030.6 | -6.4 | 50 |
| 20260917/E25_door_LH_RR/s_125429 | E25_door_LH_RR | 20 | 0 | 1029.9 | -7.1 | 50 |
| 20260917/E25_door_LH_RR/s_153907 | E25_door_LH_RR | 20 | 0 | 1027.7 | -9.3 | 50 |
| 20260917/E25_door_RH/s_074245 | E25_door_RH | 20 | 0 | 892.4 | +6.4 | 74 |
| 20260917/E25_door_RH/s_103841 | E25_door_RH | 20 | 0 | 883.5 | -2.5 | 63 |
| 20260917/E30_E38_door_RH/s_133729 | E30_E38_door_RH | 20 | 0 | 1086.9 | -0.1 | 43 |
| 20260917/E38_door_LH_RR/s_170221 | E38_door_LH_RR | 20 | 0 | 1341.5 | -10.5 | 169 |
| 20260917/E38_door_LH_RR/s_173807 | E38_door_LH_RR | 20 | 0 | 1339.3 | -12.7 | 164 |
| 20260918/E25_door_RH/s_080409 | E25_door_RH | 20 | 0 | 887.2 | +1.2 | 74 |
| 20260918/E30_E38_door_RH/s_095707 | E30_E38_door_RH | 20 | 0 | 1089.7 | +2.7 | 48 |
| 20260918/E30_E38_door_RH/s_101954 | E30_E38_door_RH | 19 | 0 | 1098.0 | +11.0 | 40 |
| 20260918/E30_door_LH_FRT/s_130044 | E30_door_LH_FRT | 20 | 0 | 760.6 | -4.4 | 25 |
| 20260918/E30_door_LH_FRT/s_132636 | E30_door_LH_FRT | 20 | 0 | 765.9 | +0.9 | 33 |
| 20260918/E30_door_LH_RR/s_105355 | E30_door_LH_RR | 20 | 0 | 1157.1 | -0.9 | 55 |
| 20260918/E30_door_LH_RR/s_122814 | E30_door_LH_RR | 20 | 0 | 1150.9 | -7.1 | 53 |
| 20260921/E38_door_LH_RR/s_155719 | E38_door_LH_RR | 5 | 0 | 1345.4 | -6.6 | 179 |
| 20260922/E25_door_LH_RR/s_135225 | E25_door_LH_RR | 20 | 0 | 1031.8 | -5.2 | 50 |
| 20260922/E25_door_LH_RR/s_144307 | E25_door_LH_RR | 20 | 0 | 1027.9 | -9.1 | 50 |
| 20260922/E25_door_RH/s_151323 | E25_door_RH | 20 | 0 | 898.4 | +12.4 | 74 |
| 20260922/E25_door_RH/s_154542 | E25_door_RH | 20 | 0 | 877.3 | -8.7 | 49 |
| 20260922/E30_E38_door_RH/s_084434 | E30_E38_door_RH | 20 | 0 | 1100.4 | +13.4 | 32 |
| 20260922/E30_E38_door_RH/s_094707 | E30_E38_door_RH | 20 | 0 | 1086.1 | -0.9 | 42 |
| 20260922/E30_E38_door_RH/s_103147 | E30_E38_door_RH | 20 | 0 | 1074.5 | -12.5 | 22 |
| 20260922/E30_door_LH_RR/s_163059 | E30_door_LH_RR | 20 | 0 | 1161.3 | +3.3 | 71 |
| 20260922/E38_door_LH_FRT/s_122130 | E38_door_LH_FRT | 20 | 0 | 818.6 | +6.6 | 46 |
| 20260922/E38_door_LH_FRT/s_124410 | E38_door_LH_FRT | 20 | 0 | 821.5 | +9.5 | 45 |
| 20260922/E38_door_LH_FRT/s_130857 | E38_door_LH_FRT | 20 | 0 | 825.8 | +13.8 | 14 |
| 20260922/E38_door_LH_RR/s_080559 | E38_door_LH_RR | 1 | 0 | 1331.3 | -20.7 | 153 |
| 20260922/E38_door_LH_RR/s_081005 | E38_door_LH_RR | 20 | 0 | 1343.0 | -9.0 | 174 |
| 20260923/E30_E38_door_RH/s_072354 | E30_E38_door_RH | 20 | 0 | 1079.2 | -7.8 | 31 |
## 산출물
- CNN: `rgbe_noaux_448_seed916_datasets_factory_v2/factory_eval_results.json`(+png), `rgbe_noaux_448_seed42_datasets_factory_v2/factory_eval_results.json`(+png) — 이 폴더. DB evaluation_results(cross_domain, dataset=`datasets_blind_20260923/cnn`).
- 홀: `attribute_models/hole_landmarks/eval_classifier_datasets_blind_20260923_collect.json`(+혼동행렬 png), DB evaluation_results(inference_pipeline).

## 남은 작업
1. **재부팅**(9/23 07:14 unattended-upgrades 가 nvidia 580.178 설치, 커널 모듈은 580.173 → CUDA 불가. `/var/run/reboot-required` 있음, 7.0.0-31 커널용 nvidia 모듈 dkms 빌드 확인됨).
2. 5-fold 세션 CV 재실행(새 레시피 검증, fold당 1~1.5시간): `python scripts/session_cv.py --out <scratch>/cv --tag cv20260923` → val loss>2 스파이크 fold 0개·클래스 붕괴 없음이면 레시피 확정. 비교군: `--tag cv20260923inv -- --class_weight inv`.
3. 35세션 정식 편입: `db/build_dataset.py auto-split` → `build` → `20_session_drift.py`. E25_LH_FRT 는 이제 2세션이라 auto-split 규칙(세션<3 전부 train)상 아직 평가 제외.
4. 커밋·`scripts/model_sync.sh push`(seed916 model.pth LFS → Gitea, 추론 PC 는 `model_sync.sh pull`). 이전 run #10 의 model.pth·split_info.json 은 `git rm --cached`(이력 유지).
