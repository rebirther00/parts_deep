# 통합모델 SIDE DOOR 현장 수집 데이터 명세서 (품번 14종)

작성 2026-09-23 · 데이터 기준일 2026-09-23 · 생성 스크립트 `partno/build_dataset_spec.py` (DB `db/door_pipeline.db` 기준, 재실행으로 갱신)

## 1. 개요

- **목적**: 명성공업 도어 용접 라인(경주)에서 무인 수집한 굴착기 통합모델 SIDE DOOR RGB-D 데이터를, 도어 **어셈블리 품번 14종** 기준으로 정의·집계한다. 도어 분류 AI(홀 랜드마크 판별기 + CNN 폴백)와 자세 추정의 학습·평가 데이터이며, 품번 판정 = 형상군(9) × 레이더 옵션(유/무).
- **정본 위치**: 세종 NAS `nas:Guest/weld/<날짜>/<클래스>/s_HHMMSS/` (전량). 학습 PC 로컬 미러 `datasets_factory_collect/`(세션당 20장 샘플), 학습·평가 뷰 `datasets_factory_v2/`(심볼릭 링크), 메타데이터 DB `db/door_pipeline.db`.
- **규모**: 유효 세션 190 · NAS 수집 쌍 18,074 · 로컬 샘플 3,789장 (기간 20260827~20260923, 무인 수집 정식 설치 2026-08-27). 무효 세션 6(시험 촬영·빈 지그), 라벨 미정 4, 라벨 정정 세션 10.
- **주요 결정(2026-09-23)**: 품번 판정은 2단(형상군 → 레이더 옵션 → 품번 매핑), 레이더 옵션은 CAD 브래킷 투영 규칙 검사(학습 없음)로 자동 판정 후 사용자가 육안 확정, 클래스 체계(9종)와 태블릿 라벨은 유지하고 DB 에 옵션 컬럼만 추가, 공식 지표는 품번 정확도, CAD 는 필요한 것만 교체. AVM 옵션은 구분하지 않음.

## 2. 품번 체계 (14)

사양표·도면(2026-09-23 입수 `cad/통합모델 SIDE DOOR 도면 모음/`) 기준. 키는 리비전을 뺀 기본 품번, 리비전은 별도 필드. E23/E25 는 LH REAR·RH 를 공용하고 E30/E35/E38 은 RH 를 공용하므로 E23 고유 품번은 LH FRT 하나뿐이다. RADAR 옵션은 LH REAR·RH 에만 있다.

| 품번 | rev | 명칭 | 적용 기종 | 위치 | 레이더 | 사양표 폭(mm) | CAD D(mm) | 파이프라인 클래스 | 도면/STEP |
|---|---|---|---|---|---|---:|---:|---|---|
| **110982-02444** | B | DOOR,SIDE;LH FRT | E23 | LH FRT | — | 562 | 458 | E23_door_LH_FRT | E23) 110982-02444B LH FRT |
| **110982-02431** | B | DOOR,SIDE;LH;FRONT | E25 | LH FRT | — | 828 | 724 | E25_door_LH_FRT | E25) 110982-02431B LH FRT |
| **110982-02414** | H | DOOR ASSY,SIDE;LH,FR | E30 | LH FRT | — | 869 | 765 | E30_door_LH_FRT | E30) 110982-02414H LH FRT |
| **110982-02447** | L | DOOR ASSY,SIDE;LH,FR,W_AVM | E35/E38 | LH FRT | — | 916 | 812 | E38_door_LH_FRT | E35,E38) 110982-02447L LH FRT |
| **110982-02427** | C | DOOR,SIDE;LH,RR,W_O RADAR | E23/E25 | LH REAR | X | 1141 | 1037 | E25_door_LH_RR | E23,E25) 110982-02427C LH REAR |
| **110982-02375** | C | DOOR,SIDE;LH;REAR (RADAR) | E23/E25 | LH REAR | O | 1141 | 1037 | E25_door_LH_RR | E23,E25) 110982-02375C LH REAR RADAR |
| **110982-02352** | H | DOOR ASSY,SIDE;LH,REAR | E30 | LH REAR | X | 1262 | 1158 | E30_door_LH_RR | E30) 110982-02352H LH REAR |
| **110982-02360** | K | DOOR ASSY,SIDE;LH,RR (RADAR) | E30 | LH REAR | O | 1262 | 1158 | E30_door_LH_RR | E30) 110982-02360K LH REAR RADAR |
| **110982-02292** | M | DOOR ASSY,SIDE;LH,RR,W_O RADAR | E35/E38 | LH REAR | X | 1456 | 1352 | E38_door_LH_RR | E35,E38) 110982-02292M LH REAR |
| **110982-02325** | M | DOOR ASSY,SIDE;LH,RR,W_RADAR | E35/E38 | LH REAR | O | 1456 | 1352 | E38_door_LH_RR | E35,E38) 110982-02325M LH REAR RADAR |
| **110982-02320** | E | DOOR,SIDE;RH | E23/E25 | RH | X | 990 | 886 | E25_door_RH | E23,E25) 110982-02320E RH |
| **110982-02424** | E | DOOR,SIDE;RH,W_RADAR | E23/E25 | RH | O | 990 | 886 | E25_door_RH | E23,E25) 110982-02424E RH RADAR |
| **110982-02334** | M | DOOR ASSY;RH (W_O RADAR) | E30/E35/E38 | RH | X | 1191 | 1087 | E30_E38_door_RH | E30,E38) 110982-02334M RH |
| **110982-02327** | N | DOOR ASSY,SIDE;W_RADAR | E30/E35/E38 | RH | O | 1191 | 1087 | E30_E38_door_RH | E30,E38) 110982-02327N RH RADAR |

- **형상군 클래스(9)**: 좌우 세로 프레임 상단 모서리 홀(Ø6) 2개 거리 D = 폭 − 106mm 로 판정(`hole_classifier.CAD_D`, 최근접 CAD, 편차 > 30mm 는 unknown). D 는 힌지↔래치 픽셀 폭 × S_PIXEL/fx (S_PIXEL[54910212] = 1477.8 mm). 그룹: FRT ['E23_door_LH_FRT', 'E25_door_LH_FRT', 'E30_door_LH_FRT', 'E38_door_LH_FRT'], RR ['E25_door_LH_RR', 'E30_door_LH_RR', 'E38_door_LH_RR'], RH ['E25_door_RH', 'E30_E38_door_RH'].
- **레이더 옵션 정의**: 중앙 가로 보강재(MID) 위 브래킷 판(약 346×127mm, 6홀 평면보다 카메라 쪽 70mm)에 Ø34 홀 2개(간격 91.6mm)가 있으면 레이더 사양(1), 없으면 미장착(0). FR/RR 세로 보강재(코너 홀)·상하 보강재·힌지·볼트 사각(157×96)은 두 사양이 같은 품번이라 형상군 D 와 자세 랜드마크는 옵션과 무관. FRT 는 옵션 없음(NULL).
- **브래킷 좌표(도어 프레임, mm)**: E25_door_LH_RR [[-13.3, 470.5], [78.3, 470.6]]; E25_door_RH [[62.2, -471.5], [153.8, -471.5]]; E30_E38_door_RH [[162.7, -471.6], [254.3, -471.5]]; E30_door_LH_RR [[198.2, 470.6], [289.8, 470.6]]; E38_door_LH_RR [[23.2, 470.6], [114.8, 470.6]] (`partno/radar_bracket.json`, CAD STL 카메라 쪽 최대 z 지도에서 추출).
- 사양표 폭은 CAD D + 106 보다 일관되게 2mm 작다(코너 홀 오프셋 52/53mm 차이). 판정에는 현장 실측으로 검증된 CAD D 를 쓴다.

## 3. 파일 구조와 형식

```
<날짜 YYYYMMDD>/<태블릿 클래스명>/s_HHMMSS/        # 세션 = 도어 1장, 06_factory_capture.py 가 2초 간격 최대 5분 저장
  rgb_NNNN.png     1920×1200 RGB 8-bit PNG (ZED X Mini 좌안, rectified)
  depth_NNNN.png   1920×1200 16-bit PNG, 단위 mm (0 = 무효)
  meta.json        class_name, started_at/ended_at, camera, capture_interval_s, session_duration_s, intrinsics{serial,width,height,fx,fy,cx,cy,rectified}, saved_pairs, finished, stop_reason
datasets_factory_collect/  위 구조의 로컬 미러(세션당 20장 균등 샘플, db/pull_nas.py)
datasets_factory_v2/{all,train,val,test}/<DB 클래스>/rgb_<날짜>_<세션>_<idx>.png (+depth_)   # 심볼릭 링크 뷰 + manifest.json
```

- DB(`db/schema.sql`): `capture_sessions`(세션 메타·태블릿 원본 라벨 `class_name`·이력 `notes`·**옵션 `option_radar/option_source/option_note/option_at`**), `images`(정정 라벨 `class_id`·`is_valid`·`split`·`synced_local`), `session_hole_metrics`(세션별 홀 판정 지표·다수결 `pred_major`), `session_radar_check`(레이더 자동 검사 근거), `evaluation_results`.
- 카메라: ZED X Mini(SN 54910212) 1920×1200, fx=fy≈1269.7, 고정 지그 위 약 1.45m, 도어 안쪽 면(보강재 면)을 본다. 실측 intrinsics 는 meta.json 에 기록되며 판정은 픽셀 폭 기준이라 depth 스케일 변동(세션 ±1.4%)에 무관.

## 4. 라벨 정의

- **형상군 라벨**: 태블릿(06) 입력 8종(E23 없음)이 원본(`capture_sessions.class_name`, 불변). DB 정정 라벨(`images.class_id`)이 정본이며 정정 이력은 `notes` 에 남는다. 정정 규칙: 홀 판별기 세션 다수결이 라벨과 다르고 두 클래스 CAD D 차 ≥100mm·판정 ≥10장·일치 ≥90% 이면 자동 재배정(`db/build_dataset.py auto-relabel`, E23↔E25 FRT 등), 인접 쌍(E30↔E38 FRT 47mm)은 육안 확인.
- **레이더 옵션 라벨(세션 단위)**: `partno/radar_check.py` 가 프레임별로 브래킷 홀 2개 투영 위치의 어두운 홀 유무를 점수화(≥0.30 있음 / ≤0.12 없음)하고 세션 표(판정 ≥5장·다수 ≥80%)로 자동 플래그(`option_source='auto'`). 사용자가 웹 도구 `/options` 에서 크롭 몽타주를 보고 확정하면 `option_source='user'`(자동 재검사가 덮어쓰지 않음). 품번 정확도의 GT 는 사용자 확정값만 쓴다.
- **무효화**: 촬영 조건(설치 전 시험 촬영, 빈 지그, 작업자 유입)이 이유일 때만 `invalidate`. 판정이 틀렸다는 이유로는 무효화하지 않는다.
- **홀 랜드마크 라벨**(검출기 학습용, 별도): `labels/holes/*.json` 134장(볼트 4·코너 2 점 좌표, `tools/label_holes.py`).
- **분할**: 세션 단위만(프레임 단위 금지). 클래스별 시간순 test=max(1,15%) → val → train, 가장 이른 세션이 test 로 고정(현장 벤치마크), 세션 <3 클래스는 전부 train(평가 제외).

## 5. 수량

### 5.1 형상군 클래스별

| 클래스 | 유효 세션 | NAS 쌍 | 로컬 샘플 | 기간 | test/val/train 세션 | test 장수 |
|---|---:|---:|---:|---|---|---:|
| E23_door_LH_FRT | 6 | 468 | 120 | 20260903~20260912 | 1/1/4 | 20 |
| E25_door_LH_FRT | 3 | 447 | 40 | 20260904~20260923 | 0/0/1 (+미지정 1) | 0 |
| E25_door_LH_RR | 37 | 3,444 | 698 | 20260827~20260923 | 4/4/20 (+미지정 8) | 80 |
| E25_door_RH | 33 | 3,472 | 659 | 20260828~20260923 | 4/4/18 (+미지정 6) | 95 |
| E30_E38_door_RH | 41 | 3,777 | 797 | 20260828~20260923 | 5/5/24 (+미지정 7) | 106 |
| E30_door_LH_FRT | 16 | 1,549 | 428 | 20260827~20260923 | 2/2/9 (+미지정 2) | 169 |
| E30_door_LH_RR | 20 | 1,919 | 401 | 20260827~20260922 | 3/3/11 (+미지정 3) | 59 |
| E38_door_LH_FRT | 15 | 1,517 | 298 | 20260831~20260922 | 2/2/8 (+미지정 3) | 39 |
| E38_door_LH_RR | 19 | 1,481 | 348 | 20260828~20260922 | 2/2/10 (+미지정 5) | 28 |
| **계** | **190** | **18,074** | **3,789** | 20260827~20260923 | | |

### 5.2 품번(옵션)별 세션 — 레이더 확정값(user)과 자동 판정(auto) 구분

| 클래스 | 품번(레이더 X) | X 확정 | X 자동 | 품번(레이더 O) | O 확정 | O 자동 | 미정·미검사 |
|---|---|---:|---:|---|---:|---:|---:|
| E23_door_LH_FRT | 110982-02444 (옵션 없음) | | | | | | 6 |
| E25_door_LH_FRT | 110982-02431 (옵션 없음) | | | | | | 3 |
| E25_door_LH_RR | 110982-02427 | 0 | 21 | 110982-02375 | 0 | 15 | 1 |
| E25_door_RH | 110982-02320 | 0 | 17 | 110982-02424 | 0 | 15 | 1 |
| E30_E38_door_RH | 110982-02334 | 0 | 31 | 110982-02327 | 0 | 9 | 1 |
| E30_door_LH_FRT | 110982-02414 (옵션 없음) | | | | | | 16 |
| E30_door_LH_RR | 110982-02352 | 0 | 16 | 110982-02360 | 0 | 3 | 1 |
| E38_door_LH_FRT | 110982-02447 (옵션 없음) | | | | | | 15 |
| E38_door_LH_RR | 110982-02292 | 0 | 9 | 110982-02325 | 0 | 9 | 1 |

자동 검사 두 단서의 세션 중앙값 분포 — 레이더 O 51세션: 밝기 0.12~0.80, 판 높이 40~74mm · X 94세션: 밝기 -0.01~0.04, 판 높이 -6~37mm · 미정 3세션(프레임 1~3장·빈 지그). 판 높이 단서에서 두 사양이 겹치지 않는다(브래킷 판 CAD 70mm vs 보강재 29mm, depth 는 가장자리를 뭉개 낮게 읽힘).

### 5.3 생산계획 대조 (월 단위 비율, 1:1 매칭 아님)

생산계획(`cad/9-10월_통합모델_일별생산계획.xlsx`, 고객사 L/ON 일자 기준 210대)의 레이더 표기('레이더 X'=미장착, 무표기=장착, '확인하기'=미정)로 집계한 기종별 장착 비율. 도어 제작은 L/ON 보다 1주 이상 앞서고 카메라는 도어의 일부(주별 40~60%)만 잡으므로 세션↔개체 매칭은 하지 않는다.

| 월 | 그룹 | 개체 | 레이더 장착 | 미장착 | 미정 | 장착 비율 |
|---|---|---:|---:|---:|---:|---:|
| 2026-09 | E23 | 48 | 6 | 39 | 3 | 13% |
| 2026-09 | E25 | 5 | 2 | 3 | 0 | 40% |
| 2026-09 | E30 | 45 | 4 | 39 | 2 | 9% |
| 2026-09 | E35/38 | 25 | 5 | 20 | 0 | 20% |
| 2026-10 | E23 | 25 | 0 | 25 | 0 | 0% |
| 2026-10 | E25 | 5 | 1 | 4 | 0 | 20% |
| 2026-10 | E30 | 34 | 3 | 31 | 0 | 9% |
| 2026-10 | E35/38 | 23 | 4 | 19 | 0 | 17% |

현장 자동 판정의 레이더 비율(RR/RH 세션)은 계획 비율보다 높다(예: 9월 E23/E25 RR·RH 40%대 vs 계획 15%). 카메라가 잡은 도어의 편중 또는 표기 해석 차이일 수 있어 공장 확인 항목으로 둔다.

## 6. 평가 정의와 현재 수치

- **판정률** = 품번이 나온 세션(프레임) 비율(형상군 판정 + RR/RH 면 레이더 판정 모두 있음). **판정 정확도** = 판정된 것 중 정답 비율. 공식 수치는 고정 test 세션(형상군 벤치마크와 동일 세션), 전량은 참고(`partno/evaluate_partno.py`).
- 품번 GT = DB 정정 클래스 × 사용자 확정 레이더. 확정 전 세션은 분모에서 제외하고 수를 적는다.
- 형상군(홀 판별기) 고정 test: 596/596 = 100.0% (2026-09-23).
- CNN 폴백(rgbe_noaux_448_seed916_datasets_factory_v2) 고정 test: 595/596 = 99.8% (2026-09-16).
- 품번 정확도: 사용자 레이더 확정 후 `partno/evaluate_partno.py` 실행 시 기록.

## 7. 도구

`partno/README.md` 의 루틴: ingest/pull → `20_session_drift.py`(다수결) → `auto-relabel` → `partno/radar_check.py` → 웹 `/options` 확정 → auto-split/build → `17_evaluate_hole_classifier.py` → `partno/evaluate_partno.py` → `partno/production_plan_summary.py`. 임의 폴더 추론은 `partno/infer_partno.py`.

## 8. 한계·미확인

- AVM 옵션 미구분(E35/E38 FRT 도면은 W_AVM 1종만 입수). 생산계획의 '확인하기' 표기 개체는 미정. 레이더 표기 해석(X=미장착)은 공장 확인 전.
- 레이더 검사는 브래킷이 클램프·손에 가리면 미정. 카메라 관측면·거치 거리가 바뀌면 시차 보정값(z_plate 70mm)·탐색 폭 재확인 필요. 새 리비전에서 브래킷 위치가 바뀌면 `extract_radar_bracket.py` 재실행.
- E23 자세 CAD 는 어셈블리 STEP(110982-02444B) 메시로 2026-09-23 정식 등록(코너 홀 거리 457.6mm → CAD_D 458, 현장 정합 3.1mm, 잠정 등록 해제). E30_door_LH_RR 는 종전 메시(구 리비전, RADAR 사양) 유지 — 신규 02360K 는 홀 추출이 되는 세밀 메시(약 420MB) 재변환이 필요해 보류(`cad/door_stl/README_mesh_20260923.md`).
- 카메라는 라인 도어의 일부만 촬영하므로 수량 비율은 생산 비율과 다를 수 있다. 명성 자체 작업 실적(품번·시각)을 받으면 세션 1:1 GT 로 확장 가능.
