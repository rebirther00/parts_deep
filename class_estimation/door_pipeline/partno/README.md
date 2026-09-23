# partno/ — 품번(14) 판정 확장 (2026-09-23)

도어 9종 형상군(홀 판별기, 폭 D) 위에 **RADAR 옵션 유무**를 얹어 통합모델 SIDE DOOR **어셈블리 품번 14종**을 판정한다.
새로 학습하는 모델은 없다. 기존 홀 랜드마크 검출기(`attribute_models/hole_landmarks/model.pth`)가 찾는 6점으로
도어 좌표계를 만들고, CAD 에서 뽑은 브래킷 홀 2개 위치를 투영해 홀 유무를 규칙으로 판정한다(AI 검출 + CAD 기하 규칙).

```
품번 = 형상군 클래스(9, hole_classifier.CAD_D) × 레이더(1/0)      → part_numbers.json
FRT 3 위치(E23·E25·E30·E35/38 LH FRT)는 옵션 없음 → 클래스만으로 품번 확정
LH REAR 3 폭 × RADAR 유무 = 6,  RH 2 폭 × RADAR 유무 = 4          (E23/E25 는 REAR·RH 공용, E30/E35/E38 은 RH 공용)
```

## 파일

| 파일 | 역할 |
|---|---|
| `part_numbers.json` | 14 품번표 — 키 = 리비전 제외 기본 품번(`110982-02424`), rev·적용 기종·위치·radar·사양표 폭·외판/MID 품번·도면/STEP 파일명 |
| `radar_bracket.json` | RR/RH 5 클래스의 브래킷 홀 2개 도어 프레임 좌표(mm), 반지름 17·간격 91.6·판 사각·판 높이 z≈70 — `extract_radar_bracket.py` 출력 |
| `extract_radar_bracket.py` | RADAR STEP→STL 메시에서 브래킷 홀 좌표 추출(카메라 쪽 최대 z 지도의 빈 원형 영역 쌍). 디버그 `artifacts/bracket_<class>.png` |
| `radar_check.py` | **규칙 검사 도구** — DB 세션 전량 검사 → `session_radar_check` + `capture_sessions.option_*`(auto) + 몽타주 `artifacts/radar_crops/` |
| `infer_partno.py` | 임의 폴더 추론(형상군·D·레이더·품번), 세션/프레임 표, JSON |
| `evaluate_partno.py` | 품번 정확도(판정률·판정 정확도) — 세션 단위 전량/고정 test, 프레임 단위 고정 test → `eval_partno.md/.json` + DB |
| `production_plan_summary.py` | 생산계획 xlsx 월별 집계(기종×레이더, 클래스별 기대 도어 수 vs 현장 세션 비율) → `production_plan_summary.md` |

DB 쪽(기존 파일 수정): `db/migrate_options.py`(옵션 컬럼·검사 표, 멱등), `db/build_dataset.py auto-relabel`(홀 판정 다수결로 명백한 오라벨 자동 재배정),
`db/webapp.py` **/options** 페이지(브래킷 크롭 보고 레이더 O/X 확정 → 품번 표시), `20_session_drift.py`(세션 다수결 `pred_major` 기록).

## 터미널 사용 순서 (신규 세션 유입 시 루틴)

```bash
cd class_estimation/door_pipeline
PY=../../venv/parts_deep/bin/python
$PY db/ingest_nas.py && $PY db/pull_nas.py               # ① NAS 세션 등록·샘플 20장 pull
$PY 20_session_drift.py                                   # ② 세션별 홀 판정 다수결(pred_major)·드리프트 지표
$PY db/build_dataset.py auto-relabel [--dry-run]          # ③ 태블릿 오라벨 자동 재배정 (E23↔E25 FRT 등 CAD D 차 ≥100mm·표 10장·일치 90%)
$PY 20_session_drift.py                                   #    (재배정 세션은 지표 재계산)
$PY partno/radar_check.py                                 # ④ RR/RH 세션 레이더 규칙 검사 → 자동 플래그 + 몽타주
$PY db/webapp.py                                          # ⑤ http://localhost:5050/options 에서 육안 확인·확정(user)
$PY db/build_dataset.py auto-split && $PY db/build_dataset.py build   # ⑥ split·뷰
$PY 17_evaluate_hole_classifier.py --base datasets_factory_v2/test    # ⑦ 형상군 벤치마크(고정 test)
$PY partno/evaluate_partno.py                             # ⑧ 품번 정확도 (GT = DB 클래스 × 사용자 확정 레이더)
$PY partno/production_plan_summary.py                     # ⑨ 생산계획 집계 대조(월 단위 비율)
```

단발 명령:

```bash
$PY partno/radar_check.py --session 20260914/E25_door_RH/s_124221 --dry-run   # 특정 세션 점수·몽타주만
$PY partno/radar_check.py --recompute                                          # 임계값·브래킷 좌표 변경 후 전량 재검사(사용자 확정은 유지)
$PY partno/radar_check.py --print                                              # 현재 표
$PY partno/radar_check.py --accept-auto                                        # /options 육안 확인을 마쳤으면 자동 판정을 사용자 확정으로 일괄 승인(미정 제외)
$PY partno/radar_check.py --recompute --cls E30_door_LH_RR                     # 브래킷 좌표 갱신 후 해당 클래스만 재검사
$PY partno/infer_partno.py datasets_factory_collect/20260914/E25_door_RH/s_124221 --frames   # 폴더 추론
$PY partno/infer_partno.py datasets_factory_v2/test --recursive --json /tmp/partno_test.json
$PY partno/extract_radar_bracket.py --stl E30_door_LH_RR=cad/door_stl/E30_door_LH_RR_02360K.stl   # 새 CAD 로 브래킷 좌표 갱신
$PY partno/evaluate_partno.py --gt auto                                        # 사용자 확정 전 자동 판정 일치도(공식 수치 아님)
```

## 판정 규칙 요약

- 프레임: 6점 검출·기하 게이트 통과 → 6점↔도어 XY 최소자승 아핀(잔차 ≤12px) → 브래킷 홀 2개 투영(판 높이 70mm 시차 보정, depth 평면 깊이 사용)
  → 밝기 점수 = (고리 밝기 − 홀 안 밝기)/고리 밝기, 두 홀 공통 오프셋 ±12px 탐색, min(홀1, 홀2)
  → 판 높이 h = 홀 주변 고리 depth 가 6홀 평면보다 카메라 쪽인 거리(브래킷 판 실측 50~56mm, 보강재 13~19mm; 조명 무관)
  → 점수 ≥ 0.30 → 레이더. 점수 ≥ 0.08 이고 h ≥ 37(판 높이) → 레이더. 점수 ≤ 0.12 이고 h < 49 → 없음. 점수 ≤ 0.04(홀 흔적 없음) → 없음. 나머지 미정. depth 없으면 점수만(0.30/0.12). 판이 프레임 밖이면 미판정.
- 세션: 판정 프레임 ≥3 이고 다수 표 ≥80% 일 때만 자동 플래그(1/0), 아니면 미정(NULL). 사용자 확정(user)은 자동 재검사가 덮어쓰지 않는다.
- 품번 정확도: 판정률(품번이 나온 비율)과 판정 정확도(판정 중 정답)를 따로 보고. 공식 수치는 고정 test 세션(형상군 벤치마크와 동일), 전량은 참고.
- 자동 재배정: 홀 판정 다수결 ≠ DB 라벨이고 두 클래스 CAD D 차 ≥100mm·판정 ≥10장·일치 ≥90% 일 때만 자동(이력 notes 기록), E30↔E38 FRT(47mm) 같은 인접 쌍은 확인 목록만.

## 근거·한계

- RADAR 사양 차이 = 외판(W_RADAR) + MID 보강재(브래킷 판, Ø34 홀 2개, 간격 91.6mm) + COVER;RADAR. FR/RR 세로 보강재(코너 홀)·볼트 사각·힌지는 공용 → 폭 D·자세 랜드마크는 RADAR 불변.
- 기존 `cad/door_stp` RR/RH 5종은 모두 RADAR 사양(브래킷 있음)이었고, 현장 라벨 안에는 RADAR/비RADAR 가 섞여 있다(2026-09-23 확인).
- 브래킷이 클램프·손에 가리면 미판정(미정) → 다른 프레임으로 보완. 카메라 관측면이 바뀌거나 지그 거리가 달라지면 시차 보정값(z_plate)·탐색 폭 재확인.
- AVM 옵션은 구분하지 않는다(E35/E38 FRT 도면은 W_AVM 1종만 입수). 생산계획의 '확인하기' 표기 개체는 미정.
