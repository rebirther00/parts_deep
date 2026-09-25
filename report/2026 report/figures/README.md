# 보고서 그림 폴더 (report/2026 report/figures)

2차년도 연차보고서 초안에 삽입된 그림. 파일을 교체·수정한 뒤 `build_annual_report_y2.py`를 다시 실행하면 문서에 반영됨(파일명 유지). 초안 작성일 2026-09-23.

| 그림 번호 | 파일 | 캡션 | 출처(원본) |
|---|---|---|---|
| 그림 1 | fig01_연구범위_공정흐름.png | 2차년도 연구 범위 — 현재 수동 공정, 자동화 전 선행 요건(본 연구), 목표 자동 공정 | report/poster_20260908/fig1_process_flow.png |
| 그림 2 | fig23_홀검출기_학습곡선.png | 홀 랜드마크 검출기 학습 곡선 | class_estimation/door_pipeline/report/hole_analysis/samples/training_curve.png |
| 그림 3 | fig02_현장수집장치_설치.jpg | 현장 무인 수집 장치 및 설치 전경 | 내부회의 보고 자료 20260910 4P 사진, 실태조사 발표자료 v2 2P 사진 |
| 그림 4 | fig03_작업자_키오스크_UI.jpg | 작업자 키오스크 UI(취득 중 화면) — 기종 선택 → 취득 시작 → 자동 저장 | 내부회의 보고 자료 20260910 4P |
| 그림 5 | fig04_현장수집_시스템_구성도.png | 현장 수집 시스템 구성도(현장 장비 → NAS 정본 → 학습 PC 메타데이터 DB) | class_estimation/door_pipeline/report/fig_20260821/arch.png |
| 그림 6 | fig05_NAS_자동업로드_흐름.png | NAS 자동 업로드 흐름(세션 종료 킥 + cron 보완, 마커 기반 재시도) | class_estimation/door_pipeline/report/fig_20260821/seq.png |
| 그림 7 | fig07_클래스별_세션_쌍수.png | 기종별 유효 세션 수·RGB-D 쌍 수(2026-08-27~09-23) | DB 조회로 생성(build_annual_report_y2.py 이전 단계) |
| 그림 8 | fig13_라벨검증_대조시트.jpg | 라벨 자동 검증 대조 시트 — (a) 미등록 기종 E23이 E25 앞문으로 입력된 세션, (b) 짧은 세션의 기종 오선택 | report/hole_analysis/field_20260907/cmp1, cmp2 |
| 그림 9 | fig08_세션단위_분할_개념도.png | 세션 단위 분할 개념도(실제 split, 8/27~9/5) — 한 세션은 통째로 train/val/test 중 한 쪽에만 배치 | report/poster_20260908/fig3_session_split.png |
| 그림 10 | fig09_DB_ERD.png | 메타데이터 DB ERD(2026-08-21 구현 기준, 이후 session_hole_metrics 추가) | class_estimation/door_pipeline/report/fig_20260821/erd.png |
| 그림 11 | fig06_수집검증체계_구성도.png | 현장 데이터 수집·검증 체계 구성도(엣지 캡처 UI → NAS 정본 → 메타 DB → 라벨 검증 루프 → 학습·평가) | report/poster_20260908/fig2_system_compact.png |
| 그림 12 | fig10_DB_웹도구_대시보드.png | DB 웹 도구 대시보드(2026-09-23 캡처) — 요약 카드·최신 벤치마크·데이터셋·기종별 수집 표 | webapp.py 화면 캡처 |
| 그림 13 | fig11_DB_웹도구_화면모음.png | DB 웹 도구 화면 — (a) 세션 목록 (b) 세션 상세 (c) 이미지 브라우저 (d) 홀 라벨 뷰 (e) 학습·평가 이력 (f) 드리프트 시계열 | webapp.py 화면 캡처 |
| 그림 14 | fig14_판별기준_홀.jpg | 판별 기준 홀 — 래치 각공 볼트홀 4점(스케일·기하 게이트) + 좌우 모서리 프레임 홀 2점(거리 D) | class_estimation/door_pipeline/report/hole_analysis/holes_def.jpg |
| 그림 15 | fig15_홀검출_현장판정예.jpg | 현장 판정 예 — 홀 6점 검출과 모서리 홀 거리 D=1,037mm → E25_door_LH_RR | 내부회의 보고 자료 20260910 9P |
| 그림 16 | fig16_미등록도어_보류_판별예.jpg | 미등록 기종 처리 — (a) E23 도어(D=461mm)는 D 범위·unknown 게이트로 보류, (b) E25 LH FRT(D=723mm) 판별 | report/poster_20260908/photo1_hole_detection.jpg |
| 그림 17 | fig18_CNN_혼동행렬_실패사례.png | 현장 재학습 CNN 결과 — (a) seed916 test 596장 혼동행렬, (b) 초기 세션 실패 사례(프로토콜 이전 촬영 조건) | artifacts/rgbe_noaux_448_seed916_datasets_factory_v2/factory_confusion_matrix.png, report/poster_20260908/fig8_failure_case.jpg |
| 그림 18 | fig25_레이더_브래킷_CAD_크롭.png | 레이더 옵션 규칙 검사 — (a) RADAR 사양 CAD의 브래킷 홀 2개 추출, (b) 레이더 O 세션 크롭(어두운 홀 2개, 초록), (c) 레이더 X 세션 크롭(빨강) | class_estimation/door_pipeline/partno/artifacts/bracket_E25_door_RH.png, radar_crops/20260828_s_073428.jpg·s_085518.jpg |
| 그림 19 | fig26_웹도구_옵션확인_화면.png | DB 웹 도구 옵션·품번 확인 화면(/options, 2026-09-23 캡처) — 자동 판정 크롭·점수·표를 보고 세션별 레이더 O/X를 확정 | webapp.py 화면 캡처 |
| 그림 20 | fig27_4채널검출기_판정샘플.jpg | 4채널 홀 랜드마크 검출기 판정 예(test·블라인드) — 볼트홀 4(파랑)·힌지 코너(빨강)·래치 코너(주황)·레이더 브래킷 홀 2(자홍), 형상군 D·마진·레이더·품번 출력 | class_estimation/door_partno/report/samples_hole4.jpg |
| 그림 21 | fig17_홀판별기_혼동행렬.png | 홀 판별기 혼동행렬 — (a) 현장 test 596장, (b) 블라인드 663장(보류 열 포함, 보류 0) | attribute_models/hole_landmarks/*_confusion_matrix.png |
| 그림 22 | fig24_홀판별기_판정_보류_샘플.png | 홀 판별기 판정·보류 샘플 — (a) 기종별 판정 성공 예, (b) 게이트 보류 예(홀 프레임 밖·미검출·기하 불일치 등) | report/hole_analysis/samples/montage_success.png, montage_abstain.png |
| 그림 23 | fig21_드리프트_시계열.png | 세션별 판정 마진 소모량 dev 시계열(185세션, 2026-08-27~09-23) — 위: 깊이 역투영 거리 기준(전환 전), 아래: 픽셀 폭 기준(전환 후). 4채널 정식 검출기 재계산 기준 |dev| p95 15.2 → 4.3mm, 경보 10 → 0 | DB session_hole_metrics(4채널 정식 모델 행)로 생성 |
| 그림 24 | fig20_CAD_홀추출.png | CAD 홀 추출 결과 — 관측면 투영에 볼트홀 4점(파랑)과 코너 홀(주황)을 표시 | pos_estimation/pos_pipeline/artifacts/cad_holes_debug/*.png |
| 그림 25 | fig19_자세추정_오버레이.jpg | 현장 프레임 자세 추정 결과 — 도어 좌표축과 추정 자세로 재투영한 CAD 홀(o) 오버레이 | 내부회의 보고 자료 20260910 11P |
| 그림 26 | fig22_자동공정_로드맵.png | 자동 공정 로드맵(계획 부분은 가안) | report/poster_20260908/fig5_roadmap.png |

- fig10·fig11: `db/webapp.py`(포트 5051 임시 인스턴스, 현재 코드) 화면을 headless Chromium으로 캡처. 대시보드 CNN 카드는 캡처 시점의 마지막 평가 기록(run #10 블라인드 84.3%)을 표시하고 있으므로 필요 시 재캡처.
- fig07·fig21: DB(door_pipeline.db) 조회로 생성한 matplotlib 차트(세션·쌍 수, 세션별 K_session·dev). 데이터가 늘면 재생성 필요.
- fig02·fig13·fig17·fig18·fig20·fig24: 원본 2장을 좌우로 합성한 그림. 원본은 출처 열 참조.
