# door_pipeline 데이터셋 메타데이터 DB

`report/DBMS_SCHEMA_DESIGN.md`의 1단계(SQLite) 구현. 이미지 파일은 파일시스템/NAS에
그대로 두고, 메타데이터(경로·세션·수량)만 DB에 넣는 하이브리드 구조다.

- 정본 데이터: 세종 NAS `nas:Guest/weld` (06_factory_capture.py → nas_sync.sh 업로드)
- 이 DB: 학습 PC(5090)에서 "무엇이 어디에 얼마나 있는지"를 조회하는 인덱스

## 파일

| 파일 | 역할 |
|------|------|
| `schema.sql` | 테이블/인덱스 정의 (멱등, `IF NOT EXISTS`) |
| `ingest_local.py` | 로컬 `datasets*/` 스캔 → datasets/classes/images 등록 |
| `ingest_nas.py` | NAS 세션 트리 스캔 → capture_sessions/images 등록 (`synced_local=FALSE`) |
| `db_log.py` | 학습·평가 스크립트 → DB 자동 기록 헬퍼 (3단계) |
| `webapp.py` | 열람·관리 웹 도구 — 대시보드/세션/이미지 썸네일/홀 라벨 뷰(점 오버레이)/학습 이력 + 라벨정정·split·무효화 버튼 |
| `door_pipeline.db` | SQLite DB 본체 (git 미추적) |

## 사용

```bash
# 로컬 데이터셋 재스캔 (멱등 — 새 파일만 추가됨)
python db/ingest_local.py

# NAS 스캔 — rclone 리모트 설정 후
python db/ingest_nas.py --remote nas:Guest/weld
# 또는 NAS 마운트/복사본 경로로
python db/ingest_nas.py --local-dir /mnt/nas/weld

# 브라우저로 열람·관리 (권장) — http://localhost:5050
python db/webapp.py                # --readonly 로 열람 전용, --port 로 포트 변경
# SQL로 직접 보고 싶을 때 (선택)
pipx run sqlite-web db/door_pipeline.db     # 또는: pipx run datasette db/door_pipeline.db
```

## 주의

- 라벨 정정은 파일을 옮기지 말고 DB에서: `capture_sessions.class_name`은 현장 입력
  원본이고, 정정 이력은 `classes`/`images` 재배정 + `notes`에 남긴다.
  (예: 대차 세션 #12~66 → E30_E38_door_RH 정정 사례)
- `door_aug`/`door_aug2`는 강건성 평가 전용 — 학습에 사용 금지 (description에 명시).
## 학습·평가 자동 기록 (3단계, 2026-08-25)

02/03/04/12/13 스크립트가 `db/db_log.py`를 통해 실행 시 자동으로 기록한다.
별도 조작 불필요. DB 오류는 경고만 남기고 학습/평가는 계속된다.

| 스크립트 | 기록 테이블 | 비고 |
|---|---|---|
| `02_train.py` | models, training_sessions, training_metrics(에폭별) | status: running→completed/stopped(Ctrl+C)/failed |
| `03_evaluate.py` | evaluation_results | original→`in_domain`(door_real), aug/aug2→`cross_domain` |
| `04_evaluate_factory.py` | evaluation_results | `cross_domain`(door_factory) |
| `12_train_vent_unet.py` | models, training_sessions, training_metrics | `val_accuracy` 컬럼 = val IoU(%) |
| `13_evaluate_attribute_pipeline.py` | evaluation_results | `inference_pipeline`, accuracy=class_acc, per_class_results에 group_acc |

- 정확도류는 JSON 산출물과 같은 백분율(0~100).
- 평가 행의 `session_id`는 같은 모델의 최신 training_sessions에 자동 연결
  (DB 도입 전에 학습된 모델은 NULL).
- 비활성: `DOOR_DB_LOG=0 python 02_train.py ...` / 다른 DB: `DOOR_DB_PATH=...`
- 조회 예:
  ```sql
  SELECT m.name, d.name AS dataset, e.eval_type, e.accuracy, e.f1_macro, e.evaluated_at
    FROM evaluation_results e JOIN models m ON m.id=e.model_id
    JOIN datasets d ON d.id=e.dataset_id ORDER BY e.evaluated_at DESC;
  SELECT epoch, train_loss, val_loss, val_accuracy FROM training_metrics
   WHERE session_id = ? ORDER BY epoch;   -- 학습 곡선
  ```

## 시각 기준 (2026-08-28)

- DB의 모든 TIMESTAMP는 **한국 시간(KST)** 이다. 스키마 기본값은 `datetime('now','localtime')`,
  코드에서 `CURRENT_TIMESTAMP`(UTC) 사용 금지.
- 2026-08-28 이전에는 `datasets.created_at/updated_at`, `models.created_at`만 UTC로 기록되어 있어
  `db/migrate_kst.py`로 +9시간 보정·재생성했다 (`PRAGMA user_version=1`, 원본은 `*.bak_utc_*`로 보존).

## 스키마 버전 (PRAGMA user_version)

| 버전 | 마이그레이션 | 내용 |
|---|---|---|
| 1 | `migrate_kst.py` (2026-08-28) | UTC→KST 시각 보정 |
| 2 | `migrate_eval_types.py` (2026-08-31) | `eval_type`에 `pose_pipeline` 추가 — pos_pipeline/03의 자세 평가 기록이 CHECK 위반으로 조용히 유실되던 문제 수정. 기존 현장 평가 1건은 JSON에서 소급 기록 |

## 현장 데이터 활용 흐름 (2026-08-28)

NAS(전량, 정본) ⊃ `datasets_factory_collect/`(미러: NAS와 같은 `날짜/클래스/s_*/` 구조, **세션당 20장 샘플만**)
← `datasets_factory_v2/`(학습·평가용 **심볼릭 링크 뷰**, 파일 없음). 2초 간격 연속 프레임은 거의 중복이라
학습엔 샘플이면 충분하고, 전량은 NAS에서 필요 시 `--all`로 받는다.

```bash
python db/ingest_nas.py [--refresh]        # ① NAS 세션 → DB 등록 (메타만). 수집 중 등록된 세션은 --refresh로 수량 갱신
python db/pull_nas.py [--date YYYYMMDD]    # ② 세션당 20장 고르게 pull → 미러, synced_local 갱신
python db/build_dataset.py status          # ③ 세션별 라벨·split 현황
python db/build_dataset.py relabel <session_dir> <class>   #    라벨 정정/Unknown 확정 (DB만, 파일 이동 없음)
python db/build_dataset.py auto-split      # ④ 클래스별 첫 세션=test, 나머지=train (미지정 세션만)
python db/build_dataset.py split <session_dir> test|train|val|none
python db/build_dataset.py build           # ⑤ datasets_factory_v2/{all,train,val,test}/<class>/rgb_<날짜>_<세션>_<idx>.png + manifest.json
python 17_evaluate_hole_classifier.py --base datasets_factory_v2/test   # ⑥ 평가 (DB 자동 기록)
python 02_train.py --model_type rgbe --no_aux --image_size 448 --dataset_dir datasets_factory_v2 --presplit
                                           # ⑦ CNN fine-tune 시: train/val/test 폴더 그대로 사용(세션 격리), run 이름에 _datasets_factory_v2
```

- split은 **세션 단위**만 허용(프레임 단위 분할 금지). 클래스별 첫 확보 세션을 test로 고정해 현장 벤치마크를 유지한다.
- `Unknown` 세션·`is_valid=0`·미동기화 이미지는 뷰에서 제외된다.
- `datasets_field/`, `datasets_factory/`(2026-04 ZED 2i)는 레거시 — 신규 작업은 v2 뷰를 쓴다.

## 다음 작업 (TODO, 2026-09-01 분류·추정 전체 검토로 갱신)

우선순위 순. 완료 시 줄을 지우거나 취소선 처리.

### 바로 실행 가능 (막힌 데 없음)

1. **E38_LH_FRT 현장 자세 평가** — 8/31 저녁 5세션 648쌍 수집 완료로 차단 해제됨
   (기존 "수집 대기" 문구는 낡음). `pos_pipeline/03_evaluate_field` 실행 → 자세 평가 7종으로 확대.
2. **05_evaluate_tracker_gt.py에 db_log 연동** — 레이저 트래커 실측 전에 필수.
   03처럼 eval_type='pose_pipeline'으로 기록 (04 합성 검증은 일회성이라 생략 가능).
3. **라벨 검수** — webapp 라벨 탭에서. 재학습·K_DEPTH 재보정의 선행 작업:
   - 미기입 점 라벨 31건 확인. 특히 6점 중 5점 미기입 2건
     (`E30_E38_door_RH__0030`, `E30_door_LH_FRT__0050`)은 재라벨 또는 제외 결정.
   - 힌지/래치 스왑 오라벨 `E25_door_LH_FRT__0000` 재라벨 (15_label_holes.py).
4. **9/1 이후 유입 세션 pull+평가 루틴 유지** — 발표 수치는 8/31 동결이지만 내부 검증은 세션마다 계속.

### 단기 (수집 유입·일정에 의존)

5. **E25_door_LH_FRT 수집 → 8종 검증 완결** — 생산 순서 대기. 유입 즉시 평가하고,
   이 시점에 세션 단위 분할로 현장 공식 인식률 수치 산출(1차년도 분류의 종착점).
6. **Unknown 4세션 61쌍 라벨 확정** — 홀 판별기 판정 참조로 빠르게 정리 가능.
7. **홀 판별기 거리 정밀화** — K_DEPTH 상수를 실측 intrinsics 기반으로 재유도(자세 쪽은 반영 완료).
   FRT 3종 간 마진 41~47mm 대비 오차 20~58mm 구간의 오판 여지 축소가 목적.
8. **레이저 트래커 실측 일정·장비 수배** — 프로토콜·양식·selftest 준비 완료
   (`pos_pipeline/GT_TRACKER_PROTOCOL.md`). KPI(위치 3mm) 최종 입증 수단으로 최중요 미결.
   9/3 현장 방문 때 측정 여건(도어 고정·공간) 확인.

### 중기 (구조 작업)

9. **엣지 자동 판정 루프** — 홀 검출기 TensorRT/FP16을 ZED Box 배포, 세션 종료 시
   자동 판정→meta/DB 기록. 현장 서버 DB 직접 연동(DBMS 2단계)과 한 묶음(R1 공표 계획).
10. **DB 전문가 미팅 준비** — 브리핑 문서 §11 빈칸(예산·일정·인원·보안 규정) 직접 기입,
    문서 공유 링크 전달. 미팅 안건: webapp 정식화 vs 기성 도구, SQLite→PostgreSQL 시점,
    스키마 리뷰 Q5 (a)~(g).

### 방향 결정 대기 (착수 전 판단 필요)

- **CNN(RGBE) 현장 혼합 재학습 지속 여부** — 홀 판별기가 주 판별기(현장 100%)인 지금,
  보조 판별기 재학습의 우선순위 재평가. 8종 완결·트래커 실측 전까지 보류 권고(2026-09-01).
  **auto-split 이미지 수 가중 개선**은 이 결정에 종속 — 현재 세션 개수 기준이라 클래스별
  이미지 비율(70/15/15)이 왜곡됨(예: E30_door_LH_FRT는 세션 1개 = 전량 test).
  결정 전까지 신규 세션 split 미지정 유지가 정상.
- **U-Net 속성 파이프라인의 지위** — 발표 자료에서는 제외했으나 코드에는 하이브리드
  폴백으로 잔존. 유지보수 범위(폴백 유지 vs 정리) 결정.
- **2차년도 확장 구체화** — 로봇 연계 파이프라인, 도장 라인 확장(R1 공표 항목).

메모: 08-31 수집 세션 4개(E25_RH 2·E38_LH_RR 2)는 pull·평가 완료, split은 재학습 노선 결정 전까지 미지정 유지.
