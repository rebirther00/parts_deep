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
| `door_pipeline.db` | SQLite DB 본체 (git 미추적) |

## 사용

```bash
# 로컬 데이터셋 재스캔 (멱등 — 새 파일만 추가됨)
python db/ingest_local.py

# NAS 스캔 — rclone 리모트 설정 후
python db/ingest_nas.py --remote nas:Guest/weld
# 또는 NAS 마운트/복사본 경로로
python db/ingest_nas.py --local-dir /mnt/nas/weld

# 브라우저로 열람 (선택)
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
