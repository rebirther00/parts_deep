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
- 학습/평가 이력(training_sessions, evaluation_results) 기록은 3단계에서
  02_train/03·04·13_evaluate 스크립트에 연동 예정.
