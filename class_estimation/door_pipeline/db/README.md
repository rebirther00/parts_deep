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
