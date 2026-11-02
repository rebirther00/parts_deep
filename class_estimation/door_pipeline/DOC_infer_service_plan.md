# 경주 공장 실시간 추론 서비스 전환 계획

작성: 2026-11-02. 수집 서비스(`06_factory_capture.py`, 작업자 버튼 라벨링)를 **무인 실시간 추론 서비스**로 교체한다.
작업자 조작·터치모니터 없이 도어가 지그에 올라오면 자동으로 클래스·레이더·품번을 판정하고, 판정 프레임을 저장해
NAS → DB → 대시보드로 올린다. 개발자가 매일 웹으로 확인한다.

## 1. 현황 (2026-11-02 확인)

| 항목 | 상태 |
|---|---|
| 수집 서비스 | `door-capture.service` **disable 완료**(2026-11-02), 키오스크(`door-kiosk.desktop`) 프로세스 종료. 추론 서비스가 이 자리를 대체 |
| 추론 코드 | `18_realtime_inference_hole.py` — 홀 판별기(4채널 브래킷 모델) + CNN 폴백 + 레이더/품번 구현 완료. 도어 교체는 수동 `/api/reset` 뿐 |
| Jetson 실측 | ZED Box(Orin NX 16GB, zed_env torch 2.8 CUDA): 홀 판별기 **117ms/프레임(FP16)**·135ms(FP32), CNN 46ms, GPU 메모리 342MB → TensorRT 없이 약 8fps |
| 카메라 | ZED X Mini S/N 54910212, GMSL, SDK 인식 정상 |
| 현장 리듬 | 397세션(44일) 기준 하루 9도어, 도어 간 간격 중앙값 33분(p10 22·p90 102), 촬영 2.7분/도어, 세션당 약 600MB |
| NAS 업로드 | 세션당 약 12분(LTE ≈0.8MB/s). 로컬 디스크 여유 38GB, `datasets_factory_collect/` 141GB |
| DB·대시보드 | SQLite·webapp은 학습 PC(5090)에만 있음. 흐름: 박스 → NAS(rclone) → `ingest_nas.py` → DB → webapp |
| 미해결 | 빈 지그를 CNN이 도어로 오판(README) — 도어 유무 게이트 없음 |

## 2. 운영 전제 (사용자 결정, 2026-11-02)

1. 용접은 **사람**이 지그 앞에서 한다. 우리 기종 외 부품(Unknown)도 같은 지그에서 작업한다.
2. 도어 사이에 지그는 비운다(교체). 재작업으로 뺐다 다시 올리는 경우는 없다. 도어 주기 30분~1시간.
3. **판정 프레임만 저장**한다. NAS 업로드 완료분은 로컬에서 지워도 된다.
4. 보는 사람은 개발자 혼자. 터치모니터에는 올리지 않는다. 박스 `:5000` = 당일 실시간 진행, 학습 PC webapp = 전체 통계.
5. 작업자 라벨은 없다. **모델 판정이 잠정값**이고, 판정 대표 이미지 옆에 기존 작업자 라벨 데이터의 클래스 대표 이미지를 나란히 놓아 개발자가 육안 비교·확정한다.
6. 전원은 작업자 또는 개발자가 OFF. 안 되면 **저녁 20:00 자동 종료**.
7. 출력은 클래스 + 레이더 + 품번 (6DoF 자세는 제외).
8. 알림은 대시보드 표시만 (푸시 없음).

## 3. 설계

### 3.1 도어 생애주기 (상태기계) — 작업자 버튼 대체

```
EMPTY ─점유 감지─▶ OCCUPIED/JUDGING ─판정 확정─▶ JUDGED(잠금, 이탈 감시만)
  ▲                      │ 10분 내 미확정 → UNKNOWN / UNRECOGNIZED                │
  └──────── 이탈 20초 지속 → 세션 마감(result.json 확정, NAS 킥) ◀────────────────┘
```

- 판정은 적재 직후 사람이 물러난 **정지 구간**에서 끝난다. 8fps·게이트 통과 3프레임이면 1~2초 → 용접 시작 전 확정.
- 확정 뒤에는 추론을 멈추고 이탈 감시만 한다(용접 중 사람·스파크는 판정에 무관).
- 미확정 분기: 홀 6점은 잡히는데 D가 CAD와 `UNKNOWN_MM`(30mm) 넘게 어긋나면 `unknown`(미등록 도어),
  6점 자체가 안 잡히면 `unrecognized`(우리 기종 외 부품). 둘 다 대표 프레임 5장 저장 → 두 화면에 표시.
- 같은 클래스 도어가 연달아 와도 빈 지그 구간으로 세션이 갈린다.

### 3.2 가림·정지 게이트 (새 모델 없음)

프레임이 다음을 **모두** 통과해야 판정 윈도에 들어가고 저장된다.

| 게이트 | 내용 | 출처 |
|---|---|---|
| 기하 | 6점 검출·공선·경계·`D_RANGE`·`pose_warn`(z 1400~1500, tilt 15°) | 기존 `hole_classifier` |
| 정지 | 도어 ROI 프레임 간 차분이 K프레임 연속 임계 이하 | 신규 |
| 침입 | 도어 평면보다 150mm 이상 앞의 depth 픽셀이 ROI의 일정 비율 초과 → 가림 | 신규 |
| CNN 폴백 조건 | "도어는 보이는데 기하 실패"인 프레임에서만 실행 (빈 지그 오판 차단) | 변경 |

### 3.3 저장·업로드·정리

- 새 루트 `datasets_factory_infer/<YYYYMMDD>/s_HHMMSS/` — 작업자 라벨 트리(`datasets_factory_collect/`)와 **분리**. NAS도 별도 폴더(`nas:Guest/weld_infer`).
  - `rgb_NNNN.png`/`depth_NNNN.png`: 판정에 쓰인 게이트 통과 프레임 최대 10쌍 (미확정 세션은 대표 5장)
  - `result.json`: 상태(judged/unknown/unrecognized/abstain)·클래스·품번·레이더·D·마진·신뢰도·n_judged·게이트 분포·source(hole/cnn)·conflict·모델 경로/해시·시각·체류 시간
  - `thumb.jpg`: 대표 프레임(점 오버레이) 썸네일
  - `meta.json`: 카메라·intrinsics·서비스 버전 (06과 동일 필드 유지 → ingest 재사용)
- 도어당 약 60MB(기존 600MB) → LTE 업로드 1~2분. `nas_sync.sh`는 루트·목적지만 바꿔 재사용, `.uploaded` 마커 방식 유지.
- 보존: `.uploaded` 후 3일 지나면 png 삭제, `thumb.jpg`·`result.json`은 90일 보관(`:5000` 이력용). 일별 `events.jsonl`(append 전용)에 상태 전이·판정을 기록 → 전원 급차단에도 안전.
- `result.json`은 임시 파일 → rename 으로 원자적 기록.

### 3.4 화면 2개

**박스 `:5000` (당일 실시간)** — 템플릿 `factory_infer.html`
- 오늘 도어 목록: 시각·체류·상태·클래스·품번·레이더·신뢰도·D/마진.
- 각 행: **판정 대표 이미지 | 판정 클래스의 작업자 라벨 대표 이미지 | 2순위 후보 대표 이미지** (FRT 41mm 간격 혼동을 눈으로 가림).
- 라이브 프리뷰(점 오버레이)·상태기계 상태·카메라/GPU/디스크/마지막 업로드·재판정(`/api/reset`)·원격 종료(개발자용, 기존 sudoers).
- 로컬 jsonl/result.json 로 서비스 → NAS 불통이어도 당일 확인 가능.

**학습 PC webapp (전체 통계·확정)**
- `ingest_nas.py`(또는 `ingest_infer.py`)가 `result.json`을 읽어 `capture_sessions`(dataset `door_factory_infer`, class_name=판정, option_source='auto') + 신규 `inference_results` 에 등록.
- `/inference`: 날짜별 목록 → 세션 상세(판정 프레임·비교 이미지·후보 표) → **확정/정정 버튼**(`confirmed_class/confirmed_radar/confirmed_by/at`). 확정값이 재학습 GT.
- 대시보드 타일: 오늘 도어 수, 보류·unknown·unrecognized 수, 마지막 판정 시각, **서비스 하트비트**(박스가 10분마다 NAS에 올리는 파일 시각).

**클래스 대표 이미지**: 기존 라벨 데이터(`datasets_factory_v2` test 분할, 게이트 통과 프레임)에서 클래스×레이더 14장을 골라
`reference_images/<class>__radar{0,1}.jpg`(작은 jpg)로 저장소에 넣어 두 PC가 같은 것을 쓴다.

### 3.5 DB 스키마 초안 (`inference_results`)

```sql
CREATE TABLE IF NOT EXISTS inference_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER NOT NULL UNIQUE REFERENCES capture_sessions(id) ON DELETE CASCADE,
    status          VARCHAR(16) NOT NULL CHECK (status IN ('judged','unknown','unrecognized','abstain')),
    pred_class      VARCHAR(100),  pred_group VARCHAR(8),
    part_no         VARCHAR(30),   radar INTEGER CHECK (radar IN (0,1)),
    source          VARCHAR(8),    conflict BOOLEAN,
    confidence      REAL, d_mm REAL, margin_mm REAL, n_judged INTEGER, n_frames INTEGER,
    gate_counts     TEXT,          candidates TEXT,          -- json
    model_hole      VARCHAR(200),  model_cnn VARCHAR(200),
    dwell_s         REAL,          judged_at TIMESTAMP,
    confirmed_class VARCHAR(100),  confirmed_radar INTEGER,
    confirmed_by    VARCHAR(30),   confirmed_at TIMESTAMP,  note TEXT,
    created_at      TIMESTAMP NOT NULL DEFAULT (datetime('now','localtime'))
);
```

### 3.6 서비스·전원

- `deploy/door-infer.service` 가 `door-capture.service` 를 대체. `ExecStart=zed_env/bin/python 07_factory_infer.py`.
  시계 동기 대기(ExecStartPre)·CUDA 프로브·`Restart=always`·`StartLimitIntervalSec=0` 은 기존 유닛 그대로.
  `TimeoutStopSec=60`, SIGTERM 시 현재 세션 마감 + NAS 킥.
- `install_factory.sh` → `install_infer.sh`: 서비스 등록, 키오스크 autostart **제거**, 크론(nas_sync 5분 + **20:00 자동 종료** `sudo -n /usr/sbin/shutdown -h now`, 기존 sudoers 규칙으로 user 크론에서 실행 가능).
- 작업자가 20:00 전에 멀티탭을 내려도 jsonl·원자적 쓰기·업로드 성공 후 마커 구조라 데이터는 안전. 아침 전원 ON 으로 자동 기동.

## 4. 위험과 투입 방식

- **점유 감지**(구현에서 확정, 2026-11-02): 지그는 평판+레일+클램프로 **도어 종류·날짜와 무관하게 고정**이고 "교체"는 도어를 말한다.
  빈 지그 평판이 카메라 정면 ~1475mm, 도어 면은 그 위 ~33mm 라 절대 z 나 "돌출 픽셀 비율"(지그 레일·클램프가 40mm+ 돌출, 빈 지그 0.18 vs 도어 0.24)로는
  구분이 안 됐다. **빈 지그 배경 기준맵(depth 1/4) 대비 25mm 넘게 다른 픽셀 비율 `diff`**는 빈 지그 0.015 vs 도어 0.68~0.74 로 극명 →
  채택. 시드 `reference_images/empty_jig_depth_small.png`(8/27 9장 중앙값), EMPTY 에서 EMA(τ 5분) 추종, UNRESOLVED 에서 느린 흡수(τ 20분, 기종 외 부품 세션 종료용), `background.npz` 보존.
- **침입 게이트**: 도어 ROI 안에서 배경보다 250mm 이상 가까운 픽셀 비율 — 도어만 0.007~0.016, 작업자 유입 0.033~0.062 → 임계 0.025.
  (150mm 기준은 클램프 열림/닫힘 차이로 도어만 있어도 0.03 이 나와 폐기. 전체 프레임 기준은 현장 첫 가동에서 ROI 밖 우측 토치 케이블·장비가 0.10 을 내 판정을 막아 폐기 → 진단값 `intr_full` 로만 기록)
- **그림자 모드** 투입: 첫 며칠은 상태 전이·지표(`metrics_<날짜>.jsonl`)·판정을 기록만 하고 개발자가 `:5000`에서 확인하며 임계값을 맞춘 뒤 NAS·DB 연동(`DOOR_INFER_UPLOAD=1`)을 켠다.
- 로컬 디스크: 기존 `datasets_factory_collect/` 141GB 는 전량 NAS 업로드 확인 후 정리(별도 결정).

## 5. 구현 순서·현황

1. ~~박스: `07_factory_infer.py`(상태기계·게이트·저장) + `templates/factory_infer.html` + `:5000` API — 그림자 모드~~ → **완료 2026-11-02**
   (리플레이 `--replay "빈지그:20,도어:45,빈지그:25,작업자:40,…"` 로 EMPTY→JUDGING→JUDGED→EMPTY 전이·보류·시한 초과 검증)
2. ~~박스: `deploy/door-infer.service`·`install_infer.sh`·크론(nas_sync_infer 5분, 20:00 종료), 보존 정리~~ → **완료 2026-11-02** (보존 정리는 서비스 내부 시간당 1회)
3. ~~학습 PC: `inference_results` 마이그레이션, ingest 확장, webapp `/inference`·대시보드 타일·확정 버튼~~ → **완료 2026-11-02** (`db/migrate_inference.py`, `db/ingest_infer.py`, webapp `/inference`·`/inference/<sid>`·확정 POST·대시보드 "현장 판정 서비스" 카드; 임시 DB 로 인제스트→열람→확정→재인제스트 보존 검증)
4. ~~`reference_images/` 클래스×레이더 대표 이미지~~ → **완료 2026-11-02** (`tools/make_reference_images.py`, 14장 + 점 오버레이본, 출처 `manifest.json`)
5. 현장 설치(`sudo bash deploy/install_infer.sh`) → 그림자 모드 며칠 → 임계값 확정 → `DOOR_INFER_UPLOAD=1` → NAS·DB 연동 ON

## 6. 관련 문서

- 수집 서비스 운영: `DOC_factory_collect.md` (서비스 중지·키오스크 종료 절차 포함)
- DB 흐름·TODO 9 "엣지 자동 판정 루프": `db/README.md`
- 외부 열람(funnel): `DOC_external_access.md`
