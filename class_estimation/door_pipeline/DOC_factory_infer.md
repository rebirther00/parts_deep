# 경주 공장 도어 실시간 판정 — 운영 가이드 (추론 서비스)

작성: 2026-11-02. 수집 서비스(`DOC_factory_collect.md`, 2026-11-02 중지)를 대체하는 **무인 판정 서비스**.
장비: ZED X mini + Zed Box(zedbox-weld, Jetson Orin NX) + LTE 라우터. 터치모니터는 쓰지 않는다.
설계·결정 사항: `DOC_infer_service_plan.md`. 코드 요약: `README.md` §E.

## 구성 요소

| 파일 | 역할 |
|---|---|
| `07_factory_infer.py` | 판정 서비스 (Flask, 포트 5000). 상태기계·게이트·판정·저장·보존 정리·하트비트 |
| `factory_camera.py` | ZED 상시 구동 카메라 (06 에서 분리, 자동 재연결·CUDA 프로브) |
| `templates/factory_infer.html` | 개발자용 웹 페이지 (라이브·지표 차트·현재 세션·오늘 도어 목록·재판정·원격 종료) |
| `reference_images/` | 클래스×레이더 대표 이미지 14장(`tools/make_reference_images.py`) + 빈 지그 배경 시드 `empty_jig_depth_small.png` |
| `deploy/door-infer.service` | systemd 유닛 (부팅 자동시작, 죽으면 5초 후 재시작, door-capture 와 Conflicts) |
| `deploy/install_infer.sh` | 설치 스크립트 (`sudo bash deploy/install_infer.sh` 1회) |
| `deploy/nas_sync_infer.sh` | NAS 업로드 (`nas:Guest/weld_infer`), 세션 마감 킥 + cron 5분 |
| `deploy/infer_ctl.sh` | sudo 없이 수동 제어 `start|stop|restart|status` (systemd 설치 전 임시 운영용 — 2026-11-02 현장 첫 가동은 이걸로 그림자 모드 기동) |
| `~/.config/door-infer.env` | 운영 설정 (그림자 모드/업로드, 임계값). 유닛을 고치지 않고 여기서 바꾼다 |

## 동작 사양

- 도어 생애주기: 빈 지그 → 점유(3초 지속) → **판정 중** → 확정(잠금) → 이탈(20초 지속) → 세션 마감·업로드.
  점유 후 10분 안에 못 잠그면 미확정(`unknown` 미등록 도어 / `unrecognized` 기종 외 부품 / `abstain` 도어는 봤으나 게이트 실패).
- 판정에 쓰는 프레임: 정지(움직임 없음 3프레임) · 침입 없음(작업자·물체) · 홀 기하 게이트 통과. 작업자가 용접 중이면 그 프레임은 버린다.
- 확정 뒤에는 추론을 멈추고 이탈만 감시한다. 보통 적재 직후 1~5초 안에 확정된다(리플레이 실측 5초, 10/10 프레임).
- 저장: 판정 프레임 최대 10쌍 + `result.json` + `thumb.jpg` (미확정은 정지 후보 5장). 하루 수십 MB.
- 업로드 완료(`.uploaded`) 3일 뒤 png 삭제, 90일 뒤 세션 삭제, 지표 로그 14일 — 서비스가 시간당 1회 정리. 업로드 안 된 세션은 지우지 않는다.
- 전원: 작업자/개발자가 OFF. 안 되면 **매일 20:00 자동 종료**(크론, 1분 예고). 아침 멀티탭 ON 으로 자동 기동(2~3분).

## 저장 구조

```
datasets_factory_infer/
  events.jsonl                       # 상태 전이·판정·마감 (append)
  heartbeat.json                     # 10분마다 (NAS 에도 복사 → 대시보드 "서비스 살아있음")
  background.npz                     # 학습된 빈 지그 배경 (없으면 reference_images 시드)
  metrics/metrics_YYYYMMDD.jsonl     # 1Hz 지표: diff·intr·motion·pts·gate·pred (임계값 튜닝용, 14일)
  20261102/s_143417/
    rgb_0000.png depth_0000.png ...  # 판정 프레임 (≤10쌍)
    result.json                      # 판정 결과 (status·pred_class·part_no·radar·D·margin·confidence·frames·models·config)
    meta.json                        # 06 호환 (class_name=판정, infer=true)
    thumb.jpg                        # 대표 프레임 + 6점 오버레이
    .uploaded                        # NAS 업로드 완료 마커
```

`result.json.status`: `judged`(판정) · `unknown`(홀은 보이나 CAD D 와 30mm+ 어긋남) · `unrecognized`(홀 자체가 안 잡힘 = 기종 외 부품) · `abstain`(도어는 봤으나 게이트 실패).

## 개발자 확인 (세종)

- 당일 실시간: `http://100.70.228.127:5000` (tailscale) — 라이브·지표·현재 세션·오늘 목록(판정 이미지 | 작업자 라벨 대표 | 2순위 후보).
  헤더에 **"그림자 모드"** 배지가 있으면 NAS 업로드가 꺼진 상태.
- 전체 통계·확정: 학습 PC webapp `/inference` (3단계 구현 후).
- SSH: `tail -f logs/infer_$(date +%Y%m%d).log`, `journalctl -u door-infer -f`, `tail datasets_factory_infer/events.jsonl`.

## 운영 명령어

- **설치(1회, sudo)**: `cd ~/workspace/parts_deep/class_estimation/door_pipeline && sudo bash deploy/install_infer.sh`
  → 수집 서비스 비활성·키오스크 제거·door-infer 등록·크론(업로드 5분, 20:00 종료)·`~/.config/door-infer.env` 생성(그림자 모드).
- **크론 수동 등록** (install 스크립트의 크론 단계가 실패했을 때 — 2026-11-02 현장 설치에서 빈 크론이 설치된 사례, 스크립트는 수정됨):
  ```bash
  ( crontab -l 2>/dev/null | grep -v -E 'nas_sync(_infer)?\.sh|door-infer-autoshutdown' ; \
    echo "*/5 * * * * /home/user/workspace/parts_deep/class_estimation/door_pipeline/deploy/nas_sync_infer.sh" ; \
    echo "0 20 * * * sudo -n /usr/sbin/shutdown -h +1 'door-infer-autoshutdown' # door-infer-autoshutdown" ) | crontab - && crontab -l
  ```
- **그림자 모드 해제(NAS·DB 연동 ON)**: `~/.config/door-infer.env` 에서 `DOOR_INFER_UPLOAD=1` → `sudo systemctl restart door-infer`
  (sudo 없이: `kill $(systemctl show door-infer -p MainPID --value)` → 5초 내 자동 재시작. SIGTERM 이라 진행 중 세션은 마감된다)
- **임계값 조정**: 같은 env 파일에 `DOOR_INFER_OCC_ON=0.15` 등 (`07_factory_infer.py` 상단 CFG 가 전체 목록) → 재시작.
  기본값 근거: diff 빈 지그 0.015 / 도어 0.7 (OCC_ON 0.15·OFF 0.08), 침입(ROI) 도어만 ≤0.016 / 작업자 ≥0.033 (INTR_FRAC 0.025).
- **재판정**: 웹 [재판정] 또는 `curl -X POST http://localhost:5000/api/reset` — 현재 세션 윈도 초기화 후 다시 판정.
- **서비스 종료/복귀**: `sudo systemctl stop door-infer` / `sudo systemctl start door-infer`. 수집 서비스로 되돌리기:
  `sudo systemctl disable --now door-infer && sudo systemctl enable --now door-capture` (키오스크는 `deploy/door-kiosk.desktop` 을 `~/.config/autostart/` 에 복사).
- **원격 종료**: 웹 [시스템 종료] (1분 후 shutdown, systemd 가 서비스를 먼저 내려 세션 마감) 또는 `sudo shutdown -h +1`.
- **리플레이 검증(카메라 없이)**: `python 07_factory_infer.py --port 5055 --no_cuda_probe --replay "빈지그폴더:20,도어폴더:45,빈지그폴더:25"`
  (`DOOR_INFER_OUTPUT_ROOT=/tmp/x` 로 산출물 분리, `DOOR_INFER_LEAVE_S=8` 등으로 시간 단축)
- **부팅 시 정상 시퀀스**: CUDA 준비 전 `CUDA 미준비 — 프로세스 종료 후 systemd 재시작 대기` 수 회 → `카메라 연결:` → `배경 기준맵:` → `추론 서비스 시작`.

## 그림자 모드에서 먼저 볼 것 (2026-11-02 첫 가동 관찰)

- 현장에는 **토치 케이블 밸런서(파란 봉)·호스가 도어 위를 가로질러** 놓이는 시간이 길다. ROI 침입 지표가 0.09~0.14 로 임계(0.025)를 넘어
  판정이 막히고, 호스가 래치 코너 홀을 덮으면 `no_corner` 로 보류된다. 적재 직후 봉·호스가 치워진 조용한 구간이 매일 있는지 `metrics_<날짜>.jsonl` 의
  `intr`·`pts`·`gate` 추이로 확인하고, 없으면 침입 게이트를 끄고(`DOOR_INFER_INTR_FRAC=1.0`) 기하 게이트·윈도 중앙값에만 맡기는 운용으로 바꾼다.
- 정지 지표는 조용한 장면에서 1.9~3.2 (임계 4.0). 작업 중은 3~15.

## 트러블슈팅

| 증상 | 조치 |
|---|---|
| 헤더 "카메라 연결 대기" | 10초 간격 자동 재시도. 지속되면 GMSL 케이블 확인 후 재부팅 |
| 빈 지그인데 "판정 중"이 계속됨 | 배경 기준맵 어긋남(카메라 이동·조명). `:5000` 지표 띠의 diff 확인 → `rm datasets_factory_infer/background.npz` 후 재시작(시드로 복귀), 또는 빈 지그 상태에서 10분 두면 EMA 로 추종 |
| 도어가 있는데 "빈 지그" | diff 가 OCC_ON(0.15) 미만 — 지표 확인 후 env 로 낮춤. pts≥4 면 diff 와 무관하게 점유로 잡힌다 |
| 보류(abstain)가 잦음 | 세션 상세의 gate_counts 확인: `near_border`(도어가 프레임 밖) → 카메라 화각, `pose_warn`(z·tilt) → 거치 상태, 침입(intr)만 높으면 INTR_FRAC 상향 |
| 미인식(unrecognized)이 우리 도어 | 홀 검출 실패 — 저장 후보 프레임을 보고 조명·가림 확인. 모델 갱신은 `scripts/model_sync.sh pull` |
| 업로드 "대기 N" 이 안 줄어듦 | `tail logs/nas_sync_infer.log`, `~/bin/rclone lsd nas:Guest/weld_infer` 로 NAS 연결 확인 |
| 디스크 여유 20GB 미만 | 업로드 완료 세션은 자동 정리됨. `datasets_factory_collect/`(수집기 141GB) 는 NAS 확인 후 수동 삭제 |
