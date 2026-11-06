# 경주 공장 도어 실시간 판정 — 운영 가이드 (추론 서비스)

작성: 2026-11-02. 수집 서비스(`DOC_factory_collect.md`, 2026-11-02 중지)를 대체하는 **무인 판정 서비스**.
장비: ZED X mini + Zed Box(zedbox-weld, Jetson Orin NX) + LTE 라우터. 터치모니터는 쓰지 않는다.
설계·결정 사항: `DOC_infer_service_plan.md`. 코드 요약: `README.md` §E.

## 구성 요소

| 파일 | 역할 |
|---|---|
| `07_factory_infer.py` | 판정 서비스 (Flask, 포트 5000). 상태기계·게이트·판정·저장·보존 정리·하트비트 |
| `factory_camera.py` | ZED 상시 구동 카메라 (06 에서 분리, 자동 재연결·CUDA 프로브) |
| `templates/factory_infer.html` | 개발자용 웹 페이지 (라이브·지표 차트·현재 세션·오늘 도어 목록·재판정·빈 지그 배경 등록·배경 보기·원격 종료) |
| `depth_view.py`, `tools/depth_colorize.py` | depth(mm) → 컬러 그림. 웹 `/bg.jpg` 와 CLI 공용 (16비트 depth PNG 는 뷰어에서 검게 보임) |
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
- 판정은 **잠정 → 최종** 두 단계다(2026-11-02 사용자 지적 반영). 현장은 밑판을 먼저 올리고 브라켓을 차례로 올린 뒤 용접하므로,
  적재 직후의 첫 판정(보통 1~5초)은 브라켓이 덜 올라간 상태일 수 있다. 그래서 잠정 판정 뒤에도 초당 2회 깨끗한 프레임으로 계속 재평가해
  클래스·레이더가 바뀌면 갱신(이력 `pred_history`)하고, **도어가 빠질 때 마지막 깨끗한 프레임 10장**으로 최종 확정·저장한다.
  NAS 에는 최종(`final: true`)만 올라간다. 화면의 "잠정" 표시가 사라지면 최종이다.
- 잠정 판정 변경 규칙(2026-11-02 오후, 24분 동안 14회 흔들린 사례 반영): ① 홀 판정이 한 번 난 세션은 CNN 폴백이 덮어쓰지 못한다
  ② 레이더 표가 윈도에서 빠져 '미정'이 되면 이전 확정값을 유지한다 ③ (클래스, 레이더) 변경은 연속 3회(`CHANGE_CONFIRM_N`) 같은 결과가 나와야 반영한다.
- CNN 폴백은 홀 6점이 잡혔는데 기하가 안 맞는 프레임에서만 돈다. 홀이 없는 맨 밑판은 CNN 을 돌리지 않는다(맨 밑판을 E23 으로 오판한 사례 차단).
- **레이더 옵션 규칙 v2 (2026-11-06, 서비스 0.2.0)**: 11/4 E25_RH 2세션(s_075733·s_151538)이 브래킷 없는 도어인데 보강재 위에 놓인 바 클램프·렌치가
  CAD 홀 자리의 어두움+depth 높이를 흉내내 규칙 경로에서 O 로 잠겼다(VLM 2차 의견은 둘 다 X). 수정(`partno/radar_check.py`, 규칙 상세는 `partno/README.md`):
  ① 피크 경로는 depth 평면 법선을 쓰는 3D 프레임으로 브래킷 피크를 판 위 점으로 풀어 CAD 홀 ≤25mm·간격 91.6±12mm 검사(이전 radial 시차 근사는 도어 기울기 8~9° 를
  무시해 E38_LH_RR 이 늘 규칙 경로로 떨어졌음) ② 규칙 경로의 O 는 브래킷 피크 동반 + 판 전체 높이(격자 중앙값 ≥37mm) 전제 위에서만 — 공구는 둘 다 못 넘어 **미정**이 된다.
  레이더 미정이면 품번이 안 나오므로 `/inference` 에서 사람이 확정한다. `result.json` 에 `radar_rule`·`option_frames`(윈도 프레임별 피크/규칙·점수·높이·동반)·`frames[].option` 이 추가됐다.
- 저장: 판정 프레임 최대 10쌍 + `result.json` + `thumb.jpg` (미확정은 정지 후보 5장). 하루 수십 MB.
- 업로드 완료(`.uploaded`) 3일 뒤 png 삭제, 90일 뒤 세션 삭제, 지표 로그 14일 — 서비스가 시간당 1회 정리. 업로드 안 된 세션은 지우지 않는다.
- 전원: 작업자/개발자가 OFF. 안 되면 **매일 20:00 자동 종료**(크론, 1분 예고). 아침 멀티탭 ON 으로 자동 기동(2~3분).

## 비움(이탈) 판정과 배경맵 — 2026-11-04 개정

- 비움 신호는 두 가지 중 하나면 된다(홀 점 수 < 2 전제): ① 배경맵 대비 중앙(OCC_ROI) 차이 `diff ≤ OCC_OFF`, ② **평판 높이** — OCC_ROI depth 중앙값 `z_occ` 가
  `z_plate − Z_EMPTY_TOL_MM(15)` 이상(= 평판 높이, 도어 면은 ≈33mm 앞). ②는 배경맵이 틀려도 동작하는 독립 신호다.
- 배경맵 EMA 갱신은 EMPTY 에서, **기존 배경과 BG_UPDATE_MM(10mm) 안쪽으로 같은 픽셀만** 추종한다(보수적). 11/3 오후 EMA 가 밑판(중앙 diff 0.09~0.12 로
  '비움' 판정 통과)을 흡수해 11/4 하루 종일 빈 지그가 diff ≥ 0.2 로 읽혀 도어 5대가 세션 하나에 합쳐진 사고의 재발 방지. UNRESOLVED 흡수는 폐지.
- **자가 복구**: 평판 높이로는 비었는데(z_empty·홀 0·정지 60초) 배경맵은 점유(diff ≥ OCC_ON)라고 하면 배경맵이 틀린 것이므로 지금 화면을 빈 지그 배경으로
  자동 재등록한다(events `bg_auto`, 10분에 1회). `z_plate` 는 등록(수동/자동) 때 OCC_ROI 중앙값으로 갱신해 `background.npz` 에 보존, 없으면 `Z_PLATE_MM`(1468).
- 확인: 지표 띠의 `zo 1500/1468 Z-EMPTY` (z_occ/z_plate), `/bg.jpg` 오른쪽 DIFF 가 빈 지그에서 회색이어야 정상.

## 재시작·정전과 열린 세션 (2026-11-04)

- 서비스가 SIGTERM 으로 내려가면(재시작·20:00 종료) 진행 중 세션은 `service_stop` 으로 마감된다. 급정전·kill -9 면 `final=false` 로 남는다(고아 세션).
- 기동 시 고아 세션을 정리한다: 마지막 기록이 `ORPHAN_RESUME_S`(15분) 이내인 최신 1개는 **이어받기 후보** — 기동 직후 도어가 그대로 있으면 같은 세션 디렉터리로
  이어서 판정(result.json `resumed` 에 공백 기록, 시작 시각 유지). 빈 지그가 20초 이상이면 후보도 마감. 나머지 고아는 마지막 기록 시각으로 `service_stop(orphan)` 마감 → NAS 업로드.
- 11/2 정전·11/3 전원 OFF·11/4 07:57 재부팅으로 남아 있던 고아 3건은 이 로직이 첫 기동 때 마감한다.
- 수정본 배포로 재시작할 때는 **빈 지그 구간에 하라**(`deploy/infer_ctl.sh restart-when-empty`). 11/4 14:49 재시작으로 같은 도어가 세션 2개로 갈린 사례.

## 타임랩스 (2026-11-04, 육안 판정 근거)

- 상태와 무관하게 **1분 1장** `timelapse/<날짜>/HHMMSS.jpg`(960px, 6점 오버레이) + `index.jsonl`(시각·상태·세션·잠정 클래스·z_occ·diff·pts·게이트),
  10분마다 배경 비교 그림 `bg_HHMMSS.jpg`(/bg.jpg 와 동일). 하루 약 60~70MB, `TL_RETAIN_DAYS`(30일) 뒤 삭제.
- `:5000` 지표 차트 아래 스트립(테두리 색 = 상태, 클릭 → 원본·배경 비교), 세션 모달에 **세션 구간 타임랩스** 표시 → 세션 경계·브라켓 조립 시점·오판 원인을 눈으로 확인.
- NAS: `nas_sync_infer.sh` 가 최근 2일치를 새 파일만 올림(`weld_infer/timelapse/`). 학습 PC: `ingest_infer.py --timelapse-days 3`(기본) 이 미러 →
  webapp 세션 상세에 세션 구간 타임랩스.

## 저장 구조

```
datasets_factory_infer/
  events.jsonl                       # 상태 전이·판정·마감 (append)
  heartbeat.json                     # 10분마다 (NAS 에도 복사 → 대시보드 "서비스 살아있음")
  background.npz                     # 학습된 빈 지그 배경 (없으면 reference_images 시드) + z_plate(평판 높이)
  timelapse/YYYYMMDD/HHMMSS.jpg      # 1분 1장 + index.jsonl + bg_HHMMSS.jpg(10분)
  metrics/metrics_YYYYMMDD.jsonl     # 1Hz 지표: diff·intr·motion·pts·gate·pred (임계값 튜닝용, 14일)
  20261102/s_143417/
    rgb_0000.png depth_0000.png ...  # 판정 프레임 (≤10쌍)
    result.json                      # 판정 결과 (status·pred_class·part_no·radar·D·margin·confidence·frames·models·config)
    meta.json                        # 06 호환 (class_name=판정, infer=true)
    thumb.jpg                        # 대표 프레임 + 6점 오버레이
    .uploaded                        # NAS 업로드 완료 마커
```

`result.json`: `provisional`(잠정 여부) · `pred_history`(잠정 판정 변경 이력) · `final_at`. `status`: `judged`(판정) · `unknown`(홀은 보이나 CAD D 와 30mm+ 어긋남) · `unrecognized`(홀 자체가 안 잡힘 = 기종 외 부품) · `abstain`(도어는 봤으나 게이트 실패).

## 개발자 확인 (세종)

- 당일 실시간: `http://100.70.228.127:5000` (tailscale) — 라이브·지표·현재 세션·오늘 목록(판정 이미지 | 작업자 라벨 대표 | 2순위 후보).
  헤더에 **"그림자 모드"** 배지가 있으면 NAS 업로드가 꺼진 상태.
- 전체 통계·확정: 학습 PC webapp `/inference` (3단계 구현 후).
- **VLM 독립 2차 의견(2026-11-03~)**: 학습 PC cron(`vlm/cron_audit_infer.py`, ingest 2분 뒤)이 새 세션의 원본 rgb 2장을 Claude 에 보내 형상군·레이더·이상(호스·손·공구·밑판만 등)을 독립 판정해 `/inference` 'VLM 2차 의견' 열과 대시보드 'VLM ≠ 판정' 카드에 띄운다. 판정 결과는 Claude 에 주지 않는다. 무료 API 크레딧 한도(11/17)까지만 돌고 자동 중지 — 연장은 `VLM_AUDIT_UNTIL`. 제드 박스 무관.
- SSH: `tail -f logs/infer_$(date +%Y%m%d).log`, `journalctl -u door-infer -f`, `tail datasets_factory_infer/events.jsonl`.

## 운영 명령어

- **코드 갱신(모델 변경 없음)**: `cd ~/workspace/parts_deep && git pull && sudo systemctl restart door-infer && journalctl -u door-infer -n 20 --no-pager`
  → 로그 첫 줄 `추론 서비스 시작 v0.2.0` 확인. 2026-11-06 레이더 규칙 v2 는 모델 파일을 바꾸지 않으므로 `model_sync.sh pull` 불필요.
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
- **점유 영역 `OCC_ROI`**(기본 `0.42,0.10,0.66,0.90`, 화면의 초록 상자): diff 는 이 지그 중앙 빈 부분에서만 잰다. 양쪽 세로 바가 도어 기종마다
  레일을 따라 가로로 움직여 도어 ROI(노랑) 전체로 재면 빈 지그도 15~20% 가 변화로 잡히기 때문(2026-11-03). 가장 작은 E23 도 중앙은 덮는다(사용자 확정).
  침입 게이트·홀 검출은 도어 ROI 그대로. 도어 ROI 전체 diff 는 지표 `diff_door` 로 남긴다. 중앙에 잡동사니(판·공구)를 두면 점유로 잡히니 비워 둘 것.
- **재판정**: 웹 [재판정] 또는 `curl -X POST http://localhost:5000/api/reset` — 현재 세션 윈도 초기화 후 다시 판정.
- **빈 지그 배경 등록**: 지그가 비어 있을 때 웹 [빈 지그 배경 등록] 또는 `curl -X POST http://localhost:5000/api/bg_learn` — 지금 depth(1/4) 를
  빈 지그 기준맵으로 채택해 `background.npz` 에 저장(재시작 후에도 유지, 이후 EMPTY 에서 EMA 로 추종). 홀이 보이거나 움직이면 거부(웹은 확인 후 강제 `?force=1`).
  시드(8/27)가 현재 카메라·지그와 안 맞아 빈 지그도 diff≈0.6 으로 읽히면 EMPTY 로 못 들어가 세션이 안 끝난다(2026-11-03: 도어 5대가 세션 하나로) — 이 버튼이 유일한 탈출구.
- **배경 보기**: 웹 [배경 보기] 또는 `http://localhost:5000/bg.jpg` — [등록된 배경 | 지금 depth | 차이] 를 색으로(가까움=빨강 … 멀리=파랑, 차이: 빨강=배경보다 가까움,
  파랑=멀어짐, 회색=같음 ±25mm). 16비트 depth PNG 는 뷰어에서 검게 보이므로 파일은 `python tools/depth_colorize.py <depth.png|background.npz> [--ref 기준]` 으로 컬러 PNG 를 만든다(`depth_view.py` 공용).
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
| 빈 지그인데 "판정 중"이 계속됨 | 배경 기준맵 어긋남(카메라 이동·조명). `:5000` 지표 띠의 diff 확인 → 지그를 비우고 웹 [빈 지그 배경 등록](`/api/bg_learn`). EMPTY 로 못 들어가면 EMA 추종도 안 된다(2026-11-03). 시드 복귀는 `rm datasets_factory_infer/background.npz` 후 재시작 |
| 도어가 있는데 "빈 지그" | diff 가 OCC_ON(0.15) 미만 — 지표 확인 후 env 로 낮춤. pts≥4 면 diff 와 무관하게 점유로 잡힌다 |
| 빈 지그인데 점유로 잡힘(배경은 맞는데) | 중앙 초록 상자(OCC_ROI) 안에 물건이 놓였거나 세로 바가 상자에 걸침 — [배경 보기]로 확인, 물건 치우기 또는 `DOOR_INFER_OCC_ROI` 조정 |
| 보류(abstain)가 잦음 | 세션 상세의 gate_counts 확인: `near_border`(도어가 프레임 밖) → 카메라 화각, `pose_warn`(z·tilt) → 거치 상태, 침입(intr)만 높으면 INTR_FRAC 상향 |
| 미인식(unrecognized)이 우리 도어 | 홀 검출 실패 — 저장 후보 프레임을 보고 조명·가림 확인. 모델 갱신은 `scripts/model_sync.sh pull` |
| 업로드 "대기 N" 이 안 줄어듦 | `tail logs/nas_sync_infer.log`, `~/bin/rclone lsd nas:Guest/weld_infer` 로 NAS 연결 확인 |
| 디스크 여유 20GB 미만 | 업로드 완료 세션은 자동 정리됨. `datasets_factory_collect/`(수집기 141GB) 는 NAS 확인 후 수동 삭제 |
