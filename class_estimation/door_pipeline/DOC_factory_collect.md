# 경주 공장 도어 이미지 수집 — 운영 가이드

작성: 2026-07-29. 4~6주간 현장(경주)에서 분류 학습용 도어 이미지를 수집한다.
장비: ZED X mini + Zed Box(zedbox-weld) + LTE 라우터 + 16" 터치모니터.
개발자는 세종에서 tailscale(주) / ZeroTier(백업)로 원격 확인.

## 구성 요소

| 파일 | 역할 |
|---|---|
| `06_factory_capture.py` | 수집 서비스 (Flask, 포트 5000). 키오스크 UI + 상태 API |
| `templates/factory.html` | 터치 UI (9버튼 + 취득 시작/종료 + 시스템 종료 + 상태/프리뷰) |
| `deploy/door-capture.service` | systemd 유닛 (부팅 자동시작, 죽으면 5초 후 재시작) |
| `deploy/kiosk.sh` + `kiosk_webview.py` + `door-kiosk.desktop` | GUI 자동로그인 시 전체화면 키오스크 자동 실행 (snap Firefox가 이 장비에서 실행 불가라 WebKit2GTK 웹뷰 사용) |
| `deploy/install_factory.sh` | 설치 스크립트 (`sudo bash deploy/install_factory.sh` 1회) |

## 동작 사양

- 취득: 시작 후 **2초 워밍업**(노출 안정화, 저장 안 함) → **2초에 1장** rgb/depth 쌍 저장 → **[취득 종료]** 버튼으로 종료 (안 누르면 300초 후 자동 종료).
- 종료 시 클래스 선택이 **Unknown으로 자동 복귀** (라벨 실수 방지). Unknown 상태로 시작해도 취득은 됨 → 나중에 수동 라벨링.
- 취득 중 클래스 변경/중복 시작/시스템 종료는 거부됨.
- 카메라: HD1200(1920×1200) NEURAL depth, mm 단위. grab 60회 연속 실패 시 자동 재연결.
- 저장 구조:
  ```
  datasets_factory_collect/
    events.jsonl                  # 모든 버튼/세션 이벤트 (라벨 정정 추적용)
    20260729/E25_door_LH_FRT/s_103015/
      rgb_0000.png                # 1920x1200 BGR
      depth_0000.png              # uint16, mm
      meta.json                   # 클래스/시각/쌍수/종료사유
  ```
- 서비스 로그: `logs/factory_YYYYMMDD.log`
- 테스트용 환경변수: `DOOR_SESSION_DURATION`, `DOOR_CAPTURE_INTERVAL`, `DOOR_WARMUP`, `DOOR_PORT`

## 작업자 일일 절차 (현장 게시용)

1. 출근(07:00): 멀티탭 전원 ON → 화면에 수집 프로그램이 자동으로 뜸 (2~3분)
2. 도어 세팅 후 용접 전: **도어 종류 버튼 선택 → [취득 시작]** → 카메라 밖으로 이동
3. 용접이 끝나면 **[취득 종료]** 버튼 (종료되면 도어 종류가 [모름/기타]로 돌아감. 안 눌러도 5분 후 자동 종료). 다음 도어부터 2번 반복
4. 도어 종류를 모르면 **[모름/기타]** 선택 후 시작
5. 퇴근(16~18시): **[시스템 종료] → [종료]** → 화면 꺼진 후 멀티탭 OFF

## 개발자 원격 확인 (세종)

- 브라우저: `http://100.70.228.127:5000` (tailscale) — 현장과 같은 화면·프리뷰·오늘 통계
- tailscale 불통 시: ZeroTier IP `10.138.38.89`로 동일 접속
- SSH: `ssh user@100.70.228.127` → `tail logs/factory_$(date +%Y%m%d).log`, `cat datasets_factory_collect/events.jsonl`
- 백업(수동): MobaXterm/WinSCP로 `datasets_factory_collect/` 전체를 NAS에 복사 (매일 저녁 권장, 최소 주 2회)

## 운영 명령어 (개발자)

- **서비스 재시작**: `sudo systemctl restart door-capture`
  - sudo 없이: `kill $(systemctl show door-capture -p MainPID --value)` → systemd가 5초 내 자동 재시작
  - `06_factory_capture.py`·`templates/factory.html` 수정 후에는 재시작 필수 (Flask debug off라 템플릿 캐시됨)
- **상태·로그**: `systemctl status door-capture`, `journalctl -u door-capture -n 50`
- **키오스크 화면 완전 종료**: `pkill -f 'deploy/kiosk.sh'; pkill -f kiosk_webview.py`
  - 루프를 먼저 죽여야 함. Alt+F4는 3초 뒤 자동 재실행됨(현장 오조작 대비 의도된 동작)
  - 다시 띄우기: 재부팅/재로그인 또는 `DISPLAY=:0 bash deploy/kiosk.sh &`
- **부팅 시 정상 시퀀스**: CUDA 준비 전이면 journal에 `CUDA 미준비 — 프로세스 종료 후 systemd 재시작 대기`가 수 회 반복된 뒤 `카메라 연결:`이 찍힘 — 오류 아님

## 설치 (1회, sudo 필요)

```bash
cd ~/workspace/parts_deep/class_estimation/door_pipeline
sudo bash deploy/install_factory.sh
sudo reboot   # 재부팅 후 전체 자동 실행 확인
```

## 현장 설치일 체크리스트

- [ ] LTE 라우터 연결 후 IP/인터넷/tailscale/ZeroTier 접속 확인
- [ ] 카메라 화각: 도어 전체가 프레임에 들어오는지, 마운트 고정
- [ ] 멀티탭 OFF→ON 전체 사이클: 자동부팅 → 키오스크 표시 → 테스트 세션 1회 → 파일 확인
- [ ] [시스템 종료] 버튼 → 화면 꺼짐 → 멀티탭 OFF 확인
- [ ] 작업자 시연 + 작업자가 직접 1회 수행 + 안내문 부착
- [ ] 세종에서 원격 접속(양 경로) 최종 확인

## 트러블슈팅

| 증상 | 조치 |
|---|---|
| 화면에 "카메라 연결 대기 중" | 10초 간격 자동 재시도 중. 지속되면 GMSL 케이블 확인 후 재부팅 |
| 키오스크가 안 뜸 | `systemctl status door-capture` 확인, `journalctl -u door-capture -n 50` |
| 원격 접속 불가 (양 경로 모두) | 현장에 전화 → 멀티탭 재부팅 요청 |
| 디스크 여유 20GB 미만(화면 주황색) | 오래된 날짜 폴더를 NAS 백업 확인 후 삭제 |
