#!/bin/bash
# 추론 서비스 설치 — 수집 서비스(door-capture)·키오스크를 대체한다. sudo 로 실행:
#   sudo bash deploy/install_infer.sh
# 되돌리기: sudo systemctl disable --now door-infer && sudo systemctl enable --now door-capture
set -e
DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
DOOR_DIR="$(dirname "$DEPLOY_DIR")"
USER_HOME=/home/user

echo "[1/5] 수집 서비스 중지·비활성 + 키오스크 autostart 제거"
systemctl disable --now door-capture.service 2>/dev/null || true
sudo -u user pkill -f 'deploy/kiosk.sh' 2>/dev/null || true
sudo -u user pkill -f kiosk_webview.py 2>/dev/null || true
rm -f "$USER_HOME/.config/autostart/door-kiosk.desktop"

echo "[2/5] 추론 서비스 등록 (그림자 모드: NAS 업로드 꺼짐 — ~/.config/door-infer.env 에서 DOOR_INFER_UPLOAD=1 로 켬)"
if [ ! -e "$USER_HOME/.config/door-infer.env" ]; then
    sudo -u user mkdir -p "$USER_HOME/.config"
    sudo -u user tee "$USER_HOME/.config/door-infer.env" >/dev/null <<'ENV'
# door-infer.service 환경 — 바꾼 뒤: sudo systemctl restart door-infer (또는 kill $(systemctl show door-infer -p MainPID --value))
DOOR_INFER_UPLOAD=0
# 임계값 기본값(07_factory_infer.py CFG) — 바꿀 것만 주석 해제
# DOOR_INFER_OCC_ON=0.15        # 배경 대비 차이 비율 이상이면 점유 (빈 지그 ≈0.015, 도어 ≈0.7)
# DOOR_INFER_OCC_OFF=0.08
# DOOR_INFER_MOTION_THR=4.0     # 정지 판정 (조용한 장면 1.9~3.2, 작업 중 3~15)
# DOOR_INFER_INTR_FRAC=0.025    # 도어 ROI 침입 비율 (도어만 ≤0.016, 작업자 ≥0.033). 1.0 이면 침입 게이트 끔
ENV
fi
cp "$DEPLOY_DIR/door-infer.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now door-infer.service

echo "[3/5] 종료용 sudoers (shutdown 만 무비밀번호 — 웹 종료 버튼·20:00 자동 종료)"
echo "user ALL=(ALL) NOPASSWD: /usr/sbin/shutdown" > /etc/sudoers.d/door-kiosk
chmod 440 /etc/sudoers.d/door-kiosk

echo "[4/5] user 크론: NAS 업로드 보완(5분) + 20:00 자동 종료 (수집용 nas_sync.sh 줄은 제거)"
chmod +x "$DEPLOY_DIR/nas_sync_infer.sh"
# 주의: set -e 아래에서 grep -v 가 아무 줄도 못 남기면(기존 크론이 nas_sync.sh 한 줄뿐일 때) 서브셸이 거기서 끝나 빈 크론이 설치되던
# 버그가 있었음(2026-11-02 현장 설치 때 발생) → || true 로 막는다
( { sudo -u user crontab -l 2>/dev/null | grep -v 'deploy/nas_sync.sh' | grep -v 'deploy/nas_sync_infer.sh' | grep -v 'door-infer-autoshutdown' || true; } ;
  echo "*/5 * * * * $DEPLOY_DIR/nas_sync_infer.sh" ;
  echo "0 20 * * * sudo -n /usr/sbin/shutdown -h +1 'door-infer-autoshutdown' # door-infer-autoshutdown" ) | sudo -u user crontab -
sudo -u user crontab -l | grep -q nas_sync_infer || { echo "크론 등록 실패 — 수동 등록 필요 (DOC_factory_infer.md)"; exit 1; }

echo "[5/5] 상태 확인"
sleep 3
systemctl --no-pager status door-infer.service | head -8
echo
echo "완료. 브라우저에서 http://<tailscale-ip>:5000 확인. 로그: journalctl -u door-infer -f"
