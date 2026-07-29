#!/bin/bash
# 현장 수집 서비스 설치 스크립트 — sudo로 실행:
#   sudo bash deploy/install_factory.sh
set -e
DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[1/4] systemd 서비스 등록"
cp "$DEPLOY_DIR/door-capture.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now door-capture.service

echo "[2/4] 종료 버튼용 sudoers 등록 (shutdown만 무비밀번호 허용)"
echo "user ALL=(ALL) NOPASSWD: /usr/sbin/shutdown" > /etc/sudoers.d/door-kiosk
chmod 440 /etc/sudoers.d/door-kiosk

echo "[3/4] 키오스크 자동시작 등록"
chmod +x "$DEPLOY_DIR/kiosk.sh"
sudo -u user mkdir -p /home/user/.config/autostart
sudo -u user cp "$DEPLOY_DIR/door-kiosk.desktop" /home/user/.config/autostart/

echo "[4/4] 상태 확인"
sleep 3
systemctl --no-pager status door-capture.service | head -8
echo
echo "완료. 브라우저에서 http://localhost:5000 확인 (재부팅 시 전체 자동 실행)"
