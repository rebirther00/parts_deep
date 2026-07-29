#!/bin/bash
# GUI 로그인 후 자동 실행: 화면 절전 차단 + 수집 서비스 대기 + 전체화면 키오스크
# (snap Firefox는 이 장비에서 실행 불가 → WebKit2GTK 웹뷰 사용)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
xset s off -dpms 2>/dev/null

for i in $(seq 1 60); do
    curl -sf -o /dev/null http://localhost:5000/api/status && break
    sleep 2
done

# 웹뷰가 죽어도 화면이 계속 뜨도록 무한 재실행
while true; do
    python3 "$SCRIPT_DIR/kiosk_webview.py" http://localhost:5000
    sleep 3
done
