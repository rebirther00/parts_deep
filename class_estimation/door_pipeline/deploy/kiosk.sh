#!/bin/bash
# GUI 로그인 후 자동 실행: 화면 절전 차단 + 수집 서비스 대기 + Firefox 키오스크
xset s off -dpms 2>/dev/null

for i in $(seq 1 60); do
    curl -sf -o /dev/null http://localhost:5000/api/status && break
    sleep 2
done

exec firefox --kiosk http://localhost:5000
