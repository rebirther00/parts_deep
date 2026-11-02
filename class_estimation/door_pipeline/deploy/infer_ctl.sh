#!/bin/bash
# 추론 서비스 수동 제어 (systemd 설치 전 / sudo 없이): bash deploy/infer_ctl.sh start|stop|restart|status
# systemd 로 설치한 뒤에는 systemctl 을 쓴다 (둘을 동시에 띄우면 포트 5000·카메라 충돌).
set -u
DOOR_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PY=/home/user/workspace/zed_env/bin/python
PAT="^$PY 07_factory_infer.py"        # 명령행 앞부분 고정 → 이 스크립트 자신은 매칭되지 않음
LOG="$DOOR_DIR/logs/infer_nohup.log"
ENVF="$HOME/.config/door-infer.env"

pids() { pgrep -f "$PAT"; }

do_stop() {
    local p; p=$(pids | tr '\n' ' ')
    [ -z "$p" ] && { echo "stopped (not running)"; return 0; }
    kill -TERM $p
    for _ in $(seq 1 30); do pids >/dev/null || break; sleep 1; done
    if pids >/dev/null; then echo "강제 종료"; kill -KILL $(pids); sleep 1; fi
    echo "stopped ($p) — 진행 중 세션은 마감됨"
}

do_start() {
    if pids >/dev/null; then echo "already running ($(pids | tr '\n' ' '))"; return 0; fi
    if systemctl is-active --quiet door-infer.service 2>/dev/null; then echo "systemd door-infer 가 이미 실행 중 — systemctl 로 제어"; return 1; fi
    mkdir -p "$DOOR_DIR/logs"
    [ -f "$ENVF" ] && { set -a; . "$ENVF"; set +a; }
    cd "$DOOR_DIR" && nohup setsid "$PY" 07_factory_infer.py "$@" >> "$LOG" 2>&1 < /dev/null &
    disown
    for _ in $(seq 1 60); do curl -sf -o /dev/null http://127.0.0.1:${DOOR_INFER_PORT:-5000}/api/status && break; sleep 1; done
    echo "started ($(pids | tr '\n' ' ')) — http://$(tailscale ip -4 2>/dev/null || hostname -I | awk '{print $1}'):${DOOR_INFER_PORT:-5000}"
}

case "${1:-status}" in
    start)   shift; do_start "$@" ;;
    stop)    do_stop ;;
    restart) shift; do_stop; do_start "$@" ;;
    status)
        if pids >/dev/null; then echo "running ($(pids | tr '\n' ' '))"; else echo "not running"; fi
        curl -s http://127.0.0.1:${DOOR_INFER_PORT:-5000}/api/status | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit()
m=d['metrics']; s=d['session'] or {}
print('state', d['state'], '%.0fs' % d['state_for_s'], '| camera', d['camera_ok'], '| shadow', d['service']['shadow'], '| today', d['today'])
print('diff', m.get('diff'), 'intr', m.get('intr'), 'intr_full', m.get('intr_full'), 'motion', m.get('motion'), 'pts', m.get('pts'), 'static', m.get('static'), 'clean', m.get('clean'))
s and print('session', s.get('session_dir'), s.get('status'), s.get('class'), 'judged', s.get('n_judged'), '/', s.get('window'), s.get('gate_counts'))" ;;
    *) echo "usage: $0 start|stop|restart|status"; exit 2 ;;
esac
