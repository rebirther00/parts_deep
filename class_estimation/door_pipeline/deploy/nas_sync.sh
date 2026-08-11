#!/bin/bash
# 수집 세션을 세종 NAS(nas:Guest/weld)로 업로드
#  - 취득 종료 시 06_factory_capture.py가 한 번 킥 + cron 5분 주기로 밀린 것 보완
#  - meta.json에 finished:true 인 세션만, 업로드 성공 시 .uploaded 마커 생성
#  - 로컬 원본은 지우지 않음 (NAS는 백업/공유용)
set -u
RCLONE=/home/user/bin/rclone
ROOT=/home/user/workspace/parts_deep/class_estimation/door_pipeline/datasets_factory_collect
DEST=nas:Guest/weld
LOG=/home/user/workspace/parts_deep/class_estimation/door_pipeline/logs/nas_sync.log

# 중복 실행 방지 (cron과 종료-킥이 겹칠 수 있음)
exec 9>/tmp/nas_sync.lock
flock -n 9 || exit 0

mkdir -p "$(dirname "$LOG")"

shopt -s nullglob
for meta in "$ROOT"/*/*/s_*/meta.json; do
    dir=$(dirname "$meta")
    [ -e "$dir/.uploaded" ] && continue
    grep -q '"finished": true' "$meta" || continue
    rel=${dir#"$ROOT"/}
    if "$RCLONE" copy "$dir" "$DEST/$rel" \
        --transfers 4 --timeout 60s --retries 3 >>"$LOG" 2>&1; then
        touch "$dir/.uploaded"
        echo "$(date '+%F %T') OK   $rel" >>"$LOG"
    else
        echo "$(date '+%F %T') FAIL $rel (다음 주기에 재시도)" >>"$LOG"
    fi
done

# 이벤트 로그도 함께 백업 (작은 파일이라 매번 통째로)
[ -e "$ROOT/events.jsonl" ] && \
    "$RCLONE" copyto "$ROOT/events.jsonl" "$DEST/events.jsonl" >>"$LOG" 2>&1
exit 0
