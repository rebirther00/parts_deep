#!/bin/bash
# 추론 세션을 세종 NAS(nas:Guest/weld_infer)로 업로드 — nas_sync.sh(수집용)와 같은 방식
#  - 세션 마감 시 07_factory_infer.py 가 한 번 킥(DOOR_INFER_UPLOAD=1 일 때) + cron 5분 주기로 밀린 것 보완
#  - result.json 에 "final": true 인 세션만, 업로드 성공 시 .uploaded 마커 생성(보존 정리의 기준)
#  - 로컬 png 는 서비스가 .uploaded 후 RETAIN_PNG_DAYS 지나면 지운다 (NAS 가 정본)
#  - events.jsonl·heartbeat.json 도 함께 (작은 파일이라 매번 통째로)
set -u
RCLONE=/home/user/bin/rclone
ROOT=/home/user/workspace/parts_deep/class_estimation/door_pipeline/datasets_factory_infer
DEST=${NAS_INFER_DEST:-nas:Guest/weld_infer}
LOG=/home/user/workspace/parts_deep/class_estimation/door_pipeline/logs/nas_sync_infer.log

exec 9>/tmp/nas_sync_infer.lock
flock -n 9 || exit 0

mkdir -p "$(dirname "$LOG")"
[ -d "$ROOT" ] || exit 0

shopt -s nullglob
for res in "$ROOT"/*/s_*/result.json; do
    dir=$(dirname "$res")
    [ -e "$dir/.uploaded" ] && continue
    grep -q '"final": true' "$res" || continue
    rel=${dir#"$ROOT"/}
    if "$RCLONE" copy "$dir" "$DEST/$rel" --exclude '.png_purged' \
        --transfers 4 --timeout 60s --retries 3 >>"$LOG" 2>&1; then
        touch "$dir/.uploaded"
        echo "$(date '+%F %T') OK   $rel" >>"$LOG"
    else
        echo "$(date '+%F %T') FAIL $rel (다음 주기에 재시도)" >>"$LOG"
    fi
done

for f in events.jsonl heartbeat.json; do
    [ -e "$ROOT/$f" ] && "$RCLONE" copyto "$ROOT/$f" "$DEST/$f" >>"$LOG" 2>&1
done
exit 0
