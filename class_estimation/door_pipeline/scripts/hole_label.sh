#!/bin/bash
# 홀 라벨링 도구 실행 → 브라우저 http://<IP>:8090
#   사용: scripts/hole_label.sh [클래스당 장수=15] [추가 디렉터리...]
source "$(dirname "$0")/hole_common.sh"
N=${1:-15}; shift
EXTRA=""; for d in "$@"; do EXTRA="$EXTRA --extra $d"; done
[ -z "$EXTRA" ] && [ -d datasets_field ] && EXTRA="--extra datasets_field"
python 15_label_holes.py --per-class "$N" $EXTRA --port 8090
