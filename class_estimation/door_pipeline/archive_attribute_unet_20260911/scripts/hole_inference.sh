#!/bin/bash
# 실시간 추론 서버 (홀 판별기 1순위 + 속성 파이프라인 폴백) → http://<IP>:5003
#   사용: scripts/hole_inference.sh                  # ZED 카메라
#         scripts/hole_inference.sh <리플레이 폴더>   # 카메라 없이 검증, 예: datasets_field/E25_door_RH_s_091317
#   옵션 전달: PORT=5003 EXTRA="--no_holes" scripts/hole_inference.sh
source "$(dirname "$0")/hole_common.sh"
REPLAY=""; [ -n "$1" ] && REPLAY="--replay $1"
python 14_realtime_inference_attribute.py --port "${PORT:-5003}" $REPLAY ${EXTRA:-}
