#!/bin/bash
# 홀 랜드마크 판별기 **전용** 실시간 추론 서버 (SAM·U-Net 미사용) → http://<IP>:5004
#   사용: scripts/hole_only_inference.sh                  # ZED 카메라
#         scripts/hole_only_inference.sh <리플레이 폴더>   # 카메라 없이 검증, 예: datasets_field/E25_door_RH_s_091317
#   옵션 전달: PORT=5004 EXTRA="--fp16" scripts/hole_only_inference.sh
# 속성 파이프라인 폴백까지 쓰는 통합 서버(:5003)는 scripts/hole_inference.sh (14번).
source "$(dirname "$0")/hole_common.sh"
REPLAY=""; [ -n "$1" ] && REPLAY="--replay $1"
python 18_realtime_inference_hole.py --port "${PORT:-5004}" $REPLAY ${EXTRA:-}
