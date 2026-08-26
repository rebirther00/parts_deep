#!/bin/bash
# 홀 랜드마크 검출기 학습 (labels/holes → attribute_models/hole_landmarks/model.pth)
#   사용: scripts/hole_train.sh [epochs=80]   (학습 이력은 db/door_pipeline.db에 자동 기록)
source "$(dirname "$0")/hole_common.sh"
python 16_train_hole_landmarks.py --epochs "${1:-80}" 2>&1 | grep -v Warning | tee "logs/hole_train_$(date +%Y%m%d_%H%M%S).log"
