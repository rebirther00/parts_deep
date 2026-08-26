#!/bin/bash
# 홀 판별기 평가 + 샘플 이미지 생성
#   사용: scripts/hole_evaluate.sh              # test 분할 + datasets 전체 + datasets_field
#         scripts/hole_evaluate.sh <디렉터리>   # 임의 폴더(<class>/rgb_*.png), 예: datasets_field
source "$(dirname "$0")/hole_common.sh"
if [ -n "$1" ]; then python 17_evaluate_hole_classifier.py --base "$1"; else python 17_evaluate_hole_classifier.py; fi 2>&1 | grep -v Warning
python tools/make_hole_samples.py 2>&1 | grep -v Warning
echo "결과: attribute_models/hole_landmarks/eval_classifier.json, report/hole_analysis/samples/"
