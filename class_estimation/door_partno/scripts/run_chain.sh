#!/bin/bash
# 학습이 끝나는 순서대로 평가·리포트를 자동 실행 (2026-09-23). 로그: logs/chain.log
PY=/home/koceti/parts_deep/venv/parts_deep/bin/python; D=/home/koceti/parts_deep/class_estimation/door_partno; cd $D
wait_file() { until [ -f "$1" ]; do sleep 30; done; }
# ① 4채널 검출기 → 17 평가
wait_file attribute_models/hole_landmarks_bracket/eval.json
$PY -u 17_evaluate_hole_classifier.py > logs/eval_hole4.log 2>&1; $PY report/build_report.py > /dev/null 2>&1
# ② CNN 3종 → 03 평가 (각 run 의 train_log.json 이 생기면 학습 종료)
for run in partno_multi_448_seed42 partno_part14_448_seed42 partno_multi_640_seed42; do
  wait_file artifacts/$run/train_log.json; sleep 5
  $PY -u 03_evaluate_multitask.py --run $run --db-log > logs/eval_$run.log 2>&1; $PY report/build_report.py > /dev/null 2>&1
done
# ③ 5-fold 세션 CV (multi 448) → 리포트
$PY -u scripts/session_cv.py --mode multi --image_size 448 --tag cvA --epochs 30 > logs/cv_cvA.log 2>&1
$PY report/build_report.py > logs/build_report.log 2>&1
echo "chain done $(date)" >> logs/chain.log
