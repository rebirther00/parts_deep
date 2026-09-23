#!/bin/bash
# 신규 세션 블라인드 루틴 (2026-09-23): NAS 등록 → 샘플 pull → 세션 지표·다수결 → 오라벨 자동 재배정 → 레이더 자동 검사(확정용 크롭)
#   → 확정 전 블라인드 예측 저장(규칙·검출기·CNN).  확정 후 같은 19 명령을 다시 돌리면 채점된다.
#   scripts/new_sessions.sh 20260924 [태그]
set -e
DATE=$1; TAG=${2:-new_$DATE}; [ -z "$DATE" ] && { echo "usage: $0 YYYYMMDD [tag]"; exit 1; }
PY=/home/koceti/parts_deep/venv/parts_deep/bin/python
DP=/home/koceti/parts_deep/class_estimation/door_pipeline; DR=/home/koceti/parts_deep/class_estimation/door_partno
cd $DP
$PY db/ingest_nas.py --refresh
$PY db/pull_nas.py
$PY 20_session_drift.py
$PY db/build_dataset.py auto-relabel
$PY partno/radar_check.py
cd $DR
$PY partno_labels.py
$PY 19_blind_eval.py --since $DATE --tag $TAG
echo "→ 확정: 웹 /options 또는  $PY $DP/partno/radar_check.py --set <세션> 1|0   /  일괄: --accept-auto"
echo "→ 채점: $PY $DR/19_blind_eval.py --since $DATE --tag $TAG   (report/blind_$TAG.md)"
