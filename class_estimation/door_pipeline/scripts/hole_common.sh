#!/bin/bash
# 공통: venv 활성화 + door_pipeline 디렉터리로 이동
DOOR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HOME/parts_deep/venv/parts_deep/bin/activate"
cd "$DOOR"
