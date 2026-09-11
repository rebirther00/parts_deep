#!/bin/bash
# 배포 모델(pth) 동기화 — GitHub + Gitea LFS (README "모델 배포 동기화" 참조)
#   scripts/model_sync.sh push ["커밋 메시지"]   # 학습 PC: 배포 모델 add → commit → 양쪽 push (LFS 객체는 Gitea)
#   scripts/model_sync.sh pull                    # 추론 PC: git pull + git lfs pull + 실체 검사
#   scripts/model_sync.sh status                  # LFS 추적 파일·포인터 여부 확인
set -e
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
MODELS=(
  class_estimation/door_pipeline/attribute_models/hole_landmarks/model.pth
  class_estimation/door_pipeline/artifacts/rgbe_noaux_448_seed42_datasets_factory_v2/model.pth
)

is_pointer() { head -c 40 "$1" 2>/dev/null | grep -q '^version https://git-lfs'; }

check() {
  local bad=0
  for f in "${MODELS[@]}"; do
    if [ ! -f "$f" ]; then echo "  [없음]   $f"; bad=1
    elif is_pointer "$f"; then echo "  [포인터] $f  ← git lfs pull 필요"; bad=1
    else echo "  [OK]     $f ($(du -h "$f" | cut -f1))"; fi
  done
  return $bad
}

case "${1:-status}" in
  push)
    git add "${MODELS[@]}"
    if git diff --cached --quiet; then echo "변경된 배포 모델 없음"; exit 0; fi
    git lfs ls-files -s | grep -E "$(IFS='|'; echo "${MODELS[*]}")" || true
    git commit -m "${2:-models: 배포 모델 갱신 ($(date +%Y-%m-%d))}"
    git push origin HEAD          # pushurl 2개(GitHub+Gitea) 순차 push, LFS 객체는 .lfsconfig(Gitea)로
    ;;
  pull)
    git pull --ff-only
    git lfs pull
    echo "배포 모델 상태:"; check
    ;;
  status)
    echo "LFS 추적 파일:"; git lfs ls-files -s
    echo "배포 모델 상태:"; check || true
    ;;
  *) echo "usage: $0 {push [msg]|pull|status}"; exit 1;;
esac
