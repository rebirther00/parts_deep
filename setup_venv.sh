#!/bin/bash
# 가상환경 설정 스크립트
# 사용법: ./setup_venv.sh

set -e  # 에러 발생 시 스크립트 중단

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv/parts_deep"
REQUIREMENTS_FILE="${PROJECT_DIR}/class_estimation/requirements.txt"

echo "=========================================="
echo "가상환경 설정 스크립트"
echo "=========================================="
echo "프로젝트 디렉토리: ${PROJECT_DIR}"
echo "가상환경 경로: ${VENV_DIR}"
echo ""

# Python 버전 확인
echo "[1/5] Python 버전 확인..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3가 설치되어 있지 않습니다."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✅ Python ${PYTHON_VERSION} 발견"

# 가상환경 생성
echo ""
echo "[2/5] 가상환경 생성..."
if [ -d "${VENV_DIR}" ]; then
    echo "⚠️  가상환경이 이미 존재합니다: ${VENV_DIR}"
    read -p "기존 가상환경을 삭제하고 새로 만들까요? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "기존 가상환경 삭제 중..."
        rm -rf "${VENV_DIR}"
    else
        echo "기존 가상환경을 사용합니다."
    fi
fi

if [ ! -d "${VENV_DIR}" ]; then
    echo "가상환경 생성 중: ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
    echo "✅ 가상환경 생성 완료"
else
    echo "✅ 기존 가상환경 사용"
fi

# 가상환경 활성화
echo ""
echo "[3/5] 가상환경 활성화..."
source "${VENV_DIR}/bin/activate"
echo "✅ 가상환경 활성화 완료"

# pip 업그레이드
echo ""
echo "[4/5] pip 업그레이드..."
pip install --upgrade pip setuptools wheel
echo "✅ pip 업그레이드 완료"

# 패키지 설치
echo ""
echo "[5/5] 패키지 설치..."
if [ ! -f "${REQUIREMENTS_FILE}" ]; then
    echo "⚠️  requirements.txt를 찾을 수 없습니다: ${REQUIREMENTS_FILE}"
    echo "기본 패키지만 설치합니다..."
    
    # RTX 5090용 PyTorch 설치 (CUDA 12.4+)
    echo "PyTorch 설치 중 (RTX 5090 최적화)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    
    # 기본 패키지
    pip install numpy pillow scikit-learn psutil
else
    echo "requirements.txt에서 패키지 설치 중..."
    
    # RTX 5090용 PyTorch 먼저 설치 (CUDA 12.4+)
    echo "PyTorch 설치 중 (RTX 5090 최적화)..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    
    # 나머지 패키지 설치
    pip install -r "${REQUIREMENTS_FILE}"
fi

echo ""
echo "=========================================="
echo "✅ 가상환경 설정 완료!"
echo "=========================================="
echo ""
echo "가상환경 활성화 방법:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "또는 프로젝트 루트에서:"
echo "  source venv/parts_deep/bin/activate"
echo ""
echo "가상환경 비활성화:"
echo "  deactivate"
echo ""
echo "설치된 패키지 확인:"
echo "  pip list"
echo ""
