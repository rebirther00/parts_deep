# 가상환경 설정 가이드

## 📋 개요

이 프로젝트는 **가상환경(venv)** 사용을 권장합니다. 가상환경을 사용하면 프로젝트별로 독립적인 Python 패키지 환경을 구성할 수 있어 버전 충돌을 방지하고 재현 가능한 환경을 유지할 수 있습니다.

## 🎯 가상환경을 사용해야 하는 이유

### ✅ 장점

1. **의존성 격리**: 프로젝트별로 다른 패키지 버전 사용 가능
2. **버전 충돌 방지**: 다른 프로젝트와 패키지 버전 충돌 없음
3. **재현성**: 동일한 환경을 쉽게 재구성 가능
4. **시스템 보호**: 시스템 Python 환경을 깨끗하게 유지
5. **프로젝트 구조**: `class_estimation`, `pos_estimation` 등 여러 하위 프로젝트가 있어 각각 다른 의존성이 필요할 수 있음

### ❌ 글로벌 설치의 단점

1. **버전 충돌**: 다른 프로젝트와 패키지 버전이 충돌할 수 있음
2. **의존성 관리 어려움**: 어떤 프로젝트가 어떤 패키지를 사용하는지 추적 어려움
3. **시스템 오염**: 시스템 Python 환경이 복잡해짐
4. **재현성 저하**: 다른 환경에서 동일한 설정 재현 어려움

## 🚀 빠른 시작

### 방법 1: 자동 설정 스크립트 사용 (권장)

```bash
# 프로젝트 루트에서 실행
cd /home/koceti/parts_deep
./setup_venv.sh
```

스크립트가 다음을 자동으로 수행합니다:
- 가상환경 생성 (`venv/parts_deep`)
- pip 업그레이드
- RTX 5090 최적화 PyTorch 설치 (CUDA 12.4+)
- requirements.txt 패키지 설치

### 방법 2: 수동 설정

```bash
# 1. 프로젝트 루트로 이동
cd /home/koceti/parts_deep

# 2. 가상환경 생성
python3 -m venv venv/parts_deep

# 3. 가상환경 활성화
source venv/parts_deep/bin/activate

# 4. pip 업그레이드
pip install --upgrade pip setuptools wheel

# 5. RTX 5090용 PyTorch 설치 (CUDA 12.4+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 6. 나머지 패키지 설치
pip install -r class_estimation/requirements.txt
```

## 📝 가상환경 사용법

### 가상환경 활성화

```bash
# 프로젝트 루트에서
source venv/parts_deep/bin/activate

# 또는 절대 경로 사용
source /home/koceti/parts_deep/venv/parts_deep/bin/activate
```

활성화되면 프롬프트 앞에 `(parts_deep)`이 표시됩니다:
```bash
(parts_deep) user@host:~/parts_deep$
```

### 가상환경 비활성화

```bash
deactivate
```

### 스크립트 실행

가상환경 활성화 후 스크립트 실행:

```bash
# 가상환경 활성화
source venv/parts_deep/bin/activate

# 학습 스크립트 실행
cd class_estimation
python 02_parts_classification_5090.py
```

## 🔧 RTX 5090 최적화 PyTorch 설치

RTX 5090은 CUDA 12.4+를 지원하므로, PyTorch를 CUDA 12.4 버전으로 설치해야 합니다:

```bash
# 가상환경 활성화 후
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

설치 확인:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## 📦 패키지 관리

### 설치된 패키지 확인

```bash
pip list
```

### requirements.txt 업데이트

```bash
# 현재 설치된 패키지를 requirements.txt로 저장
pip freeze > class_estimation/requirements.txt
```

### 새 패키지 설치

```bash
# 가상환경 활성화 후
pip install 패키지명
```

## 🗂️ 프로젝트 구조

```
parts_deep/
├── venv/
│   └── parts_deep/          # 가상환경 (이 디렉토리는 .gitignore에 추가 권장)
│       ├── bin/
│       ├── lib/
│       └── ...
├── class_estimation/
│   ├── requirements.txt
│   └── ...
├── pos_estimation/
│   └── ...
└── setup_venv.sh            # 자동 설정 스크립트
```

## ⚠️ 주의사항

1. **가상환경 디렉토리는 Git에 커밋하지 마세요**
   - `.gitignore`에 `venv/` 추가 권장

2. **가상환경 활성화 확인**
   - 스크립트 실행 전 항상 가상환경이 활성화되어 있는지 확인
   - 프롬프트에 `(parts_deep)` 표시 확인

3. **CUDA 버전 확인**
   - RTX 5090은 CUDA 12.4+ 필요
   - `nvidia-smi`로 CUDA 버전 확인

4. **다른 프로젝트와의 충돌**
   - 각 프로젝트마다 별도 가상환경 사용 권장

## 🔍 문제 해결

### 가상환경이 활성화되지 않을 때

```bash
# Python 경로 확인
which python3

# 가상환경 재생성
rm -rf venv/parts_deep
python3 -m venv venv/parts_deep
source venv/parts_deep/bin/activate
```

### PyTorch가 GPU를 인식하지 못할 때

```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch CUDA 지원 확인
python -c "import torch; print(torch.cuda.is_available())"

# CUDA 12.4용 PyTorch 재설치
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 패키지 버전 충돌

```bash
# 가상환경 재생성
rm -rf venv/parts_deep
./setup_venv.sh
```

## 📚 추가 자료

- [Python venv 공식 문서](https://docs.python.org/3/library/venv.html)
- [PyTorch 설치 가이드](https://pytorch.org/get-started/locally/)
- [RTX 5090 사양](https://www.nvidia.com/en-us/geforce/graphics-cards/rtx-5090/)
