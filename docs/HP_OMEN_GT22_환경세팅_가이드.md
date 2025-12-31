# 🖥️ HP OMEN GT22-3002KL Linux 환경 세팅 가이드

## 📋 개요

기존 개발 환경을 **HP OMEN GT22-3002KL (울트라9 + RTX 5090)** 새 PC로 마이그레이션하기 위한 Linux 설치 및 환경 세팅 가이드입니다.

---

## 🔄 환경 비교

### 기존 PC vs 새 PC (HP OMEN GT22-3002KL)

| 항목 | 기존 PC | HP OMEN GT22-3002KL |
|------|---------|---------------------|
| **CPU** | Intel Core Ultra 9 185H (22코어) | Intel Core Ultra 9 285K (24코어, 5.7GHz) |
| **GPU** | NVIDIA RTX 4070 SUPER (8GB) | NVIDIA RTX 5090 (32GB GDDR7) |
| **RAM** | 30GB | 64GB DDR5-5600 (16GB x 4) |
| **스토리지** | - | 2TB NVMe SSD + 2TB HDD |
| **PSU** | - | 1,200W 80 Plus Gold |
| **네트워크** | - | 1Gbps LAN, Wi-Fi 6E, Bluetooth |

### 기존 소프트웨어 환경

| 항목 | 버전 | 비고 |
|------|------|------|
| **OS** | Ubuntu 22.04.5 LTS | 커널 6.8.0-90 |
| **NVIDIA Driver** | 580.95.05 | CUDA 13.0 포함 |
| **Python** | 3.10.12 | 시스템 기본 |
| **ROS2** | Humble | /opt/ros/humble |
| **Isaac Sim** | 5.1.0-rc.19 | ~/isaac-sim |
| **PyTorch** | 2.9.1 | torchvision 0.24.1 |

---

## 📦 1단계: Ubuntu 22.04 LTS 설치

### 1.1 부팅 USB 준비

1. **ISO 다운로드**: https://releases.ubuntu.com/22.04/
2. **USB 제작 도구**:
   - Windows: [Rufus](https://rufus.ie/) 또는 [balenaEtcher](https://etcher.balena.io/)
   - Linux: `dd` 명령어 또는 balenaEtcher

```bash
# Linux에서 USB 제작 (예시)
sudo dd if=ubuntu-22.04.5-desktop-amd64.iso of=/dev/sdX bs=4M status=progress
sync
```

### 1.2 HP OMEN BIOS 설정

> ⚠️ HP OMEN은 F10 또는 ESC로 BIOS 진입

1. **Secure Boot**: `Disabled` (NVIDIA 드라이버 호환)
2. **CSM/Legacy Boot**: `Disabled` (UEFI 모드 사용)
3. **Fast Boot**: `Disabled` (USB 부팅 인식 위해)
4. **Boot Order**: USB를 최상위로 설정

### 1.3 Ubuntu 설치 옵션

- **설치 유형**: "디스크를 지우고 Ubuntu 설치" 또는 수동 파티션
- **권장 파티션**:
  - `/boot/efi`: 512MB (EFI System Partition)
  - `/`: 나머지 SSD (ext4)
  - HDD는 설치 후 마운트
- ✅ **"Install third-party software for graphics..."** 체크

### 1.4 설치 후 초기 업데이트

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget
```

---

## 🎮 2단계: NVIDIA RTX 5090 드라이버 설치

> ⚠️ **중요**: RTX 50 시리즈는 최신 드라이버 필수 (565+)

### 2.1 그래픽 드라이버 PPA 추가

```bash
# 그래픽 드라이버 PPA 추가
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update
```

### 2.2 사용 가능한 드라이버 확인

```bash
ubuntu-drivers devices
```

### 2.3 RTX 5090용 드라이버 설치

```bash
# RTX 5090 지원 최신 드라이버 설치 (565+ 이상)
# 출시 시점에 따라 버전이 다를 수 있음
sudo apt install nvidia-driver-565 -y

# 또는 자동 추천 드라이버 설치
sudo ubuntu-drivers autoinstall
```

### 2.4 재부팅 및 확인

```bash
sudo reboot

# 재부팅 후 드라이버 확인
nvidia-smi
```

**예상 출력**:
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 565.xx.xx              Driver Version: 565.xx.xx      CUDA Version: 13.x    |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5090        Off |   00000000:XX:00.0  On |                  N/A |
| N/A   XXC    P0             XXW /  450W |    XXXXMB /  32768MiB  |      X%      Default |
+-----------------------------------------+------------------------+----------------------+
```

---

## 🐍 3단계: Python 환경 설정

### 3.1 시스템 Python 확인

```bash
python3 --version  # 3.10.12 확인
pip3 --version
```

### 3.2 필수 시스템 패키지 설치

```bash
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libssl-dev \
    libffi-dev \
    libopencv-dev \
    cmake \
    pkg-config
```

### 3.3 pip 업그레이드

```bash
pip3 install --upgrade pip setuptools wheel
```

---

## 🤖 4단계: ROS2 Humble 설치

### 4.1 로케일 설정

```bash
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
```

### 4.2 ROS2 저장소 설정

```bash
# 필수 패키지 설치
sudo apt install software-properties-common
sudo add-apt-repository universe

# ROS2 GPG 키 추가
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

# ROS2 저장소 추가
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | \
    sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
```

### 4.3 ROS2 Humble Desktop 설치

```bash
sudo apt update
sudo apt upgrade -y

# ROS2 Humble Desktop Full 설치
sudo apt install ros-humble-desktop -y

# 개발 도구 설치
sudo apt install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-argcomplete

# rosdep 초기화
sudo rosdep init
rosdep update
```

### 4.4 ROS2 환경 테스트

```bash
source /opt/ros/humble/setup.bash
ros2 --version
ros2 topic list
```

---

## 🎮 5단계: Isaac Sim 설치

### 5.1 NVIDIA Omniverse Launcher 설치

1. **다운로드**: https://www.nvidia.com/en-us/omniverse/download/
2. **NVIDIA 계정 로그인 필요**

```bash
# 다운로드한 AppImage 실행 권한 부여
chmod +x omniverse-launcher-linux.AppImage
./omniverse-launcher-linux.AppImage
```

### 5.2 Omniverse Launcher에서 Isaac Sim 설치

1. Omniverse Launcher 실행
2. **Exchange** 탭 클릭
3. **Isaac Sim** 검색
4. **Install** 버튼 클릭 (버전 5.1.0 이상 권장)
5. 설치 경로: `~/isaac-sim` 권장

### 5.3 Isaac Sim 호환성 확인

```bash
cd ~/isaac-sim
./isaac-sim.compatibility_check.sh
```

### 5.4 Isaac Sim 실행 테스트

```bash
cd ~/isaac-sim
./isaac-sim.sh
```

> 💡 **팁**: RTX 5090의 32GB VRAM으로 대규모 시뮬레이션도 원활하게 실행 가능

---

## 🔥 6단계: PyTorch 설치 (CUDA 지원)

### 6.1 RTX 5090용 PyTorch 설치

```bash
# CUDA 12.4 호환 PyTorch 설치 (RTX 50 시리즈 지원)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 6.2 설치 확인

```bash
python3 << 'EOF'
import torch
print(f"PyTorch 버전: {torch.__version__}")
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
EOF
```

**예상 출력**:
```
PyTorch 버전: 2.x.x+cu124
CUDA 사용 가능: True
GPU 이름: NVIDIA GeForce RTX 5090
VRAM: 32.0 GB
```

---

## 📦 7단계: 프로젝트 패키지 설치

### 7.1 기존 환경 패키지 목록 (주요 패키지)

```bash
pip3 install \
    numpy==2.2.6 \
    scipy==1.15.2 \
    opencv-python==4.12.0.88 \
    matplotlib \
    pandas \
    pillow \
    pyyaml \
    flask \
    aiohttp \
    anthropic
```

### 7.2 전체 패키지 복원 (requirements.txt 사용)

```bash
# 기존 PC에서 패키지 목록 생성
pip3 freeze > ~/requirements_backup.txt

# 새 PC에서 복원
pip3 install -r ~/requirements_backup.txt
```

---

## 🛠️ 8단계: 추가 개발 도구 설치

### 8.1 NVM (Node.js 버전 관리자)

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install --lts
nvm use --lts
node --version
```

### 8.2 Git 설정

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
git config --global init.defaultBranch main
```

### 8.3 Cursor IDE 설치

```bash
# Cursor AppImage 다운로드
wget https://downloader.cursor.sh/linux/appImage/x64 -O ~/cursor.AppImage
chmod +x ~/cursor.AppImage

# 실행
~/cursor.AppImage
```

### 8.4 기타 유용한 도구

```bash
sudo apt install -y \
    htop \
    tmux \
    tree \
    neofetch \
    net-tools \
    openssh-server
```

---

## 📝 9단계: ~/.bashrc 환경 설정

### 9.1 환경 변수 추가

```bash
cat << 'EOF' >> ~/.bashrc

# ================================================================
# HP OMEN GT22-3002KL 개발 환경 설정
# ================================================================

# NVM (Node.js)
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"

# MuJoCo GL 설정 (Isaac Sim 호환)
export MUJOCO_GL=egl

# ROS2 Humble alias
alias humble='source /opt/ros/humble/setup.bash; echo "ROS2 humble activated"'

# Isaac Sim alias
alias isaacsim='cd ~/isaac-sim && ./isaac-sim.sh'

# 유용한 alias
alias ll='ls -alF'
alias la='ls -A'
alias gpu='nvidia-smi'
alias gpuwatch='watch -n 1 nvidia-smi'

EOF
```

### 9.2 환경 적용

```bash
source ~/.bashrc
```

---

## 📂 10단계: 프로젝트 마이그레이션

### 10.1 Git 저장소 클론

```bash
cd ~
git clone <your-repository-url> isaac_data_output
cd isaac_data_output
```

### 10.2 대용량 데이터 전송 (네트워크 사용)

```bash
# 기존 PC에서 새 PC로 rsync
rsync -avzP --progress \
    기존PC_IP:/home/rebirther/isaac_data_output/ \
    ~/isaac_data_output/
```

### 10.3 외장 드라이브 사용 시

```bash
# 외장 드라이브 마운트
sudo mount /dev/sdX1 /mnt

# 데이터 복사
cp -r /mnt/isaac_data_output ~/

# 마운트 해제
sudo umount /mnt
```

---

## 💾 11단계: HDD 마운트 설정 (2TB HDD)

### 11.1 HDD 파티션 확인

```bash
lsblk
sudo fdisk -l
```

### 11.2 영구 마운트 설정

```bash
# 마운트 포인트 생성
sudo mkdir -p /mnt/data

# UUID 확인
sudo blkid

# /etc/fstab에 추가 (UUID로 마운트)
echo "UUID=<HDD_UUID> /mnt/data ext4 defaults 0 2" | sudo tee -a /etc/fstab

# 마운트 테스트
sudo mount -a
df -h
```

---

## ✅ 12단계: 최종 설치 확인 체크리스트

### 12.1 확인 스크립트

```bash
#!/bin/bash
echo "=========================================="
echo "HP OMEN GT22-3002KL 환경 확인"
echo "=========================================="

echo ""
echo "1. 시스템 정보:"
echo "----------------------------------------"
lsb_release -a 2>/dev/null
echo ""

echo "2. CPU 정보:"
echo "----------------------------------------"
lscpu | grep "Model name"
echo ""

echo "3. 메모리 정보:"
echo "----------------------------------------"
free -h | head -2
echo ""

echo "4. NVIDIA 드라이버:"
echo "----------------------------------------"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo ""

echo "5. Python 버전:"
echo "----------------------------------------"
python3 --version
echo ""

echo "6. PyTorch CUDA:"
echo "----------------------------------------"
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch 미설치"
echo ""

echo "7. ROS2:"
echo "----------------------------------------"
source /opt/ros/humble/setup.bash 2>/dev/null && ros2 --version 2>/dev/null || echo "ROS2 미설치 또는 source 필요"
echo ""

echo "8. Isaac Sim:"
echo "----------------------------------------"
if [ -f ~/isaac-sim/VERSION ]; then
    echo "Isaac Sim $(cat ~/isaac-sim/VERSION)"
else
    echo "Isaac Sim 미설치"
fi
echo ""

echo "=========================================="
echo "✅ 환경 확인 완료"
echo "=========================================="
```

### 12.2 스크립트 실행

```bash
# 스크립트 저장 및 실행
cat > ~/check_environment.sh << 'SCRIPT'
# 위 스크립트 내용
SCRIPT

chmod +x ~/check_environment.sh
~/check_environment.sh
```

---

## ⚠️ RTX 5090 특별 고려사항

### 전력 관리
- **TDP**: 약 450W
- **PSU**: HP OMEN GT22는 1,200W PSU 탑재로 충분

### 드라이버 호환성
- RTX 50 시리즈는 **NVIDIA Driver 565+** 필수
- 새 드라이버 출시 시 업데이트 권장

### Isaac Sim 최적화
- RTX 5090의 32GB VRAM 활용
- Ray Tracing 성능 대폭 향상 예상

### 딥러닝 최적화
```python
# 32GB VRAM 활용 예시
import torch
torch.cuda.set_per_process_memory_fraction(0.9)  # 90% VRAM 사용
```

---

## 📚 참고 자료

- [HP OMEN GT22-3002KL 지원 페이지](https://support.hp.com/kr-ko/product/omen-45l-gt22-3002kl)
- [Ubuntu 22.04 공식 문서](https://ubuntu.com/server/docs)
- [NVIDIA Linux 드라이버](https://www.nvidia.com/Download/index.aspx)
- [ROS2 Humble 설치 가이드](https://docs.ros.org/en/humble/Installation.html)
- [Isaac Sim 문서](https://developer.nvidia.com/isaac-sim)
- [PyTorch 설치 가이드](https://pytorch.org/get-started/locally/)

---

## 📅 문서 정보

- **작성일**: 2025년 12월 31일
- **대상 하드웨어**: HP OMEN GT22-3002KL (Ultra 9 285K + RTX 5090)
- **기존 환경 소스**: Intel Ultra 9 185H + RTX 4070 SUPER

---

> 💡 **팁**: 이 문서는 `/home/rebirther/isaac_data_output/docs/` 폴더에 저장되어 있습니다.

