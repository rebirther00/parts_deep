# Isaac Sim 굴착기 부품 데이터셋 명세서

## 1. 데이터셋 개요

### 1.1 기본 정보
- **데이터셋 종류**: 2종
  - **분류/탐지용 데이터셋**: `datasets/` (기존)
  - **분류 + 6DoF 포즈용 데이터셋**: `dataset_pos/` (신규)
- **생성 도구**: Isaac Sim Replicator
- **주요 생성 스크립트**
  - 분류/탐지: `01_generate_multi_class_dataset.py`
  - 분류 + 6DoF 포즈: `pos_estimation/01_generate_mult_class_dataset_with_pos.py`

### 1.2 데이터셋 통계(공통 기본)
- **클래스별 샘플 수(기본값)**: 각 500개
- **이미지 해상도(기본값)**: 1024 × 1024 픽셀
- **이미지 형식**: PNG (RGBA, 8-bit/color)
- **배경**: 랜덤 배경 (없음/단색/팩토리)

### 1.3 포함된 클래스(현재 `datasets/` 기준 예시)
클래스는 `assets/` 폴더의 USD 파일을 자동 스캔하여 결정됩니다. 현재 `datasets/dataset_info.json` 기준 예시는 다음과 같습니다.

| 클래스 이름 | 설명 | 샘플 수 |
|-----------|------|---------|
| `arm_link_25_nomat` | 굴착기 암 링크 | 500 |
| `arm_link_30_notmat` | 굴착기 암 링크(변형) | 500 |
| `boom_link_25_nomat` | 굴착기 붐 링크 | 500 |
| `boom_link_30_notmat` | 굴착기 붐 링크(변형) | 500 |

### 1.4 데이터셋 목적
- **분류/탐지(`datasets/`)**: 굴착기 부품 객체들의 2D 바운딩 박스 탐지 및 분류 모델 학습을 위한 합성 데이터셋
- **분류 + 6DoF 포즈(`dataset_pos/`)**: 이미지로부터 부품을 분류하고, 동시에 **카메라 기준 6DoF 위치/자세(^cam T_obj)** 를 추정하는 모델 학습용 데이터셋

---

## 2. 전체 데이터셋 구조

### 2.1 디렉토리 구조
```
datasets/
├── dataset_info.json                    # 전체 데이터셋 메타데이터
├── arm_cylinder_25/                     # 암 실린더 데이터셋
│   ├── rgb_0000.png
│   ├── rgb_0001.png
│   ├── ...
│   ├── rgb_0499.png
│   ├── bounding_box_2d_tight_0000.npy
│   ├── bounding_box_2d_tight_0001.npy
│   ├── ...
│   ├── bounding_box_2d_tight_0499.npy
│   ├── bounding_box_2d_tight_labels_0000.json
│   ├── bounding_box_2d_tight_labels_0001.json
│   └── ...
├── arm_link_25/                         # 암 링크 데이터셋
│   ├── rgb_0000.png
│   ├── rgb_0001.png
│   ├── ...
│   ├── rgb_0499.png
│   ├── bounding_box_2d_tight_0000.npy
│   ├── bounding_box_2d_tight_0001.npy
│   ├── ...
│   ├── bounding_box_2d_tight_0499.npy
│   ├── bounding_box_2d_tight_labels_0000.json
│   ├── bounding_box_2d_tight_labels_0001.json
│   └── ...
└── boom_link_25/                        # 붐 링크 데이터셋
    ├── rgb_0000.png
    ├── rgb_0001.png
    ├── ...
    ├── rgb_0499.png
    ├── bounding_box_2d_tight_0000.npy
    ├── bounding_box_2d_tight_0001.npy
    ├── ...
    ├── bounding_box_2d_tight_0499.npy
    ├── bounding_box_2d_tight_labels_0000.json
    ├── bounding_box_2d_tight_labels_0001.json
    └── ...
```

### 2.2 파일 명명 규칙
각 서브데이터셋 폴더 내에서:
- **RGB 이미지**: `rgb_{프레임번호:04d}.png`
- **바운딩 박스 데이터**: `bounding_box_2d_tight_{프레임번호:04d}.npy`
- **레이블 매핑**: `bounding_box_2d_tight_labels_{프레임번호:04d}.json`

**참고**: 프레임 번호는 각 서브데이터셋마다 0000부터 0499까지 연속적으로 할당됩니다.

### 2.3 메타데이터 파일 (`dataset_info.json`)

전체 데이터셋의 메타정보를 담고 있는 JSON 파일입니다.

#### 구조
```json
{
  "dataset_name": "Excavator Parts Classification Dataset",
  "num_classes": 4,
  "images_per_class": 500,
  "total_images": 2000,
  "classes": {
    "arm_link_25_nomat": "arm_link_25_nomat",
    "arm_link_30_notmat": "arm_link_30_notmat",
    "boom_link_25_nomat": "boom_link_25_nomat",
    "boom_link_30_notmat": "boom_link_30_notmat"
  },
  "background_mode": "random",
  "background_ratios": {
    "none": 0.2,
    "solid": 0.3,
    "factory": 0.5
  },
  "created_at": "2025-12-22 14:05:08",
  "note": "배경의 유무가 학습에 큰 영향을 줍니다. background_impact_analysis.md 참조."
}
```

#### 필드 설명
- `dataset_name`: 데이터셋 이름
- `num_classes`: 클래스 개수
- `images_per_class`: 클래스별 이미지 수
- `total_images`: 전체 이미지 수
- `classes`: 클래스 이름 매핑
- `background_mode`: 배경 생성 모드
- `background_ratios`: 배경 타입별 비율
- `created_at`: 생성 일시
- `note`: 참고 사항

---

## 3. 데이터 형식 상세

### 3.1 RGB 이미지 파일 (`rgb_*.png`)

#### 형식
- **파일 형식**: PNG
- **색상 모드**: RGBA (Red, Green, Blue, Alpha)
- **비트 깊이**: 8-bit per channel
- **해상도**: 1024 × 1024 픽셀
- **인터레이스**: Non-interlaced

#### 내용
- Isaac Sim에서 렌더링된 굴착기 부품 객체의 RGB 이미지
- 다양한 카메라 각도와 위치에서 촬영된 합성 이미지
- 배경은 랜덤 모드로 생성 (없음/단색/팩토리 배경)

#### 사용 예시
```python
from PIL import Image
import numpy as np

img = Image.open('arm_cylinder_25/rgb_0075.png')
img_array = np.array(img)  # Shape: (1024, 1024, 4)
rgb_array = img_array[:, :, :3]  # Alpha 채널 제거: (1024, 1024, 3)
```

---

### 3.2 바운딩 박스 데이터 파일 (`bounding_box_2d_tight_*.npy`)

#### 형식
- **파일 형식**: NumPy structured array (`.npy`)
- **인코딩**: NumPy binary format

#### 데이터 구조
각 파일은 NumPy structured array로, 다음 필드를 가진 튜플들의 배열입니다:

| 필드명 | 데이터 타입 | 설명 | 범위/예시 |
|--------|------------|------|-----------|
| `semanticId` | `uint32` | 시맨틱 ID (클래스 인덱스) | 0, 1, ... |
| `x_min` | `int32` | 바운딩 박스 최소 X 좌표 (픽셀) | 0 ~ 1023 |
| `y_min` | `int32` | 바운딩 박스 최소 Y 좌표 (픽셀) | 0 ~ 1023 |
| `x_max` | `int32` | 바운딩 박스 최대 X 좌표 (픽셀) | 0 ~ 1023 |
| `y_max` | `int32` | 바운딩 박스 최대 Y 좌표 (픽셀) | 0 ~ 1023 |
| `occlusionRatio` | `float32` | 가림 비율 (0.0 ~ 1.0) | 0.0 (완전히 보임) ~ 1.0 (완전히 가려짐) |

#### 좌표계
- **원점**: 이미지 좌상단 (0, 0)
- **X축**: 왼쪽 → 오른쪽 (0 ~ 1023)
- **Y축**: 위 → 아래 (0 ~ 1023)
- **바운딩 박스**: `[x_min, y_min, x_max, y_max]` 형식 (포함 범위)

#### 데이터 예시
```python
import numpy as np

# 파일 로드
bbox_data = np.load('arm_cylinder_25/bounding_box_2d_tight_0075.npy', allow_pickle=True)

# Shape: (N,) - N은 해당 프레임의 바운딩 박스 개수
print(f"바운딩 박스 개수: {len(bbox_data)}")

# 첫 번째 바운딩 박스 접근
first_bbox = bbox_data[0]
print(f"Semantic ID: {first_bbox['semanticId']}")
print(f"좌표: ({first_bbox['x_min']}, {first_bbox['y_min']}, {first_bbox['x_max']}, {first_bbox['y_max']})")
print(f"가림 비율: {first_bbox['occlusionRatio']}")

# 모든 필드 접근
for field_name in bbox_data.dtype.names:
    print(f"{field_name}: {first_bbox[field_name]}")
```

---

### 3.3 레이블 매핑 파일 (`bounding_box_2d_tight_labels_*.json`)

#### 형식
- **파일 형식**: JSON
- **인코딩**: UTF-8

#### 데이터 구조
각 파일은 JSON 객체로, 시맨틱 ID(문자열 키)를 클래스 정보로 매핑합니다:

```json
{
  "0": {
    "class": "arm_cylinder_25"
  },
  "1": {
    "class": "background"
  }
}
```

#### 필드 설명
- **키**: 시맨틱 ID (문자열 형태의 숫자)
- **값**: 클래스 정보 객체
  - `class`: 클래스 이름 (문자열)

#### 클래스 매핑 (서브데이터셋별)
각 서브데이터셋마다 시맨틱 ID 0은 해당 부품 클래스를, ID 1은 배경을 나타냅니다:

| 서브데이터셋 | 시맨틱 ID 0 | 시맨틱 ID 1 |
|------------|-----------|-----------|
| `arm_cylinder_25` | `arm_cylinder_25` | `background` |
| `arm_link_25` | `arm_link_25` | `background` |
| `boom_link_25` | `boom_link_25` | `background` |

#### 사용 예시
```python
import json
import numpy as np

# 레이블 파일 로드
with open('arm_cylinder_25/bounding_box_2d_tight_labels_0075.json', 'r') as f:
    labels = json.load(f)

# 바운딩 박스 데이터 로드
bbox_data = np.load('arm_cylinder_25/bounding_box_2d_tight_0075.npy', allow_pickle=True)

# 바운딩 박스와 클래스 매칭
for bbox in bbox_data:
    semantic_id = str(bbox['semanticId'])
    class_name = labels[semantic_id]['class']
    print(f"BBox: ({bbox['x_min']}, {bbox['y_min']}, {bbox['x_max']}, {bbox['y_max']}) -> {class_name}")
```

---

## 4. 데이터 생성 방법

### 4.1 생성 프로세스
1. **USD 파일 로드**: 각 부품의 USD 파일을 Isaac Sim 스테이지에 로드
2. **바운딩 박스 계산**: 부품 객체의 3D 바운딩 박스 계산
3. **Semantics 추가**: 모든 프림에 해당 부품의 semantics 추가
4. **카메라 설정**: 부품 전체가 보이도록 카메라 거리 및 위치 계산
5. **랜덤화**: 각 서브데이터셋마다 500개 프레임에 대해 카메라 위치 랜덤화
6. **배경 생성**: 랜덤 모드로 배경 생성 (없음/단색/팩토리 배경)
7. **렌더링**: 각 프레임마다 RGB 이미지 렌더링
8. **어노테이션**: `bounding_box_2d_tight` annotator로 2D 바운딩 박스 추출
9. **저장**: BasicWriter로 이미지 및 어노테이션 데이터 저장

### 4.2 카메라 설정
- **해상도**: 1024 × 1024
- **FOV**: 60도
- **카메라 위치**: 부품 중심 기준 구면 좌표계에서 랜덤 배치
- **카메라 거리**: 부품 대각선 길이를 기준으로 계산된 최소/최대 거리 범위 내

### 4.3 조명 설정
- **조명 타입**: Dome Light
- **강도**: 1000.0
- **회전**: (270, 0, 0)

### 4.4 배경 설정
- **모드**: 랜덤
- **배경 비율**:
  - 없음 (none): 20%
  - 단색 (solid): 30%
  - 팩토리 (factory): 50%

---

## 5. 데이터 사용 가이드

### 5.1 전체 데이터셋 로더 예시 (PyTorch)
```python
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import json
import os
import glob

class ExcavatorPartsDataset(Dataset):
    """전체 굴착기 부품 데이터셋 로더"""
    
    def __init__(self, data_root, transform=None, include_background=False):
        """
        Args:
            data_root: datasets 폴더 경로
            transform: 이미지 변환 함수
            include_background: 배경 바운딩 박스 포함 여부
        """
        self.data_root = data_root
        self.transform = transform
        self.include_background = include_background
        self.samples = []
        
        # 클래스 이름 매핑
        self.class_to_idx = {
            'arm_cylinder_25': 0,
            'arm_link_25': 1,
            'boom_link_25': 2,
            'background': 3
        }
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # 모든 서브데이터셋에서 샘플 수집
        subdatasets = ['arm_cylinder_25', 'arm_link_25', 'boom_link_25']
        
        for subdataset in subdatasets:
            subdataset_dir = os.path.join(data_root, subdataset)
            if not os.path.exists(subdataset_dir):
                continue
            
            rgb_files = sorted(glob.glob(os.path.join(subdataset_dir, 'rgb_*.png')))
            
            for rgb_file in rgb_files:
                frame_num = os.path.basename(rgb_file).replace('rgb_', '').replace('.png', '')
                bbox_file = os.path.join(subdataset_dir, f'bounding_box_2d_tight_{frame_num}.npy')
                label_file = os.path.join(subdataset_dir, f'bounding_box_2d_tight_labels_{frame_num}.json')
                
                if os.path.exists(bbox_file) and os.path.exists(label_file):
                    self.samples.append({
                        'rgb': rgb_file,
                        'bbox': bbox_file,
                        'label': label_file,
                        'subdataset': subdataset
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 이미지 로드
        image = Image.open(sample['rgb']).convert('RGB')
        
        # 바운딩 박스 로드
        bboxes = np.load(sample['bbox'], allow_pickle=True)
        
        # 레이블 로드
        with open(sample['label'], 'r') as f:
            labels = json.load(f)
        
        # 바운딩 박스와 클래스 리스트 생성
        bbox_list = []
        class_list = []
        
        for bbox in bboxes:
            semantic_id = str(bbox['semanticId'])
            class_name = labels[semantic_id]['class']
            
            # 배경 필터링 (선택사항)
            if not self.include_background and class_name == 'background':
                continue
            
            bbox_list.append([
                bbox['x_min'],
                bbox['y_min'],
                bbox['x_max'],
                bbox['y_max']
            ])
            class_list.append(self.class_to_idx[class_name])
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'boxes': torch.tensor(bbox_list, dtype=torch.float32) if bbox_list else torch.zeros((0, 4)),
            'labels': torch.tensor(class_list, dtype=torch.long) if class_list else torch.zeros((0,), dtype=torch.long),
            'subdataset': sample['subdataset']
        }
```

### 5.2 단일 서브데이터셋 로더 예시
```python
class SinglePartDataset(Dataset):
    """단일 부품 데이터셋 로더"""
    
    def __init__(self, subdataset_dir, transform=None):
        self.subdataset_dir = subdataset_dir
        self.transform = transform
        self.samples = []
        
        rgb_files = sorted(glob.glob(os.path.join(subdataset_dir, 'rgb_*.png')))
        
        for rgb_file in rgb_files:
            frame_num = os.path.basename(rgb_file).replace('rgb_', '').replace('.png', '')
            bbox_file = os.path.join(subdataset_dir, f'bounding_box_2d_tight_{frame_num}.npy')
            label_file = os.path.join(subdataset_dir, f'bounding_box_2d_tight_labels_{frame_num}.json')
            
            if os.path.exists(bbox_file):
                self.samples.append({
                    'rgb': rgb_file,
                    'bbox': bbox_file,
                    'label': label_file
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = Image.open(sample['rgb']).convert('RGB')
        bboxes = np.load(sample['bbox'], allow_pickle=True)
        
        with open(sample['label'], 'r') as f:
            labels = json.load(f)
        
        bbox_list = []
        for bbox in bboxes:
            semantic_id = str(bbox['semanticId'])
            if labels[semantic_id]['class'] != 'background':
                bbox_list.append([
                    bbox['x_min'],
                    bbox['y_min'],
                    bbox['x_max'],
                    bbox['y_max']
                ])
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'boxes': torch.tensor(bbox_list, dtype=torch.float32) if bbox_list else torch.zeros((0, 4))
        }
```

### 5.3 데이터 검증 스크립트
```python
import numpy as np
import json
from PIL import Image
import os
import glob

def validate_dataset(data_root):
    """전체 데이터셋 무결성 검증"""
    errors = []
    warnings = []
    
    subdatasets = ['arm_cylinder_25', 'arm_link_25', 'boom_link_25']
    
    for subdataset in subdatasets:
        subdataset_dir = os.path.join(data_root, subdataset)
        
        if not os.path.exists(subdataset_dir):
            errors.append(f"Subdataset directory not found: {subdataset}")
            continue
        
        rgb_files = sorted(glob.glob(os.path.join(subdataset_dir, 'rgb_*.png')))
        
        print(f"\n=== Validating {subdataset} ===")
        print(f"Found {len(rgb_files)} RGB images")
        
        for rgb_file in rgb_files:
            frame_num = os.path.basename(rgb_file).replace('rgb_', '').replace('.png', '')
            bbox_file = os.path.join(subdataset_dir, f'bounding_box_2d_tight_{frame_num}.npy')
            label_file = os.path.join(subdataset_dir, f'bounding_box_2d_tight_labels_{frame_num}.json')
            
            # 파일 존재 확인
            if not os.path.exists(bbox_file):
                errors.append(f"Missing bbox file for {rgb_file}")
                continue
            if not os.path.exists(label_file):
                errors.append(f"Missing label file for {rgb_file}")
                continue
            
            # 이미지 크기 확인
            try:
                img = Image.open(rgb_file)
                if img.size != (1024, 1024):
                    errors.append(f"Invalid image size for {rgb_file}: {img.size}")
            except Exception as e:
                errors.append(f"Failed to open image {rgb_file}: {e}")
            
            # 바운딩 박스 좌표 검증
            try:
                bboxes = np.load(bbox_file, allow_pickle=True)
                with open(label_file, 'r') as f:
                    labels = json.load(f)
                
                for bbox in bboxes:
                    # 좌표 범위 확인
                    if not (0 <= bbox['x_min'] < bbox['x_max'] <= 1023):
                        errors.append(f"Invalid x coordinates in {bbox_file}")
                    if not (0 <= bbox['y_min'] < bbox['y_max'] <= 1023):
                        errors.append(f"Invalid y coordinates in {bbox_file}")
                    
                    # 레이블 매핑 확인
                    semantic_id = str(bbox['semanticId'])
                    if semantic_id not in labels:
                        errors.append(f"Missing label mapping for semanticId {semantic_id} in {label_file}")
            except Exception as e:
                errors.append(f"Failed to process {bbox_file}: {e}")
    
    return errors, warnings

# 검증 실행
if __name__ == '__main__':
    data_root = '/home/rebirther/isaac_data_output/datasets'
    errors, warnings = validate_dataset(data_root)
    
    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors[:20]:  # 처음 20개만 출력
            print(f"  - {error}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more errors")
    else:
        print("\n✅ Dataset validation passed!")
    
    if warnings:
        print(f"\n⚠️  Found {len(warnings)} warnings:")
        for warning in warnings[:10]:
            print(f"  - {warning}")
```

### 5.4 데이터셋 통계 확인 스크립트
```python
import json
import os
import glob
import numpy as np
from collections import defaultdict

def get_dataset_statistics(data_root):
    """데이터셋 통계 정보 수집"""
    stats = {
        'subdatasets': {},
        'total_samples': 0,
        'total_bboxes': 0,
        'class_distribution': defaultdict(int)
    }
    
    subdatasets = ['arm_cylinder_25', 'arm_link_25', 'boom_link_25']
    
    for subdataset in subdatasets:
        subdataset_dir = os.path.join(data_root, subdataset)
        
        if not os.path.exists(subdataset_dir):
            continue
        
        rgb_files = glob.glob(os.path.join(subdataset_dir, 'rgb_*.png'))
        bbox_count = 0
        class_counts = defaultdict(int)
        
        for rgb_file in rgb_files[:10]:  # 샘플링
            frame_num = os.path.basename(rgb_file).replace('rgb_', '').replace('.png', '')
            bbox_file = os.path.join(subdataset_dir, f'bounding_box_2d_tight_{frame_num}.npy')
            label_file = os.path.join(subdataset_dir, f'bounding_box_2d_tight_labels_{frame_num}.json')
            
            if os.path.exists(bbox_file):
                bboxes = np.load(bbox_file, allow_pickle=True)
                bbox_count += len(bboxes)
                
                with open(label_file, 'r') as f:
                    labels = json.load(f)
                
                for bbox in bboxes:
                    semantic_id = str(bbox['semanticId'])
                    class_name = labels[semantic_id]['class']
                    class_counts[class_name] += 1
        
        stats['subdatasets'][subdataset] = {
            'num_images': len(rgb_files),
            'avg_bboxes_per_image': bbox_count / min(10, len(rgb_files)) if rgb_files else 0,
            'class_distribution': dict(class_counts)
        }
        stats['total_samples'] += len(rgb_files)
    
    return stats

# 통계 출력
if __name__ == '__main__':
    data_root = '/home/rebirther/isaac_data_output/datasets'
    stats = get_dataset_statistics(data_root)
    
    print("=== Dataset Statistics ===")
    print(f"Total samples: {stats['total_samples']}")
    print("\nPer subdataset:")
    for subdataset, sub_stats in stats['subdatasets'].items():
        print(f"\n  {subdataset}:")
        print(f"    Images: {sub_stats['num_images']}")
        print(f"    Avg bboxes per image: {sub_stats['avg_bboxes_per_image']:.2f}")
        print(f"    Class distribution: {sub_stats['class_distribution']}")
```

---

## 6. 데이터 특성

### 6.1 데이터 다양성
- **카메라 각도**: 다양한 시점에서 촬영 (수평 -45~45도, 수직 15~75도)
- **카메라 거리**: 부품 크기에 비례하여 다양한 거리에서 촬영
- **조명**: 균일한 Dome Light 조명
- **배경**: 랜덤 배경 (없음/단색/팩토리)

### 6.2 데이터 제한사항
- **배경**: 일부 프레임은 단순한 배경을 가짐
- **객체 수**: 각 이미지당 주로 하나의 부품 객체만 포함
- **가림**: 일부 프레임에서 객체가 부분적으로 가려질 수 있음 (`occlusionRatio` 필드 참조)

### 6.3 데이터 품질
- **해상도**: 고해상도 (1024×1024)
- **어노테이션 정확도**: Isaac Sim의 정확한 3D-2D 프로젝션 기반
- **일관성**: 모든 프레임에 대해 동일한 형식과 구조 유지

---

## 7. 사용 시나리오

### 7.1 단일 클래스 탐지
각 서브데이터셋을 독립적으로 사용하여 특정 부품의 탐지 모델 학습

### 7.2 다중 클래스 분류
전체 데이터셋을 통합하여 다중 클래스를 구분하는 분류 모델 학습  
(클래스 목록은 `dataset_info.json`의 `classes` 필드 참조)

### 7.3 전이 학습
한 부품에서 학습한 모델을 다른 부품에 전이 학습

---

## 8. 참고 사항

### 8.1 파일 크기
- **PNG 이미지**: 약 400KB ~ 1MB (압축률에 따라 다름)
- **NPY 파일**: 약 400 ~ 500 bytes (바운딩 박스 개수에 따라 다름)
- **JSON 파일**: 약 50 ~ 100 bytes
- **전체 데이터셋 크기**: 약 750MB ~ 1.5GB (압축률에 따라 다름)

### 8.2 호환성
- **Python 버전**: Python 3.6 이상
- **필수 라이브러리**: 
  - NumPy (NPY 파일 읽기)
  - PIL/Pillow (이미지 처리)
  - JSON (표준 라이브러리)
  - PyTorch (선택사항, 데이터 로더 예시용)

### 8.3 라이선스 및 사용
- 이 데이터셋은 Isaac Sim Replicator를 사용하여 생성되었습니다.
- 학습 및 연구 목적으로 자유롭게 사용 가능합니다.

### 8.4 관련 문서
- 생성 스크립트: `generate_data.py`
- 배경 영향 분석: `background_impact_analysis.md`
- Isaac Sim Replicator 문서: [NVIDIA Omniverse 문서](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/replicator.html)

---

## 9. 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| 1.0 | 2024-12-22 | 초기 데이터셋 생성 및 명세서 작성 |

---

## 10. 문의 및 지원

데이터셋에 대한 문의사항이나 문제가 있으면 다음을 확인하세요:
- 생성 스크립트: `generate_data.py`
- 메타데이터 파일: `dataset_info.json`
- Isaac Sim Replicator 문서: [NVIDIA Omniverse 문서](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/replicator.html)

---

## 11. 분류 + 6DoF 포즈 데이터셋 (`dataset_pos/`) 명세

### 11.1 개요
이 데이터셋은 “부품 분류 + 6DoF 포즈(위치/자세)” 학습을 위해 생성됩니다.

- **출력 경로**: `/home/rebirther/isaac_data_output/dataset_pos/`
- **생성 스크립트**: `pos_estimation/01_generate_mult_class_dataset_with_pos.py`
- **기본 설정**
  - 해상도: 1024×1024
  - FOV(가정): 60deg (실카메라 intrinsics 없음 → 고정된 가상 카메라로 정의)
  - 단위: translation은 **미터(m)**
  - 좌표계: 라벨은 **카메라(optical) 기준**으로 저장

### 11.2 생성 방식(A안)
- **카메라**: 고정, 상단 사선 시점(작업대를 내려다보는 시점)
- **부품**: 작업대 위에 평평하게 놓이며 이동/회전
  - roll/pitch: 작은 범위(평평하게)로 제한
  - yaw: 0~360deg 범위에서 자유
- **배경/조명**: 기존과 동일하게 도메인 랜덤화 유지(바닥/벽 색, 조명 위치 등)

### 11.3 디렉토리 구조
```
dataset_pos/
├── dataset_info.json
└── <class_name>/
    ├── rgb_0000.png
    ├── bounding_box_2d_tight_0000.npy
    ├── bounding_box_2d_tight_labels_0000.json
    ├── pose_0000.json
    ├── ...
    └── metadata.json
```

### 11.4 `pose_####.json` (핵심 라벨)
각 이미지에 대해 카메라(optical) 기준 물체 포즈를 저장합니다.

#### 좌표계 규약
- **cam_optical**: x=오른쪽, y=아래, z=전방
- **object frame(obj)**: CAD 원점(USD 로컬 프레임) 그대로 사용

#### 구조(요약)
```json
{
  "class_name": "boom_link_30_notmat",
  "frame_idx": 0,
  "unit": "m",
  "camera": {
    "width": 1024,
    "height": 1024,
    "K_assumed": [[fx,0,cx],[0,fy,cy],[0,0,1]],
    "convention": "cam_optical: x-right, y-down, z-forward"
  },
  "pose_cam_optical_obj": {
    "t_xyz_m": [x, y, z],
    "q_xyzw": [qx, qy, qz, qw]
  },
  "raw_pose_world": {
    "t_xyz_m": [xw, yw, zw],
    "r_xyz_deg": [roll, pitch, yaw],
    "camera_pos_world_m": [cx, cy, cz],
    "camera_lookat_world_m": [tx, ty, tz]
  },
  "stage": {
    "up_axis": "Y 또는 Z",
    "meters_per_unit": 1.0
  }
}
```

#### 필드 설명
- `pose_cam_optical_obj.t_xyz_m`: 카메라 기준 위치(미터)
- `pose_cam_optical_obj.q_xyzw`: 카메라 기준 자세(쿼터니언 x,y,z,w)
- `raw_pose_world`: 디버깅/재현성을 위한 world 기준 샘플링 정보(학습에는 cam 기준 사용 권장)

### 11.5 주의사항(중요)
- 실카메라 intrinsics가 확보되면, `dataset_pos`를 **재생성**하거나, 기존 데이터로 사전학습 후 실데이터/실카메라로 **파인튜닝**을 권장합니다.
- 대칭 물체의 경우(형상이 회전 대칭), yaw가 본질적으로 모호할 수 있으므로 평가 지표(ADD-S 등) 고려가 필요합니다.

