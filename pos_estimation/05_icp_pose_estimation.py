#!/usr/bin/env python3
# ==========================================
# ICP 기반 6DoF Pose Estimation
# 고전적 Point Cloud Registration 방식
# ==========================================
#
# 파이프라인:
# 1. RGB로 분류 → 해당 CAD 선택
# 2. Depth → Point Cloud 변환
# 3. CAD → Point Cloud 변환
# 4. ICP 정합 (Open3D)
# 5. 결과: 6DoF Pose
#
# 필요 라이브러리:
# pip install open3d numpy torch torchvision pillow
# ==========================================

import os
import sys
import json
import glob
import numpy as np
from pathlib import Path

# Open3D 임포트 (설치 필요: pip install open3d)
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("⚠️  Open3D가 설치되지 않았습니다. pip install open3d")
    OPEN3D_AVAILABLE = False

# PyTorch (분류 모델용)
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ==========================================
# 설정
# ==========================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset_pos_depth")
ARTIFACTS_DIR = os.path.join(PROJECT_DIR, "artifacts")
CAD_DIR = "/home/rebirther/isaac-sim/assets"  # CAD(USD) 파일 경로

# 카메라 내재 파라미터 (Isaac Sim 기본값, 1024x1024 해상도 기준)
CAMERA_INTRINSICS = {
    "fx": 768.0,  # focal length x (픽셀)
    "fy": 768.0,  # focal length y (픽셀)
    "cx": 512.0,  # principal point x
    "cy": 512.0,  # principal point y
    "width": 1024,
    "height": 1024
}

# ICP 설정
ICP_THRESHOLD = 0.05  # 최대 대응점 거리 (미터)
ICP_MAX_ITERATION = 50


def load_classification_model(model_path, num_classes=4):
    """분류 모델 로드"""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ 분류 모델 로드: {model_path}")
    else:
        print(f"⚠️  분류 모델 없음: {model_path}")
        return None, None
    
    model.eval()
    
    # 클래스 이름 로드
    class_names = checkpoint.get('class_names', None)
    return model, class_names


def classify_image(model, image_path, class_names, device='cpu'):
    """이미지 분류"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
    
    class_idx = pred.item()
    class_name = class_names[class_idx] if class_names else str(class_idx)
    return class_name, class_idx


def depth_to_pointcloud(depth_npy_path, intrinsics):
    """Depth 맵을 Point Cloud로 변환"""
    if not os.path.exists(depth_npy_path):
        return None
    
    depth = np.load(depth_npy_path)
    
    # Depth 맵이 2D인지 확인
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]  # 첫 번째 채널만 사용
    
    height, width = depth.shape
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    # 픽셀 좌표 그리드 생성
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)
    
    # 유효한 depth만 사용 (0 또는 inf 제외)
    valid_mask = (depth > 0.1) & (depth < 100.0) & np.isfinite(depth)
    
    # 3D 좌표 계산 (카메라 좌표계)
    z = depth[valid_mask]
    x = (u[valid_mask] - cx) * z / fx
    y = (v[valid_mask] - cy) * z / fy
    
    points = np.stack([x, y, z], axis=1)
    
    # Open3D Point Cloud 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd


def load_cad_pointcloud(cad_usd_path, num_points=10000):
    """CAD 파일(USD)에서 Point Cloud 생성
    
    참고: Open3D는 USD를 직접 읽지 못하므로,
    사전에 CAD를 .ply 또는 .pcd로 변환해두어야 함
    """
    # USD 대신 .ply 파일 찾기
    ply_path = cad_usd_path.replace('.usd', '.ply')
    obj_path = cad_usd_path.replace('.usd', '.obj')
    
    if os.path.exists(ply_path):
        mesh = o3d.io.read_triangle_mesh(ply_path)
    elif os.path.exists(obj_path):
        mesh = o3d.io.read_triangle_mesh(obj_path)
    else:
        print(f"⚠️  CAD 메시 파일 없음: {ply_path} 또는 {obj_path}")
        print(f"   USD를 PLY/OBJ로 변환 필요")
        return None
    
    # 메시에서 포인트 샘플링
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    return pcd


def create_dummy_cad_pointcloud(class_name, part_size_m=4.0):
    """테스트용: 간단한 박스 Point Cloud 생성"""
    # 실제로는 CAD 파일에서 로드해야 함
    # 여기서는 테스트용으로 박스 생성
    half_size = part_size_m / 2.0
    
    # 박스 표면의 포인트 생성
    n = 50  # 한 면당 포인트 수
    points = []
    
    # 6면에 포인트 생성
    for axis in range(3):
        for sign in [-1, 1]:
            other_axes = [i for i in range(3) if i != axis]
            for _ in range(n * n // 6):
                pt = np.random.uniform(-half_size, half_size, 3)
                pt[axis] = sign * half_size
                points.append(pt)
    
    points = np.array(points)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def run_icp(source_pcd, target_pcd, threshold=ICP_THRESHOLD, max_iteration=ICP_MAX_ITERATION):
    """ICP 정합 실행
    
    Args:
        source_pcd: CAD Point Cloud (정합할 소스)
        target_pcd: Depth에서 생성된 Point Cloud (타겟)
        threshold: 최대 대응점 거리
        max_iteration: 최대 반복 횟수
    
    Returns:
        transformation: 4x4 변환 행렬 (source → target)
        fitness: 정합 품질 (0~1)
        rmse: Root Mean Square Error
    """
    # 초기 추정: 중심점 정렬
    source_center = np.asarray(source_pcd.get_center())
    target_center = np.asarray(target_pcd.get_center())
    
    init_transform = np.eye(4)
    init_transform[:3, 3] = target_center - source_center
    
    # Point-to-Point ICP
    reg_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        threshold,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration)
    )
    
    return reg_result.transformation, reg_result.fitness, reg_result.inlier_rmse


def transformation_to_pose(transformation):
    """4x4 변환 행렬에서 위치(xyz)와 회전(roll, pitch, yaw) 추출"""
    # 위치 추출
    t = transformation[:3, 3]
    
    # 회전 행렬에서 Euler 각도 추출 (XYZ 순서)
    R = transformation[:3, :3]
    
    # Roll, Pitch, Yaw 계산
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    return {
        "position_m": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
        "rotation_deg": {
            "roll": float(np.degrees(roll)),
            "pitch": float(np.degrees(pitch)),
            "yaw": float(np.degrees(yaw))
        }
    }


def evaluate_icp_on_dataset(dataset_dir, classification_model_path):
    """데이터셋에서 ICP 평가"""
    if not OPEN3D_AVAILABLE:
        print("Open3D가 필요합니다. pip install open3d")
        return
    
    # 분류 모델 로드
    model, class_names = load_classification_model(classification_model_path)
    if model is None:
        print("분류 모델이 필요합니다.")
        return
    
    # 클래스별 폴더 스캔
    class_dirs = sorted(glob.glob(os.path.join(dataset_dir, "*")))
    class_dirs = [d for d in class_dirs if os.path.isdir(d)]
    
    if not class_dirs:
        print(f"데이터셋이 없습니다: {dataset_dir}")
        return
    
    print(f"\n{'='*60}")
    print("ICP 기반 6DoF Pose Estimation 평가")
    print(f"{'='*60}")
    print(f"데이터셋: {dataset_dir}")
    print(f"클래스 수: {len(class_dirs)}")
    
    total_samples = 0
    total_position_error = 0.0
    total_rotation_error = 0.0
    
    for class_dir in class_dirs:
        class_name = os.path.basename(class_dir)
        
        # Depth 파일 찾기
        depth_files = sorted(glob.glob(os.path.join(class_dir, "distance_to_camera_*.npy")))
        pose_files = sorted(glob.glob(os.path.join(class_dir, "pose_*.json")))
        
        if not depth_files:
            print(f"\n  {class_name}: Depth 파일 없음")
            continue
        
        print(f"\n  {class_name}: {len(depth_files)}개 샘플")
        
        # CAD Point Cloud 로드 (또는 생성)
        cad_path = os.path.join(CAD_DIR, f"{class_name}.usd")
        cad_pcd = load_cad_pointcloud(cad_path)
        
        if cad_pcd is None:
            print(f"    → CAD 파일 없음, 테스트용 더미 생성")
            cad_pcd = create_dummy_cad_pointcloud(class_name)
        
        # 일부 샘플만 테스트 (전체는 시간 오래 걸림)
        test_indices = list(range(0, min(10, len(depth_files))))
        
        class_pos_errors = []
        class_rot_errors = []
        
        for idx in test_indices:
            depth_path = depth_files[idx]
            
            # Ground Truth 로드
            frame_idx = int(os.path.basename(depth_path).split('_')[-1].split('.')[0])
            pose_path = os.path.join(class_dir, f"pose_{frame_idx:04d}.json")
            
            if not os.path.exists(pose_path):
                continue
            
            with open(pose_path, 'r') as f:
                gt_pose = json.load(f)
            
            gt_position = gt_pose['camTobj']['t_xyz_m']
            gt_rotation = gt_pose['camTobj']['r_xyz_deg']
            
            # Depth → Point Cloud
            scene_pcd = depth_to_pointcloud(depth_path, CAMERA_INTRINSICS)
            if scene_pcd is None or len(scene_pcd.points) < 100:
                continue
            
            # ICP 실행
            try:
                transform, fitness, rmse = run_icp(cad_pcd, scene_pcd)
                pred_pose = transformation_to_pose(transform)
                
                # 위치 오차 계산 (mm)
                pos_error = np.sqrt(
                    (pred_pose['position_m']['x'] - gt_position[0])**2 +
                    (pred_pose['position_m']['y'] - gt_position[1])**2 +
                    (pred_pose['position_m']['z'] - gt_position[2])**2
                ) * 1000  # 미터 → mm
                
                # 회전 오차 계산 (도)
                rot_error = np.sqrt(
                    (pred_pose['rotation_deg']['roll'] - gt_rotation[0])**2 +
                    (pred_pose['rotation_deg']['pitch'] - gt_rotation[1])**2 +
                    (pred_pose['rotation_deg']['yaw'] - gt_rotation[2])**2
                )
                
                class_pos_errors.append(pos_error)
                class_rot_errors.append(rot_error)
                
            except Exception as e:
                print(f"    ⚠️  ICP 실패 (frame {frame_idx}): {e}")
                continue
        
        if class_pos_errors:
            mean_pos_err = np.mean(class_pos_errors)
            mean_rot_err = np.mean(class_rot_errors)
            print(f"    → 평균 위치 오차: {mean_pos_err:.2f}mm")
            print(f"    → 평균 회전 오차: {mean_rot_err:.2f}°")
            
            total_samples += len(class_pos_errors)
            total_position_error += sum(class_pos_errors)
            total_rotation_error += sum(class_rot_errors)
    
    # 전체 결과
    if total_samples > 0:
        print(f"\n{'='*60}")
        print("전체 결과")
        print(f"{'='*60}")
        print(f"테스트 샘플: {total_samples}개")
        print(f"평균 위치 오차: {total_position_error / total_samples:.2f}mm")
        print(f"평균 회전 오차: {total_rotation_error / total_samples:.2f}°")


def demo_single_image(rgb_path, depth_path, class_name):
    """단일 이미지에서 ICP 데모"""
    if not OPEN3D_AVAILABLE:
        print("Open3D가 필요합니다. pip install open3d")
        return
    
    print(f"\n{'='*60}")
    print("ICP 6DoF Pose Estimation 데모")
    print(f"{'='*60}")
    print(f"RGB: {rgb_path}")
    print(f"Depth: {depth_path}")
    print(f"Class: {class_name}")
    
    # Depth → Point Cloud
    scene_pcd = depth_to_pointcloud(depth_path, CAMERA_INTRINSICS)
    if scene_pcd is None:
        print("Depth 로드 실패")
        return
    
    print(f"\n장면 Point Cloud: {len(scene_pcd.points)}개 포인트")
    
    # CAD Point Cloud (테스트용 더미)
    cad_pcd = create_dummy_cad_pointcloud(class_name)
    print(f"CAD Point Cloud: {len(cad_pcd.points)}개 포인트")
    
    # ICP 실행
    print("\nICP 정합 실행 중...")
    transform, fitness, rmse = run_icp(cad_pcd, scene_pcd)
    
    print(f"\n결과:")
    print(f"  Fitness: {fitness:.4f} (1.0 = 완벽)")
    print(f"  RMSE: {rmse:.4f}m")
    
    # Pose 추출
    pose = transformation_to_pose(transform)
    print(f"\n추정 Pose:")
    print(f"  위치: x={pose['position_m']['x']:.3f}m, y={pose['position_m']['y']:.3f}m, z={pose['position_m']['z']:.3f}m")
    print(f"  회전: roll={pose['rotation_deg']['roll']:.1f}°, pitch={pose['rotation_deg']['pitch']:.1f}°, yaw={pose['rotation_deg']['yaw']:.1f}°")
    
    # 시각화 (옵션)
    visualize = input("\n시각화하시겠습니까? (y/n): ").strip().lower() == 'y'
    if visualize:
        # 변환된 CAD
        cad_transformed = cad_pcd.transform(transform)
        
        # 색상 설정
        scene_pcd.paint_uniform_color([0.0, 0.5, 1.0])  # 파란색 (장면)
        cad_transformed.paint_uniform_color([1.0, 0.5, 0.0])  # 주황색 (CAD)
        
        o3d.visualization.draw_geometries([scene_pcd, cad_transformed],
                                          window_name="ICP Result (Blue=Scene, Orange=CAD)")


# ==========================================
# 메인 실행
# ==========================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ICP 기반 6DoF Pose Estimation")
    parser.add_argument('--mode', type=str, default='demo', choices=['demo', 'eval'],
                        help='실행 모드: demo(단일 이미지), eval(전체 평가)')
    parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR,
                        help='데이터셋 경로')
    parser.add_argument('--class_name', type=str, default='arm_link_25_nomat',
                        help='데모용 클래스 이름')
    parser.add_argument('--frame_idx', type=int, default=0,
                        help='데모용 프레임 인덱스')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        # 데모: 단일 이미지
        class_dir = os.path.join(args.dataset_dir, args.class_name)
        rgb_path = os.path.join(class_dir, f"rgb_{args.frame_idx:04d}.png")
        depth_path = os.path.join(class_dir, f"distance_to_camera_{args.frame_idx:04d}.npy")
        
        if not os.path.exists(depth_path):
            print(f"Depth 파일 없음: {depth_path}")
            print("먼저 01_generate_mult_class_dataset_with_pos.py를 실행하여 데이터셋을 생성하세요.")
        else:
            demo_single_image(rgb_path, depth_path, args.class_name)
    
    elif args.mode == 'eval':
        # 전체 평가
        classification_model_path = os.path.join(ARTIFACTS_DIR, "parts_xy_best.pt")
        evaluate_icp_on_dataset(args.dataset_dir, classification_model_path)

