#!/usr/bin/env python3
"""
ICP 문제 분석 스크립트
Depth Point Cloud, CAD Point Cloud, GT 좌표계 비교
"""

import os
import sys
import json
import glob
import numpy as np

# Open3D
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("⚠️  Open3D가 설치되지 않았습니다. pip install open3d")
    OPEN3D_AVAILABLE = False
    sys.exit(1)

# 설정
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset_pos_depth")
CAD_DIR = "/home/rebirther/isaac-sim/assets"

# 카메라 내재 파라미터
CAMERA_INTRINSICS = {
    "fx": 768.0,
    "fy": 768.0,
    "cx": 512.0,
    "cy": 512.0,
    "width": 1024,
    "height": 1024
}


def analyze_depth_pointcloud(depth_path, intrinsics):
    """Depth 파일에서 Point Cloud 분석"""
    if not os.path.exists(depth_path):
        print(f"  ❌ Depth 파일 없음: {depth_path}")
        return None
    
    depth = np.load(depth_path)
    print(f"\n  [Depth 분석]")
    print(f"    Shape: {depth.shape}")
    print(f"    dtype: {depth.dtype}")
    print(f"    Raw 값 범위: {depth.min():.6f} ~ {depth.max():.6f}")
    print(f"    유효 픽셀 (>0): {(depth > 0).sum()} / {depth.size}")
    
    # 유효한 값만 통계
    valid_depth = depth[(depth > 0) & np.isfinite(depth)]
    if len(valid_depth) > 0:
        print(f"    유효 Depth 범위: {valid_depth.min():.4f} ~ {valid_depth.max():.4f}")
        print(f"    유효 Depth 평균: {valid_depth.mean():.4f}")
    
    # Point Cloud 생성
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    
    height, width = depth.shape
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)
    
    valid_mask = (depth > 0.01) & (depth < 100.0) & np.isfinite(depth)
    
    if valid_mask.sum() == 0:
        print(f"    ⚠️  유효한 Point Cloud 없음")
        return None
    
    z = depth[valid_mask]
    x = (u[valid_mask] - cx) * z / fx
    y = (v[valid_mask] - cy) * z / fy
    
    points = np.stack([x, y, z], axis=1)
    
    print(f"\n  [Depth Point Cloud 분석]")
    print(f"    포인트 수: {len(points)}")
    print(f"    X 범위: {points[:, 0].min():.4f} ~ {points[:, 0].max():.4f}")
    print(f"    Y 범위: {points[:, 1].min():.4f} ~ {points[:, 1].max():.4f}")
    print(f"    Z 범위: {points[:, 2].min():.4f} ~ {points[:, 2].max():.4f}")
    center = points.mean(axis=0)
    print(f"    중심점: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    
    return points


def analyze_cad_mesh(class_name):
    """CAD 메시 분석"""
    obj_path = os.path.join(CAD_DIR, f"{class_name}.obj")
    ply_path = os.path.join(CAD_DIR, f"{class_name}.ply")
    
    mesh = None
    if os.path.exists(obj_path):
        mesh = o3d.io.read_triangle_mesh(obj_path)
        print(f"\n  [CAD 분석: {obj_path}]")
    elif os.path.exists(ply_path):
        mesh = o3d.io.read_triangle_mesh(ply_path)
        print(f"\n  [CAD 분석: {ply_path}]")
    else:
        print(f"\n  ❌ CAD 파일 없음: {obj_path}")
        return None
    
    vertices = np.asarray(mesh.vertices)
    
    if len(vertices) == 0:
        print(f"    ⚠️  CAD 정점 없음")
        return None
    
    print(f"    정점 수: {len(vertices)}")
    print(f"    X 범위: {vertices[:, 0].min():.4f} ~ {vertices[:, 0].max():.4f}")
    print(f"    Y 범위: {vertices[:, 1].min():.4f} ~ {vertices[:, 1].max():.4f}")
    print(f"    Z 범위: {vertices[:, 2].min():.4f} ~ {vertices[:, 2].max():.4f}")
    
    # 크기 계산
    x_size = vertices[:, 0].max() - vertices[:, 0].min()
    y_size = vertices[:, 1].max() - vertices[:, 1].min()
    z_size = vertices[:, 2].max() - vertices[:, 2].min()
    print(f"    전체 크기: {x_size:.4f} x {y_size:.4f} x {z_size:.4f}")
    
    center = vertices.mean(axis=0)
    print(f"    중심점: ({center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f})")
    
    # 스케일 추정
    max_coord = np.abs(vertices).max()
    if max_coord > 1000:
        print(f"    ⚠️  좌표가 매우 큼 → mm 단위로 추정 (스케일 0.001 필요)")
    elif max_coord > 100:
        print(f"    ⚠️  좌표가 큼 → cm 단위로 추정 (스케일 0.01 필요)")
    else:
        print(f"    ✓ 좌표가 적절함 → m 단위로 추정")
    
    return vertices


def analyze_gt_pose(pose_path):
    """Ground Truth pose 분석"""
    if not os.path.exists(pose_path):
        print(f"\n  ❌ Pose 파일 없음: {pose_path}")
        return None
    
    with open(pose_path, 'r') as f:
        pose = json.load(f)
    
    print(f"\n  [Ground Truth 분석]")
    
    # 월드 좌표계 객체 위치
    world_pos = pose['raw_pose_world']['t_xyz_m']
    world_rot = pose['raw_pose_world']['r_xyz_deg']
    print(f"    객체 월드 위치: ({world_pos[0]:.4f}, {world_pos[1]:.4f}, {world_pos[2]:.4f})")
    print(f"    객체 월드 회전: ({world_rot[0]:.2f}°, {world_rot[1]:.2f}°, {world_rot[2]:.2f}°)")
    
    # 카메라 위치
    cam_pos = pose['camera_pose_world']['t_xyz_m']
    cam_rot = pose['camera_pose_world']['r_xyz_deg']
    print(f"    카메라 월드 위치: ({cam_pos[0]:.4f}, {cam_pos[1]:.4f}, {cam_pos[2]:.4f})")
    print(f"    카메라 월드 회전: ({cam_rot[0]:.2f}°, {cam_rot[1]:.2f}°, {cam_rot[2]:.2f}°)")
    
    # camTobj (카메라 기준 객체)
    camTobj_t = pose['camTobj']['t_xyz_m']
    camTobj_r = pose['camTobj']['r_xyz_deg']
    print(f"    camTobj 위치 (GT): ({camTobj_t[0]:.4f}, {camTobj_t[1]:.4f}, {camTobj_t[2]:.4f})")
    print(f"    camTobj 회전 (GT): ({camTobj_r[0]:.2f}°, {camTobj_r[1]:.2f}°, {camTobj_r[2]:.2f}°)")
    
    # 거리 계산
    distance = np.sqrt(sum(x**2 for x in camTobj_t))
    print(f"    카메라-객체 거리: {distance:.4f}m")
    
    # 부품 크기
    if 'stage_info' in pose:
        part_size = pose['stage_info'].get('part_size_m', 'N/A')
        meters_per_unit = pose['stage_info'].get('meters_per_unit', 1.0)
        print(f"    부품 크기: {part_size}m")
        print(f"    meters_per_unit: {meters_per_unit}")
    
    return pose


def check_coordinate_systems():
    """좌표계 일치 여부 확인"""
    print("\n" + "="*60)
    print("좌표계 분석")
    print("="*60)
    
    print("""
    [카메라 좌표계 (일반적)]
    - X: 오른쪽
    - Y: 아래쪽  
    - Z: 앞쪽 (깊이 방향)
    
    [Isaac Sim 좌표계]
    - Y-up 또는 Z-up 가능
    - Stage 설정에 따라 다름
    
    [문제점]
    - camTobj.t_xyz_m의 Z가 음수 (-3.7) → 카메라 뒤에 있다는 의미?
    - 또는 좌표축 방향이 다를 수 있음
    - Depth에서 생성된 Point Cloud의 Z는 양수 (카메라 앞)
    """)


def run_simple_icp_test(depth_points, cad_vertices, gt_camTobj):
    """간단한 ICP 테스트"""
    print("\n" + "="*60)
    print("ICP 테스트")
    print("="*60)
    
    # Point Cloud 생성
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(depth_points)
    
    cad_pcd = o3d.geometry.PointCloud()
    cad_pcd.points = o3d.utility.Vector3dVector(cad_vertices)
    
    # 중심점
    scene_center = np.asarray(scene_pcd.get_center())
    cad_center = np.asarray(cad_pcd.get_center())
    
    print(f"\n  Scene Point Cloud 중심: ({scene_center[0]:.4f}, {scene_center[1]:.4f}, {scene_center[2]:.4f})")
    print(f"  CAD Point Cloud 중심: ({cad_center[0]:.4f}, {cad_center[1]:.4f}, {cad_center[2]:.4f})")
    print(f"  GT camTobj 위치: ({gt_camTobj[0]:.4f}, {gt_camTobj[1]:.4f}, {gt_camTobj[2]:.4f})")
    
    # 중심점 차이
    center_diff = scene_center - cad_center
    print(f"\n  중심점 차이 (Scene - CAD): ({center_diff[0]:.4f}, {center_diff[1]:.4f}, {center_diff[2]:.4f})")
    print(f"  중심점 차이 크기: {np.linalg.norm(center_diff):.4f}m")
    
    # GT와 Scene 중심점 비교
    gt_diff = scene_center - np.array(gt_camTobj)
    print(f"\n  Scene중심 vs GT 차이: ({gt_diff[0]:.4f}, {gt_diff[1]:.4f}, {gt_diff[2]:.4f})")
    print(f"  Scene중심 vs GT 거리: {np.linalg.norm(gt_diff):.4f}m")
    
    # ICP 실행
    print("\n  ICP 정합 실행...")
    
    # 초기 변환: 중심점 정렬
    init_transform = np.eye(4)
    init_transform[:3, 3] = scene_center - cad_center
    
    reg_result = o3d.pipelines.registration.registration_icp(
        cad_pcd, scene_pcd,
        0.05,  # threshold
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    
    print(f"\n  ICP 결과:")
    print(f"    Fitness: {reg_result.fitness:.4f}")
    print(f"    RMSE: {reg_result.inlier_rmse:.4f}m")
    
    # 변환 결과
    T = reg_result.transformation
    pred_pos = T[:3, 3]
    print(f"    예측 위치: ({pred_pos[0]:.4f}, {pred_pos[1]:.4f}, {pred_pos[2]:.4f})")
    
    # GT와 비교
    pos_error = np.linalg.norm(pred_pos - np.array(gt_camTobj))
    print(f"\n  ⚠️  위치 오차: {pos_error:.4f}m ({pos_error*1000:.2f}mm)")
    
    if pos_error > 5.0:
        print(f"\n  🔴 심각한 오차! 가능한 원인:")
        print(f"     1. CAD와 Depth의 스케일 불일치")
        print(f"     2. 좌표계 방향 불일치 (Y-up vs Z-up)")
        print(f"     3. CAD 원점이 객체 중심이 아닐 수 있음")
        print(f"     4. camTobj의 의미가 다를 수 있음")


def main():
    print("="*60)
    print("ICP 문제 분석 스크립트")
    print("="*60)
    
    # 테스트할 클래스와 프레임
    class_name = "arm_link_25_nomat"
    frame_idx = 0
    
    class_dir = os.path.join(DATASET_DIR, class_name)
    depth_path = os.path.join(class_dir, f"distance_to_camera_{frame_idx:04d}.npy")
    pose_path = os.path.join(class_dir, f"pose_{frame_idx:04d}.json")
    
    print(f"\n분석 대상: {class_name}, frame {frame_idx}")
    
    # 1. Depth Point Cloud 분석
    depth_points = analyze_depth_pointcloud(depth_path, CAMERA_INTRINSICS)
    
    # 2. CAD 메시 분석
    cad_vertices = analyze_cad_mesh(class_name)
    
    # 3. Ground Truth 분석
    gt_pose = analyze_gt_pose(pose_path)
    
    # 4. 좌표계 분석
    check_coordinate_systems()
    
    # 5. ICP 테스트
    if depth_points is not None and cad_vertices is not None and gt_pose is not None:
        gt_camTobj = gt_pose['camTobj']['t_xyz_m']
        run_simple_icp_test(depth_points, cad_vertices, gt_camTobj)
    
    print("\n" + "="*60)
    print("분석 완료")
    print("="*60)


if __name__ == "__main__":
    main()

