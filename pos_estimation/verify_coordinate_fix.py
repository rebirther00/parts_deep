#!/usr/bin/env python3
"""
좌표계 수정 검증 스크립트
기존 데이터의 월드 좌표와 카메라 회전 정보를 사용해서
올바른 camTobj 값을 계산하고, 기존 값과 비교합니다.
"""

import os
import sys
import json
import glob
import numpy as np
import math

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset_pos_depth")

# 카메라 내재 파라미터
CAMERA_INTRINSICS = {
    "fx": 768.0,
    "fy": 768.0,
    "cx": 512.0,
    "cy": 512.0,
    "width": 1024,
    "height": 1024
}


def euler_to_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """Euler XYZ angles (degrees) → 3x3 Rotation Matrix"""
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    
    # Rotation matrices for each axis
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])
    
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])
    
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation: R = Rz * Ry * Rx (XYZ order)
    R = Rz @ Ry @ Rx
    return R


def compute_correct_camTobj(camera_pos, camera_rot_deg, object_pos):
    """올바른 카메라 좌표계에서의 객체 위치 계산 (OpenCV 좌표계)
    
    변환 과정:
    1. 월드 좌표 차이 계산
    2. 카메라 회전 역행렬 적용 (월드 → USD 카메라 좌표계)
    3. USD → OpenCV 좌표계 변환 (Y, Z 반전)
    """
    # 월드 좌표 차이
    world_diff = np.array([
        object_pos[0] - camera_pos[0],
        object_pos[1] - camera_pos[1],
        object_pos[2] - camera_pos[2]
    ])
    
    # 카메라 회전 행렬
    R_cam = euler_to_rotation_matrix(
        camera_rot_deg[0],  # roll
        camera_rot_deg[1],  # pitch
        camera_rot_deg[2]   # yaw
    )
    
    # 카메라 좌표계로 변환: R_cam^T * world_diff (USD 좌표계)
    R_cam_inv = R_cam.T
    camTobj_usd = R_cam_inv @ world_diff
    
    # USD/OpenGL → OpenCV/Depth 좌표계 변환
    # USD: Y-up, -Z forward → OpenCV: Y-down, +Z forward
    camTobj_opencv = np.array([
        camTobj_usd[0],   # X 유지
        -camTobj_usd[1],  # Y 반전
        -camTobj_usd[2]   # Z 반전
    ])
    
    return camTobj_opencv, world_diff


def depth_to_pointcloud_center(depth_path, intrinsics, bbox=None):
    """Depth에서 Point Cloud 중심점 계산
    
    Args:
        depth_path: Depth 맵 파일 경로
        intrinsics: 카메라 내재 파라미터
        bbox: 객체 영역 {"x_min", "y_min", "x_max", "y_max"} (없으면 전체)
    """
    if not os.path.exists(depth_path):
        return None, None
    
    depth = np.load(depth_path)
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    
    height, width = depth.shape
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    u = np.arange(width)
    v = np.arange(height)
    u, v = np.meshgrid(u, v)
    
    # 유효 Depth 마스크
    valid_mask = (depth > 0.01) & (depth < 100.0) & np.isfinite(depth)
    
    # bbox 영역으로 제한
    object_mask = valid_mask.copy()
    if bbox is not None and bbox.get('x_max', 0) > 0:
        x_min, y_min = int(bbox['x_min']), int(bbox['y_min'])
        x_max, y_max = int(bbox['x_max']), int(bbox['y_max'])
        
        # bbox 마스크 생성
        bbox_mask = np.zeros_like(valid_mask)
        bbox_mask[y_min:y_max, x_min:x_max] = True
        object_mask = valid_mask & bbox_mask
    
    # 전체 장면 Point Cloud
    if valid_mask.sum() > 0:
        z_all = depth[valid_mask]
        x_all = (u[valid_mask] - cx) * z_all / fx
        y_all = (v[valid_mask] - cy) * z_all / fy
        points_all = np.stack([x_all, y_all, z_all], axis=1)
        center_all = points_all.mean(axis=0)
    else:
        center_all = None
    
    # 객체 영역 Point Cloud
    if object_mask.sum() > 0:
        z_obj = depth[object_mask]
        x_obj = (u[object_mask] - cx) * z_obj / fx
        y_obj = (v[object_mask] - cy) * z_obj / fy
        points_obj = np.stack([x_obj, y_obj, z_obj], axis=1)
        center_obj = points_obj.mean(axis=0)
    else:
        center_obj = None
    
    return center_all, center_obj


def analyze_sample(class_dir, frame_idx):
    """단일 샘플 분석"""
    pose_path = os.path.join(class_dir, f"pose_{frame_idx:04d}.json")
    depth_path = os.path.join(class_dir, f"distance_to_camera_{frame_idx:04d}.npy")
    
    if not os.path.exists(pose_path):
        return None
    
    with open(pose_path, 'r') as f:
        pose = json.load(f)
    
    # 기존 데이터
    object_pos = pose['raw_pose_world']['t_xyz_m']
    camera_pos = pose['camera_pose_world']['t_xyz_m']
    camera_rot = pose['camera_pose_world']['r_xyz_deg']
    old_camTobj = pose['camTobj']['t_xyz_m']
    bbox = pose.get('bbox_2d', None)
    
    # 올바른 camTobj 계산
    correct_camTobj, world_diff = compute_correct_camTobj(camera_pos, camera_rot, object_pos)
    
    # Depth에서 중심점 계산 (전체 장면 + 객체 영역)
    depth_center_all, depth_center_obj = depth_to_pointcloud_center(depth_path, CAMERA_INTRINSICS, bbox)
    
    return {
        'object_pos': object_pos,
        'camera_pos': camera_pos,
        'camera_rot': camera_rot,
        'old_camTobj': old_camTobj,
        'world_diff': world_diff,
        'correct_camTobj': correct_camTobj,
        'depth_center_all': depth_center_all,
        'depth_center_obj': depth_center_obj,
        'bbox': bbox
    }


def main():
    print("="*70)
    print("좌표계 수정 검증")
    print("="*70)
    
    # 테스트할 클래스
    class_name = "arm_link_25_nomat"
    class_dir = os.path.join(DATASET_DIR, class_name)
    
    if not os.path.exists(class_dir):
        print(f"❌ 데이터셋 없음: {class_dir}")
        return
    
    print(f"\n테스트 클래스: {class_name}")
    
    # 여러 샘플 분석
    test_frames = [0, 10, 50, 100, 200]
    
    for frame_idx in test_frames:
        result = analyze_sample(class_dir, frame_idx)
        if result is None:
            continue
        
        print(f"\n{'='*60}")
        print(f"Frame {frame_idx}")
        print(f"{'='*60}")
        
        print(f"\n[월드 좌표]")
        print(f"  객체 위치:  ({result['object_pos'][0]:.4f}, {result['object_pos'][1]:.4f}, {result['object_pos'][2]:.4f})")
        print(f"  카메라 위치: ({result['camera_pos'][0]:.4f}, {result['camera_pos'][1]:.4f}, {result['camera_pos'][2]:.4f})")
        print(f"  카메라 회전: ({result['camera_rot'][0]:.2f}°, {result['camera_rot'][1]:.2f}°, {result['camera_rot'][2]:.2f}°)")
        
        print(f"\n[camTobj 비교]")
        old = result['old_camTobj']
        new = result['correct_camTobj']
        wd = result['world_diff']
        
        print(f"  월드 차이 (old 방식):   ({wd[0]:.4f}, {wd[1]:.4f}, {wd[2]:.4f})")
        print(f"  기존 camTobj:          ({old[0]:.4f}, {old[1]:.4f}, {old[2]:.4f})")
        print(f"  수정된 camTobj:         ({new[0]:.4f}, {new[1]:.4f}, {new[2]:.4f})")
        
        # bbox 정보
        bbox = result['bbox']
        if bbox and bbox.get('x_max', 0) > 0:
            print(f"\n[BBox]")
            print(f"  범위: ({bbox['x_min']:.0f}, {bbox['y_min']:.0f}) ~ ({bbox['x_max']:.0f}, {bbox['y_max']:.0f})")
        
        # Depth 중심 비교
        print(f"\n[Depth Point Cloud 중심]")
        
        if result['depth_center_all'] is not None:
            dc_all = result['depth_center_all']
            print(f"  전체 장면:   ({dc_all[0]:.4f}, {dc_all[1]:.4f}, {dc_all[2]:.4f})")
        
        if result['depth_center_obj'] is not None:
            dc_obj = result['depth_center_obj']
            print(f"  객체 영역:   ({dc_obj[0]:.4f}, {dc_obj[1]:.4f}, {dc_obj[2]:.4f})")
            
            # 객체 영역 기준 오차 계산
            old_error = np.linalg.norm(np.array(old) - dc_obj)
            new_error = np.linalg.norm(new - dc_obj)
            
            print(f"\n[객체 영역 Depth 중심과의 오차]")
            print(f"  기존 camTobj 오차:   {old_error:.4f}m ({old_error*1000:.2f}mm)")
            print(f"  수정된 camTobj 오차: {new_error:.4f}m ({new_error*1000:.2f}mm)")
            
            if new_error < old_error:
                improvement = (old_error - new_error) * 1000
                improvement_pct = (old_error - new_error) / old_error * 100
                print(f"  ✓ 수정 후 오차 {improvement:.2f}mm 감소! ({improvement_pct:.1f}% 개선)")
            else:
                print(f"  ⚠️ 수정 후에도 오차가 큼 - 추가 조사 필요")
        elif result['depth_center_all'] is not None:
            # bbox가 없어서 전체 장면으로 비교
            dc_all = result['depth_center_all']
            print(f"  (bbox 없음, 전체 장면 사용)")
            
            old_error = np.linalg.norm(np.array(old) - dc_all)
            new_error = np.linalg.norm(new - dc_all)
            
            print(f"\n[전체 Depth 중심과의 오차]")
            print(f"  기존 camTobj 오차:   {old_error:.4f}m")
            print(f"  수정된 camTobj 오차: {new_error:.4f}m")
    
    print("\n" + "="*70)
    print("검증 완료")
    print("="*70)
    print("""
참고사항:
- Depth Point Cloud 중심은 "보이는 면"의 중심이므로 객체 중심과 다를 수 있음
- 수정된 camTobj가 Depth 중심과 더 가깝다면 좌표계 변환이 올바름
- 완벽한 일치는 기대하기 어려움 (부분 Point Cloud vs 객체 중심)
- 데이터셋을 새로 생성하면 camTobj가 올바르게 저장됨
""")


if __name__ == "__main__":
    main()

