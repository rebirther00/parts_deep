#!/usr/bin/env python3
"""
ICP 결과 시각화 스크립트
- RGB 이미지 + Depth 맵
- Point Cloud 정합 결과 (3D)
- GT vs 예측 비교
"""

import os
import sys
import json
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 헤드리스 모드
import matplotlib.pyplot as plt
import math

# Open3D
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("Warning: Open3D is not installed. pip install open3d")
    OPEN3D_AVAILABLE = False
    sys.exit(1)

from PIL import Image

# 설정
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_DIR, "dataset_pos_depth")
CAD_DIR = "/home/rebirther/isaac-sim/assets"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "visualization_results")

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
    """Euler XYZ angles → 3x3 Rotation Matrix"""
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    
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
    
    R = Rz @ Ry @ Rx
    return R


def depth_to_pointcloud(depth_path, intrinsics, downsample_factor=4):
    """Depth → Point Cloud (다운샘플링 포함)"""
    if not os.path.exists(depth_path):
        return None, None
    
    depth = np.load(depth_path)
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    
    height, width = depth.shape
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    # 다운샘플링
    depth_ds = depth[::downsample_factor, ::downsample_factor]
    h_ds, w_ds = depth_ds.shape
    
    u = np.arange(0, width, downsample_factor)
    v = np.arange(0, height, downsample_factor)
    u, v = np.meshgrid(u, v)
    
    valid_mask = (depth_ds > 0.1) & (depth_ds < 50.0) & np.isfinite(depth_ds)
    
    if valid_mask.sum() == 0:
        return None, None
    
    z = depth_ds[valid_mask]
    x = (u[valid_mask] - cx) * z / fx
    y = (v[valid_mask] - cy) * z / fy
    
    points = np.stack([x, y, z], axis=1)
    
    # Open3D Point Cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd, points


def load_cad_pointcloud(class_name, num_points=5000):
    """CAD에서 Point Cloud 로드"""
    obj_path = os.path.join(CAD_DIR, f"{class_name}.obj")
    ply_path = os.path.join(CAD_DIR, f"{class_name}.ply")
    
    mesh = None
    if os.path.exists(obj_path):
        mesh = o3d.io.read_triangle_mesh(obj_path)
    elif os.path.exists(ply_path):
        mesh = o3d.io.read_triangle_mesh(ply_path)
    else:
        return None, None
    
    if not mesh.has_vertices():
        return None, None
    
    # 포인트 샘플링
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(pcd.points)
    
    return pcd, points


def compute_correct_camTobj(camera_pos, camera_rot_deg, object_pos):
    """올바른 카메라 좌표계에서의 객체 위치 계산 (OpenCV 좌표계)"""
    world_diff = np.array([
        object_pos[0] - camera_pos[0],
        object_pos[1] - camera_pos[1],
        object_pos[2] - camera_pos[2]
    ])
    
    R_cam = euler_to_rotation_matrix(
        camera_rot_deg[0], camera_rot_deg[1], camera_rot_deg[2]
    )
    
    R_cam_inv = R_cam.T
    camTobj_usd = R_cam_inv @ world_diff
    
    # USD → OpenCV 변환
    camTobj_opencv = np.array([
        camTobj_usd[0],
        -camTobj_usd[1],
        -camTobj_usd[2]
    ])
    
    return camTobj_opencv


def run_icp(source_pcd, target_pcd, threshold=0.1, max_iter=50):
    """ICP 정합 실행"""
    source_center = np.asarray(source_pcd.get_center())
    target_center = np.asarray(target_pcd.get_center())
    
    init_transform = np.eye(4)
    init_transform[:3, 3] = target_center - source_center
    
    reg_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        threshold,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    
    return reg_result.transformation, reg_result.fitness, reg_result.inlier_rmse


def visualize_sample(class_name, frame_idx, save_path=None):
    """단일 샘플 시각화"""
    class_dir = os.path.join(DATASET_DIR, class_name)
    
    rgb_path = os.path.join(class_dir, f"rgb_{frame_idx:04d}.png")
    depth_path = os.path.join(class_dir, f"distance_to_camera_{frame_idx:04d}.npy")
    pose_path = os.path.join(class_dir, f"pose_{frame_idx:04d}.json")
    
    if not all(os.path.exists(p) for p in [rgb_path, depth_path, pose_path]):
        print(f"  Warning: Files not found: {class_name} frame {frame_idx}")
        return None
    
    # 데이터 로드
    rgb_img = Image.open(rgb_path)
    depth = np.load(depth_path)
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    
    with open(pose_path, 'r') as f:
        pose = json.load(f)
    
    # GT 정보
    object_pos = pose['raw_pose_world']['t_xyz_m']
    camera_pos = pose['camera_pose_world']['t_xyz_m']
    camera_rot = pose['camera_pose_world']['r_xyz_deg']
    old_camTobj = pose['camTobj']['t_xyz_m']
    
    # 올바른 camTobj 계산
    correct_camTobj = compute_correct_camTobj(camera_pos, camera_rot, object_pos)
    
    # Point Cloud 생성
    scene_pcd, scene_points = depth_to_pointcloud(depth_path, CAMERA_INTRINSICS, downsample_factor=8)
    cad_pcd, cad_points = load_cad_pointcloud(class_name, num_points=3000)
    
    if scene_pcd is None or cad_pcd is None:
        print(f"  Warning: Point Cloud generation failed: {class_name} frame {frame_idx}")
        return None
    
    # ICP 실행
    transform, fitness, rmse = run_icp(cad_pcd, scene_pcd, threshold=0.2)
    
    # 변환된 CAD
    cad_transformed = np.asarray(cad_pcd.points) @ transform[:3, :3].T + transform[:3, 3]
    
    # 예측 위치
    pred_pos = transform[:3, 3]
    
    # 오차 계산
    old_error = np.linalg.norm(np.array(old_camTobj) - scene_points.mean(axis=0))
    new_error = np.linalg.norm(correct_camTobj - scene_points.mean(axis=0))
    icp_error = np.linalg.norm(pred_pos - correct_camTobj)
    
    # ==========================================
    # 시각화
    # ==========================================
    fig = plt.figure(figsize=(20, 12))
    
    # 1. RGB 이미지
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(rgb_img)
    ax1.set_title(f'RGB: {class_name}\nFrame {frame_idx}', fontsize=12)
    ax1.axis('off')
    
    # 2. Depth 맵
    ax2 = fig.add_subplot(2, 3, 2)
    depth_vis = np.clip(depth, 0, 20)
    im = ax2.imshow(depth_vis, cmap='viridis')
    ax2.set_title(f'Depth Map\nRange: {depth[depth>0].min():.2f}m ~ {depth.max():.2f}m', fontsize=12)
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046)
    
    # 3. Point Cloud (Scene) - 위에서 본 뷰 (XZ plane)
    ax3 = fig.add_subplot(2, 3, 3)
    # 장면 Point Cloud
    ax3.scatter(scene_points[::5, 0], scene_points[::5, 2], 
                c='blue', s=1, alpha=0.3, label='Scene (Depth)')
    # GT 위치 (수정된)
    ax3.scatter(correct_camTobj[0], correct_camTobj[2], 
                c='green', s=200, marker='*', label=f'GT (corrected)')
    # 기존 GT 위치 (잘못된)
    ax3.scatter(old_camTobj[0], old_camTobj[2], 
                c='red', s=200, marker='x', label=f'Old GT (wrong)')
    # ICP 예측
    ax3.scatter(pred_pos[0], pred_pos[2], 
                c='orange', s=200, marker='^', label=f'ICP Pred')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Top View (XZ)\nScene Point Cloud + Positions', fontsize=12)
    ax3.legend(fontsize=8)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # 4. Point Cloud (ICP 전) - Side View (XY plane)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(scene_points[::10, 0], scene_points[::10, 1], 
                c='blue', s=1, alpha=0.3, label='Scene')
    ax4.scatter(cad_points[::5, 0], cad_points[::5, 1], 
                c='red', s=1, alpha=0.3, label='CAD (Original)')
    ax4.set_xlabel('X (m)')
    ax4.set_ylabel('Y (m)')
    ax4.set_title('Side View (XY)\n(Before ICP)', fontsize=12)
    ax4.legend(fontsize=8)
    ax4.set_aspect('equal')
    ax4.grid(True, alpha=0.3)
    
    # 5. Point Cloud (ICP 후) - Side View (XY plane)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(scene_points[::10, 0], scene_points[::10, 1], 
                c='blue', s=1, alpha=0.3, label='Scene')
    ax5.scatter(cad_transformed[::5, 0], cad_transformed[::5, 1], 
                c='green', s=1, alpha=0.5, label='CAD (ICP)')
    ax5.set_xlabel('X (m)')
    ax5.set_ylabel('Y (m)')
    ax5.set_title(f'Side View (XY)\n(After ICP, Fitness={fitness:.3f})', fontsize=12)
    ax5.legend(fontsize=8)
    ax5.set_aspect('equal')
    ax5.grid(True, alpha=0.3)
    
    # 6. 오차 비교
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    
    info_text = f"""
    === Analysis Results ===
    
    Class: {class_name}
    Frame: {frame_idx}
    
    Camera Position (World): 
      ({camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f}) m
    
    Camera Rotation (World):
      ({camera_rot[0]:.1f}, {camera_rot[1]:.1f}, {camera_rot[2]:.1f}) deg
    
    === GT Comparison ===
    
    Old camTobj (Wrong):
      ({old_camTobj[0]:.3f}, {old_camTobj[1]:.3f}, {old_camTobj[2]:.3f}) m
    
    Corrected camTobj (OpenCV):
      ({correct_camTobj[0]:.3f}, {correct_camTobj[1]:.3f}, {correct_camTobj[2]:.3f}) m
    
    === ICP Results ===
    
    ICP Predicted Position:
      ({pred_pos[0]:.3f}, {pred_pos[1]:.3f}, {pred_pos[2]:.3f}) m
    
    ICP Fitness: {fitness:.4f}
    ICP RMSE: {rmse:.4f} m
    
    === Errors ===
    
    SceneCenter vs OldGT: {old_error:.3f} m ({old_error*1000:.1f} mm)
    SceneCenter vs NewGT: {new_error:.3f} m ({new_error*1000:.1f} mm)
    ICPPred vs NewGT: {icp_error:.3f} m ({icp_error*1000:.1f} mm)
    """
    
    ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    
    plt.close()
    
    return {
        'class_name': class_name,
        'frame_idx': frame_idx,
        'old_error': old_error,
        'new_error': new_error,
        'icp_error': icp_error,
        'fitness': fitness,
        'rmse': rmse
    }


def main():
    print("="*60)
    print("ICP Results Visualization")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Test samples
    test_samples = [
        ("arm_link_25_nomat", [0, 50, 100, 150, 200]),
        ("boom_link_25_nomat", [0, 50, 100, 150, 200]),
    ]
    
    all_results = []
    
    for class_name, frames in test_samples:
        print(f"\nClass: {class_name}")
        
        for frame_idx in frames:
            save_path = os.path.join(OUTPUT_DIR, f"{class_name}_frame{frame_idx:04d}.png")
            result = visualize_sample(class_name, frame_idx, save_path)
            
            if result:
                all_results.append(result)
    
    # Summary output
    if all_results:
        print("\n" + "="*60)
        print("Results Summary")
        print("="*60)
        print(f"{'Class':<25} {'Frame':>6} {'Old Err':>10} {'New Err':>10} {'ICP Err':>10} {'Fitness':>8}")
        print("-"*75)
        
        for r in all_results:
            print(f"{r['class_name']:<25} {r['frame_idx']:>6} "
                  f"{r['old_error']*1000:>8.1f}mm {r['new_error']*1000:>8.1f}mm "
                  f"{r['icp_error']*1000:>8.1f}mm {r['fitness']:>8.3f}")
        
        # 평균
        avg_old = np.mean([r['old_error'] for r in all_results]) * 1000
        avg_new = np.mean([r['new_error'] for r in all_results]) * 1000
        avg_icp = np.mean([r['icp_error'] for r in all_results]) * 1000
        avg_fit = np.mean([r['fitness'] for r in all_results])
        
        print("-"*75)
        print(f"{'Average':<25} {'':<6} {avg_old:>8.1f}mm {avg_new:>8.1f}mm {avg_icp:>8.1f}mm {avg_fit:>8.3f}")
    
    print(f"\nVisualization results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

