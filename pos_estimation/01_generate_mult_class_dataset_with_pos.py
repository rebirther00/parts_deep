# ==========================================
# 굴착기 부품 데이터셋 생성 스크립트 (6DoF Pose Estimation용)
# RGB + Depth + Pose 라벨 생성
# ==========================================
#
# 핵심:
# 1. RGB 이미지 + Depth 맵 동시 생성
# 2. 각 프레임마다 카메라 prim에서 월드 좌표 읽기
# 3. 카메라→오브젝트 상대 pose 계산하여 pose_####.json 저장
#
# 생성 파일:
# - rgb_####.png: RGB 이미지 (1024x1024)
# - distance_to_camera_####.npy: Depth 맵 (float32, 미터 단위)
# - pose_####.json: 6DoF 라벨 (위치 + 자세)
# ==========================================

import os
import sys
import glob
import shutil
import json
import time
import math
import numpy as np
from PIL import Image

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PROJECT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Isaac Sim 초기화
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# Isaac Sim 모듈 (App 실행 후 임포트)
import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom, Semantics, Gf

# ==========================================
# 설정
# ==========================================
ASSETS_DIR = "/home/rebirther/isaac-sim/assets"
BASE_OUTPUT_DIR = os.path.join(PROJECT_DIR, "dataset_pos_depth")  # Depth 포함 데이터셋
IMAGES_PER_CLASS = 2000  # 클래스당 이미지 수 (공장 스타일 정제 데이터)
RESOLUTION = (1024, 1024)
CLEAR_EXISTING = True
ENABLE_DEPTH = True  # Depth 맵 생성 활성화


def scan_usd_files(assets_dir):
    """assets 폴더에서 모든 USD 파일 스캔"""
    configs = {}
    usd_files = sorted(glob.glob(os.path.join(assets_dir, "*.usd")))
    for usd_path in usd_files:
        name = os.path.splitext(os.path.basename(usd_path))[0]
        configs[name] = {
            "usd_path": usd_path,
            "class_name": name,
            "display_name": name
        }
    return configs


def compute_world_aabb(stage):
    """스테이지의 모든 Mesh에 대해 월드 AABB 계산"""
    time_code = Usd.TimeCode.Default()
    xform_cache = UsdGeom.XformCache(time_code)
    
    mesh_prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    if not mesh_prims:
        return None
    
    world_min = Gf.Vec3d(float("inf"), float("inf"), float("inf"))
    world_max = Gf.Vec3d(float("-inf"), float("-inf"), float("-inf"))
    
    for prim in mesh_prims:
        mesh = UsdGeom.Mesh(prim)
        M = xform_cache.GetLocalToWorldTransform(prim)
        pts = mesh.GetPointsAttr().Get(time_code)
        if pts:
            for p in pts:
                wp = M.Transform(Gf.Vec3d(p[0], p[1], p[2]))
                world_min = Gf.Vec3d(min(world_min[0], wp[0]), min(world_min[1], wp[1]), min(world_min[2], wp[2]))
                world_max = Gf.Vec3d(max(world_max[0], wp[0]), max(world_max[1], wp[1]), max(world_max[2], wp[2]))
    
    if world_min[0] == float("inf"):
        return None
    
    size = world_max - world_min
    center = (world_min + world_max) / 2.0
    up_axis = UsdGeom.GetStageUpAxis(stage)
    axis_index = 2 if up_axis == UsdGeom.Tokens.z else (1 if up_axis == UsdGeom.Tokens.y else 0)
    
    return {
        "min": world_min,
        "max": world_max,
        "size": size,
        "center": center,
        "floor": world_min[axis_index],
        "up_axis": str(up_axis),
        "axis_index": axis_index
    }


def rotation_matrix_to_euler_xyz(rot_matrix):
    """3x3 회전 행렬 → Euler XYZ angles (degrees) 변환"""
    # rot_matrix는 Gf.Matrix3d 또는 row-major 3x3
    # XYZ 순서 (Roll, Pitch, Yaw)
    import math
    
    # Gf.Matrix3d에서 요소 추출
    r00, r01, r02 = rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2]
    r10, r11, r12 = rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2]
    r20, r21, r22 = rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2]
    
    # Gimbal lock 체크
    if abs(r20) >= 1.0 - 1e-6:
        # Gimbal lock
        yaw = 0.0
        if r20 < 0:
            pitch = math.pi / 2.0
            roll = math.atan2(r01, r02)
        else:
            pitch = -math.pi / 2.0
            roll = math.atan2(-r01, -r02)
    else:
        pitch = -math.asin(r20)
        roll = math.atan2(r21 / math.cos(pitch), r22 / math.cos(pitch))
        yaw = math.atan2(r10 / math.cos(pitch), r00 / math.cos(pitch))
    
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def get_camera_world_transform(stage, camera_prim_path):
    """카메라 prim의 월드 좌표 변환 행렬 가져오기"""
    camera_prim = stage.GetPrimAtPath(camera_prim_path)
    if not camera_prim or not camera_prim.IsValid():
        return None, None, None
    
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    world_matrix = xform_cache.GetLocalToWorldTransform(camera_prim)
    
    # 위치 추출
    position = world_matrix.ExtractTranslation()
    
    # 회전 행렬 추출 → Euler angles (degrees)
    rot_matrix = world_matrix.ExtractRotationMatrix()
    euler_xyz_deg = rotation_matrix_to_euler_xyz(rot_matrix)
    
    return (position[0], position[1], position[2]), euler_xyz_deg, rot_matrix


def compute_relative_pose(camera_pos, object_center, meters_per_unit, camera_rot_matrix=None):
    """카메라 좌표계에서 오브젝트 상대 위치 계산 (미터 단위)
    
    OpenCV/Depth 좌표계로 변환:
    - X: 오른쪽 (+)
    - Y: 아래쪽 (+)  
    - Z: 앞쪽 (+, forward)
    
    USD/OpenGL 카메라 좌표계:
    - X: 오른쪽 (+)
    - Y: 위쪽 (+)
    - Z: 뒤쪽 (+, backward)
    
    Args:
        camera_pos: 카메라 월드 위치 (x, y, z)
        object_center: 오브젝트 월드 위치 (x, y, z)
        meters_per_unit: 스케일 계수
        camera_rot_matrix: 카메라 회전 행렬 (Gf.Matrix3d) - 없으면 월드 좌표 차이만 반환
    
    Returns:
        카메라 좌표계에서의 오브젝트 위치 (x, y, z) in meters (OpenCV 좌표계)
    """
    # 1. 월드 좌표계에서의 차이 (미터 단위)
    world_diff = np.array([
        (object_center[0] - camera_pos[0]) * meters_per_unit,
        (object_center[1] - camera_pos[1]) * meters_per_unit,
        (object_center[2] - camera_pos[2]) * meters_per_unit
    ])
    
    if camera_rot_matrix is None:
        # 회전 행렬이 없으면 월드 좌표 차이만 반환 (이전 동작)
        return tuple(world_diff)
    
    # 2. 카메라 회전 행렬을 numpy 배열로 변환
    # Gf.Matrix3d는 row-major
    R_cam = np.array([
        [camera_rot_matrix[0][0], camera_rot_matrix[0][1], camera_rot_matrix[0][2]],
        [camera_rot_matrix[1][0], camera_rot_matrix[1][1], camera_rot_matrix[1][2]],
        [camera_rot_matrix[2][0], camera_rot_matrix[2][1], camera_rot_matrix[2][2]]
    ])
    
    # 3. 카메라 회전의 역행렬 적용 → 카메라 좌표계로 변환
    # R_cam은 카메라의 월드 회전이므로, R_cam^T (전치 = 역행렬)로 월드→카메라 변환
    R_cam_inv = R_cam.T
    camTobj_usd = R_cam_inv @ world_diff
    
    # 4. USD/OpenGL 좌표계 → OpenCV/Depth 좌표계 변환
    # USD: Y-up, -Z forward → OpenCV: Y-down, +Z forward
    # 변환: X' = X, Y' = -Y, Z' = -Z
    camTobj_opencv = np.array([
        camTobj_usd[0],   # X 유지
        -camTobj_usd[1],  # Y 반전
        -camTobj_usd[2]   # Z 반전
    ])
    
    return tuple(camTobj_opencv)


def find_camera_prim_path(stage):
    """스테이지에서 카메라 prim 경로 찾기"""
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            return str(prim.GetPath())
    return None


def parse_bbox_from_file(bbox_npy_path, labels_json_path, class_name):
    """bbox 파일에서 클래스에 해당하는 bbox 추출"""
    if not os.path.exists(bbox_npy_path) or not os.path.exists(labels_json_path):
        return None
    
    try:
        bbox_data = np.load(bbox_npy_path)
        with open(labels_json_path, 'r') as f:
            labels_data = json.load(f)
        
        if len(bbox_data) == 0:
            return None
        
        # idToLabels 키가 있으면 사용, 없으면 labels_data 자체가 id→label 매핑
        id_to_labels = labels_data.get("idToLabels", labels_data)
        
        best_bbox = None
        best_area = 0
        
        for box in bbox_data:
            if hasattr(box, 'dtype') and box.dtype.names:
                semantic_id = str(int(box['semanticId']))
                x_min, y_min = float(box['x_min']), float(box['y_min'])
                x_max, y_max = float(box['x_max']), float(box['y_max'])
                occlusion = float(box['occlusionRatio']) if 'occlusionRatio' in box.dtype.names else 0.0
            else:
                continue
            
            if semantic_id not in id_to_labels:
                continue
            
            label_info = id_to_labels[semantic_id]
            label_class = label_info.get("class", str(label_info)) if isinstance(label_info, dict) else str(label_info)
            
            if class_name not in label_class:
                continue
            
            area = (x_max - x_min) * (y_max - y_min)
            if area > best_area:
                best_area = area
                best_bbox = {
                    "x_min": x_min, "y_min": y_min,
                    "x_max": x_max, "y_max": y_max,
                    "center_x": (x_min + x_max) / 2.0,
                    "center_y": (y_min + y_max) / 2.0,
                    "visibility": 1.0 - occlusion
                }
        
        return best_bbox
    except:
        return None


def generate_class_dataset(part_config, class_index, total_classes):
    """한 클래스의 데이터셋 생성 (step 루프 + 카메라 위치 추적)"""
    usd_path = part_config["usd_path"]
    class_name = part_config["class_name"]
    display_name = part_config["display_name"]
    
    class_output_dir = os.path.join(BASE_OUTPUT_DIR, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"[{class_index+1}/{total_classes}] {display_name}")
    print(f"{'='*60}")
    print(f"  USD: {usd_path}")
    print(f"  출력: {class_output_dir}")
    
    if not os.path.exists(usd_path):
        print(f"  ⚠️  USD 파일 없음")
        return
    
    # 이전 클래스 정리
    if class_index > 0:
        for _ in range(10):
            simulation_app.update()
            time.sleep(0.05)
    
    # USD 로드
    print(f"  USD 로딩 중...")
    omni.usd.get_context().open_stage(usd_path)
    time.sleep(1.0)
    
    stage = omni.usd.get_context().get_stage()
    if not stage:
        print(f"  ⚠️  스테이지 로드 실패")
        return
    
    meters_per_unit_raw = UsdGeom.GetStageMetersPerUnit(stage)
    
    # metersPerUnit이 비정상적으로 작으면 (< 0.1) 1.0으로 간주
    # 이유: 일부 USD가 센티미터 단위로 모델링되어 metersPerUnit=0.01이지만,
    #       실제로는 다른 부품과 같은 크기 (약 4m)여야 함
    if meters_per_unit_raw < 0.1:
        meters_per_unit = 1.0
        print(f"  metersPerUnit: {meters_per_unit_raw} → 1.0으로 보정 (스케일 통일)")
    else:
        meters_per_unit = meters_per_unit_raw
        print(f"  metersPerUnit: {meters_per_unit}")
    
    # AABB 계산
    aabb = compute_world_aabb(stage)
    if not aabb:
        print(f"  ⚠️  AABB 계산 실패")
        return
    
    part_size = max(aabb["size"][0], aabb["size"][1], aabb["size"][2])
    part_center = (aabb["center"][0], aabb["center"][1], aabb["center"][2])
    floor_height = aabb["floor"]
    
    # 오브젝트 월드 좌표 (고정)
    object_world_pos = part_center
    object_world_pos_m = (
        object_world_pos[0] * meters_per_unit,
        object_world_pos[1] * meters_per_unit,
        object_world_pos[2] * meters_per_unit
    )
    
    print(f"  부품 크기: {part_size:.4f} ({part_size * meters_per_unit:.4f} m)")
    print(f"  부품 중심 (world): {part_center}")
    print(f"  부품 중심 (m): {object_world_pos_m}")
    
    # Semantics 추가
    print(f"  Semantics 추가 중...")
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Imageable):
            sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
            sem.CreateSemanticTypeAttr("class")
            sem.CreateSemanticDataAttr().Set(class_name)
    
    # ==========================================
    # Replicator 설정
    # ==========================================
    print(f"  Replicator 설정 중...")
    
    camera_prim_path = None
    generated_frames = 0
    
    try:
        with rep.new_layer():
            # 메시 참조
            mesh_prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
            print(f"  메시 개수: {len(mesh_prims)}")
            
            if len(mesh_prims) == 0:
                print(f"  ⚠️  메시 없음")
                return
            
            part_prim = rep.get.prims(semantics=[("class", class_name)])
            
            # 배경 생성
            floor_size = part_size * 5
            floor_plane = rep.create.plane(
                scale=(floor_size, floor_size, 1),
                position=(part_center[0], part_center[1], floor_height),
                rotation=(0, 0, 0),
                semantics=[("class", "background")]
            )
            
            back_wall = rep.create.plane(
                scale=(floor_size, floor_size * 0.5, 1),
                position=(part_center[0] - floor_size * 0.4, part_center[1], part_center[2]),
                rotation=(0, 90, 0),
                semantics=[("class", "background")]
            )
            
            # 조명 생성
            dome_light = rep.create.light(light_type="Dome", intensity=800.0, rotation=(270, 0, 0))
            point_light1 = rep.create.light(
                light_type="Sphere", intensity=50000.0,
                position=(part_center[0] + part_size, part_center[1] + part_size, part_center[2] + part_size * 2),
                scale=0.5
            )
            point_light2 = rep.create.light(
                light_type="Sphere", intensity=30000.0,
                position=(part_center[0] - part_size, part_center[1] - part_size * 0.5, part_center[2] + part_size),
                scale=0.3
            )
            print(f"  ✓ 배경/조명 생성")
            
            # 카메라 설정
            size_value = (aabb["size"][0], aabb["size"][1], aabb["size"][2])
            part_diagonal = np.sqrt(size_value[0]**2 + size_value[1]**2 + size_value[2]**2)
            camera_fov_rad = np.radians(60)
            min_camera_distance = (part_diagonal / 2.0) / np.tan(camera_fov_rad / 2.0) / 0.8
            camera_distance_min = min_camera_distance * 0.9
            camera_distance_max = min_camera_distance * 1.5
            initial_camera_distance = (camera_distance_min + camera_distance_max) / 2.0
            
            camera = rep.create.camera(
                position=(part_center[0] + initial_camera_distance * 0.7,
                          part_center[1] + initial_camera_distance * 0.5,
                          part_center[2] + initial_camera_distance * 0.5),
                look_at=part_center
            )
            render_product = rep.create.render_product(camera, resolution=RESOLUTION)
            print(f"  ✓ 카메라 생성 (거리: {initial_camera_distance:.2f})")
            
            # 프레임별 랜덤화
            with rep.trigger.on_frame(max_execs=IMAGES_PER_CLASS):
                # 카메라 위치 랜덤화
                # 사선 구도: 측면 + 윗면 동시에 보이게
                # X: 0.3~0.7 (측면), Y: 0.1~0.5 (앞쪽), Z: 0.2~0.4 (다양한 사선 각도)
                with rep.create.group([camera]):
                    rep.modify.pose(
                        position=rep.distribution.uniform(
                            (part_center[0] + camera_distance_max * 0.3,
                             part_center[1] + camera_distance_max * 0.1,
                             part_center[2] + camera_distance_max * 0.2),
                            (part_center[0] + camera_distance_max * 0.7,
                             part_center[1] + camera_distance_max * 0.5,
                             part_center[2] + camera_distance_max * 0.4)
                        ),
                        look_at=part_center
                    )
                
                # 배경 색상 랜덤화
                with floor_plane:
                    rep.randomizer.color(colors=rep.distribution.uniform((0.2, 0.2, 0.2), (0.6, 0.5, 0.4)))
                with back_wall:
                    rep.randomizer.color(colors=rep.distribution.uniform((0.4, 0.4, 0.4), (0.9, 0.9, 0.85)))
                
                # 조명 위치 랜덤화
                with point_light1:
                    rep.modify.pose(position=rep.distribution.uniform(
                        (part_center[0] + part_size * 0.5, part_center[1] - part_size, part_center[2] + part_size * 1.5),
                        (part_center[0] + part_size * 1.5, part_center[1] + part_size, part_center[2] + part_size * 3)
                    ))
                with point_light2:
                    rep.modify.pose(position=rep.distribution.uniform(
                        (part_center[0] - part_size * 1.5, part_center[1] - part_size, part_center[2] + part_size * 0.5),
                        (part_center[0] - part_size * 0.5, part_center[1] + part_size, part_center[2] + part_size * 2)
                    ))
            
            print(f"  ✓ 프레임별 랜덤화 설정")
            
            # Writer 설정 (RGB + Depth + BBox)
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(
                output_dir=class_output_dir,
                rgb=True,
                distance_to_camera=ENABLE_DEPTH,  # Depth 맵 (카메라로부터 거리)
                bounding_box_2d_tight=True
            )
            writer.attach([render_product])
            print(f"  ✓ Writer 설정 (depth={ENABLE_DEPTH})")
            
            # 시뮬레이션 준비
            print(f"\n  🎬 데이터 생성 시작 ({IMAGES_PER_CLASS}장, step 루프)...")
            for _ in range(10):
                simulation_app.update()
                time.sleep(0.1)
            
            # 카메라 prim 경로 찾기
            stage = omni.usd.get_context().get_stage()
            camera_prim_path = find_camera_prim_path(stage)
            print(f"  카메라 prim: {camera_prim_path}")
            
            # ==========================================
            # Step 루프: 각 프레임마다 카메라 위치 추적
            # ==========================================
            for frame_idx in range(IMAGES_PER_CLASS):
                # 1프레임 생성
                rep.orchestrator.step()
                simulation_app.update()
                
                # 카메라 월드 좌표 및 회전 읽기
                stage = omni.usd.get_context().get_stage()
                camera_pos, camera_euler_deg, camera_rot_matrix = get_camera_world_transform(stage, camera_prim_path)
                
                if camera_pos is None:
                    camera_pos = (0, 0, 0)
                    camera_euler_deg = (0, 0, 0)
                
                # 카메라 위치 (미터)
                camera_pos_m = (
                    camera_pos[0] * meters_per_unit,
                    camera_pos[1] * meters_per_unit,
                    camera_pos[2] * meters_per_unit
                )
                
                # 카메라 좌표계에서 오브젝트 상대 위치 (미터)
                # camera_rot_matrix를 전달하여 올바른 카메라 좌표계로 변환
                rel_pos_m = compute_relative_pose(camera_pos, object_world_pos, meters_per_unit, camera_rot_matrix)
                
                # 카메라 기준 오브젝트 상대 회전 (오브젝트 고정이므로 카메라 회전의 역)
                # 카메라가 오브젝트를 바라볼 때, 카메라 좌표계에서 오브젝트의 방향
                # 간단히: 카메라 회전의 반대 = 오브젝트가 카메라에서 어떻게 보이는지
                camTobj_euler_deg = (
                    -camera_euler_deg[0],  # roll
                    -camera_euler_deg[1],  # pitch
                    -camera_euler_deg[2]   # yaw
                )
                
                # 파일 경로
                idx_str = f"{frame_idx:04d}"
                rgb_path = os.path.join(class_output_dir, f"rgb_{idx_str}.png")
                bbox_npy_path = os.path.join(class_output_dir, f"bounding_box_2d_tight_{idx_str}.npy")
                labels_json_path = os.path.join(class_output_dir, f"bounding_box_2d_tight_labels_{idx_str}.json")
                pose_path = os.path.join(class_output_dir, f"pose_{idx_str}.json")
                
                # 파일 대기 (최대 2초)
                wait_start = time.time()
                while time.time() - wait_start < 2.0:
                    if os.path.exists(rgb_path) and os.path.getsize(rgb_path) > 1000:
                        break
                    time.sleep(0.02)
                
                # bbox 파싱
                bbox_info = parse_bbox_from_file(bbox_npy_path, labels_json_path, class_name)
                if bbox_info is None:
                    bbox_info = {"x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0, "center_x": 0, "center_y": 0, "visibility": 0}
                
                # 100장당 1장씩 crop 샘플 저장 (bbox 품질 확인용)
                if frame_idx % 100 == 0 and bbox_info["x_max"] > 0:
                    try:
                        crop_samples_dir = os.path.join(class_output_dir, "crop_samples")
                        os.makedirs(crop_samples_dir, exist_ok=True)
                        
                        # RGB 이미지 로드 및 crop
                        if os.path.exists(rgb_path):
                            img = Image.open(rgb_path)
                            x_min = int(bbox_info["x_min"])
                            y_min = int(bbox_info["y_min"])
                            x_max = int(bbox_info["x_max"])
                            y_max = int(bbox_info["y_max"])
                            
                            # crop 영역이 유효한지 확인
                            if x_max > x_min and y_max > y_min:
                                cropped = img.crop((x_min, y_min, x_max, y_max))
                                crop_filename = f"crop_{idx_str}_bbox_{x_min}_{y_min}_{x_max}_{y_max}.png"
                                cropped.save(os.path.join(crop_samples_dir, crop_filename))
                                print(f"    📷 Crop 샘플 저장: {crop_filename}")
                    except Exception as crop_err:
                        print(f"    ⚠️ Crop 저장 실패: {crop_err}")
                
                # Pose 라벨 저장
                pose_data = {
                    "class_name": class_name,
                    "frame_index": frame_idx,
                    "raw_pose_world": {
                        "t_xyz_m": list(object_world_pos_m),  # 오브젝트 월드 좌표 (고정)
                        "r_xyz_deg": [0.0, 0.0, 0.0]  # 오브젝트 회전 (고정)
                    },
                    "camera_pose_world": {
                        "t_xyz_m": list(camera_pos_m),  # 카메라 월드 좌표
                        "r_xyz_deg": list(camera_euler_deg)  # 카메라 월드 회전 (roll, pitch, yaw)
                    },
                    "camTobj": {
                        "t_xyz_m": list(rel_pos_m),  # 카메라 기준 오브젝트 상대 위치
                        "r_xyz_deg": list(camTobj_euler_deg)  # 카메라 기준 오브젝트 상대 회전
                    },
                    "bbox_2d": bbox_info,
                    "image_info": {
                        "resolution": list(RESOLUTION),
                        "rgb_file": f"rgb_{idx_str}.png",
                        "depth_file": f"distance_to_camera_{idx_str}.npy" if ENABLE_DEPTH else None
                    },
                    "stage_info": {
                        "meters_per_unit": meters_per_unit,
                        "meters_per_unit_raw": meters_per_unit_raw,
                        "scale_corrected": meters_per_unit_raw < 0.1,
                        "part_size_m": part_size * meters_per_unit
                    }
                }
                
                with open(pose_path, 'w', encoding='utf-8') as f:
                    json.dump(pose_data, f, indent=2, ensure_ascii=False)
                
                generated_frames += 1
                
                # 진행 로그 (50장마다)
                if (frame_idx + 1) % 50 == 0 or frame_idx == 0:
                    print(f"    진행: {frame_idx + 1}/{IMAGES_PER_CLASS} | 카메라: ({camera_pos_m[0]:.2f}, {camera_pos_m[1]:.2f}, {camera_pos_m[2]:.2f})m")
            
            print(f"  ✓ 이미지 생성 완료: {generated_frames}장")
            
    except Exception as e:
        print(f"  ⚠️  에러: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        print(f"  정리 중...")
        for _ in range(20):
            simulation_app.update()
            time.sleep(0.1)
    
    # 메타데이터 저장
    metadata = {
        "class_name": class_name,
        "num_images": generated_frames,
        "part_size_m": float(part_size * meters_per_unit),
        "part_center_world_m": list(object_world_pos_m),
        "meters_per_unit": meters_per_unit,
        "meters_per_unit_raw": meters_per_unit_raw,
        "scale_corrected": meters_per_unit_raw < 0.1
    }
    with open(os.path.join(class_output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ {display_name} 완료!")


# ==========================================
# 메인 실행
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("굴착기 부품 Pose Estimation 데이터셋 생성")
    print("(B안: step 루프 + 카메라 위치 추적)")
    print("="*60)
    
    parts_config = scan_usd_files(ASSETS_DIR)
    print(f"발견된 USD: {len(parts_config)}개")
    for name in parts_config:
        print(f"  - {name}")
    
    if CLEAR_EXISTING and os.path.exists(BASE_OUTPUT_DIR):
        print(f"\n기존 데이터 삭제 중: {BASE_OUTPUT_DIR}")
        shutil.rmtree(BASE_OUTPUT_DIR)
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    total_classes = len(parts_config)
    for idx, (name, config) in enumerate(parts_config.items()):
        generate_class_dataset(config, idx, total_classes)
        
        if idx < total_classes - 1:
            print("\n스테이지 정리 중...")
            rep.orchestrator.stop()
            for _ in range(30):
                simulation_app.update()
                time.sleep(0.1)
            time.sleep(1.0)
    
    # 전체 메타데이터
    dataset_info = {
        "name": "Excavator Parts 6DoF Pose Dataset (RGB + Depth)",
        "num_classes": len(parts_config),
        "images_per_class": IMAGES_PER_CLASS,
        "classes": list(parts_config.keys()),
        "depth_enabled": ENABLE_DEPTH,
        "resolution": list(RESOLUTION),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(BASE_OUTPUT_DIR, "dataset_info.json"), 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("데이터셋 생성 완료!")
    print("="*60)
    
    for name in parts_config:
        class_dir = os.path.join(BASE_OUTPUT_DIR, name)
        if os.path.exists(class_dir):
            rgb_count = len(glob.glob(os.path.join(class_dir, "rgb_*.png")))
            depth_count = len(glob.glob(os.path.join(class_dir, "distance_to_camera_*.npy")))
            pose_count = len(glob.glob(os.path.join(class_dir, "pose_*.json")))
            print(f"  {name}: RGB {rgb_count}장, Depth {depth_count}장, Pose {pose_count}개")
    
    print("\n종료하려면 Ctrl+C를 누르세요.")
    while simulation_app.is_running():
        simulation_app.update()
    
    simulation_app.close()
