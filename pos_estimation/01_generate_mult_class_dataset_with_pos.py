# ==========================================
# 굴착기 부품 데이터셋 생성 스크립트 (Pose Estimation용)
# 완전 재작성: 단순하고 확실한 Xform 조작
# ==========================================
#
# 핵심 원칙:
# 1. 카메라 고정, 오브젝트 조정
# 2. Z를 먼저 충분히 들어올린 후 회전 (바닥 파묻힘 방지)
# 3. Wrapper Xform을 만들어 단순한 transform만 적용
# 4. 80% 이상 가시성 검증 (rejection sampling)
# ==========================================

import os
import sys
import shutil
import json
import time
import math
import random
import glob
import numpy as np
from pathlib import Path

# 프로젝트 경로 설정
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
from pxr import Usd, UsdGeom, Semantics, Gf, Sdf

# ==========================================
# 설정
# ==========================================
ASSETS_DIR = "/home/rebirther/isaac-sim/assets"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "dataset_pos")
IMAGES_PER_CLASS = 500
RESOLUTION = (1024, 1024)
CLEAR_EXISTING = True

# 카메라 설정 (oblique view - 35도 각도로 내려다봄)
CAMERA_ELEVATION_DEG = 35.0  # 수평에서 아래로 내려다보는 각도
CAMERA_DISTANCE_FACTOR = 2.5  # 부품 크기 대비 카메라 거리 배율

# 오브젝트 이동 범위 (미터 단위로 생각, 나중에 스테이지 단위로 변환)
XY_RANGE_M = 0.15  # XY 이동 범위 (카메라 시야 내에서 약간 이동)
Z_LIFT_FACTOR = 0.7  # 부품 크기 대비 Z 들어올림 비율 (회전해도 안 묻히게)
Z_RANGE_FACTOR = 0.3  # 추가 Z 변동 범위 (부품 크기 대비)

# 회전 범위
ROLL_RANGE_DEG = (-45.0, 45.0)
PITCH_RANGE_DEG = (-45.0, 45.0)
YAW_RANGE_DEG = (0.0, 360.0)

# 가시성 검증 설정
MIN_VISIBILITY = 0.80  # 80% 이상 가시성
MIN_BBOX_AREA_RATIO = 0.02  # 이미지 면적 대비 최소 bbox 면적 (2%)
EDGE_MARGIN = 10  # 이미지 가장자리 여백 (픽셀)
MAX_ATTEMPTS_PER_FRAME = 50  # 한 프레임 채우기 위한 최대 시도 횟수


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
    """
    스테이지의 모든 Mesh에 대해 월드 좌표계 AABB 계산.
    모든 vertex를 월드 좌표로 변환하여 정확한 min/max 계산.
    """
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
    return {
        "min": world_min,
        "max": world_max,
        "size": size,
        "center": center,
        "floor": world_min[2]  # Z-up 가정
    }


def create_wrapper_xform(stage, wrapper_path="/World/ObjectWrapper"):
    """
    새로운 Wrapper Xform 생성.
    이 Xform에 translate, rotateXYZ 연산만 적용하여 단순하게 조작.
    """
    # 기존에 있으면 삭제
    existing = stage.GetPrimAtPath(wrapper_path)
    if existing:
        stage.RemovePrim(wrapper_path)
    
    # 새 Xform 생성
    wrapper = UsdGeom.Xform.Define(stage, wrapper_path)
    
    # 명시적으로 xformOpOrder 설정: translate 먼저, 그 다음 rotateXYZ
    # 이렇게 하면 "Z를 올린 후 회전"이 됨
    xformable = UsdGeom.Xformable(wrapper.GetPrim())
    xformable.ClearXformOpOrder()
    
    # translate 연산 추가
    xformable.AddTranslateOp(opSuffix="pos")
    # rotateXYZ 연산 추가 (ZYX 순서로 적용됨 - 일반적인 Euler)
    xformable.AddRotateXYZOp(opSuffix="rot")
    
    return wrapper.GetPrim()


def reparent_all_to_wrapper(stage, wrapper_path="/World/ObjectWrapper"):
    """
    /World 아래의 모든 프림들을 Wrapper 아래로 이동.
    단, Wrapper 자신과 카메라, 조명은 제외.
    """
    wrapper_prim = stage.GetPrimAtPath(wrapper_path)
    if not wrapper_prim:
        print(f"  ⚠️  Wrapper prim not found: {wrapper_path}")
        return False
    
    world_prim = stage.GetPrimAtPath("/World")
    if not world_prim:
        # /World가 없으면 루트 프림들을 대상으로
        root_prims = [p for p in stage.GetPseudoRoot().GetChildren()]
    else:
        root_prims = list(world_prim.GetChildren())
    
    moved_count = 0
    for prim in root_prims:
        prim_path = str(prim.GetPath())
        
        # Wrapper 자신은 건너뜀
        if prim_path == wrapper_path:
            continue
        
        # 카메라, 조명은 건너뜀
        if prim.IsA(UsdGeom.Camera) or "light" in prim_path.lower() or "Light" in prim_path:
            continue
        
        # 이미 Wrapper 아래에 있으면 건너뜀
        if prim_path.startswith(wrapper_path + "/"):
            continue
        
        # Wrapper 아래로 이동
        try:
            new_path = wrapper_path + "/" + prim.GetName()
            # Sdf layer에서 직접 이동
            edit = Sdf.BatchNamespaceEdit()
            edit.Add(prim_path, new_path)
            if stage.GetRootLayer().Apply(edit):
                moved_count += 1
            else:
                print(f"    ⚠️  이동 실패: {prim_path}")
        except Exception as e:
            print(f"    ⚠️  이동 예외: {prim_path} - {e}")
    
    print(f"  ✓ {moved_count}개 프림을 Wrapper로 이동 완료")
    return moved_count > 0


def set_wrapper_pose(stage, wrapper_path, translate, rotate_xyz_deg):
    """
    Wrapper Xform의 위치와 회전 설정.
    translate: (x, y, z) - 월드 좌표
    rotate_xyz_deg: (roll, pitch, yaw) - 도 단위
    """
    wrapper = stage.GetPrimAtPath(wrapper_path)
    if not wrapper:
        return False
    
    xformable = UsdGeom.Xformable(wrapper)
    
    # translate 설정
    for op in xformable.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:translate:pos":
            op.Set(Gf.Vec3d(translate[0], translate[1], translate[2]))
        elif op.GetOpName() == "xformOp:rotateXYZ:rot":
            op.Set(Gf.Vec3f(rotate_xyz_deg[0], rotate_xyz_deg[1], rotate_xyz_deg[2]))
    
    return True


def create_fixed_camera(stage, position, look_at, cam_path="/World/FixedCamera"):
    """고정 카메라 생성"""
    # 기존에 있으면 삭제
    existing = stage.GetPrimAtPath(cam_path)
    if existing:
        stage.RemovePrim(cam_path)
    
    camera = UsdGeom.Camera.Define(stage, cam_path)
    
    # 카메라 위치 설정
    xformable = UsdGeom.Xformable(camera.GetPrim())
    xformable.ClearXformOpOrder()
    
    # look_at 방향 계산
    eye = Gf.Vec3d(position[0], position[1], position[2])
    target = Gf.Vec3d(look_at[0], look_at[1], look_at[2])
    up = Gf.Vec3d(0, 0, 1)  # Z-up
    
    # 카메라 방향 벡터
    forward = (target - eye).GetNormalized()
    right = Gf.Cross(forward, up).GetNormalized()
    actual_up = Gf.Cross(right, forward).GetNormalized()
    
    # 변환 행렬 생성
    matrix = Gf.Matrix4d()
    matrix.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0))
    matrix.SetRow(1, Gf.Vec4d(actual_up[0], actual_up[1], actual_up[2], 0))
    matrix.SetRow(2, Gf.Vec4d(-forward[0], -forward[1], -forward[2], 0))
    matrix.SetRow(3, Gf.Vec4d(eye[0], eye[1], eye[2], 1))
    
    xformable.AddTransformOp().Set(matrix)
    
    return camera.GetPrim()


def validate_bbox(bbox_data, labels_data, resolution, class_name):
    """
    BBox 데이터 검증.
    Returns: (valid, reason, bbox_info)
    - valid: True if 80% visible and good position
    - reason: 거부 이유 (valid=False일 때)
    - bbox_info: (x_min, y_min, x_max, y_max, occlusion)
    """
    if bbox_data is None or len(bbox_data) == 0:
        return False, "no_bbox_data", None
    
    # 클래스에 해당하는 bbox 찾기
    id_to_labels = {v: k for k, v in labels_data.get("idToLabels", {}).items()}
    
    best_bbox = None
    best_area = 0
    
    for box in bbox_data:
        # numpy.void (structured array) 처리
        if hasattr(box, 'dtype') and box.dtype.names:
            semantic_id = int(box['semanticId'])
            x_min = float(box['x_min'])
            y_min = float(box['y_min'])
            x_max = float(box['x_max'])
            y_max = float(box['y_max'])
            occlusion = float(box['occlusionRatio']) if 'occlusionRatio' in box.dtype.names else 0.0
        else:
            # 일반 dict 또는 tuple
            try:
                semantic_id = int(box.get('semanticId', box[0]))
                x_min = float(box.get('x_min', box[1]))
                y_min = float(box.get('y_min', box[2]))
                x_max = float(box.get('x_max', box[3]))
                y_max = float(box.get('y_max', box[4]))
                occlusion = float(box.get('occlusionRatio', 0.0))
            except:
                continue
        
        # 클래스 이름 확인
        sem_id_str = str(semantic_id)
        if sem_id_str not in id_to_labels:
            continue
        
        label_info = id_to_labels[sem_id_str]
        if class_name not in str(label_info):
            continue
        
        area = (x_max - x_min) * (y_max - y_min)
        if area > best_area:
            best_area = area
            best_bbox = (x_min, y_min, x_max, y_max, occlusion)
    
    if best_bbox is None:
        return False, "no_class_bbox", None
    
    x_min, y_min, x_max, y_max, occlusion = best_bbox
    
    # 1. 가시성 검사 (80% 이상)
    visibility = 1.0 - occlusion
    if visibility < MIN_VISIBILITY:
        return False, f"low_visibility_{visibility:.2f}", best_bbox
    
    # 2. 이미지 경계 검사
    if x_min < EDGE_MARGIN or y_min < EDGE_MARGIN:
        return False, "near_edge_min", best_bbox
    if x_max > resolution[0] - EDGE_MARGIN or y_max > resolution[1] - EDGE_MARGIN:
        return False, "near_edge_max", best_bbox
    
    # 3. bbox 면적 검사
    image_area = resolution[0] * resolution[1]
    bbox_area = (x_max - x_min) * (y_max - y_min)
    if bbox_area < image_area * MIN_BBOX_AREA_RATIO:
        return False, f"small_bbox_{bbox_area/image_area:.3f}", best_bbox
    
    return True, "ok", best_bbox


def generate_class_dataset(part_config, class_index, total_classes):
    """한 클래스의 데이터셋 생성"""
    usd_path = part_config["usd_path"]
    class_name = part_config["class_name"]
    display_name = part_config["display_name"]
    
    class_output_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"[{class_index+1}/{total_classes}] {display_name}")
    print(f"{'='*60}")
    print(f"  USD: {usd_path}")
    print(f"  출력: {class_output_dir}")
    
    if not os.path.exists(usd_path):
        print(f"  ⚠️  USD 파일 없음, 건너뜀")
        return
    
    # USD 로드
    print(f"  USD 로딩 중...")
    omni.usd.get_context().open_stage(usd_path)
    time.sleep(1.0)
    
    stage = omni.usd.get_context().get_stage()
    if not stage:
        print(f"  ⚠️  스테이지 로드 실패")
        return
    
    # 스테이지 단위 확인
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"  metersPerUnit: {meters_per_unit}")
    
    # AABB 계산
    aabb = compute_world_aabb(stage)
    if not aabb:
        print(f"  ⚠️  AABB 계산 실패")
        return
    
    part_size = max(aabb["size"][0], aabb["size"][1], aabb["size"][2])
    part_center = aabb["center"]
    floor_height = aabb["floor"]
    
    # 미터 단위로 환산
    part_size_m = part_size * meters_per_unit
    
    print(f"  부품 크기: {part_size:.4f} (단위) = {part_size_m:.4f} m")
    print(f"  부품 중심: ({part_center[0]:.4f}, {part_center[1]:.4f}, {part_center[2]:.4f})")
    print(f"  바닥 높이: {floor_height:.4f}")
    
    # XY 범위를 스테이지 단위로 변환
    xy_range = XY_RANGE_M / meters_per_unit
    z_lift = part_size * Z_LIFT_FACTOR  # 부품 크기의 70%를 들어올림
    z_range = part_size * Z_RANGE_FACTOR  # 추가 Z 변동
    
    print(f"  XY 이동 범위: ±{xy_range:.4f} 단위 (±{XY_RANGE_M:.4f} m)")
    print(f"  Z 기본 들어올림: {z_lift:.4f} 단위")
    print(f"  Z 추가 변동: ±{z_range:.4f} 단위")
    print(f"  Roll/Pitch 범위: {ROLL_RANGE_DEG}")
    print(f"  Yaw 범위: {YAW_RANGE_DEG}")
    
    # Wrapper Xform 생성
    print(f"  Wrapper Xform 생성 중...")
    wrapper_path = "/World/ObjectWrapper"
    wrapper_prim = create_wrapper_xform(stage, wrapper_path)
    
    # 모든 오브젝트를 Wrapper 아래로 이동
    print(f"  오브젝트를 Wrapper로 이동 중...")
    if not reparent_all_to_wrapper(stage, wrapper_path):
        print(f"  ⚠️  Reparent 실패, 직접 조작 시도...")
    
    # Semantics 추가
    print(f"  Semantics 추가 중...")
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Imageable):
            sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
            sem.CreateSemanticTypeAttr("class")
            sem.CreateSemanticDataAttr().Set(class_name)
    
    # 카메라 위치 계산 (oblique view)
    camera_distance = part_size * CAMERA_DISTANCE_FACTOR
    elev_rad = math.radians(CAMERA_ELEVATION_DEG)
    
    # 오브젝트의 초기 위치 (Wrapper가 이동하기 전 기준점)
    base_z = floor_height + z_lift  # 바닥에서 들어올린 높이
    
    # 카메라는 오브젝트 중심 + Z 들어올림 위치를 바라봄
    look_at = (part_center[0], part_center[1], base_z)
    
    # 카메라 위치 (X 방향에서 비스듬히 내려다봄)
    cam_x = part_center[0] + camera_distance * math.cos(elev_rad)
    cam_y = part_center[1]
    cam_z = base_z + camera_distance * math.sin(elev_rad)
    camera_pos = (cam_x, cam_y, cam_z)
    
    print(f"  카메라 위치: ({cam_x:.4f}, {cam_y:.4f}, {cam_z:.4f})")
    print(f"  카메라 타겟: ({look_at[0]:.4f}, {look_at[1]:.4f}, {look_at[2]:.4f})")
    
    # 카메라 생성
    camera_prim = create_fixed_camera(stage, camera_pos, look_at)
    
    # 조명 생성
    print(f"  조명 생성 중...")
    dome_light = rep.create.light(light_type="Dome", intensity=1000.0, rotation=(270, 0, 0))
    
    # Render product
    camera = rep.get.prim_at_path(str(camera_prim.GetPath()))
    render_product = rep.create.render_product(camera, resolution=RESOLUTION)
    
    # 임시 Writer 디렉토리
    tmp_dir = os.path.join(class_output_dir, "_tmp")
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    
    # Writer 설정
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(output_dir=tmp_dir, rgb=True, bounding_box_2d_tight=True)
    writer.attach([render_product])
    
    # 시뮬레이션 준비
    print(f"  시뮬레이션 준비 중...")
    for _ in range(10):
        simulation_app.update()
        time.sleep(0.05)
    
    # Rejection sampling 루프
    print(f"\n  🔄 Rejection Sampling 시작 (목표: {IMAGES_PER_CLASS}장)")
    
    accepted = 0
    total_attempts = 0
    reject_reasons = {}
    frame_idx = 0
    
    while accepted < IMAGES_PER_CLASS:
        total_attempts += 1
        
        if total_attempts > IMAGES_PER_CLASS * MAX_ATTEMPTS_PER_FRAME:
            print(f"  ⚠️  최대 시도 횟수 초과, {accepted}장만 생성됨")
            break
        
        # 랜덤 pose 생성
        dx = random.uniform(-xy_range, xy_range)
        dy = random.uniform(-xy_range, xy_range)
        dz = random.uniform(0, z_range)
        roll = random.uniform(*ROLL_RANGE_DEG)
        pitch = random.uniform(*PITCH_RANGE_DEG)
        yaw = random.uniform(*YAW_RANGE_DEG)
        
        # Wrapper 위치 설정
        # 핵심: Z를 먼저 충분히 올린 후 회전
        new_x = part_center[0] + dx
        new_y = part_center[1] + dy
        new_z = base_z + dz  # 이미 z_lift만큼 올라간 상태
        
        set_wrapper_pose(stage, wrapper_path, (new_x, new_y, new_z), (roll, pitch, yaw))
        
        # 렌더링
        if hasattr(rep.orchestrator, 'step'):
            rep.orchestrator.step()
        else:
            rep.orchestrator.run(num_frames=1)
        
        simulation_app.update()
        time.sleep(0.05)
        
        # BBox 파일 읽기
        bbox_npy = os.path.join(tmp_dir, f"bounding_box_2d_tight_{frame_idx:04d}.npy")
        bbox_json = os.path.join(tmp_dir, f"bounding_box_2d_tight_labels_{frame_idx:04d}.json")
        rgb_png = os.path.join(tmp_dir, f"rgb_{frame_idx:04d}.png")
        
        # 파일 대기 (최대 2초)
        wait_start = time.time()
        while time.time() - wait_start < 2.0:
            if os.path.exists(bbox_npy) and os.path.exists(bbox_json):
                break
            time.sleep(0.05)
        
        frame_idx += 1
        
        if not os.path.exists(bbox_npy) or not os.path.exists(bbox_json):
            reason = "file_not_found"
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            continue
        
        # BBox 검증
        try:
            bbox_data = np.load(bbox_npy)
            with open(bbox_json, 'r') as f:
                labels_data = json.load(f)
            
            valid, reason, bbox_info = validate_bbox(bbox_data, labels_data, RESOLUTION, class_name)
            
            if not valid:
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                continue
            
            # RGB 파일 대기
            wait_start = time.time()
            while time.time() - wait_start < 2.0:
                if os.path.exists(rgb_png) and os.path.getsize(rgb_png) > 1000:
                    break
                time.sleep(0.05)
            
            if not os.path.exists(rgb_png) or os.path.getsize(rgb_png) < 1000:
                reason = "rgb_not_ready"
                reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
                continue
            
            # 유효한 프레임! 저장
            final_rgb = os.path.join(class_output_dir, f"rgb_{accepted:04d}.png")
            final_pose = os.path.join(class_output_dir, f"pose_{accepted:04d}.json")
            
            shutil.copy(rgb_png, final_rgb)
            
            # Pose 정보 저장
            x_min, y_min, x_max, y_max, occlusion = bbox_info
            pose_data = {
                "class_name": class_name,
                "frame_index": accepted,
                "raw_pose_world": {
                    "t_xyz_m": [
                        (new_x - part_center[0]) * meters_per_unit,
                        (new_y - part_center[1]) * meters_per_unit,
                        (new_z - floor_height) * meters_per_unit
                    ],
                    "r_xyz_deg": [roll, pitch, yaw]
                },
                "bbox_2d": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "visibility": 1.0 - occlusion
                },
                "camera": {
                    "position_m": [c * meters_per_unit for c in camera_pos],
                    "look_at_m": [c * meters_per_unit for c in look_at],
                    "resolution": list(RESOLUTION)
                },
                "stage_info": {
                    "meters_per_unit": meters_per_unit,
                    "part_size_m": part_size_m
                }
            }
            
            with open(final_pose, 'w', encoding='utf-8') as f:
                json.dump(pose_data, f, indent=2, ensure_ascii=False)
            
            accepted += 1
            
            if accepted % 50 == 0 or accepted == IMAGES_PER_CLASS:
                top_rejects = sorted(reject_reasons.items(), key=lambda x: -x[1])[:3]
                print(f"    진행: {accepted}/{IMAGES_PER_CLASS} (시도: {total_attempts}, 거부 상위: {top_rejects})")
            
        except Exception as e:
            reason = f"exception_{type(e).__name__}"
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            continue
    
    # 정리
    print(f"\n  ✓ {display_name} 완료: {accepted}장 생성")
    print(f"    총 시도: {total_attempts}, 채택률: {accepted/max(1,total_attempts)*100:.1f}%")
    print(f"    거부 사유: {reject_reasons}")
    
    # 임시 폴더 삭제
    shutil.rmtree(tmp_dir, ignore_errors=True)
    
    # 메타데이터 저장
    metadata = {
        "class_name": class_name,
        "num_images": accepted,
        "part_size_m": part_size_m,
        "settings": {
            "xy_range_m": XY_RANGE_M,
            "z_lift_factor": Z_LIFT_FACTOR,
            "roll_pitch_range": ROLL_RANGE_DEG,
            "yaw_range": YAW_RANGE_DEG,
            "camera_elevation_deg": CAMERA_ELEVATION_DEG
        }
    }
    with open(os.path.join(class_output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# ==========================================
# 메인 실행
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("굴착기 부품 Pose Estimation 데이터셋 생성")
    print("="*60)
    
    # USD 파일 스캔
    parts_config = scan_usd_files(ASSETS_DIR)
    print(f"발견된 USD: {len(parts_config)}개")
    for name in parts_config:
        print(f"  - {name}")
    
    # 출력 디렉토리 정리
    if CLEAR_EXISTING and os.path.exists(OUTPUT_DIR):
        print(f"\n기존 데이터 삭제 중: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 각 클래스 처리
    for idx, (name, config) in enumerate(parts_config.items()):
        generate_class_dataset(config, idx, len(parts_config))
        
        # 다음 클래스 전 정리
        if idx < len(parts_config) - 1:
            print("\n스테이지 정리 중...")
            rep.orchestrator.stop()
            for _ in range(20):
                simulation_app.update()
                time.sleep(0.05)
            time.sleep(1.0)
    
    # 전체 메타데이터
    dataset_info = {
        "name": "Excavator Parts Pose Estimation Dataset",
        "num_classes": len(parts_config),
        "images_per_class": IMAGES_PER_CLASS,
        "classes": list(parts_config.keys()),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(OUTPUT_DIR, "dataset_info.json"), 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*60)
    print("데이터셋 생성 완료!")
    print("="*60)
    
    # 결과 출력
    for name in parts_config:
        class_dir = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(class_dir):
            count = len(glob.glob(os.path.join(class_dir, "rgb_*.png")))
            print(f"  {name}: {count}장")
    
    print("\n종료하려면 Ctrl+C를 누르세요.")
    while simulation_app.is_running():
        simulation_app.update()
    
    simulation_app.close()
