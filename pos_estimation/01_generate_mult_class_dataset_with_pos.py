"""
굴착기 부품 데이터셋 생성 스크립트 (분류 + 6DoF 포즈)

- 출력: /home/rebirther/isaac_data_output/pos_estimation/dataset_pos
- 방식(A안): 카메라 고정(상단 사선), 부품 이동(작업대 위, 평평하게)
- 저장:
  - rgb_{frame:04d}.png
  - bounding_box_2d_tight_{frame:04d}.npy
  - bounding_box_2d_tight_labels_{frame:04d}.json
  - pose_{frame:04d}.json  (카메라(optical) 기준 ^cam T_obj, 단위 m)

주의:
- Isaac Sim/Replicator 실행 환경에서 동작합니다.
- intrinsics(실카메라)가 없으므로, FOV/해상도를 고정하고 K를 계산하여 메타에 기록합니다.
"""

# 로깅 설정 (SimulationApp 초기화 전에 설정해야 함)
import os
import sys
import json
import time
import math
from pathlib import Path
import numpy as np
import shutil

SCRIPT_DIR = "/home/rebirther/isaac_data_output"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(PROJECT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from utils.logger import setup_logging, reinit_logging, finish_logging

LOG_PATH = setup_logging("01_generate_pos")

from isaacsim import SimulationApp

# 시뮬레이터 초기화
simulation_app = SimulationApp({"headless": False})

# App 실행 후 임포트
import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom, Semantics, Gf

reinit_logging(LOG_PATH)


# =========================
# 설정
# =========================
ASSETS_DIR = "/home/rebirther/isaac-sim/assets"
BASE_OUTPUT_DIR = os.path.join(PROJECT_DIR, "dataset_pos")
IMAGES_PER_CLASS = 500
CLEAR_EXISTING_DATA = True

# 카메라 설정 (가상 카메라: intrinsics 실측값 없음)
RESOLUTION = (1024, 1024)
FOV_DEG = 60.0  # 거리 계산/메타 기록용

# 배경 설정 (기존 스크립트와 동일한 랜덤 비율)
BACKGROUND_MODE = "random"
BACKGROUND_RATIOS = {
    "none": 0.2,
    "solid": 0.3,
    "factory": 0.5,
}

# A안: 상단 사선(작업대를 내려다보는) 카메라 고정
CAM_ELEVATION_DEG = 35.0  # 아래로 내려다보는 각도 느낌(명세용)
CAM_AZIMUTH_DEG = 0.0     # 좌우 회전(명세용, 필요 시 변경)

# 부품 포즈 분포(평평하게 놓임)
ROLL_DEG_RANGE = (-8.0, 8.0)
PITCH_DEG_RANGE = (-8.0, 8.0)
YAW_DEG_RANGE = (0.0, 360.0)

# XY 이동 범위(보수적으로: 부품 크기에 비례)
XY_RANGE_RATIO = 0.15  # part_size의 15% 범위에서만 이동

# 유효 프레임 조건(부품이 최소 80% 이상 보이게)
# - occlusionRatio <= 0.2 를 기준으로 "80% 이상 보임"으로 간주
# - bbox가 이미지 경계에 닿으면 잘림(truncation)으로 보고 실패 처리
MIN_VISIBLE_RATIO = 0.80
BBOX_MARGIN_PX = 5
MIN_BBOX_AREA_RATIO = 0.02  # 이미지 대비 bbox 면적이 너무 작으면 실패 처리(2%)

# 리젝션 샘플링 안전장치
MAX_ATTEMPTS_MULTIPLIER = 10  # IMAGES_PER_CLASS의 N배까지 시도
XY_RANGE_SHRINK_EVERY_REJECTS = 200
XY_RANGE_SHRINK_FACTOR = 0.7


# =========================
# 유틸
# =========================
def scan_usd_files(assets_dir: str):
    """assets 폴더의 모든 USD 파일을 스캔하여 설정 딕셔너리 생성"""
    import glob
    configs = {}
    usd_files = glob.glob(os.path.join(assets_dir, "*.usd"))
    usd_files.sort()
    for usd_path in usd_files:
        file_name = os.path.basename(usd_path)
        name_without_ext = os.path.splitext(file_name)[0]
        configs[name_without_ext] = {
            "usd_path": usd_path,
            "class_name": name_without_ext,
            "display_name": name_without_ext,
        }
    return configs


def _get_up_axis_and_index(stage):
    """스테이지 UpAxis를 읽고, 높이축 인덱스(0=X,1=Y,2=Z)를 반환"""
    up_axis = UsdGeom.GetStageUpAxis(stage)
    axis_index = 2
    if up_axis == UsdGeom.Tokens.y:
        axis_index = 1
    elif up_axis == UsdGeom.Tokens.x:
        axis_index = 0
    return up_axis, axis_index


def _mesh_local_min_max(mesh: UsdGeom.Mesh, time_code: Usd.TimeCode):
    extent = mesh.GetExtentAttr().Get(time_code)
    if extent and len(extent) == 2:
        mn = extent[0]
        mx = extent[1]
        return (Gf.Vec3d(mn[0], mn[1], mn[2]), Gf.Vec3d(mx[0], mx[1], mx[2]))
    pts = mesh.GetPointsAttr().Get(time_code)
    if not pts:
        return None, None
    mn = Gf.Vec3d(float("inf"), float("inf"), float("inf"))
    mx = Gf.Vec3d(float("-inf"), float("-inf"), float("-inf"))
    for p in pts:
        mn = Gf.Vec3d(min(mn[0], p[0]), min(mn[1], p[1]), min(mn[2], p[2]))
        mx = Gf.Vec3d(max(mx[0], p[0]), max(mx[1], p[1]), max(mx[2], p[2]))
    return mn, mx


def compute_world_aabb_from_meshes(stage, time_code: Usd.TimeCode):
    """
    Mesh vertex를 LocalToWorld로 변환하여 월드 AABB 계산(orient 반영).
    """
    xform_cache = UsdGeom.XformCache(time_code)
    mesh_prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    if not mesh_prims:
        return None

    world_min = Gf.Vec3d(float("inf"), float("inf"), float("inf"))
    world_max = Gf.Vec3d(float("-inf"), float("-inf"), float("-inf"))
    used_meshes = 0
    used_points = 0
    used_extent_fallback = 0
    total_points = 0

    for prim in mesh_prims:
        mesh = UsdGeom.Mesh(prim)
        M = xform_cache.GetLocalToWorldTransform(prim)
        used_meshes += 1

        pts = mesh.GetPointsAttr().Get(time_code)
        if pts:
            used_points += 1
            total_points += len(pts)
            for p in pts:
                wp = M.Transform(Gf.Vec3d(p[0], p[1], p[2]))
                world_min = Gf.Vec3d(min(world_min[0], wp[0]), min(world_min[1], wp[1]), min(world_min[2], wp[2]))
                world_max = Gf.Vec3d(max(world_max[0], wp[0]), max(world_max[1], wp[1]), max(world_max[2], wp[2]))
            continue

        used_extent_fallback += 1
        local_min, local_max = _mesh_local_min_max(mesh, time_code)
        if local_min is None or local_max is None:
            continue
        for x in (local_min[0], local_max[0]):
            for y in (local_min[1], local_max[1]):
                for z in (local_min[2], local_max[2]):
                    wp = M.Transform(Gf.Vec3d(x, y, z))
                    world_min = Gf.Vec3d(min(world_min[0], wp[0]), min(world_min[1], wp[1]), min(world_min[2], wp[2]))
                    world_max = Gf.Vec3d(max(world_max[0], wp[0]), max(world_max[1], wp[1]), max(world_max[2], wp[2]))

    size = world_max - world_min
    center = (world_min + world_max) / 2.0
    return {
        "world_min": world_min,
        "world_max": world_max,
        "size": size,
        "center": center,
        "mesh_count": len(mesh_prims),
        "used_meshes": used_meshes,
        "used_points": used_points,
        "used_extent_fallback": used_extent_fallback,
        "total_points": total_points,
    }


def _find_root_xform_prim(stage: Usd.Stage):
    """
    부품을 '하나의 강체'처럼 이동시키기 위한 루트 Xform prim을 찾습니다.
    우선순위:
    1) DefaultPrim이 Xformable이면 그것 사용
    2) /World/* 형태의 첫 번째 Xformable prim 사용
    """
    default_prim = stage.GetDefaultPrim()
    if default_prim and default_prim.IsA(UsdGeom.Xformable):
        return default_prim

    root_candidate = None
    for prim in stage.Traverse():
        if prim.GetPath().pathString.count("/") == 2 and prim.IsA(UsdGeom.Xformable) and prim.GetPath() != "/World":
            root_candidate = prim
            break
    return root_candidate


def _set_xform_translate_rotate_xyz(prim: Usd.Prim, t_xyz, r_xyz_deg):
    """
    prim에 translate + rotateXYZ를 설정합니다(간단/명시적).
    - translate: meters
    - rotateXYZ: degrees
    """
    xform = UsdGeom.Xformable(prim)
    ops = xform.GetOrderedXformOps()
    # 기존 op가 있으면 재사용, 없으면 생성
    t_op = None
    r_op = None
    for op in ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            t_op = op
        if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            r_op = op
    if t_op is None:
        t_op = xform.AddTranslateOp()
    if r_op is None:
        r_op = xform.AddRotateXYZOp()
    t_op.Set(Gf.Vec3d(float(t_xyz[0]), float(t_xyz[1]), float(t_xyz[2])))
    r_op.Set(Gf.Vec3f(float(r_xyz_deg[0]), float(r_xyz_deg[1]), float(r_xyz_deg[2])))


def _compute_K_from_fov_deg(width: int, height: int, fov_deg: float):
    """정사각형 해상도/수평FOV 가정으로 K 근사 계산"""
    fov_rad = math.radians(float(fov_deg))
    fx = (width / 2.0) / math.tan(fov_rad / 2.0)
    fy = fx
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]


def _rotmat_to_quat_xyzw(R: np.ndarray):
    """3x3 회전행렬 -> quaternion(x,y,z,w)"""
    # 안정적인 변환(표준 알고리즘)
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    # 정규화
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return [qx / n, qy / n, qz / n, qw / n]


def _camera_basis_from_lookat(cam_pos, target_pos, world_up_vec):
    """
    cam_optical 기준 축을 world에서 계산:
    - z: forward (카메라 -> 타겟)
    - x: right
    - y: down  (world_up에 기반해 up을 만들고 부호 반전)
    반환:
    - R_world_cam (3x3): cam축들이 world에서 어떻게 놓이는지(열벡터 = [x y z])
    """
    cam_pos = np.asarray(cam_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    up_guess = np.asarray(world_up_vec, dtype=float)

    forward = target_pos - cam_pos
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    right = np.cross(forward, up_guess)
    right = right / (np.linalg.norm(right) + 1e-12)
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-12)
    down = -up

    R_world_cam = np.column_stack([right, down, forward])
    return R_world_cam

def _select_object_bbox_from_files(bbox_npy_path: str, label_json_path: str):
    """
    Writer가 저장한 bbox npy/json을 기반으로 "background가 아닌" bbox 중 가장 큰 bbox를 선택.
    반환: (bbox_xyxy, occlusion_ratio)
    """
    def _get_field(bb, key: str, default=None):
        """bbox row(dict 또는 numpy structured row)에서 필드 값을 안전하게 꺼냅니다."""
        try:
            # numpy.void / structured array row
            return bb[key]
        except Exception:
            try:
                # dict-like
                return bb.get(key, default)
            except Exception:
                return default

    try:
        bboxes = np.load(bbox_npy_path, allow_pickle=True)
        with open(label_json_path, "r", encoding="utf-8") as f:
            labels_map = json.load(f)
    except Exception:
        return None, None

    best = None
    best_area = -1
    best_occ = None
    for bb in bboxes:
        sid_raw = _get_field(bb, "semanticId", None)
        if sid_raw is None:
            continue
        sid = str(int(sid_raw))
        cls = labels_map.get(sid, {}).get("class", "unknown")
        if cls == "background":
            continue
        x_min = int(_get_field(bb, "x_min", 0)); y_min = int(_get_field(bb, "y_min", 0))
        x_max = int(_get_field(bb, "x_max", 0)); y_max = int(_get_field(bb, "y_max", 0))
        area = max(0, x_max - x_min) * max(0, y_max - y_min)
        if area > best_area:
            best_area = area
            best = (x_min, y_min, x_max, y_max)
            occ_raw = _get_field(bb, "occlusionRatio", 0.0)
            best_occ = float(occ_raw) if occ_raw is not None else 0.0
    return best, best_occ

def _is_frame_valid(bbox_xyxy, occlusion_ratio: float, width: int, height: int):
    """
    유효 프레임 판단:
    - visible ratio >= MIN_VISIBLE_RATIO (occlusionRatio 기반 근사)
    - bbox가 이미지 경계에 닿지 않음(잘림 방지)
    - bbox 면적이 너무 작지 않음
    """
    if bbox_xyxy is None:
        return False, "no_object_bbox"

    x_min, y_min, x_max, y_max = bbox_xyxy
    if x_max <= x_min or y_max <= y_min:
        return False, "invalid_bbox"

    visible_ratio = 1.0 - float(occlusion_ratio if occlusion_ratio is not None else 1.0)
    if visible_ratio < MIN_VISIBLE_RATIO:
        return False, f"visible_ratio<{MIN_VISIBLE_RATIO:.2f}"

    # 경계 닿음(truncation) 체크
    if (
        x_min < BBOX_MARGIN_PX
        or y_min < BBOX_MARGIN_PX
        or x_max > (width - 1 - BBOX_MARGIN_PX)
        or y_max > (height - 1 - BBOX_MARGIN_PX)
    ):
        return False, "truncated_bbox"

    bbox_area = float((x_max - x_min) * (y_max - y_min))
    img_area = float(width * height)
    if bbox_area / img_area < MIN_BBOX_AREA_RATIO:
        return False, "bbox_too_small"

    return True, "ok"

def _wait_for_files(paths, timeout_s: float, poll_s: float = 0.05):
    """Replicator writer 출력이 비동기라서, 필요한 파일들이 생성될 때까지 기다립니다."""
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        if all(os.path.exists(p) for p in paths):
            return True
        time.sleep(poll_s)
    return False

def _move_or_delete_attempt_files(tmp_dir: str, attempt_idx: int, out_dir: str, out_idx: int, accept: bool):
    """
    attempt_idx로 생성된 writer 파일을 accept 여부에 따라 처리.
    - accept=True: tmp의 파일들을 out_dir로 out_idx 번호로 rename/move
    - accept=False: tmp의 attempt_idx 파일들을 삭제
    """
    # writer가 생성하는 파일 패턴(현재 사용)
    patterns = [
        ("rgb_{:04d}.png", "rgb_{:04d}.png"),
        ("bounding_box_2d_tight_{:04d}.npy", "bounding_box_2d_tight_{:04d}.npy"),
        ("bounding_box_2d_tight_labels_{:04d}.json", "bounding_box_2d_tight_labels_{:04d}.json"),
        # BasicWriter가 함께 생성하는 prim_paths 파일도 같이 정리(남아있으면 폴더가 더러워짐)
        ("bounding_box_2d_tight_prim_paths_{:04d}.json", "bounding_box_2d_tight_prim_paths_{:04d}.json"),
    ]
    for src_fmt, dst_fmt in patterns:
        src = os.path.join(tmp_dir, src_fmt.format(attempt_idx))
        if not os.path.exists(src):
            continue
        if accept:
            dst = os.path.join(out_dir, dst_fmt.format(out_idx))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
        else:
            try:
                os.remove(src)
            except Exception:
                pass


# =========================
# 메인
# =========================
EXCAVATOR_PARTS_CONFIG = scan_usd_files(ASSETS_DIR)

print("=" * 60)
print("굴착기 부품 데이터셋 생성 (분류 + 6DoF 포즈)")
print("=" * 60)
print(f"출력 경로: {BASE_OUTPUT_DIR}")
print(f"클래스 수: {len(EXCAVATOR_PARTS_CONFIG)}")
print(f"클래스당 이미지: {IMAGES_PER_CLASS}")
print(f"해상도: {RESOLUTION[0]}x{RESOLUTION[1]}, FOV: {FOV_DEG}deg")
print(f"방식: 카메라 고정(A안, 상단 사선) / 부품 이동(평평)")
print("=" * 60)

if CLEAR_EXISTING_DATA and os.path.exists(BASE_OUTPUT_DIR):
    import shutil
    print(f"\n⚠️  기존 dataset_pos 폴더 삭제 중: {BASE_OUTPUT_DIR}")
    shutil.rmtree(BASE_OUTPUT_DIR, ignore_errors=True)
    print("✓ 삭제 완료")

os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# 전역 메타데이터(dataset_info.json)
K = _compute_K_from_fov_deg(RESOLUTION[0], RESOLUTION[1], FOV_DEG)
dataset_metadata = {
    "dataset_name": "Excavator Parts Classification + 6DoF Pose Dataset",
    "output_dir": BASE_OUTPUT_DIR,
    "num_classes": len(EXCAVATOR_PARTS_CONFIG),
    "images_per_class": IMAGES_PER_CLASS,
    "total_images": IMAGES_PER_CLASS * len(EXCAVATOR_PARTS_CONFIG),
    "classes": {key: cfg["display_name"] for key, cfg in EXCAVATOR_PARTS_CONFIG.items()},
    "background_mode": BACKGROUND_MODE,
    "background_ratios": BACKGROUND_RATIOS if BACKGROUND_MODE == "random" else None,
    "camera": {
        "resolution": [RESOLUTION[0], RESOLUTION[1]],
        "fov_deg_assumed": FOV_DEG,
        "K_assumed": K,
        "convention": "cam_optical: x-right, y-down, z-forward",
        "note": "실카메라 intrinsics가 없으므로 FOV 기반으로 K를 근사 계산"
    },
    "pose_label": {
        "frame": "cam_optical",
        "object_frame": "CAD origin (USD object local frame)",
        "unit_translation": "m",
        "rotation": "quaternion_xyzw",
        "note": "카메라 고정/부품 이동(A안). roll/pitch 제한, yaw 자유."
    },
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}

with open(os.path.join(BASE_OUTPUT_DIR, "dataset_info.json"), "w", encoding="utf-8") as f:
    json.dump(dataset_metadata, f, indent=2, ensure_ascii=False)


def generate_class_dataset(part_config, class_index: int):
    usd_path = part_config["usd_path"]
    class_name = part_config["class_name"]
    display_name = part_config["display_name"]

    class_output_dir = os.path.join(BASE_OUTPUT_DIR, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    print(f"\n[{class_index+1}/{len(EXCAVATOR_PARTS_CONFIG)}] {display_name} 생성 시작")
    print(f"  USD: {usd_path}")
    print(f"  OUT: {class_output_dir}")

    if not os.path.exists(usd_path):
        print(f"  ⚠️  USD 파일 없음: {usd_path} (스킵)")
        return

    omni.usd.get_context().open_stage(usd_path)
    time.sleep(1.0)

    stage = omni.usd.get_context().get_stage()
    if not stage:
        print("  ⚠️  스테이지 로드 실패")
        return

    time_code = Usd.TimeCode.Default()
    up_axis, axis_index = _get_up_axis_and_index(stage)
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    world_up = np.array([0.0, 0.0, 1.0])
    if axis_index == 1:
        world_up = np.array([0.0, 1.0, 0.0])
    elif axis_index == 0:
        world_up = np.array([1.0, 0.0, 0.0])

    aabb = compute_world_aabb_from_meshes(stage, time_code)
    if not aabb:
        print("  ⚠️  AABB 계산 실패(메시 없음?)")
        return

    world_min = aabb["world_min"]
    world_max = aabb["world_max"]
    size = aabb["size"]
    center = aabb["center"]
    part_center = (float(center[0]), float(center[1]), float(center[2]))
    part_size = float(max(size[0], size[1], size[2]))
    floor_height = float(world_min[axis_index])

    # 루트 프림 찾기(부품 이동용)
    root_prim = _find_root_xform_prim(stage)
    if not root_prim:
        print("  ⚠️  루트 Xform prim을 찾지 못했습니다(스킵)")
        return

    root_path = root_prim.GetPath().pathString
    print(f"  Root prim: {root_path}")
    print(f"  UpAxis: {up_axis}, metersPerUnit: {meters_per_unit}")

    # Semantics 추가
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Imageable):
            sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
            sem.CreateSemanticTypeAttr("class")
            sem.CreateSemanticDataAttr().Set(class_name)

    # 클래스 메타 저장
    metadata = {
        "class_name": class_name,
        "display_name": display_name,
        "usd_path": usd_path,
        "num_images": IMAGES_PER_CLASS,
        "root_prim_path": root_path,
        "part_center_world": list(part_center),
        "part_size_world": part_size,
        "floor_height_world": floor_height,
        "up_axis": str(up_axis),
        "axis_index": int(axis_index),
        "meters_per_unit": meters_per_unit,
        "background_mode": BACKGROUND_MODE,
        "background_ratios": BACKGROUND_RATIOS if BACKGROUND_MODE == "random" else None,
    }
    with open(os.path.join(class_output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Replicator 레이어
    with rep.new_layer():
        # 바닥/뒷벽 (기존 방식 유지)
        floor_size = part_size * 5.0
        if axis_index == 2:
            floor_pos = (part_center[0], part_center[1], floor_height)
            floor_rot = (0, 0, 0)
        elif axis_index == 1:
            floor_pos = (part_center[0], floor_height, part_center[2])
            floor_rot = (90, 0, 0)
        else:
            floor_pos = (floor_height, part_center[1], part_center[2])
            floor_rot = (0, 90, 0)

        floor_plane = rep.create.plane(
            scale=(floor_size, floor_size, 1),
            position=floor_pos,
            rotation=floor_rot,
            semantics=[("class", "background")],
        )
        back_wall = rep.create.plane(
            scale=(floor_size, floor_size * 0.5, 1),
            position=(part_center[0] - floor_size * 0.4, part_center[1], part_center[2]),
            rotation=(0, 90, 0),
            semantics=[("class", "background")],
        )

        # 조명(기존 유지 + 프레임별 위치 랜덤화는 trigger에서)
        dome_light = rep.create.light(light_type="Dome", intensity=800.0, rotation=(270, 0, 0))
        point_light1 = rep.create.light(
            light_type="Sphere",
            intensity=50000.0,
            position=(part_center[0] + part_size, part_center[1] + part_size, part_center[2] + part_size * 2),
            scale=0.5,
        )
        point_light2 = rep.create.light(
            light_type="Sphere",
            intensity=30000.0,
            position=(part_center[0] - part_size, part_center[1] - part_size * 0.5, part_center[2] + part_size),
            scale=0.3,
        )

        # 카메라 고정(A안 느낌): "부품 전체가 보이도록" 거리 자동 산정 + 사선 위치 고정
        part_diagonal = float(math.sqrt(float(size[0]) ** 2 + float(size[1]) ** 2 + float(size[2]) ** 2))
        camera_fov_rad = math.radians(FOV_DEG)
        min_camera_distance = (part_diagonal / 2.0) / math.tan(camera_fov_rad / 2.0) / 0.8
        camera_distance = min_camera_distance * 1.15

        # 카메라 위치(사선): up축 기준으로 위쪽(+up) + 약간 옆(+right)
        # 간단히 기존 벡터(0.7,0.5,0.5)를 유지하되, 카메라를 고정한다.
        cam_pos = (
            part_center[0] + camera_distance * 0.7,
            part_center[1] + camera_distance * 0.5,
            part_center[2] + camera_distance * 0.5,
        )
        cam_lookat = part_center

        camera = rep.create.camera(position=cam_pos, look_at=cam_lookat)
        render_product = rep.create.render_product(camera, resolution=RESOLUTION)

        # bbox annotator + writer
        bbox_annotator = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
        bbox_annotator.attach([render_product])

        # 리젝션 샘플링을 위해 writer는 임시 디렉토리에 기록하고,
        # "유효 프레임만" 최종 디렉토리(class_output_dir)로 이동/재번호 부여한다.
        tmp_output_dir = os.path.join(class_output_dir, "_tmp_writer")
        if os.path.exists(tmp_output_dir):
            shutil.rmtree(tmp_output_dir, ignore_errors=True)
        os.makedirs(tmp_output_dir, exist_ok=True)

        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(output_dir=tmp_output_dir, rgb=True, bounding_box_2d_tight=True)
        writer.attach([render_product])

        # 프레임별 랜덤화(배경/조명만): 카메라는 고정, 부품은 외부 루프로 이동
        # 리젝션으로 시도 횟수가 늘어날 수 있으므로 max_execs를 크게 잡는다.
        max_attempts = int(IMAGES_PER_CLASS * MAX_ATTEMPTS_MULTIPLIER)
        with rep.trigger.on_frame(max_execs=max_attempts):
            with floor_plane:
                rep.randomizer.color(
                    colors=rep.distribution.uniform((0.2, 0.2, 0.2), (0.6, 0.5, 0.4))
                )
            with back_wall:
                rep.randomizer.color(
                    colors=rep.distribution.uniform((0.4, 0.4, 0.4), (0.9, 0.9, 0.85))
                )
            with point_light1:
                rep.modify.pose(
                    position=rep.distribution.uniform(
                        (part_center[0] + part_size * 0.5, part_center[1] - part_size, part_center[2] + part_size * 1.5),
                        (part_center[0] + part_size * 1.5, part_center[1] + part_size, part_center[2] + part_size * 3),
                    )
                )
            with point_light2:
                rep.modify.pose(
                    position=rep.distribution.uniform(
                        (part_center[0] - part_size * 1.5, part_center[1] - part_size, part_center[2] + part_size * 0.5),
                        (part_center[0] - part_size * 0.5, part_center[1] + part_size, part_center[2] + part_size * 2),
                    )
                )

        # 시뮬레이션 준비
        for _ in range(10):
            simulation_app.update()
            time.sleep(0.05)

        # cam_optical 기준(우리 정의) 카메라 외부파라미터(고정)
        R_world_cam = _camera_basis_from_lookat(cam_pos, cam_lookat, world_up)
        R_cam_world = R_world_cam.T
        t_world_cam = np.asarray(cam_pos, dtype=float)

        # 프레임 루프(리젝션 샘플링):
        # - attempt는 writer 파일 인덱스로 증가
        # - accept된 프레임만 최종 인덱스(0..IMAGES_PER_CLASS-1)로 이동
        # (카메라는 고정, 조명/배경은 trigger에서 변함)
        xy_range = part_size * XY_RANGE_RATIO
        reject_count = 0

        if not (hasattr(rep.orchestrator, "step") or hasattr(rep.orchestrator, "run")):
            raise RuntimeError("rep.orchestrator.step/run API를 찾을 수 없습니다. Isaac Sim 버전을 확인하세요.")

        accepted = 0
        attempt_idx = 0
        try:
            while accepted < IMAGES_PER_CLASS and attempt_idx < max_attempts:
                # 1) 부품 pose 샘플링(world 기준)
                dx = float(np.random.uniform(-xy_range, xy_range))
                dy = float(np.random.uniform(-xy_range, xy_range))
                # 높이축 외에는 거의 0, 높이축은 바닥에 맞춰둔 후 작은 노이즈만(옵션)
                dz = 0.0

                roll = float(np.random.uniform(ROLL_DEG_RANGE[0], ROLL_DEG_RANGE[1]))
                pitch = float(np.random.uniform(PITCH_DEG_RANGE[0], PITCH_DEG_RANGE[1]))
                yaw = float(np.random.uniform(YAW_DEG_RANGE[0], YAW_DEG_RANGE[1]))

                # base 위치: 바닥 위에 놓이도록(초기 바닥 기준)
                # 여기서는 "기준 중심(part_center)"를 테이블 기준점으로 쓰고, XY만 이동
                t_world_obj = np.array([part_center[0] + dx, part_center[1] + dy, part_center[2]], dtype=float)
                # 높이축 정렬: z-up이면 z를 조정. y-up/x-up도 동일 로직으로 처리
                t_world_obj[axis_index] = part_center[axis_index] + dz

                # 2) USD에 적용(rotateXYZ)
                _set_xform_translate_rotate_xyz(root_prim, t_world_obj, (roll, pitch, yaw))

                # 3) 렌더/어노테이션 1프레임 진행
                if hasattr(rep.orchestrator, "step"):
                    rep.orchestrator.step()
                else:
                    rep.orchestrator.run(num_frames=1)

                # writer 출력은 비동기일 수 있으므로, 필요한 파일들이 실제로 생길 때까지 기다린다.
                fid = f"{attempt_idx:04d}"
                rgb_png = os.path.join(tmp_output_dir, f"rgb_{fid}.png")
                bbox_npy = os.path.join(tmp_output_dir, f"bounding_box_2d_tight_{fid}.npy")
                bbox_lbl = os.path.join(tmp_output_dir, f"bounding_box_2d_tight_labels_{fid}.json")

                ready = _wait_for_files([rgb_png, bbox_npy, bbox_lbl], timeout_s=3.0, poll_s=0.05)
                if not ready:
                    # 파일이 늦게 생성되는 경우가 있어, 일단 거부 처리하되 가능한 범위에서 정리한다.
                    _move_or_delete_attempt_files(tmp_output_dir, attempt_idx, class_output_dir, accepted, accept=False)
                    reject_count += 1
                    attempt_idx += 1
                    continue

                bbox_xyxy, occ = _select_object_bbox_from_files(bbox_npy, bbox_lbl)
                ok, reason = _is_frame_valid(bbox_xyxy, occ, RESOLUTION[0], RESOLUTION[1])

                if not ok:
                    # 무효 프레임: tmp 파일 삭제
                    _move_or_delete_attempt_files(tmp_output_dir, attempt_idx, class_output_dir, accepted, accept=False)
                    reject_count += 1
                    # 거부가 너무 많으면 XY 이동 범위를 줄여 "화면 밖으로 나가는" 빈도를 줄인다.
                    if reject_count % XY_RANGE_SHRINK_EVERY_REJECTS == 0:
                        xy_range *= XY_RANGE_SHRINK_FACTOR
                        print(f"  ⚠️  reject {reject_count}회 발생 → XY 이동 범위 축소: {xy_range:.4f} (reason={reason})")
                    attempt_idx += 1
                    continue

                # 4) 포즈 라벨 계산(카메라(optical) 기준) - accept된 프레임만 저장
                # object 회전행렬(world 기준): rotateXYZ(roll,pitch,yaw) 순서 가정
                rx, ry, rz = math.radians(roll), math.radians(pitch), math.radians(yaw)
                Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]], dtype=float)
                Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]], dtype=float)
                Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]], dtype=float)
                R_world_obj = (Rz @ Ry @ Rx)

                t_cam_obj = R_cam_world @ (t_world_obj - t_world_cam)
                R_cam_obj = R_cam_world @ R_world_obj
                q_cam_obj = _rotmat_to_quat_xyzw(R_cam_obj)

                pose = {
                    "class_name": class_name,
                    "frame_idx": accepted,
                    "unit": "m",
                    "camera": {
                        "width": RESOLUTION[0],
                        "height": RESOLUTION[1],
                        "K_assumed": K,
                        "convention": "cam_optical: x-right, y-down, z-forward",
                    },
                    "pose_cam_optical_obj": {
                        "t_xyz_m": [float(t_cam_obj[0]), float(t_cam_obj[1]), float(t_cam_obj[2])],
                        "q_xyzw": [float(q_cam_obj[0]), float(q_cam_obj[1]), float(q_cam_obj[2]), float(q_cam_obj[3])],
                    },
                    "raw_pose_world": {
                        "t_xyz_m": [float(t_world_obj[0]), float(t_world_obj[1]), float(t_world_obj[2])],
                        "r_xyz_deg": [roll, pitch, yaw],
                        "camera_pos_world_m": [float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])],
                        "camera_lookat_world_m": [float(cam_lookat[0]), float(cam_lookat[1]), float(cam_lookat[2])],
                    },
                    "stage": {
                        "up_axis": str(up_axis),
                        "meters_per_unit": meters_per_unit,
                    },
                }

                # accept된 writer 파일을 최종 디렉토리로 이동/재번호 부여
                _move_or_delete_attempt_files(tmp_output_dir, attempt_idx, class_output_dir, accepted, accept=True)

                pose_path = os.path.join(class_output_dir, f"pose_{accepted:04d}.json")
                with open(pose_path, "w", encoding="utf-8") as f:
                    json.dump(pose, f, indent=2, ensure_ascii=False)

                accepted += 1
                attempt_idx += 1

                if accepted % 50 == 0:
                    print(f"  진행: {accepted}/{IMAGES_PER_CLASS} (attempt={attempt_idx}, reject={reject_count}, xy_range={xy_range:.4f})")
        finally:
            # tmp 디렉토리 정리(중간 중단/예외가 발생해도 남지 않게)
            shutil.rmtree(tmp_output_dir, ignore_errors=True)

        if accepted < IMAGES_PER_CLASS:
            print(f"  ⚠️  유효 프레임 부족: {accepted}/{IMAGES_PER_CLASS} (attempt={attempt_idx}, reject={reject_count})")
        else:
            print(f"  ✓ {display_name}: {IMAGES_PER_CLASS} 프레임 생성 완료 (reject={reject_count}, attempt={attempt_idx})")


print("\n전체 dataset_pos 생성 시작...")
for idx, (_, cfg) in enumerate(EXCAVATOR_PARTS_CONFIG.items()):
    generate_class_dataset(cfg, idx)

print("\n" + "=" * 60)
print("dataset_pos 생성 완료!")
print("=" * 60)

finish_logging()

print("\n시뮬레이션을 계속 실행합니다. 종료하려면 Ctrl+C")
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
