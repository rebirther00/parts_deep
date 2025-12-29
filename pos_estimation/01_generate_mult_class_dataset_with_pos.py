# ==========================================
# 굴착기 부품 데이터셋 생성 스크립트 (Pose Estimation용)
# Replicator rep.modify.pose() 사용 버전
# ==========================================
#
# 핵심 원칙:
# 1. 카메라 고정 (oblique view)
# 2. 오브젝트에 rep.modify.pose() 적용하여 이동/회전
# 3. Domain Randomization (조명, 배경 색상)
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
from pxr import Usd, UsdGeom, Semantics, Gf

# ==========================================
# 설정
# ==========================================
ASSETS_DIR = "/home/rebirther/isaac-sim/assets"
OUTPUT_DIR = os.path.join(PROJECT_DIR, "dataset_pos")
IMAGES_PER_CLASS = 500
RESOLUTION = (1024, 1024)
CLEAR_EXISTING = True

# 카메라 설정 (oblique view - 35도 각도로 내려다봄)
CAMERA_ELEVATION_DEG = 35.0

# 오브젝트 이동/회전 범위
XY_OFFSET_RATIO = 0.15  # 부품 크기 대비 XY 오프셋 비율
Z_LIFT_RATIO = 0.6  # 부품 크기 대비 Z 들어올림 비율
Z_RANGE_RATIO = 0.2  # 추가 Z 변동 비율

ROLL_RANGE = (-45.0, 45.0)
PITCH_RANGE = (-45.0, 45.0)
YAW_RANGE = (0.0, 360.0)

# 가시성 검증
MIN_VISIBILITY = 0.80
MIN_BBOX_AREA_RATIO = 0.02
EDGE_MARGIN = 10
MAX_REJECTS_PER_ACCEPT = 20


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
    """스테이지의 모든 Mesh에 대해 월드 좌표계 AABB 계산"""
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
        "floor": world_min[2]
    }


def validate_bbox_data(bbox_data, labels_data, resolution, class_name):
    """BBox 데이터 검증"""
    if bbox_data is None or len(bbox_data) == 0:
        return False, "no_bbox", None
    
    id_to_labels = labels_data.get("idToLabels", {})
    
    best_bbox = None
    best_area = 0
    
    for box in bbox_data:
        try:
            if hasattr(box, 'dtype') and box.dtype.names:
                semantic_id = str(int(box['semanticId']))
                x_min, y_min = float(box['x_min']), float(box['y_min'])
                x_max, y_max = float(box['x_max']), float(box['y_max'])
                occlusion = float(box['occlusionRatio']) if 'occlusionRatio' in box.dtype.names else 0.0
            else:
                continue
        except:
            continue
        
        # 클래스 확인
        if semantic_id not in id_to_labels:
            continue
        label_info = id_to_labels[semantic_id]
        if class_name not in str(label_info):
            continue
        
        area = (x_max - x_min) * (y_max - y_min)
        if area > best_area:
            best_area = area
            best_bbox = (x_min, y_min, x_max, y_max, occlusion)
    
    if best_bbox is None:
        return False, "no_class_bbox", None
    
    x_min, y_min, x_max, y_max, occlusion = best_bbox
    visibility = 1.0 - occlusion
    
    if visibility < MIN_VISIBILITY:
        return False, f"low_vis_{visibility:.2f}", best_bbox
    
    if x_min < EDGE_MARGIN or y_min < EDGE_MARGIN:
        return False, "edge_min", best_bbox
    if x_max > resolution[0] - EDGE_MARGIN or y_max > resolution[1] - EDGE_MARGIN:
        return False, "edge_max", best_bbox
    
    bbox_area = (x_max - x_min) * (y_max - y_min)
    if bbox_area < resolution[0] * resolution[1] * MIN_BBOX_AREA_RATIO:
        return False, "small_bbox", best_bbox
    
    return True, "ok", best_bbox


def generate_class_dataset(part_config, class_index, total_classes):
    """한 클래스의 데이터셋 생성 (Replicator 방식)"""
    usd_path = part_config["usd_path"]
    class_name = part_config["class_name"]
    display_name = part_config["display_name"]
    
    class_output_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"[{class_index+1}/{total_classes}] {display_name}")
    print(f"{'='*60}")
    print(f"  USD: {usd_path}")
    
    if not os.path.exists(usd_path):
        print(f"  ⚠️  USD 파일 없음")
        return
    
    # USD 로드
    print(f"  USD 로딩 중...")
    omni.usd.get_context().open_stage(usd_path)
    time.sleep(1.0)
    
    stage = omni.usd.get_context().get_stage()
    if not stage:
        print(f"  ⚠️  스테이지 로드 실패")
        return
    
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"  metersPerUnit: {meters_per_unit}")
    
    # AABB 계산
    aabb = compute_world_aabb(stage)
    if not aabb:
        print(f"  ⚠️  AABB 계산 실패")
        return
    
    part_size = max(aabb["size"][0], aabb["size"][1], aabb["size"][2])
    part_center = (aabb["center"][0], aabb["center"][1], aabb["center"][2])
    floor_height = aabb["floor"]
    
    print(f"  부품 크기: {part_size:.4f}")
    print(f"  부품 중심: {part_center}")
    print(f"  바닥 높이: {floor_height:.4f}")
    
    # 오브젝트 이동 범위 계산
    xy_offset = part_size * XY_OFFSET_RATIO
    z_base = floor_height + part_size * Z_LIFT_RATIO  # Z 들어올림
    z_range = part_size * Z_RANGE_RATIO
    
    print(f"  XY 오프셋: ±{xy_offset:.4f}")
    print(f"  Z 기본 높이: {z_base:.4f}")
    print(f"  Z 범위: ±{z_range:.4f}")
    print(f"  Roll/Pitch: {ROLL_RANGE}, Yaw: {YAW_RANGE}")
    
    # 카메라 위치 (oblique view)
    camera_distance = part_size * 2.5
    elev_rad = math.radians(CAMERA_ELEVATION_DEG)
    cam_x = part_center[0] + camera_distance * math.cos(elev_rad)
    cam_y = part_center[1]
    cam_z = z_base + camera_distance * math.sin(elev_rad)
    look_at = (part_center[0], part_center[1], z_base)
    
    print(f"  카메라: ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}) -> {look_at}")
    
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
    
    try:
        with rep.new_layer():
            # 기존 오브젝트 참조
            part_prim = rep.get.prims(semantics=[("class", class_name)])
            print(f"  ✓ 오브젝트 참조 완료")
            
            # ------------------------------------------
            # Domain Randomization: 배경
            # ------------------------------------------
            floor_size = part_size * 5
            
            # 바닥 평면
            floor_plane = rep.create.plane(
                scale=(floor_size, floor_size, 1),
                position=(part_center[0], part_center[1], floor_height),
                rotation=(0, 0, 0),
                semantics=[("class", "background")]
            )
            print(f"  ✓ 바닥 생성")
            
            # 뒷벽 평면
            back_wall = rep.create.plane(
                scale=(floor_size, floor_size * 0.5, 1),
                position=(part_center[0] - floor_size * 0.4, part_center[1], part_center[2]),
                rotation=(0, 90, 0),
                semantics=[("class", "background")]
            )
            print(f"  ✓ 뒷벽 생성")
            
            # ------------------------------------------
            # Domain Randomization: 조명
            # ------------------------------------------
            dome_light = rep.create.light(
                light_type="Dome",
                intensity=800.0,
                rotation=(270, 0, 0)
            )
            
            point_light1 = rep.create.light(
                light_type="Sphere",
                intensity=50000.0,
                position=(part_center[0] + part_size, part_center[1] + part_size, z_base + part_size * 2),
                scale=0.5
            )
            
            point_light2 = rep.create.light(
                light_type="Sphere",
                intensity=30000.0,
                position=(part_center[0] - part_size, part_center[1] - part_size * 0.5, z_base + part_size),
                scale=0.3
            )
            print(f"  ✓ 조명 생성 (Dome + Sphere x2)")
            
            # ------------------------------------------
            # 카메라 생성 (고정)
            # ------------------------------------------
            camera = rep.create.camera(
                position=(cam_x, cam_y, cam_z),
                look_at=look_at
            )
            render_product = rep.create.render_product(camera, resolution=RESOLUTION)
            print(f"  ✓ 카메라 생성 (고정)")
            
            # ------------------------------------------
            # 프레임별 랜덤화
            # ------------------------------------------
            with rep.trigger.on_frame(max_execs=IMAGES_PER_CLASS * (MAX_REJECTS_PER_ACCEPT + 1)):
                # 오브젝트 위치/회전 랜덤화
                with part_prim:
                    rep.modify.pose(
                        position=rep.distribution.uniform(
                            (part_center[0] - xy_offset, part_center[1] - xy_offset, z_base - z_range),
                            (part_center[0] + xy_offset, part_center[1] + xy_offset, z_base + z_range)
                        ),
                        rotation=rep.distribution.uniform(
                            (ROLL_RANGE[0], PITCH_RANGE[0], YAW_RANGE[0]),
                            (ROLL_RANGE[1], PITCH_RANGE[1], YAW_RANGE[1])
                        )
                    )
                
                # 바닥 색상 랜덤화
                with floor_plane:
                    rep.randomizer.color(
                        colors=rep.distribution.uniform((0.2, 0.2, 0.2), (0.6, 0.5, 0.4))
                    )
                
                # 뒷벽 색상 랜덤화
                with back_wall:
                    rep.randomizer.color(
                        colors=rep.distribution.uniform((0.4, 0.4, 0.4), (0.9, 0.9, 0.85))
                    )
                
                # 조명 위치 랜덤화
                with point_light1:
                    rep.modify.pose(
                        position=rep.distribution.uniform(
                            (part_center[0] + part_size * 0.5, part_center[1] - part_size, z_base + part_size * 1.5),
                            (part_center[0] + part_size * 1.5, part_center[1] + part_size, z_base + part_size * 3)
                        )
                    )
                
                with point_light2:
                    rep.modify.pose(
                        position=rep.distribution.uniform(
                            (part_center[0] - part_size * 1.5, part_center[1] - part_size, z_base + part_size * 0.5),
                            (part_center[0] - part_size * 0.5, part_center[1] + part_size, z_base + part_size * 2)
                        )
                    )
            
            print(f"  ✓ 프레임별 랜덤화 설정 완료")
            print(f"    - 오브젝트 위치/회전")
            print(f"    - 바닥/벽 색상")
            print(f"    - 조명 위치")
            
            # ------------------------------------------
            # Writer 및 Annotator
            # ------------------------------------------
            tmp_dir = os.path.join(class_output_dir, "_tmp")
            if os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir)
            os.makedirs(tmp_dir)
            
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=tmp_dir, rgb=True, bounding_box_2d_tight=True)
            writer.attach([render_product])
            print(f"  ✓ Writer 설정 완료")
            
            # 시뮬레이션 준비
            print(f"  시뮬레이션 준비 중...")
            for _ in range(10):
                simulation_app.update()
                time.sleep(0.05)
            
            # ------------------------------------------
            # Rejection Sampling 루프
            # ------------------------------------------
            print(f"\n  🔄 Rejection Sampling 시작 (목표: {IMAGES_PER_CLASS}장)")
            
            accepted = 0
            frame_idx = 0
            reject_counts = {}
            
            while accepted < IMAGES_PER_CLASS:
                # 최대 시도 횟수 체크
                if frame_idx >= IMAGES_PER_CLASS * (MAX_REJECTS_PER_ACCEPT + 1):
                    print(f"  ⚠️  최대 시도 초과, {accepted}장만 생성")
                    break
                
                # 1프레임 실행
                rep.orchestrator.step()
                simulation_app.update()
                
                # 파일 경로
                bbox_npy = os.path.join(tmp_dir, f"bounding_box_2d_tight_{frame_idx:04d}.npy")
                bbox_json = os.path.join(tmp_dir, f"bounding_box_2d_tight_labels_{frame_idx:04d}.json")
                rgb_png = os.path.join(tmp_dir, f"rgb_{frame_idx:04d}.png")
                
                frame_idx += 1
                
                # 파일 대기 (최대 1초)
                wait_end = time.time() + 1.0
                while time.time() < wait_end:
                    if os.path.exists(bbox_npy) and os.path.exists(bbox_json):
                        break
                    time.sleep(0.02)
                
                if not os.path.exists(bbox_npy) or not os.path.exists(bbox_json):
                    reject_counts["no_file"] = reject_counts.get("no_file", 0) + 1
                    continue
                
                # BBox 검증
                try:
                    bbox_data = np.load(bbox_npy)
                    with open(bbox_json, 'r') as f:
                        labels_data = json.load(f)
                    
                    valid, reason, bbox_info = validate_bbox_data(bbox_data, labels_data, RESOLUTION, class_name)
                    
                    if not valid:
                        reject_counts[reason] = reject_counts.get(reason, 0) + 1
                        continue
                    
                    # RGB 대기
                    wait_end = time.time() + 1.0
                    while time.time() < wait_end:
                        if os.path.exists(rgb_png) and os.path.getsize(rgb_png) > 1000:
                            break
                        time.sleep(0.02)
                    
                    if not os.path.exists(rgb_png) or os.path.getsize(rgb_png) < 1000:
                        reject_counts["no_rgb"] = reject_counts.get("no_rgb", 0) + 1
                        continue
                    
                    # 유효! 저장
                    x_min, y_min, x_max, y_max, occlusion = bbox_info
                    
                    final_rgb = os.path.join(class_output_dir, f"rgb_{accepted:04d}.png")
                    final_pose = os.path.join(class_output_dir, f"pose_{accepted:04d}.json")
                    
                    shutil.copy(rgb_png, final_rgb)
                    
                    # Pose 정보 (간소화)
                    pose_data = {
                        "class_name": class_name,
                        "frame_index": accepted,
                        "bbox_2d": {
                            "x_min": x_min, "y_min": y_min,
                            "x_max": x_max, "y_max": y_max,
                            "visibility": 1.0 - occlusion
                        },
                        "camera": {
                            "position": [cam_x, cam_y, cam_z],
                            "look_at": list(look_at),
                            "resolution": list(RESOLUTION)
                        },
                        "stage_info": {
                            "meters_per_unit": meters_per_unit,
                            "part_size": part_size
                        }
                    }
                    
                    with open(final_pose, 'w', encoding='utf-8') as f:
                        json.dump(pose_data, f, indent=2, ensure_ascii=False)
                    
                    accepted += 1
                    
                    # 진행 로그
                    if accepted % 50 == 0 or accepted == IMAGES_PER_CLASS:
                        top_rejects = sorted(reject_counts.items(), key=lambda x: -x[1])[:3]
                        rate = accepted / frame_idx * 100
                        print(f"    ✓ {accepted}/{IMAGES_PER_CLASS} (시도: {frame_idx}, 채택률: {rate:.1f}%, 거부: {top_rejects})")
                    
                except Exception as e:
                    reject_counts[f"err_{type(e).__name__}"] = reject_counts.get(f"err_{type(e).__name__}", 0) + 1
                    continue
            
            print(f"\n  ✓ {display_name} 완료: {accepted}장")
            print(f"    총 시도: {frame_idx}, 최종 채택률: {accepted/max(1,frame_idx)*100:.1f}%")
            print(f"    거부 통계: {reject_counts}")
            
    except Exception as e:
        print(f"  ⚠️  Replicator 에러: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 정리
    shutil.rmtree(tmp_dir, ignore_errors=True)
    
    # 메타데이터 저장
    metadata = {
        "class_name": class_name,
        "num_images": accepted,
        "part_size": part_size,
        "meters_per_unit": meters_per_unit
    }
    with open(os.path.join(class_output_dir, "metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# ==========================================
# 메인 실행
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("굴착기 부품 Pose Estimation 데이터셋 생성")
    print("(Replicator rep.modify.pose 방식)")
    print("="*60)
    
    parts_config = scan_usd_files(ASSETS_DIR)
    print(f"발견된 USD: {len(parts_config)}개")
    for name in parts_config:
        print(f"  - {name}")
    
    if CLEAR_EXISTING and os.path.exists(OUTPUT_DIR):
        print(f"\n기존 데이터 삭제 중: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for idx, (name, config) in enumerate(parts_config.items()):
        generate_class_dataset(config, idx, len(parts_config))
        
        if idx < len(parts_config) - 1:
            print("\n스테이지 정리 중...")
            rep.orchestrator.stop()
            for _ in range(20):
                simulation_app.update()
                time.sleep(0.05)
            time.sleep(1.0)
    
    # 전체 메타데이터
    dataset_info = {
        "name": "Excavator Parts Pose Dataset",
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
    
    for name in parts_config:
        class_dir = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(class_dir):
            count = len(glob.glob(os.path.join(class_dir, "rgb_*.png")))
            print(f"  {name}: {count}장")
    
    print("\n종료하려면 Ctrl+C를 누르세요.")
    while simulation_app.is_running():
        simulation_app.update()
    
    simulation_app.close()
