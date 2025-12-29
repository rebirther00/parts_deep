# ==========================================
# 굴착기 부품 데이터셋 생성 스크립트 (Pose Estimation용)
# class_estimation 코드 기반 - 안정적으로 작동하는 버전
# ==========================================
#
# 핵심: class_estimation과 동일한 방식으로 이미지 생성 후,
#       bbox 파일을 파싱하여 pose 라벨(pose_####.json) 추가 저장
# ==========================================

import os
import sys
import glob
import shutil
import json
import time
import numpy as np

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
BASE_OUTPUT_DIR = os.path.join(PROJECT_DIR, "dataset_pos")
IMAGES_PER_CLASS = 500
RESOLUTION = (1024, 1024)
CLEAR_EXISTING = True


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


def parse_bbox_file(bbox_npy_path, labels_json_path, class_name):
    """bbox 파일을 파싱하여 클래스에 해당하는 bbox 정보 추출"""
    if not os.path.exists(bbox_npy_path) or not os.path.exists(labels_json_path):
        return None
    
    try:
        bbox_data = np.load(bbox_npy_path)
        with open(labels_json_path, 'r') as f:
            labels_data = json.load(f)
        
        if len(bbox_data) == 0:
            return None
        
        id_to_labels = labels_data.get("idToLabels", {})
        
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
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "width": x_max - x_min,
                    "height": y_max - y_min,
                    "center_x": (x_min + x_max) / 2.0,
                    "center_y": (y_min + y_max) / 2.0,
                    "area": area,
                    "visibility": 1.0 - occlusion
                }
        
        return best_bbox
    except Exception as e:
        print(f"    ⚠️  bbox 파싱 에러: {e}")
        return None


def generate_pose_labels(class_output_dir, class_name, part_center, part_size, meters_per_unit):
    """생성된 bbox 파일들을 읽어서 pose 라벨 생성"""
    print(f"  📝 Pose 라벨 생성 중...")
    
    bbox_files = sorted(glob.glob(os.path.join(class_output_dir, "bounding_box_2d_tight_*.npy")))
    generated_count = 0
    
    for bbox_npy_path in bbox_files:
        # 파일 인덱스 추출
        basename = os.path.basename(bbox_npy_path)
        idx_str = basename.replace("bounding_box_2d_tight_", "").replace(".npy", "")
        try:
            frame_idx = int(idx_str)
        except:
            continue
        
        labels_json_path = os.path.join(class_output_dir, f"bounding_box_2d_tight_labels_{idx_str}.json")
        rgb_path = os.path.join(class_output_dir, f"rgb_{idx_str}.png")
        
        if not os.path.exists(rgb_path):
            continue
        
        bbox_info = parse_bbox_file(bbox_npy_path, labels_json_path, class_name)
        
        if bbox_info is None:
            # bbox가 없어도 pose 파일은 생성 (빈 bbox로)
            bbox_info = {
                "x_min": 0, "y_min": 0, "x_max": 0, "y_max": 0,
                "width": 0, "height": 0, "center_x": 0, "center_y": 0,
                "area": 0, "visibility": 0
            }
        
        # pose 라벨 생성
        pose_data = {
            "class_name": class_name,
            "frame_index": frame_idx,
            "bbox_2d": bbox_info,
            "normalized_center": {
                "x": bbox_info["center_x"] / RESOLUTION[0],
                "y": bbox_info["center_y"] / RESOLUTION[1]
            },
            "object_info": {
                "part_center_world": list(part_center) if hasattr(part_center, '__iter__') else [part_center, part_center, part_center],
                "part_size": part_size,
                "meters_per_unit": meters_per_unit
            },
            "image_info": {
                "resolution": list(RESOLUTION),
                "rgb_file": f"rgb_{idx_str}.png"
            }
        }
        
        pose_path = os.path.join(class_output_dir, f"pose_{idx_str}.json")
        with open(pose_path, 'w', encoding='utf-8') as f:
            json.dump(pose_data, f, indent=2, ensure_ascii=False)
        
        generated_count += 1
    
    print(f"  ✓ Pose 라벨 {generated_count}개 생성 완료")
    return generated_count


def generate_class_dataset(part_config, class_index, total_classes):
    """한 클래스의 데이터셋 생성 (class_estimation 방식)"""
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
    
    # Semantics 추가
    print(f"  Semantics 추가 중...")
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Imageable):
            sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
            sem.CreateSemanticTypeAttr("class")
            sem.CreateSemanticDataAttr().Set(class_name)
    
    # ==========================================
    # Replicator 설정 (class_estimation과 동일)
    # ==========================================
    print(f"  Replicator 설정 중...")
    
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
                with rep.create.group([camera]):
                    rep.modify.pose(
                        position=rep.distribution.uniform(
                            (part_center[0] - camera_distance_max * 0.7,
                             part_center[1] - camera_distance_max * 0.5,
                             part_center[2] + camera_distance_min * 0.3),
                            (part_center[0] + camera_distance_max * 0.7,
                             part_center[1] + camera_distance_max * 0.5,
                             part_center[2] + camera_distance_max * 0.9)
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
            
            # Writer 설정
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=class_output_dir, rgb=True, bounding_box_2d_tight=True)
            writer.attach([render_product])
            print(f"  ✓ Writer 설정")
            
            # 데이터 생성
            print(f"\n  🎬 데이터 생성 시작 ({IMAGES_PER_CLASS}장)...")
            
            for i in range(10):
                simulation_app.update()
                time.sleep(0.1)
            
            files_before = len(glob.glob(os.path.join(class_output_dir, "rgb_*.png")))
            
            # 핵심: run_until_complete() 사용
            rep.orchestrator.run_until_complete()
            
            time.sleep(0.5)
            for i in range(10):
                simulation_app.update()
                time.sleep(0.1)
            
            files_after = len(glob.glob(os.path.join(class_output_dir, "rgb_*.png")))
            print(f"  ✓ 이미지 생성 완료: {files_after}장")
            
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
    
    # Pose 라벨 생성 (bbox 파일 파싱)
    pose_count = generate_pose_labels(class_output_dir, class_name, part_center, part_size, meters_per_unit)
    
    # 메타데이터 저장
    metadata = {
        "class_name": class_name,
        "num_images": files_after,
        "num_pose_labels": pose_count,
        "part_size": float(part_size),
        "part_center": list(part_center),
        "meters_per_unit": meters_per_unit
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
    print("(class_estimation 기반 안정 버전)")
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
        "name": "Excavator Parts Pose Dataset",
        "num_classes": len(parts_config),
        "images_per_class": IMAGES_PER_CLASS,
        "classes": list(parts_config.keys()),
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
            pose_count = len(glob.glob(os.path.join(class_dir, "pose_*.json")))
            print(f"  {name}: RGB {rgb_count}장, Pose {pose_count}개")
    
    print("\n종료하려면 Ctrl+C를 누르세요.")
    while simulation_app.is_running():
        simulation_app.update()
    
    simulation_app.close()
