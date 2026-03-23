# ==========================================
# Door CAD 합성 데이터셋 생성 스크립트
# (방향별 카메라 + Domain Randomization)
#
# 실행: ~/isaac-sim/python.sh class_estimation/door/01_generate_door_cad_dataset.py
# ==========================================

import os
import sys
import time
import json
import glob
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.dirname(PROJECT_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from utils.logger import setup_logging, reinit_logging, finish_logging

LOG_PATH = setup_logging("01_generate_door_cad")

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom, Semantics, Gf

reinit_logging(LOG_PATH)

# ==========================================
# 설정
# ==========================================
ASSETS_DIR = os.path.expanduser("~/isaac-sim/assets/door")
BASE_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "datasets_cad")
IMAGES_PER_CLASS = 500
CLEAR_EXISTING_DATA = True
BACKGROUND_MODE = "random"
BACKGROUND_RATIOS = {"none": 0.2, "solid": 0.3, "factory": 0.5}

# 8클래스: USD 파일 + 카메라 관찰 방향 (복잡한 형상이 있는 후면)
# 카메라는 지정된 방향 중심으로 ±45° 범위에서 랜덤 배치
DOOR_CLASSES = {
    "E25_door_LH_FRT": {"usd": "E25_door_LH_FRT.usd", "view_dir": "-Z"},
    "E25_door_LH_RR":  {"usd": "E25_door_LH_RR.usd",  "view_dir": "-Z"},
    "E25_door_RH":     {"usd": "E25_door_RH.usd",      "view_dir": "-Z"},
    "E30_door_LH_FRT": {"usd": "E30_door_LH_FRT.usd", "view_dir": "+Z"},
    "E30_door_LH_RR":  {"usd": "E30_door_LH_RR.usd",  "view_dir": "+Z"},
    "E30_E38_door_RH": {"usd": "E30_E38_door_RH.usd", "view_dir": "-Z"},
    "E38_door_LH_FRT": {"usd": "E38_door_LH_FRT.usd", "view_dir": "+Z"},
    "E38_door_LH_RR":  {"usd": "E38_door_LH_RR.usd",  "view_dir": "+Z"},
}

TOTAL_FRAMES = IMAGES_PER_CLASS * len(DOOR_CLASSES)

if CLEAR_EXISTING_DATA and os.path.exists(BASE_OUTPUT_DIR):
    import shutil
    print(f"⚠️  기존 데이터셋 삭제 중: {BASE_OUTPUT_DIR}")
    shutil.rmtree(BASE_OUTPUT_DIR)

print("=" * 60)
print("Door CAD 합성 데이터셋 생성")
print("=" * 60)
print(f"클래스: {len(DOOR_CLASSES)}개 | 클래스당: {IMAGES_PER_CLASS}장 | 총: {TOTAL_FRAMES}장")
print(f"배경: {BACKGROUND_MODE} | 출력: {BASE_OUTPUT_DIR}")
print("=" * 60)


# ==========================================
# AABB 계산
# ==========================================
def compute_world_aabb(stage):
    """메시 vertex를 월드 좌표로 변환하여 AABB 계산"""
    time_code = Usd.TimeCode.Default()
    xform_cache = UsdGeom.XformCache(time_code)
    mesh_prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    if not mesh_prims:
        return None

    w_min = Gf.Vec3d(float("inf"), float("inf"), float("inf"))
    w_max = Gf.Vec3d(float("-inf"), float("-inf"), float("-inf"))

    for prim in mesh_prims:
        mesh = UsdGeom.Mesh(prim)
        M = xform_cache.GetLocalToWorldTransform(prim)
        pts = mesh.GetPointsAttr().Get(time_code)
        if not pts:
            extent = mesh.GetExtentAttr().Get(time_code)
            if not extent or len(extent) != 2:
                continue
            for x in (extent[0][0], extent[1][0]):
                for y in (extent[0][1], extent[1][1]):
                    for z in (extent[0][2], extent[1][2]):
                        wp = M.Transform(Gf.Vec3d(x, y, z))
                        w_min = Gf.Vec3d(min(w_min[0], wp[0]), min(w_min[1], wp[1]), min(w_min[2], wp[2]))
                        w_max = Gf.Vec3d(max(w_max[0], wp[0]), max(w_max[1], wp[1]), max(w_max[2], wp[2]))
            continue
        for p in pts:
            wp = M.Transform(Gf.Vec3d(p[0], p[1], p[2]))
            w_min = Gf.Vec3d(min(w_min[0], wp[0]), min(w_min[1], wp[1]), min(w_min[2], wp[2]))
            w_max = Gf.Vec3d(max(w_max[0], wp[0]), max(w_max[1], wp[1]), max(w_max[2], wp[2]))

    return {
        "world_min": w_min, "world_max": w_max,
        "size": w_max - w_min, "center": (w_min + w_max) / 2.0,
    }


# ==========================================
# 클래스별 데이터 생성
# ==========================================
def generate_class_dataset(class_name, class_config, class_index):
    usd_path = os.path.join(ASSETS_DIR, class_config["usd"])
    view_dir = class_config["view_dir"]
    class_output_dir = os.path.join(BASE_OUTPUT_DIR, class_name)
    os.makedirs(class_output_dir, exist_ok=True)

    print(f"\n[{class_index+1}/{len(DOOR_CLASSES)}] {class_name} (카메라: {view_dir})")

    # 이전 클래스 상태 정리
    if class_index > 0:
        for _ in range(10):
            simulation_app.update()
            time.sleep(0.05)

    if not os.path.exists(usd_path):
        print(f"  ⚠️  USD 파일 없음: {usd_path}")
        return

    # USD 로드
    omni.usd.get_context().open_stage(usd_path)
    time.sleep(1.0)
    stage = omni.usd.get_context().get_stage()
    if not stage:
        print(f"  ⚠️  스테이지 로드 실패")
        return

    # AABB 계산
    up_axis = UsdGeom.GetStageUpAxis(stage)
    axis_idx = 1 if up_axis == UsdGeom.Tokens.y else 2

    aabb = compute_world_aabb(stage)
    if not aabb:
        print(f"  ⚠️  AABB 계산 실패")
        return

    size, center = aabb["size"], aabb["center"]
    part_size = max(size[0], size[1], size[2])
    cx, cy, cz = center[0], center[1], center[2]
    floor_height = float(aabb["world_min"][axis_idx])

    print(f"  크기: ({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f})")
    print(f"  중심: ({cx:.4f}, {cy:.4f}, {cz:.4f})")

    # Semantics 추가
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Imageable):
            sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
            sem.CreateSemanticTypeAttr("class")
            sem.CreateSemanticDataAttr().Set(class_name)

    # 카메라 거리 계산
    part_diag = np.sqrt(size[0]**2 + size[1]**2 + size[2]**2)
    min_cam = (part_diag / 2.0) / np.tan(np.radians(30)) / 0.8
    cam_min, cam_max = min_cam * 0.9, min_cam * 1.5

    # 카메라 방향별 위치 범위 (지정 방향 중심 ±45°)
    lateral = cam_max * 0.7
    if view_dir == "-Z":
        cam_pos_lo = (cx - lateral, cy - lateral, cz - cam_max)
        cam_pos_hi = (cx + lateral, cy + lateral, cz - cam_min * 0.5)
        init_cam = (cx, cy, cz - min_cam)
        wall_z = cz + part_size * 0.4
    else:  # "+Z"
        cam_pos_lo = (cx - lateral, cy - lateral, cz + cam_min * 0.5)
        cam_pos_hi = (cx + lateral, cy + lateral, cz + cam_max)
        init_cam = (cx, cy, cz + min_cam)
        wall_z = cz - part_size * 0.4

    print(f"  카메라 거리: {cam_min:.3f} ~ {cam_max:.3f}")

    try:
        with rep.new_layer():
            mesh_prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
            if not mesh_prims:
                print(f"  ⚠️  메시 없음")
                return

            rep.get.prims(semantics=[("class", class_name)])

            # 바닥 평면
            floor_size = part_size * 5
            if axis_idx == 2:
                floor_pos, floor_rot = (cx, cy, floor_height), (0, 0, 0)
            else:
                floor_pos, floor_rot = (cx, floor_height, cz), (90, 0, 0)

            floor_plane = rep.create.plane(
                scale=(floor_size, floor_size, 1),
                position=floor_pos, rotation=floor_rot,
                semantics=[("class", "background")]
            )

            # 뒷벽 (카메라 반대편)
            back_wall = rep.create.plane(
                scale=(floor_size, floor_size * 0.5, 1),
                position=(cx, cy, wall_z), rotation=(0, 0, 0),
                semantics=[("class", "background")]
            )

            # 조명
            dome_light = rep.create.light(
                light_type="Dome", intensity=800.0, rotation=(270, 0, 0)
            )
            light1 = rep.create.light(
                light_type="Sphere", intensity=50000.0,
                position=(cx + part_size, cy + part_size, cz + part_size * 2), scale=0.5
            )
            light2 = rep.create.light(
                light_type="Sphere", intensity=30000.0,
                position=(cx - part_size, cy - part_size * 0.5, cz + part_size), scale=0.3
            )

            # 카메라
            camera = rep.create.camera(position=init_cam, look_at=(cx, cy, cz))
            render_product = rep.create.render_product(camera, resolution=(1024, 1024))

            # 프레임별 랜덤화
            with rep.trigger.on_frame(max_execs=IMAGES_PER_CLASS):
                with rep.create.group([camera]):
                    rep.modify.pose(
                        position=rep.distribution.uniform(cam_pos_lo, cam_pos_hi),
                        look_at=(cx, cy, cz)
                    )
                with floor_plane:
                    rep.randomizer.color(
                        colors=rep.distribution.uniform((0.2, 0.2, 0.2), (0.6, 0.5, 0.4))
                    )
                with back_wall:
                    rep.randomizer.color(
                        colors=rep.distribution.uniform((0.4, 0.4, 0.4), (0.9, 0.9, 0.85))
                    )
                with light1:
                    rep.modify.pose(position=rep.distribution.uniform(
                        (cx - part_size * 1.5, cy - part_size, cz - part_size * 2),
                        (cx + part_size * 1.5, cy + part_size, cz + part_size * 2)
                    ))
                with light2:
                    rep.modify.pose(position=rep.distribution.uniform(
                        (cx - part_size * 1.5, cy - part_size, cz - part_size * 2),
                        (cx + part_size * 1.5, cy + part_size, cz + part_size * 2)
                    ))

            # Writer (RGB만, bbox 없음)
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=class_output_dir, rgb=True)
            writer.attach([render_product])

            # 데이터 생성
            print(f"  데이터 생성 시작 ({IMAGES_PER_CLASS}장)...")
            for i in range(10):
                simulation_app.update()
                time.sleep(0.1)

            rep.orchestrator.run_until_complete()

            time.sleep(0.5)
            for _ in range(10):
                simulation_app.update()
                time.sleep(0.1)

            generated = glob.glob(os.path.join(class_output_dir, "rgb_*.png"))
            print(f"  ✓ {class_name} 완료 ({len(generated)}장)")

    except Exception as e:
        print(f"  ⚠️  에러: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        for _ in range(20):
            simulation_app.update()
            time.sleep(0.1)

    # 메타데이터
    metadata = {
        "class_name": class_name,
        "data_source": "isaac_sim_cad",
        "view_direction": view_dir,
        "num_images": IMAGES_PER_CLASS,
        "part_center": [cx, cy, cz],
        "part_size": float(part_size),
        "background_mode": BACKGROUND_MODE,
    }
    meta_path = os.path.join(class_output_dir, "metadata.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


# ==========================================
# 전체 데이터셋 생성
# ==========================================
print("\n전체 데이터셋 생성 시작...")

for idx, (name, config) in enumerate(DOOR_CLASSES.items()):
    print(f"\n{'='*60}")
    generate_class_dataset(name, config, idx)

    if idx < len(DOOR_CLASSES) - 1:
        print("\n스테이지 정리 중...")
        try:
            rep.orchestrator.stop()
            for _ in range(30):
                simulation_app.update()
                time.sleep(0.1)
            time.sleep(1.0)
        except Exception as e:
            print(f"  정리 에러 (무시): {e}")

# 전체 메타데이터
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
dataset_meta = {
    "dataset_name": "Excavator Door CAD Classification Dataset",
    "num_classes": len(DOOR_CLASSES),
    "images_per_class": IMAGES_PER_CLASS,
    "total_images": TOTAL_FRAMES,
    "classes": {k: k for k in DOOR_CLASSES},
    "data_source": "isaac_sim_cad",
    "background_mode": BACKGROUND_MODE,
    "background_ratios": BACKGROUND_RATIOS,
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
with open(os.path.join(BASE_OUTPUT_DIR, "dataset_info.json"), 'w', encoding='utf-8') as f:
    json.dump(dataset_meta, f, indent=2, ensure_ascii=False)

# 결과 출력
print("\n" + "=" * 60)
print("데이터셋 생성 완료!")
print("=" * 60)
for name in DOOR_CLASSES:
    d = os.path.join(BASE_OUTPUT_DIR, name)
    if os.path.exists(d):
        count = len(glob.glob(os.path.join(d, "rgb_*.png")))
        print(f"  {name}: {count}장")

finish_logging()

print("\n종료하려면 Ctrl+C를 누르세요.")
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
