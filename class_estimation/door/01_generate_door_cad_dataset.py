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
from pxr import Usd, UsdGeom, UsdShade, Semantics, Gf, Sdf

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

# 8클래스: USD 파일 + 카메라 관찰 방향
# 사용자가 Isaac Sim에서 직접 생성한 USD (m 단위, +Z 방향에서 관찰)
DOOR_CLASSES = {
    "E25_door_LH_FRT": {"usd": "E25_door_LH_FRT.usd", "view_dir": "+Z"},
    "E25_door_LH_RR":  {"usd": "E25_door_LH_RR.usd",  "view_dir": "+Z"},
    "E25_door_RH":     {"usd": "E25_door_RH.usd",      "view_dir": "+Z"},
    "E30_door_LH_FRT": {"usd": "E30_door_LH_FRT.usd", "view_dir": "+Z"},
    "E30_door_LH_RR":  {"usd": "E30_door_LH_RR.usd",  "view_dir": "+Z"},
    "E30_E38_door_RH": {"usd": "E30_E38_door_RH.usd", "view_dir": "+Z"},
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

    # USD 로드 (충분한 대기)
    omni.usd.get_context().open_stage(usd_path)
    for _ in range(30):
        simulation_app.update()
        time.sleep(0.1)

    stage = omni.usd.get_context().get_stage()
    if not stage:
        print(f"  ⚠️  스테이지 로드 실패")
        return

    # 스테이지 정보
    up_axis = UsdGeom.GetStageUpAxis(stage)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"  스테이지: upAxis={up_axis}, metersPerUnit={mpu}")

    # 프림 구조 디버깅
    prim_count = 0
    mesh_count = 0
    for prim in stage.Traverse():
        prim_count += 1
        if prim.IsA(UsdGeom.Mesh):
            mesh_count += 1
    print(f"  프림 수: {prim_count}, 메시 수: {mesh_count}")

    if mesh_count == 0:
        print(f"  ⚠️  메시가 없습니다! USD 파일 내용 확인 필요")
        for prim in stage.Traverse():
            print(f"    - {prim.GetPath()} ({prim.GetTypeName()})")
        return

    # AABB 계산
    axis_idx = 1 if up_axis == UsdGeom.Tokens.y else 2
    aabb = compute_world_aabb(stage)
    if not aabb:
        print(f"  ⚠️  AABB 계산 실패")
        return

    size, center = aabb["size"], aabb["center"]
    part_size = max(size[0], size[1], size[2])
    cx, cy, cz = center[0], center[1], center[2]
    floor_height = float(aabb["world_min"][axis_idx])

    print(f"  AABB 크기: ({size[0]:.4f}, {size[1]:.4f}, {size[2]:.4f})")
    print(f"  AABB 중심: ({cx:.4f}, {cy:.4f}, {cz:.4f})")
    print(f"  AABB min: ({aabb['world_min'][0]:.4f}, {aabb['world_min'][1]:.4f}, {aabb['world_min'][2]:.4f})")
    print(f"  AABB max: ({aabb['world_max'][0]:.4f}, {aabb['world_max'][1]:.4f}, {aabb['world_max'][2]:.4f})")
    print(f"  최대 치수: {part_size:.4f}, 바닥 높이: {floor_height:.4f}")

    # Material 확인 및 기본 material 적용 (STL 변환 USD에는 material 없음)
    mesh_prims_list = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    has_material = False
    for mp in mesh_prims_list:
        binding = UsdShade.MaterialBindingAPI(mp)
        mat, _ = binding.ComputeBoundMaterial()
        if mat:
            has_material = True
            break

    if not has_material:
        print(f"  Material 없음 → 기본 OmniPBR 생성 중...")
        mat_path = Sdf.Path("/World/DefaultDoorMaterial")
        material = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, mat_path.AppendChild("Shader"))
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set((0.6, 0.6, 0.65))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.7)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.3)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        for mp in mesh_prims_list:
            UsdShade.MaterialBindingAPI.Apply(mp).Bind(material)
        print(f"  ✓ {len(mesh_prims_list)}개 메시에 material 바인딩 완료")
    else:
        print(f"  ✓ Material 이미 존재")

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

    # 장면 구성 (참조 스크립트와 동일한 패턴: 위에서 내려다보는 구도)
    # +Z 카메라: 카메라가 위(+Z)에서 아래로 내려다봄
    #   바닥: Z=floor_height, XY 수평면 (도어 아래)
    #   뒷벽: 도어 뒤(-Y 방향), XZ 수직면
    lateral_x = cam_max * 0.7
    lateral_y = cam_max * 0.5

    if view_dir == "+Z":
        cam_pos_lo = (cx - lateral_x, cy - lateral_y, cz + cam_min * 0.3)
        cam_pos_hi = (cx + lateral_x, cy + lateral_y, cz + cam_max)
        init_cam = (cx + min_cam * 0.3, cy + min_cam * 0.2, cz + min_cam * 0.8)
        floor_pos = (cx, cy, floor_height - part_size * 0.1)
        floor_rot = (0, 0, 0)
        wall_pos = (cx, cy - part_size * 2.5, cz)
        wall_rot = (90, 0, 0)
    else:  # "-Z"
        cam_pos_lo = (cx - lateral_x, cy - lateral_y, cz - cam_max)
        cam_pos_hi = (cx + lateral_x, cy + lateral_y, cz - cam_min * 0.3)
        init_cam = (cx + min_cam * 0.3, cy + min_cam * 0.2, cz - min_cam * 0.8)
        floor_pos = (cx, cy, floor_height - part_size * 0.1)
        floor_rot = (0, 0, 0)
        wall_pos = (cx, cy + part_size * 2.5, cz)
        wall_rot = (90, 0, 0)

    print(f"  카메라 거리: {cam_min:.4f} ~ {cam_max:.4f}")
    print(f"  초기 카메라: ({init_cam[0]:.4f}, {init_cam[1]:.4f}, {init_cam[2]:.4f})")
    print(f"  카메라 lo: ({cam_pos_lo[0]:.4f}, {cam_pos_lo[1]:.4f}, {cam_pos_lo[2]:.4f})")
    print(f"  카메라 hi: ({cam_pos_hi[0]:.4f}, {cam_pos_hi[1]:.4f}, {cam_pos_hi[2]:.4f})")
    print(f"  바닥 위치: ({floor_pos[0]:.4f}, {floor_pos[1]:.4f}, {floor_pos[2]:.4f}), 회전: {floor_rot}")
    print(f"  벽 위치: ({wall_pos[0]:.4f}, {wall_pos[1]:.4f}, {wall_pos[2]:.4f}), 회전: {wall_rot}")

    try:
        with rep.new_layer():
            mesh_prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
            if not mesh_prims:
                print(f"  ⚠️  메시 없음")
                return

            rep.get.prims(semantics=[("class", class_name)])

            floor_size = part_size * 5
            floor_plane = rep.create.plane(
                scale=(floor_size, floor_size, 1),
                position=floor_pos, rotation=floor_rot,
                semantics=[("class", "background")]
            )

            back_wall = rep.create.plane(
                scale=(floor_size, floor_size * 0.5, 1),
                position=wall_pos, rotation=wall_rot,
                semantics=[("class", "background")]
            )

            dome_light = rep.create.light(
                light_type="Dome", intensity=1500.0, rotation=(270, 0, 0)
            )
            light1 = rep.create.light(
                light_type="Sphere", intensity=80000.0,
                position=(cx + part_size, cy + part_size * 0.5, cz + part_size * 2),
                scale=0.5
            )
            light2 = rep.create.light(
                light_type="Sphere", intensity=50000.0,
                position=(cx - part_size, cy - part_size * 0.3, cz + part_size * 1.5),
                scale=0.3
            )

            camera = rep.create.camera(position=init_cam, look_at=(cx, cy, cz))
            render_product = rep.create.render_product(camera, resolution=(1024, 1024))

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
                        (cx - part_size * 1.5, cy - part_size, cz + part_size * 1.5),
                        (cx + part_size * 1.5, cy + part_size, cz + part_size * 3)
                    ))
                with light2:
                    rep.modify.pose(position=rep.distribution.uniform(
                        (cx - part_size * 1.5, cy - part_size, cz + part_size * 0.5),
                        (cx + part_size * 1.5, cy + part_size, cz + part_size * 2)
                    ))

            # Writer (RGB만, bbox 없음)
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(output_dir=class_output_dir, rgb=True)
            writer.attach([render_product])

            # 렌더링 워밍업 (첫 프레임 안정화)
            print(f"  렌더링 워밍업 중...")
            for _ in range(30):
                simulation_app.update()
                time.sleep(0.1)

            # 데이터 생성
            print(f"  데이터 생성 시작 ({IMAGES_PER_CLASS}장)...")
            rep.orchestrator.run_until_complete()

            for _ in range(20):
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
