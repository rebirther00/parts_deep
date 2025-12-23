# ==========================================
# 단일 이미지 테스트 스크립트
# boom_link_30_notmat.usd 파일로 1장만 생성
# ==========================================

import os
import sys
import math

# Isaac Sim 초기화
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# 모듈 임포트
import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom, Semantics, Gf
import time
import numpy as np

# ==========================================
# 설정
# ==========================================
USD_FILE = "/home/rebirther/isaac-sim/assets/boom_link_30_notmat.usd"
OUTPUT_DIR = "/home/rebirther/isaac_data_output/test_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("단일 이미지 테스트")
print("="*60)
print(f"USD 파일: {USD_FILE}")
print(f"출력 디렉토리: {OUTPUT_DIR}")

# ==========================================
# USD 파일 로드
# ==========================================
print("\n1. USD 파일 로딩...")
omni.usd.get_context().open_stage(USD_FILE)
time.sleep(1.0)

stage = omni.usd.get_context().get_stage()
if not stage:
    print("  ⚠️  스테이지를 가져올 수 없습니다.")
    simulation_app.close()
    sys.exit(1)

# ==========================================
# 바운딩 박스 계산
# ==========================================
print("\n2. 바운딩 박스 계산...")

time_code = Usd.TimeCode.Default()
bbox_cache = UsdGeom.BBoxCache(time_code, includedPurposes=[UsdGeom.Tokens.default_])

# 모든 프림 확인
print("  스테이지 구조:")
for prim in stage.Traverse():
    prim_type = prim.GetTypeName()
    print(f"    {prim.GetPath()} ({prim_type})")
    
    # Xformable인 경우 Transform 출력
    if prim.IsA(UsdGeom.Xformable):
        xform = UsdGeom.Xformable(prim)
        local_xform = xform.GetLocalTransformation()
        print(f"      LocalTransform: {local_xform}")

# UpAxis 확인 (사용자 환경: Z-up)
up_axis = UsdGeom.GetStageUpAxis(stage)
print(f"\n  Stage UpAxis: {up_axis}")
axis_index = 2  # Z-up 기본
if up_axis == UsdGeom.Tokens.y:
    axis_index = 1
elif up_axis == UsdGeom.Tokens.x:
    axis_index = 0

# 루트 프림 찾기 (Scale이 적용된 프림)
default_prim = stage.GetDefaultPrim()
print(f"\n  Default Prim: {default_prim.GetPath() if default_prim else 'None'}")

# 루트 프림의 자식 중 Xformable 찾기
root_xform = None
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Xformable) and prim.GetPath() != "/World":
        xform = UsdGeom.Xformable(prim)
        xform_matrix = xform.GetLocalTransformation()
        # Scale이 적용된 프림 찾기
        if prim.GetPath().pathString.count('/') == 2:  # /World/xxx 형태
            root_xform = prim
            print(f"  루트 Xform: {prim.GetPath()}")
            print(f"    LocalTransformation: {xform_matrix}")
            break

# 비교용: BBoxCache(문제 재현) - root_xform이 있으면 그것의 WorldBound를 찍어둠
if root_xform:
    try:
        root_bbox = bbox_cache.ComputeWorldBound(root_xform)
        root_range = root_bbox.GetRange()
        print("\n  [비교용] BBoxCache WorldBound(root_xform):")
        print(f"    Min: {root_range.GetMin()}")
        print(f"    Max: {root_range.GetMax()}")
        print(f"    Size: {root_range.GetSize()}")
        print(f"    바닥(UpAxis min): {root_range.GetMin()[axis_index]}")
    except Exception as e:
        print(f"\n  ⚠️  BBoxCache 비교 출력 실패: {e}")

# 옵션 1: vertex 기반 월드 AABB/바닥 높이 계산 (Orient/Scale/Translate 포함)
print("\n  [옵션1] Vertex 기반 월드 AABB 계산 (Orient 포함)...")
xform_cache = UsdGeom.XformCache(time_code)
mesh_prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
print(f"  Mesh 개수: {len(mesh_prims)}개")

if not mesh_prims:
    print("  ⚠️  메시를 찾을 수 없습니다.")
    simulation_app.close()
    sys.exit(1)

world_min = Gf.Vec3d(float("inf"), float("inf"), float("inf"))
world_max = Gf.Vec3d(float("-inf"), float("-inf"), float("-inf"))
total_points = 0

for prim in mesh_prims:
    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    if not pts:
        continue

    M = xform_cache.GetLocalToWorldTransform(prim)  # orient 포함
    total_points += len(pts)

    # 정확도를 위해 모든 vertex를 변환 (단일 테스트 스크립트 목적)
    for p in pts:
        # p는 보통 Vec3f이므로 Vec3d로 변환
        wp = M.Transform(Gf.Vec3d(p[0], p[1], p[2]))
        world_min = Gf.Vec3d(
            min(world_min[0], wp[0]),
            min(world_min[1], wp[1]),
            min(world_min[2], wp[2]),
        )
        world_max = Gf.Vec3d(
            max(world_max[0], wp[0]),
            max(world_max[1], wp[1]),
            max(world_max[2], wp[2]),
        )

size = world_max - world_min
center = (world_min + world_max) / 2.0
print(f"  총 vertex 수(합계): {total_points:,}")
print("\n  [Vertex 기반 월드 AABB]")
print(f"    Min: {world_min}")
print(f"    Max: {world_max}")
print(f"    Size: {size}")
print(f"    Center: {center}")
print(f"    바닥(UpAxis min): {world_min[axis_index]}")

part_size = max(size[0], size[1], size[2])
part_center = (center[0], center[1], center[2])  # 옵션1: 오브젝트는 그대로, 중심만 사용

print(f"\n  부품 정보:")
print(f"    중심: ({part_center[0]:.3f}, {part_center[1]:.3f}, {part_center[2]:.3f})")
print(f"    크기: {part_size:.3f}")
print(f"    바닥(UpAxis min): {world_min[axis_index]:.3f}")

# ==========================================
# Semantics 추가
# ==========================================
print("\n3. Semantics 추가...")
class_name = "boom_link_30"
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Imageable):
        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr("class")
        sem.CreateSemanticDataAttr().Set(class_name)
print("  ✓ Semantics 추가 완료")

# ==========================================
# Replicator 설정
# ==========================================
print("\n4. Replicator 설정...")

with rep.new_layer():
    # 옵션 1: 오브젝트는 그대로 두고, 바닥 평면만 '진짜 바닥 높이'에 배치
    floor_size = part_size * 3
    floor_height = float(world_min[axis_index])

    # UpAxis에 따라 평면 위치/회전 결정
    if axis_index == 2:
        # Z-up: 바닥은 XY 평면, Z=floor_height
        floor_pos = (part_center[0], part_center[1], floor_height)
        floor_rot = (0, 0, 0)
    elif axis_index == 1:
        # Y-up: 바닥은 XZ 평면, Y=floor_height (plane은 기본 XY이므로 X축 90도 회전)
        floor_pos = (part_center[0], floor_height, part_center[2])
        floor_rot = (90, 0, 0)
    else:
        # X-up: 바닥은 YZ 평면, X=floor_height (plane을 Z축 90도 회전 후 Y축 90도 회전 등 복잡하므로 단순 처리)
        floor_pos = (floor_height, part_center[1], part_center[2])
        floor_rot = (0, 90, 0)

    print(f"  바닥 평면 생성(옵션1):")
    print(f"    UpAxis: {up_axis}")
    print(f"    바닥 높이: {floor_height:.3f}")
    print(f"    위치: ({floor_pos[0]:.3f}, {floor_pos[1]:.3f}, {floor_pos[2]:.3f})")
    print(f"    회전: {floor_rot}")
    print(f"    크기: {floor_size:.3f}")
    
    floor_plane = rep.create.plane(
        scale=(floor_size, floor_size, 1),
        position=floor_pos,
        rotation=floor_rot,
        semantics=[("class", "background")]
    )
    
    # 뒷벽 평면
    back_wall = rep.create.plane(
        scale=(floor_size, floor_size * 0.5, 1),
        position=(part_center[0] - floor_size * 0.4, part_center[1], part_center[2]),
        rotation=(0, 90, 0),
        semantics=[("class", "background")]
    )
    
    # 조명
    dome_light = rep.create.light(
        light_type="Dome",
        intensity=800.0,
        rotation=(270, 0, 0)
    )
    
    # 카메라 거리 계산
    part_diagonal = math.sqrt(size[0]**2 + size[1]**2 + size[2]**2)
    camera_fov_rad = np.radians(60)
    camera_distance = (part_diagonal / 2.0) / np.tan(camera_fov_rad / 2.0) / 0.8
    
    print(f"  카메라 생성:")
    print(f"    거리: {camera_distance:.3f}")
    
    # 카메라 위치 계산 (오브젝트 중심을 바라보도록)
    camera_pos = (
        part_center[0] + camera_distance * 0.7,
        part_center[1] + camera_distance * 0.5,
        part_center[2] + camera_distance * 0.5
    )
    print(f"    위치: ({camera_pos[0]:.3f}, {camera_pos[1]:.3f}, {camera_pos[2]:.3f})")
    print(f"    Look at: ({part_center[0]:.3f}, {part_center[1]:.3f}, {part_center[2]:.3f})")
    
    camera = rep.create.camera(
        position=camera_pos,
        look_at=part_center
    )
    
    # Render product
    render_product = rep.create.render_product(camera, resolution=(1024, 1024))
    
    # Writer
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=OUTPUT_DIR,
        rgb=True
    )
    writer.attach([render_product])
    
    # 1장만 생성
    with rep.trigger.on_frame(max_execs=1):
        # 바닥 색상
        with floor_plane:
            rep.randomizer.color(colors=rep.distribution.uniform((0.4, 0.4, 0.4), (0.6, 0.5, 0.4)))
        with back_wall:
            rep.randomizer.color(colors=rep.distribution.uniform((0.7, 0.7, 0.7), (0.9, 0.9, 0.85)))
    
    print("\n5. 이미지 생성...")
    
    # 시뮬레이션 업데이트
    for _ in range(10):
        simulation_app.update()
        time.sleep(0.1)
    
    # 실행
    rep.orchestrator.run_until_complete()
    print("  ✓ 이미지 생성 완료!")
    
    # 추가 업데이트
    for _ in range(10):
        simulation_app.update()
        time.sleep(0.1)

# ==========================================
# 결과 확인
# ==========================================
print("\n6. 결과 확인...")
import glob
generated_files = glob.glob(os.path.join(OUTPUT_DIR, "rgb_*.png"))
print(f"  생성된 파일 개수: {len(generated_files)}")
for f in generated_files:
    print(f"    {os.path.basename(f)}")

print("\n" + "="*60)
print("테스트 완료!")
print("="*60)
print(f"출력 디렉토리: {OUTPUT_DIR}")

# 시뮬레이터 유지 (확인용)
print("\n시뮬레이션을 계속 실행합니다. 종료하려면 Ctrl+C를 누르세요.")
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()

