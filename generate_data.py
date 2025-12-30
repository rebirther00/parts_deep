# ==========================================
# Isaac Sim Replicator를 사용한 데이터 생성 스크립트
# ==========================================
from isaacsim import SimulationApp

# 시뮬레이터 초기화
simulation_app = SimulationApp({"headless": False})

# ==========================================
# 모듈 임포트 (App 실행 후)
# ==========================================
import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom, Semantics
import os
import time
import numpy as np

# ==========================================
# 설정
# ==========================================
usd_path = "/home/rebirther/isaac-sim/assets/boom_link_25.usd"
output_dir = "/home/rebirther/isaac_data_output/datasets"

# datasets 폴더가 없으면 생성
os.makedirs(output_dir, exist_ok=True)

# USD 파일 존재 확인
if not os.path.exists(usd_path):
    print(f"ERROR: USD 파일을 찾을 수 없습니다: {usd_path}")
    simulation_app.close()
    exit(1)

print(f"USD 파일 경로: {usd_path}")
print(f"출력 디렉토리: {output_dir}")

# ==========================================
# USD 파일을 먼저 스테이지에 로드 (view_boom.py 방식 참조)
# ==========================================
print(f"\nUSD 파일을 스테이지에 로딩 중: {usd_path}")
omni.usd.get_context().open_stage(usd_path)

# 스테이지가 로드될 때까지 대기
time.sleep(1.0)
print("USD 파일 스테이지 로드 완료!")

# ==========================================
# 붐의 바운딩 박스 계산 및 Semantics 추가
# ==========================================
stage = omni.usd.get_context().get_stage()
if stage:
    print("\n붐의 바운딩 박스 계산 중...")
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), 
        includedPurposes=[UsdGeom.Tokens.default_]
    )
    
    # 루트 프림의 바운딩 박스 계산
    root_prim = stage.GetPseudoRoot()
    bbox = bbox_cache.ComputeWorldBound(root_prim)
    bbox_range = bbox.GetRange()
    
    if bbox_range.GetSize().GetLength() > 0.001:
        size = bbox_range.GetSize()
        min_point = bbox_range.GetMin()
        max_point = bbox_range.GetMax()
        center = (min_point + max_point) / 2.0
        
        boom_center = (center[0], center[1], center[2])
        boom_size = max(size[0], size[1], size[2])  # 가장 긴 축
        
        print(f"붐 바운딩 박스:")
        print(f"  최소점: ({min_point[0]:.3f}, {min_point[1]:.3f}, {min_point[2]:.3f})")
        print(f"  최대점: ({max_point[0]:.3f}, {max_point[1]:.3f}, {max_point[2]:.3f})")
        print(f"  중심점: ({boom_center[0]:.3f}, {boom_center[1]:.3f}, {boom_center[2]:.3f})")
        print(f"  크기: ({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f})")
        print(f"  주요 길이: {boom_size:.3f}")
        
        # 모든 프림에 semantics 추가 (바운딩 박스 annotator가 인식하도록)
        print("\nSemantics 추가 중...")
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Imageable):
                # Semantics API를 사용하여 semantics 추가
                sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
                sem.CreateSemanticTypeAttr("class")
                sem.CreateSemanticDataAttr().Set("excavator_boom")
        print("Semantics 추가 완료!")
        
        # 변수를 전역으로 저장 (rep.new_layer() 블록에서 사용하기 위해)
        boom_size_value = boom_size
        boom_center_value = boom_center
        size_value = (size[0], size[1], size[2])  # 튜플로 변환
    else:
        print("⚠️  경고: 바운딩 박스를 계산할 수 없습니다. 기본값 사용.")
        boom_center_value = (3.4, 0.0, 0.2)
        boom_size_value = 6.09
        size_value = (boom_size_value, boom_size_value * 0.1, boom_size_value * 0.2)
else:
    print("⚠️  경고: 스테이지를 가져올 수 없습니다. 기본값 사용.")
    boom_center_value = (3.4, 0.0, 0.2)
    boom_size_value = 6.09
    size_value = (boom_size_value, boom_size_value * 0.1, boom_size_value * 0.2)

# ==========================================
# Replicator 설정
# ==========================================
with rep.new_layer():
    
    # 조명 추가
    light = rep.create.light(
        light_type="Dome",
        intensity=1000.0,
        rotation=(270, 0, 0)
    )
    
    # Boom 객체는 이미 스테이지에 로드되어 있으므로 별도로 로드하지 않음
    print("Boom 객체는 스테이지에 이미 로드되어 있습니다.")
    
    # 카메라 생성 (붐 전체가 보이도록 설정)
    # 붐 전체가 화면에 들어오도록 카메라 거리 계산
    # FOV 고려: 일반적으로 카메라 FOV는 60도 정도
    # 붐이 화면의 70-80%를 차지하도록 설정 (여유 공간 확보)
    print("카메라 생성 중...")
    
    # 붐의 대각선 길이 계산 (3D 바운딩 박스의 대각선)
    boom_diagonal = np.sqrt(size_value[0]**2 + size_value[1]**2 + size_value[2]**2)
    
    # 카메라 FOV (라디안) - 기본값 60도
    camera_fov_rad = np.radians(60)
    
    # 붐 전체가 보이도록 하는 최소 거리 계산
    # distance = (object_size / 2) / tan(FOV / 2)
    # 화면의 80%를 차지하도록 하려면 0.8을 곱함
    min_camera_distance = (boom_diagonal / 2.0) / np.tan(camera_fov_rad / 2.0) / 0.8
    
    # 적절한 거리 범위 설정 (붐 전체가 보이면서도 충분히 크게 보이도록)
    camera_distance_min = min_camera_distance * 0.9  # 최소 거리 (약간 가까이)
    camera_distance_max = min_camera_distance * 1.5  # 최대 거리 (약간 멀리)
    
    print(f"붐 대각선 길이: {boom_diagonal:.3f}")
    print(f"카메라 거리 범위: {camera_distance_min:.3f} ~ {camera_distance_max:.3f}")
    
    # 초기 카메라 위치: 붐 중심에서 적절한 거리
    initial_camera_distance = (camera_distance_min + camera_distance_max) / 2.0
    camera = rep.create.camera(
        position=(boom_center_value[0] + initial_camera_distance * 0.7, 
                  boom_center_value[1] + initial_camera_distance * 0.5, 
                  boom_center_value[2] + initial_camera_distance * 0.5),
        look_at=boom_center_value
    )
    print(f"카메라 생성 완료! 붐 중심 {boom_center_value}을 바라봄 (거리: {initial_camera_distance:.3f})")
    
    # Render product 생성
    render_product = rep.create.render_product(camera, resolution=(1024, 1024))
    print(f"Render product 생성 완료! 해상도: 1024x1024")
    
    # 카메라 랜덤화 (20프레임)
    # 붐 전체가 보이도록 하면서 다양한 각도에서 촬영
    with rep.trigger.on_frame(max_execs=20):
        with rep.create.group([camera]):
            # 붐 중심을 기준으로 구면 좌표계에서 랜덤 위치 생성
            # 거리: 계산된 최소~최대 거리 범위
            # 각도: 수평 -45~45도, 수직 15~75도 (붐 전체가 보이도록)
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (boom_center_value[0] - camera_distance_max * 0.7, 
                     boom_center_value[1] - camera_distance_max * 0.5, 
                     boom_center_value[2] + camera_distance_min * 0.3),
                    (boom_center_value[0] + camera_distance_max * 0.7, 
                     boom_center_value[1] + camera_distance_max * 0.5, 
                     boom_center_value[2] + camera_distance_max * 0.9)
                ),
                look_at=boom_center_value
            )
    
    # 바운딩 박스 annotator 추가
    bbox_annotator = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
    bbox_annotator.attach([render_product])
    
    # Writer 설정
    print(f"\nWriter 설정 중... 출력 디렉토리: {output_dir}")
    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(
        output_dir=output_dir,
        rgb=True,
        bounding_box_2d_tight=True
    )
    writer.attach([render_product])
    print("Writer 설정 완료!")
    
    # 데이터 생성 시작
    print("\n데이터 생성 시작...")
    print(f"20개 프레임 생성 예정...")
    rep.orchestrator.run()
    print("데이터 생성 프로세스 완료!")

# ==========================================
# 완료 및 종료
# ==========================================
print("\n" + "="*50)
print("데이터 생성 완료!")
print("="*50)

# 생성된 파일 확인
import glob
png_files = glob.glob(os.path.join(output_dir, "rgb_*.png"))
npy_files = glob.glob(os.path.join(output_dir, "bounding_box_*.npy"))
json_files = glob.glob(os.path.join(output_dir, "bounding_box_*.json"))

print(f"\n생성된 파일 통계:")
print(f"  - PNG 이미지: {len(png_files)}개")
print(f"  - NPY 바운딩박스: {len(npy_files)}개")
print(f"  - JSON 메타데이터: {len(json_files)}개")
print(f"\n파일 위치: {output_dir}")

if len(png_files) == 0:
    print("\n⚠️  경고: PNG 파일이 생성되지 않았습니다!")
    print("   카메라나 렌더링 설정을 확인하세요.")

print("\n시뮬레이션을 계속 실행합니다. 종료하려면 Ctrl+C를 누르세요.")

# 시뮬레이션 유지
while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
