# ==========================================
# 붐 크기 및 카메라 거리 분석 스크립트
# ==========================================
from isaacsim import SimulationApp

# 시뮬레이터 초기화 (headless 모드로 변경하여 출력만 확인)
simulation_app = SimulationApp({"headless": True})

# ==========================================
# 모듈 임포트 (App 실행 후)
# ==========================================
import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom
import os
import time
import numpy as np

# ==========================================
# 설정
# ==========================================
usd_path = "/home/rebirther/isaac-sim/assets/boom_link.usd"
output_file = "/home/rebirther/isaac_data_output/boom_analysis_result.txt"

# 출력을 파일로도 저장
import sys
class TeeOutput:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

sys.stdout = TeeOutput(sys.stdout, open(output_file, 'w'))

if not os.path.exists(usd_path):
    print(f"ERROR: USD 파일을 찾을 수 없습니다: {usd_path}")
    simulation_app.close()
    exit(1)

print(f"USD 파일 경로: {usd_path}")

# ==========================================
# USD 파일 로드 및 붐 크기 분석
# ==========================================
print(f"\nUSD 파일을 스테이지에 로딩 중: {usd_path}")
omni.usd.get_context().open_stage(usd_path)

# 스테이지가 로드될 때까지 대기
time.sleep(2.0)
print("USD 파일 스테이지 로드 완료!")

# 스테이지 가져오기
stage = omni.usd.get_context().get_stage()

if stage:
    print("\n" + "="*60)
    print("붐 크기 분석")
    print("="*60)
    
    # 바운딩 박스 캐시 생성
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
        
        print(f"\n전체 바운딩 박스:")
        print(f"  최소점 (Min): ({min_point[0]:.3f}, {min_point[1]:.3f}, {min_point[2]:.3f})")
        print(f"  최대점 (Max): ({max_point[0]:.3f}, {max_point[1]:.3f}, {max_point[2]:.3f})")
        print(f"  중심점 (Center): ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        print(f"\n붐 크기:")
        print(f"  X축 길이: {size[0]:.3f} 단위")
        print(f"  Y축 길이: {size[1]:.3f} 단위")
        print(f"  Z축 길이: {size[2]:.3f} 단위")
        print(f"  대각선 길이: {size.GetLength():.3f} 단위")
        
        # 가장 긴 축 찾기 (붐의 주요 길이)
        max_dimension = max(size[0], size[1], size[2])
        print(f"\n붐의 대략적인 길이: {max_dimension:.3f} 단위")
        
        # 각 프림별로도 확인
        print(f"\n개별 프림 분석:")
        print("-" * 60)
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Imageable):
                prim_bbox = bbox_cache.ComputeWorldBound(prim)
                prim_range = prim_bbox.GetRange()
                prim_size = prim_range.GetSize()
                
                if prim_size.GetLength() > 0.001:
                    prim_min = prim_range.GetMin()
                    prim_max = prim_range.GetMax()
                    print(f"  {prim.GetPath()}:")
                    print(f"    크기: ({prim_size[0]:.3f}, {prim_size[1]:.3f}, {prim_size[2]:.3f})")
                    print(f"    길이: {prim_size.GetLength():.3f}")
                    print(f"    최소점: ({prim_min[0]:.3f}, {prim_min[1]:.3f}, {prim_min[2]:.3f})")
                    print(f"    최대점: ({prim_max[0]:.3f}, {prim_max[1]:.3f}, {prim_max[2]:.3f})")
    else:
        print("⚠️  바운딩 박스를 계산할 수 없습니다.")

# ==========================================
# 현재 카메라 설정 분석
# ==========================================
print("\n" + "="*60)
print("현재 카메라 설정 분석")
print("="*60)

# 현재 코드의 카메라 설정
camera_initial_pos = (3, 3, 3)
camera_min_pos = (2.0, 2.0, 2.0)
camera_max_pos = (4.0, 4.0, 4.0)
look_at = (0, 0, 0)

print(f"\n카메라 초기 위치: {camera_initial_pos}")
print(f"카메라 랜덤 범위: {camera_min_pos} ~ {camera_max_pos}")
print(f"카메라가 바라보는 지점: {look_at}")

# 카메라와 원점 사이의 거리 계산
camera_distance = np.sqrt(sum(x**2 for x in camera_initial_pos))
camera_min_distance = np.sqrt(sum(x**2 for x in camera_min_pos))
camera_max_distance = np.sqrt(sum(x**2 for x in camera_max_pos))

print(f"\n카메라 거리 (원점 기준):")
print(f"  초기 거리: {camera_distance:.3f} 단위")
print(f"  최소 거리: {camera_min_distance:.3f} 단위")
print(f"  최대 거리: {camera_max_distance:.3f} 단위")

# 붐 크기가 계산되었다면 비율 계산
if stage and bbox_range.GetSize().GetLength() > 0.001:
    boom_size = bbox_range.GetSize().GetLength()
    print(f"\n붐 크기 vs 카메라 거리 비율:")
    print(f"  붐 크기: {boom_size:.3f} 단위")
    print(f"  카메라 거리: {camera_distance:.3f} 단위")
    print(f"  비율 (카메라/붐): {camera_distance/boom_size:.2f}:1")
    
    # 권장 카메라 거리 계산 (붐이 화면의 30-50%를 차지하도록)
    # 일반적으로 화면의 30-50%를 차지하려면 카메라 거리는 객체 크기의 2-3배 정도
    recommended_distance_min = boom_size * 2.0
    recommended_distance_max = boom_size * 3.0
    
    print(f"\n권장 카메라 거리 (붐이 화면의 30-50% 차지):")
    print(f"  최소 거리: {recommended_distance_min:.3f} 단위")
    print(f"  최대 거리: {recommended_distance_max:.3f} 단위")

print("\n" + "="*60)
print("분석 완료!")
print("="*60)

# 분석 완료 후 즉시 종료
simulation_app.close()
