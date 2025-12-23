# ==========================================
# 굴착기 부품 데이터셋 생성 스크립트
# (Domain Randomization 적용)
# ==========================================

# 로깅 설정 (SimulationApp 초기화 전에 설정해야 함)
import os
import sys
SCRIPT_DIR = "/home/rebirther/isaac_data_output"
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from utils.logger import setup_logging, reinit_logging, finish_logging

# 로그 파일 생성
LOG_PATH = setup_logging("01_generate")

# Isaac Sim 초기화
from isaacsim import SimulationApp

# 시뮬레이터 초기화
simulation_app = SimulationApp({"headless": False})

# ==========================================
# 모듈 임포트 (App 실행 후)
# ==========================================
import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom, Semantics
import time
import numpy as np
import json
from pathlib import Path

# 로깅 재초기화 (SimulationApp이 stdout을 변경한 후 다시 설정)
reinit_logging(LOG_PATH)

# ==========================================
# 설정: 굴착기 부품 정의 (자동 스캔)
# ==========================================
# assets 폴더의 모든 USD 파일을 자동으로 스캔하여 데이터 생성
ASSETS_DIR = "/home/rebirther/isaac-sim/assets"

def scan_usd_files(assets_dir):
    """
    assets 폴더에서 모든 USD 파일을 스캔하여 설정 딕셔너리 생성
    class_name과 display_name은 파일 이름으로 통일
    """
    import glob
    configs = {}
    
    usd_files = glob.glob(os.path.join(assets_dir, "*.usd"))
    usd_files.sort()  # 알파벳 순 정렬
    
    for usd_path in usd_files:
        # 파일 이름에서 확장자 제거
        file_name = os.path.basename(usd_path)
        name_without_ext = os.path.splitext(file_name)[0]
        
        configs[name_without_ext] = {
            "usd_path": usd_path,
            "class_name": name_without_ext,
            "display_name": name_without_ext
        }
    
    return configs

# USD 파일 자동 스캔
EXCAVATOR_PARTS_CONFIG = scan_usd_files(ASSETS_DIR)

# 스캔 결과 출력
print(f"📂 Assets 폴더 스캔 완료: {ASSETS_DIR}")
print(f"   발견된 USD 파일: {len(EXCAVATOR_PARTS_CONFIG)}개")
for name, config in EXCAVATOR_PARTS_CONFIG.items():
    print(f"   - {name}: {config['usd_path']}")

# 데이터셋 설정
BASE_OUTPUT_DIR = "/home/rebirther/isaac_data_output/datasets"
IMAGES_PER_CLASS = 500  # 각 클래스당 생성할 이미지 수
TOTAL_FRAMES = IMAGES_PER_CLASS * len(EXCAVATOR_PARTS_CONFIG)
CLEAR_EXISTING_DATA = True  # True: 기존 데이터셋 폴더 삭제 후 새로 생성

# 기존 데이터셋 폴더 삭제 (CLEAR_EXISTING_DATA가 True일 때)
if CLEAR_EXISTING_DATA and os.path.exists(BASE_OUTPUT_DIR):
    import shutil
    print(f"\n⚠️  기존 데이터셋 폴더 삭제 중: {BASE_OUTPUT_DIR}")
    try:
        shutil.rmtree(BASE_OUTPUT_DIR)
        print(f"✓ 기존 데이터셋 폴더 삭제 완료")
    except Exception as e:
        print(f"⚠️  폴더 삭제 실패: {e}")

# 배경 설정 (중요: 배경 유무가 학습에 큰 영향을 줍니다)
# "none": 배경 없음 (검은 배경)
# "solid": 단색 배경 (랜덤 색상)
# "factory": 공장 환경 배경 (권장)
# "random": 랜덤 배경 (Domain Randomization, 최고 권장)
BACKGROUND_MODE = "random"  # "none", "solid", "factory", "random"

# 배경 비율 (BACKGROUND_MODE가 "random"일 때만 사용)
BACKGROUND_RATIOS = {
    "none": 0.2,      # 20%: 배경 없음
    "solid": 0.3,     # 30%: 단색 배경
    "factory": 0.5    # 50%: 공장 환경 배경
}

print("="*60)
print("굴착기 부품 데이터셋 생성 스크립트")
print("="*60)
print(f"생성할 클래스: {len(EXCAVATOR_PARTS_CONFIG)}개")
print(f"클래스당 이미지: {IMAGES_PER_CLASS}장")
print(f"총 이미지 수: {TOTAL_FRAMES}장")
print(f"배경 모드: {BACKGROUND_MODE}")
if BACKGROUND_MODE == "random":
    print(f"  - 배경 없음: {BACKGROUND_RATIOS['none']*100:.0f}%")
    print(f"  - 단색 배경: {BACKGROUND_RATIOS['solid']*100:.0f}%")
    print(f"  - 공장 배경: {BACKGROUND_RATIOS['factory']*100:.0f}%")
print("="*60)
print("\n💡 참고: 배경의 유무가 학습에 큰 영향을 줍니다.")
print("   자세한 내용은 background_impact_analysis.md 문서를 참조하세요.")
print("="*60)

# ==========================================
# 각 클래스별 데이터 생성 함수
# ==========================================
# 중요: 한 번에 하나의 부품만 로드하여 데이터 생성
# 여러 종류를 동시에 띄우는 것이 아니라 순차적으로 처리
def generate_class_dataset(part_config, class_index, start_frame):
    """
    특정 클래스의 데이터셋 생성
    
    중요: 한 번에 하나의 부품만 로드하여 데이터를 생성합니다.
    여러 종류를 동시에 띄우는 것이 아니라, 각 클래스를 순차적으로 처리합니다.
    
    Args:
        part_config: 부품 설정 딕셔너리
        class_index: 클래스 인덱스
        start_frame: 시작 프레임 번호
    """
    usd_path = part_config["usd_path"]
    class_name = part_config["class_name"]
    display_name = part_config["display_name"]
    
    # 클래스별 출력 디렉토리 생성
    class_output_dir = os.path.join(BASE_OUTPUT_DIR, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    print(f"\n[{class_index+1}/{len(EXCAVATOR_PARTS_CONFIG)}] {display_name} 데이터 생성 시작...")
    print(f"  USD 파일: {usd_path}")
    print(f"  출력 디렉토리: {class_output_dir}")
    
    # 이전 클래스 처리 후 남은 상태 정리 (연속 처리 시 중요)
    if class_index > 0:
        print(f"  이전 클래스 상태 정리 중...")
        try:
            # 시뮬레이션 업데이트로 정리
            for _ in range(10):
                simulation_app.update()
                time.sleep(0.05)
        except Exception as e:
            print(f"  ⚠️  상태 정리 중 에러 (무시하고 계속): {e}")
    
    # USD 파일 존재 확인
    if not os.path.exists(usd_path):
        print(f"  ⚠️  경고: USD 파일을 찾을 수 없습니다: {usd_path}")
        print(f"  이 클래스는 건너뜁니다.")
        return
    
    # USD 파일 로드 (한 번에 하나의 부품만 로드)
    # 중요: open_stage()는 이전 스테이지를 닫고 새로운 스테이지를 엽니다.
    # 따라서 여러 부품이 동시에 존재하지 않습니다.
    print(f"  USD 파일 로딩 중... (이전 스테이지는 자동으로 닫힙니다)")
    omni.usd.get_context().open_stage(usd_path)
    time.sleep(1.0)
    
    # 부품의 바운딩 박스 계산
    stage = omni.usd.get_context().get_stage()
    if not stage:
        print(f"  ⚠️  경고: 스테이지를 가져올 수 없습니다.")
        return
    
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), 
        includedPurposes=[UsdGeom.Tokens.default_]
    )
    
    # 메시 프리미티브만 대상으로 바운딩 박스 계산 (헬퍼 오브젝트 제외)
    mesh_prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    if not mesh_prims:
        print(f"  ⚠️  경고: 메시를 찾을 수 없습니다.")
        return
    
    print(f"  메시 개수: {len(mesh_prims)}개")
    
    # 각 메시의 바운딩 박스를 합침
    from pxr import Gf
    combined_range = None
    for mesh_prim in mesh_prims:
        mesh_bbox = bbox_cache.ComputeWorldBound(mesh_prim)
        mesh_range = mesh_bbox.GetRange()
        if combined_range is None:
            combined_range = Gf.Range3d(mesh_range.GetMin(), mesh_range.GetMax())
        else:
            combined_range.UnionWith(mesh_range)
    
    bbox_range = combined_range
    
    if bbox_range.GetSize().GetLength() <= 0.001:
        print(f"  ⚠️  경고: 바운딩 박스를 계산할 수 없습니다.")
        return
    
    size = bbox_range.GetSize()
    min_point = bbox_range.GetMin()
    max_point = bbox_range.GetMax()
    center = (min_point + max_point) / 2.0
    
    part_center = (center[0], center[1], center[2])
    part_size = max(size[0], size[1], size[2])
    size_value = (size[0], size[1], size[2])
    
    # 스케일 경고 (부품 크기가 100 이상이면 mm 단위로 판단)
    if part_size > 100:
        print(f"  ⚠️  경고: 부품 크기가 {part_size:.1f}로 큼 (mm 단위로 추정)")
        print(f"     USD 파일의 스케일을 확인하세요. (권장: metersPerUnit=0.001)")
        print(f"     데이터 생성은 계속 진행됩니다.")
    
    print(f"  부품 중심: {part_center}")
    print(f"  부품 크기: {part_size:.3f}")
    print(f"  부품 바닥 Z: {min_point[2]:.3f}")
    
    # 기존 메시에 Semantics 추가 (Replicator가 인식할 수 있도록)
    print(f"  Semantics 추가 중...")
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Imageable):
            sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
            sem.CreateSemanticTypeAttr("class")
            sem.CreateSemanticDataAttr().Set(class_name)
    print(f"  ✓ Semantics 추가 완료")
    
    # Replicator 설정 (기존 스테이지 유지, new_stage() 호출하지 않음)
    print(f"  Replicator 레이어 생성 중...")
    try:
        with rep.new_layer():
            # 기존 스테이지에서 메시 프리미티브 참조
            # rep.create.from_usd()를 사용하지 않고 rep.get.prims()로 기존 객체 참조
            print(f"  기존 스테이지에서 메시 참조 중...")
            
            # 스테이지의 모든 메시 가져오기
            mesh_prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
            print(f"    스테이지 메시 개수: {len(mesh_prims)}")
            
            if len(mesh_prims) == 0:
                print(f"  ⚠️  경고: 스테이지에 메시가 없습니다!")
                print(f"  USD 파일 구조를 확인하세요.")
                return
            
            # rep.get.prims()로 기존 메시 참조
            part_prim = rep.get.prims(semantics=[("class", class_name)])
            print(f"  ✓ 기존 메시를 Replicator에서 참조합니다.")
            print(f"    반환 타입: {type(part_prim)}")
            
            # ==========================================
            # Domain Randomization: 배경 설정
            # ==========================================
            # Sim-to-Real 성능 향상을 위해 다양한 배경 적용
            print(f"  🎨 Domain Randomization: 배경 설정 중...")
            
            # 바닥 평면 생성 (부품의 min_z와 동일한 높이에 배치)
            floor_size = part_size * 5  # 부품 크기의 5배
            floor_z = min_point[2]  # 부품 바닥과 정확히 동일한 높이
            floor_plane = rep.create.plane(
                scale=(floor_size, floor_size, 1),
                position=(part_center[0], part_center[1], floor_z),
                rotation=(0, 0, 0),
                semantics=[("class", "background")]
            )
            print(f"    ✓ 바닥 평면 생성 (크기: {floor_size:.1f}, Z={floor_z:.3f}, 부품 바닥과 동일)")
            
            # 뒷벽 평면 생성 (카메라 반대편)
            back_wall = rep.create.plane(
                scale=(floor_size, floor_size * 0.5, 1),
                position=(part_center[0] - floor_size * 0.4, part_center[1], part_center[2]),
                rotation=(0, 90, 0),
                semantics=[("class", "background")]
            )
            print(f"    ✓ 뒷벽 평면 생성")
            
            # ==========================================
            # Domain Randomization: 조명 설정
            # ==========================================
            print(f"  💡 Domain Randomization: 조명 설정 중...")
            
            # 메인 돔 조명 (환경광)
            dome_light = rep.create.light(
                light_type="Dome",
                intensity=800.0,  # 기본값, on_frame에서 랜덤화
                rotation=(270, 0, 0)
            )
            print(f"    ✓ 돔 조명 생성")
            
            # 추가 포인트 라이트 1 (주 조명)
            point_light1 = rep.create.light(
                light_type="Sphere",
                intensity=50000.0,
                position=(part_center[0] + part_size, part_center[1] + part_size, part_center[2] + part_size * 2),
                scale=0.5
            )
            print(f"    ✓ 포인트 라이트 1 생성 (주 조명)")
            
            # 추가 포인트 라이트 2 (보조 조명 - 그림자 완화)
            point_light2 = rep.create.light(
                light_type="Sphere",
                intensity=30000.0,
                position=(part_center[0] - part_size, part_center[1] - part_size * 0.5, part_center[2] + part_size),
                scale=0.3
            )
            print(f"    ✓ 포인트 라이트 2 생성 (보조 조명)")
            
            print(f"  ✓ Domain Randomization 설정 완료")
            
                # 카메라 거리 계산
            print(f"  카메라 거리 계산 중...")
            part_diagonal = np.sqrt(size_value[0]**2 + size_value[1]**2 + size_value[2]**2)
            camera_fov_rad = np.radians(60)
            min_camera_distance = (part_diagonal / 2.0) / np.tan(camera_fov_rad / 2.0) / 0.8
            camera_distance_min = min_camera_distance * 0.9
            camera_distance_max = min_camera_distance * 1.5
            
            print(f"    부품 대각선: {part_diagonal:.3f}")
            print(f"    카메라 거리 범위: {camera_distance_min:.3f} ~ {camera_distance_max:.3f}")
            
            # 카메라 생성
            print(f"  카메라 생성 중...")
            initial_camera_distance = (camera_distance_min + camera_distance_max) / 2.0
            camera = rep.create.camera(
                position=(part_center[0] + initial_camera_distance * 0.7, 
                          part_center[1] + initial_camera_distance * 0.5, 
                          part_center[2] + initial_camera_distance * 0.5),
                look_at=part_center
            )
            print(f"  ✓ 카메라 생성 완료 (거리: {initial_camera_distance:.3f})")
            
            # Render product 생성
            print(f"  Render product 생성 중...")
            render_product = rep.create.render_product(camera, resolution=(1024, 1024))
            print(f"  ✓ Render product 생성 완료 (해상도: 1024x1024)")
            
            # ==========================================
            # Domain Randomization: 프레임별 랜덤화
            # ==========================================
            print(f"  🎲 프레임별 랜덤화 설정 중...")
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
                
                # 바닥 색상 랜덤화 (공장 바닥 색상: 회색~갈색 계열)
                with floor_plane:
                    rep.randomizer.color(
                        colors=rep.distribution.uniform(
                            (0.2, 0.2, 0.2),  # 어두운 회색
                            (0.6, 0.5, 0.4)   # 밝은 갈색/베이지
                        )
                    )
                
                # 뒷벽 색상 랜덤화 (공장 벽 색상: 흰색~회색 계열)
                with back_wall:
                    rep.randomizer.color(
                        colors=rep.distribution.uniform(
                            (0.4, 0.4, 0.4),  # 회색
                            (0.9, 0.9, 0.85)  # 밝은 베이지
                        )
                    )
                
                # 포인트 라이트 1 위치 랜덤화
                with point_light1:
                    rep.modify.pose(
                        position=rep.distribution.uniform(
                            (part_center[0] + part_size * 0.5, part_center[1] - part_size, part_center[2] + part_size * 1.5),
                            (part_center[0] + part_size * 1.5, part_center[1] + part_size, part_center[2] + part_size * 3)
                        )
                    )
                
                # 포인트 라이트 2 위치 랜덤화
                with point_light2:
                    rep.modify.pose(
                        position=rep.distribution.uniform(
                            (part_center[0] - part_size * 1.5, part_center[1] - part_size, part_center[2] + part_size * 0.5),
                            (part_center[0] - part_size * 0.5, part_center[1] + part_size, part_center[2] + part_size * 2)
                        )
                    )
            
            print(f"  ✓ 프레임별 랜덤화 설정 완료:")
            print(f"    - 카메라 위치 랜덤화")
            print(f"    - 바닥/벽 색상 랜덤화 (공장 환경 시뮬레이션)")
            print(f"    - 조명 위치 랜덤화 (다양한 그림자)")
            print(f"    - 총 {IMAGES_PER_CLASS}장 생성 예정")
            
            # 바운딩 박스 annotator
            print(f"  바운딩 박스 annotator 설정 중...")
            bbox_annotator = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
            bbox_annotator.attach([render_product])
            print(f"  ✓ 바운딩 박스 annotator 설정 완료")
            
            # Writer 설정 (클래스별 디렉토리)
            print(f"  Writer 설정 중...")
            writer = rep.WriterRegistry.get("BasicWriter")
            writer.initialize(
                output_dir=class_output_dir,
                rgb=True,
                bounding_box_2d_tight=True
            )
            writer.attach([render_product])
            print(f"  ✓ Writer 설정 완료 (출력 디렉토리: {class_output_dir})")
            
            # 데이터 생성
            print(f"\n  {'='*50}")
            print(f"  데이터 생성 시작... ({IMAGES_PER_CLASS}장)")
            print(f"  {'='*50}")
            print(f"  ⏳ 이 작업은 시간이 걸릴 수 있습니다. 잠시만 기다려주세요...")
            
            # 시뮬레이션 업데이트 (Replicator가 준비될 때까지)
            print(f"  시뮬레이션 준비 중...")
            for i in range(10):
                simulation_app.update()
                time.sleep(0.1)
                if i % 3 == 0:
                    print(f"    준비 중... ({i+1}/10)")
            
            print(f"  rep.orchestrator.run_until_complete() 실행 중...")
            
            # 실행 전 상태 확인
            import glob
            files_before = glob.glob(os.path.join(class_output_dir, "rgb_*.png"))
            print(f"  실행 전 파일 개수: {len(files_before)}")
            
            # 핵심 수정: run_until_complete()를 사용하여 모든 프레임 완료 대기
            # run()은 비동기로 실행되어 완료 전에 다음 클래스로 넘어감
            rep.orchestrator.run_until_complete()
            print(f"  ✓ rep.orchestrator.run_until_complete() 완료!")
            
            # 실행 후 상태 확인 및 정리 (연속 처리 시 중요)
            print(f"  데이터 생성 완료 후 정리 중...")
            time.sleep(0.5)
            for i in range(10):
                simulation_app.update()
                time.sleep(0.1)
                if i % 3 == 0:
                    print(f"    정리 중... ({i+1}/10)")
            print(f"  ✓ 정리 완료")
            
            generated_files = glob.glob(os.path.join(class_output_dir, "rgb_*.png"))
            print(f"  실행 후 파일 개수: {len(generated_files)}개")
            
            if len(generated_files) > len(files_before):
                new_files = [f for f in generated_files if f not in files_before]
                print(f"  새로 생성된 파일: {len(new_files)}개")
                if len(new_files) > 0:
                    print(f"    첫 번째 파일: {os.path.basename(new_files[0])}")
                    print(f"    마지막 파일: {os.path.basename(new_files[-1])}")
            elif len(generated_files) == 0:
                print(f"  ⚠️  이미지 파일이 생성되지 않았습니다!")
                print(f"  가능한 원인:")
                print(f"    1. 카메라가 부품을 보지 못함")
                print(f"    2. Writer가 제대로 작동하지 않음")
                print(f"    3. 렌더링이 실패함")
                print(f"    4. rep.create.from_usd()가 객체를 제대로 로드하지 못함")
            else:
                print(f"  ✓ {display_name} 데이터 생성 완료! (총 {len(generated_files)}장)")
                if len(generated_files) > 0:
                    print(f"    첫 번째 파일: {os.path.basename(generated_files[0])}")
                    print(f"    마지막 파일: {os.path.basename(generated_files[-1])}")
                
    except Exception as e:
        print(f"  ⚠️  Replicator 설정 중 에러 발생: {e}")
        import traceback
        traceback.print_exc()
        print(f"  이 클래스는 건너뜁니다.")
        return
    finally:
        # Replicator 레이어 정리 (연속 처리 시 중요)
        print(f"  Replicator 레이어 정리 중...")
        try:
            # Writer 정리 - 레이어가 닫히면 자동으로 정리됨
            print(f"    Writer 정리 중...")
            
            # 시뮬레이션 업데이트로 정리 완료 대기
            for i in range(20):
                simulation_app.update()
                time.sleep(0.1)
                if i % 5 == 0:
                    print(f"    정리 중... ({i+1}/20)")
            
            print(f"  ✓ Replicator 레이어 정리 완료")
        except Exception as cleanup_error:
            print(f"  ⚠️  정리 중 에러 발생 (무시하고 계속): {cleanup_error}")
    
    # 메타데이터 파일 생성
    metadata = {
        "class_name": class_name,
        "display_name": display_name,
        "usd_path": usd_path,
        "num_images": IMAGES_PER_CLASS,
        "part_center": list(part_center),
        "part_size": float(part_size),
        "background_mode": BACKGROUND_MODE,
        "background_ratios": BACKGROUND_RATIOS if BACKGROUND_MODE == "random" else None
    }
    
    metadata_path = os.path.join(class_output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"  메타데이터 저장 완료: {metadata_path}")

# ==========================================
# 전체 데이터셋 생성
# ==========================================
# 중요: 각 클래스를 순차적으로 처리합니다.
# 한 번에 하나의 부품만 로드하여 데이터를 생성하고,
# 완료 후 다음 클래스로 이동합니다.
print("\n전체 데이터셋 생성 시작...")
print("방식: 한 번에 하나의 부품만 로드하여 순차적으로 처리\n")

for idx, (key, config) in enumerate(EXCAVATOR_PARTS_CONFIG.items()):
    start_frame = idx * IMAGES_PER_CLASS
    print(f"\n{'='*60}")
    print(f"클래스 {idx+1}/{len(EXCAVATOR_PARTS_CONFIG)} 처리 중...")
    print(f"{'='*60}")
    
    # 한 번에 하나의 부품만 로드하여 데이터 생성
    generate_class_dataset(config, idx, start_frame)
    
    # 다음 클래스를 위해 스테이지 정리 (연속 처리 시 중요)
    # 이전 클래스의 USD 파일을 언로드하고 다음 클래스 준비
    if idx < len(EXCAVATOR_PARTS_CONFIG) - 1:
        print("\n다음 클래스를 위해 스테이지 정리 중...")
        try:
            # Orchestrator 정지 (run_until_complete 완료 후에는 이미 정지 상태)
            print(f"  Orchestrator 정지 확인 중...")
            rep.orchestrator.stop()
            
            # 시뮬레이션 업데이트로 이전 클래스의 리소스 정리
            for i in range(30):
                simulation_app.update()
                time.sleep(0.1)
                if i % 5 == 0:
                    print(f"  정리 중... ({i+1}/30)")
            
            # 추가 대기 시간 (연속 처리 시 안정성을 위해)
            print(f"  추가 대기 중... (연속 처리 안정화)")
            time.sleep(1.0)
            
            print("  ✓ 스테이지 정리 완료")
        except Exception as cleanup_error:
            print(f"  ⚠️  정리 중 에러 발생 (무시하고 계속): {cleanup_error}")
            # 에러가 발생해도 계속 진행

# ==========================================
# 전체 데이터셋 메타데이터 생성
# ==========================================
print("\n" + "="*60)
print("전체 데이터셋 메타데이터 생성 중...")

dataset_metadata = {
    "dataset_name": "Excavator Parts Classification Dataset",
    "num_classes": len(EXCAVATOR_PARTS_CONFIG),
    "images_per_class": IMAGES_PER_CLASS,
    "total_images": TOTAL_FRAMES,
    "classes": {key: config["display_name"] for key, config in EXCAVATOR_PARTS_CONFIG.items()},
    "background_mode": BACKGROUND_MODE,
    "background_ratios": BACKGROUND_RATIOS if BACKGROUND_MODE == "random" else None,
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "note": "배경의 유무가 학습에 큰 영향을 줍니다. background_impact_analysis.md 참조."
}

dataset_metadata_path = os.path.join(BASE_OUTPUT_DIR, "dataset_info.json")
with open(dataset_metadata_path, 'w', encoding='utf-8') as f:
    json.dump(dataset_metadata, f, indent=2, ensure_ascii=False)

print(f"전체 메타데이터 저장 완료: {dataset_metadata_path}")

# ==========================================
# 완료
# ==========================================
print("\n" + "="*60)
print("데이터셋 생성 완료!")
print("="*60)

# 생성된 파일 통계
import glob
for key, config in EXCAVATOR_PARTS_CONFIG.items():
    class_dir = os.path.join(BASE_OUTPUT_DIR, config["class_name"])
    if os.path.exists(class_dir):
        # glob을 사용하여 파일 개수 확인
        png_files = glob.glob(os.path.join(class_dir, "rgb_*.png"))
        png_count = len(png_files)
        print(f"{config['display_name']}: {png_count}장")
        if png_count == 0:
            print(f"  ⚠️  경고: {config['display_name']} 이미지가 생성되지 않았습니다.")

# 로깅 종료
finish_logging()

print("\n시뮬레이션을 계속 실행합니다. 종료하려면 Ctrl+C를 누르세요.")

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
