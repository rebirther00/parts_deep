# ==========================================
# 여러 종류의 붐을 구분하는 데이터셋 생성 스크립트
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
import json
from pathlib import Path

# ==========================================
# 설정: 여러 종류의 붐 정의
# ==========================================
# 각 붐의 USD 파일 경로와 클래스명 정의
BOOM_CONFIGS = {
    "boom_25ton": {
        "usd_path": "/home/rebirther/isaac-sim/assets/boom_link_25.usd",
        "class_name": "boom_25ton",
        "display_name": "붐 25톤급"
    },
    "arm_25ton": {
        "usd_path": "/home/rebirther/isaac-sim/assets/arm_link_25.usd",  # 예시 경로
        "class_name": "arm_25ton",
        "display_name": "암 25톤급"
    },
    # "arm_15ton": {
    #     "usd_path": "/home/rebirther/isaac-sim/assets/arm_15.usd",  # 예시 경로
    #     "class_name": "arm_15ton",
    #     "display_name": "암 15톤급"
    # },
    # "kevin_14ton": {
    #     "usd_path": "/home/rebirther/isaac-sim/assets/kevin_14.usd",  # 예시 경로
    #     "class_name": "kevin_14ton",
    #     "display_name": "케빈 14톤급"
    # }
}

# 데이터셋 설정
BASE_OUTPUT_DIR = "/home/rebirther/isaac_data_output/datasets"
IMAGES_PER_CLASS = 500  # 각 클래스당 생성할 이미지 수
TOTAL_FRAMES = IMAGES_PER_CLASS * len(BOOM_CONFIGS)

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
print("다중 클래스 붐 데이터셋 생성 스크립트")
print("="*60)
print(f"생성할 클래스: {len(BOOM_CONFIGS)}개")
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
# 중요: 한 번에 하나의 붐만 로드하여 데이터 생성
# 여러 종류를 동시에 띄우는 것이 아니라 순차적으로 처리
def generate_class_dataset(boom_config, class_index, start_frame):
    """
    특정 클래스의 데이터셋 생성
    
    중요: 한 번에 하나의 붐만 로드하여 데이터를 생성합니다.
    여러 종류를 동시에 띄우는 것이 아니라, 각 클래스를 순차적으로 처리합니다.
    
    Args:
        boom_config: 붐 설정 딕셔너리
        class_index: 클래스 인덱스
        start_frame: 시작 프레임 번호
    """
    usd_path = boom_config["usd_path"]
    class_name = boom_config["class_name"]
    display_name = boom_config["display_name"]
    
    # 클래스별 출력 디렉토리 생성
    class_output_dir = os.path.join(BASE_OUTPUT_DIR, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
    
    print(f"\n[{class_index+1}/{len(BOOM_CONFIGS)}] {display_name} 데이터 생성 시작...")
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
    
    # USD 파일 로드 (한 번에 하나의 붐만 로드)
    # 중요: open_stage()는 이전 스테이지를 닫고 새로운 스테이지를 엽니다.
    # 따라서 여러 붐이 동시에 존재하지 않습니다.
    print(f"  USD 파일 로딩 중... (이전 스테이지는 자동으로 닫힙니다)")
    omni.usd.get_context().open_stage(usd_path)
    time.sleep(1.0)
    
    # 붐의 바운딩 박스 계산
    stage = omni.usd.get_context().get_stage()
    if not stage:
        print(f"  ⚠️  경고: 스테이지를 가져올 수 없습니다.")
        return
    
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), 
        includedPurposes=[UsdGeom.Tokens.default_]
    )
    
    root_prim = stage.GetPseudoRoot()
    bbox = bbox_cache.ComputeWorldBound(root_prim)
    bbox_range = bbox.GetRange()
    
    if bbox_range.GetSize().GetLength() <= 0.001:
        print(f"  ⚠️  경고: 바운딩 박스를 계산할 수 없습니다.")
        return
    
    size = bbox_range.GetSize()
    min_point = bbox_range.GetMin()
    max_point = bbox_range.GetMax()
    center = (min_point + max_point) / 2.0
    
    boom_center = (center[0], center[1], center[2])
    boom_size = max(size[0], size[1], size[2])
    size_value = (size[0], size[1], size[2])
    
    print(f"  붐 중심: {boom_center}")
    print(f"  붐 크기: {boom_size:.3f}")
    
    # Semantics 추가
    print(f"  Semantics 추가 중...")
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Imageable):
            sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
            sem.CreateSemanticTypeAttr("class")
            sem.CreateSemanticDataAttr().Set(class_name)
    
    # Replicator 설정
    print(f"  Replicator 레이어 생성 중...")
    try:
        with rep.new_layer():
            # 중요: rep.new_layer() 안에서는 USD 파일을 rep.create.from_usd()로 다시 로드해야 함
            # 스테이지에 이미 로드된 객체는 Replicator 레이어에 자동으로 포함되지 않음
            # 
            # 문제: rep.create.from_usd()가 일부 USD 파일에서 제대로 작동하지 않을 수 있음
            # 해결: 반환값을 확인하고, 실패하면 에러를 출력
            
            print(f"  Replicator 레이어에 USD 파일 로드 중...")
            print(f"    USD 경로: {usd_path}")
            
            # 스테이지 확인 (로드 전)
            stage_before = omni.usd.get_context().get_stage()
            mesh_count_before = len([p for p in stage_before.Traverse() if p.IsA(UsdGeom.Mesh)]) if stage_before else 0
            print(f"    로드 전 스테이지 메시 개수: {mesh_count_before}")
            
            # USD 파일을 Replicator 레이어에 로드
            boom_prim = rep.create.from_usd(
                usd_path,
                semantics=[("class", class_name)]
            )
            
            # 로드 후 확인
            stage_after = omni.usd.get_context().get_stage()
            mesh_count_after = len([p for p in stage_after.Traverse() if p.IsA(UsdGeom.Mesh)]) if stage_after else 0
            print(f"    로드 후 스테이지 메시 개수: {mesh_count_after}")
            
            if boom_prim is None:
                print(f"  ⚠️  경고: rep.create.from_usd()가 None을 반환했습니다!")
                print(f"  이는 USD 파일이 제대로 로드되지 않았음을 의미합니다.")
            else:
                print(f"  ✓ USD 파일이 Replicator 레이어에 로드되었습니다.")
                print(f"    반환 타입: {type(boom_prim)}")
            
            if mesh_count_after == 0:
                print(f"  ⚠️  경고: 스테이지에 메시가 없습니다!")
                print(f"  USD 파일 구조를 확인하세요.")
            
            # 배경 설정 (배경 유무가 학습에 큰 영향을 줍니다)
            # 참고: background_impact_analysis.md 문서 참조
            # 주의: displayColor 설정이 에러를 발생시킬 수 있으므로, 일단 배경 없음으로 설정
            # 배경이 필요하면 나중에 추가 가능
            print(f"  배경 설정: 없음 (검은 배경) - displayColor 문제로 인해 일시적으로 비활성화")
            
            # 조명 추가 (고정 intensity - 타입 미스매치 방지)
            print(f"  조명 추가 중...")
            light = rep.create.light(
                light_type="Dome",
                intensity=1000.0,  # 고정값 사용 (랜덤화 시 타입 에러 발생)
                rotation=(270, 0, 0)
            )
            print(f"  ✓ 조명 추가 완료")
            
                # 카메라 거리 계산
            print(f"  카메라 거리 계산 중...")
            boom_diagonal = np.sqrt(size_value[0]**2 + size_value[1]**2 + size_value[2]**2)
            camera_fov_rad = np.radians(60)
            min_camera_distance = (boom_diagonal / 2.0) / np.tan(camera_fov_rad / 2.0) / 0.8
            camera_distance_min = min_camera_distance * 0.9
            camera_distance_max = min_camera_distance * 1.5
            
            print(f"    붐 대각선: {boom_diagonal:.3f}")
            print(f"    카메라 거리 범위: {camera_distance_min:.3f} ~ {camera_distance_max:.3f}")
            
            # 카메라 생성
            print(f"  카메라 생성 중...")
            initial_camera_distance = (camera_distance_min + camera_distance_max) / 2.0
            camera = rep.create.camera(
                position=(boom_center[0] + initial_camera_distance * 0.7, 
                          boom_center[1] + initial_camera_distance * 0.5, 
                          boom_center[2] + initial_camera_distance * 0.5),
                look_at=boom_center
            )
            print(f"  ✓ 카메라 생성 완료 (거리: {initial_camera_distance:.3f})")
            
            # Render product 생성
            print(f"  Render product 생성 중...")
            render_product = rep.create.render_product(camera, resolution=(1024, 1024))
            print(f"  ✓ Render product 생성 완료 (해상도: 1024x1024)")
            
            # 카메라 랜덤화
            print(f"  카메라 랜덤화 설정 중...")
            with rep.trigger.on_frame(max_execs=IMAGES_PER_CLASS):
                with rep.create.group([camera]):
                    rep.modify.pose(
                        position=rep.distribution.uniform(
                            (boom_center[0] - camera_distance_max * 0.7, 
                             boom_center[1] - camera_distance_max * 0.5, 
                             boom_center[2] + camera_distance_min * 0.3),
                            (boom_center[0] + camera_distance_max * 0.7, 
                             boom_center[1] + camera_distance_max * 0.5, 
                             boom_center[2] + camera_distance_max * 0.9)
                        ),
                        look_at=boom_center
                    )
            print(f"  ✓ 카메라 랜덤화 설정 완료 ({IMAGES_PER_CLASS}장)")
            
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
                print(f"    1. 카메라가 붐을 보지 못함")
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
        "boom_center": list(boom_center),
        "boom_size": float(boom_size),
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
# 한 번에 하나의 붐만 로드하여 데이터를 생성하고,
# 완료 후 다음 클래스로 이동합니다.
print("\n전체 데이터셋 생성 시작...")
print("방식: 한 번에 하나의 붐만 로드하여 순차적으로 처리\n")

for idx, (key, config) in enumerate(BOOM_CONFIGS.items()):
    start_frame = idx * IMAGES_PER_CLASS
    print(f"\n{'='*60}")
    print(f"클래스 {idx+1}/{len(BOOM_CONFIGS)} 처리 중...")
    print(f"{'='*60}")
    
    # 한 번에 하나의 붐만 로드하여 데이터 생성
    generate_class_dataset(config, idx, start_frame)
    
    # 다음 클래스를 위해 스테이지 정리 (연속 처리 시 중요)
    # 이전 클래스의 USD 파일을 언로드하고 다음 클래스 준비
    if idx < len(BOOM_CONFIGS) - 1:
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
    "dataset_name": "Boom Classification Dataset",
    "num_classes": len(BOOM_CONFIGS),
    "images_per_class": IMAGES_PER_CLASS,
    "total_images": TOTAL_FRAMES,
    "classes": {key: config["display_name"] for key, config in BOOM_CONFIGS.items()},
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
for key, config in BOOM_CONFIGS.items():
    class_dir = os.path.join(BASE_OUTPUT_DIR, config["class_name"])
    if os.path.exists(class_dir):
        # glob을 사용하여 파일 개수 확인
        png_files = glob.glob(os.path.join(class_dir, "rgb_*.png"))
        png_count = len(png_files)
        print(f"{config['display_name']}: {png_count}장")
        if png_count == 0:
            print(f"  ⚠️  경고: {config['display_name']} 이미지가 생성되지 않았습니다.")

print("\n시뮬레이션을 계속 실행합니다. 종료하려면 Ctrl+C를 누르세요.")

while simulation_app.is_running():
    simulation_app.update()

simulation_app.close()
