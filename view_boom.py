import os
# 1. 시뮬레이터 실행 (가장 먼저 와야 함)
from isaacsim import SimulationApp

# headless=False: 화면을 띄우겠다는 뜻
simulation_app = SimulationApp({"headless": False})

import omni.usd
import omni.replicator.core as rep
import omni.kit.viewport.utility as viewport_utils
from pxr import Usd, UsdUtils
import time

# ==========================================
# [수정] 본인의 파일 경로를 정확히 입력하세요
# ==========================================
usd_path = "/home/rebirther/isaac-sim/assets/boom_link.usd"  

def check_missing_references(stage):
    """USD 스테이지에서 누락된 참조를 확인하고 출력"""
    missing_refs = []
    
    # 모든 프림을 순회하며 참조 확인
    for prim in stage.Traverse():
        # References 확인
        try:
            if prim.HasAuthoredReferences():
                refs = prim.GetReferences()
                for ref in refs.GetAddedOrExplicitItems():
                    asset_path = ref.assetPath if hasattr(ref, 'assetPath') else str(ref)
                    # 상대 경로나 절대 경로 모두 확인
                    if asset_path:
                        # 절대 경로가 아니면 USD 파일 기준 상대 경로로 변환 시도
                        if not os.path.isabs(asset_path):
                            usd_dir = os.path.dirname(usd_path)
                            full_path = os.path.join(usd_dir, asset_path)
                            if not os.path.exists(full_path):
                                missing_refs.append(f"Prim: {prim.GetPath()}, Reference: {asset_path}")
                        elif not os.path.exists(asset_path):
                            missing_refs.append(f"Prim: {prim.GetPath()}, Reference: {asset_path}")
        except Exception as e:
            # References 확인 중 오류 발생 시 무시 (API 버전 차이 등)
            pass
    
    return missing_refs

def main():
    # 파일이 실제로 있는지 먼저 체크 (실수 방지)
    if not os.path.exists(usd_path):
        print(f"ERROR: 파일을 찾을 수 없습니다! 경로를 확인하세요: {usd_path}")
        return

    print(f"Opening stage: {usd_path}")
    
    # 2. USD 파일 불러오기
    omni.usd.get_context().open_stage(usd_path)
    
    # 스테이지가 로드될 때까지 대기 (뷰포트가 준비될 때까지)
    time.sleep(1.0)
    
    # 3. 누락된 참조 확인
    stage = omni.usd.get_context().get_stage()
    if stage:
        missing_refs = check_missing_references(stage)
        if missing_refs:
            print("\n⚠️  누락된 참조가 발견되었습니다:")
            for ref in missing_refs:
                print(f"  - {ref}")
            print("\n참고: 누락된 참조가 있어도 일부 메시나 텍스처가 보이지 않을 수 있습니다.")
            print("하지만 기본 geometry는 표시될 수 있습니다.\n")
        else:
            print("✓ 모든 참조가 정상적으로 로드되었습니다.")

    # 4. (옵션) 조명 하나 추가
    # 파일에 조명이 없으면 까맣게 보일 수 있어서, 기본 조명을 하나 달아줍니다.
    rep.create.light(light_type="Dome", intensity=1000)

    # 5. 뷰포트를 객체에 맞추기 (프레임 맞추기)
    # 여러 방법을 시도해서 뷰포트를 프레임 맞추기
    try:
        # 방법 1: get_active_viewport 사용
        viewport = viewport_utils.get_active_viewport()
        if viewport:
            viewport.frame_all()
            print("Viewport framed using get_active_viewport.")
        else:
            # 방법 2: 모든 뷰포트에 대해 시도
            import omni.kit.viewport as vp
            viewport_window = vp.get_viewport_window()
            if viewport_window:
                viewport_window.frame_all()
                print("Viewport framed using get_viewport_window.")
    except Exception as e:
        print(f"Warning: Could not auto-frame viewport: {e}")
        print("You can manually frame the view by pressing 'F' in the viewport or using the frame button.")

    print("Success! Isaac Sim is running. Press Ctrl+C in terminal to exit.")
    print("Tip: If the object is not visible, press 'F' in the viewport to frame all objects.")

    # 6. 화면 유지 루프
    while simulation_app.is_running():
        simulation_app.update()

if __name__ == "__main__":
    main()
    simulation_app.close()