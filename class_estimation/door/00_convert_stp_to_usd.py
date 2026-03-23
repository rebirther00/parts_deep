# ==========================================
# Door STP(STEP) → USD 변환 스크립트
# Isaac Sim의 Asset Converter를 사용하여
# CAD 파일을 USD 형식으로 변환
# ==========================================

import os
import sys
import time
import asyncio

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # class_estimation/
REPO_DIR = os.path.dirname(PROJECT_DIR)    # parts_deep/

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from utils.logger import setup_logging, reinit_logging, finish_logging

LOG_PATH = setup_logging("00_convert_stp")

# Isaac Sim 초기화 (변환만 하므로 headless 모드)
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

# SimulationApp 초기화 후 모듈 임포트
import omni.kit.asset_converter as asset_converter

reinit_logging(LOG_PATH)

# ==========================================
# 설정
# ==========================================
# STP 파일 경로
STP_DIR = os.path.join(REPO_DIR, "cad", "door_stp")

# USD 출력 경로 (기존 부품 assets와 분리)
OUTPUT_DIR = os.path.expanduser("~/isaac-sim/assets/door")

# STP 파일 → 클래스 매핑 (8클래스)
# E30_door_RH와 E38_door_RH는 동일 부품이므로 하나의 클래스로 통합
STP_CLASS_MAP = {
    "E25_door_LH_FRT":  "E25_door_LH_FRT.stp",
    "E25_door_LH_RR":   "E25_door_LH_RR.stp",
    "E25_door_RH":      "E25_door_RH.stp",
    "E30_door_LH_FRT":  "E30_door_LH_FRT.stp",
    "E30_door_LH_RR":   "E30_door_LH_RR.stp",
    "E30_E38_door_RH":  "E30_E38_door_RH.stp",
    "E38_door_LH_FRT":  "E38_door_LH_FRT.stp",
    "E38_door_LH_RR":   "E38_door_LH_RR.stp",
}

# 각 클래스별 카메라 관찰 방향 (데이터셋 생성 시 참조용으로 메타데이터에 기록)
CAMERA_VIEW_DIRECTIONS = {
    "E25_door_LH_FRT":  "-Z",
    "E25_door_LH_RR":   "-Z",
    "E25_door_RH":      "-Z",
    "E30_door_LH_FRT":  "+Z",
    "E30_door_LH_RR":   "+Z",
    "E30_E38_door_RH":  "-Z",
    "E38_door_LH_FRT":  "+Z",
    "E38_door_LH_RR":   "+Z",
}


# ==========================================
# 변환 함수
# ==========================================
def progress_callback(current_step, total_steps):
    """변환 진행률 콜백"""
    if total_steps > 0:
        pct = (current_step / total_steps) * 100
        print(f"    진행: {current_step}/{total_steps} ({pct:.0f}%)")


async def convert_single_file(input_path, output_path):
    """
    단일 STP 파일을 USD로 변환

    Args:
        input_path: STP 파일 절대 경로
        output_path: USD 파일 출력 절대 경로

    Returns:
        bool: 변환 성공 여부
    """
    converter_manager = asset_converter.get_instance()

    context = asset_converter.AssetConverterContext()
    context.ignore_materials = False
    context.ignore_animations = True
    context.ignore_camera = True
    context.ignore_light = True
    context.export_preview_surface = False
    # CAD 파일 단위가 mm → USD 표준 단위(meter)로 변환
    context.use_meter_as_world_unit = True
    context.create_world_as_default_root_prim = True

    task = converter_manager.create_converter_task(
        input_path, output_path, progress_callback, context
    )
    success = await task.wait_until_finished()

    if not success:
        status = task.get_status()
        print(f"    변환 실패 상태: {status}")

    return success


async def convert_all():
    """모든 STP 파일을 USD로 변환"""
    print("=" * 60)
    print("Door STP → USD 변환 시작")
    print("=" * 60)
    print(f"입력 폴더: {STP_DIR}")
    print(f"출력 폴더: {OUTPUT_DIR}")
    print(f"변환 대상: {len(STP_CLASS_MAP)}개 클래스")
    print("=" * 60)

    # STP 폴더 확인
    if not os.path.exists(STP_DIR):
        print(f"\n⚠️  STP 폴더를 찾을 수 없습니다: {STP_DIR}")
        return {}

    # 출력 폴더 생성
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}
    total = len(STP_CLASS_MAP)

    for idx, (class_name, stp_filename) in enumerate(STP_CLASS_MAP.items()):
        input_path = os.path.join(STP_DIR, stp_filename)
        output_path = os.path.join(OUTPUT_DIR, f"{class_name}.usd")

        print(f"\n[{idx + 1}/{total}] {class_name}")
        print(f"    입력: {stp_filename} ({os.path.getsize(input_path) / 1e6:.1f}MB)" if os.path.exists(input_path) else "")
        print(f"    출력: {output_path}")
        print(f"    카메라 시점: {CAMERA_VIEW_DIRECTIONS.get(class_name, 'N/A')}")

        if not os.path.exists(input_path):
            print(f"    ⚠️  STP 파일 없음: {input_path}")
            results[class_name] = False
            continue

        start_time = time.time()
        try:
            success = await convert_single_file(input_path, output_path)
            elapsed = time.time() - start_time
            results[class_name] = success

            if success:
                usd_size = os.path.getsize(output_path) / 1e6 if os.path.exists(output_path) else 0
                print(f"    ✓ 변환 완료 ({elapsed:.1f}초, {usd_size:.1f}MB)")
            else:
                print(f"    ✗ 변환 실패 ({elapsed:.1f}초)")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"    ✗ 변환 중 에러 발생 ({elapsed:.1f}초): {e}")
            results[class_name] = False

    return results


def print_summary(results):
    """변환 결과 요약 출력"""
    print("\n" + "=" * 60)
    print("변환 결과 요약")
    print("=" * 60)

    success_count = 0
    for class_name, success in results.items():
        status = "✓ 성공" if success else "✗ 실패"
        view_dir = CAMERA_VIEW_DIRECTIONS.get(class_name, "N/A")
        print(f"  {class_name:25s} {status}  (카메라: {view_dir})")
        if success:
            success_count += 1

    print(f"\n  총 {success_count}/{len(results)} 변환 성공")

    if success_count == len(results):
        print(f"\n✓ 모든 파일 변환 완료!")
        print(f"  USD 파일 위치: {OUTPUT_DIR}")
        print(f"\n다음 단계: 01_generate_door_dataset.py 로 합성 데이터셋 생성")
    else:
        failed = [name for name, s in results.items() if not s]
        print(f"\n⚠️  실패한 클래스: {', '.join(failed)}")
        print(f"  Isaac Sim GUI에서 수동 Import를 시도해 보세요.")
        print(f"  (File → Import → STP 파일 선택 → File → Save As → USD)")

    print("=" * 60)


# ==========================================
# 메인 실행
# ==========================================
async def main():
    results = await convert_all()
    print_summary(results)

    # 변환 완료 후 앱 종료
    import omni.kit.app
    omni.kit.app.get_app().post_quit()


print("\nIsaac Sim Asset Converter 실행 중...")
asyncio.ensure_future(main())

while simulation_app.is_running():
    simulation_app.update()

finish_logging()
simulation_app.close()
