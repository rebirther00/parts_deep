# ==========================================
# Door STP(STEP) → USD 변환 스크립트
# 2단계 파이프라인: STP → STL (gmsh) → USD (Isaac Sim)
#
# 사전 요구사항: sudo apt install gmsh
# 실행: ~/isaac-sim/python.sh class_estimation/door/00_convert_stp_to_usd.py
# ==========================================

import os
import sys
import time
import asyncio
import subprocess
import tempfile
import shutil

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
STP_DIR = os.path.join(REPO_DIR, "cad", "door_stp")
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

# 각 클래스별 카메라 관찰 방향 (데이터셋 생성 시 참조용)
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
# Phase 1: STP → STL 변환 (gmsh 사용)
# ==========================================
def check_gmsh():
    """gmsh 설치 여부 확인"""
    result = subprocess.run(["which", "gmsh"], capture_output=True, text=True)
    if result.returncode != 0:
        print("⚠️  gmsh가 설치되어 있지 않습니다.")
        print("   설치 방법: sudo apt install gmsh")
        return False
    print(f"✓ gmsh 경로: {result.stdout.strip()}")
    return True


def convert_stp_to_stl(stp_path, stl_path):
    """
    gmsh CLI로 STP → STL 변환

    Args:
        stp_path: STP 파일 경로
        stl_path: 출력 STL 파일 경로

    Returns:
        bool: 변환 성공 여부
    """
    cmd = [
        "gmsh",
        stp_path,
        "-2",              # 2D 표면 메시 생성
        "-o", stl_path,
        "-format", "stl",
        "-clscale", "0.5", # 메시 밀도 (작을수록 촘촘, 기본 1.0)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        print(f"    gmsh 에러: {result.stderr[-200:]}")
        return False

    return os.path.exists(stl_path) and os.path.getsize(stl_path) > 0


# ==========================================
# Phase 2: STL → USD 변환 (Isaac Sim)
# ==========================================
def progress_callback(current_step, total_steps):
    """변환 진행률 콜백"""
    if total_steps > 0:
        pct = (current_step / total_steps) * 100
        print(f"    USD 변환 진행: {current_step}/{total_steps} ({pct:.0f}%)")


async def convert_stl_to_usd(stl_path, usd_path):
    """Isaac Sim Asset Converter로 STL → USD 변환"""
    converter_manager = asset_converter.get_instance()

    context = asset_converter.AssetConverterContext()
    context.ignore_materials = False
    context.ignore_animations = True
    context.ignore_camera = True
    context.ignore_light = True
    context.export_preview_surface = False
    # CAD 원본 단위가 mm → STL도 mm 단위 → meter로 변환
    context.use_meter_as_world_unit = True
    context.create_world_as_default_root_prim = True

    task = converter_manager.create_converter_task(
        stl_path, usd_path, progress_callback, context
    )
    success = await task.wait_until_finished()

    if not success:
        status = task.get_status()
        print(f"    USD 변환 실패 상태: {status}")

    return success


# ==========================================
# 전체 파이프라인: STP → STL → USD
# ==========================================
async def convert_all():
    """모든 STP 파일을 USD로 변환 (2단계 파이프라인)"""
    print("=" * 60)
    print("Door STP → USD 변환 (2단계: STP → STL → USD)")
    print("=" * 60)
    print(f"입력 폴더: {STP_DIR}")
    print(f"출력 폴더: {OUTPUT_DIR}")
    print(f"변환 대상: {len(STP_CLASS_MAP)}개 클래스")
    print("=" * 60)

    if not check_gmsh():
        return {}

    if not os.path.exists(STP_DIR):
        print(f"\n⚠️  STP 폴더를 찾을 수 없습니다: {STP_DIR}")
        return {}

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 임시 디렉토리 (STL 중간 파일 저장용)
    tmp_dir = tempfile.mkdtemp(prefix="door_stl_")
    print(f"임시 STL 폴더: {tmp_dir}")

    results = {}
    total = len(STP_CLASS_MAP)

    for idx, (class_name, stp_filename) in enumerate(STP_CLASS_MAP.items()):
        stp_path = os.path.join(STP_DIR, stp_filename)
        stl_path = os.path.join(tmp_dir, f"{class_name}.stl")
        usd_path = os.path.join(OUTPUT_DIR, f"{class_name}.usd")

        print(f"\n[{idx + 1}/{total}] {class_name}")
        print(f"    카메라 시점: {CAMERA_VIEW_DIRECTIONS.get(class_name, 'N/A')}")

        if not os.path.exists(stp_path):
            print(f"    ⚠️  STP 파일 없음: {stp_path}")
            results[class_name] = False
            continue

        stp_size = os.path.getsize(stp_path) / 1e6
        print(f"    입력: {stp_filename} ({stp_size:.1f}MB)")

        # Phase 1: STP → STL
        print(f"    [Phase 1] STP → STL 변환 중 (gmsh)...")
        start_time = time.time()
        try:
            stl_ok = convert_stp_to_stl(stp_path, stl_path)
        except subprocess.TimeoutExpired:
            print(f"    ✗ STP → STL 변환 타임아웃 (120초 초과)")
            results[class_name] = False
            continue

        if not stl_ok:
            print(f"    ✗ STP → STL 변환 실패")
            results[class_name] = False
            continue

        stl_size = os.path.getsize(stl_path) / 1e6
        stl_elapsed = time.time() - start_time
        print(f"    ✓ STL 생성 완료 ({stl_elapsed:.1f}초, {stl_size:.1f}MB)")

        # Phase 2: STL → USD
        print(f"    [Phase 2] STL → USD 변환 중 (Isaac Sim)...")
        usd_start = time.time()
        try:
            usd_ok = await convert_stl_to_usd(stl_path, usd_path)
        except Exception as e:
            print(f"    ✗ STL → USD 변환 에러: {e}")
            results[class_name] = False
            continue

        if not usd_ok:
            results[class_name] = False
            continue

        usd_elapsed = time.time() - usd_start
        total_elapsed = time.time() - start_time
        usd_size = os.path.getsize(usd_path) / 1e6 if os.path.exists(usd_path) else 0
        print(f"    ✓ USD 생성 완료 ({usd_elapsed:.1f}초, {usd_size:.1f}MB)")
        print(f"    ✓ 전체 소요: {total_elapsed:.1f}초")
        results[class_name] = True

    # 임시 STL 폴더 정리
    print(f"\n임시 STL 폴더 정리 중: {tmp_dir}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("✓ 임시 파일 정리 완료")

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
    elif success_count > 0:
        failed = [name for name, s in results.items() if not s]
        print(f"\n⚠️  일부 실패: {', '.join(failed)}")
    else:
        print(f"\n⚠️  모든 변환이 실패했습니다.")
        print(f"  gmsh 설치를 확인하세요: sudo apt install gmsh")

    print("=" * 60)


# ==========================================
# 메인 실행
# ==========================================
async def main():
    results = await convert_all()
    print_summary(results)

    import omni.kit.app
    omni.kit.app.get_app().post_quit()


print("\nIsaac Sim Asset Converter + gmsh 파이프라인 실행 중...")
asyncio.ensure_future(main())

while simulation_app.is_running():
    simulation_app.update()

finish_logging()
simulation_app.close()
