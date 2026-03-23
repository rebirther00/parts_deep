# ==========================================
# Door STL → USD 변환 스크립트
# Isaac Sim Asset Converter로 STL을 USD로 변환
#
# 실행: ~/isaac-sim/python.sh class_estimation/door/00_convert_stl_to_usd.py
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

LOG_PATH = setup_logging("00_convert_stl")

# Isaac Sim 초기화 (변환만 하므로 headless 모드)
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import omni.kit.asset_converter as asset_converter
from pxr import Usd, UsdGeom, Gf

reinit_logging(LOG_PATH)

# ==========================================
# 설정
# ==========================================
STL_DIR = os.path.join(REPO_DIR, "cad", "door_stl")
OUTPUT_DIR = os.path.expanduser("~/isaac-sim/assets/door")

# STL 파일 → 클래스 매핑 (8클래스)
# E30_door_RH와 E38_door_RH는 동일 부품이므로 하나의 클래스로 통합
STL_CLASS_MAP = {
    "E25_door_LH_FRT":  "E25_door_LH_FRT.stl",
    "E25_door_LH_RR":   "E25_door_LH_RR.stl",
    "E25_door_RH":      "E25_door_RH.stl",
    "E30_door_LH_FRT":  "E30_door_LH_FRT.stl",
    "E30_door_LH_RR":   "E30_door_LH_RR.stl",
    "E30_E38_door_RH":  "E30_E38_door_RH.stl",
    "E38_door_LH_FRT":  "E38_door_LH_FRT.stl",
    "E38_door_LH_RR":   "E38_door_LH_RR.stl",
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
# STL → USD 변환
# ==========================================
def fix_usd_units(usd_path):
    """STL(mm 좌표) → 미터 단위로 변환 + 원점 센터링"""
    stage = Usd.Stage.Open(usd_path)

    # 메시 AABB 계산 (원본 mm 좌표)
    tc = Usd.TimeCode.Default()
    xc = UsdGeom.XformCache(tc)
    w_min = Gf.Vec3d(float("inf"), float("inf"), float("inf"))
    w_max = Gf.Vec3d(float("-inf"), float("-inf"), float("-inf"))

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get(tc)
        if not pts:
            continue
        M = xc.GetLocalToWorldTransform(prim)
        for p in pts:
            wp = M.Transform(Gf.Vec3d(p[0], p[1], p[2]))
            for i in range(3):
                w_min[i] = min(w_min[i], wp[i])
                w_max[i] = max(w_max[i], wp[i])

    center = (w_min + w_max) / 2.0
    size = w_max - w_min
    max_dim = max(size[0], size[1], size[2])

    # root prim에 Xform 적용: 센터링 → mm→m 스케일
    root = stage.GetDefaultPrim()
    if root:
        xformable = UsdGeom.Xformable(root)
        xformable.ClearXformOpOrder()
        # 스케일(0.001) 먼저 정의 → 이동(-center) 나중 정의
        # USD는 ops를 왼쪽→오른쪽 합성: M = M_scale * M_translate
        # 점 v에 대해: v' = Scale * (v + Translate) = 0.001 * (v - center)
        xformable.AddScaleOp().Set(Gf.Vec3d(0.001, 0.001, 0.001))
        xformable.AddTranslateOp().Set(Gf.Vec3d(-center[0], -center[1], -center[2]))

    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.Save()

    real_size_m = max_dim * 0.001
    print(f"    원본 중심: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f}) mm")
    print(f"    원본 최대 치수: {max_dim:.1f} mm → 실제 크기: {real_size_m:.3f} m")
    print(f"    센터링 + 스케일(0.001) + metersPerUnit=1.0 적용 완료")


def progress_callback(current_step, total_steps):
    if total_steps > 0:
        pct = (current_step / total_steps) * 100
        print(f"    진행: {current_step}/{total_steps} ({pct:.0f}%)")


async def convert_stl_to_usd(stl_path, usd_path):
    """Isaac Sim Asset Converter로 STL → USD 변환"""
    converter_manager = asset_converter.get_instance()

    context = asset_converter.AssetConverterContext()
    context.ignore_materials = False
    context.ignore_animations = True
    context.ignore_camera = True
    context.ignore_light = True
    context.export_preview_surface = False
    # 변환 후 fix_usd_units()에서 스케일/센터링 처리
    context.use_meter_as_world_unit = False
    context.create_world_as_default_root_prim = True

    task = converter_manager.create_converter_task(
        stl_path, usd_path, progress_callback, context
    )
    success = await task.wait_until_finished()

    if not success:
        status = task.get_status()
        print(f"    변환 실패 상태: {status}")

    return success


async def convert_all():
    """모든 STL 파일을 USD로 변환"""
    print("=" * 60)
    print("Door STL → USD 변환")
    print("=" * 60)
    print(f"입력 폴더: {STL_DIR}")
    print(f"출력 폴더: {OUTPUT_DIR}")
    print(f"변환 대상: {len(STL_CLASS_MAP)}개 클래스")
    print("=" * 60)

    if not os.path.exists(STL_DIR):
        print(f"\n⚠️  STL 폴더를 찾을 수 없습니다: {STL_DIR}")
        print(f"   {STL_DIR} 에 STL 파일을 배치해 주세요.")
        return {}

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = {}
    total = len(STL_CLASS_MAP)

    for idx, (class_name, stl_filename) in enumerate(STL_CLASS_MAP.items()):
        stl_path = os.path.join(STL_DIR, stl_filename)
        usd_path = os.path.join(OUTPUT_DIR, f"{class_name}.usd")

        print(f"\n[{idx + 1}/{total}] {class_name}")
        print(f"    카메라 시점: {CAMERA_VIEW_DIRECTIONS.get(class_name, 'N/A')}")

        if not os.path.exists(stl_path):
            print(f"    ⚠️  STL 파일 없음: {stl_path}")
            results[class_name] = False
            continue

        stl_size = os.path.getsize(stl_path) / 1e6
        print(f"    입력: {stl_filename} ({stl_size:.1f}MB)")

        start_time = time.time()
        try:
            success = await convert_stl_to_usd(stl_path, usd_path)
            elapsed = time.time() - start_time

            if success:
                fix_usd_units(usd_path)
                usd_size = os.path.getsize(usd_path) / 1e6 if os.path.exists(usd_path) else 0
                print(f"    ✓ 변환 완료 ({elapsed:.1f}초, {usd_size:.1f}MB)")
            else:
                print(f"    ✗ 변환 실패 ({elapsed:.1f}초)")

            results[class_name] = success
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"    ✗ 에러 발생 ({elapsed:.1f}초): {e}")
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
        print(f"\n다음 단계: 합성 데이터셋 생성 스크립트 실행")
    elif success_count > 0:
        failed = [name for name, s in results.items() if not s]
        print(f"\n⚠️  일부 실패: {', '.join(failed)}")
    else:
        print(f"\n⚠️  모든 변환이 실패했습니다.")

    print("=" * 60)


# ==========================================
# 메인 실행
# ==========================================
async def main():
    results = await convert_all()
    print_summary(results)

    import omni.kit.app
    omni.kit.app.get_app().post_quit()


print("\nIsaac Sim Asset Converter 실행 중...")
asyncio.ensure_future(main())

while simulation_app.is_running():
    simulation_app.update()

finish_logging()
simulation_app.close()
