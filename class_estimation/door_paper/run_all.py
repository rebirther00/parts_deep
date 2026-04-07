"""
논문용 전체 실행 오케스트레이터
- 4 models x 5 seeds x 2 resolutions = 40 학습 + 120 평가 + 집계
- Resume 지원: 완료된 실험 자동 건너뛰기
"""
import subprocess
import sys
import os
import time
import argparse
import json

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_TYPES = ["rgbd", "texture_aug", "edge", "rgbe"]
SEEDS = [42, 123, 456, 789, 1024]
RESOLUTIONS = [448, 224]

parser = argparse.ArgumentParser(description='논문용 전체 실행')
parser.add_argument('--models', nargs='+', default=MODEL_TYPES,
                    choices=MODEL_TYPES, help='실행할 모델 타입')
parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS,
                    help='사용할 시드 목록')
parser.add_argument('--resolutions', nargs='+', type=int, default=RESOLUTIONS,
                    help='해상도 목록')
parser.add_argument('--skip_train', action='store_true',
                    help='학습 건너뛰고 평가+집계만')
parser.add_argument('--skip_eval', action='store_true',
                    help='평가 건너뛰고 집계만')
parser.add_argument('-cpu', '--cpu', action='store_true')
args = parser.parse_args()


def run_cmd(cmd, desc):
    """subprocess로 명령 실행, 실시간 출력"""
    print(f"\n{'='*70}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'='*70}")
    start = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    elapsed = time.time() - start
    status = "성공" if result.returncode == 0 else f"실패 (code={result.returncode})"
    print(f"  [{status}] {elapsed/60:.1f}분 소요")
    return result.returncode == 0


def is_trained(model_type, res, seed):
    path = os.path.join(PROJECT_DIR, "artifacts",
                        f"{model_type}_{res}_seed{seed}", "model.pth")
    return os.path.exists(path)


def is_evaluated(model_type, res, seed):
    path = os.path.join(PROJECT_DIR, "artifacts",
                        f"{model_type}_{res}_seed{seed}", "eval_results.json")
    return os.path.exists(path)


def main():
    total_start = time.time()
    python = sys.executable
    cpu_flag = ["--cpu"] if args.cpu else []

    # 실험 목록 생성
    experiments = []
    for model_type in args.models:
        for res in args.resolutions:
            for seed in args.seeds:
                experiments.append((model_type, res, seed))

    total_exps = len(experiments)
    print(f"\n논문 실험 파이프라인")
    print(f"  모델: {args.models}")
    print(f"  시드: {args.seeds}")
    print(f"  해상도: {args.resolutions}")
    print(f"  총 실험 수: {total_exps}")
    print(f"  학습 건너뛰기: {args.skip_train}")
    print(f"  평가 건너뛰기: {args.skip_eval}")

    # Phase 1: 학습
    if not args.skip_train:
        print(f"\n{'#'*70}")
        print(f"# Phase 1: 학습 ({total_exps}회)")
        print(f"{'#'*70}")

        trained, skipped, failed = 0, 0, 0
        for idx, (mt, res, seed) in enumerate(experiments, 1):
            name = f"{mt}_{res}_seed{seed}"
            if is_trained(mt, res, seed):
                print(f"[{idx}/{total_exps}] {name} - 이미 완료, 건너뜀")
                skipped += 1
                continue

            cmd = [python, "train_paper.py",
                   "--model_type", mt,
                   "--seed", str(seed),
                   "--image_size", str(res)] + cpu_flag
            ok = run_cmd(cmd, f"[{idx}/{total_exps}] 학습: {name}")
            if ok:
                trained += 1
            else:
                failed += 1

        print(f"\n학습 완료: 신규 {trained}, 건너뜀 {skipped}, 실패 {failed}")

    # Phase 2: 평가
    if not args.skip_eval:
        print(f"\n{'#'*70}")
        print(f"# Phase 2: 평가 ({total_exps}회 x 3 datasets)")
        print(f"{'#'*70}")

        evaluated, skipped, failed = 0, 0, 0
        for idx, (mt, res, seed) in enumerate(experiments, 1):
            name = f"{mt}_{res}_seed{seed}"
            if not is_trained(mt, res, seed):
                print(f"[{idx}/{total_exps}] {name} - 모델 없음, 건너뜀")
                skipped += 1
                continue
            if is_evaluated(mt, res, seed):
                print(f"[{idx}/{total_exps}] {name} - 이미 평가 완료, 건너뜀")
                skipped += 1
                continue

            cmd = [python, "evaluate_paper.py",
                   "--model_type", mt,
                   "--seed", str(seed),
                   "--image_size", str(res)] + cpu_flag
            ok = run_cmd(cmd, f"[{idx}/{total_exps}] 평가: {name}")
            if ok:
                evaluated += 1
            else:
                failed += 1

        print(f"\n평가 완료: 신규 {evaluated}, 건너뜀 {skipped}, 실패 {failed}")

    # Phase 3: 집계
    print(f"\n{'#'*70}")
    print(f"# Phase 3: 결과 집계")
    print(f"{'#'*70}")

    cmd = [python, "aggregate_results.py"]
    run_cmd(cmd, "결과 집계 + 시각화")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"전체 파이프라인 완료!")
    print(f"총 소요 시간: {total_elapsed/3600:.1f}시간 ({total_elapsed/60:.0f}분)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
