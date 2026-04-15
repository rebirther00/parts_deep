"""
논문용 전체 실행 오케스트레이터
- N models x 5 seeds x 2 resolutions 학습 + 평가 + 집계
- Resume 지원: 완료된 실험 자동 건너뛰기
- --no_aux: Aux MLP 제거 ablation 모드
"""
import subprocess
import sys
import os
import time
import argparse
import json

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

ALL_MODEL_TYPES = ["rgb", "rgbd", "texture_aug", "edge", "rgbe", "rgbe_texture_aug"]
SEEDS = [42, 123, 456, 789, 1024]
RESOLUTIONS = [448, 224]

parser = argparse.ArgumentParser(description='논문용 전체 실행')
parser.add_argument('--models', nargs='+', default=None,
                    choices=ALL_MODEL_TYPES, help='실행할 모델 타입')
parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS,
                    help='사용할 시드 목록')
parser.add_argument('--resolutions', nargs='+', type=int, default=RESOLUTIONS,
                    help='해상도 목록')
parser.add_argument('--no_aux', action='store_true',
                    help='Aux MLP 제거 ablation 실험')
parser.add_argument('--skip_train', action='store_true',
                    help='학습 건너뛰고 평가+집계만')
parser.add_argument('--skip_eval', action='store_true',
                    help='평가 건너뛰고 집계만')
parser.add_argument('--skip_aggregate', action='store_true',
                    help='집계 건너뛰기')
parser.add_argument('-cpu', '--cpu', action='store_true')
args = parser.parse_args()

if args.models is None:
    args.models = ["rgb", "rgbd", "edge", "rgbe"] if args.no_aux \
        else ["rgb", "rgbd", "texture_aug", "edge", "rgbe"]


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


def run_name(model_type, res, seed, no_aux=False):
    suffix = "_noaux" if no_aux else ""
    return f"{model_type}{suffix}_{res}_seed{seed}"


def is_trained(model_type, res, seed, no_aux=False):
    path = os.path.join(PROJECT_DIR, "artifacts",
                        run_name(model_type, res, seed, no_aux), "model.pth")
    return os.path.exists(path)


def is_evaluated(model_type, res, seed, no_aux=False):
    path = os.path.join(PROJECT_DIR, "artifacts",
                        run_name(model_type, res, seed, no_aux),
                        "eval_results.json")
    return os.path.exists(path)


def main():
    total_start = time.time()
    python = sys.executable
    cpu_flag = ["--cpu"] if args.cpu else []
    noaux_flag = ["--no_aux"] if args.no_aux else []
    mode_label = "w/o Aux" if args.no_aux else "w/ Aux"

    experiments = []
    for model_type in args.models:
        for res in args.resolutions:
            for seed in args.seeds:
                experiments.append((model_type, res, seed))

    total_exps = len(experiments)
    print(f"\n논문 실험 파이프라인 ({mode_label})")
    print(f"  모델: {args.models}")
    print(f"  시드: {args.seeds}")
    print(f"  해상도: {args.resolutions}")
    print(f"  Aux MLP: {'제거' if args.no_aux else '포함'}")
    print(f"  총 실험 수: {total_exps}")

    # Phase 1: 학습
    if not args.skip_train:
        print(f"\n{'#'*70}")
        print(f"# Phase 1: 학습 ({total_exps}회, {mode_label})")
        print(f"{'#'*70}")

        trained, skipped, failed = 0, 0, 0
        for idx, (mt, res, seed) in enumerate(experiments, 1):
            name = run_name(mt, res, seed, args.no_aux)
            if is_trained(mt, res, seed, args.no_aux):
                print(f"[{idx}/{total_exps}] {name} - 이미 완료, 건너뜀")
                skipped += 1
                continue

            cmd = [python, "train_paper.py",
                   "--model_type", mt,
                   "--seed", str(seed),
                   "--image_size", str(res)] + noaux_flag + cpu_flag
            ok = run_cmd(cmd, f"[{idx}/{total_exps}] 학습: {name}")
            if ok:
                trained += 1
            else:
                failed += 1

        print(f"\n학습 완료: 신규 {trained}, 건너뜀 {skipped}, 실패 {failed}")

    # Phase 2: 평가
    if not args.skip_eval:
        print(f"\n{'#'*70}")
        print(f"# Phase 2: 평가 ({total_exps}회 x 3 datasets, {mode_label})")
        print(f"{'#'*70}")

        evaluated, skipped, failed = 0, 0, 0
        for idx, (mt, res, seed) in enumerate(experiments, 1):
            name = run_name(mt, res, seed, args.no_aux)
            if not is_trained(mt, res, seed, args.no_aux):
                print(f"[{idx}/{total_exps}] {name} - 모델 없음, 건너뜀")
                skipped += 1
                continue
            if is_evaluated(mt, res, seed, args.no_aux):
                print(f"[{idx}/{total_exps}] {name} - 이미 평가 완료, 건너뜀")
                skipped += 1
                continue

            cmd = [python, "evaluate_paper.py",
                   "--model_type", mt,
                   "--seed", str(seed),
                   "--image_size", str(res)] + noaux_flag + cpu_flag
            ok = run_cmd(cmd, f"[{idx}/{total_exps}] 평가: {name}")
            if ok:
                evaluated += 1
            else:
                failed += 1

        print(f"\n평가 완료: 신규 {evaluated}, 건너뜀 {skipped}, 실패 {failed}")

    # Phase 3: 집계
    if not args.skip_aggregate:
        print(f"\n{'#'*70}")
        print(f"# Phase 3: 결과 집계")
        print(f"{'#'*70}")

        agg_args = ["--no_aux"] if args.no_aux else []
        cmd = [python, "aggregate_results.py"] + agg_args
        run_cmd(cmd, "결과 집계 + 시각화")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"전체 파이프라인 완료!")
    print(f"총 소요 시간: {total_elapsed/3600:.1f}시간 ({total_elapsed/60:.0f}분)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
