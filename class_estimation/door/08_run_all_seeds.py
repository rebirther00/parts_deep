"""과제 성능 보고서용 - 5 seed 순차 실행 + 결과 집계
- 07_seed_evaluation.py를 5개 seed로 실행
- mean ± std 계산, 결과 표 생성
"""
import subprocess
import sys
import os
import json
import time
import argparse

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(PROJECT_DIR, "artifacts", "report")

SEEDS = [42, 123, 456, 789, 1024]
IMAGE_SIZE = 448
DATASET_LABELS = {
    "original": "원본 (Test)",
    "datasets_aug": "증강1 (Foreground)",
    "datasets_aug2": "증강2 (Full)",
}

parser = argparse.ArgumentParser(description='과제 보고서용 5-seed 실행')
parser.add_argument('-cpu', '--cpu', action='store_true')
parser.add_argument('--skip_train', action='store_true',
                    help='학습 건너뛰고 집계만')
args = parser.parse_args()


def run_cmd(cmd, desc):
    print(f"\n{'='*70}")
    print(f"  {desc}")
    print(f"{'='*70}")
    start = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    elapsed = time.time() - start
    status = "성공" if result.returncode == 0 else f"실패 (code={result.returncode})"
    print(f"  [{status}] {elapsed/60:.1f}분 소요")
    return result.returncode == 0


def aggregate():
    """5 seed 결과 집계"""
    all_results = {}
    for seed in SEEDS:
        run_name = f"rgbd_{IMAGE_SIZE}_seed{seed}"
        eval_path = os.path.join(REPORT_DIR, run_name, "eval_results.json")
        if os.path.exists(eval_path):
            with open(eval_path) as f:
                all_results[seed] = json.load(f)

    if not all_results:
        print("수집된 결과가 없습니다.")
        return

    variants = set()
    for seed_results in all_results.values():
        variants.update(seed_results.keys())

    summary = {}
    for variant in sorted(variants):
        accs, f1s, precs, recs = [], [], [], []
        for seed in SEEDS:
            if seed in all_results and variant in all_results[seed]:
                r = all_results[seed][variant]
                accs.append(r["accuracy"])
                f1s.append(r["macro_f1"])
                precs.append(r["macro_precision"])
                recs.append(r["macro_recall"])

        if accs:
            n = len(accs)
            summary[variant] = {
                "n_seeds": n,
                "acc_mean": round(np.mean(accs), 2),
                "acc_std": round(np.std(accs, ddof=1) if n > 1 else 0, 2),
                "f1_mean": round(np.mean(f1s), 2),
                "f1_std": round(np.std(f1s, ddof=1) if n > 1 else 0, 2),
                "prec_mean": round(np.mean(precs), 2),
                "prec_std": round(np.std(precs, ddof=1) if n > 1 else 0, 2),
                "rec_mean": round(np.mean(recs), 2),
                "rec_std": round(np.std(recs, ddof=1) if n > 1 else 0, 2),
                "acc_values": accs,
                "f1_values": f1s,
            }

    # JSON 저장
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "summary.json"), 'w',
              encoding='utf-8') as f:
        json.dump({"summary": summary, "seeds": SEEDS,
                   "raw_results": {str(k): v for k, v in all_results.items()}},
                  f, ensure_ascii=False, indent=2)

    # Markdown 생성
    lines = ["# 과제 성능 보고서 - RGBD 분류 결과\n"]
    lines.append(f"- **모델**: RGBDAuxResNet18 (RGBD 4채널 + Aux)")
    lines.append(f"- **해상도**: {IMAGE_SIZE}x{IMAGE_SIZE}")
    lines.append(f"- **분할**: Train 70% / Test 30%")
    lines.append(f"- **Seeds**: {SEEDS}")
    lines.append(f"- **완료**: {len(all_results)}개 seed\n")

    lines.append("## 종합 결과 (mean ± std)\n")
    lines.append("| 데이터셋 | Accuracy (%) | Macro F1 (%) "
                 "| Macro Precision (%) | Macro Recall (%) |")
    lines.append("|----------|:---:|:---:|:---:|:---:|")

    for variant in ["original", "datasets_aug", "datasets_aug2"]:
        if variant not in summary:
            continue
        s = summary[variant]
        label = DATASET_LABELS.get(variant, variant)
        lines.append(
            f"| {label} | "
            f"{s['acc_mean']:.2f} ± {s['acc_std']:.2f} | "
            f"{s['f1_mean']:.2f} ± {s['f1_std']:.2f} | "
            f"{s['prec_mean']:.2f} ± {s['prec_std']:.2f} | "
            f"{s['rec_mean']:.2f} ± {s['rec_std']:.2f} |"
        )

    lines.append("\n## Seed별 상세 결과\n")
    lines.append("| Seed | 데이터셋 | Accuracy | Macro F1 "
                 "| Macro Precision | Macro Recall |")
    lines.append("|:---:|----------|:---:|:---:|:---:|:---:|")

    for seed in SEEDS:
        if seed not in all_results:
            continue
        for variant in ["original", "datasets_aug", "datasets_aug2"]:
            if variant not in all_results[seed]:
                continue
            r = all_results[seed][variant]
            label = DATASET_LABELS.get(variant, variant)
            lines.append(
                f"| {seed} | {label} | "
                f"{r['accuracy']:.2f} | {r['macro_f1']:.2f} | "
                f"{r['macro_precision']:.2f} | {r['macro_recall']:.2f} |"
            )

    md_content = "\n".join(lines)
    md_path = os.path.join(REPORT_DIR, "summary.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"\n결과 요약 저장: {md_path}")
    print("\n" + md_content)


def main():
    python = sys.executable
    total_start = time.time()

    print("과제 성능 보고서 - RGBD 5 Seed 실험")
    print(f"  Seeds: {SEEDS}")
    print(f"  해상도: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  분할: Train 70% / Test 30%")

    if not args.skip_train:
        trained, skipped, failed = 0, 0, 0
        cpu_flag = ["--cpu"] if args.cpu else []

        for idx, seed in enumerate(SEEDS, 1):
            run_name = f"rgbd_{IMAGE_SIZE}_seed{seed}"
            eval_path = os.path.join(REPORT_DIR, run_name,
                                     "eval_results.json")

            if os.path.exists(eval_path):
                print(f"[{idx}/{len(SEEDS)}] {run_name} - 이미 완료, 건너뜀")
                skipped += 1
                continue

            cmd = [python, "07_seed_evaluation.py",
                   "--seed", str(seed),
                   "--image_size", str(IMAGE_SIZE)] + cpu_flag
            ok = run_cmd(cmd, f"[{idx}/{len(SEEDS)}] {run_name}")
            if ok:
                trained += 1
            else:
                failed += 1

        print(f"\n학습+평가: 신규 {trained}, 건너뜀 {skipped}, 실패 {failed}")

    print(f"\n{'#'*70}")
    print("# 결과 집계")
    print(f"{'#'*70}")
    aggregate()

    total_elapsed = time.time() - total_start
    print(f"\n전체 소요: {total_elapsed/60:.1f}분")


if __name__ == "__main__":
    main()
