"""
논문용 결과 집계 스크립트
- 모든 seed/model/resolution 결과 수집
- mean +- std 계산, paired t-test, 시각화 생성
- --no_aux: Aux MLP 제거 실험 결과 별도 집계
"""
import json
import os
import sys
import glob
import itertools
import argparse

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

agg_parser = argparse.ArgumentParser(description='논문용 결과 집계')
agg_parser.add_argument('--no_aux', action='store_true',
                        help='Aux MLP 제거 실험 결과 집계')
agg_args = agg_parser.parse_args()

NO_AUX_MODE = agg_args.no_aux
NOAUX_SUFFIX = "_noaux" if NO_AUX_MODE else ""
SUMMARY_SUBDIR = "summary_noaux" if NO_AUX_MODE else "summary"

MODELS_WITH_AUX = ["rgbd", "texture_aug", "edge", "rgbe"]
MODELS_NO_AUX = ["rgbd", "texture_aug", "edge", "rgbe"]
MODEL_TYPES = MODELS_NO_AUX if NO_AUX_MODE else MODELS_WITH_AUX

MODEL_LABELS = {
    "rgbd": "Baseline RGBD",
    "texture_aug": "Texture Aug RGBD",
    "edge": "Edge-only",
    "rgbe": "RGBE Hybrid",
    "rgbe_texture_aug": "RGBE + TexAug",
}
DATASET_VARIANTS = ["original", "datasets_aug", "datasets_aug2"]
DATASET_LABELS = {
    "original": "Test Original",
    "datasets_aug": "Aug (Foreground)",
    "datasets_aug2": "Aug (Full)",
}
SEEDS = [42, 123, 456, 789, 1024]
RESOLUTIONS = [448, 224]


def make_run_name(model_type, res, seed):
    return f"{model_type}{NOAUX_SUFFIX}_{res}_seed{seed}"


def collect_results():
    """모든 결과 JSON 수집"""
    results = {}
    artifacts_dir = os.path.join(PROJECT_DIR, "artifacts")
    for model_type in MODEL_TYPES:
        for res in RESOLUTIONS:
            for seed in SEEDS:
                rn = make_run_name(model_type, res, seed)
                eval_path = os.path.join(artifacts_dir, rn,
                                         "eval_results.json")
                if os.path.exists(eval_path):
                    with open(eval_path, 'r') as f:
                        results[rn] = json.load(f)
    return results


def compute_stats(results):
    """(model_type, resolution, dataset) 그룹별 mean +- std 계산"""
    summary = {}
    for model_type in MODEL_TYPES:
        for res in RESOLUTIONS:
            for variant in DATASET_VARIANTS:
                key = f"{model_type}_{res}_{variant}"
                accs, f1s = [], []
                for seed in SEEDS:
                    rn = make_run_name(model_type, res, seed)
                    if rn in results and variant in results[rn]:
                        r = results[rn][variant]
                        accs.append(r["accuracy"])
                        f1s.append(r["macro_f1"])
                if accs:
                    summary[key] = {
                        "model": model_type,
                        "resolution": res,
                        "dataset": variant,
                        "n_seeds": len(accs),
                        "acc_mean": round(np.mean(accs), 2),
                        "acc_std": round(np.std(accs, ddof=1) if len(accs) > 1 else 0, 2),
                        "f1_mean": round(np.mean(f1s), 2),
                        "f1_std": round(np.std(f1s, ddof=1) if len(f1s) > 1 else 0, 2),
                        "acc_values": accs,
                        "f1_values": f1s,
                    }
    # 하락폭 계산
    for model_type in MODEL_TYPES:
        for res in RESOLUTIONS:
            orig_key = f"{model_type}_{res}_original"
            if orig_key not in summary:
                continue
            orig_accs = summary[orig_key]["acc_values"]
            for variant in ["datasets_aug", "datasets_aug2"]:
                aug_key = f"{model_type}_{res}_{variant}"
                if aug_key not in summary:
                    continue
                aug_accs = summary[aug_key]["acc_values"]
                n = min(len(orig_accs), len(aug_accs))
                deltas = [aug_accs[i] - orig_accs[i] for i in range(n)]
                summary[aug_key]["delta_mean"] = round(np.mean(deltas), 2)
                summary[aug_key]["delta_std"] = round(
                    np.std(deltas, ddof=1) if n > 1 else 0, 2)
    return summary


def paired_tests(results):
    """모델 쌍별 paired t-test (F1 기준)"""
    test_results = []
    for res in RESOLUTIONS:
        for variant in DATASET_VARIANTS:
            model_f1s = {}
            for model_type in MODEL_TYPES:
                vals = []
                for seed in SEEDS:
                    rn = make_run_name(model_type, res, seed)
                    if rn in results and variant in results[rn]:
                        vals.append(results[rn][variant]["macro_f1"])
                if vals:
                    model_f1s[model_type] = vals

            for m1, m2 in itertools.combinations(MODEL_TYPES, 2):
                if m1 not in model_f1s or m2 not in model_f1s:
                    continue
                v1, v2 = model_f1s[m1], model_f1s[m2]
                n = min(len(v1), len(v2))
                if n < 3:
                    continue
                t_stat, p_val = stats.ttest_rel(v1[:n], v2[:n])
                test_results.append({
                    "resolution": res,
                    "dataset": variant,
                    "model_a": m1,
                    "model_b": m2,
                    "t_statistic": round(float(t_stat), 4),
                    "p_value": round(float(p_val), 6),
                    "significant": bool(p_val < 0.05),
                    "n_pairs": n,
                })
    return test_results


def plot_learning_curves(res=448):
    """학습 곡선 (Val Accuracy vs Epoch) - 모델 오버레이"""
    artifacts_dir = os.path.join(PROJECT_DIR, "artifacts")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {'rgbd': '#1f77b4', 'texture_aug': '#ff7f0e',
              'edge': '#2ca02c', 'rgbe': '#d62728',
              'rgbe_texture_aug': '#9467bd'}

    for model_type in MODEL_TYPES:
        all_val_accs = []
        for seed in SEEDS:
            rn = make_run_name(model_type, res, seed)
            log_path = os.path.join(artifacts_dir, rn, "train_log.json")
            if not os.path.exists(log_path):
                continue
            with open(log_path) as f:
                log = json.load(f)
            val_accs = [e["val_acc"] for e in log["epochs"]]
            all_val_accs.append(val_accs)
            ax.plot(range(1, len(val_accs) + 1), val_accs,
                    color=colors.get(model_type, '#888888'),
                    alpha=0.15, linewidth=0.8)

        if all_val_accs:
            max_len = max(len(v) for v in all_val_accs)
            padded = np.full((len(all_val_accs), max_len), np.nan)
            for i, v in enumerate(all_val_accs):
                padded[i, :len(v)] = v
            mean_curve = np.nanmean(padded, axis=0)
            ax.plot(range(1, max_len + 1), mean_curve,
                    color=colors.get(model_type, '#888888'), linewidth=2.5,
                    label=MODEL_LABELS[model_type])

    mode_label = "(w/o Aux)" if NO_AUX_MODE else ""
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title(f'Learning Curves ({res}x{res}) {mode_label}', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    summary_dir = os.path.join(artifacts_dir, SUMMARY_SUBDIR)
    os.makedirs(summary_dir, exist_ok=True)
    save_path = os.path.join(summary_dir, f"learning_curves_{res}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"학습 곡선 저장: {save_path}")


def plot_robustness_bars(summary, res=448):
    """강건성 비교 막대 그래프"""
    fig, axes = plt.subplots(1, 2, figsize=(max(14, len(MODEL_TYPES)*3.5), 6))

    for ax_idx, metric in enumerate(["acc", "f1"]):
        ax = axes[ax_idx]
        metric_label = "Accuracy" if metric == "acc" else "Macro F1"
        x = np.arange(len(MODEL_TYPES))
        width = 0.25

        for d_idx, variant in enumerate(DATASET_VARIANTS):
            means, stds = [], []
            for model_type in MODEL_TYPES:
                key = f"{model_type}_{res}_{variant}"
                if key in summary:
                    means.append(summary[key][f"{metric}_mean"])
                    stds.append(summary[key][f"{metric}_std"])
                else:
                    means.append(0)
                    stds.append(0)
            ax.bar(x + d_idx * width, means, width,
                   yerr=stds, capsize=3,
                   label=DATASET_LABELS[variant], alpha=0.85)

        mode_label = "(w/o Aux)" if NO_AUX_MODE else ""
        ax.set_xlabel('Model', fontsize=11)
        ax.set_ylabel(f'{metric_label} (%)', fontsize=11)
        ax.set_title(f'{metric_label} ({res}x{res}) {mode_label}', fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_TYPES],
                          rotation=15, ha='right', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')
        ax.set_ylim(50, 105)

    plt.tight_layout()
    summary_dir = os.path.join(PROJECT_DIR, "artifacts", SUMMARY_SUBDIR)
    os.makedirs(summary_dir, exist_ok=True)
    save_path = os.path.join(summary_dir, f"robustness_comparison_{res}.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"강건성 비교 차트 저장: {save_path}")


def plot_resolution_comparison(summary):
    """448 vs 224 비교 차트"""
    fig, ax = plt.subplots(figsize=(max(12, len(MODEL_TYPES)*3), 6))
    x = np.arange(len(MODEL_TYPES))
    width = 0.15

    dataset_res_combos = list(itertools.product(DATASET_VARIANTS, RESOLUTIONS))
    for combo_idx, (variant, res) in enumerate(dataset_res_combos):
        means, stds = [], []
        for model_type in MODEL_TYPES:
            key = f"{model_type}_{res}_{variant}"
            if key in summary:
                means.append(summary[key]["acc_mean"])
                stds.append(summary[key]["acc_std"])
            else:
                means.append(0)
                stds.append(0)
        label = f"{DATASET_LABELS[variant]} ({res})"
        ax.bar(x + combo_idx * width, means, width,
               yerr=stds, capsize=2, label=label, alpha=0.8)

    mode_label = "(w/o Aux)" if NO_AUX_MODE else ""
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_title(f'Resolution Comparison (448 vs 224) {mode_label}', fontsize=13)
    total_w = len(dataset_res_combos) * width
    ax.set_xticks(x + total_w / 2 - width / 2)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_TYPES],
                      rotation=15, ha='right', fontsize=9)
    ax.legend(fontsize=7, ncol=3, loc='lower left')
    ax.grid(True, alpha=0.2, axis='y')
    ax.set_ylim(50, 105)

    plt.tight_layout()
    summary_dir = os.path.join(PROJECT_DIR, "artifacts", SUMMARY_SUBDIR)
    os.makedirs(summary_dir, exist_ok=True)
    save_path = os.path.join(summary_dir, "resolution_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"해상도 비교 차트 저장: {save_path}")


def generate_markdown_summary(summary, tests):
    """결과 요약 markdown 생성"""
    mode = "w/o Aux" if NO_AUX_MODE else "w/ Aux"
    lines = [f"# 실험 결과 요약 ({mode})\n"]

    for res in RESOLUTIONS:
        lines.append(f"\n## {res}x{res} 해상도\n")
        lines.append("### Accuracy (mean +- std)")
        lines.append("| Model | Test Original | Aug (Foreground) | Aug (Full) |")
        lines.append("|-------|:---:|:---:|:---:|")
        for mt in MODEL_TYPES:
            row = f"| {MODEL_LABELS[mt]} |"
            for var in DATASET_VARIANTS:
                key = f"{mt}_{res}_{var}"
                if key in summary:
                    s = summary[key]
                    row += f" {s['acc_mean']:.2f} +- {s['acc_std']:.2f} |"
                else:
                    row += " - |"
            lines.append(row)

        lines.append("\n### Macro F1 (mean +- std)")
        lines.append("| Model | Test Original | Aug (Foreground) | Aug (Full) |")
        lines.append("|-------|:---:|:---:|:---:|")
        for mt in MODEL_TYPES:
            row = f"| {MODEL_LABELS[mt]} |"
            for var in DATASET_VARIANTS:
                key = f"{mt}_{res}_{var}"
                if key in summary:
                    s = summary[key]
                    row += f" {s['f1_mean']:.2f} +- {s['f1_std']:.2f} |"
                else:
                    row += " - |"
            lines.append(row)

        lines.append("\n### Delta Accuracy (mean +- std)")
        lines.append("| Model | Delta Aug (FG) | Delta Aug (Full) | Avg Delta |")
        lines.append("|-------|:---:|:---:|:---:|")
        for mt in MODEL_TYPES:
            row = f"| {MODEL_LABELS[mt]} |"
            deltas = []
            for var in ["datasets_aug", "datasets_aug2"]:
                key = f"{mt}_{res}_{var}"
                if key in summary and "delta_mean" in summary[key]:
                    d = summary[key]
                    row += f" {d['delta_mean']:+.2f} +- {d['delta_std']:.2f} |"
                    deltas.append(d["delta_mean"])
                else:
                    row += " - |"
            avg_d = np.mean(deltas) if deltas else 0
            row += f" {avg_d:+.2f} |"
            lines.append(row)

    lines.append("\n## 통계 검정 (Paired t-test, Macro F1)\n")
    lines.append("| Resolution | Dataset | Model A | Model B | "
                 "t-stat | p-value | Significant |")
    lines.append("|:---:|---------|---------|---------|:---:|:---:|:---:|")
    for t in tests:
        sig = "Yes" if t["significant"] else "No"
        lines.append(
            f"| {t['resolution']} | {DATASET_LABELS.get(t['dataset'], t['dataset'])} | "
            f"{MODEL_LABELS.get(t['model_a'], t['model_a'])} | "
            f"{MODEL_LABELS.get(t['model_b'], t['model_b'])} | "
            f"{t['t_statistic']:.3f} | {t['p_value']:.4f} | {sig} |"
        )

    return "\n".join(lines)


def main():
    mode_label = "(w/o Aux)" if NO_AUX_MODE else "(w/ Aux)"
    print("=" * 70)
    print(f"결과 집계 시작 {mode_label}")
    print("=" * 70)

    summary_dir = os.path.join(PROJECT_DIR, "artifacts", SUMMARY_SUBDIR)
    os.makedirs(summary_dir, exist_ok=True)

    results = collect_results()
    if not results:
        print("[오류] 수집된 결과가 없습니다. 먼저 학습/평가를 실행하세요.")
        return

    print(f"수집된 실험 결과: {len(results)}개")

    summary = compute_stats(results)
    tests = paired_tests(results)

    with open(os.path.join(summary_dir, "results_summary.json"), 'w') as f:
        json.dump({"summary": summary, "paired_tests": tests},
                  f, indent=2, ensure_ascii=False)
    print("결과 요약 JSON 저장 완료")

    # 시각화
    for res in RESOLUTIONS:
        has_data = any(f"{mt}_{res}_original" in summary
                       for mt in MODEL_TYPES)
        if has_data:
            plot_learning_curves(res)
            plot_robustness_bars(summary, res)

    if any(f"{mt}_224_original" in summary for mt in MODEL_TYPES) and \
       any(f"{mt}_448_original" in summary for mt in MODEL_TYPES):
        plot_resolution_comparison(summary)

    md_content = generate_markdown_summary(summary, tests)
    md_path = os.path.join(summary_dir, "results_summary.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"결과 요약 Markdown 저장: {md_path}")

    # 콘솔 출력
    print("\n" + md_content)
    print("\n" + "=" * 70)
    print("결과 집계 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
