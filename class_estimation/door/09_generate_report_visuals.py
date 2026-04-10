"""report 폴더의 eval_results.json 데이터로 혼동 행렬 PNG + 클래스별 통계 생성
- 각 seed × 데이터셋별 혼동 행렬 PNG
- 클래스별 mean ± std 통계
- summary.md 업데이트
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(PROJECT_DIR, "artifacts", "report")
SUMMARY_JSON = os.path.join(REPORT_DIR, "summary.json")

DATASET_LABELS = {
    "original": "Test Original",
    "datasets_aug": "Augmented (Foreground)",
    "datasets_aug2": "Augmented (Full)",
}

SHORT_CLASS_NAMES = {
    "E25_door_LH_FRT": "E25\nLH_FRT",
    "E25_door_LH_RR": "E25\nLH_RR",
    "E25_door_RH": "E25\nRH",
    "E30_E38_door_RH": "E3x\nRH",
    "E30_door_LH_FRT": "E30\nLH_FRT",
    "E30_door_LH_RR": "E30\nLH_RR",
    "E38_door_LH_FRT": "E38\nLH_FRT",
    "E38_door_LH_RR": "E38\nLH_RR",
}


def plot_confusion_matrix(cm, class_names, title, save_path):
    """혼동 행렬 시각화 (door_paper 스타일)"""
    n = len(class_names)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_pct = cm / row_sums * 100

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm_pct, interpolation='nearest', cmap='Blues',
                   vmin=0, vmax=100)

    tick_labels = [SHORT_CLASS_NAMES.get(c, c) for c in class_names]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_yticklabels(tick_labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            pct = cm_pct[i, j]
            color = "white" if pct > 50 else "black"
            ax.text(j, i, f"{val}\n({pct:.0f}%)",
                    ha="center", va="center", color=color, fontsize=9)

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('%', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_confusion_matrices(data):
    """seed별 × 데이터셋별 혼동 행렬 PNG 생성"""
    raw = data["raw_results"]
    count = 0
    for seed, datasets in raw.items():
        seed_dir = os.path.join(REPORT_DIR, f"rgbd_448_seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)

        for ds_name, result in datasets.items():
            cm = np.array(result["confusion_matrix"])
            class_names = list(result["per_class"].keys())
            title = f"RGBD 448 seed{seed}\n{DATASET_LABELS.get(ds_name, ds_name)}"
            save_path = os.path.join(seed_dir, f"confusion_matrix_{ds_name}.png")
            plot_confusion_matrix(cm, class_names, title, save_path)
            count += 1
            print(f"  [PNG] {save_path}")
    return count


def compute_per_class_stats(data):
    """클래스별 mean ± std 계산 (모든 데이터셋 × 모든 지표)"""
    raw = data["raw_results"]
    seeds = data["seeds"]

    first_seed = str(seeds[0])
    first_ds = list(raw[first_seed].keys())[0]
    class_names = list(raw[first_seed][first_ds]["per_class"].keys())

    stats = {}
    for ds_name in raw[first_seed].keys():
        stats[ds_name] = {}
        for cls in class_names:
            prec_vals, rec_vals, f1_vals = [], [], []
            for seed in seeds:
                pc = raw[str(seed)][ds_name]["per_class"].get(cls, {})
                prec_vals.append(pc.get("precision", 0))
                rec_vals.append(pc.get("recall", 0))
                f1_vals.append(pc.get("f1", 0))

            stats[ds_name][cls] = {
                "precision_mean": round(np.mean(prec_vals), 2),
                "precision_std": round(np.std(prec_vals), 2),
                "recall_mean": round(np.mean(rec_vals), 2),
                "recall_std": round(np.std(rec_vals), 2),
                "f1_mean": round(np.mean(f1_vals), 2),
                "f1_std": round(np.std(f1_vals), 2),
            }
    return stats, class_names


def update_summary_md(data, per_class_stats, class_names):
    """summary.md에 클래스별 상세 통계 추가"""
    summary = data["summary"]
    raw = data["raw_results"]
    seeds = data["seeds"]

    lines = []
    lines.append("# 과제 성능 보고서 - RGBD 분류 결과\n")
    lines.append(f"- **모델**: RGBDAuxResNet18 (RGBD 4채널 + Aux)")
    lines.append(f"- **해상도**: 448x448")
    lines.append(f"- **분할**: Train 70% / Test 30%")
    lines.append(f"- **Seeds**: {seeds}")
    lines.append(f"- **완료**: {len(seeds)}개 seed\n")

    # 종합 결과
    lines.append("## 종합 결과 (mean ± std)\n")
    lines.append("| 데이터셋 | Accuracy (%) | Macro F1 (%) | Macro Precision (%) | Macro Recall (%) |")
    lines.append("|----------|:---:|:---:|:---:|:---:|")
    ds_label = {"original": "원본 (Test)", "datasets_aug": "증강1 (Foreground)",
                "datasets_aug2": "증강2 (Full)"}
    for ds_name in ["original", "datasets_aug", "datasets_aug2"]:
        s = summary[ds_name]
        lines.append(f"| {ds_label[ds_name]} | "
                     f"{s['acc_mean']:.2f} ± {s['acc_std']:.2f} | "
                     f"{s['f1_mean']:.2f} ± {s['f1_std']:.2f} | "
                     f"{s['prec_mean']:.2f} ± {s['prec_std']:.2f} | "
                     f"{s['rec_mean']:.2f} ± {s['rec_std']:.2f} |")
    lines.append("")

    # Seed별 상세 결과
    lines.append("## Seed별 상세 결과\n")
    lines.append("| Seed | 데이터셋 | Accuracy | Macro F1 | Macro Precision | Macro Recall |")
    lines.append("|:---:|----------|:---:|:---:|:---:|:---:|")
    for seed in seeds:
        for ds_name in ["original", "datasets_aug", "datasets_aug2"]:
            r = raw[str(seed)][ds_name]
            lines.append(f"| {seed} | {ds_label[ds_name]} | "
                         f"{r['accuracy']:.2f} | {r['macro_f1']:.2f} | "
                         f"{r['macro_precision']:.2f} | {r['macro_recall']:.2f} |")
    lines.append("")

    # 클래스별 상세 통계
    lines.append("## 클래스별 상세 통계 (mean ± std, 5 seeds)\n")
    for ds_name in ["original", "datasets_aug", "datasets_aug2"]:
        lines.append(f"### {ds_label[ds_name]}\n")
        lines.append("| 클래스 | Precision (%) | Recall (%) | F1 (%) |")
        lines.append("|--------|:---:|:---:|:---:|")
        for cls in class_names:
            s = per_class_stats[ds_name][cls]
            short = cls.replace("_door_", " ").replace("_", " ")
            lines.append(f"| {short} | "
                         f"{s['precision_mean']:.2f} ± {s['precision_std']:.2f} | "
                         f"{s['recall_mean']:.2f} ± {s['recall_std']:.2f} | "
                         f"{s['f1_mean']:.2f} ± {s['f1_std']:.2f} |")
        lines.append("")

    # 혼동 행렬 링크
    lines.append("## 혼동 행렬\n")
    lines.append("각 seed × 데이터셋 별 혼동 행렬 PNG:\n")
    for seed in seeds:
        lines.append(f"### Seed {seed}\n")
        for ds_name in ["original", "datasets_aug", "datasets_aug2"]:
            fname = f"confusion_matrix_{ds_name}.png"
            lines.append(f"- [{ds_label[ds_name]}](rgbd_448_seed{seed}/{fname})")
        lines.append("")

    md_path = os.path.join(REPORT_DIR, "summary.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  [MD] {md_path}")


def save_per_class_json(per_class_stats):
    """클래스별 통계를 별도 JSON으로 저장"""
    path = os.path.join(REPORT_DIR, "per_class_stats.json")
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(per_class_stats, f, ensure_ascii=False, indent=2)
    print(f"  [JSON] {path}")


def main():
    print("=" * 60)
    print("보고서 시각화 생성")
    print("=" * 60)

    with open(SUMMARY_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"\n1. 혼동 행렬 PNG 생성...")
    n_png = generate_confusion_matrices(data)
    print(f"   → {n_png}개 PNG 생성 완료")

    print(f"\n2. 클래스별 통계 계산...")
    per_class_stats, class_names = compute_per_class_stats(data)
    save_per_class_json(per_class_stats)

    print(f"\n3. summary.md 업데이트...")
    update_summary_md(data, per_class_stats, class_names)

    print(f"\n완료!")


if __name__ == "__main__":
    main()
