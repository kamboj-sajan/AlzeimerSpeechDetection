# Combine repeated training runs (python enhanced_train.py --seed N) into mean ± std test metrics

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import enhanced_config as config

METRICS = ["test_accuracy", "test_f1", "test_precision", "test_recall", "test_auc"]
METRIC_LABELS = {"test_accuracy": "Accuracy", "test_f1": "F1", "test_precision": "Precision",
                 "test_recall": "Recall", "test_auc": "ROC-AUC"}
# Light and dark variants so the chart reads well on either GitHub theme
THEMES = {
    "seed_metrics.png": {"surface": "#fcfcfb", "ink": "#0b0b0b", "secondary": "#52514e",
                         "muted": "#898781", "grid": "#e1e0d9", "dot": "#2a78d6"},
    "seed_metrics_dark.png": {"surface": "#1a1a19", "ink": "#f0efec", "secondary": "#c3c2b7",
                              "muted": "#898781", "grid": "#2c2c2a", "dot": "#3987e5"},
}


def plot_seed_metrics(runs, out_dir):
    """Dot plot: one row per metric, one dot per seed, a bar at the mean."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Small fixed vertical offsets keep seeds with identical scores from hiding each other
    offsets = [(i - (len(runs) - 1) / 2) * 0.16 for i in range(len(runs))]
    rows = METRICS[::-1]  # first metric at the top

    for filename, theme in THEMES.items():
        fig, ax = plt.subplots(figsize=(8, 4.4), dpi=200)
        fig.patch.set_facecolor(theme["surface"])
        ax.set_facecolor(theme["surface"])
        for y, metric in enumerate(rows):
            values = runs[metric]
            ax.scatter(values, [y + o for o in offsets], s=55, color=theme["dot"],
                       edgecolors=theme["surface"], linewidths=1.5, zorder=3)
            # Mean bar sits behind the dots; their surface ring keeps both visible where they meet
            ax.plot([values.mean()] * 2, [y - 0.36, y + 0.36], color=theme["ink"], linewidth=2, zorder=2)
            label = f"{values.mean():.2f}" + (f" ± {values.std():.2f}" if len(values) > 1 else "")
            ax.text(1.03, y, label, transform=ax.get_yaxis_transform(), va="center",
                    color=theme["ink"], fontsize=9.5)
        ax.text(1.03, len(rows) - 0.4, "mean ± std", transform=ax.get_yaxis_transform(), va="center",
                color=theme["muted"], fontsize=8.5)

        ax.set_xlim(0.6, 1.02)
        ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([METRIC_LABELS[m] for m in rows], color=theme["secondary"], fontsize=10)
        ax.tick_params(axis="x", colors=theme["muted"], labelsize=9, length=0)
        ax.tick_params(axis="y", length=0, pad=8)
        ax.grid(axis="x", color=theme["grid"], linewidth=0.8, zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        fig.text(0.02, 0.98, f"Test-set scores across {len(runs)} random seeds", color=theme["ink"],
                 fontsize=12.5, fontweight="semibold", va="top")
        fig.text(0.02, 0.915, "Each dot is one seed's run; the vertical bar marks the mean.",
                 color=theme["secondary"], fontsize=9.5, va="top")
        fig.subplots_adjust(left=0.12, right=0.84, top=0.82, bottom=0.07)
        fig.savefig(out_dir / filename, facecolor=theme["surface"])
        plt.close(fig)
    print(f"Charts saved to {out_dir / 'seed_metrics.png'} (and the _dark variant)")


def main():
    run_files = sorted(Path(config.RESULTS_DIR, "seed_runs").glob("seed_*.csv"))
    if not run_files:
        print("No seed runs found. Train first with: python enhanced_train.py --seed <n>")
        return

    rows = []
    for path in run_files:
        df = pd.read_csv(path)
        # One row per seed; for K-fold CV that is the seed's mean over folds
        row = df[METRICS + ["best_epoch"]].mean()
        row["seed"] = int(path.stem.split("_")[1])
        rows.append(row)
    runs = pd.DataFrame(rows).set_index("seed").sort_index()
    runs.index = runs.index.astype(int)

    print("Per-seed test results:")
    print(runs.round(4).to_string())
    print(f"\nAcross {len(runs)} seeds (mean ± sample std):")
    for metric in METRICS:
        values = runs[metric]
        print(f"{metric:<16} {values.mean():.4f} ± {values.std():.4f}  "
              f"(min {values.min():.4f}, max {values.max():.4f})")

    summary_path = Path(config.RESULTS_DIR) / "seed_summary.csv"
    runs.to_csv(summary_path)
    print(f"\nPer-seed table saved to {summary_path}")
    plot_seed_metrics(runs, config.PLOTS_DIR)


if __name__ == "__main__":
    main()
