# Combine repeated training runs (python enhanced_train.py --seed N) into mean ± std test metrics

from pathlib import Path

import pandas as pd

import enhanced_config as config

METRICS = ["test_accuracy", "test_f1", "test_precision", "test_recall", "test_auc"]


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


if __name__ == "__main__":
    main()
