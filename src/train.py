"""
Run the full pipeline from the command line.

Usage:
    python src/train.py --data data/Friday-WorkingHours-Morning.pcap_ISCX.csv
    python src/train.py --data data/            # a folder of CIC-IDS2017 CSVs
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # save plots to files, no pop-up windows
sys.path.insert(0, str(Path(__file__).parent))

import botnet_detection as bd  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="AI-Based Botnet Traffic Detection")
    parser.add_argument("--data", required=True, help="CSV file or folder of CSV files")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    print("1/6 Loading data ...")
    df, files = bd.load_dataset(args.data)

    print("2/6 Cleaning data ...")
    X, y, report = bd.clean_dataset(df)
    for k, v in report.items():
        print(f"    {k}: {v}")
    if report["class_counts"]["bot"] < 2:
        sys.exit("Not enough 'Bot' rows found. Use the Friday-WorkingHours-Morning CSV.")

    print("3/6 Splitting data ...")
    X_train, X_test, y_train, y_test = bd.split_data(X, y, args.test_size)
    print(f"    train: {len(X_train):,} rows | test: {len(X_test):,} rows")

    print("4/6 Training models ...")
    models = bd.train_models(bd.build_models(), X_train, y_train)

    print("5/6 Evaluating ...")
    metrics = bd.evaluate_all(models, X_test, y_test)
    print(metrics.round(4).to_string())

    print("6/6 Saving plots, metrics and model ...")
    out = args.results_dir
    bd.plot_class_distribution(y, out)
    bd.plot_confusion_matrices(models, X_test, y_test, out)
    bd.plot_metric_comparison(metrics, out)
    bd.plot_feature_importance(models["Random Forest"], list(X.columns), out_dir=out)
    bd.plot_pr_curves(models, X_test, y_test, out)
    bd.save_artifacts(
        models,
        metrics,
        bd.text_reports(models, X_test, y_test),
        report,
        files,
        list(X.columns),
        results_dir=out,
        models_dir=args.models_dir,
        test_size=args.test_size,
    )
    print(f"Done. See the '{out}/' and '{args.models_dir}/' folders.")


if __name__ == "__main__":
    main()
