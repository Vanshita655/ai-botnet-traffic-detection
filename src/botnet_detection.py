"""
AI-Based Botnet Traffic Detection - core pipeline
Computer Networks mini project, Karnavati University (UIT)

Pipeline:  load -> clean -> split -> train -> evaluate -> visualise -> save

Designed for the CIC-IDS2017 "MachineLearningCVE" CSV files, where botnet
traffic is labelled "Bot" and normal traffic is labelled "BENIGN".
"""

from __future__ import annotations

import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
BENIGN_LABEL = "BENIGN"
BOT_LABEL = "Bot"


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
def load_dataset(path: str | Path) -> tuple[pd.DataFrame, list[str]]:
    """Load one CSV file, or every CSV inside a folder, into one DataFrame."""
    path = Path(path)
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No CSV files found at {path}")

    frames = []
    for f in files:
        df = pd.read_csv(f, low_memory=False, encoding="latin-1")
        df.columns = df.columns.str.strip()  # CIC-IDS2017 headers have leading spaces
        # the original CSVs contain "Fwd Header Length" twice -> pandas names the copy ".1"
        df = df.loc[:, ~df.columns.str.contains(r"\.\d+$")]
        frames.append(df)
    return pd.concat(frames, ignore_index=True), [f.name for f in files]


# --------------------------------------------------------------------------
# 2. Clean
# --------------------------------------------------------------------------
def clean_dataset(
    df: pd.DataFrame, label_col: str = "Label"
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Returns (X, y, report).
      y = 1 for botnet traffic ("Bot"), 0 for normal traffic ("BENIGN").
    Rows belonging to other attack types are dropped so the task stays
    'botnet vs normal'.
    """
    report = {"rows_loaded": int(len(df))}

    df = df.copy()
    df[label_col] = df[label_col].astype(str).str.strip()

    # keep only normal + botnet flows
    df = df[df[label_col].str.lower().isin([BENIGN_LABEL.lower(), BOT_LABEL.lower()])]
    report["rows_after_label_filter"] = int(len(df))

    # CIC-IDS2017 has +/-inf (e.g. Flow Bytes/s when duration = 0) and NaN values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    report["rows_after_dropping_inf_nan"] = int(len(df))

    # remove exact duplicates BEFORE splitting -> prevents train/test leakage
    df = df.drop_duplicates()
    report["rows_after_dropping_duplicates"] = int(len(df))

    y = (df[label_col].str.lower() == BOT_LABEL.lower()).astype(int)
    X = df.drop(columns=[label_col]).select_dtypes(include=[np.number])

    constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
    X = X.drop(columns=constant_cols)
    report["constant_columns_dropped"] = constant_cols
    report["n_features"] = int(X.shape[1])
    report["class_counts"] = {"normal": int((y == 0).sum()), "bot": int((y == 1).sum())}
    return X.reset_index(drop=True), y.reset_index(drop=True), report


# --------------------------------------------------------------------------
# 3. Split
# --------------------------------------------------------------------------
def split_data(X, y, test_size: float = 0.3):
    """Stratified split so train and test keep the same bot/normal ratio."""
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )


# --------------------------------------------------------------------------
# 4. Models
# --------------------------------------------------------------------------
def build_models() -> dict:
    """
    Botnet flows are rare, so class_weight='balanced' makes the models pay more
    attention to the minority (Bot) class.
    """
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=15, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
            ),
        ),
    }


def train_models(models: dict, X_train, y_train) -> dict:
    for name, model in models.items():
        print(f"Training {name} ...")
        model.fit(X_train, y_train)
    return models


# --------------------------------------------------------------------------
# 5. Evaluate
# --------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    result = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        result["roc_auc"] = roc_auc_score(y_test, proba)
        result["pr_auc"] = average_precision_score(y_test, proba)
    return result


def evaluate_all(models: dict, X_test, y_test) -> pd.DataFrame:
    rows = {name: evaluate_model(m, X_test, y_test) for name, m in models.items()}
    return pd.DataFrame(rows).T


def text_reports(models: dict, X_test, y_test) -> str:
    out = []
    for name, m in models.items():
        out.append(f"===== {name} =====")
        out.append(
            classification_report(
                y_test, m.predict(X_test), target_names=["Normal", "Bot"], digits=4
            )
        )
    return "\n".join(out)


# --------------------------------------------------------------------------
# 6. Visualise
# --------------------------------------------------------------------------
def plot_class_distribution(y, out_dir=None):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = y.map({0: "Normal", 1: "Bot"}).value_counts()
    ax.bar(counts.index, counts.values, color=["#1f4e79", "#b22222"][: len(counts)])
    ax.set_yscale("log")
    ax.set_ylabel("Number of flows (log scale)")
    ax.set_title("Class distribution (after cleaning)")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom")
    return _save(fig, out_dir, "class_distribution.png")


def plot_confusion_matrices(models: dict, X_test, y_test, out_dir=None):
    for name, model in models.items():
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ConfusionMatrixDisplay.from_estimator(
            model,
            X_test,
            y_test,
            display_labels=["Normal", "Bot"],
            cmap="Blues",
            values_format="d",
            ax=ax,
        )
        ax.set_title(f"Confusion matrix - {name}")
        _save(fig, out_dir, f"confusion_matrix_{name.lower().replace(' ', '_')}.png")


def plot_metric_comparison(metrics: pd.DataFrame, out_dir=None):
    cols = ["precision", "recall", "f1", "accuracy"]
    ax = metrics[cols].astype(float).plot(kind="bar", figsize=(8, 4.5), rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison (test set)")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    return _save(ax.figure, out_dir, "model_comparison.png")


def plot_feature_importance(model, feature_names, top_n: int = 15, out_dir=None):
    if not hasattr(model, "feature_importances_"):
        return None
    imp = (
        pd.Series(model.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
        .head(top_n)[::-1]
    )
    fig, ax = plt.subplots(figsize=(7, 5.5))
    imp.plot(kind="barh", ax=ax, color="#1f4e79")
    ax.set_title(f"Top {top_n} features - Random Forest")
    ax.set_xlabel("Importance")
    return _save(fig, out_dir, "feature_importance.png")


def plot_pr_curves(models: dict, X_test, y_test, out_dir=None):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            PrecisionRecallDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
    ax.set_title("Precision-Recall curves (Bot class)")
    return _save(fig, out_dir, "precision_recall_curves.png")


def _save(fig, out_dir, filename):
    fig.tight_layout()
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / filename, dpi=200, bbox_inches="tight")
    return fig


# --------------------------------------------------------------------------
# 7. Save + reproducibility record
# --------------------------------------------------------------------------
def save_artifacts(
    models: dict,
    metrics: pd.DataFrame,
    reports_text: str,
    clean_report: dict,
    files_used: list[str],
    feature_names: list[str],
    results_dir="results",
    models_dir="models",
    test_size: float = 0.3,
):
    results_dir, models_dir = Path(results_dir), Path(models_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics.round(4).to_csv(results_dir / "metrics.csv")
    (results_dir / "classification_reports.txt").write_text(reports_text)

    joblib.dump(models["Random Forest"], models_dir / "random_forest.joblib")
    (models_dir / "features.json").write_text(json.dumps(feature_names, indent=2))

    config = {
        "dataset_files": files_used,
        "cleaning": clean_report,
        "split": {"test_size": test_size, "stratified": True, "random_state": RANDOM_STATE},
        "model_settings": {n: str(m) for n, m in models.items()},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    (results_dir / "run_config.json").write_text(json.dumps(config, indent=2))


# --------------------------------------------------------------------------
# 8. Prediction helper (used by the demo app)
# --------------------------------------------------------------------------
def predict_flows(model, feature_names: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """Return the input flows with a 'Prediction' column: Normal / Suspicious."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Input is missing {len(missing)} required columns, e.g. {missing[:5]}")
    X = df[feature_names].replace([np.inf, -np.inf], np.nan).fillna(0)
    pred = model.predict(X)
    out = df.copy()
    out["Bot_probability"] = model.predict_proba(X)[:, 1].round(4)
    out["Prediction"] = np.where(pred == 1, "Suspicious", "Normal")
    return out
