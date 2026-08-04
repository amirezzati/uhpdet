from __future__ import annotations
import argparse
import json
import logging
import os
import pathlib
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

warnings.filterwarnings("ignore")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Optional models
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


# ----------------------------------------------------
# Utilities
# ----------------------------------------------------
def ensure_dir_exists(path: str):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def sanitize_filename(s: str):
    return s.replace(" ", "_").replace("(", "").replace(")", "")


# ----------------------------------------------------
# Argument Parser
# ----------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--label-col", default="hal_label")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


# ----------------------------------------------------
# Data Loading
# ----------------------------------------------------
def load_data(train_path, test_path, label_col, random_state):

    train_df = pd.read_csv(train_path).dropna()
    test_df = pd.read_csv(test_path).dropna()

    # Force exactly 1000 test samples
    if len(test_df) < 1000:
        raise ValueError(f"Test set must contain at least 1000 rows (found {len(test_df)})")

    if len(test_df) > 1000:
        test_df = test_df.sample(n=1000, random_state=random_state)
        logger.info("Test set reduced to exactly 1000 samples.")

    if label_col not in train_df.columns:
        raise KeyError(f"Label column '{label_col}' not found in training CSV")

    # Numeric features only
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != label_col and c not in {"id", "baseline"}]

    X_train = train_df[feature_cols]
    y_train = train_df[label_col]

    X_test = test_df[feature_cols]
    y_test = test_df[label_col]

    logger.info("Train shape: %s | Test shape: %s", X_train.shape, X_test.shape)

    return X_train, y_train, X_test, y_test, feature_cols


# ----------------------------------------------------
# Plot Functions
# ----------------------------------------------------
def plot_learning_curve_fn(estimator, X, y, model_name, cv, save_path):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, scoring="accuracy", n_jobs=-1
    )

    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1))
    plt.plot(train_sizes, np.mean(val_scores, axis=1))
    plt.title(f"Learning Curve - {model_name}")
    plt.xlabel("Training Samples")
    plt.ylabel("Accuracy")
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve_fn(y_true, y_scores, model_name, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_pr_curve_fn(y_true, y_scores, model_name, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    plt.figure()
    plt.plot(recall, precision, label=f"AP={ap:.4f}")
    plt.title(f"Precision-Recall - {model_name}")
    plt.legend()
    plt.savefig(save_path)
    plt.close()


# ----------------------------------------------------
# Model Catalog
# ----------------------------------------------------
def build_model_catalog(random_state):

    models = {
        "Logistic Regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
            ]),
            {"clf__C": [0.01, 0.1, 1, 10]}
        ),
        "Ridge Classifier": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RidgeClassifier())
            ]),
            {"clf__alpha": [0.1, 1, 10]}
        ),
        "SVM (RBF)": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(probability=True, class_weight="balanced"))
            ]),
            {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", "auto"]}
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
            {"max_depth": [3, 5, 10, None]}
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=random_state, class_weight="balanced"),
            {"n_estimators": [100, 200], "max_depth": [5, 10, None]}
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=random_state),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=random_state),
            {"n_estimators": [50, 100, 200]}
        ),
    }

    if XGBClassifier:
        models["XGBoost"] = (
            XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state),
            {"n_estimators": [100, 200], "max_depth": [3, 5, 7]}
        )

    if LGBMClassifier:
        models["LightGBM"] = (
            LGBMClassifier(random_state=random_state),
            {"n_estimators": [100, 200]}
        )

    return models


# ----------------------------------------------------
# Training & Evaluation
# ----------------------------------------------------
def train_and_evaluate(X_train, y_train, X_test, y_test, feature_cols, output_dir, random_state):

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    models = build_model_catalog(random_state)

    results = {}

    for name, (estimator, param_grid) in models.items():

        logger.info("Training %s", name)

        grid = GridSearchCV(estimator, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        model_dir = os.path.join(output_dir, sanitize_filename(name))
        ensure_dir_exists(model_dir)

        # Save model
        joblib.dump(best_model, os.path.join(model_dir, "best_model.joblib"))

        # Learning curve
        plot_learning_curve_fn(best_model, X_train, y_train, name, cv,
                               os.path.join(model_dir, "learning_curve.png"))

        # Predictions
        y_pred = best_model.predict(X_test)

        if hasattr(best_model, "predict_proba"):
            y_scores = best_model.predict_proba(X_test)[:, 1]
        elif hasattr(best_model, "decision_function"):
            raw = best_model.decision_function(X_test)
            y_scores = (raw - raw.min()) / (raw.max() - raw.min())
        else:
            y_scores = None

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        roc_auc = roc_auc_score(y_test, y_scores) if y_scores is not None else None
        pr_auc = average_precision_score(y_test, y_scores) if y_scores is not None else None

        # Save plots
        if y_scores is not None:
            plot_roc_curve_fn(y_test, y_scores, name, os.path.join(model_dir, "roc.png"))
            plot_pr_curve_fn(y_test, y_scores, name, os.path.join(model_dir, "pr.png"))

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        pd.DataFrame(cm).to_csv(os.path.join(model_dir, "confusion_matrix.csv"))

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(os.path.join(model_dir, "classification_report.csv"))
        with open(os.path.join(model_dir, "classification_report.json"), "w") as f:
            json.dump(report, f, indent=2)

        results[name] = {
            "cv_auc": grid.best_score_,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "f1_weighted": f1_weighted,
            "auc_roc": roc_auc,
            "auc_pr": pr_auc
        }

    summary = pd.DataFrame(results).T.sort_values("f1_weighted", ascending=False)
    summary.to_csv(os.path.join(output_dir, "model_performance_summary.csv"))

    return summary


# ----------------------------------------------------
# Main
# ----------------------------------------------------
def main():
    args = parse_args()
    ensure_dir_exists(args.output_dir)

    X_train, y_train, X_test, y_test, feature_cols = load_data(
        args.train_csv,
        args.test_csv,
        args.label_col,
        args.random_state
    )

    summary = train_and_evaluate(
        X_train,
        y_train,
        X_test,
        y_test,
        feature_cols,
        args.output_dir,
        args.random_state
    )

    logger.info("\nTop Models:\n%s", summary.head())


if __name__ == "__main__":
    main()
