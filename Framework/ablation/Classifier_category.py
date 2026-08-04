from __future__ import annotations
import argparse
import os
import pathlib
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    log_loss,
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except:
    XGBClassifier = None

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def ensure_dir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def sanitize(name):
    return name.replace(" ", "_").replace("(", "").replace(")", "")


# --------------------------------------------------
# Argument Parser
# --------------------------------------------------
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv", required=True)
#     parser.add_argument("--label-col", default="hal_label")
#     parser.add_argument("--type-col", default="type")
#     parser.add_argument("--test-type", default="sentiment")
#     parser.add_argument("--output-dir", required=True)
#     parser.add_argument("--random-state", type=int, default=42)
#     return parser.parse_args()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv1", required=True, help="First CSV file")
    parser.add_argument("--csv2", required=True, help="Second CSV file")
    parser.add_argument("--label-col", default="hal_label")
    parser.add_argument("--type-col", default="type")
    parser.add_argument("--test-type", default="discriminative-attribute-state")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


# --------------------------------------------------
# Custom Split Logic
# --------------------------------------------------
def split_train_test(df, label_col, type_col, test_type, random_state):

    df = df.dropna().reset_index(drop=True)

    attribute_df = df[df[type_col] == test_type]
    other_df = df[df[type_col] != test_type]

    if len(attribute_df) == 0:
        raise ValueError("No rows found for test type")

    attribute_df = attribute_df.sample(frac=1.0, random_state=random_state)

    half = len(attribute_df) // 4
    test_df = attribute_df.iloc[:half]
    train_attr = attribute_df.iloc[half:]

    train_df = pd.concat([other_df, train_attr], ignore_index=False)

    # Strict leakage check
    overlap = set(test_df.index).intersection(set(train_df.index))
    if overlap:
        raise RuntimeError("Data leakage detected!")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [[c for c in numeric_cols if c != label_col and c not in {"id", "baseline"}][0]]
    print(feature_cols)
    # exit()
    return (
        train_df[feature_cols],
        train_df[label_col],
        test_df[feature_cols],
        test_df[label_col],
        feature_cols,
    )


# --------------------------------------------------
# Model Catalog
# --------------------------------------------------
def get_models(random_state):

    models = {

        "Logistic Regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))
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
                ("clf", SVC(probability=True))
            ]),
            {"clf__C": [0.1, 1, 10]}
        ),

        "Decision Tree": (
            DecisionTreeClassifier(random_state=random_state),
            {"max_depth": [None, 5, 10, 20]}
        ),

        "Random Forest": (
            RandomForestClassifier(random_state=random_state, class_weight="balanced"),
            {"n_estimators": [100, 200]}
        ),

        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=random_state),
            {"n_estimators": [100, 200]}
        ),

        "AdaBoost": (
            AdaBoostClassifier(random_state=random_state),
            {"n_estimators": [50, 100, 200]}
        ),
    }

    if XGBClassifier:
        models["XGBoost"] = (
            XGBClassifier(
                eval_metric="logloss",
                use_label_encoder=False,
                random_state=random_state
            ),
            {"n_estimators": [100, 200]}
        )

    return models


# --------------------------------------------------
# Plot Functions
# --------------------------------------------------
def plot_roc(y_true, y_scores, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.savefig(save_path)
    plt.close()


def plot_pr(y_true, y_scores, save_path):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure()
    plt.plot(recall, precision)
    plt.title("Precision-Recall Curve")
    plt.savefig(save_path)
    plt.close()


# --------------------------------------------------
# Training & Evaluation
# --------------------------------------------------
def train_and_evaluate(X_train, y_train, X_test, y_test, output_dir, random_state):

    models = get_models(random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    results = {}

    for name, (estimator, param_grid) in models.items():

        logger.info(f"Training {name}")

        grid = GridSearchCV(
            estimator,
            param_grid,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        model_dir = os.path.join(output_dir, sanitize(name))
        ensure_dir(model_dir)

        joblib.dump(best_model, os.path.join(model_dir, "best_model.joblib"))

        y_pred = best_model.predict(X_test)

        if hasattr(best_model, "predict_proba"):
            y_scores = best_model.predict_proba(X_test)[:, 1]
        else:
            raw = best_model.decision_function(X_test)
            y_scores = (raw - raw.min()) / (raw.max() - raw.min())

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_scores)
        pr_auc = average_precision_score(y_test, y_scores)
        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        ll = log_loss(y_test, y_scores)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)

        plot_roc(y_test, y_scores, os.path.join(model_dir, "roc.png"))
        plot_pr(y_test, y_scores, os.path.join(model_dir, "pr.png"))

        pd.DataFrame(confusion_matrix(y_test, y_pred)).to_csv(
            os.path.join(model_dir, "confusion_matrix.csv")
        )

        pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose().to_csv(
            os.path.join(model_dir, "classification_report.csv")
        )

        results[name] = {
            "accuracy": acc,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }

    summary = pd.DataFrame(results).T.sort_values("accuracy", ascending=False)
    summary.to_csv(os.path.join(output_dir, "model_summary.csv"))

    return summary


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    args = parse_args()
    ensure_dir(args.output_dir)

    # Read both CSV files
    df1 = pd.read_csv(args.csv1)
    df2 = pd.read_csv(args.csv2)

    # Optional: validate same columns
    if set(df1.columns) != set(df2.columns):
        raise ValueError("CSV files must have identical columns to aggregate.")

    # Aggregate (row-wise concatenation)
    df = pd.concat([df1, df2], ignore_index=True)

    logger.info(f"Aggregated dataset shape: {df.shape}")

    X_train, y_train, X_test, y_test, _ = split_train_test(
        df,
        args.label_col,
        args.type_col,
        args.test_type,
        args.random_state
    )

    summary = train_and_evaluate(
        X_train,
        y_train,
        X_test,
        y_test,
        args.output_dir,
        args.random_state
    )

    logger.info("\nTop Models:\n%s", summary.head())


if __name__ == "__main__":
    main()
