from __future__ import annotations
import argparse
import json
import logging
import os
import pathlib
import re
from typing import Dict, Iterable, List, Optional, Tuple
from sklearn.metrics import classification_report
import json
import warnings 

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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# optional libraries (XGBoost, LightGBM) — imported lazily inside function to avoid hard crash when missing
try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:
    XGBClassifier = None  # type: ignore

try:
    from lightgbm import LGBMClassifier  # type: ignore
except Exception:
    LGBMClassifier = None  # type: ignore


# ---------- Logging & plotting style ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
sns.set_style("whitegrid")


# ---------- Utilities ----------
def sanitize_filename(s: str) -> str:
    """Make a string safe for file names."""
    return re.sub(r"[^\w\-_\.]", "_", s)


def ensure_dir_exists(path: str) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


# ---------- Path helpers for feature-file extraction ----------
def feature_file_path_from_results_json(data_path: str) -> str:
    """
    Derive the feature CSV path from a results JSON filename following the pattern:
      results_full_{benchmark_name}_{model_name}.json

    Example:
      /.../results_full_AMBER_no_questions_blip.json
    yields
      <script_parent>/Results/Features/{model_name}/{benchmark_name}/data_features.csv

    If the parsed structure doesn't match, fallback to sibling data_features.csv in same dir.
    """
    data_path = os.path.abspath(data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"results JSON not found: {data_path}")

    file_name = os.path.splitext(os.path.basename(data_path))[0]
    # remove prefix
    prefix = "results_full_"
    if file_name.startswith(prefix):
        remainder = file_name[len(prefix) :]
        # attempt to split last underscore as model name
        if "_" in remainder:
            benchmark_name, model_name = remainder.rsplit("_", 1)
        else:
            # fallback
            benchmark_name = remainder
            model_name = "unknown_model"
    else:
        # fallback
        benchmark_name = file_name
        model_name = "unknown_model"

    # Determine script directory and craft default feature path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    feature_file_path = os.path.join(
        os.path.dirname(os.path.dirname(script_dir)),  # Framework/Ablation -> repo root
        "Results",
        "Features",
        model_name,
        benchmark_name,
        "data_features.csv",
    )

    # If that file exists, return it; otherwise try sibling in results dir or same dir as JSON
    if os.path.exists(feature_file_path):
        return feature_file_path

    # fallback candidates
    sibling_candidate = os.path.join(os.path.dirname(data_path), "data_features.csv")
    if os.path.exists(sibling_candidate):
        return sibling_candidate

    # final fallback: return crafted path (caller should check existence)
    return feature_file_path


# ---------- Plotting functions ----------
def plot_learning_curve_fn(estimator, X, y, model_name: str, cv, train_sizes: Iterable[float] = np.linspace(0.1, 1.0, 5), save_path: Optional[str] = None):
    plt.figure(figsize=(8, 6))
    plt.title(f"Learning Curve: {model_name}")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes_, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring="accuracy"
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.plot(train_sizes_, train_scores_mean, "o-", label="Train score")
    plt.plot(train_sizes_, test_scores_mean, "o-", label="Cross-val score")
    plt.legend(loc="best")
    plt.grid(True)
    if save_path:
        ensure_dir_exists(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Saved learning curve to %s", save_path)
    plt.show()


def plot_roc_curve_fn(y_true: np.ndarray, y_scores: np.ndarray, model_name: str, save_path: Optional[str] = None) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve: {model_name}")
    plt.legend(loc="lower right")
    if save_path:
        ensure_dir_exists(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Saved ROC curve to %s", save_path)
    plt.show()
    return roc_auc


def plot_precision_recall_fn(y_true: np.ndarray, y_scores: np.ndarray, model_name: str, save_path: Optional[str] = None) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)  # average precision (area under PR curve)
    # compute AUC(recall, precision) to show classical PR-AUC too
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(6, 6))
    plt.step(recall, precision, where="post", lw=2, label=f"{model_name} (AP={ap:.4f}, AUC={pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall: {model_name}")
    plt.legend(loc="lower left")
    if save_path:
        ensure_dir_exists(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Saved precision-recall curve to %s", save_path)
    plt.show()
    return ap


def plot_feature_importances_fn(model, feature_names: List[str], model_name: str, top_n: int = 15, save_path: Optional[str] = None):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        # handle linear models — might be shape (1, n_features)
        coef = model.coef_
        if coef.ndim > 1:
            coef = coef.ravel()
        importances = np.abs(coef)
    else:
        logger.debug("No feature importances for %s", model_name)
        return

    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
    plt.title(f"Feature Importances: {model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    if save_path:
        ensure_dir_exists(os.path.dirname(save_path))
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Saved feature importances to %s", save_path)
    plt.show()


# ---------- Model building & evaluation ----------
def build_model_catalog() -> Dict[str, Tuple[object, dict]]:
    """Create the set of estimators and parameter grids for GridSearchCV."""
    models: Dict[str, Tuple[object, dict]] = {
        "Logistic Regression": (
            Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
            {"clf__C": [0.01, 0.1, 1, 10], "clf__penalty": ["l2"]},
        ),
        "Ridge Classifier": (
            Pipeline([("scaler", StandardScaler()), ("clf", RidgeClassifier() )]),
            {"clf__alpha": [0.1, 1, 10]},
        ),
        "SVM (RBF)": (
            Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, class_weight="balanced"))]),
            {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", "auto"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42, class_weight="balanced"),
            {"max_depth": [3, 5, 10, None], "min_samples_split": [2, 5, 10], "ccp_alpha": [0.0, 0.01, 0.1]},
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42, class_weight="balanced"),
            {"n_estimators": [100, 200], "max_depth": [5, 10, None], "min_samples_split": [2, 5, 10], "max_features": ["sqrt", "log2"]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5], "subsample": [0.8, 1.0]},
        ),
        "AdaBoost": (
            AdaBoostClassifier(random_state=42),
            {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]},
        ),
    }

    # add XGBoost if available
    if XGBClassifier is not None:
        models["XGBoost"] = (
            XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
            {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5, 7], "subsample": [0.8, 1.0]},
        )
        models["XGBoost (Linear)"] = (
            XGBClassifier(booster="gblinear", use_label_encoder=False, eval_metric="logloss", random_state=42),
            {"learning_rate": [0.01, 0.05, 0.1], "reg_alpha": [0, 0.1, 1], "reg_lambda": [0, 0.1, 1]},
        )

    # optionally add LightGBM if available
    # if LGBMClassifier is not None:
    #     models["LightGBM"] = (
    #         LGBMClassifier(random_state=42),
    #         {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "num_leaves": [31, 63]},
    #     )

    return models


def evaluate_and_plot(
    best_model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: List[str],
    model_name: str,
    output_dir: str,
) -> Dict[str, Optional[float]]:
    """
    Produce predictions, compute metrics, plot ROC and PR curves, and feature importances.
    Returns a dict with evaluation metrics.
    """
    y_pred = best_model.predict(X_test)
    # try predict_proba first
    y_scores = None
    if hasattr(best_model, "predict_proba"):
        try:
            y_scores = best_model.predict_proba(X_test)[:, 1]
        except Exception:
            y_scores = None
    if y_scores is None:
        # fallback to decision_function if available (and scale to [0,1])
        if hasattr(best_model, "decision_function"):
            try:
                raw = best_model.decision_function(X_test)
                # scale to 0..1
                y_scores = (raw - raw.min()) / (raw.max() - raw.min())
            except Exception:
                y_scores = None

    # Compute metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1_weighted = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    roc_auc = None
    pr_ap = None
    if y_scores is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_scores)
        except Exception:
            roc_auc = None
        try:
            pr_ap = average_precision_score(y_test, y_scores)
        except Exception:
            pr_ap = None

    # --- Classification report ---
    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    # Pretty text version
    report_text = classification_report(
        y_test,
        y_pred,
        zero_division=0
    )

    # Log to console
    logger.info("Classification report for %s:\n%s", model_name, report_text)

    # Save paths
    txt_path = os.path.join(output_dir, "classification_report.txt")
    csv_path = os.path.join(output_dir, "classification_report.csv")
    json_path = os.path.join(output_dir, "classification_report.json")

    # Save TXT
    with open(txt_path, "w") as f:
        f.write(report_text)

    # Save CSV
    pd.DataFrame(report_dict).transpose().to_csv(csv_path)

    # Save JSON
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    logger.info("Saved classification report to %s", output_dir)

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=["neg", "pos"], columns=["pred_neg", "pred_pos"])
    cm_path = os.path.join(output_dir, f"{sanitize_filename(model_name)}_confusion_matrix.csv")
    ensure_dir_exists(os.path.dirname(cm_path))
    cm_df.to_csv(cm_path)
    logger.info("Saved confusion matrix to %s", cm_path)

    # Plots
    if y_scores is not None:
        roc_path = os.path.join(output_dir, f"{sanitize_filename(model_name)}_roc.png")
        pr_path = os.path.join(output_dir, f"{sanitize_filename(model_name)}_pr.png")
        plot_roc_curve_fn(y_test.values, y_scores, model_name, save_path=roc_path)
        plot_precision_recall_fn(y_test.values, y_scores, model_name, save_path=pr_path)
    else:
        logger.warning("No score/probabilities available for %s — ROC/PR plots skipped", model_name)

    # Feature importances
    feat_imp_path = os.path.join(output_dir, f"{sanitize_filename(model_name)}_feature_importances.png")
    try:
        # if model is a pipeline, extract the final estimator
        final_est = best_model
        if hasattr(best_model, "named_steps"):
            # attempt to find the last step name
            final_est = list(best_model.named_steps.values())[-1]
        plot_feature_importances_fn(final_est, feature_names, model_name, save_path=feat_imp_path)
    except Exception as e:
        logger.debug("Could not plot feature importances for %s: %s", model_name, str(e))

    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_weighted": float(f1_weighted),
        "f1": float(f1),
        "auc_roc": float(roc_auc) if roc_auc is not None else None,
        "auc_pr": float(pr_ap) if pr_ap is not None else None,
    }
    return metrics

def save_train_test_split(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: str,
) -> None:
    """
    Save train/test splits to CSV files (features + label).
    """
    ensure_dir_exists(output_dir)

    train_df = X_train.copy()
    train_df[y_train.name] = y_train.values

    test_df = X_test.copy()
    test_df[y_test.name] = y_test.values

    train_path = os.path.join(output_dir, "train_split.csv")
    test_path = os.path.join(output_dir, "test_split.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Saved train split to %s (shape=%s)", train_path, train_df.shape)
    logger.info("Saved test split to %s (shape=%s)", test_path, test_df.shape)



def build_and_evaluate(
    df: pd.DataFrame,
    label_col: str,
    feature_cols: List[str],
    output_dir: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Main pipeline: split, run GridSearchCV for multiple models, evaluate, save best models and metrics summary.
    Returns a pandas DataFrame with model metrics.
    """
    X = df[feature_cols]
    y = df[label_col]

    # ensure numeric features
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        logger.warning("Non-numeric feature columns detected; they will be dropped: %s", non_numeric)
        X = X.select_dtypes(include=[np.number])
        feature_cols = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=random_state, stratify=y)
    save_train_test_split(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test,
    output_dir=output_dir,
)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    models = build_model_catalog()

    results: Dict[str, dict] = {}
    for name, (estimator, param_grid) in models.items():
        logger.info("Training %s", name)
        # GridSearchCV
        grid = GridSearchCV(estimator, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0)
        try:
            grid.fit(X_train, y_train)
        except Exception as e:
            logger.exception("Grid search failed for %s: %s", name, str(e))
            continue

        best = grid.best_estimator_
        logger.info("Best params for %s: %s", name, grid.best_params_)

        # Learning curve (save)
        lc_path = os.path.join(output_dir, f"{sanitize_filename(name)}_learning_curve.png")
        try:
            plot_learning_curve_fn(best, X_train, y_train, name, cv=cv, save_path=lc_path)
        except Exception:
            logger.debug("Learning curve failed for %s", name)

        # Evaluate (compute metrics, plot ROC/PR, feature importances)
        model_output_dir = os.path.join(output_dir, sanitize_filename(name))
        ensure_dir_exists(model_output_dir)
        metrics = evaluate_and_plot(best, X_test, y_test, feature_cols, name, model_output_dir)

        # If XGBoost linear, print coefficients (where accessible)
        try:
            if name == "XGBoost (Linear)":
                final_est = best
                if hasattr(best, "named_steps"):
                    final_est = list(best.named_steps.values())[-1]
                if hasattr(final_est, "coef_"):
                    coef = np.ravel(final_est.coef_)
                    bias = float(final_est.intercept_[0]) if hasattr(final_est, "intercept_") else 0.0
                    logger.info("XGBoost (Linear) intercept: %f", bias)
                    for f, w in zip(feature_cols, coef):
                        logger.debug("coef %s: %f", f, w)
        except Exception:
            logger.debug("Could not extract linear coefficients for %s", name)

        # Save model
        model_fname = os.path.join(output_dir, f"best_model_{sanitize_filename(name)}.joblib")
        try:
            joblib.dump(best, model_fname)
            logger.info("Saved best model for %s to %s", name, model_fname)
        except Exception:
            logger.exception("Failed to save model for %s", name)

        # Save results
        results[name] = {
            "cv_auc": float(grid.best_score_) if grid.best_score_ is not None else None,
            **metrics,
        }

    # summary
    summary_df = pd.DataFrame(results).T
    
    # sort by weighted F1 first, fallback to AUCs
    if "f1_weighted" in summary_df.columns:
        sort_key = "f1_weighted"
    elif "auc_pr" in summary_df.columns:
        sort_key = "auc_pr"
    elif "auc_roc" in summary_df.columns:
        sort_key = "auc_roc"
    else:
        sort_key = "accuracy"

    summary_df = summary_df.sort_values(
        by=sort_key,
        ascending=False,
        na_position="last"
    )


    # Save summary
    summary_csv = os.path.join(output_dir, "model_performance_summary.csv")
    ensure_dir_exists(os.path.dirname(summary_csv))
    summary_df.to_csv(summary_csv, index=True)
    logger.info("Saved performance summary to %s", summary_csv)
    return summary_df


# ---------- I/O helpers ----------
def load_and_merge_csv(file1: str, file2: str, n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Sample up to n_samples from each file
    if len(df1) > n_samples:
        df1 = df1.sample(n=n_samples, random_state=random_state)
    if len(df2) > n_samples:
        df2 = df2.sample(n=n_samples, random_state=random_state)
        
    merged = pd.concat([df1, df2], ignore_index=True)
    return merged



# ... (Keep existing imports: os, pandas, logging, etc.) ...

def get_path(model: str, benchmark_base: str, q_type: str) -> str:
    """Helper to build the path for a specific type (yes/no)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir)) # Framework/Ablation -> repo root

    # Construct folder name: e.g., AMBER_yes_questions
    folder_name = None
    if benchmark_base == 'amber':
        if q_type == 'no':
            folder_name = f"AMBER_no_questions"
        else:
            folder_name = f"AMBER_yes_questions"
    elif benchmark_base == 'phd':
        if q_type == 'no':
            folder_name = f"PHD_no_questions"
        else:
            folder_name = f"PHD_yes_questions"
    else:
        raise ValueError("This benchmark does not support!")
    
    return os.path.join(
        project_root, "Results", "Features", model, folder_name, "data_features.csv"
    )

# def parse_args() -> argparse.Namespace:
#     p = argparse.ArgumentParser(description="Classifier with automatic path derivation.")
    
#     # Manual Override
#     p.add_argument("--file1", help="Manual path to first feature CSV.")
#     p.add_argument("--file2", help="Manual path to second feature CSV (for concat).")
    
#     # Automatic derivation arguments
#     p.add_argument("--model", help="Model name (e.g., blip)")
#     p.add_argument("--benchmark", help="Benchmark base name (e.g., AMBER)")
#     p.add_argument("--type", choices=["yes", "no", "both"], default="both", help="Question type to process.")
    
#     # General Config
#     p.add_argument("--label-col", default="hal_label", help="Target column.")

#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     par_dir = os.path.dirname(script_dir)
#     p.add_argument("--output-dir", default=par_dir+"/Results/Classifier/", help="Output directory.")
#     p.add_argument("--random-state", type=int, default=42)
#     return p.parse_args()

# ---------- CLI and main ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and evaluate classifiers on feature CSVs.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--file1", help="Path to first CSV (if providing two CSVs directly).")
    group.add_argument("--results-json", help="Path to results_full_*.json; used to derive feature CSV path.")
    # if file1 is provided then file2 must be provided; we'll handle that below.
    p.add_argument("--file2", help="Path to second CSV (paired with --file1).")
    p.add_argument("--dataset", help="Dataset name (e.g., AMBER_no_questions). Used with --model to construct results filename.")
    p.add_argument("--model", help="Model name (e.g., blip). Used with --dataset to construct results filename.")
    p.add_argument("--results-dir", default="/home/user01/haldet/Framework/results/Classifier", help="Directory where results_full_* files live (default shown).")
    p.add_argument("--label-col", default="hal_label", help="Name of label/target column in features CSV (default: hal_label).")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    par_dir = os.path.dirname(os.path.dirname(script_dir))  # Framework/Ablation -> repo root
    p.add_argument("--output-dir", default=par_dir+"/Results/Classifier/", help="Directory to write outputs (plots, models, summaries).")
    p.add_argument("--random-state", type=int, default=42, help="Random seed.")
    return p.parse_args()

# def main(args: argparse.Namespace) -> None:
#     merged_df = None
#     vlm_hint = args.model
#     dataset_hint = args.benchmark

#     # --- Case 1: Manual File Paths ---
#     if args.file1:
#         logger.info("Using manual file paths.")
#         df1 = pd.read_csv(args.file1)
#         if args.file2:
#             df2 = pd.read_csv(args.file2)
#             merged_df = pd.concat([df1, df2], ignore_index=True)
#             logger.info("Concatenated file1 and file2.")
#         else:
#             merged_df = df1
        
#         vlm_hint = vlm_hint or "manual"
#         dataset_hint = dataset_hint or "manual"

#     # --- Case 2: Automatic Derivation ---
#     elif args.model and args.benchmark and args.type:
#         logger.info(f"Deriving paths for model: {args.model}, benchmark: {args.benchmark}, type: {args.type}")
        
#         if args.type == "both":
#             path_yes = get_path(args.model, args.benchmark, "yes")
#             path_no = get_path(args.model, args.benchmark, "no")
            
#             logger.info(f"Loading and merging:\n1. {path_yes}\n2. {path_no}")
#             df_yes = pd.read_csv(path_yes)
#             df_no = pd.read_csv(path_no)
#             merged_df = pd.concat([df_yes, df_no], ignore_index=True)
#             dataset_hint = f"{args.benchmark}_full"
#         else:
#             # type is 'yes' or 'no'
#             path = get_path(args.model, args.benchmark, args.type)
#             logger.info(f"Loading single file: {path}")
#             merged_df = pd.read_csv(path)
#             dataset_hint = f"{args.benchmark}_{args.type}"
            
#     else:
#         raise ValueError("Provide EITHER --file1 OR (--model, --benchmark, and --type).")

#     # --- Processing ---
#     if merged_df is None or merged_df.empty:
#         logger.error("No data loaded. Check file paths.")
#         return

#     run_name = f"{vlm_hint}_{dataset_hint}"
#     run_output_dir = os.path.join(os.path.abspath(args.output_dir), run_name)
#     ensure_dir_exists(run_output_dir)
    
#     # ... (Rest of your original logic for feature selection and training) ...
#     all_feature_cols = [c for c in merged_df.columns if c != args.label_col]
#     numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
#     # choose intersection
#     feature_cols = [c for c in numeric_cols if c != args.label_col and c != 'id']
#     # feature_cols = [c for c in numeric_cols if c != args.label_col][2:]
#     print(feature_cols)
    
#     exit()
    
#     if not feature_cols:
#         # fallback to all columns except label
#         feature_cols = all_feature_cols
#         logger.warning("No numeric features found; using all columns except label as features.")

#     # # drop duplicates (use full row duplicates)
#     before = merged_df.shape[0]
#     unique_df = merged_df.drop_duplicates().reset_index(drop=True)
#     logger.info("Dropped %d duplicate rows; unique shape: %s", before - unique_df.shape[0], unique_df.shape)

#     # exit(0)

#     # call build_and_evaluate
#     out_dir = run_output_dir
#     ensure_dir_exists(out_dir)
#     summary_df = build_and_evaluate(unique_df, args.label_col, feature_cols, output_dir=out_dir, random_state=args.random_state)
#     logger.info("Top results:\n%s", summary_df.head(10).to_string())

#     # Save additional metadata
#     meta = {
#         "vlm_hint": vlm_hint,
#         "dataset_hint": dataset_hint,
#         "n_rows": int(unique_df.shape[0]),
#         "n_features": len(feature_cols),
#     }
#     if args.file1:
#         meta["input_file"] = args.file1
#         if args.file2:
#             meta["second_file"] = args.file2
#     elif args.type == "both":
#         meta["input_file"] = path_yes
#         meta["second_file"] = path_no
#     else:
#         meta["input_file"] = path
    
#     with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
#         json.dump(meta, f, indent=2)
#     logger.info("Wrote run metadata to %s", out_dir)


def main(args: argparse.Namespace) -> None:
    # Resolve files
    if args.file1:
        if not args.file2:
            raise ValueError("When using --file1 you must provide --file2 as well.")
        file1 = os.path.abspath(args.file1)
        file2 = os.path.abspath(args.file2)
        if not (os.path.exists(file1) and os.path.exists(file2)):
            raise FileNotFoundError("One of the provided CSV files does not exist.")
        merged_df = load_and_merge_csv(file1, file2)
        vlm_hint = None
        dataset_hint = None
    else:
        # results-json path or dataset+model -> derive
        if args.results_json:
            results_json_path = os.path.abspath(args.results_json)
        else:
            if not (args.dataset and args.model):
                raise ValueError("When not using --file1/--file2 you must provide --results-json OR both --dataset and --model.")
            # construct expected filename in results-dir: results_full_{dataset}_{model}.json
            expected = f"results_full_{args.dataset}_{args.model}.json"
            results_json_path = os.path.join(args.results_dir, expected)
        # derive feature file path
        feature_csv = feature_file_path_from_results_json(results_json_path)
        if not os.path.exists(feature_csv):
            logger.warning("Derived feature CSV does not exist yet: %s", feature_csv)
            # allow running if not exists (user may want to inspect); raise to be strict:
            raise FileNotFoundError(f"Feature CSV not found: {feature_csv}")
        merged_df = pd.read_csv(feature_csv)
        file1 = feature_csv
        file2 = None
    vlm_hint = args.model
    dataset_hint = args.dataset

    # --- Drop rows with NaN values ---
    before_drop = merged_df.shape[0]
    merged_df = merged_df.dropna()
    after_drop = merged_df.shape[0]
    logger.info("Dropped %d rows containing NaN values. New shape: %s", before_drop - after_drop, merged_df.shape)
    # --------------------------------
    
        # --- Create run-specific output directory ---
    if vlm_hint and dataset_hint:
        run_name = f"{vlm_hint}_{dataset_hint}"
    else:
        run_name = "unknown_run"
    print('run name: ', run_name)


    base_output_dir = os.path.abspath(args.output_dir)
    run_output_dir = os.path.join(base_output_dir, run_name)

    ensure_dir_exists(run_output_dir)

    logger.info("Results will be saved to: %s", run_output_dir)

    logger.info("Merged dataframe shape: %s", merged_df.shape)

    # Heuristics to pick feature columns and label column
    if args.label_col not in merged_df.columns:
        raise KeyError(f"Label column '{args.label_col}' not found in dataframe columns: {merged_df.columns.tolist()}")

    # determine feature columns: typically everything after first few metadata columns
    # Here we prefer to use numeric columns except the label
    all_feature_cols = [c for c in merged_df.columns if c != args.label_col]
    
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()

    # choose ONE feature explicitly
    # single_feature = [c for c in numeric_cols if c not in {args.label_col, "id", "baseline"} and "ITA" not in c]
    # single_feature = [c for c in numeric_cols if c not in {args.label_col, "id", "baseline"} and "ITC" not in c]
    # single_feature = [c for c in numeric_cols if c not in {args.label_col, "id", "baseline"} and "FSA" not in c]
    # single_feature = [c for c in numeric_cols if c not in {args.label_col, "id", "baseline"} and "FSC" not in c]
    # single_feature = [c for c in numeric_cols if c not in {args.label_col, "id", "baseline"} and "FSA" not in c and "FSC" not in c]
    # single_feature = [c for c in numeric_cols if c not in {args.label_col, "id", "baseline"} and "ITA" not in c and "ITC" not in c]
    # single_feature = [c for c in numeric_cols if c not in {args.label_col, "id", "baseline"} and "ITC" not in c and "FSC" not in c]
    # single_feature = [c for c in numeric_cols if c not in {args.label_col, "id", "baseline"} and "ITA" not in c and "FSA" not in c]
    
    # single_feature = [c for c in numeric_cols if c not in {args.label_col, "id", "baseline"} and "INCONSISTENCY" not in c]
    # single_feature = [c for c in numeric_cols if c not in {args.label_col, "id", "baseline"} and "INCONSISTENCY" in c]
    
    single_feature = [c for c in numeric_cols if c not in {args.label_col, "id", "baseline"}]
    
                                                                                  
    # IMPORTANT: keep it as a list
    feature_cols = [[*single_feature][3]]

    print((feature_cols))
    print(len(feature_cols))
    
    # The above code is written in Python and it is calling the `exit()` function which is used to
    # exit the Python interpreter. This will terminate the Python program or script that is currently
    # running.
    # exit()

    logger.info("Training using single feature: %s", feature_cols[0])

    
    if not feature_cols:
        # fallback to all columns except label
        feature_cols = all_feature_cols
        logger.warning("No numeric features found; using all columns except label as features.")

    # # drop duplicates (use full row duplicates)
    # before = merged_df.shape[0]
    # unique_df = merged_df.drop_duplicates().reset_index(drop=True)
    # logger.info("Dropped %d duplicate rows; unique shape: %s", before - unique_df.shape[0], unique_df.shape)

    

    # exit(0)

    # call build_and_evaluate
    out_dir = run_output_dir
    ensure_dir_exists(out_dir)
    summary_df = build_and_evaluate(merged_df, args.label_col, feature_cols, output_dir=out_dir, random_state=args.random_state)
    logger.info("Top results:\n%s", summary_df.head(10).to_string())

    # Save additional metadata
    meta = {
        "input_file": file1,
        "second_file": file2,
        "vlm_hint": vlm_hint,
        "dataset_hint": dataset_hint,
        "n_rows": int(merged_df.shape[0]),
        "n_features": len(feature_cols),
    }
    with open(os.path.join(out_dir, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Wrote run metadata to %s", out_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)


"""
Like classifier.py, but caps each merged CSV to n_samples rows (see
load_and_merge_csv) before training -- used for sample-size ablation studies.
Example usage (run from Framework/):

    python Ablation/Classifier_ablation_sampling.py \\
        --file1 ../Results/Features/blip/AMBER_no_questions/data_features.csv \\
        --file2 ../Results/Features/blip/AMBER_yes_questions/data_features.csv \\
        --model blip --dataset amber_ablation
"""