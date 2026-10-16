"""Random Forest training with a strict train / validation / test protocol."""

import json
import os
import time

import joblib
import matplotlib
matplotlib.use("Agg")  # no display needed; safe to call from a GUI thread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold

from preProcess import split_dataset


def create_random_forest(n_estimators=165, max_depth=30):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=1,
    )


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def compute_metrics(y_true, probabilities, threshold):
    """All performance metrics for one set of predictions."""
    y_true = np.asarray(y_true)
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
    }


def tune_threshold(y_val, val_probabilities):
    """Pick the threshold that maximises F1 on the VALIDATION set only."""
    precision, recall, thresholds = precision_recall_curve(y_val, val_probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


# ------------------------------------------------------------
# Cross-validation (training set only)
# ------------------------------------------------------------
def evaluate_fold(train_index, test_index, X, y, threshold):
    model = create_random_forest()
    model.fit(X.iloc[train_index], y.iloc[train_index])
    probabilities = model.predict_proba(X.iloc[test_index])[:, 1]
    return compute_metrics(y.iloc[test_index], probabilities, threshold)


# ------------------------------------------------------------
# Graphs (test set)
# ------------------------------------------------------------
def generate_test_visualisations(y, probabilities, threshold, graphs_directory):
    os.makedirs(graphs_directory, exist_ok=True)
    metrics = compute_metrics(y, probabilities, threshold)
    predictions = (probabilities >= threshold).astype(int)

    # ROC curve
    fpr, tpr, _ = roc_curve(y, probabilities)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2,
             label=f"Random Forest (AUC = {metrics['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], "--", label="Random Classifier")
    plt.title("Test Set ROC Curve", fontsize=14)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_directory, "test_roc_curve.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(
        confusion_matrix(y, predictions, labels=[0, 1]),
        display_labels=["Not Exoplanet", "Exoplanet"],
    ).plot(ax=ax, values_format="d")
    ax.set_title(f"Test Set Confusion Matrix (Threshold = {threshold:.3f})",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_directory, "test_confusion_matrix.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # Metric bars
    names = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc"]
    values = [metrics[n] * 100 for n in names]
    plt.figure(figsize=(10, 6))
    bars = plt.bar([n.upper() for n in names], values)
    plt.title("Test Set Performance", fontsize=14)
    plt.ylabel("Score (%)")
    plt.ylim(0, 100)
    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.2f}%",
                 ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(graphs_directory, "test_metrics.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def _summary(values):
    s = pd.Series(values)
    return {"mean": float(s.mean()), "std": float(s.std())}


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------
def modelTraining(
    input_file,
    model_output_file,
    performance_output_file,
    features,
    target,
    threshold=None,
    log=print,
):
    """Train / validate / test a Random Forest.

    threshold=None -> tuned automatically on the validation set.
    """
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0.")

    performance_dir = os.path.dirname(performance_output_file)
    project_dir = os.path.dirname(performance_dir)
    base = os.path.splitext(os.path.basename(performance_output_file))[0]
    base = base[:-5] if base.endswith("_perf") else base

    # ---- 1. Split -------------------------------------------------
    df = pd.read_csv(input_file)
    train_df, val_df, test_df = split_dataset(df, target)
    log(f"Split -> train: {len(train_df)}, validation: {len(val_df)}, "
        f"test: {len(test_df)}")

    splits_dir = os.path.join(project_dir, "Datasets", "Splits")
    os.makedirs(splits_dir, exist_ok=True)
    train_df.to_csv(os.path.join(splits_dir, f"{base}_train.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, f"{base}_validation.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, f"{base}_test.csv"), index=False)

    # Feature ranges seen in training (shown in the GUI prediction form)
    feature_ranges = {
        f: {"min": float(train_df[f].min()), "max": float(train_df[f].max()),
            "median": float(train_df[f].median()),
            "p05": float(train_df[f].quantile(0.05)),
            "p95": float(train_df[f].quantile(0.95))}
        for f in features
    }

    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]

    # ---- 2. Fit on TRAIN only ------------------------------------
    log("Training final model on the training set...")
    t0 = time.perf_counter()
    estimator = create_random_forest()
    estimator.fit(X_train, y_train)
    training_time = time.perf_counter() - t0

    # ---- 3. Threshold from VALIDATION ----------------------------
    val_prob = estimator.predict_proba(X_val)[:, 1]
    tuned = tune_threshold(y_val, val_prob)
    threshold_source = "manual" if threshold is not None else "tuned_on_validation"
    if threshold is None:
        threshold = tuned
    log(f"Decision threshold: {threshold:.3f} ({threshold_source})")

    # ---- 4. 10-fold CV on TRAIN only -----------------------------
    log("Running 10-fold stratified cross-validation on the training set...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    Xtr = X_train.reset_index(drop=True)
    ytr = y_train.reset_index(drop=True)
    t0 = time.perf_counter()
    fold_results = Parallel(n_jobs=-1)(
        delayed(evaluate_fold)(tr, te, Xtr, ytr, threshold)
        for tr, te in cv.split(Xtr, ytr)
    )
    cv_time = time.perf_counter() - t0
    metric_names = ["accuracy", "balanced_accuracy", "precision", "recall",
                    "specificity", "f1", "mcc", "roc_auc"]
    cv_summary = {m: _summary([r[m] for r in fold_results]) for m in metric_names}

    # ---- 5. Validation + TEST evaluation -------------------------
    validation_metrics = compute_metrics(y_val, val_prob, threshold)
    test_prob = estimator.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)
    test_metrics = compute_metrics(y_test, test_prob, threshold)

    # ---- 6. Testing output ---------------------------------------
    os.makedirs(performance_dir, exist_ok=True)
    predictions_path = os.path.join(performance_dir, f"{base}_test_predictions.csv")
    out = test_df.copy()
    out.insert(0, "true_label", y_test.values)
    out.insert(1, "predicted_label", test_pred)
    out.insert(2, "probability_confirmed", np.round(test_prob, 4))
    out.insert(3, "correct", (test_pred == y_test.values))
    out.drop(columns=[target], inplace=True)
    out.to_csv(predictions_path, index=False)

    report_path = os.path.join(performance_dir, f"{base}_test_report.txt")
    with open(report_path, "w") as f:
        f.write(f"TEST SET REPORT (threshold = {threshold:.3f})\n")
        f.write("=" * 60 + "\n")
        f.write(classification_report(
            y_test, test_pred, labels=[0, 1],
            target_names=["FALSE POSITIVE", "CONFIRMED"], digits=4))
        cm = test_metrics["confusion_matrix"]
        f.write(f"\nConfusion matrix: TN={cm['tn']} FP={cm['fp']} "
                f"FN={cm['fn']} TP={cm['tp']}\n")
        f.write(f"ROC-AUC: {test_metrics['roc_auc']:.4f}   "
                f"MCC: {test_metrics['mcc']:.4f}\n")

    graphs_dir = os.path.join(project_dir, "Graphs", "Final Evaluation")
    generate_test_visualisations(y_test.to_numpy(), test_prob, threshold, graphs_dir)

    # ---- 7. Save model + performance JSON ------------------------
    os.makedirs(os.path.dirname(model_output_file), exist_ok=True)
    joblib.dump(estimator, model_output_file)

    performance = {
        "model": "Random Forest",
        "model_code": "RF",
        "threshold": threshold,
        "threshold_source": threshold_source,
        "hyperparameters": create_random_forest().get_params(),
        "data_split": {
            "strategy": "Stratified hold-out, random_state=42",
            "train": len(train_df),
            "validation": len(val_df),
            "test": len(test_df),
            "fractions": {"train": 0.70, "validation": 0.15, "test": 0.15},
            "split_files_directory": splits_dir,
        },
        "cross_validation_on_train_only": {
            "method": "10-fold Stratified Cross Validation",
            "folds": 10,
            "metrics": cv_summary,
            "time_seconds": cv_time,
        },
        "features": list(features),
        "feature_ranges": feature_ranges,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "outputs": {
            "test_predictions_csv": predictions_path,
            "test_report_txt": report_path,
            "graphs_directory": graphs_dir,
        },
        "training_time_seconds": training_time,
    }
    with open(performance_output_file, "w") as f:
        json.dump(performance, f, indent=4, default=str)

    log("Done.")
    return performance
