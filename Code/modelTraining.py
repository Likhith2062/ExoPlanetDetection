import json
import os
import time

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold

#--------------------------------------------------------
# Parameters for Random Forest Classifier
#--------------------------------------------------------
#def create_random_forest(
#    n_estimators=100,
#    max_depth=None,
#    min_samples_split=2,
#    min_samples_leaf=1,
#    max_features="sqrt",
#    criterion="gini",
#    class_weight=None,
#    bootstrap=True,
#    max_samples=None
#):
#    return RandomForestClassifier(
#       n_estimators=n_estimators,
#       max_depth=max_depth,
#        min_samples_split=min_samples_split,
#        min_samples_leaf=min_samples_leaf,
#        max_features=max_features,
#        criterion=criterion,
#        class_weight=class_weight,
#        bootstrap=bootstrap,
#        max_samples=max_samples,
#        random_state=42,
#        n_jobs=1
#    )


def create_random_forest(
    n_estimators=165,
    max_depth=30,
):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=1
    )



def evaluate_fold(
    train_index,
    test_index,
    X,
    y,
    threshold
):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    fold_model = create_random_forest()

    fold_model.fit(
        X_train,
        y_train
    )

    probabilities = (
        fold_model.predict_proba(X_test)[:, 1]
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "test_index": test_index,
        "probabilities": probabilities,

        "accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }


def generate_final_visualisations(
    y,
    oof_probabilities,
    threshold,
    graphs_directory
):
    """Generate final graphs from out-of-fold predictions."""

    os.makedirs(
        graphs_directory,
        exist_ok=True
    )

    predictions = (
        oof_probabilities >= threshold
    ).astype(int)

    # --------------------------------------------------------
    # ROC Curve
    # --------------------------------------------------------

    fpr, tpr, _ = roc_curve(
        y,
        oof_probabilities
    )

    auc = roc_auc_score(
        y,
        oof_probabilities
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"Random Forest (AUC = {auc:.4f})"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.title(
        "Final Random Forest ROC Curve",
        fontsize=14
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            graphs_directory,
            "final_roc_curve.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    matrix = confusion_matrix(
        y,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Not Exoplanet",
            "Exoplanet"
        ]
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    display.plot(
        ax=ax,
        values_format="d"
    )

    ax.set_title(
        f"Final Random Forest Confusion Matrix "
        f"(Threshold = {threshold})",
        fontsize=14
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            graphs_directory,
            "final_confusion_matrix.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Final Metrics
    # --------------------------------------------------------

    metrics = {
        "Accuracy": accuracy_score(
            y,
            predictions
        ),
        "Precision": precision_score(
            y,
            predictions,
            zero_division=0
        ),
        "Recall": recall_score(
            y,
            predictions,
            zero_division=0
        ),
        "F1 Score": f1_score(
            y,
            predictions,
            zero_division=0
        ),
        "ROC-AUC": auc
    }

    plt.figure(
        figsize=(9, 6)
    )

    bars = plt.bar(
        metrics.keys(),
        [
            value * 100
            for value in metrics.values()
        ]
    )

    plt.title(
        "Final Random Forest Performance",
        fontsize=14
    )

    plt.ylabel("Score (%)")
    plt.ylim(0, 100)

    plt.xticks(
        rotation=20
    )

    for bar, value in zip(
        bars,
        metrics.values()
    ):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value * 100:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            graphs_directory,
            "final_metrics.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return {
        "roc_auc": auc,
        "confusion_matrix": matrix,
        "metrics": metrics
    }


def modelTraining(
    input_file,
    model_output_file,
    performance_output_file,
    features,
    target,
    threshold=0.50
):

    # --------------------------------------------------------
    # Validate threshold
    # --------------------------------------------------------

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Threshold must be between 0.0 and 1.0."
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(input_file)

    X = df[features]
    y = df[target]

    # --------------------------------------------------------
    # 10-fold Stratified Cross Validation
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    cv_start = time.perf_counter()

    fold_results = Parallel(
        n_jobs=-1
    )(
        delayed(evaluate_fold)(
            train_index,
            test_index,
            X,
            y,
            threshold
        )
        for train_index, test_index in cv.split(X, y)
    )

    cv_end = time.perf_counter()

    # --------------------------------------------------------
    # Collect out-of-fold probabilities
    # --------------------------------------------------------
    # Each sample is predicted by a model that did not train
    # on that sample.
    oof_probabilities = [None] * len(X)

    for result in fold_results:
        for index, probability in zip(
            result["test_index"],
            result["probabilities"]
        ):
            oof_probabilities[index] = probability

    oof_probabilities = pd.Series(
        oof_probabilities
    ).to_numpy()

    fold_accuracy = [
        result["accuracy"]
        for result in fold_results
    ]

    fold_precision = [
        result["precision"]
        for result in fold_results
    ]

    fold_recall = [
        result["recall"]
        for result in fold_results
    ]

    fold_f1 = [
        result["f1"]
        for result in fold_results
    ]

    fold_roc_auc = [
        result["roc_auc"]
        for result in fold_results
    ]

    # --------------------------------------------------------
    # Performance report
    # --------------------------------------------------------

    performance = {
        "model": "Random Forest",
        "model_code": "RF",
        "threshold": threshold,

        "hyperparameters": {
            parameter: value
            for parameter, value in create_random_forest().get_params().items()
        },

        "cross_validation": {
            "method": "10-fold Stratified Cross Validation",
            "folds": 10,
            "random_state": 42
        },

        "metrics": {
            "accuracy": {
                "mean": float(
                    pd.Series(
                        fold_accuracy
                    ).mean()
                ),
                "std": float(
                    pd.Series(
                        fold_accuracy
                    ).std()
                )
            },

            "precision": {
                "mean": float(
                    pd.Series(
                        fold_precision
                    ).mean()
                ),
                "std": float(
                    pd.Series(
                        fold_precision
                    ).std()
                )
            },

            "recall": {
                "mean": float(
                    pd.Series(
                        fold_recall
                    ).mean()
                ),
                "std": float(
                    pd.Series(
                        fold_recall
                    ).std()
                )
            },

            "f1": {
                "mean": float(
                    pd.Series(
                        fold_f1
                    ).mean()
                ),
                "std": float(
                    pd.Series(
                        fold_f1
                    ).std()
                )
            },

            "roc_auc": {
                "mean": float(
                    pd.Series(
                        fold_roc_auc
                    ).mean()
                ),
                "std": float(
                    pd.Series(
                        fold_roc_auc
                    ).std()
                )
            }
        },

        "fold_scores": {
            "accuracy": [
                float(x)
                for x in fold_accuracy
            ],

            "precision": [
                float(x)
                for x in fold_precision
            ],

            "recall": [
                float(x)
                for x in fold_recall
            ],

            "f1": [
                float(x)
                for x in fold_f1
            ],

            "roc_auc": [
                float(x)
                for x in fold_roc_auc
            ]
        },

        "cross_validation_time_seconds": (
            cv_end - cv_start
        )
    }

    # --------------------------------------------------------
    # Train final Random Forest
    # --------------------------------------------------------

    training_start = time.perf_counter()

    estimator = create_random_forest()

    estimator.fit(
        X,
        y
    )

    training_end = time.perf_counter()

    performance["training_time_seconds"] = (
        training_end - training_start
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_directory = os.path.dirname(
        model_output_file
    )

    if model_directory:
        os.makedirs(
            model_directory,
            exist_ok=True
        )

    joblib.dump(
        estimator,
        model_output_file
    )

    # --------------------------------------------------------
    # Save performance report
    # --------------------------------------------------------

    performance_directory = os.path.dirname(
        performance_output_file
    )

    if performance_directory:
        os.makedirs(
            performance_directory,
            exist_ok=True
        )

    with open(
        performance_output_file,
        "w"
    ) as file:

        json.dump(
            performance,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Final model visualisations
    # --------------------------------------------------------
    # These use the out-of-fold predictions from the same
    # 10-fold CV above, avoiding evaluation on training data.
    performance_directory = os.path.dirname(
        performance_output_file
    )

    project_directory = os.path.dirname(
        performance_directory
    )

    graphs_directory = os.path.join(
        project_directory,
        "Graphs",
        "Final Evaluation"
    )

    visualisation_results = generate_final_visualisations(
        y.to_numpy(),
        oof_probabilities,
        threshold,
        graphs_directory
    )

    performance["final_visualisations"] = {
        "graphs_directory": graphs_directory,
        "roc_auc": float(
            visualisation_results["roc_auc"]
        ),
        "confusion_matrix": (
            visualisation_results["confusion_matrix"].tolist()
        ),
        "metrics": {
            metric: float(value)
            for metric, value
            in visualisation_results["metrics"].items()
        }
    }

    # Save the graph-derived values in the performance JSON too.
    with open(
        performance_output_file,
        "w"
    ) as file:

        json.dump(
            performance,
            file,
            indent=4
        )

    return performance