import json
import os
import time

import joblib
import pandas as pd
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold


def create_random_forest():
    return RandomForestClassifier(
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

    return performance