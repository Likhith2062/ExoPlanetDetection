import json
import time

import joblib
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


def modelTraining(
    input_file,
    model_output_file,
    performance_output_file,
    features,
    target,
    model_name
):
    """
    Evaluate an ML algorithm using 10-fold stratified CV.

    LR, CART, SVM and NB are subsequently trained on the complete
    dataset and saved as model files.

    KNN is evaluated using 10-fold CV only; no separate final
    training/model file is produced.

    All performance information is written to JSON.
    This module intentionally produces no console report.
    """

    # --------------------------------------------------------
    # 1. Load preprocessed data
    # --------------------------------------------------------

    df = pd.read_csv(input_file)

    X = df[features]
    y = df[target]

    # --------------------------------------------------------
    # 2. Construct model pipeline
    # --------------------------------------------------------

    if model_name == "LR":

        estimator = Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ])

    elif model_name == "CART":

        estimator = Pipeline([
            (
                "model",
                DecisionTreeClassifier(
                    criterion="gini",
                    random_state=42
                )
            )
        ])

    elif model_name == "SVM":

        estimator = Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                SVC(
                    probability=True,
                    random_state=42
                )
            )
        ])

    elif model_name == "KNN":

        estimator = Pipeline([
            ("scaler", StandardScaler()),
            (
                "model",
                KNeighborsClassifier(
                    n_neighbors=5
                )
            )
        ])

    elif model_name == "NB":

        estimator = Pipeline([
            (
                "model",
                GaussianNB()
            )
        ])

    else:
        raise ValueError(
            "Invalid model. Choose LR, CART, SVM, KNN or NB."
        )

    # --------------------------------------------------------
    # 3. Configure 10-fold stratified cross-validation
    # --------------------------------------------------------

    cv = StratifiedKFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    # --------------------------------------------------------
    # 4. Cross-validation
    # --------------------------------------------------------

    cv_start = time.perf_counter()

    cv_results = cross_validate(
        estimator,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    cv_time = time.perf_counter() - cv_start

    # --------------------------------------------------------
    # 5. Build metric report
    # --------------------------------------------------------

    metrics = {}

    for metric in scoring:
        scores = cv_results[f"test_{metric}"]

        metrics[metric] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "fold_scores": [
                float(score)
                for score in scores
            ]
        }

    # --------------------------------------------------------
    # 6. KNN: evaluation only
    # --------------------------------------------------------

    if model_name == "KNN":

        report = {
            "model": model_name,
            "dataset_size": int(len(df)),
            "number_of_features": int(len(features)),
            "cross_validation": {
                "method": "Stratified K-Fold",
                "number_of_folds": 10,
                "shuffle": True,
                "random_state": 42
            },
            "cross_validation_time_seconds": float(cv_time),
            "training_time_seconds": None,
            "model_saved": False,
            "model_file": None,
            "metrics": metrics
        }

    # --------------------------------------------------------
    # 7. Final training for trainable models
    # --------------------------------------------------------

    else:

        training_start = time.perf_counter()

        estimator.fit(X, y)

        training_time = time.perf_counter() - training_start

        # Save complete pipeline so scaling and model are preserved.
        model_package = {
            "model": estimator,
            "features": features,
            "target": target,
            "model_name": model_name
        }

        joblib.dump(
            model_package,
            model_output_file
        )

        report = {
            "model": model_name,
            "dataset_size": int(len(df)),
            "number_of_features": int(len(features)),
            "cross_validation": {
                "method": "Stratified K-Fold",
                "number_of_folds": 10,
                "shuffle": True,
                "random_state": 42
            },
            "cross_validation_time_seconds": float(cv_time),
            "training_time_seconds": float(training_time),
            "model_saved": True,
            "model_file": model_output_file,
            "metrics": metrics
        }

    # --------------------------------------------------------
    # 8. Save JSON performance report
    # --------------------------------------------------------

    with open(performance_output_file, "w") as file:
        json.dump(
            report,
            file,
            indent=4
        )

    return report