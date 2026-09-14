import json
import os
import time

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)



def create_lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])


def create_cart():
    return DecisionTreeClassifier(criterion="gini", random_state=42)


def create_svm():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, random_state=42))
    ])


def create_knn():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ])


def create_nb():
    return GaussianNB()


def create_random_forest():
    return RandomForestClassifier(random_state=42)


def create_gradient_boosting():
    return GradientBoostingClassifier(random_state=42)


def create_adaboost():
    return AdaBoostClassifier(random_state=42)


def create_extra_trees():
    return ExtraTreesClassifier(random_state=42)


MODEL_REGISTRY = {
    "LR": {"name": "Logistic Regression", "creator": create_lr, "save_model": True},
    "CART": {"name": "CART Decision Tree", "creator": create_cart, "save_model": True},
    "SVM": {"name": "Support Vector Machine", "creator": create_svm, "save_model": True},
    "KNN": {"name": "K-Nearest Neighbours", "creator": create_knn, "save_model": False},
    "NB": {"name": "Gaussian Naive Bayes", "creator": create_nb, "save_model": True},
    "RF": {"name": "Random Forest", "creator": create_random_forest, "save_model": True},
    "GB": {"name": "Gradient Boosting", "creator": create_gradient_boosting, "save_model": True},
    "ADA": {"name": "AdaBoost", "creator": create_adaboost, "save_model": True},
    "ET": {"name": "Extra Trees", "creator": create_extra_trees, "save_model": True}
}


def create_model(model_name):
    model_name = model_name.strip().upper()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {', '.join(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[model_name]["creator"]()


def modelTraining(
    input_file,
    model_output_file,
    performance_output_file,
    features,
    target,
    model_name
):
    model_name = model_name.strip().upper()

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {', '.join(MODEL_REGISTRY.keys())}"
        )

    model_info = MODEL_REGISTRY[model_name]

    df = pd.read_csv(input_file)

    X = df[features]
    y = df[target]

    estimator = create_model(model_name)

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

    if model_name == "RF":

        thresholds = [
            round(x, 2)
            for x in list(
                np.arange(0.30, 0.71, 0.05)
            )
        ]

        threshold_predictions = {
            threshold: []
            for threshold in thresholds
        }

        threshold_actual = []

        cv_start = time.perf_counter()

        for train_index, test_index in cv.split(X, y):

            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]

            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            fold_model = create_model(model_name)

            fold_model.fit(X_train, y_train)

            probabilities = fold_model.predict_proba(X_test)[:, 1]

            threshold_actual.extend(y_test.tolist())

            for threshold in thresholds:

                predictions = (
                    probabilities >= threshold
                ).astype(int)

                threshold_predictions[threshold].extend(
                    predictions.tolist()
                )

        cv_end = time.perf_counter()

        threshold_results = {}

        for threshold in thresholds:

            predictions = threshold_predictions[threshold]

            threshold_results[str(threshold)] = {
                "accuracy": float(
                    accuracy_score(
                        threshold_actual,
                        predictions
                    )
                ),
                "precision": float(
                    precision_score(
                        threshold_actual,
                        predictions,
                        zero_division=0
                    )
                ),
                "recall": float(
                    recall_score(
                        threshold_actual,
                        predictions,
                        zero_division=0
                    )
                ),
                "f1": float(
                    f1_score(
                        threshold_actual,
                        predictions,
                        zero_division=0
                    )
                )
            }

        performance = {
            "model": model_info["name"],
            "model_code": model_name,
            "cross_validation": {
                "method": "10-fold Stratified Cross Validation",
                "folds": 10,
                "random_state": 42
            },
            "threshold_tuning": {
                "thresholds_tested": thresholds,
                "results": threshold_results
            },
            "cross_validation_time_seconds": (
                cv_end - cv_start
            )
        }

    else:

        cv_start = time.perf_counter()

        cv_results = cross_validate(
            estimator,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        cv_end = time.perf_counter()

        performance = {
            "model": model_info["name"],
            "model_code": model_name,
            "cross_validation": {
                "method": "10-fold Stratified Cross Validation",
                "folds": 10,
                "random_state": 42
            },
            "metrics": {
                metric: {
                    "mean": float(
                        cv_results[f"test_{metric}"].mean()
                    ),
                    "std": float(
                        cv_results[f"test_{metric}"].std()
                    )
                }
                for metric in scoring
            },
            "fold_scores": {
                metric: [
                    float(x)
                    for x in cv_results[f"test_{metric}"]
                ]
                for metric in scoring
            },
            "cross_validation_time_seconds": (
                cv_end - cv_start
            )
        }

    if model_info["save_model"]:
        training_start = time.perf_counter()
        estimator.fit(X, y)
        training_end = time.perf_counter()

        performance["training_time_seconds"] = (
            training_end - training_start
        )

        if model_output_file is not None:
            model_directory = os.path.dirname(model_output_file)

            if model_directory:
                os.makedirs(model_directory, exist_ok=True)

            joblib.dump(estimator, model_output_file)
    else:
        performance["training_time_seconds"] = None

    performance_directory = os.path.dirname(performance_output_file)

    if performance_directory:
        os.makedirs(performance_directory, exist_ok=True)

    with open(performance_output_file, "w") as file:
        json.dump(performance, file, indent=4)

    return performance
