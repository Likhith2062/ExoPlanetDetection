import json
import os

import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

MODELS_DIR = os.path.join(
    PROJECT_DIR,
    "Models"
)

DATASETS_DIR = os.path.join(
    PROJECT_DIR,
    "Datasets"
)

PERFORMANCE_DIR = os.path.join(
    PROJECT_DIR,
    "Performance",
    "Final Tests"
)

GRAPHS_DIR = os.path.join(
    PROJECT_DIR,
    "Graphs",
    "Final Evaluation"
)


FEATURES = [
    "koi_depth",
    "koi_duration",
    "koi_impact",
    "koi_model_snr",
    "koi_max_mult_ev",
    "koi_srad",
    "koi_smass",
    "koi_steff",
    "koi_bin_oedp_sig"
]

TARGET = "koi_disposition"

THRESHOLD = 0.46


def find_model(model_name):

    model_name = os.path.splitext(
        model_name.strip()
    )[0]

    model_file = os.path.join(
        MODELS_DIR,
        model_name + ".pkl"
    )

    if os.path.isfile(model_file):
        return model_file

    raise FileNotFoundError(
        f"Model file not found in Models directory:\n"
        f"{model_file}"
    )


def find_dataset():

    dataset_file = os.path.join(
        DATASETS_DIR,
        "Processed.csv"
    )

    if not os.path.isfile(dataset_file):
        raise FileNotFoundError(
            f"Final processed dataset not found:\\n{dataset_file}"
        )

    return dataset_file


def create_roc_curve(model, X, y):

    probabilities = model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(
        y,
        probabilities
    )

    auc = roc_auc_score(
        y,
        probabilities
    )

    plt.figure(figsize=(8, 6))

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

    output_file = os.path.join(
        GRAPHS_DIR,
        "final_roc_curve.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return auc


def create_confusion_matrix(model, X, y):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    matrix = confusion_matrix(
        y,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Not Exoplanet", "Exoplanet"]
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
        f"(Threshold = {THRESHOLD})",
        fontsize=14
    )

    plt.tight_layout()

    output_file = os.path.join(
        GRAPHS_DIR,
        "final_confusion_matrix.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return matrix


def create_metrics_graph(model, X, y):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    matrix = confusion_matrix(
        y,
        predictions
    )

    tn, fp, fn, tp = matrix.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    precision = (
        tp / (tp + fp)
        if (tp + fp) != 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) != 0
        else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) != 0
        else 0
    )

    auc = roc_auc_score(
        y,
        probabilities
    )

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": auc
    }

    plt.figure(figsize=(9, 6))

    bars = plt.bar(
        metrics.keys(),
        [value * 100 for value in metrics.values()]
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

    output_file = os.path.join(
        GRAPHS_DIR,
        "final_metrics.png"
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def main():

    print("=" * 70)
    print("FINAL RANDOM FOREST MODEL VISUALIZATION")
    print("=" * 70)

    model_name = input(
        "\nEnter final model file name: "
    ).strip()

    if not model_name:
        raise ValueError(
            "Model file name cannot be empty."
        )

    model_file = find_model(
        model_name
    )

    dataset_file = find_dataset()

    print(
        f"\nLoading model: {model_file}"
    )

    print(
        f"Using dataset: {dataset_file}"
    )

    model = joblib.load(
        model_file
    )

    data = __import__("pandas").read_csv(
        dataset_file
    )

    X = data[FEATURES]
    y = data[TARGET]

    os.makedirs(
        GRAPHS_DIR,
        exist_ok=True
    )

    print("\nGenerating ROC curve...")

    auc = create_roc_curve(
        model,
        X,
        y
    )

    print(
        f"ROC-AUC from dataset predictions: {auc:.4f}"
    )

    print("\nGenerating confusion matrix...")

    matrix = create_confusion_matrix(
        model,
        X,
        y
    )

    print(
        f"Confusion matrix:\n{matrix}"
    )

    print("\nGenerating final metrics graph...")

    create_metrics_graph(
        model,
        X,
        y
    )

    print(
        f"\nGraphs saved to:\n{GRAPHS_DIR}"
    )

    print(
        "\nFinal visualization completed successfully."
    )


if __name__ == "__main__":
    main()
