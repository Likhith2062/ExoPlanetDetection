import json
import os
import re

import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

PERFORMANCE_DIR = os.path.join(
    PROJECT_DIR,
    "Performance",
    "Threshold Tuning Tests"
)

GRAPHS_DIR = os.path.join(
    PROJECT_DIR,
    "Graphs",
    "Threshold Tuning Tests"
)


def extract_threshold(data, filename):

    if "threshold" in data:
        return float(data["threshold"])

    if "classification_threshold" in data:
        return float(data["classification_threshold"])

    match = re.search(
        r"(?:threshold|thresh)[^0-9]*([0-9]+(?:\.[0-9]+)?)",
        filename,
        re.IGNORECASE
    )

    if match:
        return float(match.group(1))

    match = re.search(
        r"(?<![0-9])0\.[0-9]+(?![0-9])",
        filename
    )

    if match:
        return float(match.group(0))

    return None


def load_results():

    if not os.path.isdir(PERFORMANCE_DIR):
        raise FileNotFoundError(
            f"Performance directory not found:\n{PERFORMANCE_DIR}"
        )

    results = []

    for filename in os.listdir(PERFORMANCE_DIR):

        if not filename.lower().endswith(".json"):
            continue

        if "(1)" in filename:
            continue

        file_path = os.path.join(
            PERFORMANCE_DIR,
            filename
        )

        try:
            with open(file_path, "r") as file:
                data = json.load(file)
        except (json.JSONDecodeError, OSError):
            continue

        if "metrics" not in data:
            continue

        threshold = extract_threshold(
            data,
            filename
        )

        if threshold is None:
            continue

        results.append({
            "threshold": threshold,
            "accuracy": data["metrics"]["accuracy"]["mean"],
            "precision": data["metrics"]["precision"]["mean"],
            "recall": data["metrics"]["recall"]["mean"],
            "f1": data["metrics"]["f1"]["mean"],
            "roc_auc": data["metrics"]["roc_auc"]["mean"]
        })

    if not results:
        raise ValueError(
            "No valid threshold performance JSON files were found."
        )

    return sorted(
        results,
        key=lambda item: item["threshold"]
    )


def create_graph(results, metric, title, filename):

    thresholds = [
        result["threshold"]
        for result in results
    ]

    values = [
        result[metric] * 100
        for result in results
    ]

    minimum = min(values)
    maximum = max(values)

    # Dynamically scale the Y-axis around the actual data.
    # A small minimum margin is used so very small changes
    # remain visible without making the graph misleading.
    data_range = maximum - minimum

    if data_range == 0:
        margin = 1.0
    else:
        margin = max(data_range * 0.20, 0.10)

    y_min = max(0, minimum - margin)
    y_max = min(100, maximum + margin)

    # Keep a little room for the value labels above the points.
    label_margin = max((y_max - y_min) * 0.08, 0.05)
    y_max = min(100, y_max + label_margin)

    plt.figure(figsize=(10, 6))

    plt.plot(
        thresholds,
        values,
        marker="o"
    )

    plt.title(title, fontsize=14)
    plt.xlabel("Classification Threshold")
    plt.ylabel("Score (%)")

    plt.ylim(
        y_min,
        y_max
    )

    plt.xticks(thresholds)

    for threshold, value in zip(thresholds, values):

        plt.annotate(
            f"{value:.2f}%",
            (threshold, value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9
        )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    output_file = os.path.join(
        GRAPHS_DIR,
        filename
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def main():

    os.makedirs(
        GRAPHS_DIR,
        exist_ok=True
    )

    results = load_results()

    create_graph(
        results,
        "accuracy",
        "Random Forest Accuracy vs Classification Threshold",
        "threshold_accuracy.png"
    )

    create_graph(
        results,
        "precision",
        "Random Forest Precision vs Classification Threshold",
        "threshold_precision.png"
    )

    create_graph(
        results,
        "recall",
        "Random Forest Recall vs Classification Threshold",
        "threshold_recall.png"
    )

    create_graph(
        results,
        "f1",
        "Random Forest F1 Score vs Classification Threshold",
        "threshold_f1.png"
    )

    create_graph(
        results,
        "roc_auc",
        "Random Forest ROC-AUC vs Classification Threshold",
        "threshold_roc_auc.png"
    )


if __name__ == "__main__":
    main()
