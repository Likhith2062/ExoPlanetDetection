import json
import os
import re

import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

PERFORMANCE_DIR = os.path.join(
    PROJECT_DIR,
    "Performance",
    "Depth Tests"
)

GRAPHS_DIR = os.path.join(
    PROJECT_DIR,
    "Graphs",
    "Depth Tests"
)


def extract_depth(data, filename):

    # First try to read the parameter from the JSON.
    parameter_sources = [
        "max_depth",
        "depth"
    ]

    for parameter in parameter_sources:
        if parameter in data:
            value = data[parameter]

            if value is None:
                return None

            return int(value)

    # Some performance files may store model parameters together.
    if "model_parameters" in data:
        parameters = data["model_parameters"]

        for parameter in parameter_sources:
            if parameter in parameters:
                value = parameters[parameter]

                if value is None:
                    return None

                return int(value)

    # Fall back to extracting the depth from the filename.
    patterns = [
        r"max[_ -]?depth[^0-9]*([0-9]+)",
        r"depth[^0-9]*([0-9]+)",
        r"([0-9]+)[_ -]?depth",
        r"(?<![0-9])([0-9]+)(?=_?depth)"
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            filename,
            re.IGNORECASE
        )

        if match:
            return int(match.group(1))

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

        depth = extract_depth(
            data,
            filename
        )

        if depth is None:
            continue

        results.append({
            "depth": depth,
            "accuracy": data["metrics"]["accuracy"]["mean"],
            "precision": data["metrics"]["precision"]["mean"],
            "recall": data["metrics"]["recall"]["mean"],
            "f1": data["metrics"]["f1"]["mean"],
            "roc_auc": data["metrics"]["roc_auc"]["mean"]
        })

    if not results:
        raise ValueError(
            "No valid depth tuning performance JSON files were found."
        )

    return sorted(
        results,
        key=lambda item: item["depth"]
    )


def create_graph(results, metric, title, filename):

    depths = [
        result["depth"]
        for result in results
    ]

    values = [
        result[metric] * 100
        for result in results
    ]

    minimum = min(values)
    maximum = max(values)

    # Dynamically scale the Y-axis around the observed values.
    data_range = maximum - minimum

    if data_range == 0:
        margin = 1.0
    else:
        margin = max(data_range * 0.20, 0.10)

    y_min = max(0, minimum - margin)
    y_max = min(100, maximum + margin)

    # Leave additional space for the value labels.
    label_margin = max((y_max - y_min) * 0.08, 0.05)
    y_max = min(100, y_max + label_margin)

    plt.figure(figsize=(10, 6))

    plt.plot(
        depths,
        values,
        marker="o"
    )

    plt.title(
        title,
        fontsize=14
    )

    plt.xlabel("Maximum Tree Depth")
    plt.ylabel("Score (%)")

    plt.ylim(
        y_min,
        y_max
    )

    plt.xticks(depths)

    for depth, value in zip(depths, values):

        plt.annotate(
            f"{value:.2f}%",
            (depth, value),
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
        "Random Forest Accuracy vs Maximum Tree Depth",
        "depth_accuracy.png"
    )

    create_graph(
        results,
        "precision",
        "Random Forest Precision vs Maximum Tree Depth",
        "depth_precision.png"
    )

    create_graph(
        results,
        "recall",
        "Random Forest Recall vs Maximum Tree Depth",
        "depth_recall.png"
    )

    create_graph(
        results,
        "f1",
        "Random Forest F1 Score vs Maximum Tree Depth",
        "depth_f1.png"
    )

    create_graph(
        results,
        "roc_auc",
        "Random Forest ROC-AUC vs Maximum Tree Depth",
        "depth_roc_auc.png"
    )


if __name__ == "__main__":
    main()
