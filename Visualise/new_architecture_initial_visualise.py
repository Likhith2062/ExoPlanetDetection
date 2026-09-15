import json
import os

import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

PERFORMANCE_DIR = os.path.join(
    PROJECT_DIR,
    "Performance",
    "New Architecture Initial Tests"
)

GRAPHS_DIR = os.path.join(
    PROJECT_DIR,
    "Graphs",
    "New Architecture Initial Tests"
)


def load_performance_files():

    if not os.path.isdir(PERFORMANCE_DIR):
        raise FileNotFoundError(
            f"Performance directory not found:\n{PERFORMANCE_DIR}"
        )

    results = []

    for filename in os.listdir(PERFORMANCE_DIR):

        if not filename.lower().endswith(".json"):
            continue

        # Ignore duplicate copies such as filename(1).json
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

        results.append({
            "model": data.get(
                "model",
                os.path.splitext(filename)[0]
            ),
            "accuracy": data["metrics"]["accuracy"]["mean"],
            "precision": data["metrics"]["precision"]["mean"],
            "recall": data["metrics"]["recall"]["mean"],
            "f1": data["metrics"]["f1"]["mean"],
            "roc_auc": data["metrics"]["roc_auc"]["mean"]
        })

    if not results:
        raise ValueError(
            "No valid performance JSON files were found."
        )

    return sorted(
        results,
        key=lambda item: item["model"]
    )


def create_graph(results, metric, title, filename):

    models = [result["model"] for result in results]
    values = [result[metric] * 100 for result in results]

    plt.figure(figsize=(10, 6))

    bars = plt.bar(models, values)

    plt.title(title, fontsize=14)
    plt.xlabel("Model")
    plt.ylabel("Score (%)")
    plt.ylim(0, 100)

    plt.xticks(
        rotation=30,
        ha="right"
    )

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9
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

    results = load_performance_files()

    create_graph(
        results,
        "accuracy",
        "Model Comparison - Accuracy",
        "model_accuracy.png"
    )

    create_graph(
        results,
        "precision",
        "Model Comparison - Precision",
        "model_precision.png"
    )

    create_graph(
        results,
        "recall",
        "Model Comparison - Recall",
        "model_recall.png"
    )

    create_graph(
        results,
        "f1",
        "Model Comparison - F1 Score",
        "model_f1.png"
    )

    create_graph(
        results,
        "roc_auc",
        "Model Comparison - ROC-AUC",
        "model_roc_auc.png"
    )


if __name__ == "__main__":
    main()
