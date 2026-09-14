import os

from preProcess import preProcess
from modelTraining import modelTraining


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


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CODE_DIR)

DATASETS_DIR = os.path.join(PROJECT_DIR, "Datasets")
MODELS_DIR = os.path.join(PROJECT_DIR, "Models")
PERFORMANCE_DIR = os.path.join(PROJECT_DIR, "Performance")


def main():

    print("=" * 70)
    print("NASA KOI EXOPLANET CLASSIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Input dataset
    # --------------------------------------------------------

    input_name = input(
        "\nEnter input data file name: "
    ).strip()

    if not input_name:
        raise ValueError(
            "Input data file name cannot be empty."
        )

    input_file = os.path.join(
        DATASETS_DIR,
        input_name
    )

    if not os.path.isfile(input_file):
        raise FileNotFoundError(
            f"Input dataset not found:\n{input_file}"
        )

    # --------------------------------------------------------
    # 2. Optional preprocessing
    # --------------------------------------------------------

    processed_name = input(
        "Enter output processed data file name "
        "(leave blank if already processed): "
    ).strip()

    if processed_name:

        processed_base = os.path.splitext(
            processed_name
        )[0]

        processed_file = os.path.join(
            DATASETS_DIR,
            processed_base + ".csv"
        )

        print("\nPreprocessing dataset...")

        preProcess(
            input_file,
            processed_file,
            FEATURES,
            TARGET
        )

        training_file = processed_file

        print(
            f"Processed dataset saved to: "
            f"{processed_file}"
        )

    else:
        training_file = input_file

    # --------------------------------------------------------
    # 3. Classification threshold
    # --------------------------------------------------------

    threshold_input = input(
        "\nEnter classification threshold "
        "(default 0.46): "
    ).strip()

    if threshold_input:

        try:
            threshold = float(threshold_input)
        except ValueError:
            raise ValueError(
                "Threshold must be a valid number."
            )

    else:
        threshold = 0.46


    # --------------------------------------------------------
    # 4. Output Files
    # --------------------------------------------------------

    result_name = input(
        "\nEnter model file name: "
    ).strip()

    if not result_name:
        raise ValueError(
            "Model file name cannot be empty."
        )

    base_name = os.path.splitext(result_name)[0]

    model_file = os.path.join(
        MODELS_DIR,
        base_name + ".pkl"
    )

    performance_file = os.path.join(
        PERFORMANCE_DIR,
        base_name + "_perf.json"
    )


    # --------------------------------------------------------
    # 5. Train and evaluate Random Forest
    # --------------------------------------------------------

    print("\nRunning Random Forest evaluation...")

    modelTraining(
        training_file,
        model_file,
        performance_file,
        FEATURES,
        TARGET,
        threshold
    )

    print(
        f"\nPerformance report saved to: "
        f"{performance_file}"
    )

    print(
        f"Model saved to: {model_file}"
    )

    print(
        "\nProject execution completed successfully."
    )


if __name__ == "__main__":
    main()