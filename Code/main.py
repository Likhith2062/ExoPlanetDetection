import os

from preProcess import preProcess
from modelTraining import modelTraining


# ============================================================
# PROJECT-WIDE FEATURE / TARGET DEFINITION
# ============================================================

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


# ============================================================
# DIRECTORY CONFIGURATION
# ============================================================

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

    input_name = input("\nEnter input data file name: ").strip()

    input_file = os.path.join(DATASETS_DIR, input_name)

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
        processed_base = os.path.splitext(processed_name)[0]
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
        print(f"Processed dataset saved to: {processed_file}")

    else:
        training_file = input_file

    # --------------------------------------------------------
    # 3. Select algorithm
    # --------------------------------------------------------

    print("\nAvailable ML algorithms:")
    print("  LR   - Logistic Regression")
    print("  CART - CART Decision Tree")
    print("  SVM  - Support Vector Machine")
    print("  KNN  - K-Nearest Neighbours")
    print("  NB   - Gaussian Naive Bayes")

    model_name = input("\nEnter ML algorithm: ").strip().upper()

    if model_name not in ["LR", "CART", "SVM", "KNN", "NB"]:
        raise ValueError(
            "Invalid algorithm. Choose LR, CART, SVM, KNN or NB."
        )

    # --------------------------------------------------------
    # 4. Determine output filenames
    # --------------------------------------------------------

    if model_name == "KNN":

        # KNN has no separately saved model.
        model_file = None

        performance_name = input(
            "\nEnter performance file name: "
        ).strip()

        if not performance_name:
            raise ValueError(
                "Performance file name cannot be empty."
            )

        performance_base = os.path.splitext(performance_name)[0]

        performance_file = os.path.join(
            PERFORMANCE_DIR,
            performance_base + ".json"
        )

    else:

        # Ask for model filename for trainable models.
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

        # Performance file automatically gets the same base name.
        performance_file = os.path.join(
            PERFORMANCE_DIR,
            base_name + "_perf.json"
        )
    # --------------------------------------------------------
    # 5. Run model module
    # --------------------------------------------------------

    print("\nRunning model evaluation...")

    modelTraining(
        training_file,
        model_file,
        performance_file,
        FEATURES,
        TARGET,
        model_name
    )

    print(f"\nPerformance report saved to: {performance_file}")

    if model_file is not None:
        print(f"Model saved to: {model_file}")

    print("\nProject execution completed successfully.")


if __name__ == "__main__":
    main()
