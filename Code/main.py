"""Command-line interface.  Run:  python main.py"""

import os

from pipeline import DATASETS_DIR, format_summary, run_pipeline


def main():
    print("=" * 70)
    print("NASA KOI EXOPLANET CLASSIFICATION")
    print("=" * 70)

    input_name = input("\nEnter input data file name: ").strip()
    if not input_name:
        raise ValueError("Input data file name cannot be empty.")

    processed_name = input(
        "Enter output processed data file name "
        "(leave blank if already processed): ").strip()

    threshold_input = input(
        "\nEnter classification threshold "
        "(leave blank to tune automatically on validation set): ").strip()
    if threshold_input:
        try:
            threshold = float(threshold_input)
        except ValueError:
            raise ValueError("Threshold must be a valid number.")
    else:
        threshold = None

    model_name = input("\nEnter model file name: ").strip()

    perf = run_pipeline(
        os.path.join(DATASETS_DIR, input_name),
        processed_name, model_name, threshold,
    )
    print("\n" + format_summary(perf))
    print("\nOutputs:")
    for k, v in perf["outputs"].items():
        print(f"  {k}: {v}")
    print("\nProject execution completed successfully.")


if __name__ == "__main__":
    main()
