"""Data preprocessing and train / validation / test splitting."""

import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def preProcess(input_file, output_file, features, target, log_file=None):
    """Preprocess the NASA KOI dataset, save it, and return the DataFrame.

    Every step records how many rows it removed so the methodology is
    auditable (optionally written to `log_file` as JSON).
    """
    steps = []

    # Step 1 - load, ignoring NASA metadata lines that start with '#'.
    df = pd.read_csv(input_file, comment="#")
    steps.append({"step": "Load CSV (skip '#' metadata lines)",
                  "rows_after": len(df)})

    # Step 2 - feature selection (only project attributes + target).
    df = df[features + [target]].copy()
    steps.append({"step": "Feature selection",
                  "columns_kept": features + [target],
                  "rows_after": len(df)})

    # Step 3 - binary classification: drop unresolved CANDIDATE rows.
    before = len(df)
    df = df[df[target].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
    steps.append({"step": "Remove CANDIDATE rows (label unknown)",
                  "rows_removed": before - len(df),
                  "rows_after": len(df)})

    # Step 4 - remove tuples with any missing feature (listwise deletion).
    before = len(df)
    missing_per_column = df[features].isna().sum().to_dict()
    df = df[~df[features].isna().any(axis=1)].copy()
    steps.append({"step": "Remove rows with missing feature values",
                  "missing_per_column": {k: int(v) for k, v in
                                         missing_per_column.items()},
                  "rows_removed": before - len(df),
                  "rows_after": len(df)})

    # Step 5 - label encoding.
    df[target] = df[target].map({"FALSE POSITIVE": 0, "CONFIRMED": 1})
    steps.append({"step": "Encode target (FALSE POSITIVE=0, CONFIRMED=1)",
                  "class_counts": {str(k): int(v) for k, v in
                                   df[target].value_counts().items()},
                  "rows_after": len(df)})

    df.reset_index(drop=True, inplace=True)

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(output_file, index=False)

    # Note: no scaling is applied - Random Forests are scale-invariant
    # (split decisions depend only on feature ordering).
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_file, "w") as f:
            json.dump({"steps": steps,
                       "note": "No scaling: Random Forest is scale-invariant."},
                      f, indent=4)

    return df


def split_dataset(df, target):
    """Stratified 70 / 15 / 15 split into train, validation and test sets.

    * train      - fit models and run cross-validation
    * validation - tune the decision threshold (never used for fitting)
    * test       - touched ONCE for the final, unbiased evaluation
    """
    train_df, temp_df = train_test_split(
        df,
        test_size=VAL_FRAC + TEST_FRAC,
        stratify=df[target],
        random_state=RANDOM_STATE,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=temp_df[target],
        random_state=RANDOM_STATE,
    )
    return train_df, val_df, test_df
