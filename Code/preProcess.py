import pandas as pd


def preProcess(input_file, output_file, features, target):
    """Preprocess the NASA KOI dataset and return the final DataFrame."""

    # Load NASA CSV while ignoring metadata/comment lines.
    df = pd.read_csv(input_file, comment="#")

    # Keep only project attributes.
    df = df[features + [target]].copy()

    # Binary classification: exclude unresolved candidates.
    df = df[df[target].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()

    # Remove tuples with at least one missing feature.
    missing_mask = df[features].isna().any(axis=1)
    df = df[~missing_mask].copy()

    # Encode target: FALSE POSITIVE = 0, CONFIRMED = 1.
    df[target] = df[target].map({
        "FALSE POSITIVE": 0,
        "CONFIRMED": 1
    })

    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_file, index=False)

    return df
