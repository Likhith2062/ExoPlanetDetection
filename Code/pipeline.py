"""Shared pipeline used by the command line (main.py) and GUI (gui.py).

Also contains the helpers for single-sample prediction used by the GUI.
"""

import json
import os

import joblib
import pandas as pd

from preProcess import preProcess
from modelTraining import modelTraining

FEATURES = [
    "koi_depth", "koi_duration", "koi_impact", "koi_model_snr",
    "koi_max_mult_ev", "koi_srad", "koi_smass", "koi_steff",
    "koi_bin_oedp_sig",
]
TARGET = "koi_disposition"

# (friendly name, unit, short description) shown in the prediction form
FEATURE_INFO = {
    "koi_depth":        ("Transit depth", "ppm",
                         "How much starlight is blocked during transit"),
    "koi_duration":     ("Transit duration", "hours",
                         "Length of the transit event"),
    "koi_impact":       ("Impact parameter", "",
                         "Sky-projected distance of transit from star centre"),
    "koi_model_snr":    ("Transit signal-to-noise", "",
                         "Strength of the transit signal vs. noise"),
    "koi_max_mult_ev":  ("Max multiple event stat", "",
                         "Multiple Event Statistic of the strongest event"),
    "koi_srad":         ("Stellar radius", "solar radii",
                         "Radius of the host star"),
    "koi_smass":        ("Stellar mass", "solar masses",
                         "Mass of the host star"),
    "koi_steff":        ("Stellar temperature", "K",
                         "Effective temperature of the host star"),
    "koi_bin_oedp_sig": ("Odd-even depth significance", "sigma",
                         "Difference between odd and even transit depths"),
}

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CODE_DIR)
DATASETS_DIR = os.path.join(PROJECT_DIR, "Datasets")
MODELS_DIR = os.path.join(PROJECT_DIR, "Models")
PERFORMANCE_DIR = os.path.join(PROJECT_DIR, "Performance")


def run_pipeline(input_file, processed_name, model_name, threshold=None,
                 log=print):
    """Preprocess (optional) -> split -> train -> validate -> test."""
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input dataset not found:\n{input_file}")
    if not model_name:
        raise ValueError("Model file name cannot be empty.")

    base = os.path.splitext(model_name)[0]

    if processed_name:
        processed_base = os.path.splitext(processed_name)[0]
        processed_file = os.path.join(DATASETS_DIR, processed_base + ".csv")
        log("Preprocessing dataset...")
        df = preProcess(
            input_file, processed_file, FEATURES, TARGET,
            log_file=os.path.join(PERFORMANCE_DIR, base + "_preprocessing.json"),
        )
        log(f"Processed dataset saved: {processed_file} ({len(df)} rows)")
        training_file = processed_file
    else:
        training_file = input_file

    return modelTraining(
        training_file,
        os.path.join(MODELS_DIR, base + ".pkl"),
        os.path.join(PERFORMANCE_DIR, base + "_perf.json"),
        FEATURES, TARGET, threshold, log=log,
    )


def format_summary(perf):
    t, v = perf["test_metrics"], perf["validation_metrics"]
    s = perf["data_split"]
    cm = t["confusion_matrix"]
    lines = [
        f"Split: train={s['train']}  validation={s['validation']}  test={s['test']}",
        f"Threshold: {perf['threshold']:.3f} ({perf['threshold_source']})",
        "",
        f"{'Metric':<20}{'Validation':>12}{'TEST':>12}",
    ]
    for m in ["accuracy", "balanced_accuracy", "precision", "recall",
              "specificity", "f1", "mcc", "roc_auc"]:
        lines.append(f"{m:<20}{v[m]:>12.4f}{t[m]:>12.4f}")
    lines += ["", f"Test confusion matrix: TN={cm['tn']} FP={cm['fp']} "
                  f"FN={cm['fn']} TP={cm['tp']}"]
    return "\n".join(lines)


# ------------------------------------------------------------
# Single-sample prediction
# ------------------------------------------------------------
def list_models():
    """Names (without extension) of all saved models."""
    if not os.path.isdir(MODELS_DIR):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(MODELS_DIR)
                  if f.endswith(".pkl"))


def load_bundle(model_name):
    """Load a trained model plus its threshold and feature ranges."""
    base = os.path.splitext(model_name.strip())[0]
    model_path = os.path.join(MODELS_DIR, base + ".pkl")
    perf_path = os.path.join(PERFORMANCE_DIR, base + "_perf.json")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found:\n{model_path}")

    model = joblib.load(model_path)
    threshold, ranges = 0.5, {}
    if os.path.isfile(perf_path):
        with open(perf_path) as f:
            perf = json.load(f)
        threshold = float(perf.get("threshold", 0.5))
        ranges = perf.get("feature_ranges", {})

    # Older models: rebuild ranges from the saved training split.
    if not ranges:
        train_csv = os.path.join(DATASETS_DIR, "Splits", f"{base}_train.csv")
        if os.path.isfile(train_csv):
            tr = pd.read_csv(train_csv)
            ranges = {f: {"min": float(tr[f].min()), "max": float(tr[f].max()),
                          "median": float(tr[f].median()),
                          "p05": float(tr[f].quantile(0.05)),
                          "p95": float(tr[f].quantile(0.95))}
                      for f in FEATURES}
    return {"name": base, "model": model, "threshold": threshold,
            "ranges": ranges}


def predict_single(bundle, values):
    """Predict one sample.  `values` maps feature name -> float.

    probability_confirmed  P(class 1) from the Random Forest
    confidence             probability of the *predicted* class
    """
    X = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    model = bundle["model"]
    proba = model.predict_proba(X)[0]
    p_conf = float(proba[list(model.classes_).index(1)])
    threshold = bundle["threshold"]
    label = int(p_conf >= threshold)
    return {
        "label": label,
        "label_name": "CONFIRMED" if label else "FALSE POSITIVE",
        "probability_confirmed": p_conf,
        "probability_false_positive": 1.0 - p_conf,
        "confidence": p_conf if label else 1.0 - p_conf,
        "threshold": threshold,
    }
