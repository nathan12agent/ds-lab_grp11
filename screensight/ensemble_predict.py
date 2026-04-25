"""
screensight/ensemble_predict.py
────────────────────────────────
Loaded by Streamlit at startup. Reads pkl once, never retrains.

If pkl is missing, run from project root:
    python screensight/train_ensemble.py
"""
import os
import joblib
import numpy as np
import pandas as pd

from screensight.constants import FEATURE_COLS

_PKL_PATH     = os.path.join(os.path.dirname(__file__), "ensemble_model.pkl")
_MEDIANS_PATH = os.path.join(os.path.dirname(__file__), "col_medians.pkl")


def _load():
    if not os.path.exists(_PKL_PATH):
        raise RuntimeError(
            "ensemble_model.pkl not found.\n"
            "Run: python screensight/train_ensemble.py"
        )
    model   = joblib.load(_PKL_PATH)
    medians = joblib.load(_MEDIANS_PATH)
    return model, medians


_ensemble, _col_medians = _load()


def build_input_row(answers: dict) -> pd.DataFrame:
    row = {}
    for col in FEATURE_COLS:
        val = answers.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = _col_medians.get(col, 0.0)
        row[col] = float(val)
    return pd.DataFrame([row])[FEATURE_COLS]


# FIND this entire function and replace it:
def run_inference(answers: dict):
    """
    Returns:
        predictions     : dict[model_name -> int SII]
        final_pred      : int — stacking ensemble result
        input_df        : pd.DataFrame (single row, for charts)
        final_confidence: float — meta-learner max probability
        conflict        : bool — True if strong models disagree
    """
    input_df        = build_input_row(answers)
    prob_rows       = _ensemble.predict_proba(input_df)
    final_pred      = int(np.argmax(prob_rows[0]))
    final_confidence = float(np.max(prob_rows[0]))

    # Per-model predictions from base estimators
    predictions = {}
    for name, estimator in _ensemble.named_estimators_.items():
        try:
            pred = int(estimator.predict(input_df)[0])
        except Exception:
            pred = final_pred
        predictions[name] = pred

    # Check if strong models conflict
    STRONG_MODELS = {"SVM", "Random Forest", "Gradient Boosting", "XGBoost"}
    strong_preds  = {k: v for k, v in predictions.items() if k in STRONG_MODELS}
    unique_strong = set(strong_preds.values())
    conflict      = len(unique_strong) > 1

    return predictions, final_pred, input_df, final_confidence, conflict