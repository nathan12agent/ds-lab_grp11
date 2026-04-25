"""
screensight/train_ensemble.py
──────────────────────────────
Run ONCE from project root:
    python screensight/train_ensemble.py
"""

import os, sys, json, datetime, warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from screensight.constants import FEATURE_COLS
from screensight.models import MODEL_REGISTRY

# ─────────────────────────────────────────────
# 1. Load & preprocess
# ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("clean_train_final1 (1).csv")
df = df.drop(columns=["id", "PCIAT-PCIAT_Total"], errors="ignore")
df = df.fillna(df.median(numeric_only=True))
df["sii"] = df["sii"].replace(3, 2)  # merge Severe into Moderate

col_medians = df.median(numeric_only=True).to_dict()

X = df[FEATURE_COLS]
y = df["sii"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

# ─────────────────────────────────────────────
# 2. Load tuned params if available
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 2. Load tuned params if available
# ─────────────────────────────────────────────
_BEST_PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_params.json")
_best_params = {}
if os.path.exists(_BEST_PARAMS_PATH):
    with open(_BEST_PARAMS_PATH) as f:
        _best_params = {k: v["best_params"] for k, v in json.load(f).items()}
    print("  Loaded tuned hyperparameters from best_params.json")
else:
    print("  No best_params.json found — using defaults.")

# ADD THESE LINES RIGHT HERE:
_ensemble_params = _best_params.get("StackingEnsemble", {})
_meta_cv = _ensemble_params.get("cv", 5)
_meta_xgb_params = {
    k.replace("final_estimator__", ""): v
    for k, v in _ensemble_params.items()
    if k.startswith("final_estimator__")
}

# ─────────────────────────────────────────────
# 3. Build base estimators
#    Each wrapped in ImbPipeline so SMOTE only
#    sees training folds — no leakage
# ─────────────────────────────────────────────
NEEDS_SCALING = {"Logistic Regression", "SVM"}

# REMOVE the _OVERRIDES block entirely

# Keep only this in the loop:
estimators = []
for name, fn in MODEL_REGISTRY.items():
    model = fn()
    if name in _best_params:
        model.set_params(**_best_params[name])
    if name in NEEDS_SCALING:
        pipe = ImbPipeline([
            ("smote",  SMOTE(k_neighbors=2, random_state=42)),
            ("scaler", StandardScaler()),
            ("clf",    model),
        ])
    else:
        pipe = ImbPipeline([
            ("smote", SMOTE(k_neighbors=2, random_state=42)),
            ("clf",   model),
        ])
    estimators.append((name, pipe))
# ─────────────────────────────────────────────
# 4. Meta-learner (Stacking)
#    LR as meta-learner — learns which base
#    models to trust on this specific dataset
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 4. Meta-learner (Stacking)
# ─────────────────────────────────────────────
from xgboost import XGBClassifier

# Base XGB params — tuned params from best_params.json override these
_base_xgb = {
    "n_estimators":   100,
    "max_depth":      3,
    "learning_rate":  0.05,
    "min_child_weight": 5,
    "subsample":      0.8,
}
_base_xgb.update(_meta_xgb_params)  # tuned params win over defaults

meta_learner = XGBClassifier(
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    **_base_xgb,
)
print(f"  Meta-learner params: {_base_xgb}")
print(f"  Stacking CV folds: {_meta_cv}")

print("\nBuilding StackingClassifier...")
ensemble = StackingClassifier(
    estimators=estimators,
    final_estimator=meta_learner,
    cv=StratifiedKFold(n_splits=_meta_cv, shuffle=True, random_state=42),
    stack_method="predict_proba",
    passthrough=True,    # was False — gives meta-learner original features too
    n_jobs=-1,
)

# ─────────────────────────────────────────────
# 5. Train on raw training data (SMOTE is
#    inside each pipeline — applied per fold)
# ─────────────────────────────────────────────
print("Training stacking ensemble...")
ensemble.fit(X_train, y_train)
print("Training complete.")

# ─────────────────────────────────────────────
# 6. Evaluate
# ─────────────────────────────────────────────
y_pred_train = ensemble.predict(X_train)
y_pred_test  = ensemble.predict(X_test)

train_f1 = f1_score(y_train, y_pred_train, average="weighted")
test_f1  = f1_score(y_test,  y_pred_test,  average="weighted")

print("\n========== Evaluation ==========")
print(f"Train F1 (resubstitution) : {train_f1:.4f}")
print(f"Test  F1 (hold-out)       : {test_f1:.4f}")
print(f"Accuracy (test)           : {accuracy_score(y_test, y_pred_test):.4f}")
print("\nConfusion Matrix (test):\n",  confusion_matrix(y_test, y_pred_test))
print("\nClassification Report (test):\n", classification_report(y_test, y_pred_test))

gap = train_f1 - test_f1
print(f"\nOverfit gap : {gap:.4f}")
if gap > 0.10:
    print(f"⚠  WARNING: Overfitting — gap = {gap:.4f}")
elif test_f1 < 0.45:
    print(f"⚠  WARNING: Underfitting — test F1 = {test_f1:.4f}")
else:
    print(f"✓  OK: gap = {gap:.4f}")

# ─────────────────────────────────────────────
# 7. Save
# ─────────────────────────────────────────────
out_dir = os.path.dirname(os.path.abspath(__file__))
joblib.dump(ensemble,    os.path.join(out_dir, "ensemble_model.pkl"))
joblib.dump(col_medians, os.path.join(out_dir, "col_medians.pkl"))
print("\nSaved → screensight/ensemble_model.pkl")
print("Saved → screensight/col_medians.pkl")

diagnostics = {
    "trained_at":   datetime.datetime.now().isoformat(),
    "method":       "StackingClassifier",
    "train_f1":     round(train_f1, 4),
    "test_f1":      round(test_f1, 4),
    "overfit_gap":  round(gap, 4),
    "models":       list(MODEL_REGISTRY.keys()),
}
with open(os.path.join(out_dir, "ensemble_diagnostics.json"), "w") as f:
    json.dump(diagnostics, f, indent=2)
print("Saved → screensight/ensemble_diagnostics.json")