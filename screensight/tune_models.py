"""
screensight/tune_models.py
──────────────────────────
Run from project root ONCE before training the ensemble:

    python screensight/tune_models.py

Performs for each model:
  1. StratifiedKFold cross-validation (baseline)
  2. RandomizedSearchCV  (fast, broad)
  3. GridSearchCV        (fine-grained, on narrowed grid)

Saves best params to screensight/best_params.json
"""

import os, sys, json, datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score,
    GridSearchCV, RandomizedSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from screensight.constants import FEATURE_COLS
from screensight.models import MODEL_REGISTRY, PARAM_GRID_REGISTRY

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("clean_train_final1 (1).csv")
df = df.drop(columns=["id", "PCIAT-PCIAT_Total"], errors="ignore")
df = df.fillna(df.median(numeric_only=True))

X = df[FEATURE_COLS]
y = df["sii"]

X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(k_neighbors=2, random_state=42)
X_res, y_res = smote.fit_resample(X_train_raw, y_train_raw)
print(f"  After SMOTE: {X_res.shape[0]} samples")

NEEDS_SCALING = {"Logistic Regression", "SVM"}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

best_params_all = {}

for name, get_model_fn in MODEL_REGISTRY.items():
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")

    model = get_model_fn()
    param_grid = PARAM_GRID_REGISTRY[name]()

    # Wrap scalers into pipeline prefix for LR/SVM
    if name in NEEDS_SCALING:
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", model)])
        prefixed_grid = {f"clf__{k}": v for k, v in param_grid.items()}
    else:
        pipeline = model
        prefixed_grid = param_grid

    # ── 1. Baseline cross-validation ──────────────────────────────────────────
    print("  [1/3] Baseline cross-validation (10-fold)...")
    base_scores = cross_val_score(
        pipeline, X_res, y_res, cv=cv, scoring="f1_weighted", n_jobs=-1
    )
    print(f"       CV F1: {base_scores.mean():.4f} ± {base_scores.std():.4f}")

    # ── 2. RandomizedSearchCV ─────────────────────────────────────────────────
    print("  [2/3] RandomizedSearchCV (n_iter=20)...")
    rscv = RandomizedSearchCV(
        pipeline, prefixed_grid,
        n_iter=10, cv=cv, scoring="f1_weighted",
        random_state=42, n_jobs=-1, verbose=0,
    )
    rscv.fit(X_res, y_res)
    print(f"       Best F1: {rscv.best_score_:.4f}")
    print(f"       Best params: {rscv.best_params_}")

    # ── 3. GridSearchCV (narrowed around RandomizedSearch best) ───────────────
    print("  [3/3] GridSearchCV (full grid)...")
    gscv = GridSearchCV(
        pipeline, prefixed_grid,
        cv=cv, scoring="f1_weighted",
        n_jobs=-1, verbose=0,
    )
    gscv.fit(X_res, y_res)
    print(f"       Best F1: {gscv.best_score_:.4f}")
    print(f"       Best params: {gscv.best_params_}")

    # Strip pipeline prefix for storage
    clean_params = {
        k.replace("clf__", ""): v
        for k, v in gscv.best_params_.items()
    }
    best_params_all[name] = {
        "baseline_cv_f1":     round(float(base_scores.mean()), 4),
        "random_search_f1":   round(float(rscv.best_score_), 4),
        "grid_search_f1":     round(float(gscv.best_score_), 4),
        "best_params":        clean_params,
    }

# ── Save results ──────────────────────────────────────────────────────────────
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_params.json")
with open(out_path, "w") as f:
    json.dump(best_params_all, f, indent=2)
print(f"\n✓ Saved best params → screensight/best_params.json")

# ── Tune the Stacking meta-learner ────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"  Stacking Meta-Learner (XGBoost)")
print(f"{'='*55}")

from sklearn.ensemble import StackingClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
import math

# Rebuild base estimators with tuned params
NEEDS_SCALING = {"Logistic Regression", "SVM"}
estimators = []
for name, fn in MODEL_REGISTRY.items():
    model = fn()
    if name in best_params_all and "best_params" in best_params_all[name]:
        model.set_params(**best_params_all[name]["best_params"])
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

# Meta-learner param grid
meta_param_grid = {
    "final_estimator__n_estimators":  [50, 100, 200],
    "final_estimator__max_depth":     [2, 3, 5],
    "final_estimator__learning_rate": [0.01, 0.05, 0.1],
    "cv":                             [3, 5, 10],
}

stacking = StackingClassifier(
    estimators=estimators,
    final_estimator=XGBClassifier(
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    ),
    stack_method="predict_proba",
    passthrough=True,
    n_jobs=-1,
)

# ── 1. RandomizedSearchCV on meta-learner ─────────────────────────────────────
print("  [1/2] RandomizedSearchCV on meta-learner (n_iter=10)...")
print("        ⚠ This will take 15-30 minutes — fitting 5 models × 10 iterations")
meta_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

meta_rscv = RandomizedSearchCV(
    stacking,
    meta_param_grid,
    n_iter=10,
    cv=meta_cv,
    scoring="f1_weighted",
    random_state=42,
    n_jobs=-1,
    verbose=1,
)
meta_rscv.fit(X_res, y_res)
print(f"  Best F1:     {meta_rscv.best_score_:.4f}")
print(f"  Best params: {meta_rscv.best_params_}")

# ── 2. GridSearchCV (narrowed grid) ───────────────────────────────────────────
print("  [2/2] GridSearchCV on narrowed grid...")
best = meta_rscv.best_params_
narrow_grid = {
    "final_estimator__n_estimators":  [best["final_estimator__n_estimators"]],
    "final_estimator__max_depth":     [
        max(2, best["final_estimator__max_depth"] - 1),
        best["final_estimator__max_depth"],
    ],
    "final_estimator__learning_rate": [best["final_estimator__learning_rate"]],
    "cv":                             [best["cv"]],
}
meta_gscv = GridSearchCV(
    stacking,
    narrow_grid,
    cv=meta_cv,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1,
)
meta_gscv.fit(X_res, y_res)
print(f"  Best F1:     {meta_gscv.best_score_:.4f}")
print(f"  Best params: {meta_gscv.best_params_}")

# ── Save ensemble params into best_params.json ────────────────────────────────
best_params_all["StackingEnsemble"] = {
    "baseline_cv_f1":   round(float(meta_rscv.best_score_), 4),
    "random_search_f1": round(float(meta_rscv.best_score_), 4),
    "grid_search_f1":   round(float(meta_gscv.best_score_), 4),
    "best_params":      meta_gscv.best_params_,
}

with open(out_path, "w") as f:
    json.dump(best_params_all, f, indent=2)
print(f"\n✓ Updated best params → screensight/best_params.json")
print("\nNow run: python screensight/train_ensemble.py")