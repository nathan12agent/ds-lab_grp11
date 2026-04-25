# ScreenSight v2 — Problematic Internet Use Predictor

A research-backed ML screening tool that estimates a child's Severity Index (SII) for Problematic Internet Use (PIU).

## Stack
- **Frontend**: Streamlit (dark glassmorphism UI)
- **ML Pipeline**: Scikit-learn, XGBoost, Imbalanced-learn
- **Ensemble**: StackingClassifier (LR + SVM + RF + GB + XGB) with tuned XGBoost meta-learner

## ML Pipeline

```
tune_models.py      → 10-fold CV + GridSearchCV + RandomizedSearchCV
                       on all 5 base models + stacking meta-learner
train_ensemble.py   → StackingClassifier with SMOTE inside ImbPipeline
                       (no data leakage), saves ensemble_model.pkl
app.py              → Streamlit app loads pkl and runs inference
```

## Results

| Model | GridSearch F1 |
|---|---|
| Logistic Regression | 64.2% |
| SVM | 85.8% |
| Random Forest | 88.5% |
| Gradient Boosting | 88.4% |
| XGBoost | 87.9% |
| **Stacking Ensemble (Test)** | **62% accuracy, 0.08 overfit gap** |

## Setup

```bash
pip install -r requirements.txt
python screensight/tune_models.py      # optional — best_params.json included
python screensight/train_ensemble.py   # generates ensemble_model.pkl
streamlit run app.py
```

## Dataset

Healthy Brain Network dataset (`clean_train_final1.csv`, `clean_test_final1.csv`)  
~2500 samples, 54 features, 3-class SII target (None / Mild / Moderate)

## Notes

- Not a clinical diagnosis
- SMOTE applied inside cross-validation folds to prevent data leakage
- Model self-reports confidence and flags conflicting predictions