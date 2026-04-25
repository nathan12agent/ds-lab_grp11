import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from xgboost import XGBClassifier


def get_model():
    return XGBClassifier(
        n_estimators=100,        # was 300
        max_depth=6,
        learning_rate=0.1,       # was 0.05
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",      # ADD THIS — much faster than default
    )

def get_param_grid():
    return {
        "n_estimators":  [100, 200],   # removed 300
        "max_depth":     [6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample":     [0.8],        # fixed, no search needed
    }
        
if __name__ == "__main__":
    df = pd.read_csv("clean_train_final1 (1).csv")
    X = df.drop(columns=["id", "sii"])
    y = df["sii"]
    X = X.fillna(X.mean())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = get_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))