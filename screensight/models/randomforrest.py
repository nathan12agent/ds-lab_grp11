import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report


def get_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

def get_param_grid():
    """For GridSearchCV / RandomizedSearchCV"""
    return {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [None, 10, 20],
        "min_samples_leaf": [1, 3, 5],
        "max_features":     ["sqrt", "log2"],
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