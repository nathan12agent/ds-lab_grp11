import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def get_model():
    return SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=42,
        probability=True,
    )

def get_param_grid():
    return {
        "C":      [0.1, 1.0, 10.0],
        "gamma":  ["scale", "auto"],
        "kernel": ["rbf"],           # removed poly
    }

if __name__ == "__main__":
    df = pd.read_csv("clean_train_final1 (1).csv")
    X = df.drop(columns=["id", "sii"])
    y = df["sii"]
    X = X.fillna(X.mean())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = Pipeline([("scaler", StandardScaler()), ("svc", get_model())])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))