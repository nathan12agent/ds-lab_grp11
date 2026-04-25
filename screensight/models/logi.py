import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def get_model():
    return LogisticRegression(
        max_iter=1000,
        solver='lbfgs',        # removed multi_class='multinomial' (deprecated)
        class_weight='balanced',
        C=10.0,
    )

def get_param_grid():
    return {
        "C":        [0.01, 0.1, 0.5, 1.0, 10.0],
        "solver":   ["lbfgs"],    # remove "saga"
        "max_iter": [2000],       # fixed high value
    }

if __name__ == "__main__":
    df = pd.read_csv("clean_train_final1 (1).csv")

    X = df.drop(columns=["id", "sii"])
    y = df["sii"]
    X = X.fillna(X.mean())

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = get_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print(confusion_matrix(y_test, y_pred))