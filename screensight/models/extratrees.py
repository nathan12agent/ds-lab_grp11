from sklearn.ensemble import ExtraTreesClassifier


def get_model():
    return ExtraTreesClassifier(
        n_estimators=200,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
    )


def get_param_grid():
    return {
        "n_estimators":     [100, 200, 300],
        "max_features":     ["sqrt", "log2"],
        "min_samples_leaf": [1, 3, 5],
        "max_depth":        [None, 10, 20],
    }


if __name__ == "__main__":
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("clean_train_final1 (1).csv")
    X = df.drop(columns=["id", "sii", "PCIAT-PCIAT_Total"], errors="ignore")
    y = df["sii"]
    X = X.fillna(X.mean())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = get_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:          ", accuracy_score(y_test, y_pred))
    print("F1 Score:          ", f1_score(y_test, y_pred, average='weighted'))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))